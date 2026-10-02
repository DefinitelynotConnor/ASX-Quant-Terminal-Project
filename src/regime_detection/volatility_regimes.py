"""
volatility_regimes.py
=====================
Rule-based and statistical volatility regime detection.

Complements HMM and Markov Switching with interpretable, fast methods
that are useful for live trading and strategy switching.

Methods
-------
    Rolling Vol Percentile  : Low/Med/High based on trailing percentile rank
    VIX-Based               : Uses VIX level thresholds (global risk proxy)
    GARCH-Based             : Uses GARCH conditional vol for classification
    Trend Regime            : Bull/Bear/Sideways using moving average rules
    Combined Regime Engine  : Consensus across all methods

Usage
-----
    from src.regime_detection.volatility_regimes import (
        VolatilityRegimeDetector,
        TrendRegimeDetector,
        RegimeEngine,
    )

    engine = RegimeEngine(returns, macro_data=macro_df)
    current = engine.current_regime()
    history = engine.regime_history()
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Volatility Regime Detector
# ─────────────────────────────────────────────

class VolatilityRegimeDetector:
    """
    Classify volatility into discrete regimes using multiple methods.

    Parameters
    ----------
    returns : pd.Series   Daily return series.
    prices  : pd.DataFrame, optional   OHLCV for range-based estimators.
    """

    VOL_LABELS_2 = {0: "Low Vol", 1: "High Vol"}
    VOL_LABELS_3 = {0: "Low Vol", 1: "Medium Vol", 2: "High Vol"}

    def __init__(
        self,
        returns: pd.Series,
        prices: Optional[pd.DataFrame] = None,
    ) -> None:
        self.returns = returns.dropna().copy()
        self.prices = prices

    # ──────────────────────────────────────
    #  Rolling percentile method
    # ──────────────────────────────────────

    def percentile_regimes(
        self,
        window: int = 21,
        lookback: int = 252,
        n_regimes: int = 3,
    ) -> pd.Series:
        """
        Classify volatility using its rolling percentile rank.

        Parameters
        ----------
        window    : int   Short window for realised vol calculation.
        lookback  : int   Historical window for percentile ranking.
        n_regimes : int   2 or 3 regimes.

        Returns
        -------
        pd.Series  Regime labels ('Low Vol', 'Medium Vol', 'High Vol').
        """
        vol = self.returns.rolling(window).std() * np.sqrt(252)
        vol_pct = vol.rolling(lookback).rank(pct=True)

        if n_regimes == 2:
            labels = pd.cut(
                vol_pct,
                bins=[0, 0.5, 1.0],
                labels=["Low Vol", "High Vol"],
            )
        else:
            labels = pd.cut(
                vol_pct,
                bins=[0, 0.33, 0.67, 1.0],
                labels=["Low Vol", "Medium Vol", "High Vol"],
            )

        return labels.astype(str).rename("Vol_Regime_Percentile")

    # ──────────────────────────────────────
    #  Absolute threshold method
    # ──────────────────────────────────────

    def threshold_regimes(
        self,
        window: int = 21,
        low_threshold: float = 0.12,
        high_threshold: float = 0.25,
    ) -> pd.Series:
        """
        Classify volatility using fixed annualised thresholds.
        Useful for consistent comparison across time periods.

        Parameters
        ----------
        low_threshold  : float   Below = Low Vol (e.g. 12% ann.)
        high_threshold : float   Above = High Vol (e.g. 25% ann.)
        """
        vol = self.returns.rolling(window).std() * np.sqrt(252)

        regime = pd.Series("Medium Vol", index=vol.index, name="Vol_Regime_Threshold")
        regime[vol < low_threshold] = "Low Vol"
        regime[vol > high_threshold] = "High Vol"
        regime[vol.isna()] = "Unknown"

        return regime

    # ──────────────────────────────────────
    #  VIX-based method
    # ──────────────────────────────────────

    def vix_regimes(
        self,
        vix: pd.Series,
        low_threshold: float = 15.0,
        high_threshold: float = 25.0,
        extreme_threshold: float = 35.0,
    ) -> pd.Series:
        """
        Classify market regimes using VIX (fear index) levels.

        VIX thresholds (historically):
            < 15  = Low volatility, complacency
            15-25 = Normal volatility
            25-35 = Elevated fear
            > 35  = Panic/crisis

        Parameters
        ----------
        vix : pd.Series   VIX daily close prices.
        """
        vix_aligned = vix.reindex(self.returns.index).ffill()

        regime = pd.Series("Normal", index=self.returns.index, name="VIX_Regime")
        regime[vix_aligned < low_threshold] = "Low Vol"
        regime[
            (vix_aligned >= low_threshold) & (vix_aligned < high_threshold)
        ] = "Normal"
        regime[
            (vix_aligned >= high_threshold) & (vix_aligned < extreme_threshold)
        ] = "High Vol"
        regime[vix_aligned >= extreme_threshold] = "Extreme Vol"

        return regime

    # ──────────────────────────────────────
    #  GARCH-based regime
    # ──────────────────────────────────────

    def garch_regimes(
        self,
        low_pct: float = 0.33,
        high_pct: float = 0.67,
    ) -> pd.Series:
        """
        Use GARCH conditional volatility for regime classification.
        Falls back to EWMA if GARCH unavailable.
        """
        try:
            from src.analytics.volatility_models import VolatilityModels
            vm = VolatilityModels(self.returns)
            g = vm.garch_volatility()
            if "conditional_vol" in g:
                garch_vol = g["conditional_vol"]
            else:
                garch_vol = vm.ewma_volatility()
        except Exception:
            garch_vol = (
                self.returns.ewm(alpha=0.06, adjust=False).std() * np.sqrt(252)
            )

        p_low = garch_vol.quantile(low_pct)
        p_high = garch_vol.quantile(high_pct)

        regime = pd.Series("Medium Vol", index=garch_vol.index, name="GARCH_Regime")
        regime[garch_vol <= p_low] = "Low Vol"
        regime[garch_vol > p_high] = "High Vol"

        return regime

    # ──────────────────────────────────────
    #  Summary
    # ──────────────────────────────────────

    def all_regimes(self) -> pd.DataFrame:
        """Compute all volatility regime series and return as DataFrame."""
        pct = self.percentile_regimes()
        thr = self.threshold_regimes()
        return pd.DataFrame({
            "Percentile_Regime": pct,
            "Threshold_Regime": thr,
        }).dropna()


# ─────────────────────────────────────────────
#  Trend Regime Detector
# ─────────────────────────────────────────────

class TrendRegimeDetector:
    """
    Detect Bull, Bear, and Sideways market regimes using
    price-based trend rules.

    Parameters
    ----------
    prices  : pd.Series or pd.DataFrame   Close price series.
    returns : pd.Series                   Daily return series.
    """

    def __init__(
        self,
        prices: pd.Series | pd.DataFrame,
        returns: Optional[pd.Series] = None,
    ) -> None:
        if isinstance(prices, pd.DataFrame):
            self.prices = prices.iloc[:, 0]
        else:
            self.prices = prices.dropna().copy()

        self.returns = returns if returns is not None else self.prices.pct_change().dropna()

    def moving_average_regime(
        self,
        short: int = 50,
        long: int = 200,
    ) -> pd.Series:
        """
        Classic dual moving average trend regime.
        Bull    : price above both MAs, short MA above long MA
        Bear    : price below both MAs, short MA below long MA
        Sideways: mixed signals

        Returns
        -------
        pd.Series  'Bull', 'Bear', or 'Sideways'.
        """
        sma_s = self.prices.rolling(short).mean()
        sma_l = self.prices.rolling(long).mean()

        regime = pd.Series("Sideways", index=self.prices.index, name="MA_Regime")
        bull = (self.prices > sma_s) & (sma_s > sma_l)
        bear = (self.prices < sma_s) & (sma_s < sma_l)
        regime[bull] = "Bull"
        regime[bear] = "Bear"

        return regime

    def drawdown_regime(
        self,
        bear_threshold: float = -0.20,
        bull_threshold: float = 0.20,
    ) -> pd.Series:
        """
        Classify based on drawdown from peak.
        Bear Market  : drawdown < -20% (formal definition)
        Bull Market  : within 20% of all-time-high
        Recovery     : between -20% and 0%

        Parameters
        ----------
        bear_threshold : float   Drawdown threshold for Bear (default -20%).
        bull_threshold : float   Within % of ATH for Bull (default 20%).
        """
        peak = self.prices.cummax()
        drawdown = (self.prices / peak) - 1

        regime = pd.Series("Recovery", index=self.prices.index, name="DD_Regime")
        regime[drawdown <= bear_threshold] = "Bear"
        regime[drawdown >= -0.05] = "Bull"   # within 5% of ATH = Bull

        return regime

    def rsi_regime(
        self,
        period: int = 14,
        overbought: float = 65,
        oversold: float = 35,
    ) -> pd.Series:
        """
        RSI-based regime classification.
        Bull     : RSI > overbought (strong upward momentum)
        Bear     : RSI < oversold (strong downward momentum)
        Sideways : RSI between thresholds
        """
        delta = self.prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)

        regime = pd.Series("Sideways", index=self.prices.index, name="RSI_Regime")
        regime[rsi > overbought] = "Bull"
        regime[rsi < oversold] = "Bear"

        return regime

    def breadth_regime(
        self,
        universe_prices: Optional[pd.DataFrame] = None,
        threshold: float = 0.60,
    ) -> pd.Series:
        """
        Market breadth regime: % of stocks above their 200-day MA.
        Bull     : > threshold of stocks above 200MA
        Bear     : < (1-threshold) of stocks above 200MA
        Sideways : in between

        Parameters
        ----------
        universe_prices : pd.DataFrame   Close prices for ASX universe.
        threshold       : float          % required for Bull/Bear classification.
        """
        if universe_prices is None:
            return pd.Series("Unknown", index=self.prices.index, name="Breadth_Regime")

        ma_200 = universe_prices.rolling(200).mean()
        above_200 = (universe_prices > ma_200).mean(axis=1)

        regime = pd.Series("Sideways", index=above_200.index, name="Breadth_Regime")
        regime[above_200 > threshold] = "Bull"
        regime[above_200 < (1 - threshold)] = "Bear"

        return regime.reindex(self.prices.index).fillna("Sideways")

    def all_trend_regimes(
        self,
        universe_prices: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute all trend regime series."""
        ma = self.moving_average_regime()
        dd = self.drawdown_regime()
        rsi = self.rsi_regime()

        df = pd.DataFrame({
            "MA_Regime": ma,
            "Drawdown_Regime": dd,
            "RSI_Regime": rsi,
        })

        if universe_prices is not None:
            df["Breadth_Regime"] = self.breadth_regime(universe_prices)

        return df.dropna()


# ─────────────────────────────────────────────
#  Combined Regime Engine
# ─────────────────────────────────────────────

class RegimeEngine:
    """
    Master regime detection engine.
    Combines HMM, MRS, volatility, and trend signals into a
    single consensus regime classification.

    Parameters
    ----------
    returns      : pd.Series   Daily return series.
    prices       : pd.Series   Close price series (for trend detection).
    macro_data   : pd.DataFrame, optional   Macro variables (VIX, AUD/USD).
    universe_prices : pd.DataFrame, optional   For breadth calculation.
    """

    def __init__(
        self,
        returns: pd.Series,
        prices: Optional[pd.Series] = None,
        macro_data: Optional[pd.DataFrame] = None,
        universe_prices: Optional[pd.DataFrame] = None,
    ) -> None:
        self.returns = returns.dropna().copy()
        self.prices = prices
        self.macro = macro_data
        self.universe = universe_prices

    def run_all(self, use_hmm: bool = True) -> pd.DataFrame:
        """
        Run all regime detection methods and return a combined DataFrame.

        Parameters
        ----------
        use_hmm : bool   Include HMM (slower but more sophisticated).

        Returns
        -------
        pd.DataFrame  One column per method + consensus column.
        """
        results = {}

        # Volatility regimes
        vrd = VolatilityRegimeDetector(self.returns)
        results["Vol_Percentile"] = vrd.percentile_regimes()
        results["Vol_Threshold"] = vrd.threshold_regimes()

        if self.macro is not None and "VIX" in self.macro.columns:
            results["VIX_Regime"] = vrd.vix_regimes(self.macro["VIX"])

        # Trend regimes
        if self.prices is not None:
            trd = TrendRegimeDetector(self.prices, self.returns)
            trend = trd.all_trend_regimes(self.universe)
            for col in trend.columns:
                results[col] = trend[col]

        # HMM
        if use_hmm:
            try:
                from src.regime_detection.hidden_markov_models import HiddenMarkovModel
                hmm = HiddenMarkovModel(self.returns, n_states=3)
                hmm.fit()
                results["HMM_Regime"] = hmm.predict_regimes()
            except Exception as e:
                print(f"  [RegimeEngine] HMM skipped: {e}")

        df = pd.DataFrame(results).dropna(how="all")
        df["Consensus"] = df.apply(self._consensus_regime, axis=1)

        return df

    def _consensus_regime(self, row: pd.Series) -> str:
        """
        Majority vote across all regime signals.
        Maps diverse labels to Bull/Bear/Sideways for consistency.
        """
        bull_words = {"bull", "low vol", "low"}
        bear_words = {"bear", "high vol", "extreme vol", "high"}
        sideways_words = {"sideways", "medium vol", "normal", "recovery"}

        votes = {"Bull": 0, "Bear": 0, "Sideways": 0}
        for val in row:
            v = str(val).lower()
            if any(w in v for w in bull_words):
                votes["Bull"] += 1
            elif any(w in v for w in bear_words):
                votes["Bear"] += 1
            else:
                votes["Sideways"] += 1

        return max(votes, key=votes.get)

    def current_regime(self) -> dict:
        """
        Return the current consensus regime and supporting signals.

        Returns
        -------
        dict  {consensus, signals, confidence}
        """
        df = self.run_all(use_hmm=False)   # Fast version without HMM
        latest = df.iloc[-1]

        votes = {"Bull": 0, "Bear": 0, "Sideways": 0}
        signals = {}
        for col in df.columns:
            if col == "Consensus":
                continue
            val = str(latest[col]).lower()
            signals[col] = str(latest[col])
            if "bull" in val or "low vol" in val:
                votes["Bull"] += 1
            elif "bear" in val or "high vol" in val or "extreme" in val:
                votes["Bear"] += 1
            else:
                votes["Sideways"] += 1

        consensus = max(votes, key=votes.get)
        total = sum(votes.values())
        confidence = votes[consensus] / total if total > 0 else 0

        return {
            "consensus": consensus,
            "confidence": round(confidence, 2),
            "votes": votes,
            "signals": signals,
            "date": str(df.index[-1].date()),
        }

    def regime_history(self, use_hmm: bool = True) -> pd.DataFrame:
        """Return the full historical regime DataFrame."""
        return self.run_all(use_hmm=use_hmm)

    def regime_statistics(self) -> pd.DataFrame:
        """
        Summary statistics for each consensus regime:
        average return, volatility, Sharpe, % of time.
        """
        df = self.run_all(use_hmm=False)
        consensus = df["Consensus"].reindex(self.returns.index).dropna()

        rows = []
        for regime in ["Bull", "Sideways", "Bear"]:
            mask = consensus == regime
            ret = self.returns[mask]
            if len(ret) == 0:
                continue
            rows.append({
                "Regime": regime,
                "Ann. Return": float(ret.mean() * 252),
                "Ann. Volatility": float(ret.std() * np.sqrt(252)),
                "Sharpe": float(
                    ret.mean() / ret.std() * np.sqrt(252)
                ) if ret.std() > 0 else 0,
                "% of Time": float(mask.mean()),
                "Observations": int(mask.sum()),
                "Best Day": float(ret.max()),
                "Worst Day": float(ret.min()),
            })

        return pd.DataFrame(rows).set_index("Regime")

    def strategy_regime_filter(
        self,
        signal: pd.Series,
        allowed_regimes: list[str] = None,
    ) -> pd.Series:
        """
        Filter a strategy signal to only trade in allowed regimes.
        Sets signal to 0 (flat) in disallowed regimes.

        Parameters
        ----------
        signal          : pd.Series   Trading signal (+1, 0, -1).
        allowed_regimes : list        Regimes where trading is permitted.
                                      e.g. ['Bull', 'Sideways']

        Returns
        -------
        pd.Series  Filtered signal.
        """
        allowed_regimes = allowed_regimes or ["Bull", "Sideways"]
        regimes = self.run_all(use_hmm=False)["Consensus"]
        regimes = regimes.reindex(signal.index).ffill()

        filtered = signal.copy()
        filtered[~regimes.isin(allowed_regimes)] = 0

        return filtered
