"""
factor_strategy.py  /  regime_adaptive_strategy.py
===================================================
Factor-based and regime-adaptive strategies for ASX equities.

Strategies Implemented
----------------------
FactorStrategy          : Rank and trade on combined factor scores
RegimeAdaptiveStrategy  : Switch strategy based on detected market regime
FactorTimingStrategy    : Time factor exposure based on macro conditions

Usage
-----
    from src.strategies.factor_strategy import FactorStrategy
    from src.strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
"""

from __future__ import annotations

import warnings
from typing import Optional, Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════
#  FACTOR STRATEGY
# ═══════════════════════════════════════════════

class FactorStrategy:
    """
    Multi-factor long-only or long/short strategy.

    Combines momentum, value, and quality factor scores into
    a composite ranking. Long top-ranked stocks, optionally
    short bottom-ranked stocks.

    Parameters
    ----------
    universe_prices  : pd.DataFrame   Close prices.
    factor_weights   : dict           {'momentum': 0.4, 'value': 0.3, 'quality': 0.3}
    n_long           : int            Number of long positions.
    n_short          : int            Number of short positions.
    rebalance_freq   : str            'monthly' | 'quarterly'.
    min_score        : float          Minimum composite score to enter long.
    """

    def __init__(
        self,
        universe_prices: pd.DataFrame,
        factor_weights: Optional[dict] = None,
        n_long: int = 10,
        n_short: int = 0,
        rebalance_freq: str = "monthly",
        min_score: float = 0.0,
    ) -> None:
        self.prices = universe_prices.copy()
        self.factor_weights = factor_weights or {
            "momentum": 0.40,
            "value": 0.30,
            "quality": 0.30,
        }
        self.n_long = n_long
        self.n_short = n_short
        self.rebalance_freq = rebalance_freq
        self.min_score = min_score

    def _compute_composite_score(self) -> pd.DataFrame:
        """Compute weighted composite factor score."""
        from src.factor_models.momentum_factor import (
            MomentumFactor, ValueFactor, QualityFactor
        )

        factors = {}
        if "momentum" in self.factor_weights:
            mom = MomentumFactor(self.prices)
            factors["momentum"] = mom.compute_signals(zscore=True)

        if "value" in self.factor_weights:
            val = ValueFactor(self.prices)
            factors["value"] = val.compute_signals(zscore=True)

        if "quality" in self.factor_weights:
            qual = QualityFactor(self.prices)
            factors["quality"] = qual.compute_signals(zscore=True)

        if not factors:
            return pd.DataFrame()

        composite = None
        for fname, fdf in factors.items():
            w = self.factor_weights.get(fname, 0)
            if composite is None:
                composite = fdf * w
            else:
                common = composite.index.intersection(fdf.index)
                composite = composite.loc[common] + fdf.loc[common] * w

        return composite

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate factor strategy signals.

        Returns
        -------
        pd.DataFrame  Position weights.
        """
        score = self._compute_composite_score()
        if score is None or score.empty:
            return pd.DataFrame()

        signals = pd.DataFrame(0.0, index=score.index, columns=score.columns)

        freq_map = {"monthly": "ME", "quarterly": "QE", "weekly": "W"}
        pandas_freq = freq_map.get(self.rebalance_freq, "ME")
        rebalance_dates = set(
            score.resample(pandas_freq).last().index.normalize()
        )

        last_signal = pd.Series(0.0, index=score.columns)

        for date in score.index:
            if pd.Timestamp(date).normalize() in rebalance_dates:
                row = score.loc[date].dropna()
                if len(row) < self.n_long:
                    last_signal = pd.Series(0.0, index=score.columns)
                    signals.loc[date] = last_signal
                    continue

                new_signal = pd.Series(0.0, index=score.columns)

                # Long: top N by composite score above min_score
                longs = row.nlargest(self.n_long)
                longs = longs[longs >= self.min_score]
                if not longs.empty:
                    new_signal[longs.index] = 1.0 / len(longs)

                # Short: bottom N by composite score
                if self.n_short > 0:
                    shorts = row.nsmallest(self.n_short)
                    if not shorts.empty:
                        new_signal[shorts.index] = -1.0 / len(shorts)

                last_signal = new_signal

            signals.loc[date] = last_signal

        return signals

    def backtest(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        benchmark: Optional[pd.Series] = None,
    ):
        """Run factor strategy backtest."""
        from src.backtesting.backtester import Backtester, BacktestConfig

        signals = self.generate_signals()
        config = BacktestConfig(
            initial_capital=initial_capital,
            commission=commission,
            slippage=0.0005,
            allow_short=self.n_short > 0,
            signal_lag=1,
        )
        bt = Backtester(
            prices=self.prices,
            signals=signals,
            config=config,
            benchmark=benchmark,
            strategy_name="Factor_Strategy",
        )
        return bt.run()


# ═══════════════════════════════════════════════
#  REGIME ADAPTIVE STRATEGY
# ═══════════════════════════════════════════════

class RegimeAdaptiveStrategy:
    """
    Regime-adaptive strategy that switches between sub-strategies
    based on detected market regime.

    Concept
    -------
    Different strategies work in different regimes:
        Bull markets   → Momentum performs best
        Bear markets   → Move to defensive / cash / low-vol
        Sideways       → Mean reversion / value works

    Parameters
    ----------
    universe_prices : pd.DataFrame   Close prices.
    market_prices   : pd.Series      Market index for regime detection.
    regime_map      : dict           {regime_label: strategy_class_or_signals}
    n_positions     : int            Number of positions per regime.
    """

    def __init__(
        self,
        universe_prices: pd.DataFrame,
        market_prices: pd.Series,
        n_positions: int = 10,
        use_hmm: bool = False,
    ) -> None:
        self.prices = universe_prices.copy()
        self.market = market_prices.dropna()
        self.n_positions = n_positions
        self.use_hmm = use_hmm

    def _detect_regimes(self) -> pd.Series:
        """Detect market regimes using TrendRegimeDetector."""
        from src.regime_detection.volatility_regimes import TrendRegimeDetector

        trd = TrendRegimeDetector(
            prices=self.market,
            returns=self.market.pct_change().dropna(),
        )
        regimes = trd.moving_average_regime()
        return regimes.reindex(self.prices.index).ffill().fillna("Sideways")

    def _bull_signals(self) -> pd.DataFrame:
        """Momentum signals for bull regime."""
        from src.strategies.momentum_strategy import CrossSectionalMomentum
        strategy = CrossSectionalMomentum(
            self.prices, n_long=self.n_positions, n_short=0
        )
        return strategy.generate_signals()

    def _bear_signals(self) -> pd.DataFrame:
        """Defensive signals for bear regime — move to lowest vol stocks."""
        returns = self.prices.pct_change()
        vol = returns.rolling(63).std() * np.sqrt(252)
        low_vol = -vol   # Invert: low vol = high score

        signals = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)
        rebalance_dates = set(
            signals.resample("ME").last().index.normalize()
        )
        last_signal = pd.Series(0.0, index=self.prices.columns)

        for date in self.prices.index:
            if pd.Timestamp(date).normalize() in rebalance_dates:
                if date in low_vol.index:
                    row = low_vol.loc[date].dropna()
                    if len(row) >= self.n_positions:
                        new_sig = pd.Series(0.0, index=self.prices.columns)
                        longs = row.nlargest(self.n_positions).index
                        new_sig[longs] = 1.0 / self.n_positions
                        last_signal = new_sig
            signals.loc[date] = last_signal

        return signals

    def _sideways_signals(self) -> pd.DataFrame:
        """Value signals for sideways regime."""
        from src.factor_models.momentum_factor import ValueFactor
        val = ValueFactor(self.prices)
        scores = val.compute_signals(zscore=True)

        signals = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)
        rebalance_dates = set(
            signals.resample("ME").last().index.normalize()
        )
        last_signal = pd.Series(0.0, index=self.prices.columns)

        for date in self.prices.index:
            if pd.Timestamp(date).normalize() in rebalance_dates:
                if date in scores.index:
                    row = scores.loc[date].dropna()
                    if len(row) >= self.n_positions:
                        new_sig = pd.Series(0.0, index=self.prices.columns)
                        longs = row.nlargest(self.n_positions).index
                        new_sig[longs] = 1.0 / self.n_positions
                        last_signal = new_sig
            signals.loc[date] = last_signal

        return signals

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate regime-adaptive signals by blending sub-strategy signals.

        Returns
        -------
        pd.DataFrame  Combined regime-aware position weights.
        """
        regimes = self._detect_regimes()
        bull_sig = self._bull_signals()
        bear_sig = self._bear_signals()
        sideways_sig = self._sideways_signals()

        # Align all signals to same index
        common = (
            self.prices.index
            .intersection(bull_sig.index)
            .intersection(bear_sig.index)
            .intersection(sideways_sig.index)
            .intersection(regimes.index)
        )

        combined = pd.DataFrame(0.0, index=common, columns=self.prices.columns)

        for date in common:
            regime = str(regimes.loc[date])
            if regime == "Bull":
                combined.loc[date] = bull_sig.loc[date] if date in bull_sig.index else 0
            elif regime == "Bear":
                combined.loc[date] = bear_sig.loc[date] if date in bear_sig.index else 0
            else:
                combined.loc[date] = sideways_sig.loc[date] if date in sideways_sig.index else 0

        return combined

    def backtest(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        benchmark: Optional[pd.Series] = None,
    ):
        """Run regime adaptive backtest."""
        from src.backtesting.backtester import Backtester, BacktestConfig

        signals = self.generate_signals()
        config = BacktestConfig(
            initial_capital=initial_capital,
            commission=commission,
            slippage=0.0005,
            signal_lag=1,
        )
        bt = Backtester(
            prices=self.prices,
            signals=signals,
            config=config,
            benchmark=benchmark,
            strategy_name="Regime_Adaptive",
        )
        return bt.run()


# ═══════════════════════════════════════════════
#  FACTOR TIMING STRATEGY
# ═══════════════════════════════════════════════

class FactorTimingStrategy:
    """
    Dynamically tilt factor exposures based on macro conditions.

    In risk-on environments  → overweight momentum and quality.
    In risk-off environments → overweight value and low-vol.
    Reads risk sentiment from macro data.

    Parameters
    ----------
    universe_prices : pd.DataFrame
    macro_data      : pd.DataFrame   From MacroDataLoader.fetch_all().
    n_positions     : int
    """

    def __init__(
        self,
        universe_prices: pd.DataFrame,
        macro_data: Optional[pd.DataFrame] = None,
        n_positions: int = 10,
    ) -> None:
        self.prices = universe_prices.copy()
        self.macro = macro_data
        self.n_positions = n_positions

    def _get_factor_weights(self, date: pd.Timestamp) -> dict:
        """
        Determine factor weights based on macro environment.
        Returns weights for momentum, value, quality.
        """
        if self.macro is None or date not in self.macro.index:
            return {"momentum": 0.4, "value": 0.3, "quality": 0.3}

        row = self.macro.loc[date]
        risk_score = float(row.get("risk_on_score", 0))
        vix = float(row.get("VIX_level", 20))

        # Risk-on: tilt to momentum
        if risk_score > 0.3 and vix < 20:
            return {"momentum": 0.60, "value": 0.20, "quality": 0.20}
        # Risk-off: tilt to value and quality
        elif risk_score < -0.3 or vix > 30:
            return {"momentum": 0.10, "value": 0.45, "quality": 0.45}
        # Neutral
        else:
            return {"momentum": 0.40, "value": 0.30, "quality": 0.30}

    def generate_signals(self) -> pd.DataFrame:
        """Generate factor-timed signals."""
        from src.factor_models.momentum_factor import (
            MomentumFactor, ValueFactor, QualityFactor
        )

        mom_scores = MomentumFactor(self.prices).compute_signals(zscore=True)
        val_scores = ValueFactor(self.prices).compute_signals(zscore=True)
        qual_scores = QualityFactor(self.prices).compute_signals(zscore=True)

        common = (
            mom_scores.index
            .intersection(val_scores.index)
            .intersection(qual_scores.index)
        )

        signals = pd.DataFrame(0.0, index=common, columns=self.prices.columns)
        rebalance_dates = set(
            pd.Series(index=common, dtype=float)
            .resample("ME").last().index.normalize()
        )
        last_signal = pd.Series(0.0, index=self.prices.columns)

        for date in common:
            if pd.Timestamp(date).normalize() in rebalance_dates:
                fw = self._get_factor_weights(date)

                composite = pd.Series(0.0, index=self.prices.columns)
                if date in mom_scores.index:
                    composite += mom_scores.loc[date].reindex(self.prices.columns).fillna(0) * fw["momentum"]
                if date in val_scores.index:
                    composite += val_scores.loc[date].reindex(self.prices.columns).fillna(0) * fw["value"]
                if date in qual_scores.index:
                    composite += qual_scores.loc[date].reindex(self.prices.columns).fillna(0) * fw["quality"]

                valid = composite.dropna()
                if len(valid) >= self.n_positions:
                    new_sig = pd.Series(0.0, index=self.prices.columns)
                    longs = valid.nlargest(self.n_positions).index
                    new_sig[longs] = 1.0 / self.n_positions
                    last_signal = new_sig

            signals.loc[date] = last_signal

        return signals

    def backtest(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        benchmark: Optional[pd.Series] = None,
    ):
        """Run factor timing backtest."""
        from src.backtesting.backtester import Backtester, BacktestConfig

        signals = self.generate_signals()
        config = BacktestConfig(
            initial_capital=initial_capital,
            commission=commission,
            slippage=0.0005,
            signal_lag=1,
        )
        bt = Backtester(
            prices=self.prices,
            signals=signals,
            config=config,
            benchmark=benchmark,
            strategy_name="Factor_Timing",
        )
        return bt.run()
