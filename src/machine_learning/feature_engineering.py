"""
feature_engineering.py
======================
Feature engineering pipeline for ML alpha models.

Constructs a rich feature matrix from:
    - Technical indicators (momentum, volatility, volume)
    - Factor signals (momentum, value, quality)
    - Macro variables (VIX, rates, FX)
    - Regime indicators
    - Cross-sectional ranks and z-scores
    - Lagged return features

All features are designed to avoid look-ahead bias.
Every feature is computed using only data available at signal time.

Usage
-----
    from src.machine_learning.feature_engineering import FeatureEngineer

    fe = FeatureEngineer(prices, macro_data=macro_df)
    features = fe.build_features(target_horizon=21)
    X_train, X_test, y_train, y_test = fe.train_test_split(features)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """
    Build a comprehensive ML feature matrix for return prediction.

    Parameters
    ----------
    prices      : pd.DataFrame   Adjusted close prices, columns = tickers.
    macro_data  : pd.DataFrame   Macro variables from MacroDataLoader.
    ohlcv_data  : dict           {ticker: ohlcv_df} for volume features.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        macro_data: Optional[pd.DataFrame] = None,
        ohlcv_data: Optional[dict] = None,
    ) -> None:
        self.prices = prices.copy()
        self.macro = macro_data
        self.ohlcv = ohlcv_data
        self.returns = prices.pct_change()

    # ──────────────────────────────────────
    #  Return-based features
    # ──────────────────────────────────────

    def _return_features(self) -> dict[str, pd.DataFrame]:
        """Past return features at multiple horizons."""
        features = {}
        ret = self.returns

        for window in [1, 5, 10, 21, 63, 126, 252]:
            cumret = (1 + ret).rolling(window).apply(
                lambda x: x.prod() - 1, raw=True
            )
            features[f"ret_{window}d"] = cumret

        # 12-1 month momentum
        features["mom_12_1"] = (
            self.prices.shift(21) / self.prices.shift(252) - 1
        )

        # Short-term reversal
        features["reversal_1m"] = -(self.prices / self.prices.shift(21) - 1)

        # 52-week high ratio
        high_52w = self.prices.rolling(252).max()
        features["pct_52w_high"] = self.prices / high_52w.replace(0, np.nan)

        # Drawdown from peak
        rolling_max = self.prices.rolling(252).max()
        features["drawdown_1y"] = (self.prices / rolling_max) - 1

        return features

    # ──────────────────────────────────────
    #  Volatility features
    # ──────────────────────────────────────

    def _volatility_features(self) -> dict[str, pd.DataFrame]:
        """Volatility-based features."""
        features = {}
        ret = self.returns

        for window in [5, 21, 63]:
            features[f"vol_{window}d"] = ret.rolling(window).std() * np.sqrt(252)

        # Vol of vol (volatility uncertainty)
        vol_21 = ret.rolling(21).std() * np.sqrt(252)
        features["vol_of_vol"] = vol_21.rolling(63).std()

        # Volatility regime: current vol vs 1-year average
        features["vol_ratio"] = (
            ret.rolling(21).std() / ret.rolling(252).std()
        ).replace(0, np.nan)

        # EWMA volatility
        features["ewma_vol"] = (
            ret.ewm(alpha=0.06, adjust=False).std() * np.sqrt(252)
        )

        # Skewness of returns
        features["skew_63d"] = ret.rolling(63).skew()

        # Excess kurtosis
        features["kurt_63d"] = ret.rolling(63).kurt()

        return features

    # ──────────────────────────────────────
    #  Technical indicator features
    # ──────────────────────────────────────

    def _technical_features(self) -> dict[str, pd.DataFrame]:
        """Price-based technical indicators."""
        features = {}
        p = self.prices

        # Moving average ratios (price relative to MA)
        for ma in [10, 20, 50, 200]:
            sma = p.rolling(ma).mean()
            features[f"price_to_sma_{ma}"] = (p / sma.replace(0, np.nan)) - 1

        # MA crossover signals
        sma_20 = p.rolling(20).mean()
        sma_50 = p.rolling(50).mean()
        sma_200 = p.rolling(200).mean()
        features["sma_20_50_cross"] = (sma_20 - sma_50) / sma_50.replace(0, np.nan)
        features["sma_50_200_cross"] = (sma_50 - sma_200) / sma_200.replace(0, np.nan)

        # RSI
        delta = p.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        features["rsi_14"] = 100 - 100 / (1 + rs)

        # Bollinger Band position
        bb_mid = p.rolling(20).mean()
        bb_std = p.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        features["bb_pct_b"] = (p - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

        # MACD normalised by price
        ema12 = p.ewm(span=12, adjust=False).mean()
        ema26 = p.ewm(span=26, adjust=False).mean()
        features["macd_signal"] = (ema12 - ema26) / p.replace(0, np.nan)

        # Rate of change
        for roc_w in [5, 21, 63]:
            features[f"roc_{roc_w}d"] = p / p.shift(roc_w).replace(0, np.nan) - 1

        return features

    # ──────────────────────────────────────
    #  Volume features
    # ──────────────────────────────────────

    def _volume_features(self) -> dict[str, pd.DataFrame]:
        """Volume-based features (requires OHLCV data)."""
        features = {}

        if self.ohlcv is None:
            return features

        for ticker in self.prices.columns:
            if ticker not in self.ohlcv:
                continue
            vol_series = self.ohlcv[ticker].get("Volume")
            if vol_series is None:
                continue

            # Volume ratio (current vs average)
            vol_ma = vol_series.rolling(20).mean()
            vol_ratio = vol_series / vol_ma.replace(0, np.nan)

            # OBV normalised
            direction = np.sign(self.returns[ticker].fillna(0))
            obv = (direction * vol_series).cumsum()
            obv_norm = obv / vol_series.rolling(252).mean().replace(0, np.nan)

            features.setdefault("vol_ratio", {})[ticker] = vol_ratio
            features.setdefault("obv_norm", {})[ticker] = obv_norm

        result = {}
        for fname, ticker_dict in features.items():
            result[fname] = pd.DataFrame(ticker_dict)

        return result

    # ──────────────────────────────────────
    #  Cross-sectional features
    # ──────────────────────────────────────

    def _cross_sectional_features(
        self, raw_features: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """
        Cross-sectional standardisation of all features.
        For each date, z-score each feature across all assets.
        This ensures features are comparable across tickers.
        """
        cs_features = {}
        for fname, df in raw_features.items():
            if not isinstance(df, pd.DataFrame):
                continue
            mean = df.mean(axis=1)
            std = df.std(axis=1).replace(0, np.nan)
            cs_z = df.sub(mean, axis=0).div(std, axis=0).clip(-3, 3)
            cs_features[f"cs_{fname}"] = cs_z
        return cs_features

    # ──────────────────────────────────────
    #  Macro features
    # ──────────────────────────────────────

    def _macro_features(self) -> dict[str, pd.Series]:
        """Macro variable features (broadcast to all tickers)."""
        if self.macro is None:
            return {}

        features = {}
        macro_cols = [
            "VIX_level", "VIX_zscore", "VIX_20d_change",
            "AUDUSD_20d_ret", "AUDUSD_trend",
            "Gold_20d_ret", "risk_on_score",
            "yield_curve", "yield_inverted",
            "ASX200_20d_ret", "ASX200_above_200ma",
        ]

        for col in macro_cols:
            if col in self.macro.columns:
                features[f"macro_{col}"] = self.macro[col]

        return features

    # ──────────────────────────────────────
    #  Master feature builder
    # ──────────────────────────────────────

    def build_features(
        self,
        ticker: Optional[str] = None,
        target_horizon: int = 21,
        include_target: bool = True,
        cs_normalise: bool = True,
    ) -> pd.DataFrame:
        """
        Build the complete feature matrix for a single ticker.

        Parameters
        ----------
        ticker          : str, optional   Build for one ticker. If None,
                                          returns a stacked panel.
        target_horizon  : int             Forward return horizon for label.
        include_target  : bool            Include forward return as target column.
        cs_normalise    : bool            Include cross-sectional z-scored features.

        Returns
        -------
        pd.DataFrame  Features + optional target column.
        """
        tickers = [ticker] if ticker else list(self.prices.columns)

        ret_feats = self._return_features()
        vol_feats = self._volatility_features()
        tech_feats = self._technical_features()

        all_raw = {**ret_feats, **vol_feats, **tech_feats}

        cs_feats = {}
        if cs_normalise:
            cs_feats = self._cross_sectional_features(all_raw)

        macro_feats = self._macro_features()

        all_frames = []

        for t in tickers:
            rows = {}

            # Raw features
            for fname, df in all_raw.items():
                if isinstance(df, pd.DataFrame) and t in df.columns:
                    rows[fname] = df[t]
                elif isinstance(df, pd.Series):
                    rows[fname] = df

            # CS features
            for fname, df in cs_feats.items():
                if isinstance(df, pd.DataFrame) and t in df.columns:
                    rows[fname] = df[t]

            # Macro features (broadcast)
            for fname, series in macro_feats.items():
                rows[fname] = series.reindex(self.prices.index).ffill()

            ticker_df = pd.DataFrame(rows, index=self.prices.index)

            # Target: forward return
            if include_target:
                fwd_ret = self.returns[t].shift(-target_horizon) if t in self.returns.columns else None
                if fwd_ret is not None:
                    ticker_df["target"] = (
                        (1 + self.returns[t]).rolling(target_horizon).apply(
                            lambda x: x.prod() - 1, raw=True
                        ).shift(-target_horizon)
                    )

            ticker_df["ticker"] = t
            all_frames.append(ticker_df)

        if not all_frames:
            return pd.DataFrame()

        result = pd.concat(all_frames)
        result = result.replace([np.inf, -np.inf], np.nan)
        return result

    def build_universe_features(
        self,
        target_horizon: int = 21,
    ) -> pd.DataFrame:
        """
        Build features for all tickers stacked into a panel DataFrame.
        Index: (date, ticker) MultiIndex.
        """
        df = self.build_features(
            ticker=None,
            target_horizon=target_horizon,
            include_target=True,
        )
        df = df.reset_index().rename(columns={"index": "date"})
        df = df.set_index(["date", "ticker"])
        return df

    def get_feature_names(self) -> list[str]:
        """Return list of all feature column names."""
        sample = self.build_features(
            ticker=self.prices.columns[0],
            include_target=False,
        )
        return [c for c in sample.columns if c != "ticker"]

    def train_test_split(
        self,
        features: pd.DataFrame,
        test_pct: float = 0.20,
        target_col: str = "target",
        drop_cols: Optional[list] = None,
    ) -> tuple:
        """
        Time-series aware train/test split.

        Returns
        -------
        (X_train, X_test, y_train, y_test)
        """
        drop_cols = drop_cols or ["ticker", "target"]
        feature_cols = [c for c in features.columns if c not in drop_cols]

        clean = features.dropna(subset=[target_col])
        n = len(clean)
        split = int(n * (1 - test_pct))

        train = clean.iloc[:split]
        test = clean.iloc[split:]

        X_train = train[feature_cols].fillna(0)
        X_test = test[feature_cols].fillna(0)
        y_train = train[target_col]
        y_test = test[target_col]

        return X_train, X_test, y_train, y_test
