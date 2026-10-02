"""
data_cleaning.py
================
Data quality, standardisation, and preprocessing pipeline for ASX price data.

Handles
-------
    Outlier detection and removal
    Gap filling and forward/backward filling
    Returns winsorisation
    Price normalisation
    Corporate action detection (large gaps)
    Data alignment across multiple series
    Train/test split for research

Usage
-----
    from src.data.data_cleaning import DataCleaner

    cleaner = DataCleaner()
    clean_prices = cleaner.clean_prices(raw_prices)
    clean_returns = cleaner.clean_returns(raw_returns)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


class DataCleaner:
    """
    Preprocessing pipeline for ASX equity price and return data.

    Parameters
    ----------
    max_gap_fill  : int    Max consecutive NaN days to forward-fill.
    outlier_std   : float  Z-score threshold for outlier detection.
    winsor_pct    : float  Percentile for return winsorisation (e.g. 0.01 = 1%).
    min_price     : float  Minimum valid price (removes penny stocks / data errors).
    """

    def __init__(
        self,
        max_gap_fill: int = 5,
        outlier_std: float = 4.0,
        winsor_pct: float = 0.01,
        min_price: float = 0.10,
    ) -> None:
        self.max_gap_fill = max_gap_fill
        self.outlier_std = outlier_std
        self.winsor_pct = winsor_pct
        self.min_price = min_price

    # ──────────────────────────────────────
    #  Price cleaning
    # ──────────────────────────────────────

    def clean_prices(
        self,
        prices: pd.DataFrame,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Full price cleaning pipeline.

        Steps
        -----
        1. Remove prices below minimum threshold
        2. Forward-fill small gaps
        3. Remove tickers with excessive missing data
        4. Sort index chronologically
        5. Remove duplicate dates

        Parameters
        ----------
        prices  : pd.DataFrame  Raw close prices, columns = tickers.
        verbose : bool          Print cleaning summary.

        Returns
        -------
        pd.DataFrame  Cleaned price DataFrame.
        """
        df = prices.copy()

        # Sort and deduplicate
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # Remove sub-minimum prices
        df[df < self.min_price] = np.nan

        # Forward-fill small gaps
        df = df.ffill(limit=self.max_gap_fill)

        # Drop tickers where >20% of data is missing
        missing_pct = df.isna().mean()
        drop = missing_pct[missing_pct > 0.20].index.tolist()
        if drop and verbose:
            print(f"  [Clean] Dropping {len(drop)} tickers (>20% missing): {drop}")
        df = df.drop(columns=drop)

        # Drop rows that are entirely NaN
        df = df.dropna(how="all")

        if verbose:
            print(f"  [Clean] Prices: {len(prices.columns)} → "
                  f"{len(df.columns)} tickers, {len(df)} rows.")

        return df

    # ──────────────────────────────────────
    #  Returns cleaning
    # ──────────────────────────────────────

    def clean_returns(
        self,
        returns: pd.DataFrame | pd.Series,
        method: str = "winsorise",
        verbose: bool = False,
    ) -> pd.DataFrame | pd.Series:
        """
        Clean a daily return series.

        Parameters
        ----------
        returns : pd.DataFrame or pd.Series
        method  : str
            'winsorise'  — clip extreme returns at percentile thresholds
            'zscore'     — remove returns beyond N standard deviations
            'none'       — no outlier treatment

        Returns
        -------
        Same type as input.
        """
        is_series = isinstance(returns, pd.Series)
        df = returns.to_frame() if is_series else returns.copy()

        if method == "winsorise":
            df = self._winsorise(df)
        elif method == "zscore":
            df = self._zscore_filter(df)

        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        if verbose:
            n_cleaned = df.isna().sum().sum() - returns.isna().sum().sum()
            print(f"  [Clean] {n_cleaned} return observations cleaned.")

        return df.iloc[:, 0] if is_series else df

    def _winsorise(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorise returns at the lower and upper percentile thresholds.
        Extreme values are clipped rather than removed.
        """
        p_low = self.winsor_pct
        p_high = 1 - self.winsor_pct

        for col in returns.columns:
            col_data = returns[col].dropna()
            if len(col_data) < 30:
                continue
            lower = col_data.quantile(p_low)
            upper = col_data.quantile(p_high)
            returns[col] = returns[col].clip(lower=lower, upper=upper)

        return returns

    def _zscore_filter(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Replace returns beyond N standard deviations with NaN.
        """
        for col in returns.columns:
            col_data = returns[col].dropna()
            if len(col_data) < 30:
                continue
            z = np.abs(stats.zscore(col_data))
            outlier_idx = col_data[z > self.outlier_std].index
            returns.loc[outlier_idx, col] = np.nan

        return returns

    # ──────────────────────────────────────
    #  Corporate action detection
    # ──────────────────────────────────────

    def detect_corporate_actions(
        self,
        prices: pd.DataFrame,
        threshold: float = 0.25,
    ) -> dict[str, list[dict]]:
        """
        Detect likely corporate actions (large single-day price moves that
        may indicate stock splits, dividends, or data errors).

        Parameters
        ----------
        prices    : pd.DataFrame  Unadjusted close prices.
        threshold : float         Flag moves larger than this percentage.

        Returns
        -------
        dict  {ticker: [{'date': ..., 'return': ...}]}
        """
        returns = prices.pct_change()
        actions = {}

        for ticker in returns.columns:
            col = returns[ticker].dropna()
            extreme = col[col.abs() > threshold]
            if len(extreme) > 0:
                actions[ticker] = [
                    {"date": str(d.date()), "return": round(float(r), 4)}
                    for d, r in extreme.items()
                ]

        return actions

    # ──────────────────────────────────────
    #  Data alignment
    # ──────────────────────────────────────

    def align_series(
        self,
        *series: pd.DataFrame | pd.Series,
        method: str = "inner",
        fill_method: Optional[str] = "ffill",
    ) -> list[pd.DataFrame | pd.Series]:
        """
        Align multiple DataFrames or Series to a common date index.

        Parameters
        ----------
        *series     : Any number of DataFrames or Series.
        method      : 'inner' (common dates only) | 'outer' (all dates).
        fill_method : 'ffill' | 'bfill' | None (no fill after join).

        Returns
        -------
        list  Aligned series in the same order as input.
        """
        if len(series) == 0:
            return []

        # Build common index
        indices = [s.index for s in series]
        if method == "inner":
            common_index = indices[0]
            for idx in indices[1:]:
                common_index = common_index.intersection(idx)
        else:
            common_index = indices[0]
            for idx in indices[1:]:
                common_index = common_index.union(idx)

        aligned = []
        for s in series:
            s_aligned = s.reindex(common_index)
            if fill_method == "ffill":
                s_aligned = s_aligned.ffill()
            elif fill_method == "bfill":
                s_aligned = s_aligned.bfill()
            aligned.append(s_aligned)

        return aligned

    # ──────────────────────────────────────
    #  Normalisation
    # ──────────────────────────────────────

    def normalise_prices(
        self,
        prices: pd.DataFrame,
        method: str = "rebase",
        base: float = 100.0,
    ) -> pd.DataFrame:
        """
        Normalise price series for comparison.

        Parameters
        ----------
        prices : pd.DataFrame
        method : str
            'rebase'     — rebase to base value at start (default 100)
            'zscore'     — standardise to zero mean, unit variance
            'minmax'     — scale to [0, 1]
        base   : float  Starting value for rebase method.

        Returns
        -------
        pd.DataFrame  Normalised prices.
        """
        if method == "rebase":
            first_valid = prices.bfill().iloc[0]
            return prices.div(first_valid) * base

        elif method == "zscore":
            return (prices - prices.mean()) / prices.std()

        elif method == "minmax":
            return (prices - prices.min()) / (prices.max() - prices.min())

        raise ValueError(f"Unknown normalisation method: {method}")

    # ──────────────────────────────────────
    #  Cross-sectional standardisation
    # ──────────────────────────────────────

    def cross_sectional_zscore(
        self,
        data: pd.DataFrame,
        clip: float = 3.0,
    ) -> pd.DataFrame:
        """
        Standardise each row (date) cross-sectionally.
        Used for factor signal normalisation.

        For each date: z_i = (x_i - mean(x)) / std(x)

        Parameters
        ----------
        data : pd.DataFrame  Rows = dates, columns = assets.
        clip : float         Clip z-scores beyond this value.

        Returns
        -------
        pd.DataFrame  Cross-sectionally standardised values.
        """
        z = data.sub(data.mean(axis=1), axis=0).div(data.std(axis=1), axis=0)
        return z.clip(-clip, clip)

    def cross_sectional_rank(
        self,
        data: pd.DataFrame,
        pct: bool = True,
    ) -> pd.DataFrame:
        """
        Rank assets cross-sectionally on each date.

        Parameters
        ----------
        data : pd.DataFrame
        pct  : bool   If True, return percentile ranks [0, 1].

        Returns
        -------
        pd.DataFrame  Cross-sectional ranks.
        """
        return data.rank(axis=1, pct=pct)

    # ──────────────────────────────────────
    #  Train / test split
    # ──────────────────────────────────────

    def train_test_split(
        self,
        data: pd.DataFrame | pd.Series,
        test_pct: float = 0.20,
        gap_days: int = 0,
    ) -> tuple:
        """
        Time-series aware train/test split.
        Always splits chronologically — no shuffling.

        Parameters
        ----------
        data      : pd.DataFrame or pd.Series
        test_pct  : float   Fraction of data to use as test set.
        gap_days  : int     Gap between train and test to prevent leakage.

        Returns
        -------
        tuple (train, test)
        """
        n = len(data)
        split = int(n * (1 - test_pct))
        train = data.iloc[:split - gap_days]
        test = data.iloc[split:]
        return train, test

    def rolling_train_test_windows(
        self,
        data: pd.DataFrame | pd.Series,
        train_days: int = 504,
        test_days: int = 63,
        step_days: int = 21,
    ) -> list[tuple]:
        """
        Generate rolling walk-forward train/test windows.
        Used for out-of-sample strategy validation.

        Parameters
        ----------
        train_days : int   Training window size in trading days.
        test_days  : int   Test window size in trading days.
        step_days  : int   Step size between windows.

        Returns
        -------
        list of (train_df, test_df) tuples.
        """
        windows = []
        n = len(data)
        start = 0

        while start + train_days + test_days <= n:
            train = data.iloc[start: start + train_days]
            test = data.iloc[start + train_days: start + train_days + test_days]
            windows.append((train, test))
            start += step_days

        return windows

    # ──────────────────────────────────────
    #  Missing data analysis
    # ──────────────────────────────────────

    def missing_data_report(
        self, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate a per-ticker missing data report.

        Returns
        -------
        pd.DataFrame  Columns: ticker, total_days, missing_days,
                               missing_pct, first_date, last_date
        """
        rows = []
        for ticker in prices.columns:
            col = prices[ticker]
            rows.append({
                "Ticker": ticker,
                "Total Days": len(col),
                "Missing Days": col.isna().sum(),
                "Missing %": col.isna().mean(),
                "First Date": str(col.dropna().index[0].date()) if not col.dropna().empty else "N/A",
                "Last Date": str(col.dropna().index[-1].date()) if not col.dropna().empty else "N/A",
                "Min Price": round(float(col.min()), 2),
                "Max Price": round(float(col.max()), 2),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("Missing %", ascending=False)
            .reset_index(drop=True)
        )

    # ──────────────────────────────────────
    #  Full pipeline
    # ──────────────────────────────────────

    def run_pipeline(
        self,
        prices: pd.DataFrame,
        clean_method: str = "winsorise",
        normalise: bool = False,
        verbose: bool = True,
    ) -> dict:
        """
        Run the complete data cleaning pipeline and return a result dict.

        Returns
        -------
        dict with keys:
            prices         : cleaned price DataFrame
            returns        : cleaned daily return DataFrame
            log_returns    : log return DataFrame
            report         : missing data report
            actions        : detected corporate actions
        """
        if verbose:
            print("Running data cleaning pipeline…")

        # Clean prices
        clean_p = self.clean_prices(prices, verbose=verbose)

        # Compute returns
        raw_ret = clean_p.pct_change().dropna()
        log_ret = np.log(clean_p / clean_p.shift(1)).dropna()

        # Clean returns
        clean_ret = self.clean_returns(raw_ret, method=clean_method, verbose=verbose)

        # Optional normalisation
        if normalise:
            clean_p = self.normalise_prices(clean_p)

        # Reports
        report = self.missing_data_report(prices)
        actions = self.detect_corporate_actions(prices)

        if verbose:
            print(f"  [Pipeline] Complete. "
                  f"{len(clean_p.columns)} tickers, {len(clean_p)} days.")

        return {
            "prices": clean_p,
            "returns": clean_ret,
            "log_returns": log_ret,
            "report": report,
            "corporate_actions": actions,
        }


# ─────────────────────────────────────────────
#  Module-level __init__
# ─────────────────────────────────────────────

def __init_data_module():
    """Create required data directories on import."""
    for d in ["data/raw", "data/processed", "data/macro"]:
        Path(d).mkdir(parents=True, exist_ok=True)


__init_data_module()
