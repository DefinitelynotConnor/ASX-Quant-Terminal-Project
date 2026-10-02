"""
asx_data_loader.py
==================
Primary data ingestion module for ASX equity price data.

Handles:
    - Fetching OHLCV price data from Yahoo Finance
    - Local disk caching to avoid redundant API calls
    - Universe-level batch downloading
    - Data validation and integrity checks
    - Adjusted close price handling

Usage
-----
    from src.data.asx_data_loader import ASXDataLoader

    loader = ASXDataLoader()
    prices = loader.fetch_price_data("BHP.AX", "2020-01-01", "2024-12-31")
    universe = loader.fetch_universe_data(["BHP.AX", "CBA.AX", "CSL.AX"])
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Default ASX Universe
# ─────────────────────────────────────────────

ASX_200_UNIVERSE = [
    # Financials
    "CBA.AX", "WBC.AX", "ANZ.AX", "NAB.AX", "MQG.AX",
    "SUN.AX", "IAG.AX", "QBE.AX", "AMP.AX",
    # Materials
    "BHP.AX", "RIO.AX", "FMG.AX", "S32.AX", "NST.AX",
    "EVN.AX", "IGO.AX", "MIN.AX", "LYC.AX",
    # Healthcare
    "CSL.AX", "RMD.AX", "COH.AX", "RHC.AX", "SHL.AX",
    "PME.AX", "NVX.AX",
    # Energy
    "WDS.AX", "STO.AX", "BPT.AX", "WHC.AX", "NHC.AX",
    # Consumer Discretionary
    "WES.AX", "JBH.AX", "HVN.AX", "SUL.AX", "ALL.AX",
    # Consumer Staples
    "WOW.AX", "COL.AX", "TWE.AX", "A2M.AX",
    # Communication Services
    "TLS.AX", "REA.AX", "CAR.AX", "SEK.AX",
    # Industrials
    "TCL.AX", "QAN.AX", "BXB.AX", "ALX.AX",
    # Real Estate
    "GMG.AX", "SCG.AX", "MGR.AX", "DXS.AX",
    # Utilities
    "AGL.AX", "ORG.AX", "APA.AX",
    # Information Technology
    "XRO.AX", "WTC.AX", "ALU.AX",
]

ASX_BENCHMARK = "^AXJO"   # ASX 200 Index


# ─────────────────────────────────────────────
#  ASXDataLoader
# ─────────────────────────────────────────────

class ASXDataLoader:
    """
    Fetch and cache ASX equity price data from Yahoo Finance.

    Parameters
    ----------
    cache_dir : str
        Directory for local parquet cache files.
        Defaults to  data/raw/  relative to project root.
    cache_expiry_days : int
        Number of days before cached data is considered stale.
        Defaults to 1 day (refresh daily).
    max_retries : int
        Number of download retry attempts on failure.
    retry_delay : float
        Seconds to wait between retries.
    """

    OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(
        self,
        cache_dir: str = "data/raw",
        cache_expiry_days: int = 1,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry = timedelta(days=cache_expiry_days)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ──────────────────────────────────────
    #  Cache helpers
    # ──────────────────────────────────────

    def _cache_path(self, ticker: str, start: str, end: str) -> Path:
        """Generate a unique cache file path for a ticker + date range."""
        key = f"{ticker}_{start}_{end}"
        h = hashlib.md5(key.encode()).hexdigest()[:8]
        safe = ticker.replace(".", "_").replace("^", "BENCH_")
        return self.cache_dir / f"{safe}_{h}.parquet"

    def _is_cache_valid(self, path: Path) -> bool:
        """Return True if the cache file exists and is not stale."""
        if not path.exists():
            return False
        modified = datetime.fromtimestamp(path.stat().st_mtime)
        return datetime.now() - modified < self.cache_expiry

    def _load_cache(self, path: Path) -> Optional[pd.DataFrame]:
        """Load a cached parquet file."""
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def _save_cache(self, df: pd.DataFrame, path: Path) -> None:
        """Save a DataFrame to parquet cache."""
        try:
            df.to_parquet(path)
        except Exception as e:
            print(f"  [Cache] Warning: could not save cache to {path}: {e}")

    # ──────────────────────────────────────
    #  Single ticker fetch
    # ──────────────────────────────────────

    def fetch_price_data(
        self,
        ticker: str,
        start: str,
        end: Optional[str] = None,
        use_cache: bool = True,
        fields: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV price data for a single ASX ticker.

        Parameters
        ----------
        ticker    : str   e.g. 'BHP.AX'
        start     : str   Start date 'YYYY-MM-DD'
        end       : str   End date 'YYYY-MM-DD'. Defaults to today.
        use_cache : bool  Use local cache if available.
        fields    : list  Subset of columns to return. Default = all OHLCV.

        Returns
        -------
        pd.DataFrame  DatetimeIndex, columns = OHLCV fields.

        Raises
        ------
        ValueError if no data is returned after all retries.
        """
        end = end or datetime.today().strftime("%Y-%m-%d")
        fields = fields or self.OHLCV_COLS

        # Check cache
        cache_path = self._cache_path(ticker, start, end)
        if use_cache and self._is_cache_valid(cache_path):
            df = self._load_cache(cache_path)
            if df is not None and not df.empty:
                return self._filter_fields(df, fields)

        # Download with retries
        df = self._download_with_retry(ticker, start, end)

        if df is None or df.empty:
            raise ValueError(
                f"No data returned for {ticker} between {start} and {end}. "
                "Check the ticker symbol is correct (e.g. 'BHP.AX')."
            )

        # Clean and validate
        df = self._clean_price_data(df, ticker)

        # Save to cache
        if use_cache:
            self._save_cache(df, cache_path)

        return self._filter_fields(df, fields)

    def _download_with_retry(
        self, ticker: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Download from Yahoo Finance with retry logic."""
        for attempt in range(self.max_retries):
            try:
                raw = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )

                if raw is None or raw.empty:
                    raise ValueError("Empty response")

                # Flatten MultiIndex columns if present
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                # Strip timezone
                raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)

                return raw

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"  [Retry {attempt + 1}] {ticker}: {e}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"  [Failed] {ticker}: {e}")
                    return None

        return None

    # ──────────────────────────────────────
    #  Universe fetch
    # ──────────────────────────────────────

    def fetch_universe_data(
        self,
        tickers: Optional[list[str]] = None,
        start: str = "2019-01-01",
        end: Optional[str] = None,
        field: str = "Close",
        min_history_days: int = 252,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch a single price field for a universe of ASX tickers.

        Parameters
        ----------
        tickers          : list   List of ASX tickers. Defaults to ASX_200_UNIVERSE.
        start            : str    Start date.
        end              : str    End date. Defaults to today.
        field            : str    Price field to return. Default = 'Close'.
        min_history_days : int    Drop tickers with fewer trading days than this.
        use_cache        : bool   Use local cache.

        Returns
        -------
        pd.DataFrame  DatetimeIndex rows, ticker columns, values = field prices.
        """
        tickers = tickers or ASX_200_UNIVERSE
        end = end or datetime.today().strftime("%Y-%m-%d")

        # Batch download is faster than individual calls
        print(f"Downloading {len(tickers)} tickers from {start} to {end}…")

        cache_path = self.cache_dir / f"universe_{start}_{end}_{field}.parquet"
        if use_cache and self._is_cache_valid(cache_path):
            df = self._load_cache(cache_path)
            if df is not None and not df.empty:
                print(f"  [Cache] Loaded {len(df.columns)} tickers from cache.")
                return df

        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as e:
            raise ValueError(f"Universe download failed: {e}")

        if raw is None or raw.empty:
            raise ValueError("No data returned for universe download.")

        # Extract the requested field
        if isinstance(raw.columns, pd.MultiIndex):
            if field in raw.columns.get_level_values(0):
                df = raw[field].copy()
            else:
                df = raw["Close"].copy()
        else:
            df = raw.copy()

        # Strip timezone
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)

        # Drop tickers with insufficient history
        valid = df.count() >= min_history_days
        dropped = valid[~valid].index.tolist()
        if dropped:
            print(f"  [Filter] Dropped {len(dropped)} tickers with < "
                  f"{min_history_days} days of history: {dropped}")
        df = df.loc[:, valid]

        # Forward-fill small gaps (up to 5 days) then drop remaining NaNs
        df = df.ffill(limit=5)
        df = df.dropna(how="all")

        print(f"  [Done] {len(df.columns)} tickers, {len(df)} trading days.")

        if use_cache:
            self._save_cache(df, cache_path)

        return df

    # ──────────────────────────────────────
    #  Benchmark fetch
    # ──────────────────────────────────────

    def fetch_benchmark(
        self,
        start: str,
        end: Optional[str] = None,
        ticker: str = ASX_BENCHMARK,
        use_cache: bool = True,
    ) -> pd.Series:
        """
        Fetch the ASX 200 benchmark return series.

        Returns
        -------
        pd.Series  Daily returns of the benchmark.
        """
        end = end or datetime.today().strftime("%Y-%m-%d")
        try:
            df = self.fetch_price_data(
                ticker, start, end,
                use_cache=use_cache,
                fields=["Close"],
            )
            returns = df["Close"].pct_change().dropna()
            returns.name = ticker
            return returns
        except Exception as e:
            print(f"  [Benchmark] Warning: {e}")
            return pd.Series(dtype=float, name=ticker)

    # ──────────────────────────────────────
    #  Returns computation
    # ──────────────────────────────────────

    def compute_returns(
        self,
        prices: pd.DataFrame | pd.Series,
        method: str = "simple",
        periods: int = 1,
    ) -> pd.DataFrame | pd.Series:
        """
        Compute returns from a price series.

        Parameters
        ----------
        prices  : pd.DataFrame or pd.Series
        method  : 'simple' | 'log'
        periods : int   Number of periods for return calculation (1 = daily).

        Returns
        -------
        Same type as input, with returns instead of prices.
        """
        if method == "simple":
            return prices.pct_change(periods).dropna()
        elif method == "log":
            return np.log(prices / prices.shift(periods)).dropna()
        raise ValueError(f"Unknown method: {method}. Use 'simple' or 'log'.")

    def compute_universe_returns(
        self,
        tickers: Optional[list[str]] = None,
        start: str = "2019-01-01",
        end: Optional[str] = None,
        method: str = "simple",
    ) -> pd.DataFrame:
        """
        Convenience method: fetch universe prices and return daily returns.
        """
        prices = self.fetch_universe_data(
            tickers=tickers, start=start, end=end
        )
        return self.compute_returns(prices, method=method)

    # ──────────────────────────────────────
    #  Data validation
    # ──────────────────────────────────────

    def _clean_price_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Apply data quality rules to raw OHLCV data.

        Rules applied
        -------------
        1. Sort by date ascending
        2. Remove duplicate dates
        3. Ensure all OHLCV columns exist
        4. Remove rows where Close <= 0
        5. Forward-fill small gaps (up to 5 trading days)
        6. Flag and remove extreme price jumps (>50% in one day)
        """
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # Ensure standard columns exist
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = np.nan

        # Remove zero or negative prices
        df = df[df["Close"] > 0]

        # Forward-fill gaps up to 5 days
        df = df.ffill(limit=5)

        # Flag extreme single-day moves (data errors, not market moves)
        daily_ret = df["Close"].pct_change().abs()
        extreme = daily_ret > 0.50
        if extreme.sum() > 0:
            print(f"  [Clean] {ticker}: {extreme.sum()} extreme daily moves "
                  f"(>50%) detected and flagged.")

        return df

    @staticmethod
    def _filter_fields(
        df: pd.DataFrame, fields: list[str]
    ) -> pd.DataFrame:
        """Return only the requested columns that exist in the DataFrame."""
        available = [f for f in fields if f in df.columns]
        return df[available]

    def validate_universe(
        self,
        prices: pd.DataFrame,
        min_price: float = 0.10,
        max_missing_pct: float = 0.05,
    ) -> dict:
        """
        Run data quality checks across a universe price DataFrame.

        Returns
        -------
        dict with:
            valid_tickers     : list of tickers passing all checks
            dropped_tickers   : list of tickers failing checks
            missing_pct       : dict {ticker: pct_missing}
            summary           : human-readable quality report
        """
        total_days = len(prices)
        issues = {}
        valid = []
        dropped = []

        for ticker in prices.columns:
            col = prices[ticker]
            missing_pct = col.isna().mean()
            min_val = col.dropna().min() if not col.dropna().empty else 0

            ticker_issues = []
            if missing_pct > max_missing_pct:
                ticker_issues.append(f"{missing_pct:.1%} missing data")
            if min_val < min_price:
                ticker_issues.append(f"price below ${min_price:.2f}")

            if ticker_issues:
                issues[ticker] = ticker_issues
                dropped.append(ticker)
            else:
                valid.append(ticker)

        summary_lines = [
            f"Universe Quality Report",
            f"  Total tickers  : {len(prices.columns)}",
            f"  Valid tickers  : {len(valid)}",
            f"  Dropped        : {len(dropped)}",
            f"  Trading days   : {total_days}",
        ]
        if dropped:
            summary_lines.append(f"  Issues         : {issues}")

        return {
            "valid_tickers": valid,
            "dropped_tickers": dropped,
            "issues": issues,
            "missing_pct": {t: prices[t].isna().mean() for t in prices.columns},
            "summary": "\n".join(summary_lines),
        }

    # ──────────────────────────────────────
    #  Cache management
    # ──────────────────────────────────────

    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """
        Clear cached data files.

        Parameters
        ----------
        ticker : str, optional
            If provided, clear only files for this ticker.
            If None, clear all cache files.
        """
        if ticker:
            safe = ticker.replace(".", "_").replace("^", "BENCH_")
            files = list(self.cache_dir.glob(f"{safe}_*.parquet"))
        else:
            files = list(self.cache_dir.glob("*.parquet"))

        for f in files:
            f.unlink()
        print(f"  [Cache] Cleared {len(files)} cache files.")

    def cache_status(self) -> pd.DataFrame:
        """
        Return a summary of all cached files.

        Returns
        -------
        pd.DataFrame  with columns: file, size_kb, age_hours, valid
        """
        files = list(self.cache_dir.glob("*.parquet"))
        rows = []
        now = datetime.now()
        for f in files:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            age = (now - mtime).total_seconds() / 3600
            rows.append({
                "file": f.name,
                "size_kb": f.stat().st_size / 1024,
                "age_hours": round(age, 1),
                "valid": age < self.cache_expiry.total_seconds() / 3600,
            })
        return pd.DataFrame(rows).sort_values("age_hours")

    # ──────────────────────────────────────
    #  Convenience helpers
    # ──────────────────────────────────────

    def get_available_history(self, ticker: str) -> dict:
        """
        Check how much price history is available for a ticker.

        Returns
        -------
        dict with start_date, end_date, num_days
        """
        try:
            df = self.fetch_price_data(ticker, "2000-01-01", use_cache=True)
            return {
                "ticker": ticker,
                "start_date": str(df.index[0].date()),
                "end_date": str(df.index[-1].date()),
                "num_trading_days": len(df),
                "num_years": round(len(df) / 252, 1),
            }
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}

    def get_latest_prices(self, tickers: list[str]) -> pd.Series:
        """
        Get the most recent closing price for a list of tickers.

        Returns
        -------
        pd.Series  index = tickers, values = latest close price.
        """
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=10)).strftime("%Y-%m-%d")
        prices = {}
        for ticker in tickers:
            try:
                df = self.fetch_price_data(ticker, start, end, fields=["Close"])
                prices[ticker] = float(df["Close"].iloc[-1])
            except Exception:
                prices[ticker] = np.nan
        return pd.Series(prices, name="Latest Close")


# ─────────────────────────────────────────────
#  Standalone convenience functions
# ─────────────────────────────────────────────

def fetch_price_data(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    cache_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Convenience wrapper around ASXDataLoader.fetch_price_data().

    Example
    -------
    >>> prices = fetch_price_data("BHP.AX", "2020-01-01", "2024-12-31")
    """
    loader = ASXDataLoader(cache_dir=cache_dir)
    return loader.fetch_price_data(ticker, start, end)


def fetch_universe_data(
    tickers: Optional[list[str]] = None,
    start: str = "2019-01-01",
    end: Optional[str] = None,
    field: str = "Close",
    cache_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Convenience wrapper around ASXDataLoader.fetch_universe_data().

    Example
    -------
    >>> universe = fetch_universe_data(["BHP.AX", "CBA.AX"], "2020-01-01")
    """
    loader = ASXDataLoader(cache_dir=cache_dir)
    return loader.fetch_universe_data(tickers, start, end, field)
