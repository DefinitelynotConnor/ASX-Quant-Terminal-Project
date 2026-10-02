"""
macro_data_loader.py
====================
Macroeconomic data ingestion for regime identification and factor modelling.

Data Sources
------------
    Yahoo Finance   : VIX, commodity prices, FX rates
    pandas-datareader: FRED economic data (US rates, inflation)
    Manual proxies  : AUS-specific macro variables via Yahoo Finance

Variables Fetched
-----------------
    Interest Rates  : RBA cash rate proxy, US 10Y Treasury, yield curve
    Inflation       : CPI proxies
    Volatility      : VIX (US fear index), AXVI (ASX volatility)
    Commodities     : Iron ore proxy, Gold, Oil (Brent)
    FX              : AUD/USD exchange rate
    Credit          : Investment grade credit spread proxy

Usage
-----
    from src.data.macro_data_loader import MacroDataLoader

    macro = MacroDataLoader()
    data = macro.fetch_all(start="2019-01-01")
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# Optional pandas-datareader for FRED data
try:
    import pandas_datareader.data as web
    HAS_DATAREADER = True
except ImportError:
    HAS_DATAREADER = False


# ─────────────────────────────────────────────
#  Macro variable definitions
# ─────────────────────────────────────────────

# Yahoo Finance tickers for macro proxies
MACRO_TICKERS_YF = {
    # Volatility
    "VIX": "^VIX",               # CBOE Volatility Index (US fear gauge)

    # FX
    "AUDUSD": "AUDUSD=X",        # Australian Dollar vs USD

    # Commodities
    "Gold": "GC=F",              # Gold futures
    "Oil_Brent": "BZ=F",         # Brent crude oil futures
    "Copper": "HG=F",            # Copper futures (China proxy)

    # Equity indices
    "ASX200": "^AXJO",           # ASX 200 Index
    "SP500": "^GSPC",            # S&P 500
    "Shanghai": "000001.SS",     # Shanghai Composite (China exposure)

    # Bond proxies via ETFs
    "AUS_10Y_Bond": "^TNX",      # US 10Y Treasury (proxy for global rates)
}

# FRED series for US macro data
FRED_SERIES = {
    "US_CPI_YoY": "CPIAUCSL",           # US CPI all items
    "US_Fed_Funds": "FEDFUNDS",          # US Federal Funds Rate
    "US_10Y_Treasury": "GS10",           # 10-Year Treasury constant maturity
    "US_2Y_Treasury": "GS2",             # 2-Year Treasury constant maturity
    "US_Yield_Curve": None,              # Computed as 10Y - 2Y
    "US_Unemployment": "UNRATE",         # Unemployment rate
    "US_ISM_Manufacturing": "MANEMP",    # Manufacturing employment
}


# ─────────────────────────────────────────────
#  MacroDataLoader
# ─────────────────────────────────────────────

class MacroDataLoader:
    """
    Fetch and cache macroeconomic data for use in regime detection
    and factor model construction.

    Parameters
    ----------
    cache_dir         : str   Local cache directory.
    cache_expiry_days : int   Cache refresh frequency.
    """

    def __init__(
        self,
        cache_dir: str = "data/macro",
        cache_expiry_days: int = 1,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry = timedelta(days=cache_expiry_days)

    # ──────────────────────────────────────
    #  Cache helpers
    # ──────────────────────────────────────

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.parquet"

    def _is_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return age < self.cache_expiry

    # ──────────────────────────────────────
    #  Yahoo Finance macro data
    # ──────────────────────────────────────

    def fetch_yf_macro(
        self,
        start: str = "2010-01-01",
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch all Yahoo Finance macro variables and return as a single
        aligned DataFrame of daily observations.

        Returns
        -------
        pd.DataFrame  index = date, columns = macro variable names.
        """
        end = end or datetime.today().strftime("%Y-%m-%d")
        cache_path = self._cache_path(f"yf_macro_{start}_{end}")

        if use_cache and self._is_valid(cache_path):
            return pd.read_parquet(cache_path)

        tickers = list(MACRO_TICKERS_YF.values())
        names = list(MACRO_TICKERS_YF.keys())

        try:
            raw = yf.download(
                tickers, start=start, end=end,
                auto_adjust=True, progress=False,
            )

            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"].copy()
            else:
                prices = raw.copy()

            prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)

            # Rename columns to friendly names
            rename_map = {v: k for k, v in MACRO_TICKERS_YF.items()}
            prices = prices.rename(columns=rename_map)

            # Forward-fill (macro series have gaps over weekends/holidays)
            prices = prices.ffill()

            if use_cache:
                prices.to_parquet(cache_path)

            return prices

        except Exception as e:
            print(f"  [Macro YF] Warning: {e}")
            return pd.DataFrame()

    # ──────────────────────────────────────
    #  FRED macro data
    # ──────────────────────────────────────

    def fetch_fred_macro(
        self,
        start: str = "2010-01-01",
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch FRED macroeconomic time series.
        Requires pandas-datareader.

        Returns
        -------
        pd.DataFrame  Monthly observations, forward-filled to daily.
        """
        if not HAS_DATAREADER:
            print("  [FRED] pandas-datareader not installed. "
                  "Run: pip install pandas-datareader")
            return pd.DataFrame()

        end = end or datetime.today().strftime("%Y-%m-%d")
        cache_path = self._cache_path(f"fred_macro_{start}_{end}")

        if use_cache and self._is_valid(cache_path):
            return pd.read_parquet(cache_path)

        frames = {}
        series_to_fetch = {
            k: v for k, v in FRED_SERIES.items()
            if v is not None
        }

        for name, fred_code in series_to_fetch.items():
            try:
                s = web.DataReader(
                    fred_code, "fred",
                    start=start, end=end,
                )
                frames[name] = s.iloc[:, 0]
                print(f"  [FRED] Fetched {name} ({fred_code})")
            except Exception as e:
                print(f"  [FRED] Could not fetch {name}: {e}")

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)

        # Compute yield curve spread (10Y - 2Y)
        if "US_10Y_Treasury" in df and "US_2Y_Treasury" in df:
            df["US_Yield_Curve"] = df["US_10Y_Treasury"] - df["US_2Y_Treasury"]

        # Resample to daily and forward-fill (FRED is monthly)
        idx = pd.date_range(start=start, end=end, freq="B")
        df = df.reindex(idx).ffill()

        if use_cache:
            df.to_parquet(cache_path)

        return df

    # ──────────────────────────────────────
    #  Derived macro features
    # ──────────────────────────────────────

    def compute_macro_features(
        self,
        macro_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute derived macro features useful for regime detection.

        Features computed
        -----------------
        VIX_regime      : VIX level categorised as Low/Medium/High/Extreme
        VIX_change      : 20-day change in VIX
        AUDUSD_momentum : 20-day AUD/USD return
        Gold_trend      : 50-day vs 200-day gold price ratio
        Oil_trend       : 50-day vs 200-day oil price ratio
        Yield_slope     : Sign of yield curve (positive = normal, negative = inverted)
        Risk_on_off     : Composite risk sentiment score
        """
        df = macro_df.copy()
        features = pd.DataFrame(index=df.index)

        # VIX features
        if "VIX" in df.columns:
            features["VIX_level"] = df["VIX"]
            features["VIX_20d_change"] = df["VIX"].pct_change(20)
            features["VIX_regime"] = pd.cut(
                df["VIX"],
                bins=[0, 15, 20, 30, 100],
                labels=["Low", "Medium", "High", "Extreme"],
            ).astype(str)
            features["VIX_zscore"] = (
                (df["VIX"] - df["VIX"].rolling(252).mean())
                / df["VIX"].rolling(252).std()
            )

        # AUD/USD momentum
        if "AUDUSD" in df.columns:
            features["AUDUSD"] = df["AUDUSD"]
            features["AUDUSD_20d_ret"] = df["AUDUSD"].pct_change(20)
            features["AUDUSD_trend"] = np.where(
                df["AUDUSD"] > df["AUDUSD"].rolling(50).mean(), 1, -1
            )

        # Commodity trends
        for commodity in ["Gold", "Oil_Brent", "Copper"]:
            if commodity in df.columns:
                ma50 = df[commodity].rolling(50).mean()
                ma200 = df[commodity].rolling(200).mean()
                features[f"{commodity}_trend"] = np.where(ma50 > ma200, 1, -1)
                features[f"{commodity}_20d_ret"] = df[commodity].pct_change(20)

        # Yield curve
        if "US_Yield_Curve" in df.columns:
            features["yield_curve"] = df["US_Yield_Curve"]
            features["yield_inverted"] = (df["US_Yield_Curve"] < 0).astype(int)

        # ASX market momentum
        if "ASX200" in df.columns:
            asx = df["ASX200"]
            features["ASX200_20d_ret"] = asx.pct_change(20)
            features["ASX200_above_200ma"] = (
                asx > asx.rolling(200).mean()
            ).astype(int)

        # Composite risk-on / risk-off score (-1 to +1)
        scores = []
        if "AUDUSD_trend" in features:
            scores.append(features["AUDUSD_trend"])
        if "Gold_trend" in features:
            scores.append(-features["Gold_trend"])   # gold rises in risk-off
        if "ASX200_above_200ma" in features:
            scores.append(features["ASX200_above_200ma"] * 2 - 1)

        if scores:
            features["risk_on_score"] = pd.concat(scores, axis=1).mean(axis=1)

        return features.dropna(how="all")

    # ──────────────────────────────────────
    #  Master fetch
    # ──────────────────────────────────────

    def fetch_all(
        self,
        start: str = "2010-01-01",
        end: Optional[str] = None,
        include_fred: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch all macro data (Yahoo Finance + FRED) and return a single
        combined DataFrame with derived features.

        Parameters
        ----------
        start        : str   Start date.
        end          : str   End date.
        include_fred : bool  Include FRED data (requires pandas-datareader).
        use_cache    : bool  Use local cache.

        Returns
        -------
        pd.DataFrame  Daily macro variables + derived features.
        """
        end = end or datetime.today().strftime("%Y-%m-%d")
        print("Fetching macroeconomic data…")

        # Yahoo Finance macro
        yf_data = self.fetch_yf_macro(start, end, use_cache)

        # FRED macro
        fred_data = pd.DataFrame()
        if include_fred and HAS_DATAREADER:
            fred_data = self.fetch_fred_macro(start, end, use_cache)

        # Combine
        if not yf_data.empty and not fred_data.empty:
            combined = pd.concat([yf_data, fred_data], axis=1)
        elif not yf_data.empty:
            combined = yf_data
        elif not fred_data.empty:
            combined = fred_data
        else:
            print("  [Macro] Warning: no macro data retrieved.")
            return pd.DataFrame()

        # Compute derived features
        features = self.compute_macro_features(combined)
        result = pd.concat([combined, features], axis=1)
        result = result.loc[start:end]

        print(f"  [Done] {len(result.columns)} macro variables, "
              f"{len(result)} observations.")
        return result

    # ──────────────────────────────────────
    #  Specific variable getters
    # ──────────────────────────────────────

    def get_vix(
        self, start: str, end: Optional[str] = None
    ) -> pd.Series:
        """Return the VIX daily close series."""
        df = self.fetch_yf_macro(start, end)
        if "VIX" in df.columns:
            return df["VIX"].dropna()
        return pd.Series(dtype=float, name="VIX")

    def get_audusd(
        self, start: str, end: Optional[str] = None
    ) -> pd.Series:
        """Return the AUD/USD daily exchange rate."""
        df = self.fetch_yf_macro(start, end)
        if "AUDUSD" in df.columns:
            return df["AUDUSD"].dropna()
        return pd.Series(dtype=float, name="AUDUSD")

    def get_risk_sentiment(
        self, start: str, end: Optional[str] = None
    ) -> pd.Series:
        """
        Return the composite risk-on/risk-off score.
        Values closer to +1 = risk-on, closer to -1 = risk-off.
        """
        df = self.fetch_all(start, end)
        if "risk_on_score" in df.columns:
            return df["risk_on_score"].dropna()
        return pd.Series(dtype=float, name="risk_on_score")

    def macro_summary(self, start: str, end: Optional[str] = None) -> pd.DataFrame:
        """
        Return a summary table of the latest macro variable values.
        """
        df = self.fetch_all(start, end)
        if df.empty:
            return pd.DataFrame()

        latest = df.iloc[-1]
        summary = pd.DataFrame({
            "Variable": latest.index,
            "Latest Value": latest.values,
            "1M Change": (df.iloc[-1] - df.iloc[-21]).values if len(df) > 21 else None,
            "1Y Z-Score": (
                (df.iloc[-1] - df.tail(252).mean()) / df.tail(252).std()
            ).values,
        })
        return summary.set_index("Variable").dropna(how="all")
