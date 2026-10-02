"""
fama_french.py
==============
Fama-French 3-Factor Model implementation for ASX equities.

The Fama-French model extends CAPM with two additional factors:
    SMB (Small Minus Big)  : Size premium — small caps outperform large caps
    HML (High Minus Low)   : Value premium — high book/price stocks outperform

Model
-----
R_i - R_f = α + β_mkt × (R_m - R_f) + β_smb × SMB + β_hml × HML + ε

Factor Construction (from available ASX price data)
---------------------------------------------------
    MKT  : ASX 200 excess return
    SMB  : Return of small-cap portfolio minus large-cap portfolio
           (proxied by cross-sectional size quintile sorts)
    HML  : Return of high book-to-price minus low book-to-price
           (proxied by 12-month return reversal as value proxy)

Usage
-----
    from src.factor_models.fama_french import FamaFrench

    ff = FamaFrench(asset_returns, market_returns, universe_prices)
    results = ff.fit()
    factors = ff.get_factor_returns()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# ─────────────────────────────────────────────
#  Result Container
# ─────────────────────────────────────────────

@dataclass
class FamaFrenchResult:
    """Container for Fama-French regression results."""
    ticker: str = ""
    alpha: float = 0.0
    beta_mkt: float = 0.0
    beta_smb: float = 0.0
    beta_hml: float = 0.0
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    alpha_tstat: float = 0.0
    alpha_pvalue: float = 1.0
    alpha_significant: bool = False
    mkt_tstat: float = 0.0
    smb_tstat: float = 0.0
    hml_tstat: float = 0.0
    factor_contributions: dict = field(default_factory=dict)
    n_observations: int = 0

    def summary(self) -> str:
        return (
            f"\nFama-French 3-Factor: {self.ticker}\n"
            f"{'─' * 45}\n"
            f"  Alpha (ann.)  : {self.alpha:.2%}  "
            f"(t={self.alpha_tstat:.2f}, "
            f"{'✅' if self.alpha_significant else '❌'})\n"
            f"  β Market      : {self.beta_mkt:.4f}  (t={self.mkt_tstat:.2f})\n"
            f"  β SMB (Size)  : {self.beta_smb:.4f}  (t={self.smb_tstat:.2f})\n"
            f"  β HML (Value) : {self.beta_hml:.4f}  (t={self.hml_tstat:.2f})\n"
            f"  R²            : {self.r_squared:.4f}  "
            f"(Adj R²: {self.adj_r_squared:.4f})\n"
            f"  Observations  : {self.n_observations}\n"
        )


# ─────────────────────────────────────────────
#  FamaFrench
# ─────────────────────────────────────────────

class FamaFrench:
    """
    Fama-French 3-Factor Model.

    Parameters
    ----------
    asset_returns   : pd.Series or pd.DataFrame
        Daily returns for the asset(s) to analyse.
    market_returns  : pd.Series
        Daily market (benchmark) returns.
    universe_prices : pd.DataFrame, optional
        Close prices for a broad universe of ASX stocks.
        Used to construct SMB and HML factor proxies.
    factor_returns  : pd.DataFrame, optional
        Pre-computed factor return series with columns:
        ['MKT', 'SMB', 'HML']. If provided, skips factor construction.
    risk_free_rate  : float
        Annualised risk-free rate.
    periods_per_year: int
        Trading days per year.
    """

    FACTORS = ["MKT", "SMB", "HML"]

    def __init__(
        self,
        asset_returns: pd.Series | pd.DataFrame,
        market_returns: pd.Series,
        universe_prices: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0435,
        periods_per_year: int = 252,
    ) -> None:
        self.market = market_returns.dropna()
        self.universe = universe_prices
        self.rfr = risk_free_rate
        self.rfr_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        self.ppy = periods_per_year

        if isinstance(asset_returns, pd.Series):
            self.assets = asset_returns.dropna().to_frame()
        else:
            self.assets = asset_returns.dropna()

        # Build or use provided factor returns
        if factor_returns is not None:
            self._factors = factor_returns
        elif universe_prices is not None:
            self._factors = self._build_factors()
        else:
            self._factors = self._build_market_only_factors()

    # ──────────────────────────────────────
    #  Factor Construction
    # ──────────────────────────────────────

    def _build_factors(self) -> pd.DataFrame:
        """
        Construct SMB and HML factor proxies from universe price data.

        SMB (Size): Cross-sectional sort on 12-month trailing market cap proxy
                    (use cumulative return as a price-level proxy for size).
                    Long bottom 30% (small), short top 30% (large).

        HML (Value): Cross-sectional sort on book-to-market proxy.
                     Use 12-month price reversal as a value proxy
                     (high past losers = cheap/value, low past losers = growth).
                     Long top 30% (value), short bottom 30% (growth).
        """
        prices = self.universe.copy()
        ret = prices.pct_change().dropna()

        factors = {}

        # MKT excess return
        common_mkt = self.market.index.intersection(ret.index)
        factors["MKT"] = (self.market.loc[common_mkt] - self.rfr_daily)

        # SMB: sort on price level (proxy for size — higher price ≈ larger)
        size_proxy = prices.copy()
        smb_returns = []
        smb_dates = []

        # HML: sort on 12-month return reversal (negative 12M ret = value proxy)
        ret_12m = (1 + ret).rolling(252).apply(
            lambda x: x.prod() - 1, raw=True
        )
        hml_returns = []
        hml_dates = []

        for date in ret.index[252:]:
            if date not in size_proxy.index or date not in ret_12m.index:
                continue

            # Size sort
            size_row = size_proxy.loc[date].dropna()
            if len(size_row) < 6:
                continue
            n_q = max(int(len(size_row) * 0.30), 1)
            small = size_row.nsmallest(n_q).index.tolist()
            large = size_row.nlargest(n_q).index.tolist()

            small_tickers = [t for t in small if t in ret.columns]
            large_tickers = [t for t in large if t in ret.columns]

            if date in ret.index and small_tickers and large_tickers:
                r_small = float(ret.loc[date, small_tickers].mean())
                r_large = float(ret.loc[date, large_tickers].mean())
                smb_returns.append(r_small - r_large)
                smb_dates.append(date)

            # Value sort (12M reversal as value proxy)
            rev_row = ret_12m.loc[date].dropna()
            if len(rev_row) < 6:
                continue
            n_q = max(int(len(rev_row) * 0.30), 1)
            value = rev_row.nsmallest(n_q).index.tolist()   # past losers = value
            growth = rev_row.nlargest(n_q).index.tolist()   # past winners = growth

            value_tickers = [t for t in value if t in ret.columns]
            growth_tickers = [t for t in growth if t in ret.columns]

            if date in ret.index and value_tickers and growth_tickers:
                r_value = float(ret.loc[date, value_tickers].mean())
                r_growth = float(ret.loc[date, growth_tickers].mean())
                hml_returns.append(r_value - r_growth)
                hml_dates.append(date)

        if smb_dates:
            factors["SMB"] = pd.Series(smb_returns, index=smb_dates, name="SMB")
        if hml_dates:
            factors["HML"] = pd.Series(hml_returns, index=hml_dates, name="HML")

        df = pd.DataFrame(factors).dropna()
        return df

    def _build_market_only_factors(self) -> pd.DataFrame:
        """
        Fallback: build factors using market return only.
        SMB and HML are set to zero (degenerates to CAPM).
        """
        mkt_excess = self.market - self.rfr_daily
        df = pd.DataFrame({"MKT": mkt_excess})
        df["SMB"] = 0.0
        df["HML"] = 0.0
        return df

    # ──────────────────────────────────────
    #  Regression
    # ──────────────────────────────────────

    def _run_regression(
        self,
        ticker: str,
        returns: pd.Series,
    ) -> FamaFrenchResult:
        """
        Run OLS regression of asset excess returns on 3 factors.
        R_i - R_f = α + β_mkt×MKT + β_smb×SMB + β_hml×HML + ε
        """
        common = returns.index.intersection(self._factors.index)
        if len(common) < 60:
            return FamaFrenchResult(ticker=ticker)

        y = returns.loc[common] - self.rfr_daily
        X_df = self._factors.loc[common][
            [f for f in self.FACTORS if f in self._factors.columns]
        ]

        if HAS_STATSMODELS:
            X = sm.add_constant(X_df.values)
            model = sm.OLS(y.values, X)
            fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": 5})

            params = fit.params
            tstats = fit.tvalues
            pvals = fit.pvalues
            r2 = float(fit.rsquared)
            adj_r2 = float(fit.rsquared_adj)
            n = int(fit.nobs)
        else:
            X = np.column_stack([np.ones(len(y)), X_df.values])
            try:
                params, _, _, _ = np.linalg.lstsq(X, y.values, rcond=None)
            except Exception:
                return FamaFrenchResult(ticker=ticker)

            y_hat = X @ params
            residuals = y.values - y_hat
            n = len(y)
            k = len(params)
            mse = np.dot(residuals, residuals) / (n - k)
            cov = mse * np.linalg.pinv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            tstats = params / se
            pvals = 2 * (1 - stats.t.cdf(np.abs(tstats), df=n - k))
            ss_res = np.dot(residuals, residuals)
            ss_tot = np.dot(y.values - y.values.mean(), y.values - y.values.mean())
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

        alpha_d = float(params[0])
        beta_mkt = float(params[1]) if len(params) > 1 else 0.0
        beta_smb = float(params[2]) if len(params) > 2 else 0.0
        beta_hml = float(params[3]) if len(params) > 3 else 0.0

        # Factor return contributions (annualised)
        factor_contribs = {}
        factor_means = self._factors.loc[common].mean() * self.ppy
        factor_contribs["MKT"] = beta_mkt * float(factor_means.get("MKT", 0))
        factor_contribs["SMB"] = beta_smb * float(factor_means.get("SMB", 0))
        factor_contribs["HML"] = beta_hml * float(factor_means.get("HML", 0))
        factor_contribs["Alpha"] = alpha_d * self.ppy

        return FamaFrenchResult(
            ticker=ticker,
            alpha=alpha_d * self.ppy,
            beta_mkt=beta_mkt,
            beta_smb=beta_smb,
            beta_hml=beta_hml,
            r_squared=r2,
            adj_r_squared=adj_r2,
            alpha_tstat=float(tstats[0]),
            alpha_pvalue=float(pvals[0]),
            alpha_significant=float(pvals[0]) < 0.05,
            mkt_tstat=float(tstats[1]) if len(tstats) > 1 else 0,
            smb_tstat=float(tstats[2]) if len(tstats) > 2 else 0,
            hml_tstat=float(tstats[3]) if len(tstats) > 3 else 0,
            factor_contributions=factor_contribs,
            n_observations=n,
        )

    # ──────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────

    def fit(self) -> dict[str, FamaFrenchResult]:
        """Fit FF3 for all assets. Returns {ticker: FamaFrenchResult}."""
        return {
            ticker: self._run_regression(ticker, self.assets[ticker])
            for ticker in self.assets.columns
        }

    def fit_to_dataframe(self) -> pd.DataFrame:
        """Fit FF3 for all assets and return a comparison DataFrame."""
        results = self.fit()
        rows = []
        for ticker, r in results.items():
            rows.append({
                "Ticker": ticker,
                "Alpha (ann.)": r.alpha,
                "β MKT": r.beta_mkt,
                "β SMB": r.beta_smb,
                "β HML": r.beta_hml,
                "R²": r.r_squared,
                "Adj R²": r.adj_r_squared,
                "Alpha t-stat": r.alpha_tstat,
                "Alpha Sig.": r.alpha_significant,
                "MKT Contrib": r.factor_contributions.get("MKT", 0),
                "SMB Contrib": r.factor_contributions.get("SMB", 0),
                "HML Contrib": r.factor_contributions.get("HML", 0),
                "Observations": r.n_observations,
            })
        return (
            pd.DataFrame(rows)
            .set_index("Ticker")
            .sort_values("Alpha (ann.)", ascending=False)
        )

    def get_factor_returns(self) -> pd.DataFrame:
        """Return the constructed factor return time series."""
        return self._factors.copy()

    def factor_correlation(self) -> pd.DataFrame:
        """Correlation matrix between factors."""
        return self._factors.corr()

    def factor_summary(self) -> pd.DataFrame:
        """Annualised statistics for each factor."""
        f = self._factors
        rows = []
        for col in f.columns:
            s = f[col].dropna()
            rows.append({
                "Factor": col,
                "Ann. Return": s.mean() * 252,
                "Ann. Volatility": s.std() * np.sqrt(252),
                "Sharpe": s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0,
                "Skewness": float(s.skew()),
                "Kurtosis": float(s.kurtosis()),
                "Observations": len(s),
            })
        return pd.DataFrame(rows).set_index("Factor")
