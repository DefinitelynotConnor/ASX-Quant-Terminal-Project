"""
capm.py
=======
Capital Asset Pricing Model (CAPM) implementation.

The CAPM is the foundational single-factor model in finance.
It describes the relationship between systematic risk (beta) and expected return.

Model
-----
E[R_i] = R_f + β_i × (E[R_m] - R_f)

Where:
    R_i  = Asset return
    R_f  = Risk-free rate
    β_i  = Systematic risk (sensitivity to market)
    R_m  = Market return
    α_i  = Jensen's Alpha (excess return above CAPM prediction)

Usage
-----
    from src.factor_models.capm import CAPM

    capm = CAPM(asset_returns, market_returns, risk_free_rate=0.0435)
    results = capm.fit()
    print(results.summary())
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
class CAPMResult:
    """Container for CAPM regression results."""
    ticker: str = ""
    alpha: float = 0.0              # Jensen's Alpha (annualised)
    alpha_daily: float = 0.0        # Daily alpha
    beta: float = 0.0               # Market beta
    r_squared: float = 0.0          # Explanatory power
    alpha_tstat: float = 0.0
    beta_tstat: float = 0.0
    alpha_pvalue: float = 1.0
    beta_pvalue: float = 1.0
    alpha_significant: bool = False
    beta_significant: bool = False
    expected_return: float = 0.0    # CAPM-implied expected return
    actual_return: float = 0.0      # Realised annualised return
    abnormal_return: float = 0.0    # Actual - Expected
    treynor_ratio: float = 0.0
    information_ratio: float = 0.0
    residual_std: float = 0.0       # Idiosyncratic risk
    systematic_risk_pct: float = 0.0
    idiosyncratic_risk_pct: float = 0.0
    n_observations: int = 0
    start_date: str = ""
    end_date: str = ""

    def summary(self) -> str:
        return (
            f"\nCAPM Results: {self.ticker}\n"
            f"{'─' * 40}\n"
            f"  Alpha (ann.)      : {self.alpha:.2%}  "
            f"(t={self.alpha_tstat:.2f}, p={self.alpha_pvalue:.3f})"
            f"  {'✅ Significant' if self.alpha_significant else '❌ Not significant'}\n"
            f"  Beta              : {self.beta:.4f}  "
            f"(t={self.beta_tstat:.2f}, p={self.beta_pvalue:.3f})\n"
            f"  R-Squared         : {self.r_squared:.4f}\n"
            f"  Expected Return   : {self.expected_return:.2%}\n"
            f"  Actual Return     : {self.actual_return:.2%}\n"
            f"  Abnormal Return   : {self.abnormal_return:.2%}\n"
            f"  Systematic Risk   : {self.systematic_risk_pct:.1%}\n"
            f"  Idiosyncratic Risk: {self.idiosyncratic_risk_pct:.1%}\n"
            f"  Treynor Ratio     : {self.treynor_ratio:.4f}\n"
            f"  Observations      : {self.n_observations}\n"
        )


# ─────────────────────────────────────────────
#  CAPM
# ─────────────────────────────────────────────

class CAPM:
    """
    Capital Asset Pricing Model.

    Parameters
    ----------
    asset_returns  : pd.Series or pd.DataFrame
        Daily asset returns. If DataFrame, fits CAPM for each column.
    market_returns : pd.Series
        Daily market (benchmark) returns. e.g. ASX 200.
    risk_free_rate : float
        Annualised risk-free rate (default: RBA cash rate 4.35%).
    periods_per_year: int
        Trading days per year (default: 252).
    """

    def __init__(
        self,
        asset_returns: pd.Series | pd.DataFrame,
        market_returns: pd.Series,
        risk_free_rate: float = 0.0435,
        periods_per_year: int = 252,
    ) -> None:
        self.market = market_returns.dropna()
        self.rfr = risk_free_rate
        self.rfr_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        self.ppy = periods_per_year

        # Handle single Series or DataFrame
        if isinstance(asset_returns, pd.Series):
            self.assets = asset_returns.dropna().to_frame()
        else:
            self.assets = asset_returns.dropna()

    # ──────────────────────────────────────
    #  Single asset fit
    # ──────────────────────────────────────

    def fit_single(self, ticker: str, returns: pd.Series) -> CAPMResult:
        """
        Fit CAPM for a single asset return series.

        Uses OLS regression of excess asset returns on excess market returns:
        (R_i - R_f) = α + β × (R_m - R_f) + ε
        """
        # Align
        common = returns.index.intersection(self.market.index)
        if len(common) < 30:
            return CAPMResult(ticker=ticker)

        r_i = returns.loc[common]
        r_m = self.market.loc[common]

        # Excess returns
        excess_asset = r_i - self.rfr_daily
        excess_market = r_m - self.rfr_daily

        # OLS regression
        if HAS_STATSMODELS:
            X = sm.add_constant(excess_market.values)
            model = sm.OLS(excess_asset.values, X)
            fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            alpha_d = float(fit.params[0])
            beta = float(fit.params[1])
            alpha_t = float(fit.tvalues[0])
            beta_t = float(fit.tvalues[1])
            alpha_p = float(fit.pvalues[0])
            beta_p = float(fit.pvalues[1])
            r2 = float(fit.rsquared)
            resid_std = float(fit.resid.std())
        else:
            slope, intercept, r, pval, se = stats.linregress(
                excess_market.values, excess_asset.values
            )
            alpha_d = float(intercept)
            beta = float(slope)
            r2 = float(r ** 2)
            n = len(excess_asset)
            resid_std = float(
                np.std(excess_asset.values - (alpha_d + beta * excess_market.values))
            )
            se_alpha = resid_std * np.sqrt(
                np.sum(excess_market.values ** 2) /
                (n * np.sum((excess_market.values - excess_market.mean()) ** 2))
            )
            se_beta = resid_std / np.sqrt(
                np.sum((excess_market.values - excess_market.mean()) ** 2)
            )
            alpha_t = alpha_d / se_alpha if se_alpha > 0 else 0
            beta_t = beta / se_beta if se_beta > 0 else 0
            alpha_p = float(2 * (1 - stats.t.cdf(abs(alpha_t), df=n - 2)))
            beta_p = float(2 * (1 - stats.t.cdf(abs(beta_t), df=n - 2)))

        # Annualise alpha
        alpha_ann = alpha_d * self.ppy

        # CAPM-implied expected return
        market_premium = float(excess_market.mean() * self.ppy)
        expected_ret = self.rfr + beta * market_premium
        actual_ret = float((1 + r_i).prod() ** (self.ppy / len(r_i)) - 1)
        abnormal_ret = actual_ret - expected_ret

        # Risk decomposition
        var_total = float(r_i.var())
        var_systematic = beta ** 2 * float(r_m.var())
        var_idio = max(var_total - var_systematic, 0)
        pct_sys = var_systematic / var_total if var_total > 0 else 0
        pct_idio = var_idio / var_total if var_total > 0 else 0

        # Treynor ratio
        treynor = (actual_ret - self.rfr) / beta if beta != 0 else 0

        # Information ratio (alpha / tracking error)
        residuals = excess_asset - (alpha_d + beta * excess_market)
        ir = float(
            (alpha_d * self.ppy) / (residuals.std() * np.sqrt(self.ppy))
        ) if residuals.std() > 0 else 0

        return CAPMResult(
            ticker=ticker,
            alpha=alpha_ann,
            alpha_daily=alpha_d,
            beta=beta,
            r_squared=r2,
            alpha_tstat=alpha_t,
            beta_tstat=beta_t,
            alpha_pvalue=alpha_p,
            beta_pvalue=beta_p,
            alpha_significant=alpha_p < 0.05,
            beta_significant=beta_p < 0.05,
            expected_return=expected_ret,
            actual_return=actual_ret,
            abnormal_return=abnormal_ret,
            treynor_ratio=treynor,
            information_ratio=ir,
            residual_std=resid_std * np.sqrt(self.ppy),
            systematic_risk_pct=pct_sys,
            idiosyncratic_risk_pct=pct_idio,
            n_observations=len(common),
            start_date=str(common[0].date()),
            end_date=str(common[-1].date()),
        )

    # ──────────────────────────────────────
    #  Universe fit
    # ──────────────────────────────────────

    def fit(self) -> dict[str, CAPMResult]:
        """
        Fit CAPM for all assets and return a dict of results.

        Returns
        -------
        dict  {ticker: CAPMResult}
        """
        results = {}
        for ticker in self.assets.columns:
            results[ticker] = self.fit_single(ticker, self.assets[ticker])
        return results

    def fit_to_dataframe(self) -> pd.DataFrame:
        """
        Fit CAPM for all assets and return results as a comparison DataFrame.

        Returns
        -------
        pd.DataFrame  One row per ticker.
        """
        results = self.fit()
        rows = []
        for ticker, r in results.items():
            rows.append({
                "Ticker": ticker,
                "Alpha (ann.)": r.alpha,
                "Beta": r.beta,
                "R-Squared": r.r_squared,
                "Alpha t-stat": r.alpha_tstat,
                "Alpha Significant": r.alpha_significant,
                "Expected Return": r.expected_return,
                "Actual Return": r.actual_return,
                "Abnormal Return": r.abnormal_return,
                "Treynor Ratio": r.treynor_ratio,
                "Info Ratio": r.information_ratio,
                "Systematic Risk %": r.systematic_risk_pct,
                "Idiosyncratic Risk %": r.idiosyncratic_risk_pct,
                "Observations": r.n_observations,
            })
        return (
            pd.DataFrame(rows)
            .set_index("Ticker")
            .sort_values("Alpha (ann.)", ascending=False)
        )

    # ──────────────────────────────────────
    #  Rolling CAPM
    # ──────────────────────────────────────

    def rolling_capm(
        self,
        ticker: str,
        window: int = 63,
    ) -> pd.DataFrame:
        """
        Rolling CAPM estimation over a trailing window.
        Captures time-varying beta and alpha.

        Parameters
        ----------
        ticker : str   Asset to analyse.
        window : int   Rolling window in trading days.

        Returns
        -------
        pd.DataFrame  columns: alpha, beta, r_squared (at each date).
        """
        if ticker not in self.assets.columns:
            return pd.DataFrame()

        r_i = self.assets[ticker]
        common = r_i.index.intersection(self.market.index)
        r_i = r_i.loc[common]
        r_m = self.market.loc[common]

        alphas, betas, r2s = [], [], []
        dates = []

        for end in range(window, len(common)):
            ri_w = r_i.iloc[end - window: end].values
            rm_w = r_m.iloc[end - window: end].values

            slope, intercept, r, _, _ = stats.linregress(rm_w, ri_w)
            alphas.append(intercept * self.ppy)
            betas.append(slope)
            r2s.append(r ** 2)
            dates.append(common[end])

        return pd.DataFrame(
            {"Alpha": alphas, "Beta": betas, "R_Squared": r2s},
            index=dates,
        )

    # ──────────────────────────────────────
    #  Security Market Line
    # ──────────────────────────────────────

    def security_market_line(self) -> pd.DataFrame:
        """
        Generate Security Market Line data for plotting.
        Shows each asset's beta vs actual return vs CAPM-predicted return.

        Returns
        -------
        pd.DataFrame  columns: beta, actual_return, expected_return, alpha, ticker
        """
        results = self.fit()
        rows = []
        for ticker, r in results.items():
            rows.append({
                "Ticker": ticker,
                "Beta": r.beta,
                "Actual Return": r.actual_return,
                "Expected Return (CAPM)": r.expected_return,
                "Alpha": r.alpha,
                "Above SML": r.actual_return > r.expected_return,
            })
        return pd.DataFrame(rows)

    # ──────────────────────────────────────
    #  Portfolio CAPM
    # ──────────────────────────────────────

    def portfolio_capm(
        self,
        weights: dict[str, float],
    ) -> CAPMResult:
        """
        Fit CAPM for a weighted portfolio of assets.

        Parameters
        ----------
        weights : dict  {ticker: weight}  Must sum to ≈ 1.

        Returns
        -------
        CAPMResult for the combined portfolio.
        """
        tickers = [t for t in weights if t in self.assets.columns]
        w = np.array([weights[t] for t in tickers])
        w = w / w.sum()

        common = self.assets[tickers].dropna().index.intersection(
            self.market.index
        )
        port_ret = pd.Series(
            self.assets[tickers].loc[common].values @ w,
            index=common,
            name="Portfolio",
        )
        return self.fit_single("Portfolio", port_ret)
