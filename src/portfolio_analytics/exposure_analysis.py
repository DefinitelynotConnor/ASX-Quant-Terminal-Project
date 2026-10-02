"""
exposure_analysis.py
====================
Portfolio exposure diagnostics: factor loadings, sector concentration,
market beta decomposition, and style analysis.

Exposures Computed
------------------
Factor     : Value, Momentum, Quality, Size, Low-Vol factor betas
Sector     : Financials, Materials, Healthcare, … weighted exposure
Beta       : Total market beta, systematic vs idiosyncratic risk split
Style      : Return-based style analysis (á la Sharpe)
Concentration : HHI, effective N, top-N weight, Gini coefficient
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


# ─────────────────────────────────────────────
#  ASX Sector Mapping (GICS)
# ─────────────────────────────────────────────

ASX_SECTOR_MAP = {
    # Financials
    "CBA.AX": "Financials", "WBC.AX": "Financials", "ANZ.AX": "Financials",
    "NAB.AX": "Financials", "MQG.AX": "Financials", "SUN.AX": "Financials",
    "IAG.AX": "Financials", "QBE.AX": "Financials", "AMP.AX": "Financials",
    # Materials
    "BHP.AX": "Materials", "RIO.AX": "Materials", "FMG.AX": "Materials",
    "S32.AX": "Materials", "NCM.AX": "Materials", "OZL.AX": "Materials",
    "NST.AX": "Materials", "EVN.AX": "Materials", "IGO.AX": "Materials",
    # Healthcare
    "CSL.AX": "Healthcare", "RMD.AX": "Healthcare", "COH.AX": "Healthcare",
    "RHC.AX": "Healthcare", "SHL.AX": "Healthcare", "PME.AX": "Healthcare",
    # Energy
    "WDS.AX": "Energy", "STO.AX": "Energy", "BPT.AX": "Energy",
    "WHC.AX": "Energy", "NHC.AX": "Energy",
    # Consumer Discretionary
    "WES.AX": "Consumer Discretionary", "JBH.AX": "Consumer Discretionary",
    "HVN.AX": "Consumer Discretionary", "SUL.AX": "Consumer Discretionary",
    "BWX.AX": "Consumer Discretionary",
    # Consumer Staples
    "WOW.AX": "Consumer Staples", "COL.AX": "Consumer Staples",
    "TWE.AX": "Consumer Staples", "A2M.AX": "Consumer Staples",
    # Communication Services
    "TLS.AX": "Communication Services", "REA.AX": "Communication Services",
    "CAR.AX": "Communication Services", "SEK.AX": "Communication Services",
    # Industrials
    "TCL.AX": "Industrials", "SYD.AX": "Industrials", "QAN.AX": "Industrials",
    "ALX.AX": "Industrials", "BXB.AX": "Industrials",
    # Real Estate
    "GMG.AX": "Real Estate", "SCG.AX": "Real Estate", "VCX.AX": "Real Estate",
    "MGR.AX": "Real Estate", "DXS.AX": "Real Estate",
    # Utilities
    "AGL.AX": "Utilities", "ORG.AX": "Utilities", "APA.AX": "Utilities",
    # Information Technology
    "XRO.AX": "Information Technology", "WTC.AX": "Information Technology",
    "ALU.AX": "Information Technology", "APX.AX": "Information Technology",
}


# ─────────────────────────────────────────────
#  ExposureAnalyser
# ─────────────────────────────────────────────

class ExposureAnalyser:
    """
    Compute factor, sector, and style exposures for a weighted portfolio.

    Parameters
    ----------
    price_data      : pd.DataFrame  Adjusted close prices; columns = tickers.
    weights         : dict          {ticker: weight}  Must sum to ≈ 1.
    benchmark       : pd.Series     Market benchmark returns.
    factor_returns  : pd.DataFrame  Optional pre-computed factor return series.
                                     Columns: ['value', 'momentum', 'quality',
                                               'size', 'low_vol']
    risk_free_rate  : float         Annualised RFR.
    """

    FACTORS = ["value", "momentum", "quality", "size", "low_vol"]

    def __init__(
        self,
        price_data: pd.DataFrame,
        weights: dict[str, float],
        benchmark: Optional[pd.Series] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0435,
        sector_map: Optional[dict[str, str]] = None,
    ) -> None:
        self.prices = price_data.copy()
        self.weights = self._normalise_weights(weights)
        self.benchmark = benchmark
        self.factor_returns = factor_returns
        self.rfr = risk_free_rate
        self.rfr_daily = (1 + risk_free_rate) ** (1 / 252) - 1
        self.sector_map = sector_map or ASX_SECTOR_MAP

        # Asset return series
        tickers = list(self.weights.keys())
        self.asset_returns = self.prices[tickers].pct_change().dropna()
        w = np.array([self.weights[t] for t in tickers])
        self.portfolio_returns = pd.Series(
            self.asset_returns.values @ w,
            index=self.asset_returns.index,
            name="Portfolio",
        )

    # ──────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────

    @staticmethod
    def _normalise_weights(weights: dict) -> dict:
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    # ──────────────────────────────────────
    #  Sector Exposure
    # ──────────────────────────────────────

    def sector_exposure(self) -> pd.DataFrame:
        """
        Aggregate portfolio weights by GICS sector.

        Returns
        -------
        pd.DataFrame  Columns: ['sector', 'weight', 'tickers']
        """
        sector_weights: dict[str, float] = {}
        sector_tickers: dict[str, list[str]] = {}

        for ticker, weight in self.weights.items():
            sector = self.sector_map.get(ticker, "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
            sector_tickers.setdefault(sector, []).append(ticker)

        rows = []
        for sector in sorted(sector_weights):
            rows.append({
                "Sector": sector,
                "Weight": sector_weights[sector],
                "Tickers": ", ".join(sector_tickers[sector]),
                "Num Holdings": len(sector_tickers[sector]),
            })

        df = pd.DataFrame(rows).sort_values("Weight", ascending=False).reset_index(drop=True)
        return df

    def sector_concentration_risk(self) -> dict[str, float]:
        """
        Sector-level HHI (Herfindahl-Hirschman Index).
        HHI = Σ w_i²  ∈ [0, 1].  HHI > 0.25 indicates high concentration.
        """
        se = self.sector_exposure()
        weights = se["Weight"].values
        hhi = float(np.sum(weights ** 2))
        top_sector = se.iloc[0]["Sector"] if len(se) > 0 else "N/A"
        top_weight = float(se.iloc[0]["Weight"]) if len(se) > 0 else 0.0

        return {
            "sector_hhi": hhi,
            "top_sector": top_sector,
            "top_sector_weight": top_weight,
            "num_sectors": len(se),
        }

    # ──────────────────────────────────────
    #  Market Beta Decomposition
    # ──────────────────────────────────────

    def asset_betas(self) -> pd.Series:
        """
        Compute OLS beta for each individual asset vs the benchmark.
        β_i = Cov(R_i, R_m) / Var(R_m)
        """
        if self.benchmark is None:
            return pd.Series(dtype=float)

        bm = self.benchmark.reindex(self.asset_returns.index).dropna()
        common = self.asset_returns.index.intersection(bm.index)
        ar = self.asset_returns.loc[common]
        bm = bm.loc[common]

        betas = {}
        for ticker in ar.columns:
            slope, _, _, _, _ = stats.linregress(bm.values, ar[ticker].values)
            betas[ticker] = slope

        return pd.Series(betas, name="Beta")

    def portfolio_beta(self) -> float:
        """
        Weighted sum of individual asset betas = portfolio beta.
        β_p = Σ w_i × β_i
        """
        b = self.asset_betas()
        if b.empty:
            return 1.0
        return float(sum(self.weights[t] * b[t] for t in b.index if t in self.weights))

    def systematic_vs_idiosyncratic_risk(self) -> dict[str, float]:
        """
        Decompose portfolio variance into systematic and idiosyncratic components.

        σ²_p = β²_p × σ²_m + σ²_ε
        """
        if self.benchmark is None:
            return {}

        bm = self.benchmark.reindex(self.portfolio_returns.index).dropna()
        common = self.portfolio_returns.index.intersection(bm.index)
        pr = self.portfolio_returns.loc[common]
        bm = bm.loc[common]

        beta = self.portfolio_beta()
        var_market = float(bm.var())
        var_portfolio = float(pr.var())
        var_systematic = beta ** 2 * var_market
        var_idiosyncratic = max(var_portfolio - var_systematic, 0)

        return {
            "total_variance": var_portfolio,
            "systematic_variance": var_systematic,
            "idiosyncratic_variance": var_idiosyncratic,
            "pct_systematic": var_systematic / var_portfolio if var_portfolio > 0 else 0,
            "pct_idiosyncratic": var_idiosyncratic / var_portfolio if var_portfolio > 0 else 0,
            "beta": beta,
        }

    # ──────────────────────────────────────
    #  Factor Exposure (OLS Regression)
    # ──────────────────────────────────────

    def factor_exposure(self) -> pd.DataFrame:
        """
        Regress portfolio returns on factor returns to derive factor loadings.

        If factor_returns are not supplied, synthetic proxies are computed from
        the available price data using cross-sectional sorts.

        Returns
        -------
        pd.DataFrame  factor exposures with t-stats, p-values, and R².
        """
        if self.factor_returns is not None:
            return self._ols_factor_regression(self.factor_returns)

        # Build synthetic factor proxies
        synthetic = self._build_synthetic_factors()
        if synthetic is None or synthetic.empty:
            return pd.DataFrame()

        return self._ols_factor_regression(synthetic)

    def _ols_factor_regression(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        Run OLS: R_p = α + Σ β_i × F_i + ε
        Returns a DataFrame with columns: [loading, t_stat, p_value, r_squared]
        """
        common = self.portfolio_returns.index.intersection(factors.index)
        if len(common) < 30:
            return pd.DataFrame()

        y = self.portfolio_returns.loc[common].values - self.rfr_daily
        X_raw = factors.loc[common].fillna(0).values

        # Add intercept
        X = np.column_stack([np.ones(len(y)), X_raw])
        n, k = X.shape

        # OLS solution
        try:
            beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return pd.DataFrame()

        y_hat = X @ beta_hat
        residuals = y - y_hat
        sse = np.dot(residuals, residuals)
        sst = np.dot(y - y.mean(), y - y.mean())
        r2 = 1 - sse / sst if sst > 0 else 0

        # Standard errors
        var_resid = sse / (n - k)
        cov_beta = var_resid * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov_beta))
        t_stats = beta_hat / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

        factor_names = ["Alpha"] + list(factors.columns)
        result = pd.DataFrame(
            {
                "Loading": beta_hat,
                "Std Error": se,
                "T-Stat": t_stats,
                "P-Value": p_values,
                "Significant": p_values < 0.05,
            },
            index=factor_names,
        )
        result["R-Squared"] = r2   # broadcast to all rows for display

        return result

    def _build_synthetic_factors(self) -> Optional[pd.DataFrame]:
        """
        Construct cross-sectional factor proxies from available price data.

        Factors
        -------
        momentum  : 12-1 month price momentum (skip 1 month)
        low_vol   : Negative of 21-day realised volatility (low vol = positive exposure)
        size      : Negative of log market cap proxy (small = positive)
        """
        ret = self.asset_returns.copy()
        if len(ret) < 63:
            return None

        factors = {}

        # Momentum: 12M-1M return
        mom_window = min(252, len(ret))
        skip = 21
        cum_ret = (1 + ret).cumprod()
        mom = (cum_ret.shift(skip) / cum_ret.shift(mom_window)).sub(1)
        mom_cs = mom.apply(lambda row: row - row.mean(), axis=1)
        factors["momentum"] = mom_cs.mean(axis=1)

        # Low-volatility: negative 21D std
        vol = ret.rolling(21).std()
        low_vol = -vol
        low_vol_cs = low_vol.apply(lambda row: row - row.mean(), axis=1)
        factors["low_vol"] = low_vol_cs.mean(axis=1)

        df = pd.DataFrame(factors).dropna()
        return df

    def asset_factor_exposures(self) -> pd.DataFrame:
        """
        Individual asset-level factor exposures via rolling OLS.

        Returns
        -------
        pd.DataFrame  index=ticker, columns=factors
        """
        if self.factor_returns is None:
            return pd.DataFrame()

        rows = {}
        factors = self.factor_returns
        common = self.asset_returns.index.intersection(factors.index)

        for ticker in self.asset_returns.columns:
            y = self.asset_returns.loc[common, ticker].values
            X = np.column_stack(
                [np.ones(len(y)), factors.loc[common].fillna(0).values]
            )
            try:
                beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
                rows[ticker] = dict(zip(["Alpha"] + list(factors.columns), beta_hat))
            except Exception:
                pass

        return pd.DataFrame(rows).T

    # ──────────────────────────────────────
    #  Concentration Metrics
    # ──────────────────────────────────────

    def concentration_metrics(self) -> dict[str, float]:
        """
        Portfolio concentration diagnostics.

        Returns
        -------
        dict containing:
            hhi            : Herfindahl-Hirschman Index (Σ w²)
            effective_n    : Effective number of holdings (1/HHI)
            top_5_weight   : Weight of 5 largest positions
            gini           : Gini coefficient of weights
            max_weight     : Largest single position weight
        """
        weights_sorted = np.sort(list(self.weights.values()))[::-1]
        hhi = float(np.sum(weights_sorted ** 2))
        effective_n = 1 / hhi if hhi > 0 else 0
        top_5 = float(np.sum(weights_sorted[:5]))

        # Gini coefficient
        n = len(weights_sorted)
        cumw = np.cumsum(weights_sorted)
        gini = float(1 - 2 * np.sum(cumw) / (n * cumw[-1]) + 1 / n) if n > 0 else 0

        return {
            "hhi": hhi,
            "effective_n": effective_n,
            "top_5_weight": top_5,
            "max_weight": float(max(weights_sorted)),
            "gini_coefficient": gini,
            "num_holdings": len(self.weights),
        }

    # ──────────────────────────────────────
    #  Correlation Structure
    # ──────────────────────────────────────

    def correlation_matrix(self) -> pd.DataFrame:
        """Return asset return correlation matrix."""
        return self.asset_returns.corr()

    def average_pairwise_correlation(self) -> float:
        """
        Average correlation across all asset pairs.
        Lower = more diversified portfolio.
        """
        corr = self.correlation_matrix()
        n = len(corr)
        if n < 2:
            return 1.0
        # Extract upper triangle (exclude diagonal)
        upper = corr.values[np.triu_indices(n, k=1)]
        return float(np.mean(upper))

    def diversification_ratio(self) -> float:
        """
        Diversification Ratio = weighted avg volatility / portfolio volatility.
        DR > 1 indicates diversification benefit.
        """
        asset_vols = self.asset_returns.std() * np.sqrt(252)
        w = np.array([self.weights.get(t, 0) for t in asset_vols.index])
        weighted_avg_vol = float(np.dot(w, asset_vols.values))
        port_vol = float(self.portfolio_returns.std() * np.sqrt(252))
        return weighted_avg_vol / port_vol if port_vol > 0 else 1.0

    # ──────────────────────────────────────
    #  Style Analysis (Returns-Based)
    # ──────────────────────────────────────

    def style_analysis(
        self, style_benchmarks: Optional[dict[str, pd.Series]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Sharpe-style returns-based style analysis.
        Finds non-negative weights that sum to 1 to replicate portfolio returns
        using a set of style benchmarks.

        Parameters
        ----------
        style_benchmarks : dict   {style_name: return_series}
            e.g. {'Value': ..., 'Growth': ..., 'Small Cap': ...}

        Returns
        -------
        pd.DataFrame with style weights and tracking error.
        """
        if style_benchmarks is None:
            return None

        styles = list(style_benchmarks.keys())
        bench_df = pd.DataFrame(style_benchmarks)
        common = self.portfolio_returns.index.intersection(bench_df.index)

        if len(common) < 30:
            return None

        y = self.portfolio_returns.loc[common].values
        X = bench_df.loc[common].values

        # Optimise: min tracking_error subject to: weights >= 0, sum = 1
        def objective(w):
            residuals = y - X @ w
            return float(np.var(residuals))

        n_styles = len(styles)
        result = minimize(
            objective,
            x0=np.ones(n_styles) / n_styles,
            method="SLSQP",
            bounds=[(0, 1)] * n_styles,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        )

        if not result.success:
            return None

        style_weights = result.x
        y_hat = X @ style_weights
        te = float(np.std(y - y_hat) * np.sqrt(252))
        r2 = float(1 - np.var(y - y_hat) / np.var(y))

        return pd.DataFrame(
            {
                "Style": styles,
                "Weight": style_weights,
                "Tracking Error": te,
                "R-Squared": r2,
            }
        ).set_index("Style")

    # ──────────────────────────────────────
    #  Full Exposure Report
    # ──────────────────────────────────────

    def full_exposure_report(self) -> dict:
        """
        Aggregate all exposure metrics into a single dictionary.
        """
        return {
            "sector_exposure": self.sector_exposure(),
            "sector_concentration": self.sector_concentration_risk(),
            "beta_decomposition": self.systematic_vs_idiosyncratic_risk(),
            "asset_betas": self.asset_betas().to_dict(),
            "portfolio_beta": self.portfolio_beta(),
            "concentration": self.concentration_metrics(),
            "avg_pairwise_correlation": self.average_pairwise_correlation(),
            "diversification_ratio": self.diversification_ratio(),
            "factor_exposures": self.factor_exposure(),
        }
