"""
risk_decomposition.py
=====================
Institutional risk attribution and decomposition.

Identifies WHICH assets drive portfolio risk, enabling the portfolio manager
to reduce unwanted concentrations and build more efficient portfolios.

Risk Metrics Implemented
------------------------
MCTR    : Marginal Contribution to Risk — sensitivity of portfolio vol
          to a small increase in each asset's weight.
PCTR    : Percentage Contribution to Total Risk — share of total risk.
CTR     : Component (absolute) Contribution to Risk.
CVaR Decomposition : Contribution of each asset to Expected Shortfall.
PCA     : Principal Component decomposition of the covariance matrix.
Stress  : Stress test impact on portfolio from historical scenarios.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import eigh


# ─────────────────────────────────────────────
#  RiskDecomposer
# ─────────────────────────────────────────────

class RiskDecomposer:
    """
    Decompose portfolio risk into per-asset contributions.

    Parameters
    ----------
    asset_returns  : pd.DataFrame  Daily return series; columns = tickers.
    weights        : dict          {ticker: weight}  ∑w = 1 (normalised internally).
    cov_window     : int           Rolling window for covariance estimation (days).
    cov_method     : str           'sample' | 'ledoit_wolf' | 'ewma'
    ewma_lambda    : float         Decay factor for EWMA covariance.
    """

    def __init__(
        self,
        asset_returns: pd.DataFrame,
        weights: dict[str, float],
        cov_window: int = 252,
        cov_method: str = "ledoit_wolf",
        ewma_lambda: float = 0.94,
    ) -> None:
        self.tickers = [t for t in weights if t in asset_returns.columns]
        self.weights = self._normalise(weights)
        self.returns = asset_returns[self.tickers].dropna()
        self.cov_window = cov_window
        self.cov_method = cov_method
        self.ewma_lambda = ewma_lambda

        self.w = np.array([self.weights[t] for t in self.tickers])
        self.Sigma = self._estimate_covariance()

    @staticmethod
    def _normalise(w: dict) -> dict:
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    # ──────────────────────────────────────
    #  Covariance Estimation
    # ──────────────────────────────────────

    def _estimate_covariance(self) -> np.ndarray:
        """
        Estimate the annualised covariance matrix.

        Methods
        -------
        sample       : Standard sample covariance.
        ledoit_wolf  : Shrinkage estimator — reduces estimation error.
        ewma         : Exponentially weighted — more weight on recent data.
        """
        recent = self.returns.tail(self.cov_window)

        if self.cov_method == "sample":
            return recent.cov().values * 252

        elif self.cov_method == "ledoit_wolf":
            return self._ledoit_wolf_shrinkage(recent) * 252

        elif self.cov_method == "ewma":
            return self._ewma_covariance(recent) * 252

        return recent.cov().values * 252

    def _ledoit_wolf_shrinkage(self, returns: pd.DataFrame) -> np.ndarray:
        n, p = returns.shape
        if n < 2 or p < 1:
            return np.eye(p) * returns.var().mean() if p > 0 else np.eye(1)
        S = returns.cov().values
        mu_lw = np.trace(S) / p if p > 0 else 1.0
        delta = np.linalg.norm(S - mu_lw * np.eye(p), "fro") ** 2
        if delta == 0:
            return S
        beta2 = 0.0
        for i in range(n):
            x = returns.iloc[i].values
            outer = np.outer(x, x)
            beta2 += np.linalg.norm(outer - S, "fro") ** 2
        beta2 /= max(n ** 2, 1)
        alpha = min(beta2 / delta, 1.0)
        shrunk = (1 - alpha) * S + alpha * mu_lw * np.eye(p)
        return shrunk

    def _ewma_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Exponentially Weighted Moving Average covariance (RiskMetrics methodology).
        """
        lam = self.ewma_lambda
        n, p = returns.shape
        cov = returns.iloc[0:1].T @ returns.iloc[0:1]   # seed
        for i in range(1, n):
            r = returns.iloc[i].values.reshape(-1, 1)
            cov = lam * cov + (1 - lam) * (r @ r.T)
        return cov

    def portfolio_volatility(self) -> float:
        """σ_p = √(w' Σ w) — annualised portfolio volatility."""
        var_p = float(self.w @ self.Sigma @ self.w)
        return float(np.sqrt(max(var_p, 0)))

    # ──────────────────────────────────────
    #  MCTR (Marginal Contribution to Risk)
    # ──────────────────────────────────────

    def marginal_contribution_to_risk(self) -> pd.Series:
        """
        MCTR_i = ∂σ_p / ∂w_i = (Σw)_i / σ_p

        The MCTR tells the portfolio manager by how much portfolio volatility
        increases (in % terms) for a 1-unit increase in the weight of asset i.
        """
        sigma_p = self.portfolio_volatility()
        if sigma_p == 0:
            return pd.Series(0.0, index=self.tickers, name="MCTR")

        mctr = (self.Sigma @ self.w) / sigma_p
        return pd.Series(mctr, index=self.tickers, name="MCTR")

    # ──────────────────────────────────────
    #  CTR (Component / Absolute Risk Contribution)
    # ──────────────────────────────────────

    def component_risk_contribution(self) -> pd.Series:
        """
        CTR_i = w_i × MCTR_i
        The risk contribution of each asset in the same units as σ_p.
        ∑ CTR_i = σ_p  (Euler decomposition property).
        """
        mctr = self.marginal_contribution_to_risk()
        ctr = pd.Series(self.w, index=self.tickers) * mctr
        ctr.name = "CTR"
        return ctr

    # ──────────────────────────────────────
    #  PCTR (Percentage Contribution to Risk)
    # ──────────────────────────────────────

    def pct_risk_contribution(self) -> pd.Series:
        """
        PCTR_i = CTR_i / σ_p  =  w_i × MCTR_i / σ_p
        Sums to 100%. Identifies the dominant risk drivers.
        """
        ctr = self.component_risk_contribution()
        sigma_p = self.portfolio_volatility()
        if sigma_p == 0:
            return pd.Series(0.0, index=self.tickers, name="PCTR")
        return (ctr / sigma_p).rename("PCTR")

    # ──────────────────────────────────────
    #  Risk Attribution Table
    # ──────────────────────────────────────

    def risk_attribution_table(self) -> pd.DataFrame:
        """
        Full risk decomposition table per asset.

        Columns
        -------
        Weight          : Portfolio weight
        Asset Vol       : Individual asset annualised volatility
        MCTR            : Marginal contribution to risk
        CTR             : Component (absolute) risk contribution
        PCTR            : % of total portfolio risk
        Risk Budget Used: CTR / weight — risk efficiency ratio
        """
        mctr = self.marginal_contribution_to_risk()
        ctr = self.component_risk_contribution()
        pctr = self.pct_risk_contribution()

        asset_vols = self.returns.std() * np.sqrt(252)

        rows = []
        for ticker in self.tickers:
            w_i = self.weights[ticker]
            rows.append({
                "Ticker": ticker,
                "Weight": w_i,
                "Asset Volatility": float(asset_vols.get(ticker, np.nan)),
                "MCTR": float(mctr.get(ticker, 0.0)),
                "CTR": float(ctr.get(ticker, 0.0)),
                "PCTR": float(pctr.get(ticker, 0.0)),
                "Risk/Weight Ratio": float(pctr.get(ticker, 0.0) / w_i) if w_i > 0 else 0,
            })

        df = pd.DataFrame(rows).sort_values("PCTR", ascending=False).reset_index(drop=True)

        # Total row
        total = {
            "Ticker": "TOTAL",
            "Weight": df["Weight"].sum(),
            "Asset Volatility": np.nan,
            "MCTR": np.nan,
            "CTR": df["CTR"].sum(),
            "PCTR": df["PCTR"].sum(),
            "Risk/Weight Ratio": np.nan,
        }
        df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
        return df

    # ──────────────────────────────────────
    #  Variance Decomposition
    # ──────────────────────────────────────

    def variance_decomposition(self) -> dict[str, float]:
        """
        Decompose portfolio variance into pairwise asset contributions.

        σ²_p = ∑_i ∑_j w_i × w_j × σ_ij

        Returns both individual variance and cross-covariance terms.
        """
        var_matrix = np.outer(self.w, self.w) * self.Sigma
        n = len(self.tickers)

        individual_var = {}
        cross_covar = {}

        for i, ti in enumerate(self.tickers):
            individual_var[ti] = float(var_matrix[i, i])
            for j, tj in enumerate(self.tickers):
                if i != j:
                    key = f"{ti}–{tj}"
                    cross_covar[key] = float(var_matrix[i, j] + var_matrix[j, i])

        total_var = float(np.sum(var_matrix))
        individual_total = sum(individual_var.values())
        cross_total = total_var - individual_total

        return {
            "total_variance": total_var,
            "individual_variance": individual_var,
            "cross_covariance_total": cross_total,
            "pct_from_individual": individual_total / total_var if total_var > 0 else 0,
            "pct_from_cross_covariance": cross_total / total_var if total_var > 0 else 0,
        }

    # ──────────────────────────────────────
    #  CVaR Decomposition
    # ──────────────────────────────────────

    def cvar_decomposition(
        self,
        confidence: float = 0.95,
    ) -> pd.DataFrame:
        """
        Decompose portfolio CVaR into per-asset contributions.

        Component CVaR_i = E[-R_i | R_p < VaR_p]
        where R_p = portfolio return.

        Method: simulate 'historical' portfolio and asset returns,
                condition on portfolio being in the tail.
        """
        port_ret = pd.Series(
            self.returns.values @ self.w,
            index=self.returns.index,
        )

        var_threshold = np.percentile(port_ret, (1 - confidence) * 100)
        tail_mask = port_ret <= var_threshold
        tail_asset_returns = self.returns[tail_mask]
        tail_port_returns = port_ret[tail_mask]

        if len(tail_asset_returns) == 0:
            return pd.DataFrame()

        rows = []
        portfolio_cvar = float(-tail_port_returns.mean())

        for i, ticker in enumerate(self.tickers):
            asset_tail = tail_asset_returns[ticker]
            component = float(-asset_tail.mean() * self.w[i])
            rows.append({
                "Ticker": ticker,
                "Weight": self.w[i],
                "Component CVaR": component,
                "% of Portfolio CVaR": component / portfolio_cvar if portfolio_cvar != 0 else 0,
                "CVaR Contribution / Weight": (component / self.w[i]) if self.w[i] > 0 else 0,
            })

        df = pd.DataFrame(rows).sort_values("Component CVaR", ascending=False)
        return df

    # ──────────────────────────────────────
    #  Principal Component Analysis
    # ──────────────────────────────────────

    def pca_risk_decomposition(
        self, n_components: int = 5
    ) -> dict:
        """
        Decompose covariance matrix via PCA.

        Returns the fraction of variance explained by each principal component
        and the asset factor loadings.  The first PC typically represents
        'market risk'; subsequent PCs represent sector or style factors.

        Parameters
        ----------
        n_components : int   Number of principal components to retain.

        Returns
        -------
        dict containing:
            explained_variance_ratio   : ndarray of explained variance fractions
            loadings                   : DataFrame (n_assets × n_components)
            component_risk_pct         : % portfolio risk from each component
        """
        n = len(self.tickers)
        n_comp = min(n_components, n)

        # Eigendecomposition (ascending order → reverse for descending)
        eigenvalues, eigenvectors = eigh(self.Sigma)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_var = eigenvalues.sum()
        explained_ratio = eigenvalues[:n_comp] / total_var

        # Loadings DataFrame
        loadings = pd.DataFrame(
            eigenvectors[:, :n_comp],
            index=self.tickers,
            columns=[f"PC{i+1}" for i in range(n_comp)],
        )

        # Risk contribution from each PC
        pc_weights = eigenvectors[:, :n_comp].T @ self.w   # projection of port weights
        pc_risk = pc_weights ** 2 * eigenvalues[:n_comp]
        port_var = self.w @ self.Sigma @ self.w
        pc_risk_pct = pc_risk / port_var if port_var > 0 else pc_risk * 0

        return {
            "eigenvalues": eigenvalues[:n_comp],
            "explained_variance_ratio": explained_ratio,
            "cumulative_explained_variance": np.cumsum(explained_ratio),
            "loadings": loadings,
            "component_risk_pct": pd.Series(
                pc_risk_pct,
                index=[f"PC{i+1}" for i in range(n_comp)],
                name="% Portfolio Risk",
            ),
        }

    # ──────────────────────────────────────
    #  Stress Testing
    # ──────────────────────────────────────

    def stress_test(
        self,
        scenarios: Optional[dict[str, dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """
        Apply factor stress shocks to the portfolio and compute the P&L impact.

        Parameters
        ----------
        scenarios : dict
            {scenario_name: {ticker_or_factor: shock_return}}
            e.g. {'GFC-style crash': {'BHP.AX': -0.40, 'CBA.AX': -0.45}}

        Returns
        -------
        pd.DataFrame  scenario × {portfolio_return, worst_asset, best_asset}
        """
        if scenarios is None:
            scenarios = self._default_asx_scenarios()

        records = []
        for name, shocks in scenarios.items():
            port_impact = 0.0
            asset_impacts = {}
            for ticker in self.tickers:
                shock = shocks.get(ticker, shocks.get("market", 0.0))
                impact = self.weights[ticker] * shock
                port_impact += impact
                asset_impacts[ticker] = impact

            records.append({
                "Scenario": name,
                "Portfolio Impact": port_impact,
                "Worst Asset": min(asset_impacts, key=asset_impacts.get),
                "Best Asset": max(asset_impacts, key=asset_impacts.get),
                "Largest Loss": min(asset_impacts.values()),
            })

        return pd.DataFrame(records).set_index("Scenario")

    def _default_asx_scenarios(self) -> dict[str, dict[str, float]]:
        """
        Predefined historical and hypothetical stress scenarios for ASX portfolios.
        """
        return {
            "GFC 2008 (Global Financial Crisis)": {"market": -0.45},
            "COVID Crash Mar 2020": {"market": -0.35},
            "China Hard Landing": {
                "BHP.AX": -0.40, "RIO.AX": -0.38, "FMG.AX": -0.50,
                "market": -0.20,
            },
            "RBA Rate Shock +200bps": {
                "CBA.AX": -0.20, "WBC.AX": -0.22, "ANZ.AX": -0.21,
                "NAB.AX": -0.19, "GMG.AX": -0.25, "market": -0.15,
            },
            "AUD/USD -20%": {"market": -0.10},
            "Iron Ore Price -40%": {
                "BHP.AX": -0.25, "RIO.AX": -0.22, "FMG.AX": -0.40,
            },
            "Tech Sector Selloff -30%": {
                "XRO.AX": -0.30, "WTC.AX": -0.30, "ALU.AX": -0.30,
            },
            "Mild Recession": {"market": -0.20},
            "Severe Recession": {"market": -0.40},
        }

    # ──────────────────────────────────────
    #  Tracking Error & Active Risk
    # ──────────────────────────────────────

    def tracking_error(
        self,
        benchmark_returns: pd.Series,
    ) -> float:
        """
        Annualised tracking error (active risk).
        TE = σ(R_p - R_b) × √252
        """
        port_ret = pd.Series(
            self.returns.values @ self.w,
            index=self.returns.index,
        )
        common = port_ret.index.intersection(benchmark_returns.index)
        active = port_ret.loc[common] - benchmark_returns.loc[common]
        return float(active.std() * np.sqrt(252))

    def active_share(
        self,
        benchmark_weights: dict[str, float],
    ) -> float:
        """
        Active Share = 0.5 × ∑_i |w_p_i - w_b_i|
        Ranges from 0 (index) to 1 (fully active).
        Active Share > 0.6 is generally considered 'truly active'.
        """
        all_tickers = set(self.weights) | set(benchmark_weights)
        total_bw = sum(benchmark_weights.values())
        norm_bw = {t: benchmark_weights.get(t, 0.0) / total_bw for t in all_tickers}
        norm_pw = self.weights

        active_share = 0.5 * sum(
            abs(norm_pw.get(t, 0.0) - norm_bw.get(t, 0.0))
            for t in all_tickers
        )
        return float(active_share)

    # ──────────────────────────────────────
    #  Risk Budget Analysis
    # ──────────────────────────────────────

    def risk_budget_analysis(
        self,
        target_risk_budget: Optional[dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Compare actual risk contribution vs target risk budget.

        Parameters
        ----------
        target_risk_budget : dict  {ticker: target_pct_risk}
                                    e.g. equal risk: {t: 1/n for all t}

        Returns
        -------
        pd.DataFrame  with actual vs target risk contributions and deviation.
        """
        pctr = self.pct_risk_contribution()
        n = len(self.tickers)

        if target_risk_budget is None:
            target_risk_budget = {t: 1 / n for t in self.tickers}

        rows = []
        for ticker in self.tickers:
            actual = float(pctr.get(ticker, 0.0))
            target = target_risk_budget.get(ticker, 1 / n)
            rows.append({
                "Ticker": ticker,
                "Actual Risk %": actual,
                "Target Risk %": target,
                "Deviation": actual - target,
                "Over-risked": actual > target,
            })

        return (
            pd.DataFrame(rows)
            .sort_values("Deviation", ascending=False)
            .reset_index(drop=True)
        )

    # ──────────────────────────────────────
    #  Full Risk Report
    # ──────────────────────────────────────

    def full_risk_report(self) -> dict:
        """
        Aggregate all risk decomposition outputs into a single dict.
        """
        return {
            "portfolio_volatility": self.portfolio_volatility(),
            "risk_attribution": self.risk_attribution_table(),
            "variance_decomposition": self.variance_decomposition(),
            "cvar_decomposition_95": self.cvar_decomposition(0.95),
            "pca_decomposition": self.pca_risk_decomposition(),
            "stress_tests": self.stress_test(),
            "risk_budget": self.risk_budget_analysis(),
        }
