"""
black_litterman.py
==================
Black-Litterman (1990) portfolio optimisation model.

The Black-Litterman model solves the core problem with Markowitz:
    - Markowitz is extremely sensitive to expected return inputs
    - Small changes in expected returns → wildly different portfolios
    - In practice, expected returns are the hardest thing to estimate

The BL model:
    1. Starts from equilibrium (market-implied) expected returns
    2. Allows the investor to express views with confidence levels
    3. Blends the two to produce posterior expected returns
    4. Uses those posteriors as inputs to Markowitz optimisation

This produces more stable, intuitive, diversified portfolios.

Model
-----
    π = δ × Σ × w_mkt          (equilibrium returns)
    E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 × [(τΣ)^-1π + P'Ω^-1Q]

Where:
    π    = Implied equilibrium excess returns
    δ    = Risk aversion coefficient
    Σ    = Covariance matrix
    w_mkt= Market capitalisation weights
    τ    = Scaling factor (uncertainty in prior)
    P    = View matrix (which assets each view covers)
    Q    = View return vector
    Ω    = View uncertainty matrix (diagonal)

Usage
-----
    from src.optimisation.black_litterman import BlackLitterman

    bl = BlackLitterman(returns, market_weights)
    bl.add_absolute_view("BHP.AX", 0.15)   # BHP will return 15%
    bl.add_relative_view("CBA.AX", "WBC.AX", 0.03)  # CBA outperforms WBC by 3%
    result = bl.optimise()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class BLView:
    """A single investor view in the Black-Litterman model."""
    description: str
    p_vector: np.ndarray      # Pick matrix row
    q_value: float            # Expected return for this view
    confidence: float         # Confidence level [0, 1]
    view_type: str            # 'absolute' or 'relative'


@dataclass
class BLResult:
    """Container for Black-Litterman results."""
    posterior_returns: pd.Series = field(default_factory=pd.Series)
    posterior_covariance: np.ndarray = field(default_factory=lambda: np.array([]))
    equilibrium_returns: pd.Series = field(default_factory=pd.Series)
    optimal_weights: pd.Series = field(default_factory=pd.Series)
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    views_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


class BlackLitterman:
    """
    Black-Litterman portfolio optimisation.

    Parameters
    ----------
    returns         : pd.DataFrame   Daily asset returns.
    market_weights  : pd.Series, optional
        Market capitalisation weights for each asset.
        If None, equal weights are used as the prior.
    risk_aversion   : float   Risk aversion coefficient δ (default 2.5).
    tau             : float   Uncertainty in equilibrium prior (default 0.05).
    risk_free_rate  : float   Annualised risk-free rate.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        market_weights: Optional[pd.Series] = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.0435,
        periods_per_year: int = 252,
    ) -> None:
        self.returns = returns.dropna().copy()
        self.tickers = list(self.returns.columns)
        self.n = len(self.tickers)
        self.delta = risk_aversion
        self.tau = tau
        self.rfr = risk_free_rate
        self.ppy = periods_per_year

        # Annualised covariance
        self.Sigma = self._ledoit_wolf(self.returns) * self.ppy

        # Market weights (prior)
        if market_weights is not None:
            w = market_weights.reindex(self.tickers).fillna(0)
            self.w_mkt = (w / w.sum()).values
        else:
            self.w_mkt = np.full(self.n, 1 / self.n)

        # Equilibrium (implied) returns: π = δΣw
        self.pi = self.delta * self.Sigma @ self.w_mkt

        # Views storage
        self._views: list[BLView] = []

    # ──────────────────────────────────────
    #  Covariance estimation
    # ──────────────────────────────────────

    def _ledoit_wolf(self, returns: pd.DataFrame) -> np.ndarray:
        n_obs, p = returns.shape
        S = returns.cov().values
        mu_lw = np.trace(S) / p
        delta = np.linalg.norm(S - mu_lw * np.eye(p), "fro") ** 2
        beta2 = 0.0
        for i in range(n_obs):
            x = returns.iloc[i].values
            outer = np.outer(x, x)
            beta2 += np.linalg.norm(outer - S, "fro") ** 2
        beta2 /= n_obs ** 2
        alpha = min(beta2 / delta, 1.0) if delta > 0 else 0
        return (1 - alpha) * S + alpha * mu_lw * np.eye(p)

    # ──────────────────────────────────────
    #  View construction
    # ──────────────────────────────────────

    def add_absolute_view(
        self,
        ticker: str,
        expected_return: float,
        confidence: float = 0.5,
    ) -> None:
        """
        Add an absolute return view.
        "I believe {ticker} will return {expected_return} per annum."

        Parameters
        ----------
        ticker          : str    Asset ticker.
        expected_return : float  Annual return forecast (e.g. 0.15 = 15%).
        confidence      : float  Confidence in this view [0, 1].
                                 0 = no confidence (prior dominates).
                                 1 = certainty (view dominates).
        """
        if ticker not in self.tickers:
            raise ValueError(f"{ticker} not in returns DataFrame.")

        p = np.zeros(self.n)
        p[self.tickers.index(ticker)] = 1.0

        self._views.append(BLView(
            description=f"{ticker} returns {expected_return:.1%} p.a.",
            p_vector=p,
            q_value=expected_return,
            confidence=confidence,
            view_type="absolute",
        ))

    def add_relative_view(
        self,
        ticker_long: str,
        ticker_short: str,
        outperformance: float,
        confidence: float = 0.5,
    ) -> None:
        """
        Add a relative return view.
        "{ticker_long} will outperform {ticker_short} by {outperformance} p.a."

        Parameters
        ----------
        ticker_long     : str    The outperforming asset.
        ticker_short    : str    The underperforming asset.
        outperformance  : float  Predicted relative return (e.g. 0.03 = 3%).
        confidence      : float  Confidence in this view [0, 1].
        """
        for t in [ticker_long, ticker_short]:
            if t not in self.tickers:
                raise ValueError(f"{t} not in returns DataFrame.")

        p = np.zeros(self.n)
        p[self.tickers.index(ticker_long)] = 1.0
        p[self.tickers.index(ticker_short)] = -1.0

        self._views.append(BLView(
            description=(
                f"{ticker_long} outperforms {ticker_short} "
                f"by {outperformance:.1%} p.a."
            ),
            p_vector=p,
            q_value=outperformance,
            confidence=confidence,
            view_type="relative",
        ))

    def clear_views(self) -> None:
        """Remove all views and revert to pure equilibrium model."""
        self._views = []

    # ──────────────────────────────────────
    #  BL posterior computation
    # ──────────────────────────────────────

    def _compute_posterior(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior expected returns and covariance.

        BL Formula:
        E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 × [(τΣ)^-1π + P'Ω^-1Q]

        Returns
        -------
        (mu_posterior, Sigma_posterior)
        """
        tau_Sigma = self.tau * self.Sigma
        tau_Sigma_inv = np.linalg.inv(tau_Sigma + 1e-8 * np.eye(self.n))

        if not self._views:
            # No views: pure equilibrium
            return self.pi, self.Sigma + tau_Sigma

        k = len(self._views)
        P = np.array([v.p_vector for v in self._views])   # (k, n)
        Q = np.array([v.q_value for v in self._views])    # (k,)

        # Omega: diagonal view uncertainty matrix
        # Uncertainty = (1 - confidence) / confidence × P Σ P' diagonal
        P_Sigma_P = P @ self.Sigma @ P.T
        Omega = np.zeros((k, k))
        for i, view in enumerate(self._views):
            c = np.clip(view.confidence, 0.01, 0.99)
            uncertainty = ((1 - c) / c) * P_Sigma_P[i, i]
            Omega[i, i] = max(uncertainty, 1e-8)

        Omega_inv = np.diag(1 / np.diag(Omega))

        # BL posterior mean
        A = tau_Sigma_inv + P.T @ Omega_inv @ P
        try:
            A_inv = np.linalg.inv(A + 1e-8 * np.eye(self.n))
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)

        b = tau_Sigma_inv @ self.pi + P.T @ Omega_inv @ Q
        mu_posterior = A_inv @ b

        # BL posterior covariance
        Sigma_posterior = self.Sigma + A_inv

        return mu_posterior, Sigma_posterior

    # ──────────────────────────────────────
    #  Optimisation
    # ──────────────────────────────────────

    def optimise(
        self,
        allow_short: bool = False,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        objective: str = "max_sharpe",
    ) -> BLResult:
        """
        Compute BL posterior returns and optimise the portfolio.

        Parameters
        ----------
        allow_short  : bool   Allow short positions.
        min_weight   : float  Minimum weight per asset.
        max_weight   : float  Maximum weight per asset.
        objective    : str    'max_sharpe' | 'min_vol' | 'risk_parity'

        Returns
        -------
        BLResult  with posterior returns, optimal weights, and metrics.
        """
        mu_post, Sigma_post = self._compute_posterior()

        # Use posterior inputs with Markowitz
        from scipy.optimize import minimize

        n = self.n
        bounds = [(max(0, min_weight), max_weight)] * n
        if allow_short:
            bounds = [(-max_weight, max_weight)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def neg_sharpe(w):
            ret = float(w @ mu_post)
            var = float(w @ Sigma_post @ w)
            vol = np.sqrt(max(var, 1e-10))
            return -(ret - self.rfr) / vol

        def portfolio_variance(w):
            return float(w @ Sigma_post @ w)

        def risk_parity_obj(w):
            w_abs = np.abs(w)
            w_abs = w_abs / (w_abs.sum() + 1e-10)
            sigma_p = np.sqrt(w_abs @ Sigma_post @ w_abs)
            if sigma_p < 1e-10:
                return 1e10
            rc = w_abs * (Sigma_post @ w_abs) / sigma_p
            target = sigma_p / n
            return float(np.sum((rc - target) ** 2))

        obj_map = {
            "max_sharpe": neg_sharpe,
            "min_vol": portfolio_variance,
            "risk_parity": risk_parity_obj,
        }
        obj_fn = obj_map.get(objective, neg_sharpe)

        w0 = np.full(n, 1 / n)
        result = minimize(
            obj_fn, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        w = np.abs(result.x)
        if not allow_short:
            w = np.clip(w, 0, max_weight)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum

        exp_ret = float(w @ mu_post)
        exp_vol = float(np.sqrt(max(w @ Sigma_post @ w, 0)))
        sharpe = (exp_ret - self.rfr) / exp_vol if exp_vol > 0 else 0

        # Views summary
        view_rows = []
        for v in self._views:
            view_rows.append({
                "View": v.description,
                "Type": v.view_type,
                "Q (Return)": v.q_value,
                "Confidence": v.confidence,
            })
        views_df = pd.DataFrame(view_rows) if view_rows else pd.DataFrame()

        return BLResult(
            posterior_returns=pd.Series(mu_post, index=self.tickers),
            posterior_covariance=Sigma_post,
            equilibrium_returns=pd.Series(self.pi, index=self.tickers),
            optimal_weights=pd.Series(w, index=self.tickers),
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            sharpe_ratio=sharpe,
            views_summary=views_df,
        )

    # ──────────────────────────────────────
    #  Sensitivity analysis
    # ──────────────────────────────────────

    def confidence_sensitivity(
        self,
        n_levels: int = 10,
    ) -> pd.DataFrame:
        """
        Show how optimal weights change as view confidence varies from 0 to 1.
        Helps calibrate the right confidence level for each view.

        Returns
        -------
        pd.DataFrame  confidence levels × asset weights.
        """
        if not self._views:
            return pd.DataFrame()

        confidences = np.linspace(0.05, 0.95, n_levels)
        original_views = [v for v in self._views]
        records = []

        for conf in confidences:
            for v in self._views:
                v.confidence = conf
            r = self.optimise()
            row = {"Confidence": round(conf, 2)}
            for t, w in r.optimal_weights.items():
                row[t] = round(float(w), 4)
            records.append(row)

        # Restore originals
        self._views = original_views
        return pd.DataFrame(records).set_index("Confidence")

    def equilibrium_vs_posterior(self) -> pd.DataFrame:
        """
        Compare equilibrium returns vs BL posterior returns.
        Shows how views have shifted return expectations.
        """
        mu_post, _ = self._compute_posterior()
        df = pd.DataFrame({
            "Equilibrium Return": pd.Series(self.pi, index=self.tickers),
            "Posterior Return": pd.Series(mu_post, index=self.tickers),
            "Shift": pd.Series(mu_post - self.pi, index=self.tickers),
        })
        return df.sort_values("Shift", ascending=False)
