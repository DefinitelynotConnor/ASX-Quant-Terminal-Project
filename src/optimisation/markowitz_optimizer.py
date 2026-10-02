"""
markowitz_optimizer.py
======================
Markowitz (1952) Mean-Variance Portfolio Optimisation.

Finds the portfolio weights that maximise risk-adjusted return
subject to constraints. Traces the full efficient frontier.

Optimisation Objectives
-----------------------
    max_sharpe      : Maximum Sharpe Ratio portfolio
    min_volatility  : Minimum variance portfolio
    max_return      : Maximum return at given risk level
    min_vol_target  : Minimum vol subject to return target
    risk_parity     : Equal risk contribution (see RiskParity class)
    efficient_frontier : Full curve of optimal portfolios

Constraints Supported
---------------------
    Long only (no short selling)
    Position size limits (min/max weight per asset)
    Sector exposure limits
    Turnover limits
    Target return / target volatility

Usage
-----
    from src.optimisation.markowitz_optimizer import MarkowitzOptimizer

    optimizer = MarkowitzOptimizer(returns, risk_free_rate=0.0435)
    result = optimizer.max_sharpe()
    frontier = optimizer.efficient_frontier(n_points=50)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Result Container
# ─────────────────────────────────────────────

@dataclass
class OptimisationResult:
    """Container for portfolio optimisation results."""
    weights: pd.Series = field(default_factory=pd.Series)
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    objective: str = ""
    success: bool = False
    message: str = ""
    iterations: int = 0

    def summary(self) -> str:
        lines = [
            f"\nOptimisation Result: {self.objective}",
            f"  Status     : {'✅ Converged' if self.success else '❌ Failed'}",
            f"  Exp Return : {self.expected_return:.2%}",
            f"  Exp Vol    : {self.expected_volatility:.2%}",
            f"  Sharpe     : {self.sharpe_ratio:.3f}",
            f"\n  Weights:",
        ]
        for ticker, w in self.weights.sort_values(ascending=False).items():
            if abs(w) > 0.001:
                lines.append(f"    {ticker:<15} {w:.2%}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Markowitz Optimizer
# ─────────────────────────────────────────────

class MarkowitzOptimizer:
    """
    Mean-variance portfolio optimiser.

    Parameters
    ----------
    returns         : pd.DataFrame   Daily return series, columns = tickers.
    risk_free_rate  : float          Annualised risk-free rate.
    cov_method      : str            'sample' | 'ledoit_wolf' | 'ewma'
    periods_per_year: int            Trading days per year.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0435,
        cov_method: str = "ledoit_wolf",
        periods_per_year: int = 252,
    ) -> None:
        self.returns = returns.dropna().copy()
        self.tickers = list(self.returns.columns)
        self.n = len(self.tickers)
        self.rfr = risk_free_rate
        self.ppy = periods_per_year
        self.cov_method = cov_method

        # Annualised inputs
        self.mu = self.returns.mean() * self.ppy
        self.Sigma = self._estimate_covariance()

    # ──────────────────────────────────────
    #  Covariance estimation
    # ──────────────────────────────────────

    def _estimate_covariance(self) -> np.ndarray:
        """Estimate annualised covariance matrix."""
        if self.cov_method == "ledoit_wolf":
            return self._ledoit_wolf() * self.ppy
        elif self.cov_method == "ewma":
            lam = 0.94
            ret = self.returns.values
            cov = np.cov(ret[:10].T)
            for r in ret[10:]:
                r = r.reshape(-1, 1)
                cov = lam * cov + (1 - lam) * (r @ r.T)
            return cov * self.ppy
        return self.returns.cov().values * self.ppy

    def _ledoit_wolf(self) -> np.ndarray:
        """Ledoit-Wolf analytical shrinkage estimator."""
        n, p = self.returns.shape
        S = self.returns.cov().values
        mu_lw = np.trace(S) / p
        delta = np.linalg.norm(S - mu_lw * np.eye(p), "fro") ** 2
        beta2 = 0.0
        for i in range(n):
            x = self.returns.iloc[i].values
            outer = np.outer(x, x)
            beta2 += np.linalg.norm(outer - S, "fro") ** 2
        beta2 /= n ** 2
        alpha = min(beta2 / delta, 1.0) if delta > 0 else 0
        return (1 - alpha) * S + alpha * mu_lw * np.eye(p)

    # ──────────────────────────────────────
    #  Portfolio statistics
    # ──────────────────────────────────────

    def portfolio_stats(self, w: np.ndarray) -> tuple[float, float, float]:
        """Return (expected_return, volatility, sharpe) for weight vector w."""
        ret = float(w @ self.mu)
        var = float(w @ self.Sigma @ w)
        vol = np.sqrt(max(var, 0))
        sharpe = (ret - self.rfr) / vol if vol > 0 else 0
        return ret, vol, sharpe

    # ──────────────────────────────────────
    #  Constraint builders
    # ──────────────────────────────────────

    def _base_constraints(
        self,
        allow_short: bool = False,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        sector_limits: Optional[dict] = None,
        sector_map: Optional[dict] = None,
    ) -> tuple[list, list]:
        """Build scipy constraints and bounds."""

        # Weights sum to 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Sector constraints
        if sector_limits and sector_map:
            for sector, limit in sector_limits.items():
                idx = [
                    i for i, t in enumerate(self.tickers)
                    if sector_map.get(t) == sector
                ]
                if idx:
                    def sector_con(w, ix=idx, lim=limit):
                        return lim - np.sum(w[ix])
                    constraints.append({"type": "ineq", "fun": sector_con})

        # Bounds per asset
        if allow_short:
            bounds = [(-max_weight, max_weight)] * self.n
        else:
            bounds = [(max(0, min_weight), max_weight)] * self.n

        return constraints, bounds

    # ──────────────────────────────────────
    #  Maximum Sharpe
    # ──────────────────────────────────────

    def max_sharpe(
        self,
        allow_short: bool = False,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        sector_limits: Optional[dict] = None,
        sector_map: Optional[dict] = None,
    ) -> OptimisationResult:
        """
        Find the portfolio with the maximum Sharpe Ratio.
        The tangency portfolio on the efficient frontier.
        """
        constraints, bounds = self._base_constraints(
            allow_short, min_weight, max_weight, sector_limits, sector_map
        )

        def neg_sharpe(w):
            ret, vol, sharpe = self.portfolio_stats(w)
            return -sharpe if vol > 0 else 0

        w0 = np.full(self.n, 1 / self.n)
        result = minimize(
            neg_sharpe, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        w = result.x
        ret, vol, sharpe = self.portfolio_stats(w)

        return OptimisationResult(
            weights=pd.Series(w, index=self.tickers),
            expected_return=ret,
            expected_volatility=vol,
            sharpe_ratio=sharpe,
            objective="Maximum Sharpe Ratio",
            success=result.success,
            message=result.message,
            iterations=result.nit,
        )

    # ──────────────────────────────────────
    #  Minimum Volatility
    # ──────────────────────────────────────

    def min_volatility(
        self,
        allow_short: bool = False,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        target_return: Optional[float] = None,
        sector_limits: Optional[dict] = None,
        sector_map: Optional[dict] = None,
    ) -> OptimisationResult:
        """
        Find the minimum variance portfolio.
        Optionally subject to a minimum return constraint.
        """
        constraints, bounds = self._base_constraints(
            allow_short, min_weight, max_weight, sector_limits, sector_map
        )

        if target_return is not None:
            constraints.append({
                "type": "ineq",
                "fun": lambda w: float(w @ self.mu) - target_return,
            })

        def portfolio_variance(w):
            return float(w @ self.Sigma @ w)

        w0 = np.full(self.n, 1 / self.n)
        result = minimize(
            portfolio_variance, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        w = result.x
        ret, vol, sharpe = self.portfolio_stats(w)

        return OptimisationResult(
            weights=pd.Series(w, index=self.tickers),
            expected_return=ret,
            expected_volatility=vol,
            sharpe_ratio=sharpe,
            objective="Minimum Volatility",
            success=result.success,
            message=result.message,
            iterations=result.nit,
        )

    # ──────────────────────────────────────
    #  Maximum Return
    # ──────────────────────────────────────

    def max_return(
        self,
        target_vol: Optional[float] = None,
        allow_short: bool = False,
        max_weight: float = 1.0,
    ) -> OptimisationResult:
        """
        Maximum return portfolio, optionally subject to a volatility target.
        """
        constraints, bounds = self._base_constraints(allow_short, 0, max_weight)

        if target_vol is not None:
            constraints.append({
                "type": "ineq",
                "fun": lambda w: target_vol - np.sqrt(w @ self.Sigma @ w),
            })

        def neg_return(w):
            return -float(w @ self.mu)

        w0 = np.full(self.n, 1 / self.n)
        result = minimize(
            neg_return, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        w = result.x
        ret, vol, sharpe = self.portfolio_stats(w)

        return OptimisationResult(
            weights=pd.Series(w, index=self.tickers),
            expected_return=ret,
            expected_volatility=vol,
            sharpe_ratio=sharpe,
            objective="Maximum Return",
            success=result.success,
            message=result.message,
        )

    # ──────────────────────────────────────
    #  Efficient Frontier
    # ──────────────────────────────────────

    def efficient_frontier(
        self,
        n_points: int = 50,
        allow_short: bool = False,
        max_weight: float = 1.0,
    ) -> pd.DataFrame:
        """
        Trace the full efficient frontier.

        For each target return level (from min to max feasible),
        find the minimum variance portfolio.

        Returns
        -------
        pd.DataFrame  Columns: Return, Volatility, Sharpe, weights...
        """
        # Bounds for feasible returns
        min_ret = float(self.mu.min())
        max_ret = float(self.mu.max())
        target_returns = np.linspace(min_ret, max_ret, n_points)

        points = []
        constraints_base, bounds = self._base_constraints(allow_short, 0, max_weight)

        for target in target_returns:
            constraints = constraints_base + [{
                "type": "eq",
                "fun": lambda w, t=target: float(w @ self.mu) - t,
            }]

            def portfolio_variance(w):
                return float(w @ self.Sigma @ w)

            w0 = np.full(self.n, 1 / self.n)
            result = minimize(
                portfolio_variance, w0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"ftol": 1e-10, "maxiter": 500},
            )

            if result.success:
                w = result.x
                ret, vol, sharpe = self.portfolio_stats(w)
                row = {
                    "Return": ret,
                    "Volatility": vol,
                    "Sharpe": sharpe,
                }
                for i, t in enumerate(self.tickers):
                    row[t] = float(w[i])
                points.append(row)

        return pd.DataFrame(points)

    # ──────────────────────────────────────
    #  Monte Carlo frontier
    # ──────────────────────────────────────

    def monte_carlo_portfolios(
        self,
        n_portfolios: int = 5000,
        allow_short: bool = False,
    ) -> pd.DataFrame:
        """
        Generate random portfolios to visualise the feasible set.
        Used for plotting alongside the efficient frontier.

        Returns
        -------
        pd.DataFrame  Columns: Return, Volatility, Sharpe, [weights...]
        """
        np.random.seed(42)
        records = []

        for _ in range(n_portfolios):
            if allow_short:
                w = np.random.uniform(-1, 1, self.n)
            else:
                w = np.random.dirichlet(np.ones(self.n))

            ret, vol, sharpe = self.portfolio_stats(w)
            row = {"Return": ret, "Volatility": vol, "Sharpe": sharpe}
            records.append(row)

        return pd.DataFrame(records)

    # ──────────────────────────────────────
    #  Risk Parity
    # ──────────────────────────────────────

    def risk_parity(self) -> OptimisationResult:
        """
        Risk Parity (Equal Risk Contribution) portfolio.
        Each asset contributes equally to total portfolio variance.
        σ_p × w_i × (Σw)_i / σ_p = 1/N for all i.
        """
        def risk_parity_objective(w):
            w = np.abs(w)
            w /= w.sum()
            sigma_p = np.sqrt(w @ self.Sigma @ w)
            if sigma_p < 1e-10:
                return 1e10
            marginal = self.Sigma @ w
            risk_contrib = w * marginal / sigma_p
            target = sigma_p / self.n
            return float(np.sum((risk_contrib - target) ** 2))

        w0 = np.full(self.n, 1 / self.n)
        bounds = [(0, 1)] * self.n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(
            risk_parity_objective, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 2000},
        )

        w = np.abs(result.x)
        w /= w.sum()
        ret, vol, sharpe = self.portfolio_stats(w)

        return OptimisationResult(
            weights=pd.Series(w, index=self.tickers),
            expected_return=ret,
            expected_volatility=vol,
            sharpe_ratio=sharpe,
            objective="Risk Parity",
            success=result.success,
            message=result.message,
        )

    # ──────────────────────────────────────
    #  Comparison table
    # ──────────────────────────────────────

    def compare_portfolios(self) -> pd.DataFrame:
        """
        Run all optimisation objectives and compare results.

        Returns
        -------
        pd.DataFrame  One row per portfolio type.
        """
        portfolios = {
            "Max Sharpe": self.max_sharpe(),
            "Min Volatility": self.min_volatility(),
            "Risk Parity": self.risk_parity(),
            "Equal Weight": OptimisationResult(
                weights=pd.Series(
                    np.full(self.n, 1 / self.n), index=self.tickers
                ),
                **dict(zip(
                    ["expected_return", "expected_volatility", "sharpe_ratio"],
                    self.portfolio_stats(np.full(self.n, 1 / self.n)),
                )),
                objective="Equal Weight",
                success=True,
            ),
        }

        rows = []
        for name, r in portfolios.items():
            rows.append({
                "Portfolio": name,
                "Expected Return": r.expected_return,
                "Volatility": r.expected_volatility,
                "Sharpe Ratio": r.sharpe_ratio,
                "Max Weight": float(r.weights.max()),
                "Min Weight": float(r.weights[r.weights > 0.001].min())
                if (r.weights > 0.001).any() else 0,
                "N Holdings": int((r.weights > 0.001).sum()),
            })

        return pd.DataFrame(rows).set_index("Portfolio")
