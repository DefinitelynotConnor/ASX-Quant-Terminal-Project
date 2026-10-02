"""
monte_carlo_paths.py  /  stress_testing.py
==========================================
Monte Carlo simulation framework for ASX equity research.

Simulations Implemented
-----------------------
GeometricBrownianMotion : Standard GBM price path simulation
JumpDiffusion           : Merton (1976) jump-diffusion model
GARCHPaths              : Volatility clustering in simulated paths
CorrelatedPaths         : Multi-asset correlated GBM via Cholesky
PortfolioSimulation     : Full portfolio NAV simulation with distributions
StressTesting           : Scenario-based and historical stress tests
VaRBacktest             : Kupiec test for VaR model validation

Usage
-----
    from src.simulation.monte_carlo_paths import MonteCarloSimulator

    mc = MonteCarloSimulator(returns, n_simulations=10000, horizon=252)
    paths = mc.simulate_gbm()
    var_95 = mc.portfolio_var(weights, confidence=0.95)
    stress = mc.stress_test()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Result Containers
# ─────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation output."""
    paths: np.ndarray = field(default_factory=lambda: np.array([]))
    final_prices: np.ndarray = field(default_factory=lambda: np.array([]))
    horizon: int = 252
    n_simulations: int = 10000
    model: str = "GBM"
    percentiles: pd.DataFrame = field(default_factory=pd.DataFrame)
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    prob_loss: float = 0.0
    expected_return: float = 0.0


@dataclass
class StressTestResult:
    """Container for stress test results."""
    scenario_name: str = ""
    portfolio_impact: float = 0.0
    asset_impacts: dict = field(default_factory=dict)
    var_stressed: float = 0.0
    worst_case: float = 0.0
    description: str = ""


# ─────────────────────────────────────────────
#  Monte Carlo Simulator
# ─────────────────────────────────────────────

class MonteCarloSimulator:
    """
    Monte Carlo price path and portfolio simulation engine.

    Parameters
    ----------
    returns        : pd.Series or pd.DataFrame   Historical daily returns.
    n_simulations  : int    Number of simulation paths (default 10,000).
    horizon        : int    Simulation horizon in trading days (default 252).
    seed           : int    Random seed for reproducibility.
    """

    ANNUALISE = np.sqrt(252)

    def __init__(
        self,
        returns: pd.Series | pd.DataFrame,
        n_simulations: int = 10_000,
        horizon: int = 252,
        seed: int = 42,
    ) -> None:
        self.is_multi = isinstance(returns, pd.DataFrame)
        if self.is_multi:
            self.returns_df = returns.dropna().copy()
            self.returns = returns.mean(axis=1).dropna()
            self.tickers = list(returns.columns)
        else:
            self.returns = returns.dropna().copy()
            self.returns_df = None
            self.tickers = []

        self.n_sim = n_simulations
        self.horizon = horizon
        self.seed = seed

        # Fit parameters from historical data
        self.mu = float(self.returns.mean() * 252)
        self.sigma = float(self.returns.std() * np.sqrt(252))
        self.mu_daily = float(self.returns.mean())
        self.sigma_daily = float(self.returns.std())

    # ──────────────────────────────────────
    #  Geometric Brownian Motion
    # ──────────────────────────────────────

    def simulate_gbm(
        self,
        S0: float = 100.0,
        mu: Optional[float] = None,
        sigma: Optional[float] = None,
    ) -> SimulationResult:
        """
        Simulate price paths using Geometric Brownian Motion.

        dS = μS dt + σS dW_t
        Discrete: S_{t+1} = S_t × exp[(μ - σ²/2)dt + σ√dt × Z]

        Parameters
        ----------
        S0    : float   Starting price (default 100).
        mu    : float   Annualised drift (defaults to historical mean).
        sigma : float   Annualised volatility (defaults to historical vol).

        Returns
        -------
        SimulationResult with (n_sim × horizon) path matrix.
        """
        np.random.seed(self.seed)
        mu = mu if mu is not None else self.mu
        sigma = sigma if sigma is not None else self.sigma
        dt = 1 / 252

        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)

        # Generate all random shocks at once (vectorised)
        Z = np.random.standard_normal((self.n_sim, self.horizon))
        log_returns = drift + diffusion * Z
        log_price_paths = np.cumsum(log_returns, axis=1)
        paths = S0 * np.exp(log_price_paths)

        # Prepend starting price
        paths = np.column_stack([np.full(self.n_sim, S0), paths])

        return self._build_result(paths, "GBM")

    # ──────────────────────────────────────
    #  Jump Diffusion (Merton 1976)
    # ──────────────────────────────────────

    def simulate_jump_diffusion(
        self,
        S0: float = 100.0,
        jump_intensity: float = 5.0,
        jump_mean: float = -0.05,
        jump_std: float = 0.08,
    ) -> SimulationResult:
        """
        Merton (1976) jump-diffusion model.
        Adds discrete jumps (crashes) on top of GBM.

        Parameters
        ----------
        S0              : float   Starting price.
        jump_intensity  : float   Expected number of jumps per year (λ).
        jump_mean       : float   Average jump size (e.g. -0.05 = -5%).
        jump_std        : float   Jump size standard deviation.
        """
        np.random.seed(self.seed)
        dt = 1 / 252
        mu = self.mu
        sigma = self.sigma

        # Adjust drift for jump component
        jump_correction = jump_intensity * (
            np.exp(jump_mean + 0.5 * jump_std ** 2) - 1
        )
        adj_drift = (mu - jump_correction - 0.5 * sigma ** 2) * dt

        Z = np.random.standard_normal((self.n_sim, self.horizon))
        # Poisson jump counts
        N = np.random.poisson(jump_intensity * dt, (self.n_sim, self.horizon))
        # Jump sizes
        J = np.zeros_like(Z)
        for i in range(self.n_sim):
            for t in range(self.horizon):
                if N[i, t] > 0:
                    jumps = np.random.normal(jump_mean, jump_std, N[i, t])
                    J[i, t] = np.sum(jumps)

        log_returns = adj_drift + sigma * np.sqrt(dt) * Z + J
        log_price_paths = np.cumsum(log_returns, axis=1)
        paths = S0 * np.exp(log_price_paths)
        paths = np.column_stack([np.full(self.n_sim, S0), paths])

        return self._build_result(paths, "Jump Diffusion")

    # ──────────────────────────────────────
    #  Bootstrap (Historical Simulation)
    # ──────────────────────────────────────

    def simulate_bootstrap(
        self,
        S0: float = 100.0,
        block_size: int = 5,
    ) -> SimulationResult:
        """
        Block bootstrap simulation: resample blocks of historical returns.
        Preserves autocorrelation structure (volatility clustering).

        Parameters
        ----------
        S0         : float   Starting price.
        block_size : int     Bootstrap block size in days.
        """
        np.random.seed(self.seed)
        hist = self.returns.values
        n_hist = len(hist)
        paths = np.zeros((self.n_sim, self.horizon + 1))
        paths[:, 0] = S0

        n_blocks = int(np.ceil(self.horizon / block_size))

        for i in range(self.n_sim):
            sim_returns = []
            for _ in range(n_blocks):
                start = np.random.randint(0, n_hist - block_size)
                block = hist[start: start + block_size]
                sim_returns.extend(block)
            sim_returns = np.array(sim_returns[:self.horizon])
            paths[i, 1:] = paths[i, 0] * np.exp(np.cumsum(sim_returns))

        return self._build_result(paths, "Bootstrap")

    # ──────────────────────────────────────
    #  Correlated Multi-Asset GBM
    # ──────────────────────────────────────

    def simulate_correlated_gbm(
        self,
        weights: Optional[np.ndarray] = None,
        S0: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Simulate correlated price paths for multiple assets.
        Uses Cholesky decomposition to impose historical correlation structure.

        Parameters
        ----------
        weights : np.ndarray   Portfolio weights (default = equal weight).
        S0      : np.ndarray   Starting prices (default = 100 each).

        Returns
        -------
        dict with asset_paths (n_assets, n_sim, horizon) and portfolio_paths.
        """
        if self.returns_df is None:
            raise ValueError("Multi-asset simulation requires DataFrame returns.")

        np.random.seed(self.seed)
        n_assets = len(self.tickers)
        weights = weights if weights is not None else np.full(n_assets, 1 / n_assets)
        S0 = S0 if S0 is not None else np.full(n_assets, 100.0)

        mu_vec = self.returns_df.mean().values * 252
        Sigma = self.returns_df.cov().values * 252
        dt = 1 / 252

        # Cholesky decomposition for correlated normals
        try:
            L = cholesky(Sigma * dt, lower=True)
        except Exception:
            L = cholesky(Sigma * dt + 1e-8 * np.eye(n_assets), lower=True)

        drift = (mu_vec - 0.5 * np.diag(Sigma)) * dt

        asset_paths = np.zeros((n_assets, self.n_sim, self.horizon + 1))
        for k in range(n_assets):
            asset_paths[k, :, 0] = S0[k]

        for t in range(self.horizon):
            Z = np.random.standard_normal((n_assets, self.n_sim))
            corr_Z = L @ Z   # (n_assets, n_sim)
            for k in range(n_assets):
                log_ret = drift[k] + corr_Z[k]
                asset_paths[k, :, t + 1] = (
                    asset_paths[k, :, t] * np.exp(log_ret)
                )

        # Portfolio NAV paths
        port_paths = np.zeros((self.n_sim, self.horizon + 1))
        for k in range(n_assets):
            port_paths += weights[k] * (asset_paths[k] / S0[k])
        port_paths = port_paths * 100

        return {
            "asset_paths": asset_paths,
            "portfolio_paths": port_paths,
            "tickers": self.tickers,
            "weights": weights,
        }

    # ──────────────────────────────────────
    #  Risk metrics from simulated paths
    # ──────────────────────────────────────

    def portfolio_var(
        self,
        weights: np.ndarray,
        confidence: float = 0.95,
        horizon: int = 1,
    ) -> float:
        """
        Monte Carlo VaR for a portfolio over a given horizon.

        Parameters
        ----------
        weights    : np.ndarray   Portfolio weights.
        confidence : float        Confidence level.
        horizon    : int          Holding period in days.
        """
        if self.returns_df is None:
            raise ValueError("Requires DataFrame returns for portfolio VaR.")

        np.random.seed(self.seed)
        n_assets = len(self.tickers)
        Sigma = self.returns_df.cov().values * horizon
        mu_vec = self.returns_df.mean().values * horizon

        # Cholesky
        try:
            L = cholesky(Sigma, lower=True)
        except Exception:
            L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(n_assets))

        Z = np.random.standard_normal((n_assets, self.n_sim))
        corr_Z = (L @ Z).T  # (n_sim, n_assets)

        port_returns = corr_Z @ weights + weights @ mu_vec
        return float(-np.percentile(port_returns, (1 - confidence) * 100))

    def _build_result(
        self, paths: np.ndarray, model: str
    ) -> SimulationResult:
        """Compute statistics from simulation paths."""
        S0 = paths[:, 0].mean()
        final = paths[:, -1]
        returns = final / S0 - 1

        percentile_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct = {f"P{p}": float(np.percentile(final, p)) for p in percentile_levels}
        pct_df = pd.DataFrame(
            list(pct.items()), columns=["Percentile", "Price"]
        )

        var_95 = float(-np.percentile(returns, 5))
        var_99 = float(-np.percentile(returns, 1))
        tail_95 = returns[returns <= -var_95]
        tail_99 = returns[returns <= -var_99]

        return SimulationResult(
            paths=paths,
            final_prices=final,
            horizon=self.horizon,
            n_simulations=self.n_sim,
            model=model,
            percentiles=pct_df,
            var_95=var_95,
            var_99=var_99,
            cvar_95=float(-tail_95.mean()) if len(tail_95) > 0 else 0,
            cvar_99=float(-tail_99.mean()) if len(tail_99) > 0 else 0,
            prob_loss=float((returns < 0).mean()),
            expected_return=float(returns.mean()),
        )

    # ──────────────────────────────────────
    #  Summary statistics
    # ──────────────────────────────────────

    def simulation_summary(
        self, result: SimulationResult
    ) -> pd.DataFrame:
        """Return formatted summary of simulation results."""
        rows = [
            ("Model", result.model),
            ("Simulations", f"{result.n_simulations:,}"),
            ("Horizon (days)", str(result.horizon)),
            ("Expected Return", f"{result.expected_return:.2%}"),
            ("Probability of Loss", f"{result.prob_loss:.1%}"),
            ("VaR 95%", f"{result.var_95:.2%}"),
            ("VaR 99%", f"{result.var_99:.2%}"),
            ("CVaR 95%", f"{result.cvar_95:.2%}"),
            ("CVaR 99%", f"{result.cvar_99:.2%}"),
            ("5th Percentile Price", f"{result.percentiles[result.percentiles['Percentile']=='P5']['Price'].values[0]:.2f}"
             if 'P5' in result.percentiles['Percentile'].values else "N/A"),
            ("Median Price", f"{result.percentiles[result.percentiles['Percentile']=='P50']['Price'].values[0]:.2f}"
             if 'P50' in result.percentiles['Percentile'].values else "N/A"),
            ("95th Percentile Price", f"{result.percentiles[result.percentiles['Percentile']=='P95']['Price'].values[0]:.2f}"
             if 'P95' in result.percentiles['Percentile'].values else "N/A"),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"])


# ─────────────────────────────────────────────
#  Stress Testing
# ─────────────────────────────────────────────

class StressTester:
    """
    Scenario-based and historical stress testing for ASX portfolios.

    Parameters
    ----------
    returns  : pd.DataFrame   Asset daily returns.
    weights  : dict           {ticker: weight}
    """

    # Historical ASX crisis scenarios
    HISTORICAL_SCENARIOS = {
        "GFC 2008": {
            "description": "Global Financial Crisis — ASX fell ~54% peak to trough",
            "start": "2008-09-01",
            "end": "2009-03-09",
        },
        "COVID Crash 2020": {
            "description": "COVID pandemic crash — ASX fell ~37% in 5 weeks",
            "start": "2020-02-20",
            "end": "2020-03-23",
        },
        "Dot-Com Bust 2000": {
            "description": "Technology sector collapse",
            "start": "2000-03-01",
            "end": "2002-10-01",
        },
        "China Growth Scare 2015": {
            "description": "Chinese market crash and currency devaluation",
            "start": "2015-06-01",
            "end": "2016-01-31",
        },
        "RBA Rate Hikes 2022": {
            "description": "Rapid RBA rate hike cycle — ASX fell ~15%",
            "start": "2022-01-01",
            "end": "2022-10-31",
        },
    }

    # Hypothetical factor shocks
    HYPOTHETICAL_SCENARIOS = {
        "Severe Recession": {
            "description": "Deep recession — equity markets fall 40%",
            "market_shock": -0.40,
            "vol_multiplier": 3.0,
        },
        "Mild Recession": {
            "description": "Mild recession — equity markets fall 20%",
            "market_shock": -0.20,
            "vol_multiplier": 1.5,
        },
        "Rate Shock +300bps": {
            "description": "Sudden 300bps rate rise — banks and REITs hit hardest",
            "market_shock": -0.18,
            "sector_shocks": {
                "Financials": -0.22, "Real Estate": -0.28,
                "Utilities": -0.15, "Materials": -0.08,
            },
        },
        "China Hard Landing": {
            "description": "Chinese GDP growth collapses — miners devastated",
            "market_shock": -0.25,
            "sector_shocks": {
                "Materials": -0.45, "Energy": -0.30,
                "Financials": -0.20, "Healthcare": -0.05,
            },
        },
        "AUD Collapse -25%": {
            "description": "AUD/USD falls 25% — importers hurt, exporters benefit",
            "market_shock": -0.10,
            "sector_shocks": {
                "Consumer Discretionary": -0.20,
                "Materials": 0.10, "Healthcare": 0.05,
            },
        },
        "Tech Selloff -35%": {
            "description": "Global tech selloff",
            "market_shock": -0.12,
            "sector_shocks": {
                "Information Technology": -0.35,
                "Communication Services": -0.20,
            },
        },
        "Iron Ore -50%": {
            "description": "Iron ore price collapse",
            "sector_shocks": {
                "BHP.AX": -0.35, "RIO.AX": -0.30,
                "FMG.AX": -0.55, "MIN.AX": -0.45,
            },
            "market_shock": -0.08,
        },
    }

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: dict[str, float],
        sector_map: Optional[dict[str, str]] = None,
    ) -> None:
        self.returns = returns.dropna().copy()
        self.weights = self._normalise(weights)
        self.sector_map = sector_map or {}
        self.tickers = list(self.weights.keys())

        w_arr = np.array([self.weights.get(t, 0) for t in self.returns.columns])
        self.port_returns = pd.Series(
            self.returns.values @ w_arr,
            index=self.returns.index,
            name="Portfolio",
        )

    @staticmethod
    def _normalise(w: dict) -> dict:
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    # ──────────────────────────────────────
    #  Historical stress tests
    # ──────────────────────────────────────

    def historical_stress_test(self) -> pd.DataFrame:
        """
        Apply historical crisis periods to the current portfolio.
        Uses actual return data from each crisis episode.

        Returns
        -------
        pd.DataFrame  One row per scenario with portfolio impact.
        """
        results = []

        for name, scenario in self.HISTORICAL_SCENARIOS.items():
            start = scenario["start"]
            end = scenario["end"]

            mask = (
                (self.port_returns.index >= start) &
                (self.port_returns.index <= end)
            )
            period_returns = self.port_returns[mask]

            if len(period_returns) < 5:
                continue

            cumret = float((1 + period_returns).prod() - 1)
            max_dd = float(
                ((1 + period_returns).cumprod() /
                 (1 + period_returns).cumprod().cummax() - 1).min()
            )
            vol = float(period_returns.std() * np.sqrt(252))
            n_days = len(period_returns)

            results.append({
                "Scenario": name,
                "Period": f"{start} → {end}",
                "Portfolio Return": cumret,
                "Max Drawdown": max_dd,
                "Volatility (ann.)": vol,
                "Trading Days": n_days,
                "Description": scenario["description"],
            })

        return pd.DataFrame(results).set_index("Scenario")

    # ──────────────────────────────────────
    #  Hypothetical stress tests
    # ──────────────────────────────────────

    def hypothetical_stress_test(self) -> pd.DataFrame:
        """
        Apply hypothetical macro shock scenarios to the current portfolio.
        Uses predefined sector and asset shocks.

        Returns
        -------
        pd.DataFrame  One row per scenario.
        """
        results = []

        for name, scenario in self.HYPOTHETICAL_SCENARIOS.items():
            market_shock = scenario.get("market_shock", 0.0)
            sector_shocks = scenario.get("sector_shocks", {})

            port_impact = 0.0
            asset_impacts = {}

            for ticker in self.tickers:
                weight = self.weights.get(ticker, 0)
                if weight == 0:
                    continue

                # Start with market shock
                shock = market_shock

                # Apply sector shock if applicable
                sector = self.sector_map.get(ticker, "")
                if sector in sector_shocks:
                    shock = sector_shocks[sector]
                elif ticker in sector_shocks:
                    shock = sector_shocks[ticker]

                asset_impact = weight * shock
                asset_impacts[ticker] = asset_impact
                port_impact += asset_impact

            results.append({
                "Scenario": name,
                "Portfolio Impact": port_impact,
                "Worst Asset": min(asset_impacts, key=asset_impacts.get)
                if asset_impacts else "N/A",
                "Largest Loss": min(asset_impacts.values())
                if asset_impacts else 0,
                "Description": scenario["description"],
            })

        return (
            pd.DataFrame(results)
            .set_index("Scenario")
            .sort_values("Portfolio Impact")
        )

    # ──────────────────────────────────────
    #  Tail risk analysis
    # ──────────────────────────────────────

    def tail_risk_analysis(
        self,
        n_simulations: int = 50_000,
        horizon: int = 252,
    ) -> pd.DataFrame:
        """
        Monte Carlo tail risk analysis for the portfolio.
        Simulates full distribution of 1-year portfolio returns.

        Returns
        -------
        pd.DataFrame  Distribution statistics.
        """
        mc = MonteCarloSimulator(
            self.port_returns, n_simulations=n_simulations, horizon=horizon
        )
        result = mc.simulate_gbm()
        final_returns = result.final_prices / result.final_prices.mean() - 1

        rows = [
            ("Expected Return", float(final_returns.mean())),
            ("Median Return", float(np.median(final_returns))),
            ("Volatility (ann.)", float(self.port_returns.std() * np.sqrt(252))),
            ("VaR 95% (1Y)", float(-np.percentile(final_returns, 5))),
            ("VaR 99% (1Y)", float(-np.percentile(final_returns, 1))),
            ("CVaR 95% (1Y)", float(-final_returns[final_returns <= np.percentile(final_returns, 5)].mean())),
            ("CVaR 99% (1Y)", float(-final_returns[final_returns <= np.percentile(final_returns, 1)].mean())),
            ("Prob. Loss > 10%", float((final_returns < -0.10).mean())),
            ("Prob. Loss > 20%", float((final_returns < -0.20).mean())),
            ("Prob. Loss > 30%", float((final_returns < -0.30).mean())),
            ("Best 1% outcome", float(np.percentile(final_returns, 99))),
            ("Worst 1% outcome", float(np.percentile(final_returns, 1))),
        ]

        return pd.DataFrame(rows, columns=["Metric", "Value"])

    # ──────────────────────────────────────
    #  VaR backtesting (Kupiec test)
    # ──────────────────────────────────────

    def var_backtest(
        self,
        confidence: float = 0.95,
        window: int = 252,
    ) -> pd.DataFrame:
        """
        Kupiec (1995) Proportion of Failures (POF) test for VaR.
        Validates whether the VaR model is correctly calibrated.

        H₀: The VaR model is correctly specified.
        Reject H₀ if observed exceedance rate ≠ expected (1 - confidence).

        Parameters
        ----------
        confidence : float   VaR confidence level.
        window     : int     Rolling window for VaR estimation.

        Returns
        -------
        pd.DataFrame  Daily VaR, exceedances, and test statistics.
        """
        var_series = self.port_returns.rolling(window).quantile(
            1 - confidence
        ) * -1

        aligned = pd.DataFrame({
            "Return": self.port_returns,
            f"VaR_{int(confidence*100)}": var_series,
        }).dropna()

        aligned["Exceedance"] = aligned["Return"] < -aligned[f"VaR_{int(confidence*100)}"]

        n = len(aligned)
        n_exc = int(aligned["Exceedance"].sum())
        exc_rate = n_exc / n
        expected_rate = 1 - confidence

        # Kupiec LR statistic
        if exc_rate > 0 and exc_rate < 1:
            lr = -2 * (
                n_exc * np.log(expected_rate / exc_rate)
                + (n - n_exc) * np.log((1 - expected_rate) / (1 - exc_rate))
            )
        else:
            lr = np.nan

        p_val = float(1 - stats.chi2.cdf(lr, df=1)) if not np.isnan(lr) else np.nan

        summary = pd.DataFrame([
            ("Observations", n),
            ("Exceedances", n_exc),
            ("Observed Rate", f"{exc_rate:.2%}"),
            ("Expected Rate", f"{expected_rate:.2%}"),
            ("Kupiec LR Stat", f"{lr:.4f}" if not np.isnan(lr) else "N/A"),
            ("P-Value", f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"),
            ("VaR Model Valid", "✅ Yes" if (p_val or 0) > 0.05 else "❌ No"),
        ], columns=["Metric", "Value"])

        return summary
