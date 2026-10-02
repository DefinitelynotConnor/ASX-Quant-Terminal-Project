"""
parameter_optimisation.py
=========================
Strategy parameter tuning framework for the ASX Quant Terminal.

Methods Implemented
-------------------
    GridSearch      : Exhaustive search over a parameter grid
    RandomSearch    : Random sampling of parameter space
    BayesianOpt     : Gaussian Process surrogate model optimisation
    WalkForwardOpt  : Out-of-sample validated optimisation

All optimisers use the Backtester engine and return results ranked
by a configurable objective (Sharpe ratio by default).

Design Principles
-----------------
    Walk-forward validation  : Prevent in-sample overfitting
    Multiple objectives      : Sharpe, Sortino, Calmar, CAGR
    Parallel evaluation      : Vectorised where possible
    Reproducibility          : Fixed seeds, deterministic outputs

Usage
-----
    from src.optimisation.parameter_optimisation import GridSearch

    param_grid = {
        "lookback": [126, 252, 504],
        "n_long": [5, 10, 15],
        "skip": [0, 21],
    }

    gs = GridSearch(
        strategy_class=CrossSectionalMomentum,
        prices=universe_prices,
        param_grid=param_grid,
        objective="sharpe",
    )
    results = gs.run()
    print(results.best_params)
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from scipy.optimize import minimize
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    HAS_GP = True
except ImportError:
    HAS_GP = False


# ─────────────────────────────────────────────
#  Result Containers
# ─────────────────────────────────────────────

@dataclass
class OptimisationResult:
    """Container for parameter optimisation results."""
    best_params: dict = field(default_factory=dict)
    best_score: float = -np.inf
    objective: str = "sharpe"
    all_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_evaluations: int = 0
    method: str = ""
    convergence_curve: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\nOptimisation Results ({self.method})",
            f"{'─' * 45}",
            f"  Evaluations  : {self.n_evaluations}",
            f"  Objective    : {self.objective}",
            f"  Best Score   : {self.best_score:.4f}",
            f"\n  Best Parameters:",
        ]
        for k, v in self.best_params.items():
            lines.append(f"    {k:<25} {v}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Objective Evaluator
# ─────────────────────────────────────────────

class StrategyEvaluator:
    """
    Evaluates a strategy configuration and returns a scalar score.

    Parameters
    ----------
    strategy_class : type    Strategy class to instantiate.
    prices         : pd.DataFrame   Universe prices.
    benchmark      : pd.Series      Benchmark returns.
    objective      : str            Metric to optimise.
    config         : dict           Fixed BacktestConfig parameters.
    """

    VALID_OBJECTIVES = [
        "sharpe", "sortino", "calmar", "cagr",
        "information_ratio", "max_drawdown_neg",
        "profit_factor", "win_rate",
    ]

    def __init__(
        self,
        strategy_class,
        prices: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
        objective: str = "sharpe",
        config: Optional[dict] = None,
        min_trading_days: int = 252,
    ) -> None:
        self.strategy_class = strategy_class
        self.prices = prices
        self.benchmark = benchmark
        self.objective = objective.lower()
        self.config = config or {}
        self.min_days = min_trading_days

        if self.objective not in self.VALID_OBJECTIVES:
            raise ValueError(
                f"Unknown objective '{objective}'. "
                f"Valid: {self.VALID_OBJECTIVES}"
            )

    def evaluate(self, params: dict) -> float:
        """
        Instantiate strategy with params, run backtest, return objective score.

        Returns
        -------
        float   Objective value. Returns -inf on failure.
        """
        try:
            strategy = self.strategy_class(
                universe_prices=self.prices, **params
            )
            result = strategy.backtest(
                benchmark=self.benchmark,
                **self.config,
            )

            if result.returns.empty or len(result.returns) < self.min_days:
                return -np.inf

            return self._extract_objective(result)

        except Exception as e:
            return -np.inf

    def _extract_objective(self, result) -> float:
        """Extract the objective metric from a BacktestResult."""
        from src.backtesting.performance_metrics import PerformanceMetrics

        pm = PerformanceMetrics(result)

        obj_map = {
            "sharpe": pm.sharpe_ratio,
            "sortino": pm.sortino_ratio,
            "calmar": pm.calmar_ratio,
            "cagr": pm.cagr,
            "information_ratio": pm.information_ratio,
            "max_drawdown_neg": lambda: -abs(pm.max_drawdown()),
            "profit_factor": pm.profit_factor,
            "win_rate": pm.win_rate,
        }

        try:
            score = obj_map[self.objective]()
            return float(score) if not np.isnan(score) else -np.inf
        except Exception:
            return -np.inf


# ─────────────────────────────────────────────
#  Grid Search
# ─────────────────────────────────────────────

class GridSearch:
    """
    Exhaustive grid search over all combinations of parameter values.

    Best for small parameter spaces (<500 combinations).
    Guaranteed to find the global optimum within the grid.

    Parameters
    ----------
    strategy_class : type        Strategy class to optimise.
    prices         : pd.DataFrame
    param_grid     : dict        {param_name: [value1, value2, ...]}
    objective      : str         Metric to maximise.
    benchmark      : pd.Series   Optional benchmark for IR calculation.
    verbose        : bool        Print progress.
    """

    def __init__(
        self,
        strategy_class,
        prices: pd.DataFrame,
        param_grid: dict[str, list],
        objective: str = "sharpe",
        benchmark: Optional[pd.Series] = None,
        config: Optional[dict] = None,
        verbose: bool = True,
    ) -> None:
        self.strategy_class = strategy_class
        self.prices = prices
        self.param_grid = param_grid
        self.objective = objective
        self.benchmark = benchmark
        self.config = config or {}
        self.verbose = verbose
        self.evaluator = StrategyEvaluator(
            strategy_class, prices, benchmark, objective, config
        )

    def _generate_combinations(self) -> list[dict]:
        """Generate all parameter combinations from the grid."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = []
        for combo in itertools.product(*values):
            combos.append(dict(zip(keys, combo)))
        return combos

    def run(self) -> OptimisationResult:
        """
        Run exhaustive grid search.

        Returns
        -------
        OptimisationResult sorted by objective score.
        """
        combos = self._generate_combinations()
        n = len(combos)

        if self.verbose:
            print(f"GridSearch: {n} parameter combinations × {self.objective}")

        records = []
        best_score = -np.inf
        best_params = {}

        for i, params in enumerate(combos):
            score = self.evaluator.evaluate(params)

            row = {**params, "score": score}
            records.append(row)

            if score > best_score:
                best_score = score
                best_params = params.copy()

            if self.verbose and (i + 1) % max(1, n // 10) == 0:
                print(f"  [{i+1}/{n}] Best {self.objective}: {best_score:.4f}")

        df = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)

        if self.verbose:
            print(f"\n  ✅ Done. Best {self.objective}: {best_score:.4f}")
            print(f"  Best params: {best_params}")

        return OptimisationResult(
            best_params=best_params,
            best_score=best_score,
            objective=self.objective,
            all_results=df,
            n_evaluations=n,
            method="Grid Search",
        )


# ─────────────────────────────────────────────
#  Random Search
# ─────────────────────────────────────────────

class RandomSearch:
    """
    Random search over parameter space.

    More efficient than grid search for large parameter spaces.
    Random sampling covers the space better than grid search
    when not all parameters are equally important.

    Parameters
    ----------
    strategy_class  : type
    prices          : pd.DataFrame
    param_distributions : dict
        {param_name: distribution}
        distribution can be:
            list         → uniform random choice
            tuple(lo, hi)→ uniform random float
            tuple(lo,hi,'int') → uniform random int
    n_iter          : int    Number of random samples.
    objective       : str
    """

    def __init__(
        self,
        strategy_class,
        prices: pd.DataFrame,
        param_distributions: dict,
        n_iter: int = 50,
        objective: str = "sharpe",
        benchmark: Optional[pd.Series] = None,
        config: Optional[dict] = None,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.strategy_class = strategy_class
        self.prices = prices
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.objective = objective
        self.benchmark = benchmark
        self.config = config or {}
        self.seed = seed
        self.verbose = verbose
        self.evaluator = StrategyEvaluator(
            strategy_class, prices, benchmark, objective, config
        )

    def _sample_params(self, rng: np.random.Generator) -> dict:
        """Sample one set of parameters from the distributions."""
        params = {}
        for name, dist in self.param_distributions.items():
            if isinstance(dist, list):
                params[name] = rng.choice(dist)
            elif isinstance(dist, tuple):
                lo, hi = dist[0], dist[1]
                as_int = len(dist) > 2 and dist[2] == "int"
                val = rng.uniform(lo, hi)
                params[name] = int(round(val)) if as_int else float(val)
            else:
                params[name] = dist
        return params

    def run(self) -> OptimisationResult:
        """Run random search."""
        rng = np.random.default_rng(self.seed)

        if self.verbose:
            print(f"RandomSearch: {self.n_iter} samples × {self.objective}")

        records = []
        best_score = -np.inf
        best_params = {}
        convergence = []

        for i in range(self.n_iter):
            params = self._sample_params(rng)
            score = self.evaluator.evaluate(params)

            row = {**params, "score": score}
            records.append(row)

            if score > best_score:
                best_score = score
                best_params = params.copy()

            convergence.append(best_score)

            if self.verbose and (i + 1) % max(1, self.n_iter // 5) == 0:
                print(f"  [{i+1}/{self.n_iter}] Best {self.objective}: {best_score:.4f}")

        df = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)

        if self.verbose:
            print(f"\n  ✅ Done. Best {self.objective}: {best_score:.4f}")

        return OptimisationResult(
            best_params=best_params,
            best_score=best_score,
            objective=self.objective,
            all_results=df,
            n_evaluations=self.n_iter,
            method="Random Search",
            convergence_curve=convergence,
        )


# ─────────────────────────────────────────────
#  Bayesian Optimisation
# ─────────────────────────────────────────────

class BayesianOptimisation:
    """
    Gaussian Process Bayesian Optimisation.

    Uses a surrogate model (GP) to approximate the objective function
    and an acquisition function (Expected Improvement) to decide where
    to evaluate next. Far more sample-efficient than grid/random search.

    Algorithm
    ---------
    1. Evaluate n_initial random points
    2. Fit GP to observed (params, score) pairs
    3. Maximise acquisition function to find next candidate
    4. Evaluate candidate, update GP, repeat

    Parameters
    ----------
    strategy_class    : type
    prices            : pd.DataFrame
    param_bounds      : dict
        {param_name: (lo, hi)}  — continuous bounds
        {param_name: [v1, v2, v3]}  — discrete choices
    n_iter            : int    Total evaluations.
    n_initial         : int    Random initial points before GP kicks in.
    objective         : str
    """

    def __init__(
        self,
        strategy_class,
        prices: pd.DataFrame,
        param_bounds: dict,
        n_iter: int = 30,
        n_initial: int = 10,
        objective: str = "sharpe",
        benchmark: Optional[pd.Series] = None,
        config: Optional[dict] = None,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.strategy_class = strategy_class
        self.prices = prices
        self.param_bounds = param_bounds
        self.n_iter = n_iter
        self.n_initial = min(n_initial, n_iter)
        self.objective = objective
        self.benchmark = benchmark
        self.config = config or {}
        self.seed = seed
        self.verbose = verbose
        self.evaluator = StrategyEvaluator(
            strategy_class, prices, benchmark, objective, config
        )

        self._param_names = list(param_bounds.keys())
        self._discrete = {
            k: isinstance(v, list)
            for k, v in param_bounds.items()
        }

    def _decode_params(self, x: np.ndarray) -> dict:
        """Convert continuous vector back to strategy parameters."""
        params = {}
        for i, name in enumerate(self._param_names):
            val = x[i]
            if self._discrete[name]:
                choices = self.param_bounds[name]
                idx = int(round(val * (len(choices) - 1)))
                idx = max(0, min(idx, len(choices) - 1))
                params[name] = choices[idx]
            else:
                lo, hi = self.param_bounds[name]
                params[name] = float(lo + val * (hi - lo))
        return params

    def _encode_params(self, params: dict) -> np.ndarray:
        """Convert parameter dict to normalised [0, 1] vector."""
        x = np.zeros(len(self._param_names))
        for i, name in enumerate(self._param_names):
            val = params[name]
            if self._discrete[name]:
                choices = self.param_bounds[name]
                idx = choices.index(val) if val in choices else 0
                x[i] = idx / max(len(choices) - 1, 1)
            else:
                lo, hi = self.param_bounds[name]
                x[i] = (val - lo) / (hi - lo) if hi > lo else 0
        return x

    def _expected_improvement(
        self,
        x: np.ndarray,
        gp,
        y_best: float,
        xi: float = 0.01,
    ) -> float:
        """
        Expected Improvement acquisition function.
        EI(x) = E[max(f(x) - f_best, 0)]
        """
        if not HAS_GP:
            return 0.0

        x_pred = x.reshape(1, -1)
        mu, sigma = gp.predict(x_pred, return_std=True)
        mu, sigma = float(mu[0]), float(sigma[0])

        if sigma < 1e-10:
            return 0.0

        z = (mu - y_best - xi) / sigma
        ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return float(max(ei, 0))

    def run(self) -> OptimisationResult:
        """Run Bayesian optimisation."""
        if not HAS_GP:
            print("  [BayesOpt] sklearn GP not available. Falling back to RandomSearch.")
            rs = RandomSearch(
                self.strategy_class, self.prices,
                {k: (v if isinstance(v, list) else (v[0], v[1], "float"))
                 for k, v in self.param_bounds.items()},
                n_iter=self.n_iter,
                objective=self.objective,
                benchmark=self.benchmark,
                config=self.config,
                seed=self.seed,
                verbose=self.verbose,
            )
            return rs.run()

        rng = np.random.default_rng(self.seed)
        n_dims = len(self._param_names)

        X_obs = []   # Evaluated points (normalised)
        y_obs = []   # Corresponding scores
        records = []
        best_score = -np.inf
        best_params = {}
        convergence = []

        if self.verbose:
            print(f"BayesianOpt: {self.n_iter} evaluations × {self.objective}")

        # Phase 1: random initial exploration
        for i in range(self.n_initial):
            x = rng.uniform(0, 1, n_dims)
            params = self._decode_params(x)
            score = self.evaluator.evaluate(params)

            X_obs.append(x)
            y_obs.append(score if score > -np.inf else -10)
            records.append({**params, "score": score, "phase": "initial"})

            if score > best_score:
                best_score = score
                best_params = params.copy()

            convergence.append(best_score)

            if self.verbose:
                print(f"  [Init {i+1}/{self.n_initial}] score={score:.4f}")

        # Phase 2: GP-guided exploration
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            n_restarts_optimizer=5,
            random_state=self.seed,
        )

        for i in range(self.n_iter - self.n_initial):
            X_arr = np.array(X_obs)
            y_arr = np.array(y_obs)

            try:
                gp.fit(X_arr, y_arr)
            except Exception:
                x_next = rng.uniform(0, 1, n_dims)
            else:
                # Optimise acquisition function
                best_ei = -1
                x_next = rng.uniform(0, 1, n_dims)

                # Multi-start optimisation of EI
                for _ in range(20):
                    x0 = rng.uniform(0, 1, n_dims)
                    result = minimize(
                        lambda x: -self._expected_improvement(x, gp, best_score),
                        x0,
                        method="L-BFGS-B",
                        bounds=[(0, 1)] * n_dims,
                    )
                    if -result.fun > best_ei:
                        best_ei = -result.fun
                        x_next = result.x

            x_next = np.clip(x_next, 0, 1)
            params = self._decode_params(x_next)
            score = self.evaluator.evaluate(params)

            X_obs.append(x_next)
            y_obs.append(score if score > -np.inf else -10)
            records.append({**params, "score": score, "phase": "bayes"})

            if score > best_score:
                best_score = score
                best_params = params.copy()

            convergence.append(best_score)

            if self.verbose:
                print(f"  [GP {i+1}/{self.n_iter - self.n_initial}] "
                      f"score={score:.4f} | best={best_score:.4f}")

        df = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)

        if self.verbose:
            print(f"\n  ✅ Done. Best {self.objective}: {best_score:.4f}")
            print(f"  Best params: {best_params}")

        return OptimisationResult(
            best_params=best_params,
            best_score=best_score,
            objective=self.objective,
            all_results=df,
            n_evaluations=self.n_iter,
            method="Bayesian Optimisation",
            convergence_curve=convergence,
        )


# ─────────────────────────────────────────────
#  Walk-Forward Optimisation
# ─────────────────────────────────────────────

class WalkForwardOptimisation:
    """
    Walk-forward parameter optimisation with out-of-sample validation.

    Prevents overfitting by:
    1. Training on in-sample window
    2. Selecting best parameters
    3. Trading OOS with those parameters
    4. Rolling forward and repeating

    This produces a realistic out-of-sample performance estimate
    that accounts for the fact that optimal parameters change over time.

    Parameters
    ----------
    strategy_class   : type
    prices           : pd.DataFrame
    param_grid       : dict      Parameters to search.
    train_days       : int       In-sample window.
    test_days        : int       Out-of-sample window.
    step_days        : int       Roll forward step.
    optimiser        : str       'grid' | 'random' | 'bayes'
    n_random         : int       Samples for random/bayes methods.
    """

    def __init__(
        self,
        strategy_class,
        prices: pd.DataFrame,
        param_grid: dict,
        train_days: int = 504,
        test_days: int = 63,
        step_days: int = 63,
        optimiser: str = "random",
        n_random: int = 30,
        objective: str = "sharpe",
        benchmark: Optional[pd.Series] = None,
        verbose: bool = True,
    ) -> None:
        self.strategy_class = strategy_class
        self.prices = prices
        self.param_grid = param_grid
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.optimiser = optimiser
        self.n_random = n_random
        self.objective = objective
        self.benchmark = benchmark
        self.verbose = verbose

    def run(self) -> dict:
        """
        Run walk-forward optimisation across the full history.

        Returns
        -------
        dict containing:
            oos_returns    : pd.Series   Concatenated OOS returns
            window_results : list        Best params per window
            final_nav      : pd.Series   Stitched equity curve
        """
        n = len(self.prices)
        window_results = []
        oos_returns_list = []
        start = 0

        while start + self.train_days + self.test_days <= n:
            train_prices = self.prices.iloc[start: start + self.train_days]
            test_prices = self.prices.iloc[
                start + self.train_days: start + self.train_days + self.test_days
            ]

            if self.verbose:
                train_start = train_prices.index[0].date()
                test_end = test_prices.index[-1].date()
                print(f"\n  Window: train={train_start}, test ends={test_end}")

            # Optimise on training window
            bm_train = None
            if self.benchmark is not None:
                bm_train = self.benchmark.reindex(train_prices.index)

            if self.optimiser == "grid":
                opt = GridSearch(
                    self.strategy_class, train_prices, self.param_grid,
                    self.objective, bm_train, verbose=False,
                )
            elif self.optimiser == "bayes":
                opt = BayesianOptimisation(
                    self.strategy_class, train_prices, self.param_grid,
                    n_iter=self.n_random, objective=self.objective,
                    benchmark=bm_train, verbose=False,
                )
            else:
                opt = RandomSearch(
                    self.strategy_class, train_prices, self.param_grid,
                    n_iter=self.n_random, objective=self.objective,
                    benchmark=bm_train, verbose=False,
                )

            try:
                opt_result = opt.run()
                best_params = opt_result.best_params
            except Exception as e:
                if self.verbose:
                    print(f"  Optimisation failed: {e}. Using defaults.")
                best_params = {k: v[0] if isinstance(v, list) else v
                               for k, v in self.param_grid.items()}

            # Trade OOS with best params
            bm_test = None
            if self.benchmark is not None:
                bm_test = self.benchmark.reindex(test_prices.index)

            try:
                strategy = self.strategy_class(
                    universe_prices=test_prices, **best_params
                )
                oos_result = strategy.backtest(benchmark=bm_test)
                oos_returns_list.append(oos_result.returns)

                window_results.append({
                    "train_start": train_prices.index[0],
                    "train_end": train_prices.index[-1],
                    "test_start": test_prices.index[0],
                    "test_end": test_prices.index[-1],
                    "best_params": best_params,
                    "is_score": opt_result.best_score,
                    "oos_sharpe": oos_result.metrics[
                        oos_result.metrics["Metric"] == "Sharpe Ratio"
                    ]["Value"].values[0] if not oos_result.metrics.empty else "N/A",
                })

                if self.verbose:
                    print(f"  Best params: {best_params}")
                    print(f"  IS {self.objective}: {opt_result.best_score:.4f}")

            except Exception as e:
                if self.verbose:
                    print(f"  OOS evaluation failed: {e}")

            start += self.step_days

        # Stitch OOS returns
        if oos_returns_list:
            oos_returns = pd.concat(oos_returns_list)
            oos_returns = oos_returns[~oos_returns.index.duplicated(keep="last")]
            nav = (1 + oos_returns).cumprod() * 100
        else:
            oos_returns = pd.Series(dtype=float)
            nav = pd.Series(dtype=float)

        return {
            "oos_returns": oos_returns,
            "window_results": pd.DataFrame(window_results),
            "final_nav": nav,
            "n_windows": len(window_results),
        }


# ─────────────────────────────────────────────
#  Sensitivity Analysis
# ─────────────────────────────────────────────

def parameter_sensitivity(
    strategy_class,
    prices: pd.DataFrame,
    base_params: dict,
    param_to_vary: str,
    values: list,
    objective: str = "sharpe",
    benchmark: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Analyse sensitivity of strategy performance to a single parameter.
    All other parameters are held at their base values.

    Parameters
    ----------
    base_params    : dict   Default parameter values.
    param_to_vary  : str    Parameter to sweep.
    values         : list   Values to test.

    Returns
    -------
    pd.DataFrame  param value vs objective score.
    """
    evaluator = StrategyEvaluator(
        strategy_class, prices, benchmark, objective
    )

    records = []
    for val in values:
        params = {**base_params, param_to_vary: val}
        score = evaluator.evaluate(params)
        records.append({param_to_vary: val, objective: score})

    return pd.DataFrame(records).set_index(param_to_vary)
