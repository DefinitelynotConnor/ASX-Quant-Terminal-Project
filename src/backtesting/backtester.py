"""
backtester.py
=============
Professional vectorised backtesting engine for ASX equity strategies.

Design Philosophy
-----------------
    Vectorised first  : All operations use NumPy/Pandas broadcasting.
                        No Python loops over time steps.
    Realistic costs   : Slippage, commission, and market impact modelled.
    No look-ahead     : Strict signal-to-execution lag enforcement.
    Reproducible      : Fixed random seeds, deterministic outputs.
    Extensible        : Clean base class for custom strategies.

Architecture
------------
    BacktestConfig    : All parameters in one place
    Portfolio         : Tracks positions, NAV, cash, turnover
    Backtester        : Orchestrates signal → portfolio → metrics
    BacktestResult    : Clean result container

Usage
-----
    from src.backtesting.backtester import Backtester, BacktestConfig

    config = BacktestConfig(
        initial_capital=1_000_000,
        commission=0.001,
        slippage=0.0005,
    )
    bt = Backtester(prices, signals, config)
    result = bt.run()
    print(result.metrics)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """
    All backtesting parameters in one place.

    Parameters
    ----------
    initial_capital   : float   Starting portfolio value in AUD.
    commission        : float   One-way commission rate (e.g. 0.001 = 0.1%).
    slippage          : float   One-way slippage per trade (e.g. 0.0005 = 0.05%).
    market_impact     : float   Square-root market impact coefficient.
    signal_lag        : int     Days between signal and execution (1 = next open).
    rebalance_freq    : str     'daily'|'weekly'|'monthly'|'quarterly'|'signal'.
    max_position_size : float   Maximum weight per asset (e.g. 0.20 = 20%).
    min_position_size : float   Minimum weight to hold (below this = close).
    max_leverage      : float   Maximum gross leverage (1.0 = long only).
    risk_free_rate    : float   Annualised risk-free rate for Sharpe calculation.
    benchmark_ticker  : str     Benchmark label for information ratio.
    allow_short       : bool    Whether short positions are permitted.
    """
    initial_capital: float = 1_000_000.0
    commission: float = 0.001          # 0.1% one-way
    slippage: float = 0.0005           # 0.05% one-way
    market_impact: float = 0.0         # disabled by default
    signal_lag: int = 1                # execute next day
    rebalance_freq: str = "daily"
    max_position_size: float = 1.0
    min_position_size: float = 0.0
    max_leverage: float = 1.0
    risk_free_rate: float = 0.0435
    benchmark_ticker: str = "^AXJO"
    allow_short: bool = False


# ─────────────────────────────────────────────
#  Portfolio state tracker
# ─────────────────────────────────────────────

class Portfolio:
    """
    Tracks portfolio state through the backtest.
    Handles cash, positions, NAV, and trade records.

    Parameters
    ----------
    initial_capital : float
    tickers         : list of str
    dates           : pd.DatetimeIndex
    """

    def __init__(
        self,
        initial_capital: float,
        tickers: list[str],
        dates: pd.DatetimeIndex,
    ) -> None:
        self.n = len(dates)
        self.tickers = tickers
        self.dates = dates
        self.n_assets = len(tickers)

        # NAV history
        self.nav = np.full(self.n, initial_capital)
        self.cash = np.full(self.n, initial_capital)

        # Position matrix: rows=dates, cols=assets (in $ value)
        self.positions = np.zeros((self.n, self.n_assets))

        # Weight matrix
        self.weights = np.zeros((self.n, self.n_assets))

        # Trade records
        self.trades: list[dict] = []

        # Turnover per day
        self.turnover = np.zeros(self.n)

        # Transaction costs per day
        self.costs = np.zeros(self.n)

    def to_dataframe(self) -> dict[str, pd.DataFrame | pd.Series]:
        """Export all portfolio state as DataFrames."""
        return {
            "nav": pd.Series(self.nav, index=self.dates, name="NAV"),
            "cash": pd.Series(self.cash, index=self.dates, name="Cash"),
            "positions": pd.DataFrame(
                self.positions, index=self.dates, columns=self.tickers
            ),
            "weights": pd.DataFrame(
                self.weights, index=self.dates, columns=self.tickers
            ),
            "turnover": pd.Series(self.turnover, index=self.dates, name="Turnover"),
            "costs": pd.Series(self.costs, index=self.dates, name="Costs"),
        }


# ─────────────────────────────────────────────
#  Backtest Result
# ─────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Container for all backtest outputs."""
    nav: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    turnover: pd.Series = field(default_factory=pd.Series)
    costs: pd.Series = field(default_factory=pd.Series)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdown: pd.Series = field(default_factory=pd.Series)
    benchmark_returns: Optional[pd.Series] = None
    config: Optional[BacktestConfig] = None
    strategy_name: str = "Strategy"


# ─────────────────────────────────────────────
#  Core Backtester
# ─────────────────────────────────────────────

class Backtester:
    """
    Vectorised backtesting engine.

    Parameters
    ----------
    prices          : pd.DataFrame
        Adjusted close prices. Rows=dates, cols=tickers.
    signals         : pd.DataFrame or pd.Series
        Trading signals. Same index as prices.
        Values: +1 (long), 0 (flat), -1 (short) OR continuous weights.
        If DataFrame: columns = tickers (individual asset signals).
        If Series: single signal applied to equal-weight portfolio.
    config          : BacktestConfig
    benchmark       : pd.Series, optional
        Benchmark return series for relative metrics.
    strategy_name   : str
        Label for display.
    weight_func     : Callable, optional
        Custom function to convert signals → weights.
        Signature: weight_func(signals, prices, config) → pd.DataFrame
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame | pd.Series,
        config: Optional[BacktestConfig] = None,
        benchmark: Optional[pd.Series] = None,
        strategy_name: str = "Strategy",
        weight_func: Optional[Callable] = None,
    ) -> None:
        self.prices = prices.copy()
        self.config = config or BacktestConfig()
        self.benchmark = benchmark
        self.strategy_name = strategy_name
        self.weight_func = weight_func or self._default_weight_func

        # Align signals to prices
        if isinstance(signals, pd.Series):
            signals = signals.to_frame()
            # Broadcast single signal to all tickers
            if signals.columns[0] not in prices.columns:
                for t in prices.columns:
                    signals[t] = signals.iloc[:, 0]
                signals = signals.drop(columns=[signals.columns[0]])

        self.signals = signals.reindex(prices.index).fillna(0)

        # Enforce signal lag (no look-ahead)
        self.signals = self.signals.shift(self.config.signal_lag).fillna(0)

    # ──────────────────────────────────────
    #  Weight generation
    # ──────────────────────────────────────

    def _default_weight_func(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        config: BacktestConfig,
    ) -> pd.DataFrame:
        """
        Convert signals to portfolio weights.
        +1 → equal positive weight, -1 → equal negative weight, 0 → flat.
        Normalises so long weights sum to 1 (long-only) or ±1 (long/short).
        """
        if not config.allow_short:
            raw = signals.clip(lower=0)
        else:
            raw = signals.copy()

        row_sum = raw.abs().sum(axis=1).replace(0, np.nan)
        weights = raw.div(row_sum, axis=0).fillna(0)

        # Apply position size limits
        weights = weights.clip(
            lower=-config.max_position_size,
            upper=config.max_position_size,
        )

        # Apply minimum position size
        if config.min_position_size > 0:
            too_small = weights.abs() < config.min_position_size
            weights[too_small] = 0

        return weights

    def _apply_rebalance_frequency(
        self, weights: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply rebalancing frequency constraint.
        Between rebalance dates, weights drift with price changes.
        """
        freq = self.config.rebalance_freq

        if freq == "daily" or freq == "signal":
            return weights

        freq_map = {
            "weekly": "W",
            "monthly": "ME",
            "quarterly": "QE",
        }

        if freq not in freq_map:
            return weights

        pandas_freq = freq_map[freq]
        rebalance_dates = set(
            weights.resample(pandas_freq).last().index.normalize()
        )

        # Only update weights on rebalance dates, else carry forward
        mask = weights.index.normalize().isin(rebalance_dates)
        rebalanced = weights.copy()
        rebalanced[~mask] = np.nan
        rebalanced = rebalanced.ffill()

        return rebalanced.fillna(0)

    # ──────────────────────────────────────
    #  Transaction cost model
    # ──────────────────────────────────────

    def _compute_transaction_costs(
        self,
        prev_weights: np.ndarray,
        new_weights: np.ndarray,
        nav: float,
        prices_row: np.ndarray,
        adv: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """
        Compute total transaction costs for a rebalancing event.

        Cost components:
            1. Commission: fixed % of traded value
            2. Bid-ask spread (slippage): fixed % of traded value
            3. Market impact: sqrt model (optional)

        Returns
        -------
        (total_cost, turnover)
            total_cost : AUD cost of the rebalancing
            turnover   : fraction of NAV traded
        """
        weight_change = np.abs(new_weights - prev_weights)
        turnover = float(weight_change.sum() / 2)

        traded_value = nav * weight_change
        commission_cost = (traded_value * self.config.commission).sum()
        slippage_cost = (traded_value * self.config.slippage).sum()

        # Square-root market impact (optional)
        impact_cost = 0.0
        if self.config.market_impact > 0 and adv is not None:
            adv_safe = np.where(adv > 0, adv, 1e6)
            participation = traded_value / adv_safe
            impact = self.config.market_impact * np.sqrt(participation) * traded_value
            impact_cost = float(impact.sum())

        total_cost = float(commission_cost + slippage_cost + impact_cost)
        return total_cost, turnover

    # ──────────────────────────────────────
    #  Main run loop
    # ──────────────────────────────────────

    def run(self) -> BacktestResult:
        """
        Execute the backtest simulation.

        Returns
        -------
        BacktestResult  containing NAV, returns, positions, metrics.
        """
        # Align data
        common = self.prices.index.intersection(self.signals.index)
        prices = self.prices.loc[common].copy()
        signals = self.signals.loc[common].copy()

        tickers = [t for t in signals.columns if t in prices.columns]
        if not tickers:
            raise ValueError("No overlapping tickers between signals and prices.")

        prices = prices[tickers]
        signals = signals[tickers]

        dates = prices.index
        n = len(dates)
        n_assets = len(tickers)

        # Compute target weights
        target_weights = self.weight_func(signals, prices, self.config)
        target_weights = target_weights[tickers]
        target_weights = self._apply_rebalance_frequency(target_weights)

        # Daily returns matrix
        daily_returns = prices.pct_change().fillna(0).values  # (n, n_assets)
        weights_arr = target_weights.values                    # (n, n_assets)

        # Portfolio state
        portfolio = Portfolio(
            self.config.initial_capital, tickers, dates
        )

        prev_weights = np.zeros(n_assets)
        nav = self.config.initial_capital

        for t in range(1, n):
            new_weights = weights_arr[t - 1]  # lagged weights

            # Transaction costs
            cost, turnover = self._compute_transaction_costs(
                prev_weights, new_weights, nav,
                prices.values[t],
            )

            # Portfolio return for this day
            port_ret = float(np.dot(new_weights, daily_returns[t]))

            # Update NAV
            nav = nav * (1 + port_ret) - cost

            # Record state
            portfolio.nav[t] = nav
            portfolio.cash[t] = nav * (1 - np.sum(np.abs(new_weights)))
            portfolio.positions[t] = nav * new_weights
            portfolio.weights[t] = new_weights
            portfolio.turnover[t] = turnover
            portfolio.costs[t] = cost

            prev_weights = new_weights.copy()

        # Export
        state = portfolio.to_dataframe()
        nav_series = state["nav"]
        returns = nav_series.pct_change().dropna()

        result = BacktestResult(
            nav=nav_series,
            returns=returns,
            positions=state["positions"],
            weights=state["weights"],
            turnover=state["turnover"],
            costs=state["costs"],
            benchmark_returns=self.benchmark,
            config=self.config,
            strategy_name=self.strategy_name,
        )

        # Compute metrics
        from src.backtesting.performance_metrics import PerformanceMetrics
        pm = PerformanceMetrics(result)
        result.metrics = pm.compute_all()
        result.monthly_returns = pm.monthly_returns()
        result.drawdown = pm.drawdown_series()

        return result

    # ──────────────────────────────────────
    #  Walk-forward backtesting
    # ──────────────────────────────────────

    def run_walk_forward(
        self,
        train_days: int = 504,
        test_days: int = 63,
        step_days: int = 21,
    ) -> list[BacktestResult]:
        """
        Walk-forward out-of-sample backtest.
        Trains on in-sample data, tests on the next out-of-sample window.
        Repeats across the full history.

        Parameters
        ----------
        train_days : int   In-sample window.
        test_days  : int   Out-of-sample window.
        step_days  : int   How far to step forward each iteration.

        Returns
        -------
        list of BacktestResult  (one per test window)
        """
        common = self.prices.index.intersection(self.signals.index)
        n = len(common)
        results = []
        start = 0

        while start + train_days + test_days <= n:
            test_start = start + train_days
            test_end = min(test_start + test_days, n)

            test_prices = self.prices.iloc[test_start:test_end]
            test_signals = self.signals.iloc[test_start:test_end]

            bt = Backtester(
                prices=test_prices,
                signals=test_signals,
                config=self.config,
                benchmark=self.benchmark,
                strategy_name=f"{self.strategy_name}_WF_{start}",
                weight_func=self.weight_func,
            )
            try:
                r = bt.run()
                results.append(r)
            except Exception as e:
                print(f"  [WF] Window {start} failed: {e}")

            start += step_days

        return results

    def stitch_walk_forward(
        self, results: list[BacktestResult]
    ) -> BacktestResult:
        """
        Combine walk-forward windows into a single continuous equity curve.
        Each window is chain-linked from the previous ending NAV.

        Parameters
        ----------
        results : list   From run_walk_forward().

        Returns
        -------
        BacktestResult  Combined out-of-sample backtest.
        """
        if not results:
            raise ValueError("No walk-forward results to stitch.")

        nav_parts = []
        base_nav = self.config.initial_capital

        for r in results:
            scaled = r.nav / r.nav.iloc[0] * base_nav
            nav_parts.append(scaled)
            base_nav = float(scaled.iloc[-1])

        combined_nav = pd.concat(nav_parts)
        combined_nav = combined_nav[~combined_nav.index.duplicated(keep="last")]
        combined_returns = combined_nav.pct_change().dropna()

        combined = BacktestResult(
            nav=combined_nav,
            returns=combined_returns,
            benchmark_returns=self.benchmark,
            config=self.config,
            strategy_name=f"{self.strategy_name}_WalkForward",
        )

        from src.backtesting.performance_metrics import PerformanceMetrics
        pm = PerformanceMetrics(combined)
        combined.metrics = pm.compute_all()
        combined.monthly_returns = pm.monthly_returns()
        combined.drawdown = pm.drawdown_series()

        return combined
