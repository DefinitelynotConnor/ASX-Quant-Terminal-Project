"""
src/backtesting
===============
Professional vectorised backtesting engine for the ASX Quant Terminal.

Modules
-------
backtester            : Core vectorised simulation engine
transaction_cost_model: Commission, slippage, and market impact modelling
performance_metrics   : Full tearsheet with all institutional metrics

Quick Start
-----------
    from src.backtesting.backtester import Backtester, BacktestConfig
    from src.backtesting.transaction_cost_model import TransactionCostModel
    from src.backtesting.performance_metrics import PerformanceMetrics

    config = BacktestConfig(
        initial_capital=1_000_000,
        commission=0.001,
        slippage=0.0005,
        rebalance_freq='monthly',
    )
    bt = Backtester(prices, signals, config, benchmark=benchmark_returns)
    result = bt.run()
    print(result.metrics)
    print(PerformanceMetrics(result).tearsheet())
"""

from .backtester import Backtester, BacktestConfig, BacktestResult
from .transaction_cost_model import TransactionCostModel, CostBreakdown
from .performance_metrics import PerformanceMetrics

__all__ = [
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "TransactionCostModel",
    "CostBreakdown",
    "PerformanceMetrics",
]
