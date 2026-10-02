"""
performance_metrics.py
======================
Comprehensive performance metrics for backtested strategies.

Produces institutional-grade tearsheets with all metrics used by
hedge funds, asset managers, and allocators to evaluate strategies.

Metrics Computed
----------------
    Return       : CAGR, total return, monthly returns
    Risk-Adjusted: Sharpe, Sortino, Calmar, Omega, Info Ratio
    Drawdown     : Max DD, duration, recovery, underwater periods
    Market       : Beta, alpha, correlation, R²
    Tail         : VaR, CVaR, skewness, kurtosis
    Trading      : Win rate, profit factor, turnover, cost drag
    Benchmark    : Active return, tracking error, Information Ratio

Usage
-----
    from src.backtesting.performance_metrics import PerformanceMetrics

    pm = PerformanceMetrics(backtest_result)
    metrics_df = pm.compute_all()
    print(pm.tearsheet())
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

if TYPE_CHECKING:
    from src.backtesting.backtester import BacktestResult


class PerformanceMetrics:
    """
    Compute all performance metrics from a BacktestResult.

    Parameters
    ----------
    result : BacktestResult   Output from Backtester.run().
    """

    PERIODS_PER_YEAR = 252

    def __init__(self, result: "BacktestResult") -> None:
        self.result = result
        self.returns = result.returns.dropna()
        self.nav = result.nav.dropna()
        self.benchmark = result.benchmark_returns
        self.config = result.config
        self.rfr = result.config.risk_free_rate if result.config else 0.0435
        self.rfr_daily = (1 + self.rfr) ** (1 / self.PERIODS_PER_YEAR) - 1
        self.name = result.strategy_name
        self.n = len(self.returns)

        # Align benchmark
        if self.benchmark is not None:
            common = self.returns.index.intersection(self.benchmark.index)
            self.ret_aligned = self.returns.loc[common]
            self.bm_aligned = self.benchmark.loc[common]
        else:
            self.ret_aligned = self.returns
            self.bm_aligned = None

    # ──────────────────────────────────────
    #  Return Metrics
    # ──────────────────────────────────────

    def total_return(self) -> float:
        return float((1 + self.returns).prod() - 1)

    def cagr(self) -> float:
        years = self.n / self.PERIODS_PER_YEAR
        return float((1 + self.returns).prod() ** (1 / years) - 1) if years > 0 else 0

    def annualised_volatility(self) -> float:
        return float(self.returns.std() * np.sqrt(self.PERIODS_PER_YEAR))

    # ──────────────────────────────────────
    #  Risk-Adjusted Metrics
    # ──────────────────────────────────────

    def sharpe_ratio(self) -> float:
        excess = self.returns - self.rfr_daily
        return float(
            excess.mean() / excess.std() * np.sqrt(self.PERIODS_PER_YEAR)
        ) if excess.std() > 0 else 0

    def sortino_ratio(self) -> float:
        excess = self.returns - self.rfr_daily
        downside = excess[excess < 0]
        downside_std = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 1e-10
        return float(
            excess.mean() / downside_std * np.sqrt(self.PERIODS_PER_YEAR)
        )

    def calmar_ratio(self) -> float:
        mdd = abs(self.max_drawdown())
        return float(self.cagr() / mdd) if mdd > 0 else 0

    def information_ratio(self) -> float:
        if self.bm_aligned is None:
            return 0.0
        active = self.ret_aligned - self.bm_aligned
        return float(
            active.mean() / active.std() * np.sqrt(self.PERIODS_PER_YEAR)
        ) if active.std() > 0 else 0

    def omega_ratio(self, threshold: float = 0.0) -> float:
        excess = self.returns - threshold / self.PERIODS_PER_YEAR
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())
        return float(gains / losses) if losses > 0 else np.inf

    def treynor_ratio(self) -> float:
        beta = self.beta()
        return float((self.cagr() - self.rfr) / beta) if beta != 0 else 0

    # ──────────────────────────────────────
    #  Drawdown Metrics
    # ──────────────────────────────────────

    def drawdown_series(self) -> pd.Series:
        nav = (1 + self.returns).cumprod()
        return (nav / nav.cummax() - 1).rename("Drawdown")

    def max_drawdown(self) -> float:
        return float(self.drawdown_series().min())

    def max_drawdown_duration(self) -> int:
        dd = self.drawdown_series()
        in_dd = dd < 0
        if not in_dd.any():
            return 0
        max_dur = 0
        cur_dur = 0
        for val in in_dd:
            if val:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0
        return max_dur

    def avg_drawdown(self) -> float:
        dd = self.drawdown_series()
        return float(dd[dd < 0].mean()) if (dd < 0).any() else 0

    def recovery_factor(self) -> float:
        mdd = abs(self.max_drawdown())
        return float(self.total_return() / mdd) if mdd > 0 else 0

    def ulcer_index(self) -> float:
        """
        Ulcer Index: RMS of drawdown series.
        Better than max DD for measuring sustained underwater periods.
        """
        dd = self.drawdown_series()
        return float(np.sqrt(np.mean(dd ** 2)))

    # ──────────────────────────────────────
    #  Market Metrics
    # ──────────────────────────────────────

    def beta(self) -> float:
        if self.bm_aligned is None:
            return 1.0
        slope, _, _, _, _ = stats.linregress(
            self.bm_aligned.values, self.ret_aligned.values
        )
        return float(slope)

    def alpha(self) -> float:
        if self.bm_aligned is None:
            return self.cagr() - self.rfr
        _, intercept, _, _, _ = stats.linregress(
            self.bm_aligned.values, self.ret_aligned.values
        )
        return float(intercept * self.PERIODS_PER_YEAR)

    def correlation(self) -> float:
        if self.bm_aligned is None:
            return 0.0
        return float(self.ret_aligned.corr(self.bm_aligned))

    def r_squared(self) -> float:
        return float(self.correlation() ** 2)

    def tracking_error(self) -> float:
        if self.bm_aligned is None:
            return 0.0
        active = self.ret_aligned - self.bm_aligned
        return float(active.std() * np.sqrt(self.PERIODS_PER_YEAR))

    def active_return(self) -> float:
        if self.bm_aligned is None:
            return 0.0
        bm_cagr = float(
            (1 + self.bm_aligned).prod()
            ** (self.PERIODS_PER_YEAR / len(self.bm_aligned)) - 1
        )
        return self.cagr() - bm_cagr

    # ──────────────────────────────────────
    #  Tail Risk Metrics
    # ──────────────────────────────────────

    def var(self, confidence: float = 0.95) -> float:
        return float(-np.percentile(self.returns, (1 - confidence) * 100))

    def cvar(self, confidence: float = 0.95) -> float:
        threshold = np.percentile(self.returns, (1 - confidence) * 100)
        tail = self.returns[self.returns <= threshold]
        return float(-tail.mean()) if len(tail) > 0 else 0

    def skewness(self) -> float:
        return float(stats.skew(self.returns))

    def excess_kurtosis(self) -> float:
        return float(stats.kurtosis(self.returns))

    # ──────────────────────────────────────
    #  Trading Metrics
    # ──────────────────────────────────────

    def win_rate(self) -> float:
        return float((self.returns > 0).mean())

    def profit_factor(self) -> float:
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        return float(gains / losses) if losses > 0 else np.inf

    def avg_win(self) -> float:
        w = self.returns[self.returns > 0]
        return float(w.mean()) if len(w) > 0 else 0

    def avg_loss(self) -> float:
        l = self.returns[self.returns < 0]
        return float(l.mean()) if len(l) > 0 else 0

    def best_day(self) -> float:
        return float(self.returns.max())

    def worst_day(self) -> float:
        return float(self.returns.min())

    def avg_annual_turnover(self) -> float:
        if hasattr(self.result, "turnover") and not self.result.turnover.empty:
            return float(self.result.turnover.sum() / (self.n / self.PERIODS_PER_YEAR))
        return 0.0

    def total_cost_drag(self) -> float:
        """Annual cost drag as % of average NAV."""
        if hasattr(self.result, "costs") and not self.result.costs.empty:
            annual_costs = self.result.costs.sum() / (self.n / self.PERIODS_PER_YEAR)
            avg_nav = self.nav.mean()
            return float(annual_costs / avg_nav) if avg_nav > 0 else 0
        return 0

    # ──────────────────────────────────────
    #  Monthly returns
    # ──────────────────────────────────────

    def monthly_returns(self) -> pd.DataFrame:
        """Pivot table of monthly returns: rows=year, cols=month."""
        monthly = self.returns.resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )
        monthly.index = monthly.index.to_period("M")
        df = monthly.reset_index()
        df.columns = ["period", "return"]
        df["year"] = df["period"].dt.year
        df["month"] = df["period"].dt.month
        pivot = df.pivot(index="year", columns="month", values="return")
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        pivot["Annual"] = (1 + pivot.fillna(0)).prod(axis=1) - 1
        return pivot

    # ──────────────────────────────────────
    #  Master compute
    # ──────────────────────────────────────

    def compute_all(self) -> pd.DataFrame:
        """
        Compute all metrics and return as a two-column DataFrame.
        """
        years = self.n / self.PERIODS_PER_YEAR
        rows = [
            ("── STRATEGY ──", ""),
            ("Strategy", self.name),
            ("Start Date", str(self.returns.index[0].date())),
            ("End Date", str(self.returns.index[-1].date())),
            ("Trading Days", str(self.n)),
            ("Years", f"{years:.2f}"),
            ("Initial Capital", f"${self.result.config.initial_capital:,.0f}"
             if self.result.config else "N/A"),
            ("Final NAV", f"${self.nav.iloc[-1]:,.0f}"),
            ("── RETURNS ──", ""),
            ("Total Return", f"{self.total_return():.2%}"),
            ("CAGR", f"{self.cagr():.2%}"),
            ("Annualised Volatility", f"{self.annualised_volatility():.2%}"),
            ("── RISK-ADJUSTED ──", ""),
            ("Sharpe Ratio", f"{self.sharpe_ratio():.3f}"),
            ("Sortino Ratio", f"{self.sortino_ratio():.3f}"),
            ("Calmar Ratio", f"{self.calmar_ratio():.3f}"),
            ("Omega Ratio", f"{self.omega_ratio():.3f}"),
            ("Treynor Ratio", f"{self.treynor_ratio():.4f}"),
            ("Information Ratio", f"{self.information_ratio():.3f}"),
            ("── DRAWDOWN ──", ""),
            ("Max Drawdown", f"{self.max_drawdown():.2%}"),
            ("Avg Drawdown", f"{self.avg_drawdown():.2%}"),
            ("Max DD Duration (days)", f"{self.max_drawdown_duration()}"),
            ("Recovery Factor", f"{self.recovery_factor():.2f}"),
            ("Ulcer Index", f"{self.ulcer_index():.4f}"),
            ("── BENCHMARK ──", ""),
            ("Beta", f"{self.beta():.3f}"),
            ("Alpha (ann.)", f"{self.alpha():.2%}"),
            ("Active Return", f"{self.active_return():.2%}"),
            ("Tracking Error", f"{self.tracking_error():.2%}"),
            ("Correlation", f"{self.correlation():.3f}"),
            ("R-Squared", f"{self.r_squared():.3f}"),
            ("── TAIL RISK ──", ""),
            ("VaR 95%", f"{self.var(0.95):.2%}"),
            ("VaR 99%", f"{self.var(0.99):.2%}"),
            ("CVaR 95%", f"{self.cvar(0.95):.2%}"),
            ("CVaR 99%", f"{self.cvar(0.99):.2%}"),
            ("Skewness", f"{self.skewness():.3f}"),
            ("Excess Kurtosis", f"{self.excess_kurtosis():.3f}"),
            ("── TRADING ──", ""),
            ("Win Rate", f"{self.win_rate():.1%}"),
            ("Profit Factor", f"{self.profit_factor():.2f}"),
            ("Avg Win", f"{self.avg_win():.3%}"),
            ("Avg Loss", f"{self.avg_loss():.3%}"),
            ("Best Day", f"{self.best_day():.3%}"),
            ("Worst Day", f"{self.worst_day():.3%}"),
            ("Avg Annual Turnover", f"{self.avg_annual_turnover():.1%}"),
            ("Annual Cost Drag", f"{self.total_cost_drag():.3%}"),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"])

    def tearsheet(self) -> str:
        """Print a formatted tearsheet to console."""
        df = self.compute_all()
        lines = [
            f"\n{'=' * 55}",
            f"  BACKTEST TEARSHEET: {self.name}",
            f"{'=' * 55}",
        ]
        for _, row in df.iterrows():
            if row["Value"] == "":
                lines.append(f"\n  {row['Metric']}")
            else:
                lines.append(f"  {row['Metric']:<30} {row['Value']}")
        lines.append(f"{'=' * 55}\n")
        return "\n".join(lines)

    # ──────────────────────────────────────
    #  Multi-strategy comparison
    # ──────────────────────────────────────

    @staticmethod
    def compare_strategies(
        results: dict[str, "BacktestResult"],
        benchmark: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple strategy backtest results side by side.

        Parameters
        ----------
        results   : dict   {strategy_name: BacktestResult}
        benchmark : pd.Series, optional   Benchmark return series.

        Returns
        -------
        pd.DataFrame   One row per strategy, key metrics as columns.
        """
        rows = []
        for name, result in results.items():
            if benchmark is not None and result.benchmark_returns is None:
                result.benchmark_returns = benchmark
            pm = PerformanceMetrics(result)
            rows.append({
                "Strategy": name,
                "CAGR": pm.cagr(),
                "Volatility": pm.annualised_volatility(),
                "Sharpe": pm.sharpe_ratio(),
                "Sortino": pm.sortino_ratio(),
                "Max DD": pm.max_drawdown(),
                "Calmar": pm.calmar_ratio(),
                "Beta": pm.beta(),
                "Alpha": pm.alpha(),
                "Info Ratio": pm.information_ratio(),
                "Win Rate": pm.win_rate(),
                "Profit Factor": pm.profit_factor(),
                "CVaR 95%": pm.cvar(0.95),
                "Turnover": pm.avg_annual_turnover(),
                "Cost Drag": pm.total_cost_drag(),
            })
        return pd.DataFrame(rows).set_index("Strategy")
