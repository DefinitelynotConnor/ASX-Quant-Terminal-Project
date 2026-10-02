"""
portfolio_metrics.py
====================
Institutional-grade portfolio performance and risk statistics.

Computes the complete suite of metrics used by hedge funds and asset managers
to evaluate portfolio behaviour over time.

Metrics Implemented
-------------------
Performance : Cumulative return, CAGR, annualised volatility
Risk-Adjusted : Sharpe, Sortino, Calmar, Information ratio
Drawdown     : Maximum drawdown, drawdown duration, recovery time
Market       : Beta, Alpha, Correlation, R-squared vs benchmark
Tail Risk    : VaR, CVaR (Expected Shortfall), skewness, kurtosis
Trade Stats  : Win rate, profit factor, average win/loss
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────

@dataclass
class PerformanceSummary:
    """Container for all portfolio performance statistics."""

    # --- performance ---
    total_return: float = 0.0
    cagr: float = 0.0
    annualised_volatility: float = 0.0

    # --- risk-adjusted ---
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    omega_ratio: float = 0.0

    # --- drawdown ---
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0       # calendar days
    avg_drawdown_duration: float = 0.0
    recovery_factor: float = 0.0

    # --- market ---
    beta: float = 0.0
    alpha: float = 0.0
    correlation: float = 0.0
    r_squared: float = 0.0
    treynor_ratio: float = 0.0

    # --- tail risk ---
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    skewness: float = 0.0
    excess_kurtosis: float = 0.0

    # --- trade statistics ---
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0

    # --- period ---
    start_date: str = ""
    end_date: str = ""
    num_trading_days: int = 0
    num_years: float = 0.0

    # --- aggregate ---
    score_card: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
#  Core metrics engine
# ─────────────────────────────────────────────

class PortfolioMetrics:
    """
    Compute institutional-grade portfolio statistics from a daily return series.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns (not log returns).
    benchmark_returns : pd.Series, optional
        Benchmark return series (e.g. ^AXJO proxy).  Used for alpha, beta,
        information ratio and correlation calculations.
    risk_free_rate : float
        Annualised risk-free rate.  Defaults to the current RBA cash rate (4.35 %).
    periods_per_year : int
        Trading days per year.  Default 252.
    name : str
        Label used in display and dashboard outputs.

    Example
    -------
    >>> pm = PortfolioMetrics(port_returns, benchmark_returns)
    >>> summary = pm.compute_all()
    >>> print(summary.sharpe_ratio)
    """

    TRADING_DAYS = 252

    def __init__(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0435,
        periods_per_year: int = 252,
        name: str = "Portfolio",
    ) -> None:
        self.returns = portfolio_returns.dropna().copy()
        self.benchmark = benchmark_returns
        self.rfr = risk_free_rate
        self.ppy = periods_per_year
        self.name = name

        # daily risk-free rate
        self.rfr_daily = (1 + self.rfr) ** (1 / self.ppy) - 1

        # align benchmark to portfolio dates
        if self.benchmark is not None:
            self.benchmark = self.benchmark.reindex(self.returns.index).dropna()
            common = self.returns.index.intersection(self.benchmark.index)
            self.returns_aligned = self.returns.loc[common]
            self.benchmark_aligned = self.benchmark.loc[common]
        else:
            self.returns_aligned = self.returns
            self.benchmark_aligned = None

    # ──────────────────────────────────────
    #  Return Metrics
    # ──────────────────────────────────────

    def cumulative_return(self) -> float:
        """Total compounded return over the full period."""
        return float((1 + self.returns).prod() - 1)

    def cagr(self) -> float:
        """Compound Annual Growth Rate."""
        n = len(self.returns)
        years = n / self.ppy
        total = (1 + self.returns).prod()
        return float(total ** (1 / years) - 1) if years > 0 else 0.0

    def annualised_volatility(self) -> float:
        """Annualised standard deviation of daily returns."""
        return float(self.returns.std() * np.sqrt(self.ppy))

    def num_years(self) -> float:
        return len(self.returns) / self.ppy

    # ──────────────────────────────────────
    #  Risk-Adjusted Metrics
    # ──────────────────────────────────────

    def sharpe_ratio(self) -> float:
        """
        Sharpe Ratio: annualised excess return per unit of total volatility.
        SR = (E[R] - Rf) / σ(R)  × √T
        """
        excess = self.returns - self.rfr_daily
        if excess.std() == 0:
            return 0.0
        return float(excess.mean() / excess.std() * np.sqrt(self.ppy))

    def sortino_ratio(self) -> float:
        """
        Sortino Ratio: penalises only downside deviation.
        SR_sortino = (E[R] - Rf) / σ_downside × √T
        """
        excess = self.returns - self.rfr_daily
        downside = excess[excess < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        downside_std = np.sqrt(np.mean(downside ** 2))
        return float(excess.mean() / downside_std * np.sqrt(self.ppy))

    def calmar_ratio(self) -> float:
        """
        Calmar Ratio: CAGR / |Max Drawdown|
        Higher is better; commonly used by CTA and trend-following funds.
        """
        mdd = abs(self.max_drawdown())
        return float(self.cagr() / mdd) if mdd != 0 else 0.0

    def information_ratio(self) -> float:
        """
        Information Ratio: annualised active return / tracking error.
        Measures consistency of alpha generation vs the benchmark.
        """
        if self.benchmark_aligned is None:
            return 0.0
        active = self.returns_aligned - self.benchmark_aligned
        if active.std() == 0:
            return 0.0
        return float(active.mean() / active.std() * np.sqrt(self.ppy))

    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Omega Ratio: probability-weighted ratio of gains vs losses above
        a return threshold.
        Ω(τ) = ∫τ→∞ [1-F(r)]dr / ∫-∞→τ F(r)dr
        """
        excess = self.returns - threshold / self.ppy
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())
        return float(gains / losses) if losses != 0 else np.inf

    def treynor_ratio(self) -> float:
        """
        Treynor Ratio: excess return per unit of market (systematic) risk.
        TR = (E[R] - Rf) / β
        """
        beta = self.beta()
        if beta == 0:
            return 0.0
        ann_excess = self.cagr() - self.rfr
        return float(ann_excess / beta)

    # ──────────────────────────────────────
    #  Drawdown Analysis
    # ──────────────────────────────────────

    def drawdown_series(self) -> pd.Series:
        """
        Compute the full drawdown time series.
        DD_t = (NAV_t / peak_t) - 1
        """
        cumulative = (1 + self.returns).cumprod()
        rolling_max = cumulative.cummax()
        dd = (cumulative / rolling_max) - 1
        return dd

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough decline."""
        return float(self.drawdown_series().min())

    def avg_drawdown(self) -> float:
        """Average drawdown across all underwater periods."""
        dd = self.drawdown_series()
        return float(dd[dd < 0].mean())

    def drawdown_periods(self) -> list[dict]:
        """
        Identify all distinct drawdown episodes with:
            start, trough, end, depth, duration (days), recovery (days)
        """
        cumulative = (1 + self.returns).cumprod()
        rolling_max = cumulative.cummax()
        dd = (cumulative / rolling_max) - 1

        periods = []
        in_drawdown = False
        start_idx = None
        trough_idx = None
        trough_val = 0.0

        for i, (date, val) in enumerate(dd.items()):
            if val < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = date
                trough_val = val
                trough_idx = date
            elif val < 0 and in_drawdown:
                if val < trough_val:
                    trough_val = val
                    trough_idx = date
            elif val == 0 and in_drawdown:
                in_drawdown = False
                periods.append({
                    "start": start_idx,
                    "trough": trough_idx,
                    "end": date,
                    "depth": trough_val,
                    "duration_days": (date - start_idx).days,
                    "recovery_days": (date - trough_idx).days,
                })

        # Handle open drawdown
        if in_drawdown:
            last_date = dd.index[-1]
            periods.append({
                "start": start_idx,
                "trough": trough_idx,
                "end": None,
                "depth": trough_val,
                "duration_days": (last_date - start_idx).days,
                "recovery_days": None,
            })

        return periods

    def max_drawdown_duration(self) -> int:
        """Duration of the longest drawdown episode in calendar days."""
        periods = self.drawdown_periods()
        if not periods:
            return 0
        durations = [p["duration_days"] for p in periods]
        return max(durations)

    def avg_drawdown_duration(self) -> float:
        """Average drawdown episode duration in calendar days."""
        periods = self.drawdown_periods()
        if not periods:
            return 0.0
        durations = [p["duration_days"] for p in periods]
        return float(np.mean(durations))

    def recovery_factor(self) -> float:
        """
        Recovery Factor: total return / |max drawdown|
        Indicates how efficiently the strategy recovers from losses.
        """
        mdd = abs(self.max_drawdown())
        return float(self.cumulative_return() / mdd) if mdd != 0 else 0.0

    # ──────────────────────────────────────
    #  Market / Benchmark Metrics
    # ──────────────────────────────────────

    def _ols_regression(self) -> tuple[float, float, float]:
        """
        OLS regression of portfolio returns on benchmark returns.
        Returns (alpha_daily, beta, r_squared).
        """
        if self.benchmark_aligned is None or len(self.benchmark_aligned) < 10:
            return 0.0, 1.0, 0.0

        X = self.benchmark_aligned.values
        y = self.returns_aligned.values
        slope, intercept, r_val, _, _ = stats.linregress(X, y)
        return float(intercept), float(slope), float(r_val ** 2)

    def beta(self) -> float:
        """
        Market beta: sensitivity to benchmark moves.
        β = Cov(Rp, Rm) / Var(Rm)
        """
        _, beta, _ = self._ols_regression()
        return beta

    def alpha(self) -> float:
        """
        Jensen's Alpha (annualised): risk-adjusted excess return.
        α = Rp - [Rf + β(Rm - Rf)]
        """
        alpha_daily, beta, _ = self._ols_regression()
        ann_alpha = alpha_daily * self.ppy
        return float(ann_alpha)

    def correlation(self) -> float:
        """Pearson correlation with benchmark."""
        if self.benchmark_aligned is None:
            return 0.0
        return float(self.returns_aligned.corr(self.benchmark_aligned))

    def r_squared(self) -> float:
        """R² of portfolio vs benchmark: proportion of variance explained."""
        _, _, r2 = self._ols_regression()
        return r2

    # ──────────────────────────────────────
    #  Tail Risk Metrics
    # ──────────────────────────────────────

    def var(self, confidence: float = 0.95, method: str = "historical") -> float:
        """
        Value at Risk (loss is expressed as a positive number).

        Parameters
        ----------
        confidence : float   0.95 or 0.99
        method     : str     'historical' | 'parametric' | 'cornish_fisher'
        """
        if method == "historical":
            return float(-np.percentile(self.returns, (1 - confidence) * 100))

        elif method == "parametric":
            mu = self.returns.mean()
            sigma = self.returns.std()
            z = stats.norm.ppf(1 - confidence)
            return float(-(mu + z * sigma))

        elif method == "cornish_fisher":
            # Modified VaR with skewness & kurtosis correction
            mu = self.returns.mean()
            sigma = self.returns.std()
            s = stats.skew(self.returns)
            k = stats.kurtosis(self.returns)
            z = stats.norm.ppf(1 - confidence)
            z_cf = (
                z
                + (z**2 - 1) * s / 6
                + (z**3 - 3 * z) * k / 24
                - (2 * z**3 - 5 * z) * s**2 / 36
            )
            return float(-(mu + z_cf * sigma))

        raise ValueError(f"Unknown VaR method: {method}")

    def cvar(self, confidence: float = 0.95) -> float:
        """
        Conditional VaR (Expected Shortfall):
        Average loss in the worst (1-confidence)% of cases.
        CVaR = E[R | R < VaR_α]
        """
        threshold = np.percentile(self.returns, (1 - confidence) * 100)
        tail = self.returns[self.returns <= threshold]
        return float(-tail.mean()) if len(tail) > 0 else 0.0

    def skewness(self) -> float:
        """Return distribution skewness (negative = left tail risk)."""
        return float(stats.skew(self.returns))

    def excess_kurtosis(self) -> float:
        """Excess kurtosis (positive = fat tails)."""
        return float(stats.kurtosis(self.returns))

    # ──────────────────────────────────────
    #  Trade / Day Statistics
    # ──────────────────────────────────────

    def win_rate(self) -> float:
        """Percentage of days with positive returns."""
        return float((self.returns > 0).mean())

    def profit_factor(self) -> float:
        """
        Gross profit / gross loss.
        PF > 1 indicates net profitability.
        """
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        return float(gains / losses) if losses != 0 else np.inf

    def avg_win(self) -> float:
        wins = self.returns[self.returns > 0]
        return float(wins.mean()) if len(wins) > 0 else 0.0

    def avg_loss(self) -> float:
        losses = self.returns[self.returns < 0]
        return float(losses.mean()) if len(losses) > 0 else 0.0

    def best_day(self) -> float:
        return float(self.returns.max())

    def worst_day(self) -> float:
        return float(self.returns.min())

    # ──────────────────────────────────────
    #  Rolling Analytics
    # ──────────────────────────────────────

    def rolling_sharpe(self, window: int = 63) -> pd.Series:
        """Rolling Sharpe ratio over a trailing window (default = 1 quarter)."""
        def _sharpe(x):
            excess = x - self.rfr_daily
            return excess.mean() / excess.std() * np.sqrt(self.ppy) if excess.std() > 0 else 0.0
        return self.returns.rolling(window).apply(_sharpe, raw=False)

    def rolling_volatility(self, window: int = 21) -> pd.Series:
        """Rolling annualised volatility (default = 1 month)."""
        return self.returns.rolling(window).std() * np.sqrt(self.ppy)

    def rolling_beta(self, window: int = 63) -> pd.Series:
        """Rolling market beta over a trailing window."""
        if self.benchmark_aligned is None:
            return pd.Series(dtype=float)

        def _beta(idx):
            p = self.returns_aligned.iloc[max(0, idx - window): idx]
            b = self.benchmark_aligned.iloc[max(0, idx - window): idx]
            if len(p) < 5:
                return np.nan
            slope, _, _, _, _ = stats.linregress(b.values, p.values)
            return slope

        return pd.Series(
            [_beta(i) for i in range(len(self.returns_aligned))],
            index=self.returns_aligned.index,
        )

    def rolling_drawdown(self) -> pd.Series:
        """Full drawdown time series."""
        return self.drawdown_series()

    # ──────────────────────────────────────
    #  Monthly Return Matrix
    # ──────────────────────────────────────

    def monthly_returns(self) -> pd.DataFrame:
        """
        Pivot table of monthly returns: rows = year, cols = month.
        Useful for visualising seasonality.
        """
        monthly = self.returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
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
        # Annual total
        pivot["Annual"] = (1 + pivot.fillna(0)).prod(axis=1) - 1
        return pivot

    # ──────────────────────────────────────
    #  Master Summary
    # ──────────────────────────────────────

    def compute_all(self) -> PerformanceSummary:
        """
        Compute the full suite of metrics and return a PerformanceSummary.
        """
        s = PerformanceSummary()

        # Performance
        s.total_return = self.cumulative_return()
        s.cagr = self.cagr()
        s.annualised_volatility = self.annualised_volatility()

        # Risk-adjusted
        s.sharpe_ratio = self.sharpe_ratio()
        s.sortino_ratio = self.sortino_ratio()
        s.calmar_ratio = self.calmar_ratio()
        s.information_ratio = self.information_ratio()
        s.omega_ratio = self.omega_ratio()
        s.treynor_ratio = self.treynor_ratio()

        # Drawdown
        s.max_drawdown = self.max_drawdown()
        s.avg_drawdown = self.avg_drawdown()
        s.max_drawdown_duration = self.max_drawdown_duration()
        s.avg_drawdown_duration = self.avg_drawdown_duration()
        s.recovery_factor = self.recovery_factor()

        # Market
        s.beta = self.beta()
        s.alpha = self.alpha()
        s.correlation = self.correlation()
        s.r_squared = self.r_squared()

        # Tail risk
        s.var_95 = self.var(0.95)
        s.var_99 = self.var(0.99)
        s.cvar_95 = self.cvar(0.95)
        s.cvar_99 = self.cvar(0.99)
        s.skewness = self.skewness()
        s.excess_kurtosis = self.excess_kurtosis()

        # Trade stats
        s.win_rate = self.win_rate()
        s.profit_factor = self.profit_factor()
        s.avg_win = self.avg_win()
        s.avg_loss = self.avg_loss()
        s.best_day = self.best_day()
        s.worst_day = self.worst_day()

        # Period
        s.start_date = str(self.returns.index[0].date())
        s.end_date = str(self.returns.index[-1].date())
        s.num_trading_days = len(self.returns)
        s.num_years = self.num_years()

        # Graded score card
        s.score_card = self._build_score_card(s)

        return s

    def _build_score_card(self, s: PerformanceSummary) -> dict:
        """
        Grade each key metric against institutional benchmarks.
        Returns a dict of {metric: (value, grade, comment)}.
        """

        def grade(val, thresholds, labels):
            for t, l in zip(thresholds, labels):
                if val >= t:
                    return l
            return labels[-1]

        return {
            "Sharpe Ratio": (
                s.sharpe_ratio,
                grade(s.sharpe_ratio, [2.0, 1.5, 1.0, 0.5], ["Excellent", "Good", "Acceptable", "Poor", "Unacceptable"]),
            ),
            "Sortino Ratio": (
                s.sortino_ratio,
                grade(s.sortino_ratio, [3.0, 2.0, 1.5, 0.8], ["Excellent", "Good", "Acceptable", "Poor", "Unacceptable"]),
            ),
            "Max Drawdown": (
                s.max_drawdown,
                grade(-s.max_drawdown, [0.9, 0.85, 0.8, 0.7], ["Excellent", "Good", "Acceptable", "Poor", "Unacceptable"]),
            ),
            "Calmar Ratio": (
                s.calmar_ratio,
                grade(s.calmar_ratio, [3.0, 2.0, 1.0, 0.5], ["Excellent", "Good", "Acceptable", "Poor", "Unacceptable"]),
            ),
            "Win Rate": (
                s.win_rate,
                grade(s.win_rate, [0.60, 0.55, 0.50, 0.45], ["Excellent", "Good", "Acceptable", "Poor", "Unacceptable"]),
            ),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export all metrics as a two-column DataFrame for display."""
        s = self.compute_all()
        rows = [
            ("── PERFORMANCE ──", ""),
            ("Total Return", f"{s.total_return:.2%}"),
            ("CAGR", f"{s.cagr:.2%}"),
            ("Annualised Volatility", f"{s.annualised_volatility:.2%}"),
            ("── RISK-ADJUSTED ──", ""),
            ("Sharpe Ratio", f"{s.sharpe_ratio:.3f}"),
            ("Sortino Ratio", f"{s.sortino_ratio:.3f}"),
            ("Calmar Ratio", f"{s.calmar_ratio:.3f}"),
            ("Information Ratio", f"{s.information_ratio:.3f}"),
            ("Omega Ratio", f"{s.omega_ratio:.3f}"),
            ("Treynor Ratio", f"{s.treynor_ratio:.3f}"),
            ("── DRAWDOWN ──", ""),
            ("Max Drawdown", f"{s.max_drawdown:.2%}"),
            ("Avg Drawdown", f"{s.avg_drawdown:.2%}"),
            ("Max DD Duration (days)", f"{s.max_drawdown_duration}"),
            ("Avg DD Duration (days)", f"{s.avg_drawdown_duration:.1f}"),
            ("Recovery Factor", f"{s.recovery_factor:.2f}"),
            ("── MARKET ──", ""),
            ("Beta", f"{s.beta:.3f}"),
            ("Alpha (ann.)", f"{s.alpha:.2%}"),
            ("Correlation", f"{s.correlation:.3f}"),
            ("R-Squared", f"{s.r_squared:.3f}"),
            ("── TAIL RISK ──", ""),
            ("VaR 95%", f"{s.var_95:.2%}"),
            ("VaR 99%", f"{s.var_99:.2%}"),
            ("CVaR 95%", f"{s.cvar_95:.2%}"),
            ("CVaR 99%", f"{s.cvar_99:.2%}"),
            ("Skewness", f"{s.skewness:.3f}"),
            ("Excess Kurtosis", f"{s.excess_kurtosis:.3f}"),
            ("── TRADE STATISTICS ──", ""),
            ("Win Rate", f"{s.win_rate:.1%}"),
            ("Profit Factor", f"{s.profit_factor:.2f}"),
            ("Avg Win", f"{s.avg_win:.3%}"),
            ("Avg Loss", f"{s.avg_loss:.3%}"),
            ("Best Day", f"{s.best_day:.3%}"),
            ("Worst Day", f"{s.worst_day:.3%}"),
            ("── PERIOD ──", ""),
            ("Start Date", s.start_date),
            ("End Date", s.end_date),
            ("Trading Days", str(s.num_trading_days)),
            ("Years", f"{s.num_years:.2f}"),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"])


# ─────────────────────────────────────────────
#  Standalone helper functions
# ─────────────────────────────────────────────

def calculate_portfolio_returns(
    price_data: pd.DataFrame,
    weights: dict[str, float],
    rebalance: str = "monthly",
) -> pd.Series:
    """
    Compute weighted portfolio daily returns from a price DataFrame.

    Parameters
    ----------
    price_data  : pd.DataFrame   Adjusted close prices, columns = tickers.
    weights     : dict           {ticker: weight}, should sum to 1.
    rebalance   : str            'daily' | 'monthly' | 'quarterly' | 'never'

    Returns
    -------
    pd.Series   Portfolio daily return series.
    """
    tickers = list(weights.keys())
    prices = price_data[tickers].copy()
    asset_returns = prices.pct_change().dropna()

    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()   # normalise to 1

    if rebalance == "daily":
        port_returns = asset_returns @ w
        return pd.Series(port_returns, index=asset_returns.index, name="Portfolio")

    # Rebalance at frequency boundaries
    port_nav = pd.Series(1.0, index=asset_returns.index)
    holdings = w.copy()   # fractional holdings

    freq_map = {"monthly": "ME", "quarterly": "QE", "never": None}
    freq = freq_map.get(rebalance, "ME")

    if freq is None:
        port_returns = asset_returns @ w
        return pd.Series(port_returns, index=asset_returns.index, name="Portfolio")

    rebalance_dates = set(
        asset_returns.resample(freq).last().index.normalize()
    )

    nav = 1.0
    prev_nav = 1.0
    port_ret = []
    current_holdings = w.copy()

    for date, row in asset_returns.iterrows():
        daily_asset_ret = row.values
        nav = prev_nav * (1 + np.dot(current_holdings, daily_asset_ret))
        port_ret.append(nav / prev_nav - 1)

        if pd.Timestamp(date).normalize() in rebalance_dates:
            current_holdings = w.copy()   # reset to target weights

        prev_nav = nav

    return pd.Series(port_ret, index=asset_returns.index, name="Portfolio")


def calculate_portfolio_volatility(returns: pd.Series) -> float:
    """Annualised portfolio volatility. Convenience function."""
    return float(np.std(returns) * np.sqrt(252))


def compare_portfolios(
    portfolio_dict: dict[str, pd.Series],
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0435,
) -> pd.DataFrame:
    """
    Compare multiple portfolio return series side by side.

    Parameters
    ----------
    portfolio_dict : dict    {label: pd.Series of daily returns}
    benchmark      : pd.Series, optional
    risk_free_rate : float

    Returns
    -------
    pd.DataFrame   One row per portfolio, columns = key metrics.
    """
    records = []
    for label, ret in portfolio_dict.items():
        pm = PortfolioMetrics(ret, benchmark_returns=benchmark, risk_free_rate=risk_free_rate, name=label)
        s = pm.compute_all()
        records.append({
            "Portfolio": label,
            "CAGR": s.cagr,
            "Volatility": s.annualised_volatility,
            "Sharpe": s.sharpe_ratio,
            "Sortino": s.sortino_ratio,
            "Max DD": s.max_drawdown,
            "Calmar": s.calmar_ratio,
            "Beta": s.beta,
            "Alpha": s.alpha,
            "IR": s.information_ratio,
            "Win Rate": s.win_rate,
            "Profit Factor": s.profit_factor,
            "CVaR 95%": s.cvar_95,
            "Skewness": s.skewness,
        })
    return pd.DataFrame(records).set_index("Portfolio")
