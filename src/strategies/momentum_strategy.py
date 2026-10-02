"""
momentum_strategy.py
====================
Cross-sectional and time-series momentum strategies for ASX equities.

Strategies Implemented
----------------------
CrossSectionalMomentum  : Rank stocks by past returns, long top / short bottom
TimeSeriesMomentum      : Long when own past return positive, flat/short otherwise
DualMomentumStrategy    : Combines absolute and relative momentum (Antonacci)
RiskAdjustedMomentum    : Momentum normalised by volatility (Sharpe-like score)

All strategies produce a signal DataFrame compatible with Backtester.

Usage
-----
    from src.strategies.momentum_strategy import CrossSectionalMomentum

    strategy = CrossSectionalMomentum(
        universe_prices=prices,
        lookback=252,
        skip=21,
        n_long=5,
        n_short=0,
    )
    signals = strategy.generate_signals()
    result = strategy.backtest(benchmark=benchmark_returns)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class CrossSectionalMomentum:
    """
    Cross-sectional momentum strategy.

    Each month, rank all stocks by their past 12-1 month return.
    Go long the top N stocks and optionally short the bottom N.
    Rebalance monthly.

    Parameters
    ----------
    universe_prices : pd.DataFrame   Adjusted close prices.
    lookback        : int            Momentum lookback in days (default 252 = 12M).
    skip            : int            Skip period in days (default 21 = 1M).
    n_long          : int            Number of long positions.
    n_short         : int            Number of short positions (0 = long only).
    weighting       : str            'equal' | 'rank' | 'score'
    rebalance_freq  : str            'monthly' | 'weekly'
    """

    def __init__(
        self,
        universe_prices: pd.DataFrame,
        lookback: int = 252,
        skip: int = 21,
        n_long: int = 10,
        n_short: int = 0,
        weighting: str = "equal",
        rebalance_freq: str = "monthly",
    ) -> None:
        self.prices = universe_prices.copy()
        self.lookback = lookback
        self.skip = skip
        self.n_long = n_long
        self.n_short = n_short
        self.weighting = weighting
        self.rebalance_freq = rebalance_freq

    def _momentum_score(self) -> pd.DataFrame:
        """Compute 12-1 month momentum score for each ticker."""
        return self.prices.shift(self.skip) / self.prices.shift(self.lookback) - 1

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate daily position signals.

        Returns
        -------
        pd.DataFrame  index=dates, columns=tickers, values=weights.
        """
        scores = self._momentum_score()
        signals = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)

        # Only rebalance at frequency boundaries
        if self.rebalance_freq == "monthly":
            rebalance_dates = set(
                scores.resample("ME").last().index.normalize()
            )
        else:
            rebalance_dates = set(
                scores.resample("W").last().index.normalize()
            )

        last_signal = pd.Series(0.0, index=scores.columns)

        for date in scores.index:
            if pd.Timestamp(date).normalize() in rebalance_dates:
                row = scores.loc[date].dropna()
                if len(row) < self.n_long:
                    last_signal = pd.Series(0.0, index=scores.columns)
                    signals.loc[date] = last_signal
                    continue

                new_signal = pd.Series(0.0, index=scores.columns)

                # Long positions
                longs = row.nlargest(self.n_long).index
                # Short positions
                shorts = row.nsmallest(self.n_short).index if self.n_short > 0 else []

                if self.weighting == "equal":
                    n_total = self.n_long + self.n_short
                    w = 1.0 / n_total if n_total > 0 else 0.0
                    new_signal[longs] = w
                    if self.n_short > 0:
                        new_signal[shorts] = -w

                elif self.weighting == "rank":
                    # Weight proportional to rank
                    ranks = row.rank()
                    long_ranks = ranks[longs]
                    new_signal[longs] = long_ranks / long_ranks.sum()
                    if self.n_short > 0:
                        short_ranks = ranks[shorts]
                        new_signal[shorts] = -(short_ranks / short_ranks.sum())

                elif self.weighting == "score":
                    # Weight proportional to momentum score
                    long_scores = row[longs].clip(lower=0)
                    if long_scores.sum() > 0:
                        new_signal[longs] = long_scores / long_scores.sum()
                    if self.n_short > 0:
                        short_scores = (-row[shorts]).clip(lower=0)
                        if short_scores.sum() > 0:
                            new_signal[shorts] = -(short_scores / short_scores.sum())

                last_signal = new_signal

            signals.loc[date] = last_signal

        return signals

    def backtest(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        benchmark: Optional[pd.Series] = None,
    ):
        """Run a full backtest of this strategy."""
        from src.backtesting.backtester import Backtester, BacktestConfig

        signals = self.generate_signals()
        config = BacktestConfig(
            initial_capital=initial_capital,
            commission=commission,
            slippage=0.0005,
            rebalance_freq="monthly",
            signal_lag=1,
        )
        bt = Backtester(
            prices=self.prices,
            signals=signals,
            config=config,
            benchmark=benchmark,
            strategy_name="CrossSectional_Momentum",
        )
        return bt.run()


class TimeSeriesMomentum:
    """
    Time-series (absolute) momentum strategy.
    Each asset is traded based on its own past return — not relative ranking.

    Long  : If own 12-month return is positive.
    Flat  : If own 12-month return is negative (long-only version).
    Short : If own 12-month return is negative (long/short version).

    Parameters
    ----------
    universe_prices : pd.DataFrame   Close prices.
    lookback        : int            Lookback period in days.
    allow_short     : bool           Allow short positions.
    volatility_scale: bool           Scale position size by inverse volatility.
    target_vol      : float          Target annualised portfolio volatility.
    """

    def __init__(
        self,
        universe_prices: pd.DataFrame,
        lookback: int = 252,
        allow_short: bool = False,
        volatility_scale: bool = True,
        target_vol: float = 0.15,
    ) -> None:
        self.prices = universe_prices.copy()
        self.lookback = lookback
        self.allow_short = allow_short
        self.volatility_scale = volatility_scale
        self.target_vol = target_vol

    def generate_signals(self) -> pd.DataFrame:
        """Generate time-series momentum signals."""
        returns = self.prices.pct_change()
        past_ret = self.prices / self.prices.shift(self.lookback) - 1

        # Raw signal: +1 if positive momentum, -1 or 0 if negative
        if self.allow_short:
            raw_signal = np.sign(past_ret)
        else:
            raw_signal = (past_ret > 0).astype(float)

        # Volatility scaling
        if self.volatility_scale:
            vol = returns.rolling(63).std() * np.sqrt(252)
            vol = vol.replace(0, np.nan)
            scaling = (self.target_vol / vol).clip(0, 2)
            signals = raw_signal * scaling
        else:
            signals = raw_signal

        # Normalise weights
        row_sum = signals.abs().sum(axis=1).replace(0, np.nan)
        signals = signals.div(row_sum, axis=0).fillna(0)

        return signals

    def backtest(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        benchmark: Optional[pd.Series] = None,
    ):
        """Run a full backtest."""
        from src.backtesting.backtester import Backtester, BacktestConfig

        signals = self.generate_signals()
        config = BacktestConfig(
            initial_capital=initial_capital,
            commission=commission,
            slippage=0.0005,
            allow_short=self.allow_short,
            signal_lag=1,
        )
        bt = Backtester(
            prices=self.prices,
            signals=signals,
            config=config,
            benchmark=benchmark,
            strategy_name="TimeSeries_Momentum",
        )
        return bt.run()


class DualMomentumStrategy:
    """
    Antonacci (2014) Dual Momentum Strategy.

    Combines:
        1. Relative momentum  : Select the best-performing asset class.
        2. Absolute momentum  : Only hold if its own return beats T-bills.
           Otherwise hold safe-haven (bonds/cash proxy).

    Adapted for ASX: compares ASX 200, global equities (VGS proxy),
    and a defensive asset (e.g. bonds ETF or cash).

    Parameters
    ----------
    risky_assets    : pd.DataFrame   Risky asset prices (equities).
    safe_asset      : pd.Series      Safe haven asset (bonds/cash).
    lookback        : int            Momentum lookback (days).
    risk_free_rate  : float          Annualised risk-free rate.
    """

    def __init__(
        self,
        risky_assets: pd.DataFrame,
        safe_asset: pd.Series,
        lookback: int = 252,
        risk_free_rate: float = 0.0435,
    ) -> None:
        self.risky = risky_assets.copy()
        self.safe = safe_asset.copy()
        self.lookback = lookback
        self.rfr = risk_free_rate
        self.rfr_daily = (1 + risk_free_rate) ** (1 / 252) - 1

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate dual momentum signals.

        Returns
        -------
        pd.DataFrame  Weights for risky assets + safe asset.
        """
        all_assets = pd.concat([self.risky, self.safe.rename("Safe")], axis=1)
        momentum = all_assets / all_assets.shift(self.lookback) - 1

        all_tickers = list(self.risky.columns) + ["Safe"]
        signals = pd.DataFrame(0.0, index=momentum.index, columns=all_tickers)

        rebalance_dates = set(
            momentum.resample("ME").last().index.normalize()
        )

        last_signal = pd.Series(0.0, index=all_tickers)

        for date in momentum.index:
            if pd.Timestamp(date).normalize() in rebalance_dates:
                row = momentum.loc[date].dropna()
                risky_mom = row[self.risky.columns].dropna()

                if risky_mom.empty:
                    last_signal = pd.Series(0.0, index=all_tickers)
                    last_signal["Safe"] = 1.0
                    signals.loc[date] = last_signal
                    continue

                # Step 1: Which risky asset has highest momentum?
                best_risky = risky_mom.idxmax()
                best_mom = risky_mom[best_risky]

                # Step 2: Absolute momentum — beat risk-free rate?
                rfr_lookback = self.rfr_daily * self.lookback
                new_signal = pd.Series(0.0, index=all_tickers)

                if best_mom > rfr_lookback:
                    new_signal[best_risky] = 1.0
                else:
                    new_signal["Safe"] = 1.0

                last_signal = new_signal

            signals.loc[date] = last_signal

        return signals

    def backtest(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        benchmark: Optional[pd.Series] = None,
    ):
        """Run dual momentum backtest."""
        from src.backtesting.backtester import Backtester, BacktestConfig

        all_prices = pd.concat(
            [self.risky, self.safe.rename("Safe")], axis=1
        )
        signals = self.generate_signals()
        config = BacktestConfig(
            initial_capital=initial_capital,
            commission=commission,
            slippage=0.0005,
            signal_lag=1,
        )
        bt = Backtester(
            prices=all_prices,
            signals=signals,
            config=config,
            benchmark=benchmark,
            strategy_name="Dual_Momentum",
        )
        return bt.run()


class RiskAdjustedMomentum:
    """
    Risk-adjusted momentum: momentum score divided by trailing volatility.
    Equivalent to a trailing Sharpe ratio as a ranking signal.

    Avoids the lottery-ticket problem of pure momentum strategies
    (ranking volatile stocks with large recent gains).

    Parameters
    ----------
    universe_prices  : pd.DataFrame
    lookback         : int   Momentum lookback.
    vol_window       : int   Volatility estimation window.
    n_long           : int   Long positions.
    """

    def __init__(
        self,
        universe_prices: pd.DataFrame,
        lookback: int = 252,
        skip: int = 21,
        vol_window: int = 63,
        n_long: int = 10,
    ) -> None:
        self.prices = universe_prices.copy()
        self.lookback = lookback
        self.skip = skip
        self.vol_window = vol_window
        self.n_long = n_long

    def generate_signals(self) -> pd.DataFrame:
        """Generate risk-adjusted momentum signals."""
        returns = self.prices.pct_change()
        mom = self.prices.shift(self.skip) / self.prices.shift(self.lookback) - 1
        vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        score = mom / vol.replace(0, np.nan)

        signals = pd.DataFrame(0.0, index=score.index, columns=score.columns)
        rebalance_dates = set(
            score.resample("ME").last().index.normalize()
        )
        last_signal = pd.Series(0.0, index=score.columns)

        for date in score.index:
            if pd.Timestamp(date).normalize() in rebalance_dates:
                row = score.loc[date].dropna()
                if len(row) < self.n_long:
                    last_signal = pd.Series(0.0, index=score.columns)
                    signals.loc[date] = last_signal
                    continue
                new_signal = pd.Series(0.0, index=score.columns)
                longs = row.nlargest(self.n_long).index
                new_signal[longs] = 1.0 / self.n_long
                last_signal = new_signal
            signals.loc[date] = last_signal

        return signals

    def backtest(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        benchmark: Optional[pd.Series] = None,
    ):
        """Run backtest."""
        from src.backtesting.backtester import Backtester, BacktestConfig

        signals = self.generate_signals()
        config = BacktestConfig(
            initial_capital=initial_capital,
            commission=commission,
            slippage=0.0005,
            signal_lag=1,
        )
        bt = Backtester(
            prices=self.prices,
            signals=signals,
            config=config,
            benchmark=benchmark,
            strategy_name="RiskAdjusted_Momentum",
        )
        return bt.run()
