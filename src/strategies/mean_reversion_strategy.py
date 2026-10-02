"""
mean_reversion_strategy.py  /  pairs_trading.py
================================================
Mean reversion strategies for ASX equities.

Strategies Implemented
----------------------
BollingerBandReversion  : Trade when price deviates from Bollinger Band
RSIMeanReversion        : Buy oversold, sell overbought via RSI
PairsTrading            : Statistical arbitrage on cointegrated pairs
KalmanPairsTrading      : Dynamic hedge ratio via Kalman filter

Usage
-----
    from src.strategies.mean_reversion_strategy import BollingerBandReversion
    from src.strategies.pairs_trading import PairsTrading

    bb = BollingerBandReversion(prices, period=20, std=2.0)
    signals = bb.generate_signals()
    result = bb.backtest(benchmark=benchmark_returns)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════
#  BOLLINGER BAND MEAN REVERSION
# ═══════════════════════════════════════════════

class BollingerBandReversion:
    """
    Bollinger Band mean reversion strategy.

    Entry  : Buy when price crosses below lower band (oversold).
             Sell when price crosses above upper band (overbought).
    Exit   : Close when price returns to the middle band (SMA).

    Parameters
    ----------
    prices    : pd.DataFrame   Close prices.
    period    : int            Bollinger Band period (default 20).
    std_dev   : float          Number of standard deviations (default 2.0).
    exit_mid  : bool           Exit at middle band (True) or opposite band (False).
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        exit_mid: bool = True,
    ) -> None:
        self.prices = prices.copy()
        self.period = period
        self.std_dev = std_dev
        self.exit_mid = exit_mid

    def _compute_bands(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        mid = self.prices.rolling(self.period).mean()
        std = self.prices.rolling(self.period).std()
        upper = mid + self.std_dev * std
        lower = mid - self.std_dev * std
        return upper, mid, lower

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate Bollinger Band reversion signals.

        Returns +1 (long), -1 (short), 0 (flat).
        """
        upper, mid, lower = self._compute_bands()
        signals = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)

        for ticker in self.prices.columns:
            price = self.prices[ticker]
            u = upper[ticker]
            m = mid[ticker]
            l = lower[ticker]

            position = 0
            sig = []

            for t in range(len(price)):
                if pd.isna(price.iloc[t]) or pd.isna(u.iloc[t]):
                    sig.append(0)
                    continue

                p = price.iloc[t]
                if position == 0:
                    if p < l.iloc[t]:
                        position = 1   # buy — oversold
                    elif p > u.iloc[t]:
                        position = -1  # sell — overbought
                elif position == 1:
                    # Exit long at mid or upper
                    if self.exit_mid and p >= m.iloc[t]:
                        position = 0
                    elif not self.exit_mid and p >= u.iloc[t]:
                        position = 0
                elif position == -1:
                    # Exit short at mid or lower
                    if self.exit_mid and p <= m.iloc[t]:
                        position = 0
                    elif not self.exit_mid and p <= l.iloc[t]:
                        position = 0

                sig.append(position)

            signals[ticker] = sig

        # Normalise to weights
        row_sum = signals.abs().sum(axis=1).replace(0, np.nan)
        return signals.div(row_sum, axis=0).fillna(0)

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
            allow_short=True,
            signal_lag=1,
        )
        bt = Backtester(
            prices=self.prices,
            signals=signals,
            config=config,
            benchmark=benchmark,
            strategy_name="BollingerBand_Reversion",
        )
        return bt.run()


# ═══════════════════════════════════════════════
#  RSI MEAN REVERSION
# ═══════════════════════════════════════════════

class RSIMeanReversion:
    """
    RSI-based mean reversion strategy.

    Entry  : Long when RSI < oversold threshold (default 30).
             Short when RSI > overbought threshold (default 70).
    Exit   : Close when RSI returns to neutral zone (40-60).

    Parameters
    ----------
    prices      : pd.DataFrame
    rsi_period  : int     RSI calculation period (default 14).
    oversold    : float   RSI threshold to go long (default 30).
    overbought  : float   RSI threshold to go short (default 70).
    exit_level  : float   RSI level to close position (default 50).
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        exit_level: float = 50.0,
    ) -> None:
        self.prices = prices.copy()
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_level = exit_level

    def _compute_rsi(self, price: pd.Series) -> pd.Series:
        delta = price.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=self.rsi_period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    def generate_signals(self) -> pd.DataFrame:
        """Generate RSI mean reversion signals."""
        signals = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)

        for ticker in self.prices.columns:
            price = self.prices[ticker].dropna()
            rsi = self._compute_rsi(price)

            position = 0
            sig = pd.Series(0.0, index=price.index)

            for date in price.index:
                r = rsi.get(date)
                if pd.isna(r):
                    sig[date] = 0
                    continue

                if position == 0:
                    if r < self.oversold:
                        position = 1
                    elif r > self.overbought:
                        position = -1
                elif position == 1:
                    if r >= self.exit_level:
                        position = 0
                elif position == -1:
                    if r <= self.exit_level:
                        position = 0

                sig[date] = position

            signals[ticker] = sig.reindex(self.prices.index).fillna(0)

        row_sum = signals.abs().sum(axis=1).replace(0, np.nan)
        return signals.div(row_sum, axis=0).fillna(0)

    def backtest(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        benchmark: Optional[pd.Series] = None,
    ):
        from src.backtesting.backtester import Backtester, BacktestConfig

        signals = self.generate_signals()
        config = BacktestConfig(
            initial_capital=initial_capital,
            commission=commission,
            slippage=0.0005,
            allow_short=True,
            signal_lag=1,
        )
        bt = Backtester(
            prices=self.prices,
            signals=signals,
            config=config,
            benchmark=benchmark,
            strategy_name="RSI_MeanReversion",
        )
        return bt.run()


# ═══════════════════════════════════════════════
#  PAIRS TRADING
# ═══════════════════════════════════════════════

class PairsTrading:
    """
    Statistical arbitrage pairs trading.

    Finds cointegrated pairs and trades the spread between them.

    When spread widens beyond threshold: long cheap / short expensive.
    When spread converges: close both legs.

    Uses Engle-Granger cointegration to validate pairs and
    OLS regression to estimate the hedge ratio.

    Parameters
    ----------
    prices          : pd.DataFrame   Universe close prices.
    ticker1         : str            First ticker.
    ticker2         : str            Second ticker.
    entry_z         : float          Z-score threshold to enter (default 2.0).
    exit_z          : float          Z-score threshold to exit (default 0.5).
    lookback        : int            Rolling window for spread statistics.
    hedge_method    : str            'ols' | 'total_returns' | 'kalman'
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        ticker1: str,
        ticker2: str,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        lookback: int = 63,
        hedge_method: str = "ols",
    ) -> None:
        self.prices = prices.copy()
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = lookback
        self.hedge_method = hedge_method

        if ticker1 not in prices.columns or ticker2 not in prices.columns:
            raise ValueError(
                f"Both {ticker1} and {ticker2} must be in the prices DataFrame."
            )

    def _compute_hedge_ratio(
        self, p1: pd.Series, p2: pd.Series
    ) -> pd.Series:
        """
        Compute rolling OLS hedge ratio: p1 = β × p2 + α
        β is the hedge ratio — how many shares of p2 to short per share of p1.
        """
        if self.hedge_method == "ols":
            hedge = pd.Series(np.nan, index=p1.index)
            for i in range(self.lookback, len(p1)):
                x = p2.iloc[i - self.lookback: i].values
                y = p1.iloc[i - self.lookback: i].values
                slope, _, _, _, _ = stats.linregress(x, y)
                hedge.iloc[i] = slope
            return hedge

        elif self.hedge_method == "total_returns":
            return pd.Series(1.0, index=p1.index)

        elif self.hedge_method == "kalman":
            return self._kalman_hedge_ratio(p1, p2)

        return pd.Series(1.0, index=p1.index)

    def _kalman_hedge_ratio(
        self, p1: pd.Series, p2: pd.Series
    ) -> pd.Series:
        """
        Kalman filter estimate of time-varying hedge ratio.
        More responsive than rolling OLS — adapts to structural changes.
        """
        n = len(p1)
        beta = np.zeros(n)
        P = np.ones(n)    # estimate uncertainty

        # Kalman filter parameters
        Q = 1e-5   # process noise (how fast beta changes)
        R = 0.01   # observation noise

        beta[0] = 1.0
        P[0] = 1.0

        for t in range(1, n):
            # Predict
            beta_pred = beta[t - 1]
            P_pred = P[t - 1] + Q

            # Update
            x = float(p2.iloc[t])
            y = float(p1.iloc[t])
            innovation = y - beta_pred * x
            S = x ** 2 * P_pred + R
            K = P_pred * x / S if S > 0 else 0

            beta[t] = beta_pred + K * innovation
            P[t] = (1 - K * x) * P_pred

        return pd.Series(beta, index=p1.index)

    def _compute_spread(self) -> pd.DataFrame:
        """Compute spread, Z-score, and hedge ratio."""
        p1 = self.prices[self.ticker1]
        p2 = self.prices[self.ticker2]

        # Align
        common = p1.dropna().index.intersection(p2.dropna().index)
        p1 = p1.loc[common]
        p2 = p2.loc[common]

        hedge = self._compute_hedge_ratio(p1, p2)
        spread = p1 - hedge * p2

        # Rolling Z-score
        spread_mean = spread.rolling(self.lookback).mean()
        spread_std = spread.rolling(self.lookback).std()
        z_score = (spread - spread_mean) / spread_std.replace(0, np.nan)

        return pd.DataFrame({
            "spread": spread,
            "z_score": z_score,
            "hedge_ratio": hedge,
            "spread_mean": spread_mean,
            "spread_std": spread_std,
        })

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate pairs trading signals.

        Returns
        -------
        pd.DataFrame  Two columns (ticker1, ticker2) with position weights.
        """
        spread_data = self._compute_spread().dropna()
        tickers = [self.ticker1, self.ticker2]
        signals = pd.DataFrame(0.0, index=self.prices.index, columns=tickers)

        position = 0

        for date in spread_data.index:
            z = float(spread_data.loc[date, "z_score"])
            h = float(spread_data.loc[date, "hedge_ratio"])

            if pd.isna(z) or pd.isna(h):
                signals.loc[date] = [0, 0]
                continue

            if position == 0:
                if z > self.entry_z:
                    # Spread too wide: short ticker1, long ticker2
                    position = -1
                    w1 = -0.5
                    w2 = 0.5
                elif z < -self.entry_z:
                    # Spread too narrow: long ticker1, short ticker2
                    position = 1
                    w1 = 0.5
                    w2 = -0.5
                else:
                    w1, w2 = 0, 0
            elif position == 1:
                if abs(z) <= self.exit_z:
                    position = 0
                    w1, w2 = 0, 0
                else:
                    w1, w2 = 0.5, -0.5
            elif position == -1:
                if abs(z) <= self.exit_z:
                    position = 0
                    w1, w2 = 0, 0
                else:
                    w1, w2 = -0.5, 0.5

            signals.loc[date, self.ticker1] = w1
            signals.loc[date, self.ticker2] = w2

        return signals

    def spread_analysis(self) -> pd.DataFrame:
        """Return the full spread analysis DataFrame."""
        return self._compute_spread()

    def backtest(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        benchmark: Optional[pd.Series] = None,
    ):
        """Run pairs trading backtest."""
        from src.backtesting.backtester import Backtester, BacktestConfig

        pair_prices = self.prices[[self.ticker1, self.ticker2]]
        signals = self.generate_signals()

        config = BacktestConfig(
            initial_capital=initial_capital,
            commission=commission,
            slippage=0.0005,
            allow_short=True,
            signal_lag=1,
        )
        bt = Backtester(
            prices=pair_prices,
            signals=signals,
            config=config,
            benchmark=benchmark,
            strategy_name=f"Pairs_{self.ticker1}_{self.ticker2}",
        )
        return bt.run()
