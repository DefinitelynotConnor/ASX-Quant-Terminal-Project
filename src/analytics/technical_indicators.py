"""
technical_indicators.py
=======================
Comprehensive technical indicator library for ASX equity research.

All indicators are implemented as vectorised NumPy/Pandas operations
for maximum performance across large universes.

Indicators Implemented
----------------------
Trend       : SMA, EMA, DEMA, TEMA, WMA, HMA, VWAP
Momentum    : RSI, MACD, ROC, Williams %R, CMO, PPO
Volatility  : Bollinger Bands, ATR, Keltner Channels, Donchian
Volume      : OBV, VWAP, CMF, MFI, Volume SMA
Oscillators : Stochastic, CCI, Aroon, DPO, Ultimate Oscillator
Trend Str   : ADX, Ichimoku Cloud, Parabolic SAR
Signals     : Golden/Death Cross, Breakout signals

Usage
-----
    from src.analytics.technical_indicators import TechnicalIndicators

    ti = TechnicalIndicators(price_df)
    signals = ti.compute_all()
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class TechnicalIndicators:
    """
    Compute technical indicators from OHLCV price data.

    Parameters
    ----------
    prices : pd.DataFrame
        Must contain columns: Open, High, Low, Close, Volume.
        Index must be a DatetimeIndex.
    ticker : str
        Ticker label for display purposes.

    Example
    -------
    >>> ti = TechnicalIndicators(ohlcv_df, ticker="BHP.AX")
    >>> rsi = ti.rsi(period=14)
    >>> macd_line, signal, hist = ti.macd()
    >>> all_signals = ti.compute_all()
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        ticker: str = "Unknown",
    ) -> None:
        self.ticker = ticker
        self.df = prices.copy()

        # Standardise column names
        col_map = {c.lower(): c for c in self.df.columns}
        self.close = self._get_col(["Close", "close", "Adj Close"])
        self.high = self._get_col(["High", "high"])
        self.low = self._get_col(["Low", "low"])
        self.open = self._get_col(["Open", "open"])
        self.volume = self._get_col(["Volume", "volume"])

    def _get_col(self, candidates: list[str]) -> pd.Series:
        for c in candidates:
            if c in self.df.columns:
                return self.df[c].copy()
        return pd.Series(dtype=float, index=self.df.index)

    # ──────────────────────────────────────
    #  Moving Averages
    # ──────────────────────────────────────

    def sma(self, period: int = 20) -> pd.Series:
        """Simple Moving Average."""
        return self.close.rolling(period).mean().rename(f"SMA_{period}")

    def ema(self, period: int = 20) -> pd.Series:
        """Exponential Moving Average."""
        return self.close.ewm(span=period, adjust=False).mean().rename(f"EMA_{period}")

    def dema(self, period: int = 20) -> pd.Series:
        """
        Double EMA — reduces lag vs standard EMA.
        DEMA = 2 × EMA - EMA(EMA)
        """
        e = self.close.ewm(span=period, adjust=False).mean()
        ee = e.ewm(span=period, adjust=False).mean()
        return (2 * e - ee).rename(f"DEMA_{period}")

    def tema(self, period: int = 20) -> pd.Series:
        """
        Triple EMA — further reduces lag.
        TEMA = 3×EMA - 3×EMA(EMA) + EMA(EMA(EMA))
        """
        e = self.close.ewm(span=period, adjust=False).mean()
        ee = e.ewm(span=period, adjust=False).mean()
        eee = ee.ewm(span=period, adjust=False).mean()
        return (3 * e - 3 * ee + eee).rename(f"TEMA_{period}")

    def wma(self, period: int = 20) -> pd.Series:
        """
        Weighted Moving Average — linearly weighted, more weight to recent prices.
        """
        weights = np.arange(1, period + 1)

        def _wma(x):
            if len(x) < period:
                return np.nan
            return np.dot(x, weights) / weights.sum()

        return (
            self.close.rolling(period)
            .apply(_wma, raw=True)
            .rename(f"WMA_{period}")
        )

    def hma(self, period: int = 20) -> pd.Series:
        """
        Hull Moving Average — smooth and responsive.
        HMA = WMA(2×WMA(n/2) - WMA(n), √n)
        """
        half = max(int(period / 2), 1)
        sqrt_n = max(int(np.sqrt(period)), 1)

        wma_half = self.wma(half)
        wma_full = self.wma(period)
        diff = 2 * wma_half - wma_full

        weights = np.arange(1, sqrt_n + 1)

        def _wma(x):
            if len(x) < sqrt_n or np.isnan(x).any():
                return np.nan
            return np.dot(x, weights) / weights.sum()

        return (
            diff.rolling(sqrt_n)
            .apply(_wma, raw=True)
            .rename(f"HMA_{period}")
        )

    def vwap(self) -> pd.Series:
        """
        Volume Weighted Average Price.
        VWAP = Σ(Typical Price × Volume) / Σ(Volume)
        Resets daily.
        """
        typical = (self.high + self.low + self.close) / 3
        cum_vol = self.volume.groupby(self.df.index.date).cumsum()
        cum_tp_vol = (typical * self.volume).groupby(self.df.index.date).cumsum()
        return (cum_tp_vol / cum_vol).rename("VWAP")

    # ──────────────────────────────────────
    #  Momentum Indicators
    # ──────────────────────────────────────

    def rsi(self, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.
        RSI = 100 - 100 / (1 + RS)
        RS = avg_gain / avg_loss over period.

        Overbought > 70, Oversold < 30.
        """
        delta = self.close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.rename(f"RSI_{period}")

    def macd(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD — Moving Average Convergence Divergence.

        Returns
        -------
        macd_line   : EMA(fast) - EMA(slow)
        signal_line : EMA(macd_line, signal)
        histogram   : macd_line - signal_line
        """
        ema_fast = self.close.ewm(span=fast, adjust=False).mean()
        ema_slow = self.close.ewm(span=slow, adjust=False).mean()
        macd_line = (ema_fast - ema_slow).rename(f"MACD_{fast}_{slow}")
        signal_line = macd_line.ewm(span=signal, adjust=False).mean().rename(
            f"MACD_Signal_{signal}"
        )
        histogram = (macd_line - signal_line).rename("MACD_Hist")
        return macd_line, signal_line, histogram

    def roc(self, period: int = 12) -> pd.Series:
        """
        Rate of Change: percentage change over N periods.
        ROC = (Close / Close[N]) - 1
        """
        return (
            (self.close / self.close.shift(period) - 1) * 100
        ).rename(f"ROC_{period}")

    def williams_r(self, period: int = 14) -> pd.Series:
        """
        Williams %R: measures overbought/oversold on -100 to 0 scale.
        Above -20 = overbought, below -80 = oversold.
        """
        highest_high = self.high.rolling(period).max()
        lowest_low = self.low.rolling(period).min()
        wr = ((highest_high - self.close) / (highest_high - lowest_low)) * -100
        return wr.rename(f"Williams_R_{period}")

    def cmo(self, period: int = 14) -> pd.Series:
        """
        Chande Momentum Oscillator.
        CMO = 100 × (sum_up - sum_down) / (sum_up + sum_down)
        Range: -100 to +100.
        """
        delta = self.close.diff()
        up = delta.clip(lower=0).rolling(period).sum()
        down = (-delta.clip(upper=0)).rolling(period).sum()
        cmo = 100 * (up - down) / (up + down).replace(0, np.nan)
        return cmo.rename(f"CMO_{period}")

    def ppo(self, fast: int = 12, slow: int = 26) -> pd.Series:
        """
        Percentage Price Oscillator — MACD as a percentage.
        PPO = (EMA_fast - EMA_slow) / EMA_slow × 100
        """
        ema_fast = self.close.ewm(span=fast, adjust=False).mean()
        ema_slow = self.close.ewm(span=slow, adjust=False).mean()
        return ((ema_fast - ema_slow) / ema_slow * 100).rename(f"PPO_{fast}_{slow}")

    def momentum(self, period: int = 10) -> pd.Series:
        """Raw price momentum: Close - Close[N]."""
        return (self.close - self.close.shift(period)).rename(f"MOM_{period}")

    # ──────────────────────────────────────
    #  Volatility Indicators
    # ──────────────────────────────────────

    def atr(self, period: int = 14) -> pd.Series:
        """
        Average True Range — measures market volatility.
        TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        ATR = EMA(TR, period)
        """
        prev_close = self.close.shift(1)
        tr = pd.concat([
            self.high - self.low,
            (self.high - prev_close).abs(),
            (self.low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean().rename(f"ATR_{period}")

    def bollinger_bands(
        self,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.

        Returns
        -------
        upper  : SMA + std_dev × σ
        middle : SMA
        lower  : SMA - std_dev × σ
        """
        middle = self.close.rolling(period).mean()
        std = self.close.rolling(period).std()
        upper = (middle + std_dev * std).rename(f"BB_Upper_{period}")
        lower = (middle - std_dev * std).rename(f"BB_Lower_{period}")
        middle = middle.rename(f"BB_Mid_{period}")
        return upper, middle, lower

    def bollinger_pct_b(self, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """
        Bollinger %B: position of price within the bands.
        %B = (Close - Lower) / (Upper - Lower)
        0 = at lower band, 1 = at upper band, >1 = above upper.
        """
        upper, middle, lower = self.bollinger_bands(period, std_dev)
        band_width = upper - lower
        return ((self.close - lower) / band_width.replace(0, np.nan)).rename(f"BB_PctB_{period}")

    def bollinger_bandwidth(self, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """
        Bollinger Bandwidth: (Upper - Lower) / Middle.
        Measures band width — expands during volatile periods.
        """
        upper, middle, lower = self.bollinger_bands(period, std_dev)
        return ((upper - lower) / middle).rename(f"BB_BW_{period}")

    def keltner_channels(
        self,
        ema_period: int = 20,
        atr_period: int = 14,
        multiplier: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels — ATR-based volatility envelope.

        Returns
        -------
        upper, middle (EMA), lower
        """
        middle = self.close.ewm(span=ema_period, adjust=False).mean()
        atr = self.atr(atr_period)
        upper = (middle + multiplier * atr).rename("KC_Upper")
        lower = (middle - multiplier * atr).rename("KC_Lower")
        return upper, middle.rename("KC_Mid"), lower

    def donchian_channels(
        self, period: int = 20
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Donchian Channels — highest high and lowest low over N periods.
        Widely used in trend-following and breakout strategies.
        """
        upper = self.high.rolling(period).max().rename(f"DC_Upper_{period}")
        lower = self.low.rolling(period).min().rename(f"DC_Lower_{period}")
        middle = ((upper + lower) / 2).rename(f"DC_Mid_{period}")
        return upper, middle, lower

    def realised_volatility(self, period: int = 21) -> pd.Series:
        """
        Annualised realised volatility from daily log returns.
        σ_realised = std(log returns) × √252
        """
        log_ret = np.log(self.close / self.close.shift(1))
        return (log_ret.rolling(period).std() * np.sqrt(252)).rename(
            f"RealVol_{period}"
        )

    # ──────────────────────────────────────
    #  Volume Indicators
    # ──────────────────────────────────────

    def obv(self) -> pd.Series:
        """
        On-Balance Volume — running total of volume in the direction of price.
        OBV rises when close > previous close, falls when close < previous close.
        """
        direction = np.sign(self.close.diff()).fillna(0)
        return (direction * self.volume).cumsum().rename("OBV")

    def cmf(self, period: int = 20) -> pd.Series:
        """
        Chaikin Money Flow — measures buying/selling pressure.
        CMF = Σ(MFV) / Σ(Volume) over period.
        MFV = ((Close - Low) - (High - Close)) / (High - Low) × Volume
        """
        high_low = (self.high - self.low).replace(0, np.nan)
        mfv = ((self.close - self.low) - (self.high - self.close)) / high_low
        mfv = mfv * self.volume
        return (
            mfv.rolling(period).sum() / self.volume.rolling(period).sum()
        ).rename(f"CMF_{period}")

    def mfi(self, period: int = 14) -> pd.Series:
        """
        Money Flow Index — RSI applied to money flow (volume-weighted).
        Overbought > 80, Oversold < 20.
        """
        typical = (self.high + self.low + self.close) / 3
        raw_mf = typical * self.volume

        delta = typical.diff()
        pos_mf = raw_mf.where(delta > 0, 0)
        neg_mf = raw_mf.where(delta < 0, 0)

        pos_sum = pos_mf.rolling(period).sum()
        neg_sum = neg_mf.rolling(period).sum().replace(0, np.nan)

        mfr = pos_sum / neg_sum
        return (100 - 100 / (1 + mfr)).rename(f"MFI_{period}")

    def volume_sma(self, period: int = 20) -> pd.Series:
        """Volume simple moving average."""
        return self.volume.rolling(period).mean().rename(f"Vol_SMA_{period}")

    def volume_ratio(self, period: int = 20) -> pd.Series:
        """
        Volume ratio: current volume vs average volume.
        > 1 = above average volume (confirms moves).
        """
        return (self.volume / self.volume_sma(period)).rename(f"Vol_Ratio_{period}")

    # ──────────────────────────────────────
    #  Oscillators
    # ──────────────────────────────────────

    def stochastic(
        self,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.
        %K = (Close - Lowest Low) / (Highest High - Lowest Low) × 100
        %D = SMA(%K, d_period)

        Overbought > 80, Oversold < 20.
        """
        lowest = self.low.rolling(k_period).min()
        highest = self.high.rolling(k_period).max()
        k = (
            (self.close - lowest) / (highest - lowest).replace(0, np.nan) * 100
        ).rename(f"Stoch_K_{k_period}")
        d = k.rolling(d_period).mean().rename(f"Stoch_D_{d_period}")
        return k, d

    def cci(self, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index.
        CCI = (Typical Price - SMA) / (0.015 × Mean Deviation)
        Overbought > 100, Oversold < -100.
        """
        typical = (self.high + self.low + self.close) / 3
        sma = typical.rolling(period).mean()
        mean_dev = typical.rolling(period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        return ((typical - sma) / (0.015 * mean_dev)).rename(f"CCI_{period}")

    def aroon(self, period: int = 25) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Aroon Indicator — identifies trend direction and strength.
        Aroon Up   = (period - days since highest high) / period × 100
        Aroon Down = (period - days since lowest low) / period × 100
        Aroon Osc  = Aroon Up - Aroon Down
        """
        def _days_since_max(x):
            return float(period - np.argmax(x[::-1]))

        def _days_since_min(x):
            return float(period - np.argmin(x[::-1]))

        aroon_up = (
            self.high.rolling(period + 1)
            .apply(_days_since_max, raw=True) / period * 100
        ).rename(f"Aroon_Up_{period}")

        aroon_down = (
            self.low.rolling(period + 1)
            .apply(_days_since_min, raw=True) / period * 100
        ).rename(f"Aroon_Down_{period}")

        aroon_osc = (aroon_up - aroon_down).rename(f"Aroon_Osc_{period}")
        return aroon_up, aroon_down, aroon_osc

    def ultimate_oscillator(
        self,
        p1: int = 7,
        p2: int = 14,
        p3: int = 28,
    ) -> pd.Series:
        """
        Ultimate Oscillator — combines three timeframes.
        Reduces false divergences common in single-period oscillators.
        """
        prev_close = self.close.shift(1)
        bp = self.close - pd.concat([self.low, prev_close], axis=1).min(axis=1)
        tr = pd.concat([
            self.high - self.low,
            (self.high - prev_close).abs(),
            (self.low - prev_close).abs(),
        ], axis=1).max(axis=1)

        def _avg(bp_s, tr_s, p):
            return bp_s.rolling(p).sum() / tr_s.rolling(p).sum().replace(0, np.nan)

        avg1 = _avg(bp, tr, p1)
        avg2 = _avg(bp, tr, p2)
        avg3 = _avg(bp, tr, p3)

        return (
            100 * (4 * avg1 + 2 * avg2 + avg3) / 7
        ).rename(f"UltOsc_{p1}_{p2}_{p3}")

    # ──────────────────────────────────────
    #  Trend Strength
    # ──────────────────────────────────────

    def adx(self, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index — measures trend strength (not direction).
        ADX > 25 = strong trend, < 20 = weak/no trend.

        Returns
        -------
        adx, plus_di, minus_di
        """
        prev_high = self.high.shift(1)
        prev_low = self.low.shift(1)
        prev_close = self.close.shift(1)

        plus_dm = (self.high - prev_high).clip(lower=0)
        minus_dm = (prev_low - self.low).clip(lower=0)

        # Nullify when opposite direction is larger
        plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

        tr = pd.concat([
            self.high - self.low,
            (self.high - prev_close).abs(),
            (self.low - prev_close).abs(),
        ], axis=1).max(axis=1)

        # Wilder smoothing
        atr_w = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = (
            100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w
        ).rename(f"DI+_{period}")
        minus_di = (
            100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w
        ).rename(f"DI-_{period}")

        dx = (
            100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        )
        adx = dx.ewm(alpha=1 / period, adjust=False).mean().rename(f"ADX_{period}")

        return adx, plus_di, minus_di

    def parabolic_sar(
        self,
        step: float = 0.02,
        max_step: float = 0.20,
    ) -> pd.Series:
        """
        Parabolic SAR — trailing stop and reverse indicator.
        Below price = uptrend (long signal), above price = downtrend (short signal).
        """
        close = self.close.values
        high = self.high.values
        low = self.low.values
        n = len(close)

        sar = np.full(n, np.nan)
        bull = True
        ep = low[0]
        af = step
        sar[0] = high[0]

        for i in range(1, n):
            prev_sar = sar[i - 1]
            if bull:
                sar[i] = prev_sar + af * (ep - prev_sar)
                sar[i] = min(sar[i], low[i - 1], low[max(0, i - 2)])
                if low[i] < sar[i]:
                    bull = False
                    sar[i] = ep
                    ep = low[i]
                    af = step
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + step, max_step)
            else:
                sar[i] = prev_sar + af * (ep - prev_sar)
                sar[i] = max(sar[i], high[i - 1], high[max(0, i - 2)])
                if high[i] > sar[i]:
                    bull = True
                    sar[i] = ep
                    ep = high[i]
                    af = step
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + step, max_step)

        return pd.Series(sar, index=self.df.index, name="ParSAR")

    def ichimoku(
        self,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b: int = 52,
    ) -> pd.DataFrame:
        """
        Ichimoku Cloud.

        Returns
        -------
        pd.DataFrame with columns:
            Tenkan_Sen   : Conversion line (short-term trend)
            Kijun_Sen    : Base line (medium-term trend)
            Senkou_A     : Leading span A (cloud boundary)
            Senkou_B     : Leading span B (cloud boundary)
            Chikou_Span  : Lagging span
        """
        def mid(h, l, p):
            return (h.rolling(p).max() + l.rolling(p).min()) / 2

        t = mid(self.high, self.low, tenkan).rename("Tenkan_Sen")
        k = mid(self.high, self.low, kijun).rename("Kijun_Sen")
        sa = ((t + k) / 2).shift(kijun).rename("Senkou_A")
        sb = mid(self.high, self.low, senkou_b).shift(kijun).rename("Senkou_B")
        chikou = self.close.shift(-kijun).rename("Chikou_Span")

        return pd.concat([t, k, sa, sb, chikou], axis=1)

    # ──────────────────────────────────────
    #  Cross Signals
    # ──────────────────────────────────────

    def golden_death_cross(
        self,
        fast: int = 50,
        slow: int = 200,
    ) -> pd.DataFrame:
        """
        Golden Cross (bullish) and Death Cross (bearish) signals.

        Golden Cross : SMA(fast) crosses above SMA(slow)
        Death Cross  : SMA(fast) crosses below SMA(slow)

        Returns
        -------
        pd.DataFrame with columns:
            SMA_fast, SMA_slow, signal (1=golden, -1=death, 0=none)
        """
        fast_ma = self.sma(fast)
        slow_ma = self.sma(slow)
        above = (fast_ma > slow_ma).astype(int)
        signal = above.diff()
        signal = signal.map({1: 1, -1: -1}).fillna(0)

        return pd.DataFrame({
            f"SMA_{fast}": fast_ma,
            f"SMA_{slow}": slow_ma,
            "Cross_Signal": signal,
        })

    def breakout_signal(
        self,
        period: int = 20,
    ) -> pd.Series:
        """
        Donchian breakout signal.
        +1 = price breaks above N-day high (long signal)
        -1 = price breaks below N-day low (short signal)
         0 = no breakout
        """
        upper, _, lower = self.donchian_channels(period)
        signal = pd.Series(0, index=self.df.index, name=f"Breakout_{period}")
        signal[self.close > upper.shift(1)] = 1
        signal[self.close < lower.shift(1)] = -1
        return signal

    def ma_trend_signal(
        self,
        short: int = 20,
        long: int = 50,
    ) -> pd.Series:
        """
        Simple MA trend signal.
        +1 = short MA above long MA (uptrend)
        -1 = short MA below long MA (downtrend)
        """
        s = self.sma(short)
        l = self.sma(long)
        return (
            pd.Series(
                np.where(s > l, 1, -1),
                index=self.df.index,
                name=f"MA_Trend_{short}_{long}",
            )
        )

    def rsi_signal(
        self,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
    ) -> pd.Series:
        """
        RSI mean-reversion signal.
        -1 = overbought (potential sell)
        +1 = oversold (potential buy)
         0 = neutral
        """
        rsi = self.rsi(period)
        signal = pd.Series(0, index=self.df.index, name=f"RSI_Signal_{period}")
        signal[rsi < oversold] = 1
        signal[rsi > overbought] = -1
        return signal

    # ──────────────────────────────────────
    #  Composite Signal
    # ──────────────────────────────────────

    def compute_all(self) -> pd.DataFrame:
        """
        Compute a comprehensive set of indicators and return as a DataFrame.
        This is the primary method used by the dashboard and ML pipeline.

        Returns
        -------
        pd.DataFrame  All indicators as columns, DatetimeIndex rows.
        """
        results = {}

        # Trend
        for p in [10, 20, 50, 100, 200]:
            results[f"SMA_{p}"] = self.sma(p)
            results[f"EMA_{p}"] = self.ema(p)

        results["DEMA_20"] = self.dema(20)
        results["HMA_20"] = self.hma(20)

        # Momentum
        for p in [7, 14, 21]:
            results[f"RSI_{p}"] = self.rsi(p)
        results["RSI_Signal"] = self.rsi_signal()

        macd_l, macd_s, macd_h = self.macd()
        results["MACD"] = macd_l
        results["MACD_Signal"] = macd_s
        results["MACD_Hist"] = macd_h

        for p in [5, 10, 20]:
            results[f"ROC_{p}"] = self.roc(p)

        results["Williams_R"] = self.williams_r(14)
        results["CMO"] = self.cmo(14)
        results["PPO"] = self.ppo()

        # Volatility
        for p in [14, 21]:
            results[f"ATR_{p}"] = self.atr(p)
            results[f"RealVol_{p}"] = self.realised_volatility(p)

        bb_u, bb_m, bb_l = self.bollinger_bands()
        results["BB_Upper"] = bb_u
        results["BB_Mid"] = bb_m
        results["BB_Lower"] = bb_l
        results["BB_PctB"] = self.bollinger_pct_b()
        results["BB_BW"] = self.bollinger_bandwidth()

        dc_u, dc_m, dc_l = self.donchian_channels(20)
        results["DC_Upper"] = dc_u
        results["DC_Mid"] = dc_m
        results["DC_Lower"] = dc_l

        # Volume
        results["OBV"] = self.obv()
        results["CMF"] = self.cmf()
        results["MFI"] = self.mfi()
        results["Vol_Ratio"] = self.volume_ratio()

        # Oscillators
        stoch_k, stoch_d = self.stochastic()
        results["Stoch_K"] = stoch_k
        results["Stoch_D"] = stoch_d
        results["CCI"] = self.cci()
        results["Ultimate_Osc"] = self.ultimate_oscillator()

        # Trend strength
        adx_v, plus_di, minus_di = self.adx()
        results["ADX"] = adx_v
        results["DI_Plus"] = plus_di
        results["DI_Minus"] = minus_di
        results["ParSAR"] = self.parabolic_sar()

        # Ichimoku
        ichi = self.ichimoku()
        for col in ichi.columns:
            results[col] = ichi[col]

        # Cross signals
        crosses = self.golden_death_cross()
        results["Cross_Signal"] = crosses["Cross_Signal"]
        results["Breakout_20"] = self.breakout_signal(20)
        results["MA_Trend"] = self.ma_trend_signal()

        df = pd.DataFrame(results, index=self.df.index)
        return df


# ─────────────────────────────────────────────
#  Universe-level indicator computation
# ─────────────────────────────────────────────

def compute_universe_indicators(
    price_universe: pd.DataFrame,
    ohlcv_data: Optional[dict[str, pd.DataFrame]] = None,
    indicators: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute a specific indicator across a universe of tickers.

    Parameters
    ----------
    price_universe : pd.DataFrame
        Close prices, columns = tickers.
    ohlcv_data     : dict, optional
        {ticker: ohlcv_df} for volume/high/low indicators.
        If None, only close-based indicators are computed.
    indicators     : list, optional
        Specific indicator names to compute.
        Defaults to ['RSI_14', 'MACD', 'BB_PctB', 'ATR_14', 'ROC_20'].

    Returns
    -------
    dict  {indicator_name: pd.DataFrame (rows=dates, cols=tickers)}
    """
    default_indicators = ["RSI_14", "ROC_20", "BB_PctB", "ATR_14", "MOM_10"]
    indicators = indicators or default_indicators

    results = {ind: {} for ind in indicators}

    for ticker in price_universe.columns:
        # Build minimal OHLCV DataFrame
        if ohlcv_data and ticker in ohlcv_data:
            ohlcv = ohlcv_data[ticker]
        else:
            close = price_universe[ticker].dropna()
            ohlcv = pd.DataFrame({
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Volume": pd.Series(1e6, index=close.index),
            })

        if len(ohlcv) < 30:
            continue

        ti = TechnicalIndicators(ohlcv, ticker=ticker)

        for ind in indicators:
            try:
                if ind == "RSI_14":
                    results[ind][ticker] = ti.rsi(14)
                elif ind == "ROC_20":
                    results[ind][ticker] = ti.roc(20)
                elif ind == "BB_PctB":
                    results[ind][ticker] = ti.bollinger_pct_b()
                elif ind == "ATR_14":
                    results[ind][ticker] = ti.atr(14)
                elif ind == "MOM_10":
                    results[ind][ticker] = ti.momentum(10)
                elif ind == "MACD":
                    macd_l, _, _ = ti.macd()
                    results[ind][ticker] = macd_l
                elif ind == "ADX_14":
                    adx_v, _, _ = ti.adx(14)
                    results[ind][ticker] = adx_v
                elif ind == "OBV":
                    results[ind][ticker] = ti.obv()
            except Exception:
                pass

    return {
        ind: pd.DataFrame(data)
        for ind, data in results.items()
        if data
    }
