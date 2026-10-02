"""
momentum_factor.py  /  value_factor.py  /  quality_factor.py
=============================================================
Cross-sectional factor signal construction for ASX equities.

All three factors are implemented in this single file for efficiency
and are combined in the FactorLibrary class.

Factor Definitions
------------------
Momentum
    12-1 month price momentum (skip most recent month to avoid reversal)
    1-month short-term reversal
    Risk-adjusted momentum (momentum / volatility)

Value
    12-month price reversal (long-term reversal as cheap proxy)
    Book-to-market proxy (use low P/E proxy via earnings yield)
    52-week low proximity (price vs 52-week low)

Quality
    Earnings stability (low return volatility = stable earnings proxy)
    Profitability proxy (high 3-year cumulative return)
    Low leverage proxy (low price drawdown as financial stress proxy)

Output
------
    Cross-sectional z-scored signals ready for use in:
        - Factor strategy construction
        - ML feature engineering
        - Factor attribution

Usage
-----
    from src.factor_models.momentum_factor import MomentumFactor
    from src.factor_models.value_factor import ValueFactor
    from src.factor_models.quality_factor import QualityFactor

    mom = MomentumFactor(universe_prices)
    signals = mom.compute_signals()
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Helper: cross-sectional z-score
# ─────────────────────────────────────────────

def _cs_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """
    Cross-sectional z-score: standardise each row across assets.
    For each date: z_i = (x_i - mean(x)) / std(x)
    """
    z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
    return z.clip(-clip, clip)


def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank [0, 1] on each date."""
    return df.rank(axis=1, pct=True)


# ═══════════════════════════════════════════════
#  MOMENTUM FACTOR
# ═══════════════════════════════════════════════

class MomentumFactor:
    """
    Cross-sectional momentum signal construction.

    Parameters
    ----------
    universe_prices : pd.DataFrame
        Adjusted close prices. Rows = dates, columns = tickers.
    """

    def __init__(self, universe_prices: pd.DataFrame) -> None:
        self.prices = universe_prices.copy()
        self.returns = self.prices.pct_change().dropna()

    def momentum_12_1(self) -> pd.DataFrame:
        """
        Classic 12-1 month momentum signal.
        Return from 12 months ago to 1 month ago (skip most recent month).
        Higher = stronger recent momentum.

        Jegadeesh & Titman (1993): winners continue to win.
        """
        ret_12m = self.prices.shift(21) / self.prices.shift(252) - 1
        return ret_12m.dropna(how="all")

    def momentum_6_1(self) -> pd.DataFrame:
        """6-1 month momentum (intermediate horizon)."""
        return self.prices.shift(21) / self.prices.shift(126) - 1

    def short_term_reversal(self) -> pd.DataFrame:
        """
        1-month short-term reversal signal.
        Recent losers tend to outperform next month.
        Signal is INVERTED: negative past return = positive signal.
        """
        return -(self.prices / self.prices.shift(21) - 1)

    def risk_adjusted_momentum(self, lookback: int = 252) -> pd.DataFrame:
        """
        Risk-adjusted momentum: momentum / volatility.
        Divides the momentum signal by trailing volatility to
        avoid taking large positions in volatile stocks.
        """
        mom = self.momentum_12_1()
        vol = self.returns.rolling(lookback).std() * np.sqrt(252)
        return (mom / vol.replace(0, np.nan)).dropna(how="all")

    def momentum_acceleration(self) -> pd.DataFrame:
        """
        Momentum acceleration: recent 3M momentum minus 12-1M momentum.
        Positive = momentum is picking up speed.
        """
        mom_3m = self.prices / self.prices.shift(63) - 1
        mom_12_1 = self.momentum_12_1()
        return (mom_3m - mom_12_1).dropna(how="all")

    def compute_signals(self, zscore: bool = True) -> pd.DataFrame:
        """
        Compute all momentum signals and return a composite score.

        Parameters
        ----------
        zscore : bool   Cross-sectionally standardise signals.

        Returns
        -------
        pd.DataFrame  Composite momentum score (rows=dates, cols=tickers).
        """
        signals = {
            "mom_12_1": self.momentum_12_1(),
            "mom_6_1": self.momentum_6_1(),
            "risk_adj_mom": self.risk_adjusted_momentum(),
        }

        if zscore:
            signals = {k: _cs_zscore(v) for k, v in signals.items()}

        # Equally weighted composite
        composite = pd.concat(signals.values()).groupby(level=0).mean()
        if zscore:
            composite = _cs_zscore(composite)

        return composite.rename(columns=lambda c: c)

    def turnover_signal(
        self,
        n_long: int = 10,
        n_short: int = 10,
    ) -> pd.DataFrame:
        """
        Binary long/short signal based on momentum ranking.
        +1 = top n_long tickers (long), -1 = bottom n_short (short), 0 = neutral.
        """
        mom = self.momentum_12_1()
        signal = pd.DataFrame(0, index=mom.index, columns=mom.columns)

        for date in mom.index:
            row = mom.loc[date].dropna()
            if len(row) < n_long + n_short:
                continue
            longs = row.nlargest(n_long).index
            shorts = row.nsmallest(n_short).index
            signal.loc[date, longs] = 1
            signal.loc[date, shorts] = -1

        return signal


# ═══════════════════════════════════════════════
#  VALUE FACTOR
# ═══════════════════════════════════════════════

class ValueFactor:
    """
    Cross-sectional value signal construction.

    Without fundamental data (P/B, P/E) we construct value proxies
    from price data using well-researched empirical relationships.

    Parameters
    ----------
    universe_prices : pd.DataFrame
        Adjusted close prices.
    """

    def __init__(self, universe_prices: pd.DataFrame) -> None:
        self.prices = universe_prices.copy()
        self.returns = self.prices.pct_change().dropna()

    def long_term_reversal(self) -> pd.DataFrame:
        """
        36-12 month long-term reversal.
        Long-term past losers tend to outperform (value proxy).
        De Bondt & Thaler (1985): excessive pessimism creates value opportunities.
        Signal is INVERTED: negative past 36-12M return = positive value signal.
        """
        ret_36_12 = self.prices.shift(252) / self.prices.shift(756) - 1
        return (-ret_36_12).dropna(how="all")

    def distance_from_52w_low(self) -> pd.DataFrame:
        """
        Distance from 52-week low — stocks near 52W low = cheap/value.
        Signal: lower price relative to 52W high = better value.
        George & Hwang (2004): 52W high is a strong anchor for value.
        """
        high_52w = self.prices.rolling(252).max()
        # Stocks near 52W low relative to range = value signal
        low_52w = self.prices.rolling(252).min()
        dist = (self.prices - low_52w) / (high_52w - low_52w).replace(0, np.nan)
        # Invert: closer to 52W low = higher value signal
        return (1 - dist).dropna(how="all")

    def earnings_yield_proxy(self) -> pd.DataFrame:
        """
        Earnings yield proxy using earnings momentum.
        Stocks with lower recent price appreciation but positive longer-term
        returns are likely undervalued (high earnings yield proxy).
        """
        ret_1m = self.prices / self.prices.shift(21) - 1
        ret_12m = self.prices / self.prices.shift(252) - 1
        # Value = positive long-term but negative short-term
        yield_proxy = ret_12m - ret_1m
        return yield_proxy.dropna(how="all")

    def price_to_high_ratio(self) -> pd.DataFrame:
        """
        Price / 52-week high ratio (inverted).
        Stocks trading well below their 52W high may be undervalued.
        """
        high_52w = self.prices.rolling(252).max()
        ratio = self.prices / high_52w.replace(0, np.nan)
        # Invert: lower ratio = more undervalued = better value signal
        return (1 - ratio).dropna(how="all")

    def compute_signals(self, zscore: bool = True) -> pd.DataFrame:
        """
        Compute composite value signal.

        Returns
        -------
        pd.DataFrame  Composite value score (rows=dates, cols=tickers).
        """
        signals = {
            "lt_reversal": self.long_term_reversal(),
            "52w_low": self.distance_from_52w_low(),
            "earnings_yield": self.earnings_yield_proxy(),
        }

        if zscore:
            signals = {k: _cs_zscore(v) for k, v in signals.items()}

        composite = pd.concat(signals.values()).groupby(level=0).mean()
        if zscore:
            composite = _cs_zscore(composite)

        return composite

    def value_vs_growth_spread(self) -> pd.Series:
        """
        Time series of value vs growth spread.
        Measures the premium of value stocks over growth stocks over time.
        """
        val = self.compute_signals(zscore=True)
        returns = self.returns

        common = val.index.intersection(returns.index)
        value_ret = []
        dates = []

        for date in common[1:]:
            prev_date = common[common.get_loc(date) - 1]
            scores = val.loc[prev_date].dropna()
            if len(scores) < 10:
                continue
            n_q = max(int(len(scores) * 0.30), 1)
            value_tickers = scores.nlargest(n_q).index
            growth_tickers = scores.nsmallest(n_q).index

            v_tickers = [t for t in value_tickers if t in returns.columns]
            g_tickers = [t for t in growth_tickers if t in returns.columns]

            if v_tickers and g_tickers and date in returns.index:
                r_v = returns.loc[date, v_tickers].mean()
                r_g = returns.loc[date, g_tickers].mean()
                value_ret.append(r_v - r_g)
                dates.append(date)

        if not dates:
            return pd.Series(dtype=float)

        spread = pd.Series(value_ret, index=dates, name="Value_Growth_Spread")
        return spread


# ═══════════════════════════════════════════════
#  QUALITY FACTOR
# ═══════════════════════════════════════════════

class QualityFactor:
    """
    Cross-sectional quality signal construction.

    Quality = stable, profitable, low-leverage businesses.
    Without balance sheet data, we construct quality proxies from
    return series (earnings stability, profitability, financial health).

    Parameters
    ----------
    universe_prices : pd.DataFrame
        Adjusted close prices.
    """

    def __init__(self, universe_prices: pd.DataFrame) -> None:
        self.prices = universe_prices.copy()
        self.returns = self.prices.pct_change().dropna()

    def earnings_stability(self, window: int = 252) -> pd.DataFrame:
        """
        Earnings stability: inverse of return volatility.
        Low volatility = stable/predictable business = high quality.
        """
        vol = self.returns.rolling(window).std()
        # Invert and normalise: lower vol = higher quality score
        stability = 1 / vol.replace(0, np.nan)
        return stability.dropna(how="all")

    def profitability(self, lookback: int = 756) -> pd.DataFrame:
        """
        Profitability proxy: 3-year cumulative return.
        High long-term returners = profitable businesses.
        """
        ret_3y = self.prices / self.prices.shift(lookback) - 1
        return ret_3y.dropna(how="all")

    def financial_health(self) -> pd.DataFrame:
        """
        Financial health proxy: low maximum drawdown over 12 months.
        Companies with small drawdowns have financial resilience.
        Low drawdown = high quality = positive signal.
        """
        rolling_max = self.prices.rolling(252).max()
        drawdown = (self.prices / rolling_max) - 1   # negative or zero
        # Higher (less negative) = healthier = higher quality
        return drawdown.dropna(how="all")

    def return_consistency(self, window: int = 252) -> pd.DataFrame:
        """
        Return consistency: % of positive days over trailing window.
        Consistent winners = quality businesses.
        """
        positive_days = (
            self.returns.rolling(window)
            .apply(lambda x: (x > 0).mean(), raw=True)
        )
        return positive_days.dropna(how="all")

    def low_beta_quality(
        self,
        market_returns: pd.Series,
        window: int = 252,
    ) -> pd.DataFrame:
        """
        Low-beta quality: defensive stocks with above-average returns.
        Quality firms should generate good returns with below-market beta.
        """
        betas = {}
        for ticker in self.returns.columns:
            common = self.returns[ticker].dropna().index.intersection(
                market_returns.index
            )
            if len(common) < window:
                betas[ticker] = pd.Series(dtype=float)
                continue

            roll_beta = []
            roll_dates = []
            ret_a = self.returns[ticker].loc[common]
            ret_m = market_returns.loc[common]

            for end in range(window, len(common)):
                ra = ret_a.iloc[end - window: end].values
                rm = ret_m.iloc[end - window: end].values
                if len(ra) < 10:
                    continue
                slope, _, _, _, _ = stats.linregress(rm, ra)
                roll_beta.append(slope)
                roll_dates.append(common[end])

            betas[ticker] = pd.Series(roll_beta, index=roll_dates)

        beta_df = pd.DataFrame(betas)
        # Invert: lower beta = better quality signal
        return (-beta_df).dropna(how="all")

    def compute_signals(
        self,
        zscore: bool = True,
        market_returns: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Compute composite quality signal.

        Returns
        -------
        pd.DataFrame  Composite quality score (rows=dates, cols=tickers).
        """
        signals = {
            "stability": self.earnings_stability(),
            "profitability": self.profitability(),
            "health": self.financial_health(),
            "consistency": self.return_consistency(),
        }

        if zscore:
            signals = {k: _cs_zscore(v) for k, v in signals.items()}

        composite = pd.concat(signals.values()).groupby(level=0).mean()
        if zscore:
            composite = _cs_zscore(composite)

        return composite


# ═══════════════════════════════════════════════
#  Combined Factor Library
# ═══════════════════════════════════════════════

class FactorLibrary:
    """
    Combined library that constructs all factor signals in one place.
    Output feeds directly into:
        - FactorStrategy (strategies module)
        - ML feature engineering (machine_learning module)
        - Factor attribution (portfolio_analytics module)

    Parameters
    ----------
    universe_prices : pd.DataFrame   Close prices for ASX universe.
    market_returns  : pd.Series, optional   Benchmark return series.
    """

    def __init__(
        self,
        universe_prices: pd.DataFrame,
        market_returns: Optional[pd.Series] = None,
    ) -> None:
        self.prices = universe_prices.copy()
        self.market = market_returns
        self.momentum = MomentumFactor(universe_prices)
        self.value = ValueFactor(universe_prices)
        self.quality = QualityFactor(universe_prices)

    def compute_all_factors(
        self,
        zscore: bool = True,
    ) -> pd.DataFrame:
        """
        Compute all factor signals and return as a multi-level DataFrame.
        This is the master factor matrix used by the ML pipeline.

        Returns
        -------
        pd.DataFrame  columns = MultiIndex (factor, ticker)
                       or single-level with factor prefix.
        """
        print("Computing factor signals…")

        mom = self.momentum.compute_signals(zscore=zscore)
        val = self.value.compute_signals(zscore=zscore)
        qual = self.quality.compute_signals(zscore=zscore)

        # Align all to common date index
        common = mom.index.intersection(val.index).intersection(qual.index)
        common = common.intersection(mom.columns
                                     .intersection(val.columns)
                                     .intersection(qual.columns)
                                     if False else common)

        # Stack into wide format with factor prefix
        tickers = (
            set(mom.columns)
            .intersection(val.columns)
            .intersection(qual.columns)
        )
        tickers = sorted(tickers)

        records = {}
        for ticker in tickers:
            if ticker in mom.columns:
                records[f"MOM_{ticker}"] = mom[ticker]
            if ticker in val.columns:
                records[f"VAL_{ticker}"] = val[ticker]
            if ticker in qual.columns:
                records[f"QUAL_{ticker}"] = qual[ticker]

        df = pd.DataFrame(records)
        print(f"  [Done] {len(tickers)} tickers × 3 factors.")
        return df

    def factor_returns_by_quintile(
        self,
        factor_name: str = "momentum",
        n_quintiles: int = 5,
        holding_period: int = 21,
    ) -> pd.DataFrame:
        """
        Compute returns for each factor quintile portfolio.
        Used to visualise the factor return spread and validate signal quality.

        Parameters
        ----------
        factor_name   : str   'momentum' | 'value' | 'quality'
        n_quintiles   : int   Number of quantile groups.
        holding_period: int   Holding period in trading days.

        Returns
        -------
        pd.DataFrame  Quintile × average annualised return.
        """
        factor_map = {
            "momentum": self.momentum.compute_signals,
            "value": self.value.compute_signals,
            "quality": self.quality.compute_signals,
        }

        if factor_name not in factor_map:
            raise ValueError(f"Unknown factor: {factor_name}")

        signals = factor_map[factor_name](zscore=False)
        prices = self.prices
        returns = prices.pct_change()

        quintile_returns = {q: [] for q in range(1, n_quintiles + 1)}

        for i, date in enumerate(signals.index[:-holding_period]):
            row = signals.loc[date].dropna()
            if len(row) < n_quintiles * 2:
                continue

            try:
                quintile_labels = pd.qcut(
                    row, n_quintiles,
                    labels=range(1, n_quintiles + 1),
                    duplicates="drop",
                )
            except Exception:
                continue

            future_date = signals.index[i + holding_period]

            for q in range(1, n_quintiles + 1):
                tickers = quintile_labels[quintile_labels == q].index.tolist()
                valid = [t for t in tickers if t in returns.columns]
                if not valid:
                    continue

                period_ret = (
                    (1 + returns.loc[date:future_date, valid]).prod() - 1
                )
                ann_ret = float(
                    (1 + period_ret.mean()) ** (252 / holding_period) - 1
                )
                quintile_returns[q].append(ann_ret)

        results = {
            f"Q{q}": np.mean(v) if v else np.nan
            for q, v in quintile_returns.items()
        }
        spread = (
            results.get(f"Q{n_quintiles}", 0) - results.get("Q1", 0)
            if results else 0
        )
        results["Spread (Q5-Q1)"] = spread

        return pd.DataFrame(
            list(results.items()),
            columns=["Quintile", "Ann. Return"]
        ).set_index("Quintile")

    def factor_correlation_matrix(self) -> pd.DataFrame:
        """
        Correlation matrix between factor signals.
        Low correlation between factors = genuine diversification of risk premia.
        """
        mom = self.momentum.compute_signals().mean(axis=1)
        val = self.value.compute_signals().mean(axis=1)
        qual = self.quality.compute_signals().mean(axis=1)

        df = pd.DataFrame({
            "Momentum": mom,
            "Value": val,
            "Quality": qual,
        }).dropna()

        return df.corr()

    def get_latest_factor_scores(self) -> pd.DataFrame:
        """
        Return the most recent factor scores for all tickers.
        Used by the dashboard to show current factor exposures.
        """
        mom = self.momentum.compute_signals().iloc[-1]
        val = self.value.compute_signals().iloc[-1]
        qual = self.quality.compute_signals().iloc[-1]

        df = pd.DataFrame({
            "Momentum": mom,
            "Value": val,
            "Quality": qual,
        })

        # Composite score (equal weight)
        df["Composite"] = df.mean(axis=1)
        return df.sort_values("Composite", ascending=False)
