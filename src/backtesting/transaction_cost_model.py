"""
transaction_cost_model.py
=========================
Realistic transaction cost modelling for ASX equity strategies.

Cost Components
---------------
    Commission      : Brokerage fee per trade (flat rate or tiered)
    Bid-Ask Spread  : Half-spread cost on entry and exit
    Slippage        : Price movement between signal and execution
    Market Impact   : Square-root model for larger orders
    Stamp Duty      : ASX stamp duty where applicable
    Short Costs     : Stock borrow costs for short positions

ASX-Specific Defaults
---------------------
    Online brokers (e.g. CommSec, SelfWealth): ~$10 flat or 0.12%
    Institutional (prime broker): 0.05-0.10% one-way
    Typical ASX bid-ask spread: 0.05-0.20% for liquid stocks
    Short borrow cost: 0.5-3% per annum for ASX stocks

Usage
-----
    from src.backtesting.transaction_cost_model import TransactionCostModel

    tcm = TransactionCostModel(model="percentage", commission=0.001)
    cost = tcm.compute_cost(traded_value=50000, price=45.20)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Cost presets for common ASX broker types
# ─────────────────────────────────────────────

BROKER_PRESETS = {
    "retail_flat": {
        "description": "Retail broker flat fee ($9.95/trade)",
        "model": "flat",
        "flat_fee": 9.95,
        "slippage": 0.001,
        "spread": 0.001,
    },
    "retail_percentage": {
        "description": "Retail broker percentage (0.12%)",
        "model": "percentage",
        "commission": 0.0012,
        "slippage": 0.001,
        "spread": 0.001,
    },
    "institutional": {
        "description": "Institutional / prime broker (0.05%)",
        "model": "percentage",
        "commission": 0.0005,
        "slippage": 0.0003,
        "spread": 0.0003,
    },
    "algorithmic": {
        "description": "Algo trading DMA (0.03%)",
        "model": "percentage",
        "commission": 0.0003,
        "slippage": 0.0002,
        "spread": 0.0002,
    },
    "zero_cost": {
        "description": "No transaction costs (theoretical only)",
        "model": "percentage",
        "commission": 0.0,
        "slippage": 0.0,
        "spread": 0.0,
    },
}


@dataclass
class CostBreakdown:
    """Detailed breakdown of transaction costs for a single trade."""
    gross_value: float = 0.0
    commission: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    market_impact: float = 0.0
    stamp_duty: float = 0.0
    short_borrow: float = 0.0
    total_cost: float = 0.0
    cost_bps: float = 0.0        # basis points
    effective_price: float = 0.0
    direction: str = "buy"

    def summary(self) -> str:
        return (
            f"Trade Cost Breakdown ({self.direction.upper()})\n"
            f"  Gross Value   : ${self.gross_value:,.2f}\n"
            f"  Commission    : ${self.commission:.2f}\n"
            f"  Spread        : ${self.spread_cost:.2f}\n"
            f"  Slippage      : ${self.slippage_cost:.2f}\n"
            f"  Market Impact : ${self.market_impact:.2f}\n"
            f"  Total Cost    : ${self.total_cost:.2f}\n"
            f"  Cost (bps)    : {self.cost_bps:.1f}\n"
        )


# ─────────────────────────────────────────────
#  TransactionCostModel
# ─────────────────────────────────────────────

class TransactionCostModel:
    """
    Modular transaction cost model for ASX equity backtesting.

    Parameters
    ----------
    model          : str    'percentage' | 'flat' | 'tiered' | 'zero'
    commission     : float  One-way commission rate (for percentage model).
    flat_fee       : float  Fixed dollar fee per trade (for flat model).
    spread         : float  Half bid-ask spread (one-way cost).
    slippage       : float  Additional execution slippage (one-way).
    market_impact  : float  Square-root impact coefficient. 0 = disabled.
    stamp_duty     : float  Stamp duty rate (ASX: 0 for equities, applies to CFDs).
    short_borrow_rate: float  Annual cost to borrow stock for short selling.
    min_commission : float  Minimum commission per trade in AUD.
    """

    def __init__(
        self,
        model: str = "percentage",
        commission: float = 0.001,
        flat_fee: float = 9.95,
        spread: float = 0.0005,
        slippage: float = 0.0005,
        market_impact: float = 0.0,
        stamp_duty: float = 0.0,
        short_borrow_rate: float = 0.02,
        min_commission: float = 0.0,
    ) -> None:
        self.model = model
        self.commission = commission
        self.flat_fee = flat_fee
        self.spread = spread
        self.slippage = slippage
        self.market_impact = market_impact
        self.stamp_duty = stamp_duty
        self.short_borrow_rate = short_borrow_rate
        self.min_commission = min_commission

    @classmethod
    def from_preset(cls, preset: str) -> "TransactionCostModel":
        """
        Create a cost model from a named preset.

        Presets: 'retail_flat', 'retail_percentage',
                 'institutional', 'algorithmic', 'zero_cost'
        """
        if preset not in BROKER_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. "
                f"Available: {list(BROKER_PRESETS.keys())}"
            )
        p = BROKER_PRESETS[preset]
        return cls(
            model=p["model"],
            commission=p.get("commission", 0.001),
            flat_fee=p.get("flat_fee", 9.95),
            spread=p.get("spread", 0.0005),
            slippage=p.get("slippage", 0.0005),
        )

    # ──────────────────────────────────────
    #  Single trade cost
    # ──────────────────────────────────────

    def compute_cost(
        self,
        traded_value: float,
        price: float = 1.0,
        direction: str = "buy",
        shares: Optional[float] = None,
        adv: Optional[float] = None,
        holding_days: int = 0,
    ) -> CostBreakdown:
        """
        Compute the full cost breakdown for a single trade.

        Parameters
        ----------
        traded_value : float  Gross dollar value of trade.
        price        : float  Execution price per share.
        direction    : str    'buy' or 'sell'.
        shares       : float  Number of shares (optional, derived from value/price).
        adv          : float  Average daily volume in AUD (for market impact).
        holding_days : int    Days held (for short borrow cost).

        Returns
        -------
        CostBreakdown
        """
        if traded_value <= 0:
            return CostBreakdown(direction=direction)

        # Commission
        if self.model == "percentage":
            comm = max(traded_value * self.commission, self.min_commission)
        elif self.model == "flat":
            comm = self.flat_fee
        elif self.model == "tiered":
            comm = self._tiered_commission(traded_value)
        else:
            comm = 0.0

        # Spread cost (half-spread on entry AND exit)
        spread_cost = traded_value * self.spread

        # Slippage (market orders filled worse than midpoint)
        slippage_cost = traded_value * self.slippage

        # Market impact (square-root model)
        impact = 0.0
        if self.market_impact > 0 and adv is not None and adv > 0:
            participation = traded_value / adv
            impact = self.market_impact * np.sqrt(participation) * traded_value

        # Stamp duty (mostly 0 for ASX equities)
        duty = traded_value * self.stamp_duty if direction == "buy" else 0

        # Short borrow (daily cost on short positions)
        borrow = 0.0
        if direction == "short" and holding_days > 0:
            daily_rate = self.short_borrow_rate / 252
            borrow = traded_value * daily_rate * holding_days

        total = comm + spread_cost + slippage_cost + impact + duty + borrow
        cost_bps = (total / traded_value * 10_000) if traded_value > 0 else 0

        # Effective execution price (worse than quoted)
        if direction in ("buy", "long"):
            eff_price = price * (1 + self.slippage + self.spread)
        else:
            eff_price = price * (1 - self.slippage - self.spread)

        return CostBreakdown(
            gross_value=traded_value,
            commission=comm,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            market_impact=impact,
            stamp_duty=duty,
            short_borrow=borrow,
            total_cost=total,
            cost_bps=cost_bps,
            effective_price=eff_price,
            direction=direction,
        )

    def _tiered_commission(self, value: float) -> float:
        """
        Tiered commission schedule (institutional example).
        Tiers:
            < $10k   : 0.20%
            $10k-$100k: 0.10%
            > $100k  : 0.05%
        """
        if value < 10_000:
            return value * 0.002
        elif value < 100_000:
            return value * 0.001
        else:
            return value * 0.0005

    # ──────────────────────────────────────
    #  Portfolio rebalancing cost
    # ──────────────────────────────────────

    def rebalance_cost(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        nav: float,
        prices: Optional[pd.Series] = None,
        adv: Optional[pd.Series] = None,
    ) -> dict:
        """
        Compute total cost of a portfolio rebalancing.

        Parameters
        ----------
        current_weights : pd.Series   Current portfolio weights.
        target_weights  : pd.Series   Target portfolio weights.
        nav             : float       Current portfolio NAV in AUD.
        prices          : pd.Series   Current prices per ticker.
        adv             : pd.Series   Average daily volume per ticker.

        Returns
        -------
        dict with total_cost, turnover, cost_bps, per_ticker breakdown.
        """
        all_tickers = set(current_weights.index) | set(target_weights.index)
        curr = current_weights.reindex(all_tickers).fillna(0)
        tgt = target_weights.reindex(all_tickers).fillna(0)

        weight_change = (tgt - curr).abs()
        turnover = float(weight_change.sum() / 2)

        total_cost = 0.0
        per_ticker = {}

        for ticker in all_tickers:
            delta_w = abs(float(tgt.get(ticker, 0)) - float(curr.get(ticker, 0)))
            if delta_w < 1e-6:
                continue

            traded_val = nav * delta_w
            price = float(prices.get(ticker, 1.0)) if prices is not None else 1.0
            daily_vol = float(adv.get(ticker, 1e6)) if adv is not None else None
            direction = "buy" if float(tgt.get(ticker, 0)) > float(curr.get(ticker, 0)) else "sell"

            breakdown = self.compute_cost(
                traded_value=traded_val,
                price=price,
                direction=direction,
                adv=daily_vol,
            )
            total_cost += breakdown.total_cost
            per_ticker[ticker] = breakdown.total_cost

        cost_bps = (total_cost / nav * 10_000) if nav > 0 else 0

        return {
            "total_cost": total_cost,
            "turnover": turnover,
            "cost_bps": cost_bps,
            "cost_pct_nav": total_cost / nav if nav > 0 else 0,
            "per_ticker": per_ticker,
        }

    # ──────────────────────────────────────
    #  Annual cost analysis
    # ──────────────────────────────────────

    def annual_cost_analysis(
        self,
        turnover_series: pd.Series,
        nav_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute annualised transaction cost analysis from backtest output.

        Parameters
        ----------
        turnover_series : pd.Series   Daily turnover (fraction of NAV traded).
        nav_series      : pd.Series   Daily NAV.

        Returns
        -------
        pd.DataFrame   Annual cost breakdown.
        """
        common = turnover_series.index.intersection(nav_series.index)
        to = turnover_series.loc[common]
        nav = nav_series.loc[common]

        daily_costs = to * nav * (self.commission + self.spread + self.slippage)
        annual = daily_costs.resample("YE").sum()
        annual_nav = nav.resample("YE").mean()
        annual_to = to.resample("YE").sum()

        df = pd.DataFrame({
            "Annual Cost ($)": annual,
            "Annual Turnover": annual_to,
            "Cost % NAV": (annual / annual_nav * 100).round(3),
            "Cost (bps)": (annual / annual_nav * 10_000).round(1),
        })

        return df

    # ──────────────────────────────────────
    #  Break-even analysis
    # ──────────────────────────────────────

    def breakeven_alpha(
        self,
        annual_turnover: float,
    ) -> float:
        """
        Minimum annual alpha required to cover transaction costs.

        Parameters
        ----------
        annual_turnover : float   Annual one-way turnover (e.g. 2.0 = 200%).

        Returns
        -------
        float  Minimum required annual alpha (as decimal).
        """
        one_way_cost = self.commission + self.spread + self.slippage
        return 2 * one_way_cost * annual_turnover

    def cost_drag_table(
        self,
        turnover_range: list[float] = None,
    ) -> pd.DataFrame:
        """
        Table showing cost drag at various turnover levels.
        Useful for understanding strategy viability before building it.

        Parameters
        ----------
        turnover_range : list   Annual turnover values to tabulate.

        Returns
        -------
        pd.DataFrame   Turnover vs cost drag.
        """
        turnover_range = turnover_range or [0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 20.0]
        one_way = self.commission + self.spread + self.slippage

        rows = []
        for to in turnover_range:
            annual_cost = 2 * one_way * to
            rows.append({
                "Annual Turnover": f"{to:.1f}x",
                "One-Way Cost": f"{one_way:.3%}",
                "Annual Cost Drag": f"{annual_cost:.3%}",
                "Cost (bps/yr)": f"{annual_cost * 10_000:.0f}",
                "Break-even Alpha": f"{annual_cost:.3%}",
            })

        return pd.DataFrame(rows).set_index("Annual Turnover")
