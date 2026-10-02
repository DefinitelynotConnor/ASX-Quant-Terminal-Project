"""
performance_attribution.py
===========================
Brinson-Hood-Beebower (BHB) performance attribution model.

Decomposes active portfolio return into:
    • Asset Allocation Effect    - over/underweight sectors vs benchmark
    • Security Selection Effect  - stock picking within sectors
    • Interaction Effect         - combined allocation and selection
    • Factor Attribution         - return contribution from factor exposures

The BHB model is the industry-standard attribution framework used by
institutional asset managers, pension funds, and consultants.

Reference
---------
Brinson, Hood, Beebower (1986): "Determinants of Portfolio Performance"
Financial Analysts Journal.
"""

from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  BHB Attribution Engine
# ─────────────────────────────────────────────

@dataclass
class AttributionResult:
    """Container for BHB attribution decomposition."""
    total_active_return: float = 0.0
    allocation_effect: float = 0.0
    selection_effect: float = 0.0
    interaction_effect: float = 0.0
    attribution_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    cumulative_attribution: pd.DataFrame = field(default_factory=pd.DataFrame)


class PerformanceAttributor:
    """
    Brinson-Hood-Beebower performance attribution.

    Parameters
    ----------
    portfolio_weights  : dict or pd.DataFrame
        {ticker or sector: weight} for the active portfolio.
        If pd.DataFrame, index = dates, columns = tickers (time-varying weights).
    benchmark_weights  : dict or pd.DataFrame
        {ticker or sector: weight} for the benchmark.
    asset_returns      : pd.DataFrame
        Daily return series, columns = tickers.
    sector_map         : dict
        {ticker: sector_name}
    benchmark_returns  : pd.Series, optional
        Aggregate benchmark return series.
    """

    def __init__(
        self,
        portfolio_weights: dict | pd.DataFrame,
        benchmark_weights: dict | pd.DataFrame,
        asset_returns: pd.DataFrame,
        sector_map: Optional[dict[str, str]] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> None:
        self.port_w = portfolio_weights
        self.bench_w = benchmark_weights
        self.asset_returns = asset_returns.copy()
        self.sector_map = sector_map or {}
        self.benchmark_returns = benchmark_returns

        # Resolve static weights
        if isinstance(portfolio_weights, dict):
            self._port_w_static = self._normalise(portfolio_weights)
        else:
            self._port_w_static = None

        if isinstance(benchmark_weights, dict):
            self._bench_w_static = self._normalise(benchmark_weights)
        else:
            self._bench_w_static = None

    @staticmethod
    def _normalise(w: dict) -> dict:
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    # ──────────────────────────────────────
    #  Asset-Level Attribution
    # ──────────────────────────────────────

    def asset_attribution(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        BHB attribution at the individual asset level.

        For each asset:
            Allocation Effect  = (wp_i - wb_i) × (rb_i - R_b)
            Selection Effect   = wb_i × (rp_i - rb_i)
            Interaction Effect = (wp_i - wb_i) × (rp_i - rb_i)

        Where:
            wp_i = portfolio weight of asset i
            wb_i = benchmark weight of asset i
            rp_i = portfolio asset return
            rb_i = benchmark asset return
            R_b  = total benchmark return
        """
        ret = self.asset_returns.copy()
        if start:
            ret = ret[ret.index >= start]
        if end:
            ret = ret[ret.index <= end]

        if self._port_w_static is None or self._bench_w_static is None:
            raise ValueError("Static weight dicts are required for asset_attribution.")

        # Compute period total returns for each asset
        period_returns = (1 + ret).prod() - 1

        all_assets = set(self._port_w_static) | set(self._bench_w_static)
        wp = {a: self._port_w_static.get(a, 0.0) for a in all_assets}
        wb = {a: self._bench_w_static.get(a, 0.0) for a in all_assets}
        rp = {a: period_returns.get(a, 0.0) for a in all_assets}
        rb = {a: period_returns.get(a, 0.0) for a in all_assets}

        # Benchmark total return
        R_b = sum(wb[a] * rb[a] for a in all_assets)
        R_p = sum(wp[a] * rp[a] for a in all_assets)

        rows = []
        for asset in sorted(all_assets):
            alloc = (wp[asset] - wb[asset]) * (rb[asset] - R_b)
            select = wb[asset] * (rp[asset] - rb[asset])
            interact = (wp[asset] - wb[asset]) * (rp[asset] - rb[asset])
            total = alloc + select + interact

            rows.append({
                "Asset": asset,
                "Portfolio Weight": wp[asset],
                "Benchmark Weight": wb[asset],
                "Active Weight": wp[asset] - wb[asset],
                "Portfolio Return": rp[asset],
                "Benchmark Return": rb[asset],
                "Allocation Effect": alloc,
                "Selection Effect": select,
                "Interaction Effect": interact,
                "Total Attribution": total,
            })

        df = pd.DataFrame(rows).sort_values("Total Attribution", ascending=False)

        # Totals row
        totals = {
            "Asset": "TOTAL",
            "Portfolio Weight": sum(wp.values()),
            "Benchmark Weight": sum(wb.values()),
            "Active Weight": R_p - R_b,
            "Portfolio Return": R_p,
            "Benchmark Return": R_b,
            "Allocation Effect": df["Allocation Effect"].sum(),
            "Selection Effect": df["Selection Effect"].sum(),
            "Interaction Effect": df["Interaction Effect"].sum(),
            "Total Attribution": df["Total Attribution"].sum(),
        }
        df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

        return df

    # ──────────────────────────────────────
    #  Sector-Level Attribution
    # ──────────────────────────────────────

    def sector_attribution(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> AttributionResult:
        """
        BHB attribution aggregated to GICS sector level.

        Returns
        -------
        AttributionResult with:
            attribution_table   : DataFrame with one row per sector
            allocation_effect   : Total allocation contribution
            selection_effect    : Total selection contribution
            interaction_effect  : Total interaction contribution
            total_active_return : Portfolio return - Benchmark return
        """
        ret = self.asset_returns.copy()
        if start:
            ret = ret[ret.index >= start]
        if end:
            ret = ret[ret.index <= end]

        if self._port_w_static is None or self._bench_w_static is None:
            raise ValueError("Static weight dicts are required.")

        period_returns = (1 + ret).prod() - 1

        # Aggregate to sectors
        all_assets = set(self._port_w_static) | set(self._bench_w_static)
        sectors = set()
        for a in all_assets:
            s = self.sector_map.get(a, "Unknown")
            sectors.add(s)

        sector_port_w: dict[str, float] = {}
        sector_bench_w: dict[str, float] = {}
        sector_port_ret: dict[str, float] = {}
        sector_bench_ret: dict[str, float] = {}

        for sector in sectors:
            assets_in_sector = [a for a in all_assets if self.sector_map.get(a) == sector]

            # Portfolio sector weight and return
            pw_assets = {a: self._port_w_static.get(a, 0.0) for a in assets_in_sector}
            pw_sum = sum(pw_assets.values())
            sector_port_w[sector] = pw_sum
            if pw_sum > 0:
                sector_port_ret[sector] = sum(
                    (pw_assets[a] / pw_sum) * period_returns.get(a, 0.0)
                    for a in assets_in_sector
                )
            else:
                sector_port_ret[sector] = 0.0

            # Benchmark sector weight and return
            bw_assets = {a: self._bench_w_static.get(a, 0.0) for a in assets_in_sector}
            bw_sum = sum(bw_assets.values())
            sector_bench_w[sector] = bw_sum
            if bw_sum > 0:
                sector_bench_ret[sector] = sum(
                    (bw_assets[a] / bw_sum) * period_returns.get(a, 0.0)
                    for a in assets_in_sector
                )
            else:
                sector_bench_ret[sector] = 0.0

        R_b = sum(sector_bench_w[s] * sector_bench_ret[s] for s in sectors)
        R_p = sum(sector_port_w[s] * sector_port_ret[s] for s in sectors)

        rows = []
        for sector in sorted(sectors):
            wp = sector_port_w[sector]
            wb = sector_bench_w[sector]
            rp = sector_port_ret[sector]
            rb = sector_bench_ret[sector]

            alloc = (wp - wb) * (rb - R_b)
            select = wb * (rp - rb)
            interact = (wp - wb) * (rp - rb)

            rows.append({
                "Sector": sector,
                "Port Weight": wp,
                "Bench Weight": wb,
                "Active Weight": wp - wb,
                "Port Return": rp,
                "Bench Return": rb,
                "Allocation Effect": alloc,
                "Selection Effect": select,
                "Interaction Effect": interact,
                "Total": alloc + select + interact,
            })

        df = pd.DataFrame(rows).sort_values("Total", ascending=False)

        result = AttributionResult(
            total_active_return=R_p - R_b,
            allocation_effect=float(df["Allocation Effect"].sum()),
            selection_effect=float(df["Selection Effect"].sum()),
            interaction_effect=float(df["Interaction Effect"].sum()),
            attribution_table=df,
        )
        return result

    # ──────────────────────────────────────
    #  Rolling Attribution
    # ──────────────────────────────────────

    def rolling_attribution(
        self,
        window: int = 63,
    ) -> pd.DataFrame:
        """
        Rolling window BHB attribution (uses static weights).
        Returns a time series of allocation, selection, and interaction effects.

        Parameters
        ----------
        window : int   Rolling window in trading days.

        Returns
        -------
        pd.DataFrame  index=date, columns=[allocation, selection, interaction, total]
        """
        if self._port_w_static is None or self._bench_w_static is None:
            return pd.DataFrame()

        all_assets = set(self._port_w_static) | set(self._bench_w_static)
        wp = {a: self._port_w_static.get(a, 0.0) for a in all_assets}
        wb = {a: self._bench_w_static.get(a, 0.0) for a in all_assets}

        ret = self.asset_returns[[a for a in all_assets if a in self.asset_returns.columns]]

        records = []
        dates = ret.index[window:]

        for end_date in dates:
            loc = ret.index.get_loc(end_date)
            window_ret = ret.iloc[max(0, loc - window + 1): loc + 1]
            period_ret = (1 + window_ret).prod() - 1

            rp_dict = {a: float(period_ret.get(a, 0.0)) for a in all_assets}
            rb_dict = rp_dict.copy()   # same price data for both

            R_b = sum(wb[a] * rb_dict[a] for a in all_assets)
            alloc = sum((wp[a] - wb[a]) * (rb_dict[a] - R_b) for a in all_assets)
            select = sum(wb[a] * (rp_dict[a] - rb_dict[a]) for a in all_assets)
            interact = sum((wp[a] - wb[a]) * (rp_dict[a] - rb_dict[a]) for a in all_assets)

            records.append({
                "Date": end_date,
                "Allocation": alloc,
                "Selection": select,
                "Interaction": interact,
                "Total Active": alloc + select + interact,
            })

        return pd.DataFrame(records).set_index("Date")

    # ──────────────────────────────────────
    #  Factor Attribution
    # ──────────────────────────────────────

    def factor_attribution(
        self,
        factor_returns: pd.DataFrame,
        factor_exposures: pd.Series,
    ) -> pd.DataFrame:
        """
        Decompose portfolio returns into factor contributions.

        Factor Return = Exposure × Factor Return Series
        Residual (Alpha) = Total Return - Σ Factor Contributions

        Parameters
        ----------
        factor_returns   : pd.DataFrame  Daily factor return series.
                                          Columns = factor names.
        factor_exposures : pd.Series     Portfolio-level factor exposures
                                          (betas from OLS regression).

        Returns
        -------
        pd.DataFrame  One row per factor + alpha, with:
            cumulative_contribution, annualised_contribution, pct_of_total
        """
        common = self.asset_returns.index.intersection(factor_returns.index)
        if len(common) < 10:
            return pd.DataFrame()

        # Portfolio return series
        if self._port_w_static is not None:
            tickers = [t for t in self._port_w_static if t in self.asset_returns.columns]
            w = np.array([self._port_w_static[t] for t in tickers])
            w /= w.sum()
            port_ret = pd.Series(
                self.asset_returns[tickers].loc[common].values @ w,
                index=common,
                name="Portfolio",
            )
        else:
            return pd.DataFrame()

        factors = factor_returns.loc[common]
        contributions = {}

        for factor in factors.columns:
            beta = float(factor_exposures.get(factor, 0.0))
            contribution = beta * factors[factor]
            contributions[factor] = contribution

        contrib_df = pd.DataFrame(contributions)
        total_factor = contrib_df.sum(axis=1)
        alpha_series = port_ret - total_factor

        # Summary table
        rows = []
        total_return = float((1 + port_ret).prod() - 1)
        n_years = len(port_ret) / 252

        for factor in factors.columns:
            cum_contrib = float((1 + contributions[factor]).prod() - 1)
            ann_contrib = float((1 + cum_contrib) ** (1 / n_years) - 1) if n_years > 0 else 0
            rows.append({
                "Factor": factor,
                "Exposure (Beta)": float(factor_exposures.get(factor, 0.0)),
                "Cumulative Contribution": cum_contrib,
                "Annualised Contribution": ann_contrib,
                "% of Total Return": cum_contrib / total_return if total_return != 0 else 0,
            })

        # Alpha (unexplained)
        cum_alpha = float((1 + alpha_series).prod() - 1)
        ann_alpha = float((1 + cum_alpha) ** (1 / n_years) - 1) if n_years > 0 else 0
        rows.append({
            "Factor": "Alpha (Residual)",
            "Exposure (Beta)": 1.0,
            "Cumulative Contribution": cum_alpha,
            "Annualised Contribution": ann_alpha,
            "% of Total Return": cum_alpha / total_return if total_return != 0 else 0,
        })

        return pd.DataFrame(rows).set_index("Factor")

    # ──────────────────────────────────────
    #  Period Attribution Summary
    # ──────────────────────────────────────

    def period_attribution_summary(
        self,
        periods: dict[str, tuple[str, str]],
    ) -> pd.DataFrame:
        """
        Compute attribution across multiple sub-periods.

        Parameters
        ----------
        periods : dict   {label: (start_date, end_date)}
            e.g. {'2022 Bear': ('2022-01-01', '2022-12-31')}

        Returns
        -------
        pd.DataFrame  rows = periods, columns = attribution effects.
        """
        records = []
        for label, (start, end) in periods.items():
            try:
                result = self.sector_attribution(start=start, end=end)
                records.append({
                    "Period": label,
                    "Active Return": result.total_active_return,
                    "Allocation": result.allocation_effect,
                    "Selection": result.selection_effect,
                    "Interaction": result.interaction_effect,
                })
            except Exception:
                pass

        return pd.DataFrame(records).set_index("Period")
