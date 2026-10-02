"""
portfolio_dashboard.py
======================
Streamlit portfolio analytics dashboard.

Integrates all analytics modules into a single interactive interface:
    • Portfolio definition (tickers + weights)
    • Performance metrics table + score card
    • Equity curve + drawdown chart
    • Risk decomposition (MCTR / PCTR) charts
    • Sector exposure charts
    • Factor exposure analysis
    • Performance attribution (BHB)
    • Stress test results
    • Monthly return heatmap
    • Rolling metrics (Sharpe, Beta, Volatility)

Usage
-----
Run from project root:
    streamlit run src/portfolio_analytics/portfolio_dashboard.py

Or integrate into your main terminal_app.py:
    from src.portfolio_analytics.portfolio_dashboard import run_portfolio_dashboard
    run_portfolio_dashboard()
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Optional streamlit import (for standalone use)
# ─────────────────────────────────────────────
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


# ─────────────────────────────────────────────
#  Chart builders (pure Plotly, no Streamlit dependency)
# ─────────────────────────────────────────────

CHART_THEME = "plotly_dark"
COLOUR_PORTFOLIO = "#00D4FF"
COLOUR_BENCHMARK = "#FF6B35"
COLOUR_POSITIVE = "#00C853"
COLOUR_NEGATIVE = "#FF1744"
COLOUR_NEUTRAL = "#90A4AE"


def build_equity_curve_chart(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    portfolio_name: str = "Portfolio",
    benchmark_name: str = "ASX 200",
    log_scale: bool = False,
) -> go.Figure:
    """
    Interactive equity curve chart with optional benchmark overlay.
    Normalised to $100 initial NAV.
    """
    nav_port = (1 + portfolio_returns).cumprod() * 100
    nav_port.name = portfolio_name

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nav_port.index,
        y=nav_port.values,
        name=portfolio_name,
        line=dict(color=COLOUR_PORTFOLIO, width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>NAV: $%{y:.2f}<extra></extra>",
    ))

    if benchmark_returns is not None:
        common = nav_port.index.intersection(benchmark_returns.index)
        nav_bench = (1 + benchmark_returns.loc[common]).cumprod() * 100
        fig.add_trace(go.Scatter(
            x=nav_bench.index,
            y=nav_bench.values,
            name=benchmark_name,
            line=dict(color=COLOUR_BENCHMARK, width=1.5, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>NAV: $%{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Portfolio Equity Curve", font=dict(size=18)),
        yaxis=dict(
            title="NAV ($100 start)",
            type="log" if log_scale else "linear",
        ),
        xaxis=dict(title="Date"),
        template=CHART_THEME,
        height=480,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def build_drawdown_chart(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    portfolio_name: str = "Portfolio",
) -> go.Figure:
    """
    Interactive drawdown time series chart with fill.
    """
    def _dd(ret):
        nav = (1 + ret).cumprod()
        peak = nav.cummax()
        return (nav / peak) - 1

    dd_port = _dd(portfolio_returns)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dd_port.index,
        y=dd_port.values * 100,
        name=portfolio_name,
        line=dict(color=COLOUR_NEGATIVE, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255, 23, 68, 0.15)",
        hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ))

    if benchmark_returns is not None:
        common = dd_port.index.intersection(benchmark_returns.index)
        dd_bench = _dd(benchmark_returns.loc[common])
        fig.add_trace(go.Scatter(
            x=dd_bench.index,
            y=dd_bench.values * 100,
            name="Benchmark",
            line=dict(color=COLOUR_BENCHMARK, width=1, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Portfolio Drawdown", font=dict(size=18)),
        yaxis=dict(title="Drawdown (%)", ticksuffix="%"),
        xaxis=dict(title="Date"),
        template=CHART_THEME,
        height=360,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def build_risk_attribution_chart(risk_attribution_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart: % risk contribution per asset (PCTR).
    Sorted descending.
    """
    df = risk_attribution_df[risk_attribution_df["Ticker"] != "TOTAL"].copy()
    df = df.sort_values("PCTR", ascending=True)

    colours = [
        COLOUR_NEGATIVE if v > 0.15 else COLOUR_POSITIVE if v < 0.05 else COLOUR_NEUTRAL
        for v in df["PCTR"]
    ]

    fig = go.Figure(go.Bar(
        x=df["PCTR"] * 100,
        y=df["Ticker"],
        orientation="h",
        marker_color=colours,
        text=[f"{v:.1%}" for v in df["PCTR"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Risk Contribution: %{x:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="% Risk Contribution by Asset (PCTR)", font=dict(size=18)),
        xaxis=dict(title="% of Total Portfolio Risk", ticksuffix="%"),
        template=CHART_THEME,
        height=max(300, 40 * len(df)),
        showlegend=False,
    )
    return fig


def build_weight_vs_risk_chart(risk_attribution_df: pd.DataFrame) -> go.Figure:
    """
    Scatter chart: portfolio weight vs risk contribution.
    Assets above diagonal = risk-heavy; below diagonal = risk-light.
    """
    df = risk_attribution_df[risk_attribution_df["Ticker"] != "TOTAL"].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Weight"] * 100,
        y=df["PCTR"] * 100,
        mode="markers+text",
        text=df["Ticker"],
        textposition="top center",
        marker=dict(
            size=12,
            color=df["PCTR"] / df["Weight"].replace(0, np.nan),
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Risk/Weight"),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>Weight: %{x:.1f}%<br>Risk %: %{y:.1f}%<extra></extra>"
        ),
    ))

    # Diagonal = equal risk / equal weight
    max_val = max(df["Weight"].max(), df["PCTR"].max()) * 100 * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        name="Equal Risk/Weight",
        line=dict(color=COLOUR_NEUTRAL, dash="dash", width=1),
    ))

    fig.update_layout(
        title=dict(text="Weight vs Risk Contribution", font=dict(size=18)),
        xaxis=dict(title="Portfolio Weight (%)", ticksuffix="%"),
        yaxis=dict(title="% Risk Contribution", ticksuffix="%"),
        template=CHART_THEME,
        height=500,
    )
    return fig


def build_sector_exposure_chart(sector_df: pd.DataFrame) -> go.Figure:
    """
    Donut chart: sector weights in the portfolio.
    """
    fig = go.Figure(go.Pie(
        labels=sector_df["Sector"],
        values=sector_df["Weight"],
        hole=0.45,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>",
        marker=dict(
            colors=px.colors.qualitative.Set3[:len(sector_df)],
        ),
    ))

    fig.update_layout(
        title=dict(text="Portfolio Sector Exposure", font=dict(size=18)),
        template=CHART_THEME,
        height=460,
        legend=dict(orientation="v"),
    )
    return fig


def build_attribution_waterfall(
    allocation_effect: float,
    selection_effect: float,
    interaction_effect: float,
    benchmark_return: float,
    portfolio_return: float,
) -> go.Figure:
    """
    Waterfall chart decomposing active return into BHB components.
    """
    labels = [
        "Benchmark Return",
        "Allocation Effect",
        "Selection Effect",
        "Interaction Effect",
        "Portfolio Return",
    ]
    values = [
        benchmark_return,
        allocation_effect,
        selection_effect,
        interaction_effect,
        portfolio_return,
    ]
    measures = ["absolute", "relative", "relative", "relative", "total"]
    text = [f"{v:.2%}" for v in values]

    fig = go.Figure(go.Waterfall(
        name="Attribution",
        orientation="v",
        measure=measures,
        x=labels,
        y=[v * 100 for v in values],
        text=text,
        textposition="outside",
        connector=dict(line=dict(color="rgb(63, 63, 63)")),
        increasing=dict(marker=dict(color=COLOUR_POSITIVE)),
        decreasing=dict(marker=dict(color=COLOUR_NEGATIVE)),
        totals=dict(marker=dict(color=COLOUR_PORTFOLIO)),
    ))

    fig.update_layout(
        title=dict(text="BHB Performance Attribution Waterfall", font=dict(size=18)),
        yaxis=dict(title="Return (%)", ticksuffix="%"),
        template=CHART_THEME,
        height=480,
    )
    return fig


def build_monthly_returns_heatmap(monthly_df: pd.DataFrame) -> go.Figure:
    """
    Calendar heatmap of monthly returns: rows = year, columns = month.
    """
    months = [c for c in monthly_df.columns if c != "Annual"]
    data = monthly_df[months].values * 100
    years = monthly_df.index.tolist()

    fig = go.Figure(go.Heatmap(
        z=data,
        x=months,
        y=[str(y) for y in years],
        colorscale=[
            [0.0, "#FF1744"],
            [0.4, "#FF6B35"],
            [0.5, "#1A1A2E"],
            [0.6, "#00C853"],
            [1.0, "#00BFA5"],
        ],
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in data],
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="Return %", ticksuffix="%"),
    ))

    fig.update_layout(
        title=dict(text="Monthly Returns Heatmap", font=dict(size=18)),
        template=CHART_THEME,
        height=max(300, 55 * len(years)),
        xaxis=dict(title="Month"),
        yaxis=dict(title="Year", autorange="reversed"),
    )
    return fig


def build_rolling_metrics_chart(
    rolling_sharpe: pd.Series,
    rolling_vol: pd.Series,
    rolling_beta: Optional[pd.Series] = None,
) -> go.Figure:
    """
    Multi-panel chart: rolling Sharpe, volatility, and beta.
    """
    n_rows = 3 if rolling_beta is not None else 2
    subplot_titles = ["Rolling Sharpe Ratio (63D)", "Rolling Volatility (21D)"]
    if rolling_beta is not None:
        subplot_titles.append("Rolling Beta (63D)")

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index, y=rolling_sharpe.values,
        name="Sharpe", line=dict(color=COLOUR_PORTFOLIO, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color=COLOUR_NEUTRAL,
                  annotation_text="Sharpe=1", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=rolling_vol.index, y=rolling_vol.values * 100,
        name="Volatility", line=dict(color=COLOUR_BENCHMARK, width=1.5),
        fill="tozeroy", fillcolor="rgba(255, 107, 53, 0.1)",
        hovertemplate="%{x|%Y-%m-%d}<br>Vol: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    if rolling_beta is not None:
        fig.add_trace(go.Scatter(
            x=rolling_beta.index, y=rolling_beta.values,
            name="Beta", line=dict(color=COLOUR_POSITIVE, width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>Beta: %{y:.2f}<extra></extra>",
        ), row=3, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color=COLOUR_NEUTRAL,
                      annotation_text="Beta=1", row=3, col=1)
        fig.update_yaxes(title_text="Beta", row=3, col=1)

    fig.update_yaxes(title_text="Sharpe", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (%)", ticksuffix="%", row=2, col=1)
    fig.update_layout(
        template=CHART_THEME,
        height=600,
        showlegend=False,
        title=dict(text="Rolling Risk Metrics", font=dict(size=18)),
        hovermode="x unified",
    )
    return fig


def build_stress_test_chart(stress_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of stress scenario impacts on portfolio.
    """
    df = stress_df.reset_index().sort_values("Portfolio Impact")

    colours = [
        COLOUR_NEGATIVE if v < -0.10 else COLOUR_NEUTRAL
        for v in df["Portfolio Impact"]
    ]

    fig = go.Figure(go.Bar(
        x=df["Portfolio Impact"] * 100,
        y=df["Scenario"],
        orientation="h",
        marker_color=colours,
        text=[f"{v:.1%}" for v in df["Portfolio Impact"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Impact: %{x:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Portfolio Stress Test Scenarios", font=dict(size=18)),
        xaxis=dict(title="Portfolio Impact (%)", ticksuffix="%"),
        template=CHART_THEME,
        height=max(350, 50 * len(df)),
        showlegend=False,
    )
    return fig


def build_factor_exposure_chart(factor_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart: factor loadings with significance shading.
    """
    if factor_df.empty:
        return go.Figure()

    df = factor_df[factor_df.index != "Alpha"].copy()
    colours = [
        COLOUR_POSITIVE if row["Loading"] > 0 else COLOUR_NEGATIVE
        for _, row in df.iterrows()
    ]
    opacity = [
        1.0 if row.get("Significant", True) else 0.4
        for _, row in df.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x=df.index.tolist(),
        y=df["Loading"].values,
        marker=dict(color=colours, opacity=opacity),
        text=[f"{v:.3f}" for v in df["Loading"].values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Loading: %{y:.4f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_color=COLOUR_NEUTRAL, line_width=1)

    fig.update_layout(
        title=dict(text="Factor Exposure (OLS Loadings)", font=dict(size=18)),
        yaxis=dict(title="Factor Loading (Beta)"),
        xaxis=dict(title="Factor"),
        template=CHART_THEME,
        height=420,
        showlegend=False,
    )
    return fig


def build_pca_chart(pca_result: dict) -> go.Figure:
    """
    Bar chart of PCA explained variance + portfolio risk from each component.
    """
    n = len(pca_result["eigenvalues"])
    labels = [f"PC{i+1}" for i in range(n)]
    evr = pca_result["explained_variance_ratio"] * 100
    cumev = pca_result["cumulative_explained_variance"] * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=labels, y=evr,
        name="Explained Variance",
        marker_color=COLOUR_PORTFOLIO,
        hovertemplate="<b>%{x}</b><br>Explained: %{y:.1f}%<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=labels, y=cumev,
        name="Cumulative",
        mode="lines+markers",
        line=dict(color=COLOUR_BENCHMARK, width=2),
        hovertemplate="Cumulative: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        title=dict(text="PCA Explained Variance", font=dict(size=18)),
        template=CHART_THEME,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Variance Explained (%)", ticksuffix="%")
    fig.update_yaxes(title_text="Cumulative (%)", ticksuffix="%", secondary_y=True)

    return fig


def build_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """
    Portfolio correlation matrix heatmap.
    """
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1, zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="Correlation"),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Asset Correlation Matrix", font=dict(size=18)),
        template=CHART_THEME,
        height=500,
    )
    return fig




# ─────────────────────────────────────────────
#  Streamlit Dashboard
# ─────────────────────────────────────────────

def run_portfolio_dashboard(
    preloaded_prices=None,
    preloaded_benchmark=None,
    preloaded_rfr=None,
) -> None:
    """
    Main Streamlit portfolio analytics dashboard.
    Call this from your terminal_app.py to embed the portfolio analytics page.
    """
    import numpy as np
    import pandas as pd

    if not HAS_STREAMLIT:
        raise ImportError("Streamlit is required. Install: pip install streamlit")

    import yfinance as yf
    from src.portfolio_analytics.portfolio_metrics import PortfolioMetrics
    from src.portfolio_analytics.exposure_analysis import ExposureAnalyser
    from src.portfolio_analytics.performance_attribution import PerformanceAttributor
    from src.portfolio_analytics.risk_decomposition import RiskDecomposer

    # Determine if running embedded inside terminal_app or standalone
    embedded = preloaded_prices is not None

    st.title("📊 Portfolio Analytics Dashboard")
    st.markdown("*Institutional-grade portfolio diagnostics for ASX securities.*")

    # ─────────────────────────────────────────────
    #  CONFIGURATION — sidebar (standalone) or inline (embedded)
    # ─────────────────────────────────────────────

    if not embedded:
        # ── STANDALONE: sidebar config ──
        with st.sidebar:
            st.header("⚙️ Portfolio Configuration")

            tickers_input = st.text_area(
                "ASX Tickers (one per line)",
                value="BHP.AX\nCBA.AX\nCSL.AX\nWES.AX\nMQG.AX",
                height=120,
                key="pd_tickers_input",
            )
            tickers = [t.strip().upper() for t in tickers_input.split("\n") if t.strip()]

            weight_mode = st.radio(
                "Weight Method",
                ["Equal Weight", "Custom Weights"],
                index=0,
                key="pd_weight_mode",
            )

            if weight_mode == "Custom Weights":
                st.markdown("**Enter weights (must sum to 1):**")
                custom_weights = {}
                for t in tickers:
                    w = st.number_input(
                        f"{t}",
                        min_value=0.0, max_value=1.0,
                        value=1 / len(tickers), step=0.01,
                        key=f"pd_w_{t}",
                    )
                    custom_weights[t] = w
                weights = custom_weights
            else:
                weights = {t: 1 / len(tickers) for t in tickers}

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=pd.Timestamp("2019-01-01"),
                    key="pd_start_date",
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=pd.Timestamp.today(),
                    key="pd_end_date",
                )

            benchmark_ticker = st.selectbox(
                "Benchmark",
                ["^AXJO", "^AXKO", "^AORD", "^GSPC"],
                index=0,
                key="pd_benchmark_ticker",
            )

            risk_free_rate = st.number_input(
                "Risk-Free Rate (annual)",
                min_value=0.0, max_value=0.20,
                value=0.0435, step=0.0025,
                format="%.4f",
                key="pd_risk_free_rate",
            )

            cov_method = st.selectbox(
                "Covariance Method",
                ["ledoit_wolf", "sample", "ewma"],
                index=0,
                key="pd_cov_method",
            )

            run_btn = st.button(
                "▶ Run Analysis",
                type="primary",
                use_container_width=True,
                key="pd_run_btn",
            )

        if not run_btn:
            st.info("Configure your portfolio in the sidebar and click **▶ Run Analysis**.")
            st.markdown("""
            **Portfolio Analytics covers:**
            - 📈 Performance metrics (Sharpe, Sortino, CAGR, max drawdown …)
            - ⚠️ Risk decomposition (MCTR, PCTR, CVaR decomposition)
            - 🏭 Sector & factor exposure analysis
            - 📉 BHB performance attribution
            - 🌡️ Stress testing (GFC, COVID, rate shocks)
            - 🔬 PCA risk decomposition
            - 📅 Monthly return heatmap
            """)
            return

    else:
        # ── EMBEDDED: use preloaded data, show inline config ──
        price_data = preloaded_prices.copy()
        benchmark_series = preloaded_benchmark
        benchmark_ticker = "Benchmark"
        risk_free_rate = float(preloaded_rfr) if preloaded_rfr is not None else 0.0435
        start_date = price_data.index[0].date()
        end_date = price_data.index[-1].date()
        tickers = list(price_data.columns)
        available = tickers

        st.markdown("### ⚙️ Portfolio Configuration")
        col1, col2, col3 = st.columns(3)

        with col1:
            weight_mode = st.radio(
                "Weight Method",
                ["Equal Weight", "Custom Weights"],
                index=0,
                key="pd_embedded_weight_mode",
                horizontal=True,
            )
        with col2:
            cov_method = st.selectbox(
                "Covariance Method",
                ["ledoit_wolf", "sample", "ewma"],
                index=0,
                key="pd_embedded_cov",
            )
        with col3:
            risk_free_rate = st.number_input(
                "Risk-Free Rate",
                min_value=0.0, max_value=0.20,
                value=risk_free_rate,
                step=0.0025, format="%.4f",
                key="pd_embedded_rfr",
            )

        if weight_mode == "Custom Weights":
            st.markdown("**Custom Weights** (must sum to 1)")
            cols = st.columns(min(len(tickers), 6))
            custom_weights = {}
            for i, t in enumerate(tickers):
                with cols[i % len(cols)]:
                    custom_weights[t] = st.number_input(
                        t, min_value=0.0, max_value=1.0,
                        value=round(1 / len(tickers), 4),
                        step=0.01, key=f"pd_emb_w_{t}",
                    )
            weights = custom_weights
        else:
            weights = {t: 1 / len(tickers) for t in tickers}

        weights_available = {t: weights[t] for t in tickers}
        total_w = sum(weights_available.values())
        weights_available = {t: w / total_w for t, w in weights_available.items()}

        run_btn = st.button(
            "▶ Run Portfolio Analysis",
            type="primary",
            key="pd_embedded_run_btn",
        )

    # ─────────────────────────────────────────────
    #  TOP-LEVEL TABS: Guide is always available
    # ─────────────────────────────────────────────
    tab_guide_early, tab_run = st.tabs(["📖 Guide", "📊 Analytics"])

    # ─────────────────────────────────────────────
    #  GUIDE TAB — always visible, no analysis needed
    # ─────────────────────────────────────────────
    with tab_guide_early:
        st.subheader("📖 Platform Guide")
        st.markdown("""
        *A plain-English explanation of every feature, chart, and metric across the platform.
        Read this once and everything else will make sense.*
        """)

        st.info("""
        **How to use this guide:** Each section matches a tab in the main terminal.
        Click any expander to read about it. You don't need to read everything at once —
        just look up what you need when you need it.
        """)

        # ══════════════════════════════════════
        st.markdown("---")
        st.markdown("## 📊 Market Overview")
        st.markdown("*The starting point — shows how your selected stocks have performed and how they relate to each other.*")
        # ══════════════════════════════════════

        with st.expander("Normalised Price Performance Chart", expanded=False):
            st.markdown("""
            **What it shows:** All selected stocks rebased to 100 on the start date, so you can compare performance on the same scale regardless of their actual share prices.

            **How to read it:** A stock at 150 has returned 50% since the start date. A stock at 80 has lost 20%. The dotted orange line is the ASX 200 benchmark.

            **What to look for:**
            - Stocks consistently above the benchmark line = outperformers
            - Stocks that all move together at the same time = highly correlated, less diversified
            - A stock that zigs when others zag = a diversifier worth keeping
            """)

        with st.expander("Returns Summary Table", expanded=False):
            st.markdown("""
            **What it shows:** Four key statistics for each stock you've loaded.

            - **CAGR** — Compound Annual Growth Rate. The average yearly return, smoothed across the full period. 10% CAGR means the stock doubled roughly every 7 years.
            - **Ann. Vol** — Annualised Volatility. How much the stock price jumps around in a year. 20% vol means the stock could swing ±20% from its expected path in a typical year.
            - **Sharpe** — Risk-adjusted return. Return earned per unit of risk taken. Higher is better. A Sharpe above 1 is considered good.
            - **Max DD** — Maximum Drawdown. The worst peak-to-trough fall the stock ever had in your date range. Shows you the worst-case experience for a buy-and-hold investor.
            """)

        with st.expander("Correlation Matrix", expanded=False):
            st.markdown("""
            **What it shows:** How closely each pair of stocks moves together. Values range from -1 to +1.

            - **+1.0** — The two stocks move in perfect lockstep. Owning both adds no diversification.
            - **0.0** — The stocks are completely independent. Ideal for diversification.
            - **-1.0** — The stocks move in opposite directions. Rare but excellent for hedging.

            **What to look for:** A good portfolio has a mix — you want some correlation (otherwise stocks in the same sector would diverge randomly) but not everything above 0.8 (that's just buying the same stock twice).

            **Colour coding:** Dark red = high positive correlation. Dark blue = negative correlation. Light colours near zero = low correlation.

            **Practical tip:** If you see a block of stocks all correlated above 0.85 with each other, you effectively only have one bet there, not several.
            """)

        # ══════════════════════════════════════
        st.markdown("---")
        st.markdown("## 📉 Technical Analysis")
        st.markdown("*Analyses a single selected stock using price-based signals. These do not analyse the portfolio as a whole.*")
        # ══════════════════════════════════════

        with st.expander("Price + Moving Averages Chart", expanded=False):
            st.markdown("""
            **What it shows:** The stock's price history with three moving averages overlaid.

            - **SMA 20 (yellow)** — 20-day Simple Moving Average. Tracks short-term trend.
            - **SMA 50 (orange)** — 50-day SMA. Tracks medium-term trend.
            - **SMA 200 (red)** — 200-day SMA. The most widely watched long-term trend line.

            **How moving averages work:** A moving average smooths out daily noise to show the underlying direction. When price is above its 200-day MA, the long-term trend is up. When below, the trend is down.

            **Golden Cross / Death Cross:**
            - Golden Cross = SMA 50 crosses above SMA 200 → classic bullish signal, price often continues rising
            - Death Cross = SMA 50 crosses below SMA 200 → classic bearish signal

            **Note:** Moving averages are lagging indicators — they react to price after it has already moved.
            """)

        with st.expander("RSI — Relative Strength Index", expanded=False):
            st.markdown("""
            **What it measures:** Momentum of a single stock — how fast and how far it has been rising or falling recently.

            **Scale:** 0 to 100.
            - **Above 70** — Overbought. The stock has risen quickly and may be due for a pullback.
            - **Below 30** — Oversold. The stock has fallen quickly and may bounce.
            - **Between 30 and 70** — Neutral territory, no strong signal.

            **How to use it:** RSI doesn't predict with certainty — an overbought stock can stay overbought for months in a strong bull market. It's most useful as a confirmation tool alongside other signals.

            **The red dashed line at 70 and green dashed line at 30** mark these thresholds on the chart.
            """)

        with st.expander("MACD — Moving Average Convergence Divergence", expanded=False):
            st.markdown("""
            **What it measures:** Momentum and trend direction for a single stock by comparing two EMAs.

            **Three components:**
            - **MACD Line (blue)** — Difference between 12-day and 26-day Exponential Moving Averages. When positive, short-term momentum is stronger than long-term.
            - **Signal Line (orange dashed)** — 9-day EMA of the MACD line. A smoother version of the MACD.
            - **Histogram (bars)** — The gap between MACD and Signal. Green bars = MACD above signal (bullish). Red bars = MACD below signal (bearish).

            **Key signals:**
            - MACD line crosses above signal line → bullish momentum building
            - MACD line crosses below signal line → bearish momentum building
            - Histogram shrinking → momentum slowing, potential reversal
            """)

        with st.expander("Bollinger Bands", expanded=False):
            st.markdown("""
            **What it shows:** A volatility envelope around a stock's price.

            - **Middle Band** — 20-day Simple Moving Average of the stock price
            - **Upper Band** — Middle band + 2 standard deviations
            - **Lower Band** — Middle band - 2 standard deviations

            **How to read it:** About 95% of daily prices should sit inside the bands under normal conditions.
            - Price touches the upper band → statistically stretched to the upside, potential mean reversion
            - Price touches the lower band → statistically stretched to the downside, potential bounce
            - Bands squeezing together → low volatility period, a large move is often building
            - Bands widening → high volatility, strong trending market

            **Important:** Bands measure whether the price is stretched relative to recent history. They don't tell you the direction of the next move.
            """)

        # ══════════════════════════════════════
        st.markdown("---")
        st.markdown("## 🔬 Factor Analysis")
        st.markdown("*Analyses individual stocks and the portfolio to understand what drives their returns.*")
        # ══════════════════════════════════════

        with st.expander("CAPM — Capital Asset Pricing Model", expanded=False):
            st.markdown("""
            **What it does:** Measures each stock's relationship to the market. Run on individual stocks.

            **Key outputs:**
            - **Alpha** — Return the stock generated above what its market exposure alone would predict. Positive alpha = the stock added value beyond just riding the market up. Annualised figure.
            - **Beta** — How much the stock moves per 1% move in the ASX 200. Beta of 1.5 means the stock rises 1.5% when the market rises 1%, and falls 1.5% when the market falls 1%.
            - **R-Squared** — What percentage of the stock's daily moves are explained by market moves. High R² = the stock mostly just follows the index. Low R² = the stock has its own independent story.
            - **Expected Return** — What CAPM theory says the stock should return, given its beta and the market risk premium. Compare to Actual Return to see if the stock is over or underperforming theory.
            - **Abnormal Return** — Actual Return minus Expected Return. Positive = the stock beat what theory predicted. This is Jensen's Alpha.

            **The Security Market Line chart:** Each dot is a stock. Dots above the line outperformed CAPM predictions. Dots below underperformed. The line itself shows what return theory expects for each beta level.
            """)

        with st.expander("Fama-French 3-Factor Model", expanded=False):
            st.markdown("""
            **What it does:** A more sophisticated version of CAPM that adds two extra factors. Run on individual stocks.

            The three factors are:
            - **MKT (Market)** — Same as CAPM beta. Exposure to overall market moves.
            - **SMB (Small Minus Big)** — Size factor. Positive loading means the stock behaves like small-cap stocks (which have historically outperformed large-caps over long periods).
            - **HML (High Minus Low)** — Value factor. Positive loading means the stock behaves like value stocks (cheap relative to book value). Negative loading means it behaves like growth stocks.

            **Why it matters:** A stock might show positive alpha under CAPM but once you account for its size and value exposures, the alpha disappears. The FF3 model helps determine whether a fund manager's returns come from genuine skill or just from holding small-cap or value stocks.

            **R² improvement:** FF3 typically explains more of a stock's returns than CAPM alone. If R² jumps from 0.6 to 0.8, the size and value factors are important for that stock.
            """)

        with st.expander("Factor Scores Table", expanded=False):
            st.markdown("""
            **What it shows:** Cross-sectional rankings of all loaded stocks on three factors. This is a portfolio-level view showing which stocks currently rank highest on each factor.

            - **Momentum score** — How the stock has performed over the past 12 months relative to others. High score = strong recent winner. Low score = recent loser.
            - **Value score** — How cheap the stock appears relative to others based on price-based proxies. High score = potentially undervalued.
            - **Quality score** — How stable and consistent the stock's returns have been. High score = steady, reliable business.
            - **Composite** — Equal-weight average of all three scores. The stocks at the top are strongest across all dimensions simultaneously.

            **How to use it:** This table helps you understand the style tilt of your portfolio. If your top holdings all have high momentum scores, your portfolio has a momentum tilt — it will outperform when momentum is working and underperform when it reverses.
            """)

        # ══════════════════════════════════════
        st.markdown("---")
        st.markdown("## 🌡️ Regime Detection")
        st.markdown("*Analyses the overall market (average of your loaded stocks) to identify the current environment. Portfolio-level view.*")
        # ══════════════════════════════════════

        with st.expander("What is a Market Regime?", expanded=False):
            st.markdown("""
            **What it means:** The market doesn't behave the same way all the time. It cycles through distinct states — called regimes — each with different return and risk characteristics.

            **The three regimes:**
            - **Bull** — Prices trending upward, low volatility, positive average returns. Most people are optimistic. Momentum strategies tend to work.
            - **Bear** — Prices trending downward, high volatility, negative average returns. Defensive positioning matters most.
            - **Sideways** — No clear trend, prices ranging between support and resistance. Mean-reversion strategies tend to work better.

            **Why it matters for you:** A strategy or portfolio that works brilliantly in a Bull market may perform very poorly in a Bear market. Knowing the regime helps you calibrate expectations and decide whether to hold, reduce, or hedge positions.
            """)

        with st.expander("Consensus Regime and Confidence Score", expanded=False):
            st.markdown("""
            **How it works:** The platform runs four different detection methods simultaneously, each using different data. A majority vote determines the consensus regime.

            **Confidence score:** The fraction of methods that voted for the same regime.
            - 75–100% — Strong consensus. High confidence in the current regime.
            - 50–75% — Moderate consensus. Reasonable signal but some disagreement.
            - Below 50% — Mixed signals. Often happens during regime transitions. Reduce conviction.

            **When confidence is low:** This usually means the market is in transition between regimes. It's often a good time to be cautious and wait for a clearer signal rather than making big directional bets.
            """)

        with st.expander("The Four Detection Methods Explained", expanded=False):
            st.markdown("""
            **MA Regime (Moving Average):**
            Compares the 50-day and 200-day moving averages of your portfolio's average price.
            - Bull = 50MA above 200MA and price above both (Golden Cross territory)
            - Bear = 50MA below 200MA and price below both (Death Cross territory)
            - Sideways = mixed signals

            **Drawdown Regime:**
            Based on how far the market has fallen from its all-time high.
            - Bull = within 5% of the all-time high
            - Bear = drawdown greater than 20% (the formal definition of a bear market)
            - Recovery = between -5% and -20%

            **RSI Regime:**
            Uses the 14-day RSI of the average portfolio price.
            - Bull = RSI above 65 (strong upward momentum)
            - Bear = RSI below 35 (strong downward momentum)
            - Sideways = RSI between 35 and 65

            **Volatility Regime:**
            Compares current 21-day realised volatility to its own 1-year history.
            - Low Vol = bottom third of historical vol range (typically Bull)
            - Medium Vol = middle third
            - High Vol = top third (typically Bear or stressed)
            """)

        with st.expander("Regime Statistics Table", expanded=False):
            st.markdown("""
            **What it shows:** Looking back through your date range, how did the average of your portfolio's stocks perform during each detected regime?

            - **Ann. Return** — Average annualised return during that regime
            - **Ann. Volatility** — How much things jumped around
            - **Sharpe** — Risk-adjusted return during that regime
            - **% of Time** — How often that regime occurred in your date range

            **How to use it:** If your portfolio has a negative Sharpe during Bear regimes, that's a sign you have high market exposure with limited downside protection. The regime statistics help you understand what your portfolio's weak spots are.
            """)

        with st.expander("Regime History Chart", expanded=False):
            st.markdown("""
            **What it shows:** The average price of all your loaded stocks over time with colour-coded background shading.

            - **Green background** — Bull regime
            - **Yellow background** — Sideways regime
            - **Red background** — Bear regime

            **How to use it:** Look at where the major drawdowns occurred. Were they in red-shaded Bear periods? If the red shading appeared before or at the same time as the falls, the regime signal was providing useful advance warning. If large falls occurred during green-shaded periods, the signal didn't protect you in that instance.
            """)

        # ══════════════════════════════════════
        st.markdown("---")
        st.markdown("## ⚡ Strategy Backtester")
        st.markdown("*Tests how a specific trading strategy would have performed on your loaded stocks over the historical period. Portfolio-level simulation.*")
        # ══════════════════════════════════════

        with st.expander("What is Backtesting?", expanded=False):
            st.markdown("""
            **What it means:** A backtest simulates how a trading strategy would have performed if you had applied it to historical data. It answers the question: "If I had followed this strategy over the past 5 years, what would have happened?"

            **Important caveat:** Past performance does not guarantee future results. A strategy that worked historically may not continue to work. Backtesting is a research tool, not a crystal ball.

            **Transaction costs:** The backtester includes realistic costs — commission, bid-ask spread, and slippage — so results are not inflated by ignoring the cost of trading. You can adjust the commission rate.

            **Signal lag:** Signals are intentionally applied with a one-day delay (the default). This means if a strategy generates a buy signal on Tuesday, the position is entered on Wednesday. This prevents look-ahead bias — a common way backtests produce unrealistically good results.
            """)

        with st.expander("The 7 Strategies Explained", expanded=False):
            st.markdown("""
            **Cross-Sectional Momentum:**
            Every month, ranks all your loaded stocks by their 12-month return (skipping the most recent month). Buys the top N highest-ranked stocks in equal weights, rebalancing monthly.
            Based on the academic finding that recent winners tend to continue outperforming over the next 3–12 months (Jegadeesh & Titman, 1993).
            *Needs at least 13 months of data to generate its first signal.*

            **Time-Series Momentum:**
            Each stock is evaluated independently. If a stock's own 12-month return is positive, it is held. If negative, it is not held. Positions are sized by inverse volatility — volatile stocks get smaller weights so they don't dominate risk.
            Simpler than cross-sectional momentum and tends to hold fewer positions.

            **Risk-Adjusted Momentum:**
            Like cross-sectional momentum, but stocks are ranked by their momentum score divided by their trailing volatility. This is essentially a trailing Sharpe Ratio. Prevents buying volatile stocks that happened to spike recently.

            **Bollinger Band Reversion:**
            A mean-reversion strategy. Buys a stock when its price falls below its lower Bollinger Band (oversold) and sells when it returns to the middle band. Also sells short when price spikes above the upper band.
            Works best in range-bound markets. Struggles in strong trending markets.

            **RSI Mean Reversion:**
            Buys when RSI falls below 30 (oversold) and exits when RSI returns to 50. Sells short when RSI rises above 70 (overbought).
            Similar to Bollinger Bands but uses momentum rather than price range as the signal.

            **Factor Strategy:**
            Ranks all stocks by a composite score combining momentum (40%), value (30%), and quality (30%). Buys the top N stocks each month.
            The most academically grounded strategy — each factor has decades of research supporting it.

            **Regime Adaptive:**
            Automatically switches between sub-strategies based on the detected market regime:
            - Bull → uses momentum strategy
            - Bear → switches to lowest-volatility defensive stocks
            - Sideways → switches to value-ranked stocks
            The idea is to use the most appropriate strategy for current conditions.
            """)

        with st.expander("Equity Curve Chart", expanded=False):
            st.markdown("""
            **What it shows:** How $1,000,000 of starting capital grew (or shrank) over the backtest period under the selected strategy. The dotted orange line is the benchmark.

            **How to read it:**
            - A smooth upward curve = steady consistent returns
            - A jagged curve with big dips = high volatility, lots of drawdowns
            - Curve above the benchmark line = strategy outperformed the index
            - Curve that drops and stays below its previous peak = drawdown period

            **What to watch for:** A curve that rises sharply then falls sharply may be overfitting — performing well historically but unlikely to hold up in the future.
            """)

        with st.expander("Drawdown Chart", expanded=False):
            st.markdown("""
            **What it shows:** At every point in time, how far the strategy was below its previous peak. Always zero or negative.

            **How to read it:**
            - A shallow chart that stays near 0% = low drawdown strategy, capital is well protected
            - Deep troughs = the strategy suffered large losses that took time to recover from
            - Width of a trough = how long the recovery took

            **The psychological reality:** A -30% drawdown means you watched $1,000,000 become $700,000 and had to wait (possibly years) to get back to $1,000,000. Even if the long-run return looks good, can you emotionally hold through that? This chart helps you answer that question honestly.
            """)

        with st.expander("Key Backtest Metrics Explained", expanded=False):
            st.markdown("""
            **CAGR (Compound Annual Growth Rate):**
            Average yearly return across the whole period. 12% CAGR = doubled roughly every 6 years.

            **Sharpe Ratio:**
            Return earned per unit of risk. Formula: (CAGR − Risk Free Rate) ÷ Volatility.
            - Below 0.5 = poor
            - 0.5–1.0 = acceptable
            - 1.0–1.5 = good
            - Above 1.5 = strong

            **Max Drawdown:**
            The worst peak-to-trough fall. If this number would have caused you to panic-sell, the strategy is too aggressive for your risk tolerance.

            **Sortino Ratio:**
            Like Sharpe but only penalises downside moves. A Sortino higher than your Sharpe means most of the volatility was on the upside — a good sign.

            **Win Rate:**
            Percentage of days with a positive return. A 55% win rate means you made money on 55% of trading days.

            **Monthly Returns Heatmap:**
            Each cell shows the return for that month and year. Green = positive, red = negative. Helps identify seasonal patterns and which years were strongest or weakest.
            """)

        # ══════════════════════════════════════
        st.markdown("---")
        st.markdown("## ⚖️ Portfolio Optimisation")
        st.markdown("*Finds mathematically optimal weights for your portfolio using Markowitz theory. Portfolio-level analysis.*")
        # ══════════════════════════════════════

        with st.expander("What is Portfolio Optimisation?", expanded=False):
            st.markdown("""
            **The problem it solves:** If you have 8 stocks, how should you split your money between them? Equal weights? Concentrated in your best idea? This module finds the answer mathematically.

            **The core insight (Markowitz, 1952 — Nobel Prize):** Combining assets that don't move together perfectly reduces overall portfolio volatility without reducing expected return. This is the mathematics of diversification. A portfolio of 8 stocks is almost always less risky than any single one of those stocks, if they're not perfectly correlated.

            **What it needs:** Historical return data for all stocks. It uses this to estimate expected returns, volatilities, and correlations, then finds the weight combinations that produce the best risk/return trade-off.

            **Important caveat:** The optimiser uses historical data. Future returns and correlations will differ. Treat optimal weights as a reference point, not a prescription.
            """)

        with st.expander("The Efficient Frontier Chart", expanded=False):
            st.markdown("""
            **What it shows:** A scatter plot with Volatility (risk) on the x-axis and Expected Return on the y-axis.

            **The cloud of dots:** 2,000 randomly generated portfolios using different weight combinations of your stocks. Each dot is one possible portfolio.

            **The orange curve (Efficient Frontier):** The boundary of the cloud. Every portfolio on this curve is optimal — there is no other combination of your stocks that could achieve a higher return at that level of volatility. Portfolios to the right or below the frontier are suboptimal.

            **Your Portfolio (red diamond):** Where your current weights sit in this space. The further it is from the frontier, the more room there is to improve by rebalancing.

            **Max Sharpe star:** The single point on the frontier with the highest Sharpe Ratio. This is typically the most recommended target.

            **Practical message:** If your red diamond is well to the right of the orange curve at the same height, you're taking more risk than necessary for your level of expected return.
            """)

        with st.expander("The Four Portfolio Types", expanded=False):
            st.markdown("""
            **Current (grey bars):** Your actual weights as entered. The starting point.

            **Max Sharpe (blue bars):** The portfolio on the efficient frontier with the highest Sharpe Ratio. Maximises return per unit of risk. Best choice if you want the most efficient portfolio overall.

            **Min Volatility (yellow bars):** The portfolio with the lowest possible annualised volatility. Minimises risk. Best choice if preserving capital matters more than maximising growth.

            **Risk Parity (green bars):** Instead of equal capital weights, this allocates so each stock contributes equal risk. High-volatility stocks get less capital; low-volatility stocks get more. Produces more genuinely diversified portfolios than equal weight.

            **Portfolio Comparison Table:** Shows CAGR, Volatility, and Sharpe for each portfolio type so you can compare them directly.
            """)

        with st.expander("Covariance Method — Which to Choose?", expanded=False):
            st.markdown("""
            **What it is:** Before the optimiser can find the best weights, it needs to estimate how all your stocks relate to each other (the covariance matrix). Three methods are available.

            **Ledoit-Wolf (recommended):** A mathematical shrinkage technique that makes the covariance estimate more stable by reducing the influence of extreme historical correlations that may be statistical noise. Produces more balanced, less concentrated weights. Best for most situations.

            **Sample:** Uses the raw historical covariance with no adjustment. Can be noisy and produce over-concentrated portfolios, especially when there aren't many years of data. Only recommended if you have 10+ years and fewer than 10 stocks.

            **EWMA (Exponentially Weighted Moving Average):** Puts more weight on recent data. More responsive to recent changes in correlations. Useful when you believe recent market relationships are more representative than older ones.
            """)

        with st.expander("Black-Litterman (under Attribution tab)", expanded=False):
            st.markdown("""
            **What problem it solves:** Standard Markowitz is very sensitive to expected return inputs — tiny changes in estimated returns produce wildly different weight allocations. Black-Litterman fixes this.

            **How it works:**
            1. Starts from equilibrium — the market-implied return for each stock based on its size and risk
            2. You add your own views: "I think BHP will return 15%" or "CBA will outperform WBC by 3%"
            3. The model mathematically blends your views with the equilibrium based on your stated confidence level
            4. The resulting posterior expected returns are much more stable and sensible

            **Confidence level:** A confidence of 0.5 means your view gets equal weight with the market equilibrium. 0.1 means the market dominates. 0.9 means your view dominates. Start with 0.3–0.5 for most views.

            **Equilibrium vs Posterior chart:** Shows how your views have shifted return expectations away from market equilibrium. If a view barely moves the bar, either your confidence is too low or the equilibrium strongly disagrees.
            """)

        # ══════════════════════════════════════
        st.markdown("---")
        st.markdown("## 🎲 Monte Carlo Simulation")
        st.markdown("*Simulates thousands of possible future return paths for your portfolio to understand the range of outcomes. Portfolio-level analysis.*")
        # ══════════════════════════════════════

        with st.expander("What is Monte Carlo Simulation?", expanded=False):
            st.markdown("""
            **What it does:** Instead of asking "what will the market do next year?", Monte Carlo asks "across 5,000 simulated scenarios, what is the distribution of possible outcomes?" It gives you a realistic range rather than a single guess.

            **The fan chart:** Shows 100 sample simulation paths in transparent blue. The brighter coloured lines are percentile bands:
            - **P5 (red)** — Only 5% of scenarios were this bad or worse
            - **P25 (orange)** — 25% of scenarios were this bad or worse
            - **P50 (blue)** — The median outcome. Half of scenarios above, half below.
            - **P75 (orange)** — 75% of scenarios were at or below this level
            - **P95 (red)** — 95% of scenarios were at or below this level

            **The return distribution chart:** A histogram showing how often each 1-year return appeared across all simulations. The vertical line at 0 marks the break-even point.

            **Practical use:** If the P5 outcome (worst 5% of scenarios) shows a -25% return and you couldn't emotionally handle that, you need to either reduce risk or shorten your horizon.
            """)

        with st.expander("GBM — Geometric Brownian Motion", expanded=False):
            st.markdown("""
            **What it assumes:** Returns follow a normal distribution with a constant mean (drift) and constant volatility. The model is calibrated to your portfolio's own historical mean return and volatility.

            **Strengths:** Fast, mathematically clean, the industry standard for basic simulations.

            **Weaknesses:** Real markets crash harder and more suddenly than GBM predicts. GBM assigns near-zero probability to events like the GFC or COVID crash. It underestimates tail risk.

            **Best used as:** A baseline reference. If you need a quick sense of expected outcomes under normal conditions.
            """)

        with st.expander("Jump Diffusion (Merton Model)", expanded=False):
            st.markdown("""
            **What it adds:** Random discrete jumps (crashes) on top of normal GBM. The jumps follow a Poisson process — on average 5 crash events per year, each averaging -5%.

            **Why it's more realistic:** The GFC fell 50% in ways that pure GBM would call essentially impossible. Jump diffusion explicitly models these tail events, producing worse P5 outcomes but more honest VaR and CVaR figures.

            **If Jump Diffusion VaR is much worse than GBM VaR:** Your portfolio has significant crash risk that standard volatility measures are hiding. This is common in portfolios concentrated in a single sector.
            """)

        with st.expander("Bootstrap Simulation", expanded=False):
            st.markdown("""
            **What it does:** Randomly resamples blocks of your actual historical returns and strings them together. No model assumptions — it uses your real return distribution directly.

            **Why it's the most honest method:** It preserves the true distribution of your portfolio's returns, including fat tails, skewness, and volatility clustering. If your portfolio experienced the GFC, bootstrap will include GFC-like sequences in its simulations at the right frequency.

            **When to use it:** When you have at least 3 years of data that includes at least one stressed period. If your data is only from a calm bull market, bootstrap will inherit that bias.
            """)

        with st.expander("Reading the Simulation Metrics", expanded=False):
            st.markdown("""
            **Expected Return:** Average return across all simulations over the horizon. Not a forecast — the median (P50) is often more useful.

            **P(Loss):** The fraction of simulations that ended below 100 (i.e. lost money). P(Loss) = 30% means 1 in 3 simulated years ended with a loss.

            **VaR 95% (1Y):** The return level exceeded on the downside in only 5% of simulations. If VaR = 18%, then in 1-in-20 simulated years the loss exceeded 18%.

            **CVaR 95% (1Y):** The average loss in the worst 5% of simulations. Always worse than VaR. This is the expected loss "if things go badly."

            **Combining them:** P(Loss) = 25%, VaR 95% = 15%, CVaR 95% = 22% means: 1 in 4 years you lose money, 1 in 20 years you lose more than 15%, and in those bad years the average loss is 22%.
            """)

        # ══════════════════════════════════════
        st.markdown("---")
        st.markdown("## 💹 Options Pricer")
        st.markdown("*Prices European-style options on individual stocks using the Black-Scholes model. Single stock analysis.*")
        # ══════════════════════════════════════

        with st.expander("What is an Option?", expanded=False):
            st.markdown("""
            **A call option** gives you the right (but not obligation) to buy a stock at a fixed price (the strike) before a set date (expiry).

            **A put option** gives you the right to sell a stock at the strike price before expiry.

            **Why use options:**
            - To bet on a stock rising with limited downside (buy a call — maximum loss is the premium paid)
            - To protect an existing holding from falling (buy a put — like insurance on your stock)
            - To generate income from an existing holding (sell a call — receive premium in exchange for capping upside)

            **ASX note:** ASX equity options are American-style (can exercise early). The Black-Scholes model prices European-style options (can only exercise at expiry). For most practical purposes the difference is small for non-dividend paying stocks.
            """)

        with st.expander("Black-Scholes Option Price", expanded=False):
            st.markdown("""
            **What determines the price:**
            - **Spot Price (S)** — Current stock price (auto-filled from your loaded data)
            - **Strike (K)** — The price at which you can buy/sell. At-the-money = strike equals spot.
            - **Time to Expiry (T)** — Longer expiry = more expensive (more time for the stock to move in your favour)
            - **Implied Volatility (σ)** — Higher vol = more expensive (the stock could swing further in either direction)
            - **Risk-Free Rate (r)** — Small effect on pricing
            - **Dividend Yield (q)** — Reduces call prices slightly (dividends reduce the stock price)

            **Intrinsic Value vs Time Value:**
            - Intrinsic value = what the option is worth if exercised right now (e.g. call with strike $40 when stock is at $45 = $5 intrinsic value)
            - Time value = the extra you pay for the remaining time (always positive for unexpired options)
            - Total price = Intrinsic + Time value
            """)

        with st.expander("The Greeks — What They Mean in Plain English", expanded=False):
            st.markdown("""
            The Greeks measure how sensitive the option price is to different inputs.

            **Delta:**
            How much the option price moves per $1 move in the underlying stock.
            - Call delta is between 0 and 1. A delta of 0.6 means the option rises $0.60 when the stock rises $1.
            - Put delta is between -1 and 0. A delta of -0.4 means the put rises $0.40 when the stock falls $1.
            - ATM options have delta ≈ ±0.5. Deep ITM options have delta near ±1. Deep OTM options have delta near 0.
            *Practical use: If you own a call with delta 0.5, you need 2 calls to replicate owning 1 share.*

            **Gamma:**
            How fast delta changes per $1 move in the stock. High gamma = delta changes rapidly.
            *Practical use: High gamma options need frequent rebalancing if you're delta-hedging.*

            **Theta (time decay):**
            How much the option loses per calendar day from time alone, with everything else unchanged.
            Theta is always negative for long options — you're paying for time.
            An option with theta = -$0.03 loses $0.03 of value every day just from time passing.
            *Practical use: If you own options, theta works against you. If you've sold options, theta works for you.*

            **Vega:**
            How much the option price changes per 1% increase in implied volatility.
            A vega of $0.15 means the option gains $0.15 if vol rises from 20% to 21%.
            *Practical use: Buying options before expected announcements (earnings, RBA decisions) is a bet that vol will rise.*

            **Rho:**
            How much the option price changes per 1% increase in interest rates. Usually small for short-dated options.
            """)

        with st.expander("Payoff at Expiry Chart", expanded=False):
            st.markdown("""
            **What it shows:** For every possible stock price at expiry (x-axis), the profit or loss on the option position (y-axis), after accounting for the premium paid.

            **Break-even point:**
            - Call: stock must rise above Strike + Premium paid
            - Put: stock must fall below Strike - Premium paid

            **The vertical lines:**
            - Yellow dashed line = current spot price
            - Orange dashed line = strike price

            **Green fill above zero = profitable territory. Red fill below zero = loss territory.**

            **Key insight:** Buying a call limits your maximum loss to the premium paid (the chart never goes below -premium). But you only make money if the stock closes above the break-even at expiry.
            """)

        with st.expander("Delta and Gamma Sensitivity Chart", expanded=False):
            st.markdown("""
            **What it shows:** How delta and gamma change as the stock price moves.

            **Delta curve:** S-shaped curve from 0 to 1 (calls) or 0 to -1 (puts). Steepest slope occurs at the strike price — this is where delta is most sensitive.

            **Gamma curve:** Bell-shaped, peaking at the strike. Gamma is highest for at-the-money options and falls off as you move deeper in-the-money or out-of-the-money.

            **Why this matters:** Near expiry, gamma spikes dramatically for at-the-money options. A 1% move in the stock can change delta by a large amount, which is why near-expiry options are often called "lottery tickets" — small moves create outsized gains or losses.
            """)

        # ══════════════════════════════════════
        st.markdown("---")
        st.markdown("## 📊 Portfolio Analytics Dashboard")
        st.markdown("*Deep-dive analysis of a specific portfolio you define by entering tickers and weights. All analyses are portfolio-level.*")
        # ══════════════════════════════════════

        with st.expander("How to Set Up Your Portfolio", expanded=False):
            st.markdown("""
            **When embedded in the terminal:** The tickers already loaded in the main sidebar are used automatically. Choose Equal Weight or Custom Weights at the top of this page, then click **▶ Run Portfolio Analysis**.

            **When run standalone:** Configure tickers, weights, dates, and benchmark in the sidebar, then click **▶ Run Analysis**.

            **Weights:** Must sum to 1 (100%). Equal weight means each stock gets 1/n of the portfolio. Custom weights let you specify each position size directly.

            **Covariance method:** Controls how correlations are estimated for the risk decomposition. Ledoit-Wolf is the recommended default.
            """)

        with st.expander("Performance Tab — What Each Metric Means", expanded=False):
            st.markdown("""
            All metrics here are for the *combined weighted portfolio*, not individual stocks.

            **CAGR:** Average yearly return of the portfolio as a whole.
            **Sharpe Ratio:** Return per unit of total portfolio risk.
            **Max Drawdown:** Worst peak-to-trough fall the portfolio experienced.
            **Sortino Ratio:** Return per unit of downside risk only.
            **Calmar Ratio:** CAGR divided by max drawdown. Higher = better returns relative to worst losses.
            **Alpha:** Return above what the portfolio's market exposure would have predicted.
            **Beta:** Portfolio sensitivity to the ASX 200 benchmark. Beta of 1.2 = 20% more volatile than the index.
            **Win Rate:** % of days the portfolio was positive.
            **Profit Factor:** Total gains divided by total losses. Above 1.5 is considered healthy.

            **Full Metrics Table:** Expands to show 30+ metrics including VaR, CVaR, skewness, kurtosis, tracking error, and more.
            """)

        with st.expander("Risk Tab — MCTR, PCTR, CVaR Decomposition", expanded=False):
            st.markdown("""
            **This tab answers: which stocks are driving your portfolio's risk?**

            **Portfolio Volatility:** Total annualised standard deviation of the combined portfolio returns.

            **% Risk Contribution (PCTR) Bar Chart:**
            Shows each stock's percentage contribution to total portfolio volatility. All bars sum to 100%.
            - A stock at 35% PCTR but only 15% weight = it's consuming more than double its share of risk
            - A stock at 8% PCTR with 15% weight = it's a diversifier, contributing less risk than its capital weight suggests

            **Weight vs Risk Scatter Chart:**
            Each dot is a stock. The diagonal line = equal risk and weight contribution.
            - Dots above the line = risk-heavy (contributing more risk than capital)
            - Dots below the line = diversifiers (contributing less risk than capital)
            - The colour scale shows the Risk/Weight ratio — red = concerning concentration

            **CVaR Decomposition:**
            On very bad days (worst 5%), how much of the portfolio's loss does each stock explain? Stocks with high CVaR contribution are your largest sources of tail risk.

            **Full Risk Attribution Table:** Shows MCTR (Marginal Contribution to Risk — how much volatility increases if you add 1% more of this stock), CTR (absolute risk contribution), and PCTR for every holding.
            """)

        with st.expander("Exposures Tab — Sector, Factor, Correlation", expanded=False):
            st.markdown("""
            **This tab answers: what kind of bets are you actually making?**

            **Effective N:** The effective number of independent positions. A 10-stock portfolio where 3 stocks dominate the weights might only have an effective N of 4. Tells you if you're more diversified than your holding count suggests, or less.

            **Diversification Ratio:** Weighted average of individual stock vols divided by portfolio vol. A ratio of 1.4 means your portfolio is 28% less volatile than an undiversified collection of the same stocks. Higher = more diversification benefit.

            **Avg Pairwise Correlation:** Average correlation between every pair of stocks. Lower = more genuinely diversified. Above 0.7 average = your portfolio moves largely as one.

            **Sector Donut Chart:** What fraction of your portfolio is in each GICS sector (Financials, Materials, Healthcare, etc.). Heavy concentration in one sector = sector risk.

            **Asset Correlation Matrix:** Heat map showing every stock pair's correlation. Dark red clusters show groups of stocks that move together — owning multiples within a cluster adds little diversification.

            **Factor Exposures:** Regresses portfolio returns against momentum, value, quality, size, and low-vol factors to show your systematic factor tilts. A high momentum loading means your portfolio behaves like a momentum strategy — it will do well when momentum is rewarded and poorly when it reverses.
            """)

        with st.expander("Attribution Tab — BHB Performance Attribution", expanded=False):
            st.markdown("""
            **This tab answers: where did the portfolio's returns actually come from?**

            The Brinson-Hood-Beebower model breaks the difference between your portfolio return and the benchmark into three components:

            **Allocation Effect:**
            Did you put more weight in sectors that outperformed? Positive = your sector tilts added value. Negative = you over-weighted the wrong sectors.

            **Selection Effect:**
            Within each sector, did you pick better stocks than the benchmark average? Positive = good stock selection. Negative = you picked the underperformers within sectors.

            **Interaction Effect:**
            The combined effect of overweighting sectors where your picks were also the best. Usually small.

            **Waterfall Chart:**
            Visually shows how each effect built up from the benchmark return to your portfolio return. Stepping up = that effect added value. Stepping down = that effect detracted value.

            **Attribution Detail Table:** Shows all three effects broken down by sector so you can see exactly which sector decisions helped or hurt.
            """)

        with st.expander("Stress Tests Tab — Hypothetical and Historical", expanded=False):
            st.markdown("""
            **This tab answers: how would my portfolio hold up in a crisis?**

            **Hypothetical Scenarios tab:**
            Applies predefined macro shocks to your current weights. Each scenario specifies a % decline for specific sectors or stocks, then calculates the total weighted impact on your portfolio.
            - GFC 2008 — market shock of -45%
            - COVID Crash — market shock of -35%
            - China Hard Landing — heavy hit to Materials (BHP, RIO, FMG)
            - RBA Rate Shock +200bps — heavy hit to Financials and Real Estate
            - Iron Ore -40% — targeted hit to iron ore miners
            - Tech Selloff -30% — targeted hit to technology stocks
            These are simplified estimates. Real crashes are often worse due to correlation spikes.

            **Historical Crises tab:**
            Uses your actual loaded return data to replay what your portfolio *really would have experienced* during real crisis periods. More reliable than hypothetical shocks because it uses the actual daily return sequence including the path, partial recoveries, and duration.
            Shows Portfolio Return, Max Drawdown, and Volatility for each crisis period.
            """)

        with st.expander("Advanced Tab — PCA, Rolling Metrics, Risk Budget", expanded=False):
            st.markdown("""
            **Principal Component Analysis (PCA):**
            Breaks down total portfolio risk into independent components. PC1 usually represents broad market risk. PC2 and PC3 represent sector tilts or style factors.
            If PC1 explains 85% of risk, your portfolio is essentially a leveraged index fund regardless of how many stocks it holds.
            The PC Loadings table shows which stocks load onto each component — high loading = that stock is a primary driver of that risk component.

            **Rolling Risk Metrics (63-day rolling window):**
            - Rolling Sharpe: Has the portfolio's risk-adjusted return been consistent over time, or did it cluster in one period?
            - Rolling Volatility: Shows how the portfolio's risk level has changed. Spikes indicate stressed periods.
            - Rolling Beta: How market-sensitive has the portfolio been over time? Rising beta = becoming more correlated with the market.

            **Risk Budget Analysis:**
            Compares actual risk contribution of each stock against an equal-risk target. If BHP is over-budget (Actual Risk % > Target Risk %), it's consuming more than its allocated share of risk. The Deviation column shows exactly how far each stock is from its target. This is the tool for implementing a risk parity rebalancing.
            """)

        st.markdown("---")
        st.info("""
        💡 **The Big Picture:** All these tools work together. Start with Market Overview to understand the landscape.
        Use Technical Analysis to assess individual stock momentum. Check Factor Analysis to understand style exposures.
        Run the Backtester to see how strategies have worked. Use Portfolio Analytics to understand your combined risk.
        Check the Regime tab before making big allocation decisions. Use Monte Carlo to stress-test your forward expectations.
        No single number tells the whole story — the value is in seeing how they all connect.

        Additionally, No single metric tells the full story. A well-constructed portfolio has strong CAGR, reasonable volatility, 
        a Sharpe above 1, a max drawdown you can psychologically handle, risk spread evenly across assets, and weights that sit close to the efficient frontier. 
        Use the Regime tab to calibrate your expectations and the Monte Carlo tab to understand the full range of possible outcomes.
        """)

    # ─────────────────────────────────────────────
    #  ANALYTICS TAB — requires analysis to run
    # ─────────────────────────────────────────────
    with tab_run:
        has_results = "pd_analysis_cache" in st.session_state

        if not embedded:
            # Standalone: data was loaded above and run_btn was already checked
            pass
        else:
            # Embedded: show run button state
            if not run_btn and not has_results:
                st.info(
                    "Set your weights above and click **▶ Run Portfolio Analysis** "
                    "to load all analytics."
                )
                return

        # ── DATA LOADING (standalone only — embedded uses preloaded data) ──
        if not embedded:
            with st.spinner("Fetching market data…"):
                try:
                    all_tickers = tickers + [benchmark_ticker]
                    raw = yf.download(
                        all_tickers,
                        start=str(start_date),
                        end=str(end_date),
                        auto_adjust=True,
                        progress=False,
                    )

                    if raw.empty:
                        st.error("No data returned. Check your tickers and date range.")
                        return

                    if isinstance(raw.columns, pd.MultiIndex):
                        price_data = raw["Close"].copy()
                    else:
                        price_data = raw.copy()

                    if isinstance(price_data, pd.Series):
                        price_data = price_data.to_frame()

                    price_data.index = pd.DatetimeIndex(price_data.index).tz_localize(None)
                    price_data = price_data.dropna(how="all")

                    benchmark_series = None
                    if benchmark_ticker in price_data.columns:
                        bm_prices = price_data[benchmark_ticker].dropna()
                        benchmark_series = bm_prices.pct_change().dropna()
                        benchmark_series.name = benchmark_ticker
                        price_data = price_data.drop(columns=[benchmark_ticker])

                    available = [t for t in tickers if t in price_data.columns]
                    if not available:
                        st.error("No price data found for your tickers.")
                        return

                    weights_available = {t: weights[t] for t in available}
                    total_w = sum(weights_available.values())
                    weights_available = {t: w / total_w for t, w in weights_available.items()}

                except Exception as e:
                    st.error(f"Data loading failed: {e}")
                    return

        # ── COMPUTE METRICS (with session state caching) ──
        cache_key = str(sorted(weights_available.items())) + cov_method

        should_compute = (
            run_btn
            or "pd_analysis_cache" not in st.session_state
            or st.session_state.get("pd_cache_key") != cache_key
        )

        if should_compute:
            with st.spinner("Computing analytics…"):
                try:
                    asset_returns_clean = price_data.pct_change().dropna()

                    _tickers = list(weights_available.keys())
                    _w = np.array([weights_available[t] for t in _tickers])
                    _w = _w / _w.sum()
                    _asset_ret = price_data[_tickers].pct_change().dropna()
                    port_returns = pd.Series(
                        _asset_ret.values @ _w,
                        index=_asset_ret.index,
                        name="Portfolio",
                    )

                    pm = PortfolioMetrics(
                        port_returns,
                        benchmark_returns=benchmark_series,
                        risk_free_rate=risk_free_rate,
                        name="Portfolio",
                    )
                    summary = pm.compute_all()

                    rd = RiskDecomposer(
                        asset_returns_clean,
                        weights_available,
                        cov_method=cov_method,
                    )
                    risk_table = rd.risk_attribution_table()
                    stress_df = rd.stress_test()
                    pca = rd.pca_risk_decomposition()
                    cvar_decomp = rd.cvar_decomposition(0.95)

                    ea = ExposureAnalyser(
                        price_data,
                        weights_available,
                        benchmark=benchmark_series,
                    )
                    sector_df = ea.sector_exposure()
                    concentration = ea.concentration_metrics()
                    corr_matrix = ea.correlation_matrix()
                    div_ratio = ea.diversification_ratio()
                    factor_exp = ea.factor_exposure()

                    st.session_state["pd_analysis_cache"] = {
                        "port_returns": port_returns,
                        "summary": summary,
                        "pm": pm,
                        "rd": rd,
                        "risk_table": risk_table,
                        "stress_df": stress_df,
                        "pca": pca,
                        "cvar_decomp": cvar_decomp,
                        "sector_df": sector_df,
                        "concentration": concentration,
                        "corr_matrix": corr_matrix,
                        "div_ratio": div_ratio,
                        "factor_exp": factor_exp,
                        "asset_returns_clean": asset_returns_clean,
                        "weights_available": weights_available,
                        "price_data": price_data,
                        "benchmark_series": benchmark_series,
                        "benchmark_ticker": benchmark_ticker,
                        "risk_free_rate": risk_free_rate,
                        "available": available,
                    }
                    st.session_state["pd_cache_key"] = cache_key

                except Exception as e:
                    st.error(f"Analytics computation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return

        # Load from cache
        cache = st.session_state["pd_analysis_cache"]
        port_returns = cache["port_returns"]
        summary = cache["summary"]
        pm = cache["pm"]
        rd = cache["rd"]
        risk_table = cache["risk_table"]
        stress_df = cache["stress_df"]
        pca = cache["pca"]
        cvar_decomp = cache["cvar_decomp"]
        sector_df = cache["sector_df"]
        concentration = cache["concentration"]
        corr_matrix = cache["corr_matrix"]
        div_ratio = cache["div_ratio"]
        factor_exp = cache["factor_exp"]
        asset_returns_clean = cache["asset_returns_clean"]
        weights_available = cache["weights_available"]
        price_data = cache["price_data"]
        benchmark_series = cache["benchmark_series"]
        benchmark_ticker = cache["benchmark_ticker"]
        risk_free_rate = cache["risk_free_rate"]
        available = cache["available"]

        # ── INNER ANALYTICS TABS ──
        tab_perf, tab_risk, tab_exposure, tab_attribution, tab_stress, tab_advanced, tab_mc, tab_opt, tab_regime = st.tabs([
            "📈 Performance",
            "⚠️ Risk",
            "🏭 Exposures",
            "📊 Attribution",
            "🌡️ Stress Tests",
            "🔬 Advanced",
            "🎲 Monte Carlo",
            "⚖️ Optimise Weights",
            "🌡️ Regime",
        ])

        # ─────────── PERFORMANCE ───────────
        with tab_perf:
            st.subheader("Key Metrics")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("CAGR", f"{summary.cagr:.2%}")
            with col2:
                st.metric("Sharpe Ratio", f"{summary.sharpe_ratio:.3f}")
            with col3:
                st.metric("Max Drawdown", f"{summary.max_drawdown:.2%}")
            with col4:
                st.metric("Sortino Ratio", f"{summary.sortino_ratio:.3f}")
            with col5:
                st.metric("Calmar Ratio", f"{summary.calmar_ratio:.3f}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Alpha (ann.)", f"{summary.alpha:.2%}")
            with col2:
                st.metric("Beta", f"{summary.beta:.3f}")
            with col3:
                st.metric("Win Rate", f"{summary.win_rate:.1%}")
            with col4:
                st.metric("Profit Factor", f"{summary.profit_factor:.2f}")

            fig_eq = build_equity_curve_chart(
                port_returns, benchmark_series, "Portfolio", benchmark_ticker
            )
            st.plotly_chart(fig_eq, use_container_width=True, key="equity_curve")

            fig_dd = build_drawdown_chart(port_returns, benchmark_series)
            st.plotly_chart(fig_dd, use_container_width=True, key="drawdown")

            with st.expander("📋 Full Metrics Table"):
                st.dataframe(pm.to_dataframe(), use_container_width=True, height=600)

            st.subheader("Monthly Returns Heatmap")
            try:
                monthly = pm.monthly_returns()
                fig_heat = build_monthly_returns_heatmap(monthly)
                st.plotly_chart(fig_heat, use_container_width=True, key="monthly_heatmap")
            except Exception:
                st.info("Not enough data for monthly heatmap.")

        # ─────────── RISK ───────────
        with tab_risk:
            st.subheader("Risk Decomposition")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Portfolio Volatility", f"{rd.portfolio_volatility():.2%}")
            with col2:
                st.metric("VaR 95%", f"{summary.var_95:.2%}")
            with col3:
                st.metric("CVaR 95%", f"{summary.cvar_95:.2%}")
            with col4:
                st.metric("Skewness", f"{summary.skewness:.3f}")

            fig_risk = build_risk_attribution_chart(risk_table)
            st.plotly_chart(fig_risk, use_container_width=True, key="risk_attribution")

            fig_wr = build_weight_vs_risk_chart(risk_table)
            st.plotly_chart(fig_wr, use_container_width=True, key="weight_vs_risk")

            if not cvar_decomp.empty:
                st.subheader("CVaR Decomposition (95%)")
                st.dataframe(
                    cvar_decomp.style.format({
                        "Weight": "{:.1%}",
                        "Component CVaR": "{:.3%}",
                        "% of Portfolio CVaR": "{:.1%}",
                    }),
                    use_container_width=True,
                )

            with st.expander("📋 Full Risk Attribution Table"):
                st.dataframe(
                    risk_table.style.format({
                        "Weight": "{:.1%}",
                        "Asset Volatility": "{:.1%}",
                        "MCTR": "{:.4f}",
                        "CTR": "{:.4f}",
                        "PCTR": "{:.1%}",
                    }),
                    use_container_width=True,
                )

        # ─────────── EXPOSURES ───────────
        with tab_exposure:
            st.subheader("Portfolio Exposures")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Holdings", f"{concentration['num_holdings']}")
            with col2:
                st.metric("Effective N", f"{concentration['effective_n']:.1f}")
            with col3:
                st.metric("Diversification Ratio", f"{div_ratio:.2f}")
            with col4:
                st.metric(
                    "Avg Pairwise Correlation",
                    f"{ea.average_pairwise_correlation():.3f}"
                )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sector Allocation")
                fig_sector = build_sector_exposure_chart(sector_df)
                st.plotly_chart(fig_sector, use_container_width=True, key="sector_donut")
            with col2:
                st.subheader("Sector Details")
                st.dataframe(
                    sector_df.style.format({"Weight": "{:.1%}"}),
                    use_container_width=True,
                    height=350,
                )

            st.subheader("Asset Correlation Matrix")
            fig_corr = build_correlation_heatmap(corr_matrix)
            st.plotly_chart(fig_corr, use_container_width=True, key="corr_heatmap")

            st.subheader("Factor Exposures")
            if factor_exp is not None and not factor_exp.empty:
                fig_factor = build_factor_exposure_chart(factor_exp)
                st.plotly_chart(fig_factor, use_container_width=True, key="factor_chart")
                with st.expander("Factor Regression Details"):
                    st.dataframe(
                        factor_exp.style.format({
                            "Loading": "{:.4f}",
                            "T-Stat": "{:.3f}",
                            "P-Value": "{:.4f}",
                        }),
                        use_container_width=True,
                    )
            else:
                st.info("Factor exposure requires at least 252 days of price data.")

        # ─────────── ATTRIBUTION ───────────
        with tab_attribution:
            st.subheader("Performance Attribution (Brinson-Hood-Beebower)")
            try:
                from src.portfolio_analytics.exposure_analysis import ASX_SECTOR_MAP
                from src.portfolio_analytics.performance_attribution import PerformanceAttributor

                bench_weights = {t: 1 / len(available) for t in available}
                attributor = PerformanceAttributor(
                    portfolio_weights=weights_available,
                    benchmark_weights=bench_weights,
                    asset_returns=asset_returns_clean,
                    sector_map=ASX_SECTOR_MAP,
                )
                attr_result = attributor.sector_attribution()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Active Return", f"{attr_result.total_active_return:.2%}")
                with col2:
                    st.metric("Allocation Effect", f"{attr_result.allocation_effect:.2%}")
                with col3:
                    st.metric("Selection Effect", f"{attr_result.selection_effect:.2%}")
                with col4:
                    st.metric("Interaction Effect", f"{attr_result.interaction_effect:.2%}")

                bm_return = float(
                    (1 + benchmark_series.reindex(port_returns.index).fillna(0)).prod() - 1
                ) if benchmark_series is not None else 0.0
                port_total = float((1 + port_returns).prod() - 1)

                fig_wf = build_attribution_waterfall(
                    allocation_effect=attr_result.allocation_effect,
                    selection_effect=attr_result.selection_effect,
                    interaction_effect=attr_result.interaction_effect,
                    benchmark_return=bm_return,
                    portfolio_return=port_total,
                )
                st.plotly_chart(fig_wf, use_container_width=True, key="attribution_waterfall")

                with st.expander("Attribution Detail Table"):
                    st.dataframe(attr_result.attribution_table, use_container_width=True)

            except Exception as e:
                st.warning(f"Attribution calculation error: {e}")

        # ─────────── STRESS TESTS ───────────
        with tab_stress:
            st.subheader("Stress Test Scenarios")

            stress_tab1, stress_tab2 = st.tabs(["Hypothetical Scenarios", "Historical Crises"])

            with stress_tab1:
                try:
                    fig_stress = build_stress_test_chart(stress_df)
                    st.plotly_chart(fig_stress, use_container_width=True, key="stress_chart")
                    st.dataframe(
                        stress_df.style.format({
                            "Portfolio Impact": "{:.2%}",
                            "Largest Loss": "{:.2%}",
                        }),
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"Stress test error: {e}")

            with stress_tab2:
                try:
                    from src.simulation.monte_carlo_paths import StressTester

                    tickers_for_stress = [t for t in weights_available if t in asset_returns_clean.columns]
                    w_for_stress = {t: weights_available[t] for t in tickers_for_stress}

                    st_obj = StressTester(
                        returns=asset_returns_clean[tickers_for_stress],
                        weights=w_for_stress,
                    )
                    hist_stress = st_obj.historical_stress_test()

                    if not hist_stress.empty:
                        st.markdown("#### Historical Crisis Periods — Real Return Data")
                        st.dataframe(
                            hist_stress.style.format({
                                "Portfolio Return": "{:.2%}",
                                "Max Drawdown": "{:.2%}",
                                "Volatility (ann.)": "{:.2%}",
                            }),
                            use_container_width=True,
                        )
                        fig_hist = go.Figure(go.Bar(
                            x=hist_stress.index,
                            y=hist_stress["Portfolio Return"] * 100,
                            marker_color=[
                                "#FF1744" if v < 0 else "#00C853"
                                for v in hist_stress["Portfolio Return"]
                            ],
                            text=[f"{v:.1%}" for v in hist_stress["Portfolio Return"]],
                            textposition="outside",
                        ))
                        fig_hist.update_layout(
                            template="plotly_dark", height=380,
                            title="Portfolio Return During Historical Crises",
                            yaxis=dict(title="Portfolio Return (%)", ticksuffix="%"),
                        )
                        st.plotly_chart(fig_hist, use_container_width=True, key="hist_stress_chart")
                    else:
                        st.info("Not enough historical data. Try extending your date range back to 2008.")

                except Exception as e:
                    st.warning(f"Historical stress test error: {e}")

        # ─────────── ADVANCED ───────────
        with tab_advanced:
            st.subheader("Advanced Analytics")

            try:
                st.markdown("#### Principal Component Analysis")
                fig_pca = build_pca_chart(pca)
                st.plotly_chart(fig_pca, use_container_width=True, key="pca_chart")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**PC Loadings**")
                    st.dataframe(
                        pca["loadings"].style.format("{:.4f}"),
                        use_container_width=True,
                    )
                with col2:
                    st.markdown("**Component Risk Share**")
                    st.dataframe(
                        pca["component_risk_pct"].to_frame().style.format("{:.2%}"),
                        use_container_width=True,
                    )
            except Exception as e:
                st.warning(f"PCA error: {e}")

            try:
                st.markdown("#### Rolling Risk Metrics")
                rolling_sharpe = pm.rolling_sharpe(window=63)
                rolling_vol = pm.rolling_volatility(window=21)
                rolling_beta = (
                    pm.rolling_beta(window=63) if benchmark_series is not None else None
                )
                fig_rolling = build_rolling_metrics_chart(
                    rolling_sharpe, rolling_vol, rolling_beta
                )
                st.plotly_chart(fig_rolling, use_container_width=True, key="rolling_metrics")
            except Exception as e:
                st.warning(f"Rolling metrics error: {e}")

            try:
                st.markdown("#### Risk Budget Analysis")
                risk_budget = rd.risk_budget_analysis()
                st.dataframe(
                    risk_budget.style.format({
                        "Actual Risk %": "{:.1%}",
                        "Target Risk %": "{:.1%}",
                        "Deviation": "{:.1%}",
                    }),
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"Risk budget error: {e}")

        # ─────────── MONTE CARLO ───────────
        with tab_mc:
            st.subheader("Monte Carlo Simulation")
            st.markdown("Simulate 1-year portfolio return distributions using different models.")

            col1, col2 = st.columns(2)
            with col1:
                mc_model = st.selectbox(
                    "Simulation Model",
                    ["GBM", "Jump Diffusion", "Bootstrap"],
                    key="mc_model_select",
                )
            with col2:
                n_sims = st.selectbox(
                    "Simulations",
                    [1000, 5000, 10000],
                    index=1,
                    key="mc_n_sims",
                )

            if st.button("▶ Run Simulation", key="mc_run_btn"):
                try:
                    from src.simulation.monte_carlo_paths import MonteCarloSimulator

                    mc = MonteCarloSimulator(port_returns, n_simulations=n_sims, horizon=252)

                    with st.spinner(f"Running {n_sims:,} simulations…"):
                        if mc_model == "GBM":
                            mc_result = mc.simulate_gbm()
                        elif mc_model == "Jump Diffusion":
                            mc_result = mc.simulate_jump_diffusion()
                        else:
                            mc_result = mc.simulate_bootstrap()

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Expected Return", f"{mc_result.expected_return:.2%}")
                    with col2:
                        st.metric("P(Loss)", f"{mc_result.prob_loss:.1%}")
                    with col3:
                        st.metric("VaR 95% (1Y)", f"{mc_result.var_95:.2%}")
                    with col4:
                        st.metric("CVaR 95% (1Y)", f"{mc_result.cvar_95:.2%}")

                    paths = mc_result.paths
                    t_axis = list(range(paths.shape[1]))
                    fig_mc = go.Figure()

                    sample_idx = np.random.choice(n_sims, min(100, n_sims), replace=False)
                    for i in sample_idx:
                        fig_mc.add_trace(go.Scatter(
                            x=t_axis, y=paths[i], mode="lines",
                            line=dict(color="rgba(0,212,255,0.06)", width=0.5),
                            showlegend=False,
                        ))

                    for pct, colour in [(5, "#FF1744"), (25, "#FF6B35"),
                                        (50, "#00D4FF"), (75, "#FF6B35"), (95, "#FF1744")]:
                        pvals = np.percentile(paths, pct, axis=0)
                        fig_mc.add_trace(go.Scatter(
                            x=t_axis, y=pvals, name=f"P{pct}",
                            line=dict(width=2, color=colour),
                        ))

                    fig_mc.update_layout(
                        template="plotly_dark", height=440,
                        title=f"Portfolio {mc_model} Simulation ({n_sims:,} paths)",
                        xaxis_title="Trading Days",
                        yaxis_title="Portfolio NAV ($100 start)",
                    )
                    st.plotly_chart(fig_mc, use_container_width=True, key="mc_fan_chart")

                    final_returns = mc_result.final_prices / 100 - 1
                    fig_dist = go.Figure(go.Histogram(
                        x=final_returns * 100, nbinsx=80,
                        marker_color="#00D4FF", opacity=0.7,
                    ))
                    fig_dist.add_vline(x=0, line_dash="dash", line_color="#FF1744",
                                       annotation_text="Break-even")
                    fig_dist.update_layout(
                        template="plotly_dark", height=340,
                        title="Distribution of 1-Year Returns",
                        xaxis=dict(title="Return (%)", ticksuffix="%"),
                    )
                    st.plotly_chart(fig_dist, use_container_width=True, key="mc_dist_chart")

                except Exception as e:
                    st.warning(f"Monte Carlo error: {e}")
            else:
                st.info("Click **▶ Run Simulation** to generate Monte Carlo paths.")

        # ─────────── OPTIMISE WEIGHTS ───────────
        with tab_opt:
            st.subheader("⚖️ Optimise Portfolio Weights")
            st.markdown(
                "Compare your current weights against mathematically optimal allocations "
                "using Markowitz mean-variance optimisation."
            )

            cov_method_opt = st.selectbox(
                "Covariance Method",
                ["ledoit_wolf", "sample", "ewma"],
                key="opt_cov_method",
            )

            if st.button("▶ Run Optimisation", key="opt_run_btn"):
                try:
                    from src.optimisation.markowitz_optimizer import MarkowitzOptimizer

                    with st.spinner("Optimising portfolios…"):
                        optimizer = MarkowitzOptimizer(
                            asset_returns_clean,
                            risk_free_rate=risk_free_rate,
                            cov_method=cov_method_opt,
                        )
                        comparison = optimizer.compare_portfolios()
                        max_sharpe = optimizer.max_sharpe()
                        min_vol = optimizer.min_volatility()
                        risk_par = optimizer.risk_parity()

                    st.markdown("#### Portfolio Comparison")
                    st.dataframe(
                        comparison.style.format({
                            "Expected Return": "{:.2%}",
                            "Volatility": "{:.2%}",
                            "Sharpe Ratio": "{:.3f}",
                            "Max Weight": "{:.1%}",
                            "Min Weight": "{:.1%}",
                        }),
                        use_container_width=True,
                    )

                    current_w = pd.Series(weights_available, name="Current")
                    weight_df = pd.concat([
                        current_w,
                        max_sharpe.weights.rename("Max Sharpe"),
                        min_vol.weights.rename("Min Volatility"),
                        risk_par.weights.rename("Risk Parity"),
                    ], axis=1).fillna(0)

                    fig_w = go.Figure()
                    for col, colour in zip(weight_df.columns, ["#90A4AE", "#00D4FF", "#FFD600", "#00C853"]):
                        fig_w.add_trace(go.Bar(
                            name=col, x=weight_df.index,
                            y=weight_df[col] * 100, marker_color=colour,
                        ))
                    fig_w.update_layout(
                        barmode="group", template="plotly_dark", height=420,
                        title="Weight Comparison: Current vs Optimal Portfolios",
                        yaxis=dict(title="Weight (%)", ticksuffix="%"),
                    )
                    st.plotly_chart(fig_w, use_container_width=True, key="opt_weight_chart")

                    with st.spinner("Tracing efficient frontier…"):
                        frontier = optimizer.efficient_frontier(n_points=40)
                        mc_ports = optimizer.monte_carlo_portfolios(n_portfolios=2000)

                    fig_ef = go.Figure()
                    fig_ef.add_trace(go.Scatter(
                        x=mc_ports["Volatility"] * 100, y=mc_ports["Return"] * 100,
                        mode="markers", name="Random Portfolios",
                        marker=dict(size=3, color=mc_ports["Sharpe"],
                                    colorscale="Viridis", opacity=0.4),
                    ))
                    fig_ef.add_trace(go.Scatter(
                        x=frontier["Volatility"] * 100, y=frontier["Return"] * 100,
                        name="Efficient Frontier", mode="lines",
                        line=dict(color="#FF6B35", width=3),
                    ))

                    curr_ret = float(
                        asset_returns_clean.mean() @
                        pd.Series(weights_available).reindex(asset_returns_clean.columns).fillna(0) * 252
                    )
                    curr_vol = float(optimizer.portfolio_stats(
                        pd.Series(weights_available).reindex(list(optimizer.tickers)).fillna(0).values
                    )[1])
                    fig_ef.add_trace(go.Scatter(
                        x=[curr_vol * 100], y=[curr_ret * 100],
                        mode="markers+text", name="Your Portfolio",
                        text=["Your Portfolio"], textposition="top right",
                        marker=dict(size=14, color="#FF1744", symbol="diamond"),
                    ))
                    fig_ef.add_trace(go.Scatter(
                        x=[max_sharpe.expected_volatility * 100],
                        y=[max_sharpe.expected_return * 100],
                        mode="markers+text", name="Max Sharpe",
                        text=["Max Sharpe"], textposition="top right",
                        marker=dict(size=14, color="#00D4FF", symbol="star"),
                    ))
                    fig_ef.update_layout(
                        template="plotly_dark", height=500,
                        title="Efficient Frontier — Your Portfolio vs Optimal",
                        xaxis=dict(title="Volatility (%)", ticksuffix="%"),
                        yaxis=dict(title="Expected Return (%)", ticksuffix="%"),
                    )
                    st.plotly_chart(fig_ef, use_container_width=True, key="opt_frontier_chart")

                except Exception as e:
                    st.warning(f"Optimisation error: {e}")
            else:
                st.info("Click **▶ Run Optimisation** to compare weight allocations.")

        # ─────────── REGIME ───────────
        with tab_regime:
            st.subheader("🌡️ Market Regime Detection")
            st.markdown(
                "Detects whether the market is in a Bull, Bear, or Sideways regime "
                "using multiple methods with a consensus vote."
            )

            try:
                from src.regime_detection.volatility_regimes import RegimeEngine

                market_series = price_data.mean(axis=1)
                market_returns_series = market_series.pct_change().dropna()

                with st.spinner("Detecting market regimes…"):
                    engine = RegimeEngine(returns=market_returns_series, prices=market_series)
                    current = engine.current_regime()

                REGIME_COLOURS_MAP = {
                    "Bull": "#00C853",
                    "Bear": "#FF1744",
                    "Sideways": "#FFD600",
                }
                regime_colour = REGIME_COLOURS_MAP.get(current["consensus"], "#90A4AE")

                st.markdown(
                    f"<h2 style='color:{regime_colour}'>Current Regime: "
                    f"{current['consensus']}</h2>",
                    unsafe_allow_html=True,
                )

                col1, col2, col3 = st.columns(3)
                for col, (regime, votes) in zip(
                    [col1, col2, col3], current["votes"].items()
                ):
                    with col:
                        total = sum(current["votes"].values())
                        pct = votes / total if total > 0 else 0
                        st.metric(regime, f"{pct:.0%}", f"{votes} signal(s)")

                st.markdown("#### Individual Signal Breakdown")
                signal_rows = [{"Signal": k, "Value": v} for k, v in current["signals"].items()]
                if signal_rows:
                    st.dataframe(pd.DataFrame(signal_rows), use_container_width=True)

                with st.spinner("Computing regime statistics…"):
                    regime_stats = engine.regime_statistics()

                st.markdown("#### Historical Regime Statistics")
                if not regime_stats.empty:
                    st.dataframe(
                        regime_stats.style.format({
                            "Ann. Return": "{:.2%}",
                            "Ann. Volatility": "{:.2%}",
                            "Sharpe": "{:.3f}",
                            "% of Time": "{:.1%}",
                            "Best Day": "{:.2%}",
                            "Worst Day": "{:.2%}",
                        }),
                        use_container_width=True,
                    )

                regime_history = engine.regime_history(use_hmm=False)
                if "Consensus" in regime_history.columns:
                    consensus_series = regime_history["Consensus"].reindex(
                        market_series.index
                    ).ffill()

                    fig_reg = go.Figure()
                    fig_reg.add_trace(go.Scatter(
                        x=market_series.index, y=market_series.values,
                        name="Portfolio Avg Price",
                        line=dict(color="#00D4FF", width=1.5),
                    ))

                    for regime, colour in REGIME_COLOURS_MAP.items():
                        mask = consensus_series == regime
                        if not mask.any():
                            continue
                        dates_in_regime = market_series.index[mask]
                        if len(dates_in_regime) > 0:
                            fig_reg.add_vrect(
                                x0=dates_in_regime[0],
                                x1=dates_in_regime[-1],
                                fillcolor=colour,
                                opacity=0.10,
                                layer="below",
                                line_width=0,
                            )

                    fig_reg.update_layout(
                        template="plotly_dark", height=420,
                        title="Market Regime History",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_reg, use_container_width=True, key="regime_history_chart")

            except Exception as e:
                st.warning(f"Regime detection error: {e}")
                st.info(
                    "Regime detection requires `src/regime_detection/` to be in place."
                )

    # ── FOOTER ──
    st.divider()
    st.caption(
        "ASX Quant Terminal · Portfolio Analytics · "
        f"Data via Yahoo Finance · Period: {start_date} → {end_date}"
    )


# ─────────────────────────────────────────────
#  Standalone entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if HAS_STREAMLIT:
        st.set_page_config(
            page_title="ASX Portfolio Analytics",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        run_portfolio_dashboard()
    else:
        print("Install streamlit: pip install streamlit")
        print("Then run: streamlit run portfolio_dashboard.py")
