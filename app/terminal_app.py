"""
terminal_app.py
===============
ASX Quantitative Research Terminal
===================================
Main Streamlit application integrating all platform modules.

This is the single entry point for the entire quant research platform.

Run with:
    streamlit run app/terminal_app.py

Modules Integrated
------------------
    Data          : ASX price data + macro data ingestion
    Analytics     : Technical indicators, statistical tests, volatility models
    Factor Models : CAPM, Fama-French, momentum/value/quality factors
    Regime        : HMM, Markov switching, volatility/trend regime detection
    Strategies    : Momentum, mean reversion, pairs, factor, regime-adaptive
    Backtesting   : Full vectorised backtest with transaction costs
    Optimisation  : Markowitz, Black-Litterman, risk parity
    Simulation    : Monte Carlo paths, stress testing
    ML            : Feature engineering, RF/GB/XGB alpha models
    Options       : Black-Scholes pricing, Greeks, Monte Carlo options
    Portfolio     : Full portfolio analytics dashboard
    Parameter Opt : Grid, random, Bayesian optimisation
"""

from __future__ import annotations

import sys
import os
import warnings

# ── Path setup ──
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_APP_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Page configuration — MUST be first st call
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="ASX Quant Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00D4FF, #7B2FBE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #90A4AE;
        font-size: 0.95rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1E1E2E;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #2D2D3E;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem;
        font-weight: 500;
    }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Sidebar — global configuration
# ─────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render the global sidebar and return configuration dict."""

    with st.sidebar:
        st.markdown("## ⚙️ Terminal Config")
        st.divider()

        # Universe
        st.markdown("**ASX Universe**")
        default_tickers = "BHP.AX\nCBA.AX\nCSL.AX\nWES.AX\nMQG.AX\nRIO.AX\nWOW.AX\nANZ.AX"
        tickers_raw = st.text_area(
            "Tickers (one per line)",
            value=default_tickers,
            height=160,
            label_visibility="collapsed",
        )
        tickers = [t.strip().upper() for t in tickers_raw.split("\n") if t.strip()]

        # Date range
        st.markdown("**Date Range**")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=pd.Timestamp("2019-01-01"))
        with col2:
            end_date = st.date_input("End", value=pd.Timestamp.today())

        # Benchmark
        benchmark = st.selectbox(
            "Benchmark", ["^AXJO", "^AXKO", "^AORD", "^GSPC"], index=0
        )

        # Risk-free rate
        rfr = st.number_input(
            "Risk-Free Rate (p.a.)",
            min_value=0.0, max_value=0.15,
            value=0.0435, step=0.0025, format="%.4f",
        )

        st.divider()
        load_btn = st.button("📥 Load Data", type="primary", use_container_width=True)

    return {
        "tickers": tickers,
        "start": str(start_date),
        "end": str(end_date),
        "benchmark": benchmark,
        "rfr": rfr,
        "load": load_btn,
    }


# ─────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(tickers: list, start: str, end: str, benchmark: str):
    """Load and cache market data."""
    import yfinance as yf

    all_t = tickers + [benchmark]
    raw = yf.download(all_t, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw.copy()

    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    prices = prices.dropna(how="all")

    bm_series = None
    if benchmark in prices.columns:
        bm_series = prices[benchmark].pct_change().dropna()
        bm_series.name = benchmark
        prices = prices.drop(columns=[benchmark])

    avail = [t for t in tickers if t in prices.columns]
    return prices[avail], bm_series, avail


# ─────────────────────────────────────────────
#  Page renderers
# ─────────────────────────────────────────────

def page_market_overview(prices: pd.DataFrame, benchmark: pd.Series, rfr: float):
    """Market overview: price charts, returns, correlations."""
    st.subheader("📊 Market Overview")

    returns = prices.pct_change().dropna()

    # Normalised price chart
    st.markdown("#### Normalised Price Performance")
    norm = prices / prices.bfill().iloc[0] * 100
    fig = go.Figure()
    colours = px.colors.qualitative.Set2
    for i, col in enumerate(norm.columns):
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[col],
            name=col, line=dict(width=1.5, color=colours[i % len(colours)]),
        ))
    if benchmark is not None:
        bm_nav = (1 + benchmark).cumprod() * 100
        bm_nav = bm_nav.reindex(norm.index).ffill()
        fig.add_trace(go.Scatter(
            x=bm_nav.index, y=bm_nav.values,
            name="Benchmark", line=dict(dash="dot", color="#FF6B35", width=2),
        ))
    fig.update_layout(template="plotly_dark", height=420, hovermode="x unified",
                      title="Cumulative Performance (base=100)")
    st.plotly_chart(fig, use_container_width=True, key="market_overview_price")

    # Returns stats table
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Returns Summary")
        stats = pd.DataFrame({
            "CAGR": ((1 + returns).prod() ** (252 / len(returns)) - 1),
            "Ann. Vol": returns.std() * np.sqrt(252),
            "Sharpe": returns.mean() / returns.std() * np.sqrt(252),
            "Max DD": returns.apply(
                lambda r: ((1 + r).cumprod() /
                           (1 + r).cumprod().cummax() - 1).min()
            ),
        })
        st.dataframe(
            stats.style.format("{:.2%}"),
            use_container_width=True,
        )

    with col2:
        st.markdown("#### Correlation Matrix")
        corr = returns.corr()
        fig_c = px.imshow(
            corr, color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, text_auto=".2f",
            template="plotly_dark",
        )
        fig_c.update_layout(height=320)
        st.plotly_chart(fig_c, use_container_width=True, key="market_corr")


def page_technical_analysis(prices: pd.DataFrame, tickers: list):
    """Technical indicators for a selected ticker."""
    st.subheader("📉 Technical Analysis")

    ticker = st.selectbox("Select Ticker", tickers, key="ta_ticker")
    if ticker not in prices.columns:
        st.warning("Ticker not available.")
        return

    from src.analytics.technical_indicators import TechnicalIndicators

    price_df = pd.DataFrame({
        "Open": prices[ticker], "High": prices[ticker],
        "Low": prices[ticker], "Close": prices[ticker],
        "Volume": pd.Series(1e6, index=prices.index),
    })
    ti = TechnicalIndicators(price_df, ticker=ticker)
    indicators = ti.compute_all()

    # Price + MA chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices[ticker], name=ticker,
        line=dict(color="#00D4FF", width=1.5),
    ))
    for ma, color in [(20, "#FFD600"), (50, "#FF6B35"), (200, "#EF5350")]:
        col = f"SMA_{ma}"
        if col in indicators.columns:
            fig.add_trace(go.Scatter(
                x=indicators.index, y=indicators[col],
                name=f"SMA {ma}", line=dict(width=1, dash="dot", color=color),
            ))
    fig.update_layout(template="plotly_dark", height=380,
                      title=f"{ticker} Price + Moving Averages", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, key="ta_price")

    # Sub-indicators
    col1, col2 = st.columns(2)
    with col1:
        if "RSI_14" in indicators.columns:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=indicators.index, y=indicators["RSI_14"],
                name="RSI 14", line=dict(color="#00D4FF"),
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="#FF1744")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00C853")
            fig_rsi.update_layout(template="plotly_dark", height=260,
                                   title="RSI (14)", yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_rsi, use_container_width=True, key="ta_rsi")

    with col2:
        if "MACD" in indicators.columns:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=indicators.index, y=indicators["MACD"],
                name="MACD", line=dict(color="#00D4FF"),
            ))
            fig_macd.add_trace(go.Scatter(
                x=indicators.index, y=indicators["MACD_Signal"],
                name="Signal", line=dict(color="#FF6B35", dash="dot"),
            ))
            hist = indicators["MACD_Hist"]
            fig_macd.add_trace(go.Bar(
                x=indicators.index, y=hist,
                name="Histogram",
                marker_color=np.where(hist >= 0, "#00C853", "#FF1744"),
            ))
            fig_macd.update_layout(template="plotly_dark", height=260, title="MACD")
            st.plotly_chart(fig_macd, use_container_width=True, key="ta_macd")

    # Bollinger Bands
    if all(c in indicators.columns for c in ["BB_Upper", "BB_Mid", "BB_Lower"]):
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(
            x=prices.index, y=prices[ticker], name=ticker,
            line=dict(color="#00D4FF"),
        ))
        fig_bb.add_trace(go.Scatter(
            x=indicators.index, y=indicators["BB_Upper"],
            name="Upper", line=dict(color="#FFD600", dash="dot"),
        ))
        fig_bb.add_trace(go.Scatter(
            x=indicators.index, y=indicators["BB_Lower"],
            name="Lower", line=dict(color="#FFD600", dash="dot"),
            fill="tonexty", fillcolor="rgba(255, 214, 0, 0.05)",
        ))
        fig_bb.update_layout(template="plotly_dark", height=320,
                              title="Bollinger Bands (20, 2σ)", hovermode="x unified")
        st.plotly_chart(fig_bb, use_container_width=True, key="ta_bb")


def page_factor_analysis(prices: pd.DataFrame, benchmark: pd.Series, rfr: float):
    """CAPM and Fama-French factor analysis."""
    st.subheader("🔬 Factor Analysis")

    returns = prices.pct_change().dropna()

    tab1, tab2, tab3 = st.tabs(["CAPM", "Fama-French 3F", "Factor Scores"])

    with tab1:
        st.markdown("#### CAPM Regression")
        if benchmark is None:
            st.warning("Benchmark required for CAPM analysis.")
        else:
            from src.factor_models.capm import CAPM
            with st.spinner("Running CAPM…"):
                capm = CAPM(returns, benchmark, risk_free_rate=rfr)
                capm_df = capm.fit_to_dataframe()

            st.dataframe(
                capm_df.style.format({
                    "Alpha (ann.)": "{:.2%}",
                    "Beta": "{:.3f}",
                    "R-Squared": "{:.3f}",
                    "Expected Return": "{:.2%}",
                    "Actual Return": "{:.2%}",
                }),
                use_container_width=True,
            )

            # Security Market Line
            sml = capm.security_market_line()
            fig = px.scatter(
                sml, x="Beta", y="Actual Return",
                text="Ticker", color="Alpha",
                color_continuous_scale="RdYlGn",
                template="plotly_dark",
                title="Security Market Line",
            )
            betas = np.linspace(sml["Beta"].min() * 0.8, sml["Beta"].max() * 1.2, 100)
            mkt_excess = (benchmark.mean() * 252) - rfr
            sml_line = rfr + betas * mkt_excess
            fig.add_trace(go.Scatter(
                x=betas, y=sml_line, name="SML",
                line=dict(dash="dash", color="#90A4AE"),
            ))
            fig.update_layout(height=480)
            st.plotly_chart(fig, use_container_width=True, key="factor_sml")

    with tab2:
        st.markdown("#### Fama-French 3-Factor Model")
        from src.factor_models.fama_french import FamaFrench
        with st.spinner("Running Fama-French…"):
            ff = FamaFrench(returns, benchmark, universe_prices=prices, risk_free_rate=rfr)
            ff_df = ff.fit_to_dataframe()

        st.dataframe(
            ff_df.style.format({
                "Alpha (ann.)": "{:.2%}",
                "β MKT": "{:.3f}", "β SMB": "{:.3f}", "β HML": "{:.3f}",
                "R²": "{:.3f}",
            }),
            use_container_width=True,
        )

        st.markdown("#### Factor Return Summary")
        factor_stats = ff.factor_summary()
        st.dataframe(
            factor_stats.style.format({
                "Ann. Return": "{:.2%}",
                "Ann. Volatility": "{:.2%}",
                "Sharpe": "{:.3f}",
            }),
            use_container_width=True,
        )

    with tab3:
        st.markdown("#### Latest Factor Scores")
        from src.factor_models.momentum_factor import FactorLibrary
        with st.spinner("Computing factor scores…"):
            fl = FactorLibrary(prices)
            scores = fl.get_latest_factor_scores()

        st.dataframe(
            scores.style.format("{:.3f}"),
            use_container_width=True,
        )


def page_regime_detection(prices: pd.DataFrame, benchmark: pd.Series):
    """Market regime detection."""
    st.subheader("🌡️ Regime Detection")

    returns = prices.mean(axis=1).pct_change().dropna()
    market_prices = prices.mean(axis=1)

    col1, col2 = st.columns([1, 3])
    with col1:
        n_states = st.selectbox("HMM States", [2, 3, 4], index=1)
        method = st.selectbox("Method", ["HMM", "Markov Switching", "Rule-Based"])

    with col2:
        from src.regime_detection.volatility_regimes import RegimeEngine

        with st.spinner("Detecting regimes…"):
            engine = RegimeEngine(returns, prices=market_prices)
            current = engine.current_regime()

        st.success(
            f"**Current Regime: {current['consensus']}**  "
            f"(Confidence: {current['confidence']:.0%})"
        )

    if method == "HMM":
        from src.regime_detection.hidden_markov_models import HiddenMarkovModel
        with st.spinner(f"Fitting HMM ({n_states} states)…"):
            hmm = HiddenMarkovModel(returns, n_states=n_states)
            result = hmm.fit()

        regime_labels = result.regime_labels.reindex(prices.index).ffill()

        COLOURS = {"Bull": "#00C853", "Sideways": "#FFD600",
                   "Bear": "#FF1744", "High Vol": "#FF6B35"}

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prices.index, y=market_prices.reindex(prices.index),
            name="Market", line=dict(color="#00D4FF"),
        ))

        for regime in regime_labels.unique():
            mask = regime_labels == regime
            idx = prices.index[mask]
            if len(idx) == 0:
                continue
            fig.add_vrect(
                x0=idx[0], x1=idx[-1],
                fillcolor=COLOURS.get(str(regime), "#90A4AE"),
                opacity=0.12, layer="below", line_width=0,
            )

        fig.update_layout(
            template="plotly_dark", height=420,
            title=f"HMM Regime Detection ({n_states} states)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True, key="regime_chart")

        st.markdown("#### Regime Statistics")
        st.dataframe(result.state_stats.style.format({
            "Mean Return (ann.)": "{:.2%}",
            "Volatility (ann.)": "{:.2%}",
            "Sharpe": "{:.3f}",
            "% of Time": "{:.1%}",
        }), use_container_width=True)

        st.markdown("#### State Probabilities")
        probs = hmm.state_probabilities()
        fig_p = px.area(probs, template="plotly_dark", height=280,
                        title="Smoothed Regime Probabilities")
        st.plotly_chart(fig_p, use_container_width=True, key="regime_probs")

    else:
        with st.spinner("Detecting regimes…"):
            regime_df = engine.regime_history(use_hmm=False)

        st.dataframe(regime_df.tail(20), use_container_width=True)

        st.markdown("#### Regime Statistics")
        stats = engine.regime_statistics()
        st.dataframe(stats.style.format({
            "Ann. Return": "{:.2%}", "Ann. Volatility": "{:.2%}",
            "Sharpe": "{:.3f}", "% of Time": "{:.1%}",
        }), use_container_width=True)


def page_strategy_backtest(prices: pd.DataFrame, benchmark: pd.Series, rfr: float):
    """Strategy selection, configuration, and backtesting."""
    st.subheader("⚡ Strategy Backtester")

    col1, col2, col3 = st.columns(3)
    with col1:
        strategy_name = st.selectbox("Strategy", [
            "Cross-Sectional Momentum",
            "Time-Series Momentum",
            "Risk-Adjusted Momentum",
            "Bollinger Band Reversion",
            "RSI Mean Reversion",
            "Factor Strategy",
            "Regime Adaptive",
        ])
    with col2:
        initial_capital = st.number_input(
            "Capital (AUD)", min_value=10_000,
            value=1_000_000, step=100_000, format="%d"
        )
    with col3:
        commission = st.number_input(
            "Commission", min_value=0.0, max_value=0.01,
            value=0.001, step=0.0005, format="%.4f",
        )

    # Strategy params
    with st.expander("⚙️ Strategy Parameters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_long = st.slider("Long Positions", 1, 20, 10)
            lookback = st.selectbox("Lookback (days)", [63, 126, 252, 504], index=2)
        with col2:
            skip = st.selectbox("Skip Period (days)", [0, 5, 21], index=2)
            rebalance = st.selectbox("Rebalance", ["monthly", "weekly", "quarterly"])
        with col3:
            signal_lag = st.selectbox("Signal Lag (days)", [1, 2, 3], index=0)

    run_btn = st.button("▶ Run Backtest", type="primary")

    if not run_btn:
        st.info("Configure parameters above and click **▶ Run Backtest**.")
        return

    with st.spinner(f"Running {strategy_name} backtest…"):
        try:
            from src.backtesting.backtester import BacktestConfig, Backtester

            config = BacktestConfig(
                initial_capital=float(initial_capital),
                commission=commission,
                slippage=0.0005,
                signal_lag=signal_lag,
                risk_free_rate=rfr,
            )

            if strategy_name == "Cross-Sectional Momentum":
                from src.strategies.momentum_strategy import CrossSectionalMomentum
                s = CrossSectionalMomentum(
                    prices, lookback=lookback, skip=skip,
                    n_long=n_long, rebalance_freq=rebalance,
                )
                signals = s.generate_signals()

            elif strategy_name == "Time-Series Momentum":
                from src.strategies.momentum_strategy import TimeSeriesMomentum
                s = TimeSeriesMomentum(prices, lookback=lookback)
                signals = s.generate_signals()

            elif strategy_name == "Risk-Adjusted Momentum":
                from src.strategies.momentum_strategy import RiskAdjustedMomentum
                s = RiskAdjustedMomentum(prices, lookback=lookback, n_long=n_long)
                signals = s.generate_signals()

            elif strategy_name == "Bollinger Band Reversion":
                from src.strategies.mean_reversion_strategy import BollingerBandReversion
                s = BollingerBandReversion(prices)
                signals = s.generate_signals()

            elif strategy_name == "RSI Mean Reversion":
                from src.strategies.mean_reversion_strategy import RSIMeanReversion
                s = RSIMeanReversion(prices)
                signals = s.generate_signals()

            elif strategy_name == "Factor Strategy":
                from src.strategies.factor_strategy import FactorStrategy
                s = FactorStrategy(prices, n_long=n_long, rebalance_freq=rebalance)
                signals = s.generate_signals()

            else:
                from src.strategies.factor_strategy import RegimeAdaptiveStrategy
                market_p = prices.mean(axis=1)
                s = RegimeAdaptiveStrategy(prices, market_p, n_positions=n_long)
                signals = s.generate_signals()

            if signals.empty:
                st.error("Signal generation returned empty DataFrame.")
                return

            bt = Backtester(prices=prices, signals=signals, config=config,
                            benchmark=benchmark, strategy_name=strategy_name)
            result = bt.run()

        except Exception as e:
            st.error(f"Backtest failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    # Results
    if result.returns.empty or len(result.returns) < 5:
        st.warning(
            "Strategy returned no trades. This usually means the signal "
            "generator needs more historical data than the selected date range "
            "provides. Try extending the start date back to 2015 or earlier."
        )
        return

    from src.backtesting.performance_metrics import PerformanceMetrics
    pm = PerformanceMetrics(result)

    # Guard against degenerate results
    cagr = pm.cagr()
    sharpe = pm.sharpe_ratio()
    max_dd = pm.max_drawdown()
    sortino = pm.sortino_ratio()
    win_rate = pm.win_rate()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("CAGR", f"{cagr:.2%}")
    with col2:
        st.metric("Sharpe", f"{sharpe:.3f}")
    with col3:
        st.metric("Max DD", f"{max_dd:.2%}")
    with col4:
        st.metric("Sortino", f"{sortino:.3f}")
    with col5:
        st.metric("Win Rate", f"{win_rate:.1%}")

    # Equity curve
    nav = result.nav
    fig_nav = go.Figure()
    fig_nav.add_trace(go.Scatter(
        x=nav.index, y=nav.values, name=strategy_name,
        line=dict(color="#00D4FF", width=2),
    ))
    if benchmark is not None:
        bm_nav = (1 + benchmark.reindex(nav.index).fillna(0)).cumprod() * float(initial_capital)
        fig_nav.add_trace(go.Scatter(
            x=bm_nav.index, y=bm_nav.values, name="Benchmark",
            line=dict(color="#FF6B35", dash="dot", width=1.5),
        ))
    fig_nav.update_layout(template="plotly_dark", height=400,
                          title="Strategy Equity Curve", hovermode="x unified")
    st.plotly_chart(fig_nav, use_container_width=True, key="bt_nav")

    # Drawdown
    dd = pm.drawdown_series()
    fig_dd = go.Figure(go.Scatter(
        x=dd.index, y=dd.values * 100, name="Drawdown",
        line=dict(color="#FF1744"), fill="tozeroy",
        fillcolor="rgba(255,23,68,0.1)",
    ))
    fig_dd.update_layout(template="plotly_dark", height=280,
                         title="Drawdown (%)", yaxis_ticksuffix="%")
    st.plotly_chart(fig_dd, use_container_width=True, key="bt_dd")

    with st.expander("📋 Full Metrics Table"):
        st.dataframe(result.metrics, use_container_width=True, height=600)

    with st.expander("📅 Monthly Returns Heatmap"):
        try:
            monthly = pm.monthly_returns()
            fig_heat = px.imshow(
                monthly.fillna(0) * 100,
                color_continuous_scale="RdYlGn", text_auto=".1f",
                template="plotly_dark", zmin=-10, zmax=10,
                title="Monthly Returns (%)",
            )
            st.plotly_chart(fig_heat, use_container_width=True, key="bt_monthly")
        except Exception:
            pass


def page_portfolio_optimisation(prices: pd.DataFrame, rfr: float):
    """Portfolio optimisation with Markowitz and Black-Litterman."""
    st.subheader("⚖️ Portfolio Optimisation")

    returns = prices.pct_change().dropna()

    tab1, tab2 = st.tabs(["Markowitz", "Black-Litterman"])

    with tab1:
        from src.optimisation.markowitz_optimizer import MarkowitzOptimizer

        cov_method = st.selectbox(
            "Covariance Method", ["ledoit_wolf", "sample", "ewma"]
        )

        with st.spinner("Optimising portfolios…"):
            opt = MarkowitzOptimizer(returns, risk_free_rate=rfr, cov_method=cov_method)
            comparison = opt.compare_portfolios()
            frontier = opt.efficient_frontier(n_points=50)
            mc_ports = opt.monte_carlo_portfolios(n_portfolios=3000)

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

        # Efficient frontier chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mc_ports["Volatility"] * 100, y=mc_ports["Return"] * 100,
            mode="markers", name="Random Portfolios",
            marker=dict(size=3, color=mc_ports["Sharpe"],
                        colorscale="Viridis", opacity=0.5),
        ))
        fig.add_trace(go.Scatter(
            x=frontier["Volatility"] * 100, y=frontier["Return"] * 100,
            name="Efficient Frontier", mode="lines",
            line=dict(color="#FF6B35", width=3),
        ))

        # Special portfolios
        for port_name, color in [("Max Sharpe", "#00D4FF"), ("Min Volatility", "#FFD600")]:
            try:
                if "Max Sharpe" in port_name:
                    r = opt.max_sharpe()
                else:
                    r = opt.min_volatility()
                fig.add_trace(go.Scatter(
                    x=[r.expected_volatility * 100],
                    y=[r.expected_return * 100],
                    mode="markers+text",
                    name=port_name,
                    text=[port_name],
                    textposition="top center",
                    marker=dict(size=12, color=color, symbol="star"),
                ))
            except Exception:
                pass

        fig.update_layout(
            template="plotly_dark", height=500,
            title="Efficient Frontier",
            xaxis=dict(title="Volatility (%)", ticksuffix="%"),
            yaxis=dict(title="Expected Return (%)", ticksuffix="%"),
        )
        st.plotly_chart(fig, use_container_width=True, key="opt_frontier")

        # Weight allocation
        st.markdown("#### Max Sharpe Weights")
        ms = opt.max_sharpe()
        w = ms.weights[ms.weights > 0.001].sort_values(ascending=False)
        fig_w = px.bar(
            x=w.index, y=w.values * 100,
            template="plotly_dark", title="Portfolio Weights (%)",
            labels={"x": "Ticker", "y": "Weight (%)"},
        )
        st.plotly_chart(fig_w, use_container_width=True, key="opt_weights")

    with tab2:
        st.markdown("#### Black-Litterman Optimisation")
        st.markdown(
            "Express views on expected returns. The BL model blends them "
            "with market equilibrium to produce stable posterior returns."
        )

        from src.optimisation.black_litterman import BlackLitterman

        n_views = st.number_input("Number of Views", min_value=0, max_value=5, value=2)

        views = []
        for i in range(int(n_views)):
            st.markdown(f"**View {i+1}**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                vtype = st.selectbox(
                    "Type", ["Absolute", "Relative"], key=f"vtype_{i}"
                )
            with c2:
                t1 = st.selectbox(
                    "Ticker (Long)", prices.columns.tolist(), key=f"t1_{i}"
                )
            with c3:
                t2 = (
                    st.selectbox("Ticker (Short)", prices.columns.tolist(), key=f"t2_{i}")
                    if vtype == "Relative" else None
                )
            with c4:
                ret_view = st.number_input(
                    "Return (%)", value=12.0, step=1.0, key=f"ret_{i}"
                ) / 100
                conf = st.slider("Confidence", 0.1, 0.9, 0.5, key=f"conf_{i}")

            views.append((vtype, t1, t2, ret_view, conf))

        if st.button("▶ Run BL Optimisation", key="bl_run"):
            with st.spinner("Running Black-Litterman…"):
                bl = BlackLitterman(returns, risk_free_rate=rfr)
                for vtype, t1, t2, ret_view, conf in views:
                    try:
                        if vtype == "Absolute":
                            bl.add_absolute_view(t1, ret_view, conf)
                        else:
                            bl.add_relative_view(t1, t2, ret_view, conf)
                    except Exception:
                        pass

                bl_result = bl.optimise()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Expected Return", f"{bl_result.expected_return:.2%}")
            with col2:
                st.metric("Expected Vol", f"{bl_result.expected_volatility:.2%}")
            with col3:
                st.metric("Sharpe Ratio", f"{bl_result.sharpe_ratio:.3f}")

            ev = bl.equilibrium_vs_posterior()
            fig_bl = go.Figure()
            fig_bl.add_trace(go.Bar(
                name="Equilibrium", x=ev.index,
                y=ev["Equilibrium Return"] * 100,
                marker_color="#90A4AE",
            ))
            fig_bl.add_trace(go.Bar(
                name="Posterior", x=ev.index,
                y=ev["Posterior Return"] * 100,
                marker_color="#00D4FF",
            ))
            fig_bl.update_layout(
                barmode="group", template="plotly_dark", height=380,
                title="Equilibrium vs BL Posterior Returns (%)",
                yaxis_ticksuffix="%",
            )
            st.plotly_chart(fig_bl, use_container_width=True, key="bl_returns")


def page_monte_carlo(prices: pd.DataFrame):
    """Monte Carlo simulation and stress testing."""
    st.subheader("🎲 Monte Carlo Simulation")

    ticker = st.selectbox("Asset to Simulate", prices.columns.tolist(), key="mc_ticker")
    returns_series = prices[ticker].pct_change().dropna()

    col1, col2, col3 = st.columns(3)
    with col1:
        n_sim = st.selectbox("Simulations", [1000, 5000, 10000, 50000], index=1)
    with col2:
        horizon = st.selectbox("Horizon (days)", [21, 63, 126, 252], index=3)
    with col3:
        model = st.selectbox("Model", ["GBM", "Jump Diffusion", "Bootstrap"])

    if st.button("▶ Run Simulation", key="mc_run"):
        from src.simulation.monte_carlo_paths import MonteCarloSimulator

        mc = MonteCarloSimulator(returns_series, n_simulations=n_sim, horizon=horizon)

        with st.spinner(f"Running {n_sim:,} simulations…"):
            if model == "GBM":
                result = mc.simulate_gbm()
            elif model == "Jump Diffusion":
                result = mc.simulate_jump_diffusion()
            else:
                result = mc.simulate_bootstrap()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Expected Return", f"{result.expected_return:.2%}")
        with col2:
            st.metric("P(Loss)", f"{result.prob_loss:.1%}")
        with col3:
            st.metric("VaR 95%", f"{result.var_95:.2%}")
        with col4:
            st.metric("CVaR 95%", f"{result.cvar_95:.2%}")

        # Fan chart — sample 200 paths
        sample_idx = np.random.choice(n_sim, min(200, n_sim), replace=False)
        paths_sample = result.paths[sample_idx]
        t_axis = list(range(horizon + 1))

        fig = go.Figure()
        for path in paths_sample[:50]:
            fig.add_trace(go.Scatter(
                x=t_axis, y=path, mode="lines",
                line=dict(color="rgba(0,212,255,0.08)", width=0.5),
                showlegend=False,
            ))

        # Percentile bands
        for p, c in [(5, "#FF1744"), (25, "#FF6B35"), (50, "#00D4FF"),
                     (75, "#FF6B35"), (95, "#FF1744")]:
            pct_vals = np.percentile(result.paths, p, axis=0)
            fig.add_trace(go.Scatter(
                x=t_axis, y=pct_vals,
                name=f"P{p}", line=dict(width=2),
                line_color=c,
            ))

        fig.update_layout(
            template="plotly_dark", height=450,
            title=f"{ticker} {model} Simulation ({n_sim:,} paths, {horizon} days)",
            xaxis_title="Days", yaxis_title="Price",
        )
        st.plotly_chart(fig, use_container_width=True, key="mc_paths")

        st.markdown("#### Distribution of Final Prices")
        fig_hist = px.histogram(
            result.final_prices, nbins=100,
            template="plotly_dark", title="Final Price Distribution",
        )
        fig_hist.add_vline(
            x=100, line_dash="dash", line_color="#FFD600",
            annotation_text="Start",
        )
        st.plotly_chart(fig_hist, use_container_width=True, key="mc_hist")


def page_options_pricer(prices: pd.DataFrame, rfr: float):
    """Black-Scholes and Monte Carlo options pricing."""
    st.subheader("💹 Options Pricer")

    from src.options.black_scholes import BlackScholes, MonteCarloOptionPricer

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.selectbox("Underlying", prices.columns.tolist(), key="opt_ticker")
        S = float(prices[ticker].iloc[-1])
        st.metric("Current Spot", f"${S:.2f}")
        K = st.number_input("Strike (K)", value=round(S, 0), step=0.5)
        expiry_days = st.slider("Days to Expiry", 1, 730, 90)
        T = expiry_days / 365

    with col2:
        sigma = st.slider("Implied Volatility (%)", 5, 100, 25) / 100
        option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)
        q = st.number_input("Dividend Yield", 0.0, 0.10, 0.03, step=0.005)

    bs = BlackScholes(S=S, K=K, T=T, r=rfr, sigma=sigma, q=q)
    greeks = bs.greeks(option_type.lower())
    price = bs.price(option_type.lower())

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Option Price", f"${price:.4f}")
    c2.metric("Delta", f"{greeks.delta:.4f}")
    c3.metric("Gamma", f"{greeks.gamma:.6f}")
    c4.metric("Theta/day", f"${greeks.theta:.4f}")
    c5.metric("Vega/1%", f"${greeks.vega:.4f}")

    # P&L diagram
    st.markdown("#### Payoff at Expiry")
    S_range = np.linspace(S * 0.5, S * 1.5, 300)
    if option_type == "Call":
        payoff = np.maximum(S_range - K, 0) - price
    else:
        payoff = np.maximum(K - S_range, 0) - price

    fig = go.Figure()
    fill_colour = "rgba(0,200,83,0.1)" if payoff.mean() >= 0 else "rgba(255,23,68,0.1)"
    fig.add_trace(go.Scatter(
        x=S_range, y=payoff, name=f"{option_type} P&L",
        line=dict(color="#00D4FF", width=2),
        fill="tozeroy",
        fillcolor=fill_colour,
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#90A4AE")
    fig.add_vline(x=S, line_dash="dot", line_color="#FFD600",
                  annotation_text=f"Spot ${S:.2f}")
    fig.add_vline(x=K, line_dash="dot", line_color="#FF6B35",
                  annotation_text=f"Strike ${K:.2f}")
    fig.update_layout(
        template="plotly_dark", height=380,
        title=f"{option_type} Option P&L at Expiry",
        xaxis_title="Underlying Price ($)",
        yaxis_title="P&L ($)",
    )
    st.plotly_chart(fig, use_container_width=True, key="opt_payoff")

    # Greeks vs spot
    with st.expander("📊 Greeks Sensitivity"):
        S_arr = np.linspace(S * 0.7, S * 1.3, 200)
        deltas = [BlackScholes(s, K, T, rfr, sigma, q).delta(option_type.lower())
                  for s in S_arr]
        gammas = [BlackScholes(s, K, T, rfr, sigma, q).gamma() for s in S_arr]

        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=S_arr, y=deltas, name="Delta",
                                    line=dict(color="#00D4FF")))
        fig_g.add_trace(go.Scatter(x=S_arr, y=gammas, name="Gamma",
                                    line=dict(color="#FF6B35"), yaxis="y2"))
        fig_g.update_layout(
            template="plotly_dark", height=340,
            title="Delta & Gamma vs Spot",
            yaxis=dict(title="Delta"),
            yaxis2=dict(title="Gamma", overlaying="y", side="right"),
        )
        st.plotly_chart(fig_g, use_container_width=True, key="opt_greeks")


def page_portfolio_analytics(prices: pd.DataFrame, benchmark: pd.Series, rfr: float):
    """Full embedded portfolio analytics dashboard."""
    from src.portfolio_analytics.portfolio_dashboard import run_portfolio_dashboard
    run_portfolio_dashboard(
        preloaded_prices=prices,
        preloaded_benchmark=benchmark,
        preloaded_rfr=rfr,
    )


# ─────────────────────────────────────────────
#  Main app
# ─────────────────────────────────────────────

def main():
    # Header
    st.markdown('<p class="main-header">ASX Quant Terminal</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Institutional-grade quantitative research platform '
        '· ASX Equities · Python</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    config = render_sidebar()

    # Data state
    if "prices" not in st.session_state:
        st.session_state.prices = None
        st.session_state.benchmark = None
        st.session_state.available = []

    if config["load"]:
        with st.spinner("Loading market data…"):
            try:
                prices, bm, avail = load_market_data(
                    config["tickers"], config["start"],
                    config["end"], config["benchmark"],
                )
                if prices.empty:
                    st.error("No data returned. Check tickers and dates.")
                else:
                    st.session_state.prices = prices
                    st.session_state.benchmark = bm
                    st.session_state.available = avail
                    st.success(
                        f"✅ Loaded {len(avail)} tickers, "
                        f"{len(prices)} trading days."
                    )
            except Exception as e:
                st.error(f"Data load failed: {e}")

    if st.session_state.prices is None:
        st.info(
            "👈 Configure your universe in the sidebar and click "
            "**📥 Load Data** to begin."
        )
        st.markdown("""
        ### Platform Capabilities
        | Module | Description |
        |---|---|
        | 📊 Market Overview | Price charts, returns, correlations |
        | 📉 Technical Analysis | 50+ indicators: RSI, MACD, Bollinger, Ichimoku |
        | 🔬 Factor Analysis | CAPM, Fama-French, momentum/value/quality scores |
        | 🌡️ Regime Detection | HMM, Markov switching, volatility and trend regimes |
        | ⚡ Strategy Backtester | 7 strategies with transaction costs and tearsheet |
        | ⚖️ Portfolio Optimisation | Markowitz, Black-Litterman, risk parity |
        | 🎲 Monte Carlo | GBM, jump diffusion, bootstrap simulation |
        | 💹 Options Pricer | Black-Scholes pricing and full Greeks |
        | 📊 Portfolio Analytics | Full portfolio diagnostics and attribution & User Guide |
        """)
        return

    prices = st.session_state.prices
    benchmark = st.session_state.benchmark
    available = st.session_state.available
    rfr = config["rfr"]

    # Navigation tabs
    (
        tab_market, tab_tech, tab_factor, tab_regime,
        tab_backtest, tab_opt, tab_mc, tab_options, tab_portfolio, tab_guide
    ) = st.tabs([
        "📊 Market",
        "📉 Technical",
        "🔬 Factors",
        "🌡️ Regimes",
        "⚡ Backtest",
        "⚖️ Optimisation",
        "🎲 Monte Carlo",
        "💹 Options",
        "📖 Portfolio",
        "Guide"
    ])

    with tab_market:
        page_market_overview(prices, benchmark, rfr)
    with tab_tech:
        page_technical_analysis(prices, available)
    with tab_factor:
        page_factor_analysis(prices, benchmark, rfr)
    with tab_regime:
        page_regime_detection(prices, benchmark)
    with tab_backtest:
        page_strategy_backtest(prices, benchmark, rfr)
    with tab_opt:
        page_portfolio_optimisation(prices, rfr)
    with tab_mc:
        page_monte_carlo(prices)
    with tab_options:
        page_options_pricer(prices, rfr)
    with tab_portfolio:
        page_portfolio_analytics(prices, benchmark, rfr)
  
    st.divider()
    st.caption(
        "ASX Quant Terminal · Built with Python, Streamlit, Plotly · "
        f"Data via Yahoo Finance · {len(prices)} trading days loaded"
    )

if __name__ == "__main__":
    main()
