ASX Quantitative Research Terminal (Personal Project, Python)

Designed and built a full-stack quantitative research and trading platform for Australian Securities Exchange equities, modelled on the research infrastructure used by systematic hedge funds.

The platform integrates the following components:

Market data pipeline — automated ingestion, caching, and cleaning of historical OHLCV price data and macroeconomic variables (VIX, interest rates, AUD/USD, commodities) for a 60-stock ASX universe
Technical analysis engine — 50+ indicators including RSI, MACD, Bollinger Bands, ATR, ADX, Ichimoku Cloud, and Parabolic SAR
Statistical testing suite — ADF and KPSS stationarity tests, Ljung-Box autocorrelation, ARCH-LM heteroskedasticity, Jarque-Bera normality, Engle-Granger cointegration, and Hill tail index estimation
Factor models — CAPM with Jensen's Alpha and Security Market Line, Fama-French 3-Factor Model, and cross-sectional momentum, value, and quality factor signal construction
Volatility models — GARCH(1,1), EGARCH, GJR-GARCH, EWMA (RiskMetrics), Parkinson, Garman-Klass, and Yang-Zhang range-based estimators
Regime detection — Gaussian Hidden Markov Model (Baum-Welch EM, Viterbi decoding), Hamilton Markov Regime Switching, and rule-based Bull/Bear/Sideways classifiers with multi-method consensus voting
Portfolio analytics — full institutional tearsheet including Sharpe, Sortino, Calmar, Information Ratio, Jensen's Alpha, VaR, CVaR, marginal and component risk contributions (MCTR/PCTR), PCA risk decomposition, Brinson-Hood-Beebower attribution, and diversification ratio
Trading strategies — cross-sectional momentum, time-series momentum, Bollinger Band mean reversion, RSI mean reversion, Kalman filter pairs trading, multi-factor ranking, and regime-adaptive strategy switching
Portfolio optimisation — Markowitz mean-variance optimisation (max Sharpe, min volatility, risk parity), full efficient frontier with Ledoit-Wolf covariance shrinkage, and Black-Litterman model with absolute and relative investor views
Monte Carlo simulation — Geometric Brownian Motion, Merton jump-diffusion, block bootstrap, correlated multi-asset paths via Cholesky decomposition, and historical/hypothetical stress testing with Kupiec VaR backtesting
Machine learning — Random Forest, Gradient Boosting, and XGBoost return prediction models with walk-forward cross-validation, Information Coefficient evaluation, and a feature engineering pipeline covering technical, factor, and macro inputs
Options pricing — Black-Scholes-Merton analytical pricing with full Greeks suite (Delta, Gamma, Theta, Vega, Rho, Vanna, Volga, Charm), implied volatility via Brent's method, and Monte Carlo pricing for Asian, barrier, lookback, and digital options
Interactive dashboard — built in Streamlit with 9 analytical modules, session-state caching, embedded portfolio analytics, and a plain-English interpretation guide

Tools: Python, NumPy, Pandas, SciPy, statsmodels, scikit-learn, XGBoost, Plotly, Streamlit, yfinance

Note: This was built entirely with Claude AI - This project doesn't display an understanding of coding, but rather, displays an understanding of the models, the development, the testing of the outputs and the integration of the components into the broader research workflow. Overall, this project required me to research, understand and integrate a wide range of financial models and concepts into a complete and functional dashboard and system.
