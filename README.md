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

<img width="1458" height="708" alt="Screenshot 2026-10-02 at 11 18 28 am" src="https://github.com/user-attachments/assets/ed7e8c9b-6b40-432c-9445-922ce37b115a" />
<img width="1454" height="731" alt="Screenshot 2026-10-02 at 11 18 15 am" src="https://github.com/user-attachments/assets/bc515ef9-1a06-4bec-ae6b-f9b7ec190eb8" />
<img width="1459" height="742" alt="Screenshot 2026-10-02 at 11 17 17 am" src="https://github.com/user-attachments/assets/28a82c84-6f27-4f3a-ba11-8d1339283541" />
<img width="1461" height="723" alt="Screenshot 2026-10-02 at 11 16 55 am" src="https://github.com/user-attachments/assets/bca99856-1112-47e2-957f-8791649c49d6" />
<img width="1445" height="721" alt="Screenshot 2026-10-02 at 11 16 35 am" src="https://github.com/user-attachments/assets/de0ada6c-4171-4856-b26c-08d28bf87988" />
<img width="1463" height="750" alt="Screenshot 2026-10-02 at 11 16 07 am" src="https://github.com/user-attachments/assets/22a24584-0cab-4e18-9fd7-a398ea949caf" />
<img width="1450" height="734" alt="Screenshot 2026-10-02 at 11 21 58 am" src="https://github.com/user-attachments/assets/a8151ba3-4af2-4d3f-ac87-36a3a7a7dab3" />
<img width="1452" height="706" alt="Screenshot 2026-10-02 at 11 21 16 am" src="https://github.com/user-attachments/assets/484fd4e3-70cd-4aab-8162-b5fb3b95f86a" />
<img width="1455" height="712" alt="Screenshot 2026-10-02 at 11 21 08 am" src="https://github.com/user-attachments/assets/36aec798-2f4c-418b-92aa-8ec2f15558b6" />
<img width="1446" height="729" alt="Screenshot 2026-10-02 at 11 21 00 am" src="https://github.com/user-attachments/assets/29c79b13-7d5e-431f-96bb-44a913b3ebea" />
<img width="1456" height="726" alt="Screenshot 2026-10-02 at 11 20 23 am" src="https://github.com/user-attachments/assets/4881e79f-10b4-4c8c-b73e-a39eec46e04d" />
<img width="1442" height="703" alt="Screenshot 2026-10-02 at 11 19 24 am" src="https://github.com/user-attachments/assets/2debb06e-d2f4-4b16-8488-d6a5bf676d1a" />
<img width="1451" height="722" alt="Screenshot 2026-10-02 at 11 19 09 am" src="https://github.com/user-attachments/assets/89234f42-8807-4d57-b1f3-3414e8c38531" />
