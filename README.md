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

<img width="1463" height="750" alt="Screenshot 2026-10-02 at 11 16 07 am" src="https://github.com/user-attachments/assets/aa76e5cc-1234-45b4-83e2-134f2ba68d7e" />
<img width="1445" height="721" alt="Screenshot 2026-10-02 at 11 16 35 am" src="https://github.com/user-attachments/assets/380c6077-1f47-47cb-9161-07dc13a17f85" />
<img width="1461" height="723" alt="Screenshot 2026-10-02 at 11 16 55 am" src="https://github.com/user-attachments/assets/bcbce4ca-aeca-40c9-9f6b-81179083f3c2" />
<img width="1459" height="742" alt="Screenshot 2026-10-02 at 11 17 17 am" src="https://github.com/user-attachments/assets/89be202d-10a1-4af9-bdf1-42fa0fd72696" />
<img width="1454" height="731" alt="Screenshot 2026-10-02 at 11 18 15 am" src="https://github.com/user-attachments/assets/c7b4fe60-bf22-422a-a6e0-87c4096ada4b" />
<img width="1458" height="708" alt="Screenshot 2026-10-02 at 11 18 28 am" src="https://github.com/user-attachments/assets/40b8338b-e79f-4918-954c-44e20ad68d5d" />
 <img width="1451" height="722" alt="Screenshot 2026-10-02 at 11 19 09 am" src="https://github.com/user-attachments/assets/efc9ba0f-8425-4d7b-b72c-5175cccc2f2a" />
<img width="1442" height="703" alt="Screenshot 2026-10-02 at 11 19 24 am" src="https://github.com/user-attachments/assets/8e452e1b-4721-4651-812e-ec15026972b0" />
<img width="1456" height="726" alt="Screenshot 2026-10-02 at 11 20 23 am" src="https://github.com/user-attachments/assets/6e9b7454-bffc-4196-a3d9-2e694c4416dd" />
<img width="1446" height="729" alt="Screenshot 2026-10-02 at 11 21 00 am" src="https://github.com/user-attachments/assets/b895569d-265d-4f4e-9092-e8d29b16a08e" />
<img width="1455" height="712" alt="Screenshot 2026-10-02 at 11 21 08 am" src="https://github.com/user-attachments/assets/a7a0e718-2459-47aa-b371-b35518d46d93" />
<img width="1452" height="706" alt="Screenshot 2026-10-02 at 11 21 16 am" src="https://github.com/user-attachments/assets/13543391-a2f2-44a3-82fd-6e54bd68968b" />
<img width="1450" height="734" alt="Screenshot 2026-10-02 at 11 21 58 am" src="https://github.com/user-attachments/assets/cfbc6ca5-9630-40b3-918b-c68ad0f57302" />
