"""
volatility_models.py
====================
Volatility modelling suite for ASX equity research.

Models Implemented
------------------
GARCH(p,q)       : Generalised AutoRegressive Conditional Heteroskedasticity
EGARCH           : Exponential GARCH (captures leverage effect)
GJR-GARCH        : Asymmetric GARCH (bad news increases vol more than good news)
EWMA             : Exponentially Weighted Moving Average (RiskMetrics)
Realised Vol     : High-frequency realised volatility estimators
Parkinson        : High-low range estimator
Garman-Klass     : OHLC-based estimator
Yang-Zhang       : Overnight gap corrected estimator
HAR-RV           : Heterogeneous Autoregressive model for forecasting

Usage
-----
    from src.analytics.volatility_models import VolatilityModels

    vm = VolatilityModels(returns)
    garch_vol = vm.garch_volatility()
    forecast = vm.forecast_volatility(horizon=21)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# Optional arch library for proper GARCH
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("  [VolModels] Warning: arch library not installed. "
          "Using simplified GARCH. Run: pip install arch")


class VolatilityModels:
    """
    Comprehensive volatility modelling for financial return series.

    Parameters
    ----------
    returns  : pd.Series    Daily return series.
    prices   : pd.DataFrame Optional OHLCV data for range-based estimators.
    name     : str          Ticker label.
    """

    ANNUALISE = np.sqrt(252)

    def __init__(
        self,
        returns: pd.Series,
        prices: Optional[pd.DataFrame] = None,
        name: str = "Asset",
    ) -> None:
        self.returns = returns.dropna().copy()
        self.prices = prices
        self.name = name
        self.n = len(self.returns)

        # Extract OHLC if available
        if prices is not None:
            self.close = prices.get("Close", prices.iloc[:, 0])
            self.high = prices.get("High", self.close)
            self.low = prices.get("Low", self.close)
            self.open = prices.get("Open", self.close)
        else:
            self.close = None
            self.high = None
            self.low = None
            self.open = None

    # ──────────────────────────────────────
    #  Simple Volatility Estimators
    # ──────────────────────────────────────

    def rolling_volatility(
        self,
        window: int = 21,
        annualise: bool = True,
    ) -> pd.Series:
        """
        Standard rolling window volatility (standard deviation of returns).

        Parameters
        ----------
        window    : int    Rolling window in trading days.
        annualise : bool   Multiply by √252 to annualise.

        Returns
        -------
        pd.Series  Annualised volatility.
        """
        vol = self.returns.rolling(window).std()
        if annualise:
            vol = vol * self.ANNUALISE
        return vol.rename(f"RollingVol_{window}d")

    def ewma_volatility(
        self,
        lam: float = 0.94,
        annualise: bool = True,
    ) -> pd.Series:
        """
        Exponentially Weighted Moving Average volatility (RiskMetrics).
        σ²_t = λ × σ²_{t-1} + (1-λ) × r²_{t-1}

        λ = 0.94 is the standard RiskMetrics daily decay factor.
        Higher λ = more weight on historical data (slower response).

        Parameters
        ----------
        lam : float   Decay factor (0 < λ < 1).
        """
        ret_sq = self.returns ** 2
        var = ret_sq.ewm(alpha=1 - lam, adjust=False).mean()
        vol = np.sqrt(var)
        if annualise:
            vol = vol * self.ANNUALISE
        return vol.rename(f"EWMA_Vol_{lam}")

    # ──────────────────────────────────────
    #  Range-Based Estimators (more efficient than close-to-close)
    # ──────────────────────────────────────

    def parkinson_volatility(
        self,
        window: int = 21,
        annualise: bool = True,
    ) -> pd.Series:
        """
        Parkinson (1980) high-low range volatility estimator.
        σ² = 1/(4N×ln2) × Σ [ln(H/L)]²

        5× more efficient than close-to-close estimator.
        Requires High and Low prices.
        """
        if self.high is None or self.low is None:
            return pd.Series(dtype=float, name="Parkinson_Vol")

        log_hl = np.log(self.high / self.low)
        factor = 1 / (4 * np.log(2))
        var = factor * (log_hl ** 2).rolling(window).mean()
        vol = np.sqrt(var)
        if annualise:
            vol = vol * self.ANNUALISE
        return vol.rename(f"Parkinson_Vol_{window}d")

    def garman_klass_volatility(
        self,
        window: int = 21,
        annualise: bool = True,
    ) -> pd.Series:
        """
        Garman-Klass (1980) OHLC volatility estimator.
        Uses Open, High, Low, Close for higher efficiency.
        σ² = 0.5[ln(H/L)]² - (2ln2-1)[ln(C/O)]²

        7× more efficient than close-to-close estimator.
        """
        if self.high is None or self.open is None:
            return pd.Series(dtype=float, name="GarmanKlass_Vol")

        log_hl = np.log(self.high / self.low)
        log_co = np.log(self.close / self.open)

        var = (
            0.5 * log_hl ** 2
            - (2 * np.log(2) - 1) * log_co ** 2
        ).rolling(window).mean()

        vol = np.sqrt(var.clip(lower=0))
        if annualise:
            vol = vol * self.ANNUALISE
        return vol.rename(f"GarmanKlass_Vol_{window}d")

    def yang_zhang_volatility(
        self,
        window: int = 21,
        annualise: bool = True,
    ) -> pd.Series:
        """
        Yang-Zhang (2000) volatility estimator.
        Corrects for overnight gaps (opening jumps).
        The most efficient close-to-close OHLC estimator.
        """
        if self.high is None or self.open is None:
            return pd.Series(dtype=float, name="YangZhang_Vol")

        log_oc = np.log(self.open / self.close.shift(1))   # overnight return
        log_co = np.log(self.close / self.open)             # intraday return
        log_ho = np.log(self.high / self.open)
        log_lo = np.log(self.low / self.open)

        # Overnight variance
        var_oc = log_oc.rolling(window).var()
        # Open-to-close variance
        var_co = log_co.rolling(window).var()
        # Rogers-Satchell variance (drift-independent)
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        var_rs = rs.rolling(window).mean()

        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        var = var_oc + k * var_co + (1 - k) * var_rs

        vol = np.sqrt(var.clip(lower=0))
        if annualise:
            vol = vol * self.ANNUALISE
        return vol.rename(f"YangZhang_Vol_{window}d")

    # ──────────────────────────────────────
    #  GARCH Models
    # ──────────────────────────────────────

    def garch_volatility(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        forecast_horizon: int = 0,
    ) -> dict:
        """
        Fit GARCH(p,q) model and return conditional volatility.

        GARCH(1,1):  σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}

        Parameters
        ----------
        p    : int   ARCH order (lagged squared residuals).
        q    : int   GARCH order (lagged variance).
        dist : str   Error distribution: 'normal' | 't' | 'skewt'.
        forecast_horizon : int   Steps ahead to forecast.

        Returns
        -------
        dict with:
            conditional_vol  : pd.Series  Fitted conditional volatility.
            params           : dict       Model parameters.
            aic / bic        : float      Information criteria.
            forecast         : pd.Series  Volatility forecast (if horizon > 0).
        """
        if HAS_ARCH:
            return self._garch_arch_library(p, q, dist, forecast_horizon)
        else:
            return self._garch_manual(p, q)

    def _garch_arch_library(
        self,
        p: int,
        q: int,
        dist: str,
        forecast_horizon: int,
    ) -> dict:
        """GARCH via the arch library — most accurate implementation."""
        # Scale returns to percentage for numerical stability
        ret_pct = self.returns * 100

        try:
            model = arch_model(
                ret_pct,
                vol="GARCH",
                p=p,
                q=q,
                dist=dist,
                rescale=False,
            )
            fit = model.fit(disp="off", show_warning=False)

            cond_vol = fit.conditional_volatility / 100 * self.ANNUALISE
            cond_vol.name = f"GARCH({p},{q})_Vol"

            params = {
                k: round(float(v), 6)
                for k, v in fit.params.items()
            }

            result = {
                "conditional_vol": cond_vol,
                "params": params,
                "aic": round(fit.aic, 2),
                "bic": round(fit.bic, 2),
                "log_likelihood": round(fit.loglikelihood, 2),
                "model": f"GARCH({p},{q})",
                "distribution": dist,
            }

            if forecast_horizon > 0:
                fc = fit.forecast(horizon=forecast_horizon)
                fc_var = fc.variance.iloc[-1].values
                fc_vol = np.sqrt(fc_var) / 100 * self.ANNUALISE
                result["forecast"] = pd.Series(
                    fc_vol,
                    index=range(1, forecast_horizon + 1),
                    name="GARCH_Forecast",
                )

            return result

        except Exception as e:
            print(f"  [GARCH] arch library failed: {e}. Falling back to manual.")
            return self._garch_manual(p, q)

    def _garch_manual(self, p: int = 1, q: int = 1) -> dict:
        """
        Manual GARCH(1,1) implementation via maximum likelihood.
        Used as fallback when arch library is unavailable.
        """
        ret = self.returns.values
        n = len(ret)

        def garch_loglik(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10

            var = np.full(n, omega / (1 - alpha - beta))
            for t in range(1, n):
                var[t] = omega + alpha * ret[t - 1] ** 2 + beta * var[t - 1]

            var = np.maximum(var, 1e-10)
            ll = -0.5 * np.sum(np.log(2 * np.pi * var) + ret ** 2 / var)
            return -ll

        # Initial parameter guesses
        long_var = np.var(ret)
        x0 = [long_var * 0.1, 0.1, 0.8]
        bounds = [(1e-8, None), (0, 0.999), (0, 0.999)]

        result = minimize(garch_loglik, x0, method="L-BFGS-B", bounds=bounds)
        omega, alpha, beta = result.x

        # Compute conditional variance
        var_series = np.full(n, omega / (1 - alpha - beta))
        for t in range(1, n):
            var_series[t] = (
                omega + alpha * ret[t - 1] ** 2 + beta * var_series[t - 1]
            )

        cond_vol = pd.Series(
            np.sqrt(np.maximum(var_series, 0)) * self.ANNUALISE,
            index=self.returns.index,
            name="GARCH(1,1)_Vol",
        )

        return {
            "conditional_vol": cond_vol,
            "params": {
                "omega": round(omega, 8),
                "alpha": round(alpha, 6),
                "beta": round(beta, 6),
                "persistence": round(alpha + beta, 6),
            },
            "model": "GARCH(1,1) [manual]",
        }

    def egarch_volatility(self) -> dict:
        """
        EGARCH model — captures the leverage effect:
        bad news increases volatility more than good news of the same size.

        Requires the arch library.
        """
        if not HAS_ARCH:
            print("  [EGARCH] Requires arch library: pip install arch")
            return {}

        ret_pct = self.returns * 100
        try:
            model = arch_model(ret_pct, vol="EGARCH", p=1, o=1, q=1, dist="t")
            fit = model.fit(disp="off", show_warning=False)
            cond_vol = fit.conditional_volatility / 100 * self.ANNUALISE
            cond_vol.name = "EGARCH_Vol"

            return {
                "conditional_vol": cond_vol,
                "params": {k: round(float(v), 6) for k, v in fit.params.items()},
                "aic": round(fit.aic, 2),
                "bic": round(fit.bic, 2),
                "model": "EGARCH(1,1,1)",
            }
        except Exception as e:
            print(f"  [EGARCH] Failed: {e}")
            return {}

    def gjr_garch_volatility(self) -> dict:
        """
        GJR-GARCH (Glosten-Jagannathan-Runkle) model.
        Asymmetric GARCH: negative shocks have greater impact on future volatility.
        σ²_t = ω + (α + γ·I{ε<0}) × ε²_{t-1} + β × σ²_{t-1}

        Requires the arch library.
        """
        if not HAS_ARCH:
            print("  [GJR-GARCH] Requires arch library: pip install arch")
            return {}

        ret_pct = self.returns * 100
        try:
            model = arch_model(ret_pct, vol="GARCH", p=1, o=1, q=1, dist="t")
            fit = model.fit(disp="off", show_warning=False)
            cond_vol = fit.conditional_volatility / 100 * self.ANNUALISE
            cond_vol.name = "GJR_GARCH_Vol"

            return {
                "conditional_vol": cond_vol,
                "params": {k: round(float(v), 6) for k, v in fit.params.items()},
                "aic": round(fit.aic, 2),
                "bic": round(fit.bic, 2),
                "model": "GJR-GARCH(1,1)",
            }
        except Exception as e:
            print(f"  [GJR-GARCH] Failed: {e}")
            return {}

    # ──────────────────────────────────────
    #  HAR-RV Model
    # ──────────────────────────────────────

    def har_rv_model(
        self,
        forecast_horizon: int = 21,
    ) -> dict:
        """
        Heterogeneous Autoregressive Realised Volatility (HAR-RV) model.

        Models daily, weekly, and monthly volatility components:
        RV_t = c + β_d × RV_{t-1} + β_w × RV_{t-5:t-1} + β_m × RV_{t-22:t-1}

        Captures the multi-scale nature of financial market participants
        (daily traders, weekly funds, monthly pension funds).

        Parameters
        ----------
        forecast_horizon : int   Days ahead to forecast.

        Returns
        -------
        dict with fitted values, parameters, R², and forecast.
        """
        # Use squared returns as RV proxy
        rv = self.returns ** 2

        rv_d = rv                            # daily RV
        rv_w = rv.rolling(5).mean()          # weekly RV
        rv_m = rv.rolling(22).mean()         # monthly RV

        # Align
        df = pd.DataFrame({
            "RV": rv,
            "RV_d": rv_d.shift(1),
            "RV_w": rv_w.shift(1),
            "RV_m": rv_m.shift(1),
        }).dropna()

        y = df["RV"].values
        X = np.column_stack([
            np.ones(len(df)),
            df["RV_d"].values,
            df["RV_w"].values,
            df["RV_m"].values,
        ])

        # OLS
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except Exception:
            return {}

        y_hat = X @ beta
        residuals = y - y_hat
        ss_res = np.dot(residuals, residuals)
        ss_tot = np.dot(y - y.mean(), y - y.mean())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        fitted = pd.Series(
            np.sqrt(np.maximum(y_hat, 0)) * self.ANNUALISE,
            index=df.index,
            name="HAR_RV_Fitted",
        )

        # Multi-step forecast
        last_rv_d = float(rv_d.iloc[-1])
        last_rv_w = float(rv_w.iloc[-1])
        last_rv_m = float(rv_m.iloc[-1])
        c, b_d, b_w, b_m = beta

        forecasts = []
        for h in range(1, forecast_horizon + 1):
            rv_pred = c + b_d * last_rv_d + b_w * last_rv_w + b_m * last_rv_m
            rv_pred = max(rv_pred, 0)
            forecasts.append(np.sqrt(rv_pred) * self.ANNUALISE)
            last_rv_d = rv_pred

        return {
            "fitted": fitted,
            "params": {
                "intercept": round(c, 8),
                "beta_daily": round(b_d, 6),
                "beta_weekly": round(b_w, 6),
                "beta_monthly": round(b_m, 6),
            },
            "r_squared": round(r2, 4),
            "forecast": pd.Series(
                forecasts,
                index=range(1, forecast_horizon + 1),
                name="HAR_RV_Forecast",
            ),
            "model": "HAR-RV",
        }

    # ──────────────────────────────────────
    #  Volatility Regime Detection
    # ──────────────────────────────────────

    def volatility_regimes(
        self,
        window: int = 21,
        n_regimes: int = 3,
    ) -> pd.DataFrame:
        """
        Classify volatility into Low, Medium, and High regimes using
        rolling volatility percentile thresholds.

        Parameters
        ----------
        window    : int   Rolling window for volatility calculation.
        n_regimes : int   Number of regimes (2 or 3).

        Returns
        -------
        pd.DataFrame  columns: vol, regime_label, regime_code
        """
        vol = self.rolling_volatility(window)

        if n_regimes == 2:
            median = vol.quantile(0.50)
            regime = pd.cut(
                vol,
                bins=[-np.inf, median, np.inf],
                labels=["Low Vol", "High Vol"],
            )
        else:
            p33 = vol.quantile(0.33)
            p67 = vol.quantile(0.67)
            regime = pd.cut(
                vol,
                bins=[-np.inf, p33, p67, np.inf],
                labels=["Low Vol", "Medium Vol", "High Vol"],
            )

        regime_code = regime.cat.codes
        return pd.DataFrame({
            "Volatility": vol,
            "Regime": regime.astype(str),
            "Regime_Code": regime_code,
        })

    # ──────────────────────────────────────
    #  Volatility Forecast Comparison
    # ──────────────────────────────────────

    def forecast_volatility(
        self,
        horizon: int = 21,
        methods: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Forecast volatility using multiple models and return a comparison table.

        Parameters
        ----------
        horizon : int    Forecast horizon in trading days.
        methods : list   Models to use. Defaults to all available.

        Returns
        -------
        pd.DataFrame  One column per model, rows = forecast days.
        """
        methods = methods or ["EWMA", "GARCH", "HAR-RV", "Rolling"]
        forecasts = {}

        if "Rolling" in methods:
            last_vol = float(self.rolling_volatility(21).iloc[-1])
            forecasts["Rolling"] = pd.Series(
                [last_vol] * horizon,
                index=range(1, horizon + 1),
            )

        if "EWMA" in methods:
            last_ewma = float(self.ewma_volatility().iloc[-1])
            forecasts["EWMA"] = pd.Series(
                [last_ewma] * horizon,
                index=range(1, horizon + 1),
            )

        if "GARCH" in methods:
            try:
                g = self.garch_volatility(forecast_horizon=horizon)
                if "forecast" in g:
                    forecasts["GARCH"] = g["forecast"]
            except Exception:
                pass

        if "HAR-RV" in methods:
            try:
                har = self.har_rv_model(forecast_horizon=horizon)
                if "forecast" in har:
                    forecasts["HAR-RV"] = har["forecast"]
            except Exception:
                pass

        if not forecasts:
            return pd.DataFrame()

        df = pd.DataFrame(forecasts)
        df.index.name = "Days Ahead"
        return df

    # ──────────────────────────────────────
    #  Volatility Metrics Summary
    # ──────────────────────────────────────

    def volatility_summary(self) -> pd.DataFrame:
        """
        Return a summary table comparing all volatility estimators.
        """
        rows = []

        def _latest(series):
            try:
                v = series.dropna()
                return round(float(v.iloc[-1]), 4) if len(v) > 0 else np.nan
            except Exception:
                return np.nan

        rows.append(("Rolling 21D", _latest(self.rolling_volatility(21))))
        rows.append(("Rolling 63D", _latest(self.rolling_volatility(63))))
        rows.append(("EWMA (λ=0.94)", _latest(self.ewma_volatility(0.94))))
        rows.append(("EWMA (λ=0.97)", _latest(self.ewma_volatility(0.97))))

        if self.high is not None:
            rows.append(("Parkinson 21D", _latest(self.parkinson_volatility(21))))
            rows.append(("Garman-Klass 21D", _latest(self.garman_klass_volatility(21))))
            rows.append(("Yang-Zhang 21D", _latest(self.yang_zhang_volatility(21))))

        try:
            g = self.garch_volatility()
            rows.append(("GARCH(1,1) Latest", _latest(g["conditional_vol"])))
        except Exception:
            pass

        return pd.DataFrame(rows, columns=["Model", "Latest Ann. Volatility"])
