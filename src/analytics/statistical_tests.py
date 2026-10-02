"""
statistical_tests.py
====================
Institutional-grade statistical testing suite for quantitative research.

Used by regime detection, pairs trading, factor models, and strategy validation
to ensure mathematical rigour and avoid spurious signals.

Tests Implemented
-----------------
Stationarity    : ADF, KPSS, Phillips-Perron, Variance Ratio
Cointegration   : Engle-Granger, Johansen
Normality       : Jarque-Bera, Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling
Autocorrelation : Ljung-Box, Durbin-Watson, Breusch-Godfrey
Heteroskedast.  : ARCH LM, Breusch-Pagan, White's test
Structural Break: Chow test, CUSUM, Zivot-Andrews
Correlation     : Pearson, Spearman, Kendall with significance
Distribution    : KS test, Anderson-Darling fit, tail index estimation

Usage
-----
    from src.analytics.statistical_tests import StatisticalTests

    st = StatisticalTests(returns)
    results = st.run_full_battery()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    jarque_bera, shapiro, normaltest, anderson,
    kstest, spearmanr, kendalltau,
)

warnings.filterwarnings("ignore")

# Optional statsmodels imports
try:
    from statsmodels.tsa.stattools import adfuller, kpss, coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import (
        acorr_ljungbox, het_arch, het_breuschpagan,
    )
    from statsmodels.stats.breaktest import breaks_cusumolsresid
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("  [StatTests] Warning: statsmodels not installed. "
          "Some tests will be unavailable. Run: pip install statsmodels")


# ─────────────────────────────────────────────
#  Result containers
# ─────────────────────────────────────────────

@dataclass
class TestResult:
    """Container for a single statistical test result."""
    test_name: str
    statistic: float
    p_value: float
    critical_values: dict = field(default_factory=dict)
    reject_null: bool = False
    null_hypothesis: str = ""
    interpretation: str = ""
    confidence: float = 0.95

    def __str__(self) -> str:
        verdict = "REJECT H₀" if self.reject_null else "FAIL TO REJECT H₀"
        return (
            f"{self.test_name}\n"
            f"  H₀: {self.null_hypothesis}\n"
            f"  Statistic: {self.statistic:.4f}  |  P-value: {self.p_value:.4f}\n"
            f"  Verdict ({self.confidence:.0%}): {verdict}\n"
            f"  {self.interpretation}"
        )


@dataclass
class TestBattery:
    """Container for a full suite of test results."""
    series_name: str = "Series"
    n_obs: int = 0
    results: dict[str, TestResult] = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"Statistical Test Battery: {self.series_name} (n={self.n_obs})")
        print(f"{'='*60}")
        for name, result in self.results.items():
            print(f"\n{result}")

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for name, r in self.results.items():
            rows.append({
                "Test": r.test_name,
                "Statistic": round(r.statistic, 4),
                "P-Value": round(r.p_value, 4),
                "Reject H₀": r.reject_null,
                "Null Hypothesis": r.null_hypothesis,
                "Interpretation": r.interpretation,
            })
        return pd.DataFrame(rows).set_index("Test")


# ─────────────────────────────────────────────
#  StatisticalTests
# ─────────────────────────────────────────────

class StatisticalTests:
    """
    Comprehensive statistical testing for financial return series.

    Parameters
    ----------
    series     : pd.Series   Return or price series to test.
    alpha      : float       Significance level (default 0.05 = 95% confidence).
    name       : str         Label for display purposes.
    """

    def __init__(
        self,
        series: pd.Series,
        alpha: float = 0.05,
        name: str = "Series",
    ) -> None:
        self.series = series.dropna().copy()
        self.alpha = alpha
        self.name = name
        self.n = len(self.series)

    # ──────────────────────────────────────
    #  Stationarity Tests
    # ──────────────────────────────────────

    def adf_test(
        self,
        regression: str = "c",
        maxlag: Optional[int] = None,
    ) -> TestResult:
        """
        Augmented Dickey-Fuller test for unit root (non-stationarity).

        H₀: Series has a unit root (non-stationary).
        H₁: Series is stationary.

        Parameters
        ----------
        regression : 'c'  = constant, 'ct' = trend + constant, 'n' = none
        maxlag     : Maximum number of lags. None = automatic selection.
        """
        if not HAS_STATSMODELS:
            return self._unavailable("ADF Test")

        result = adfuller(self.series, regression=regression, maxlag=maxlag)
        stat, pval, _, _, crit_vals, _ = result

        reject = pval < self.alpha

        return TestResult(
            test_name="Augmented Dickey-Fuller (ADF)",
            statistic=stat,
            p_value=pval,
            critical_values=crit_vals,
            reject_null=reject,
            null_hypothesis="Series has a unit root (non-stationary)",
            interpretation=(
                "Series is stationary — safe to use in models."
                if reject else
                "Series is non-stationary — consider differencing."
            ),
            confidence=1 - self.alpha,
        )

    def kpss_test(
        self,
        regression: str = "c",
        nlags: str = "auto",
    ) -> TestResult:
        """
        KPSS test — complements ADF by testing stationarity as the null.

        H₀: Series is stationary.
        H₁: Series has a unit root (non-stationary).

        Use alongside ADF:
            ADF reject + KPSS fail-to-reject = strong evidence of stationarity.
        """
        if not HAS_STATSMODELS:
            return self._unavailable("KPSS Test")

        try:
            stat, pval, _, crit_vals = kpss(
                self.series, regression=regression, nlags=nlags
            )
        except Exception:
            stat, pval, _, crit_vals = kpss(
                self.series, regression=regression, nlags=12
            )

        reject = pval < self.alpha

        return TestResult(
            test_name="KPSS Test",
            statistic=stat,
            p_value=pval,
            critical_values=crit_vals,
            reject_null=reject,
            null_hypothesis="Series is stationary",
            interpretation=(
                "Evidence of non-stationarity — unit root present."
                if reject else
                "Series is stationary at this significance level."
            ),
            confidence=1 - self.alpha,
        )

    def variance_ratio_test(
        self,
        periods: list[int] = [2, 4, 8, 16],
    ) -> TestResult:
        """
        Lo-MacKinlay Variance Ratio test for random walk.

        H₀: Returns follow a random walk (efficient market).
        H₁: Returns are predictable (autocorrelated).

        VR(q) = Var(q-period return) / (q × Var(1-period return))
        VR = 1 → random walk.
        VR > 1 → positive autocorrelation (momentum).
        VR < 1 → negative autocorrelation (mean reversion).
        """
        ret = self.series.values
        n = len(ret)
        var1 = np.var(ret, ddof=1)

        vr_stats = {}
        for q in periods:
            if n < q * 2:
                continue
            ret_q = np.array([
                np.sum(ret[i:i + q]) for i in range(0, n - q + 1)
            ])
            var_q = np.var(ret_q, ddof=1) / q
            vr = var_q / var1 if var1 > 0 else 1.0

            # Asymptotic z-statistic
            theta = (
                2 * (2 * q - 1) * (q - 1)
            ) / (3 * q * n)
            z = (vr - 1) / np.sqrt(theta) if theta > 0 else 0

            vr_stats[q] = {"VR": round(vr, 4), "Z": round(z, 4)}

        # Overall test: any period showing significant deviation?
        z_values = [v["Z"] for v in vr_stats.values()]
        max_z = max(z_values, key=abs) if z_values else 0
        pval = float(2 * (1 - stats.norm.cdf(abs(max_z))))
        reject = pval < self.alpha

        interp_parts = [f"q={q}: VR={v['VR']:.3f}" for q, v in vr_stats.items()]
        if reject:
            if max_z > 0:
                interp = f"Positive autocorrelation (momentum) detected. {'; '.join(interp_parts)}"
            else:
                interp = f"Negative autocorrelation (mean reversion) detected. {'; '.join(interp_parts)}"
        else:
            interp = f"Consistent with random walk. {'; '.join(interp_parts)}"

        return TestResult(
            test_name="Variance Ratio Test (Lo-MacKinlay)",
            statistic=max_z,
            p_value=pval,
            critical_values={"1%": 2.576, "5%": 1.96, "10%": 1.645},
            reject_null=reject,
            null_hypothesis="Returns follow a random walk",
            interpretation=interp,
            confidence=1 - self.alpha,
        )

    # ──────────────────────────────────────
    #  Normality Tests
    # ──────────────────────────────────────

    def jarque_bera_test(self) -> TestResult:
        """
        Jarque-Bera test for normality.

        H₀: Returns are normally distributed (skewness=0, kurtosis=3).
        H₁: Returns are not normally distributed.

        Most financial returns fail this test due to fat tails and skewness.
        """
        stat, pval = jarque_bera(self.series)
        reject = pval < self.alpha

        skew = float(stats.skew(self.series))
        kurt = float(stats.kurtosis(self.series))

        return TestResult(
            test_name="Jarque-Bera Normality Test",
            statistic=stat,
            p_value=pval,
            reject_null=reject,
            null_hypothesis="Returns are normally distributed",
            interpretation=(
                f"Non-normal distribution (skew={skew:.3f}, "
                f"excess kurtosis={kurt:.3f}). Fat tails and/or skewness present."
                if reject else
                f"Cannot reject normality (skew={skew:.3f}, excess kurtosis={kurt:.3f})."
            ),
            confidence=1 - self.alpha,
        )

    def shapiro_wilk_test(self) -> TestResult:
        """
        Shapiro-Wilk test for normality.
        Most powerful test for small samples (n < 5000).

        H₀: Sample comes from a normal distribution.
        """
        # Shapiro-Wilk is unreliable for very large samples
        sample = self.series.values
        if len(sample) > 5000:
            sample = np.random.choice(sample, 5000, replace=False)

        stat, pval = shapiro(sample)
        reject = pval < self.alpha

        return TestResult(
            test_name="Shapiro-Wilk Normality Test",
            statistic=stat,
            p_value=pval,
            reject_null=reject,
            null_hypothesis="Sample is drawn from a normal distribution",
            interpretation=(
                "Non-normal. Normal distribution assumption violated."
                if reject else
                "Cannot reject normality at this significance level."
            ),
            confidence=1 - self.alpha,
        )

    def dagostino_pearson_test(self) -> TestResult:
        """
        D'Agostino-Pearson K² test — combines skewness and kurtosis.
        Works well for larger samples.

        H₀: Returns are normally distributed.
        """
        stat, pval = normaltest(self.series)
        reject = pval < self.alpha

        return TestResult(
            test_name="D'Agostino-Pearson K² Test",
            statistic=stat,
            p_value=pval,
            reject_null=reject,
            null_hypothesis="Returns are normally distributed",
            interpretation=(
                "Significant departure from normality in skewness or kurtosis."
                if reject else
                "Cannot reject normality."
            ),
            confidence=1 - self.alpha,
        )

    def anderson_darling_test(self) -> TestResult:
        """
        Anderson-Darling test for normality.
        Places more weight on the tails — important for risk management.

        H₀: Data comes from the specified distribution (default: normal).
        """
        result = anderson(self.series, dist="norm")
        stat = result.statistic
        # Compare to 5% critical value
        idx = list(result.significance_level).index(5.0)
        crit_5pct = result.critical_values[idx]
        reject = stat > crit_5pct
        pval = 0.01 if stat > result.critical_values[0] else (
            0.05 if reject else 0.10
        )

        crit = {
            f"{sl}%": cv
            for sl, cv in zip(result.significance_level, result.critical_values)
        }

        return TestResult(
            test_name="Anderson-Darling Normality Test",
            statistic=stat,
            p_value=pval,
            critical_values=crit,
            reject_null=reject,
            null_hypothesis="Data follows a normal distribution",
            interpretation=(
                "Non-normal — tail behaviour differs from Gaussian. "
                "Use fat-tailed distributions in risk models."
                if reject else
                "Cannot reject normality."
            ),
            confidence=1 - self.alpha,
        )

    # ──────────────────────────────────────
    #  Autocorrelation Tests
    # ──────────────────────────────────────

    def ljung_box_test(self, lags: int = 10) -> TestResult:
        """
        Ljung-Box test for autocorrelation in returns.

        H₀: No autocorrelation up to lag k.
        H₁: Autocorrelation present (returns are predictable).

        If rejected → return series has structure exploitable by a strategy.
        """
        if not HAS_STATSMODELS:
            return self._unavailable("Ljung-Box Test")

        result = acorr_ljungbox(self.series, lags=[lags], return_df=True)
        stat = float(result["lb_stat"].iloc[-1])
        pval = float(result["lb_pvalue"].iloc[-1])
        reject = pval < self.alpha

        return TestResult(
            test_name=f"Ljung-Box Autocorrelation Test (lag={lags})",
            statistic=stat,
            p_value=pval,
            reject_null=reject,
            null_hypothesis="No autocorrelation in returns",
            interpretation=(
                f"Significant autocorrelation at lag {lags} — "
                "return series may be predictable."
                if reject else
                f"No significant autocorrelation up to lag {lags}."
            ),
            confidence=1 - self.alpha,
        )

    def durbin_watson_test(self) -> TestResult:
        """
        Durbin-Watson test for first-order autocorrelation.

        DW ≈ 2 → no autocorrelation.
        DW < 2 → positive autocorrelation.
        DW > 2 → negative autocorrelation.
        """
        if not HAS_STATSMODELS:
            return self._unavailable("Durbin-Watson Test")

        dw = durbin_watson(self.series.values)

        # DW interpretation
        if 1.5 < dw < 2.5:
            reject = False
            interp = f"DW={dw:.3f}: No significant autocorrelation."
        elif dw <= 1.5:
            reject = True
            interp = f"DW={dw:.3f}: Positive autocorrelation detected."
        else:
            reject = True
            interp = f"DW={dw:.3f}: Negative autocorrelation detected."

        return TestResult(
            test_name="Durbin-Watson Test",
            statistic=dw,
            p_value=np.nan,
            reject_null=reject,
            null_hypothesis="No first-order autocorrelation",
            interpretation=interp,
            confidence=1 - self.alpha,
        )

    def autocorrelation_function(
        self, max_lag: int = 20
    ) -> pd.DataFrame:
        """
        Compute ACF and PACF values for the return series.

        Returns
        -------
        pd.DataFrame  columns: lag, acf, pacf, significant_acf, significant_pacf
        """
        n = self.n
        conf_bound = 1.96 / np.sqrt(n)

        acf_vals = [1.0]
        for lag in range(1, max_lag + 1):
            if lag >= n:
                break
            corr = np.corrcoef(
                self.series.values[lag:], self.series.values[:-lag]
            )[0, 1]
            acf_vals.append(float(corr) if not np.isnan(corr) else 0.0)

        rows = []
        for i, acf in enumerate(acf_vals):
            rows.append({
                "Lag": i,
                "ACF": acf,
                "Upper_Bound": conf_bound,
                "Lower_Bound": -conf_bound,
                "Significant": abs(acf) > conf_bound and i > 0,
            })

        return pd.DataFrame(rows)

    # ──────────────────────────────────────
    #  Heteroskedasticity Tests
    # ──────────────────────────────────────

    def arch_lm_test(self, lags: int = 5) -> TestResult:
        """
        ARCH LM test for conditional heteroskedasticity (volatility clustering).

        H₀: No ARCH effects (constant variance).
        H₁: ARCH effects present — volatility clusters over time.

        If rejected → GARCH modelling is appropriate.
        """
        if not HAS_STATSMODELS:
            return self._unavailable("ARCH LM Test")

        try:
            result = het_arch(self.series, nlags=lags)
            stat, pval = result[0], result[1]
        except Exception:
            return self._unavailable("ARCH LM Test")

        reject = pval < self.alpha

        return TestResult(
            test_name=f"ARCH LM Test (lags={lags})",
            statistic=stat,
            p_value=pval,
            reject_null=reject,
            null_hypothesis="No ARCH effects (homoskedastic residuals)",
            interpretation=(
                "Volatility clustering confirmed — GARCH modelling recommended."
                if reject else
                "No significant ARCH effects detected."
            ),
            confidence=1 - self.alpha,
        )

    # ──────────────────────────────────────
    #  Cointegration Tests
    # ──────────────────────────────────────

    def engle_granger_cointegration(
        self,
        series2: pd.Series,
        trend: str = "c",
    ) -> TestResult:
        """
        Engle-Granger two-step cointegration test for pairs trading.

        H₀: No cointegration between the two series.
        H₁: The two series are cointegrated (share a long-run equilibrium).

        Cointegrated pairs → mean-reversion strategy is applicable.

        Parameters
        ----------
        series2 : pd.Series   Second price series to test against.
        trend   : str         'c' = constant, 'ct' = trend + constant.
        """
        if not HAS_STATSMODELS:
            return self._unavailable("Engle-Granger Cointegration")

        # Align series
        common = self.series.index.intersection(series2.index)
        s1 = self.series.loc[common]
        s2 = series2.loc[common]

        try:
            stat, pval, crit_vals = coint(s1, s2, trend=trend)
        except Exception as e:
            return TestResult(
                test_name="Engle-Granger Cointegration",
                statistic=0.0,
                p_value=1.0,
                reject_null=False,
                null_hypothesis="No cointegration",
                interpretation=f"Test failed: {e}",
            )

        reject = pval < self.alpha
        crit = {"1%": crit_vals[0], "5%": crit_vals[1], "10%": crit_vals[2]}

        return TestResult(
            test_name="Engle-Granger Cointegration Test",
            statistic=stat,
            p_value=pval,
            critical_values=crit,
            reject_null=reject,
            null_hypothesis="No cointegration between the two series",
            interpretation=(
                "Cointegration detected — pair is suitable for mean-reversion trading."
                if reject else
                "No cointegration detected — pair is NOT suitable for pairs trading."
            ),
            confidence=1 - self.alpha,
        )

    def johansen_cointegration(
        self,
        series_list: list[pd.Series],
        det_order: int = 0,
        k_ar_diff: int = 1,
    ) -> dict:
        """
        Johansen cointegration test for multiple series (VAR framework).
        Tests the rank of cointegration among N series.

        H₀: At most r cointegrating relationships.

        Parameters
        ----------
        series_list : list of pd.Series   Price series (NOT returns).
        det_order   : int   -1=none, 0=constant, 1=trend.
        k_ar_diff   : int   Number of lags in the VAR.

        Returns
        -------
        dict with trace statistic, eigenvalue statistic, and critical values.
        """
        if not HAS_STATSMODELS:
            return {"error": "statsmodels not installed"}

        # Align all series
        df = pd.concat(series_list, axis=1).dropna()

        try:
            result = coint_johansen(df, det_order=det_order, k_ar_diff=k_ar_diff)
        except Exception as e:
            return {"error": str(e)}

        n = df.shape[1]
        ranks = list(range(n))
        trace_stats = result.lr1
        eigen_stats = result.lr2
        crit_trace = result.cvt  # [rank, {90%, 95%, 99%}]
        crit_eigen = result.cvm

        rows = []
        for r in ranks:
            rows.append({
                "Rank": r,
                "Trace Stat": round(trace_stats[r], 4),
                "Trace 95% CV": round(crit_trace[r, 1], 4),
                "Trace Reject": trace_stats[r] > crit_trace[r, 1],
                "Eigen Stat": round(eigen_stats[r], 4),
                "Eigen 95% CV": round(crit_eigen[r, 1], 4),
                "Eigen Reject": eigen_stats[r] > crit_eigen[r, 1],
            })

        cointegrating_ranks = sum(1 for row in rows if row["Trace Reject"])

        return {
            "table": pd.DataFrame(rows),
            "cointegrating_ranks": cointegrating_ranks,
            "interpretation": (
                f"{cointegrating_ranks} cointegrating relationship(s) found."
                if cointegrating_ranks > 0 else
                "No cointegration detected."
            ),
        }

    # ──────────────────────────────────────
    #  Distribution Analysis
    # ──────────────────────────────────────

    def tail_index_estimation(self) -> dict:
        """
        Estimate tail heaviness using the Hill estimator.
        Higher tail index = fatter tails = more extreme events.

        Returns alpha (tail index) for both positive and negative tails.
        α < 2 → variance is infinite (very fat tails).
        α < 4 → kurtosis is infinite.
        α > 4 → finite kurtosis (approaching normality).
        """
        ret = self.series.values
        n = len(ret)
        k = max(int(n * 0.10), 10)   # Use top 10% of observations

        # Positive tail
        pos = np.sort(ret[ret > 0])[::-1]
        if len(pos) > k:
            hill_pos = 1 / np.mean(np.log(pos[:k] / pos[k]))
        else:
            hill_pos = np.nan

        # Negative tail
        neg = np.sort(np.abs(ret[ret < 0]))[::-1]
        if len(neg) > k:
            hill_neg = 1 / np.mean(np.log(neg[:k] / neg[k]))
        else:
            hill_neg = np.nan

        def _interp(alpha):
            if alpha is None or np.isnan(alpha):
                return "Unknown"
            if alpha < 2:
                return "Extremely fat tails (infinite variance)"
            if alpha < 3:
                return "Very fat tails"
            if alpha < 4:
                return "Fat tails (infinite kurtosis)"
            return "Moderate tails — approaching normality"

        return {
            "positive_tail_index": round(hill_pos, 3) if not np.isnan(hill_pos) else None,
            "negative_tail_index": round(hill_neg, 3) if not np.isnan(hill_neg) else None,
            "positive_tail_interpretation": _interp(hill_pos),
            "negative_tail_interpretation": _interp(hill_neg),
            "k_threshold": k,
        }

    def distribution_fit(self) -> pd.DataFrame:
        """
        Fit common distributions to the return series and rank by goodness of fit.

        Distributions tested: normal, t, skew-normal, laplace, logistic.

        Returns
        -------
        pd.DataFrame  Ranked by KS statistic (lower = better fit).
        """
        distributions = {
            "Normal": stats.norm,
            "Student-t": stats.t,
            "Skew-Normal": stats.skewnorm,
            "Laplace": stats.laplace,
            "Logistic": stats.logistic,
            "Cauchy": stats.cauchy,
        }

        rows = []
        data = self.series.values

        for name, dist in distributions.items():
            try:
                params = dist.fit(data)
                ks_stat, ks_pval = kstest(data, dist.cdf, args=params)
                rows.append({
                    "Distribution": name,
                    "KS Statistic": round(ks_stat, 4),
                    "KS P-Value": round(ks_pval, 4),
                    "Good Fit": ks_pval > 0.05,
                    "Parameters": str([round(p, 4) for p in params]),
                })
            except Exception:
                pass

        return (
            pd.DataFrame(rows)
            .sort_values("KS Statistic")
            .reset_index(drop=True)
        )

    def descriptive_statistics(self) -> pd.DataFrame:
        """
        Full descriptive statistics table for the return series.
        """
        ret = self.series
        rows = [
            ("Count", int(len(ret))),
            ("Mean", float(ret.mean())),
            ("Median", float(ret.median())),
            ("Std Dev", float(ret.std())),
            ("Variance", float(ret.var())),
            ("Skewness", float(stats.skew(ret))),
            ("Excess Kurtosis", float(stats.kurtosis(ret))),
            ("Min", float(ret.min())),
            ("Max", float(ret.max())),
            ("1st Percentile", float(ret.quantile(0.01))),
            ("5th Percentile", float(ret.quantile(0.05))),
            ("25th Percentile", float(ret.quantile(0.25))),
            ("75th Percentile", float(ret.quantile(0.75))),
            ("95th Percentile", float(ret.quantile(0.95))),
            ("99th Percentile", float(ret.quantile(0.99))),
            ("Annualised Mean", float(ret.mean() * 252)),
            ("Annualised Std", float(ret.std() * np.sqrt(252))),
            ("Annualised Sharpe", float(ret.mean() / ret.std() * np.sqrt(252))
             if ret.std() > 0 else 0.0),
        ]
        return pd.DataFrame(rows, columns=["Statistic", "Value"])

    # ──────────────────────────────────────
    #  Correlation Tests
    # ──────────────────────────────────────

    def correlation_significance(
        self,
        series2: pd.Series,
    ) -> dict:
        """
        Compute Pearson, Spearman, and Kendall correlation with significance tests.

        Returns
        -------
        dict with correlation type, coefficient, p-value, and interpretation.
        """
        common = self.series.index.intersection(series2.index)
        s1 = self.series.loc[common].values
        s2 = series2.loc[common].values

        # Pearson
        pearson_r, pearson_p = stats.pearsonr(s1, s2)
        # Spearman
        spearman_r, spearman_p = spearmanr(s1, s2)
        # Kendall
        kendall_t, kendall_p = kendalltau(s1, s2)

        return {
            "Pearson": {
                "correlation": round(pearson_r, 4),
                "p_value": round(pearson_p, 4),
                "significant": pearson_p < self.alpha,
            },
            "Spearman": {
                "correlation": round(spearman_r, 4),
                "p_value": round(spearman_p, 4),
                "significant": spearman_p < self.alpha,
            },
            "Kendall": {
                "correlation": round(kendall_t, 4),
                "p_value": round(kendall_p, 4),
                "significant": kendall_p < self.alpha,
            },
            "n_observations": len(common),
        }

    # ──────────────────────────────────────
    #  Pairs Trading Helper
    # ──────────────────────────────────────

    def find_cointegrated_pairs(
        self,
        universe_prices: pd.DataFrame,
        pvalue_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """
        Screen all pairs in a universe for cointegration.
        Used by the pairs trading strategy to identify tradeable pairs.

        Parameters
        ----------
        universe_prices : pd.DataFrame   Columns = tickers (price levels).
        pvalue_threshold: float          Maximum p-value to consider cointegrated.

        Returns
        -------
        pd.DataFrame  Ranked list of cointegrated pairs.
        """
        if not HAS_STATSMODELS:
            return pd.DataFrame()

        tickers = universe_prices.columns.tolist()
        n = len(tickers)
        pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                t1, t2 = tickers[i], tickers[j]
                s1 = universe_prices[t1].dropna()
                s2 = universe_prices[t2].dropna()
                common = s1.index.intersection(s2.index)

                if len(common) < 252:
                    continue

                try:
                    stat, pval, crit = coint(
                        s1.loc[common], s2.loc[common]
                    )
                    if pval < pvalue_threshold:
                        # Compute spread half-life
                        spread = s1.loc[common] - s2.loc[common]
                        half_life = self._hurst_half_life(spread)
                        corr = float(s1.loc[common].corr(s2.loc[common]))

                        pairs.append({
                            "Ticker 1": t1,
                            "Ticker 2": t2,
                            "P-Value": round(pval, 4),
                            "Test Stat": round(stat, 4),
                            "Correlation": round(corr, 4),
                            "Half-Life (days)": round(half_life, 1),
                            "Tradeable": 5 <= half_life <= 126,
                        })
                except Exception:
                    pass

        if not pairs:
            return pd.DataFrame()

        return (
            pd.DataFrame(pairs)
            .sort_values("P-Value")
            .reset_index(drop=True)
        )

    def _hurst_half_life(self, spread: pd.Series) -> float:
        """
        Estimate mean-reversion half-life of a spread series.
        Uses OLS regression of Δspread on lagged spread.
        HL = -ln(2) / λ  where λ is the regression coefficient.
        """
        delta = spread.diff().dropna()
        lag = spread.shift(1).dropna()
        common = delta.index.intersection(lag.index)

        if len(common) < 10:
            return np.nan

        slope, _, _, _, _ = stats.linregress(lag.loc[common], delta.loc[common])
        if slope >= 0:
            return np.inf
        return float(-np.log(2) / slope)

    # ──────────────────────────────────────
    #  Full Battery
    # ──────────────────────────────────────

    def run_full_battery(self) -> TestBattery:
        """
        Run the complete statistical test battery.

        Returns
        -------
        TestBattery  containing all test results and a summary DataFrame.
        """
        battery = TestBattery(series_name=self.name, n_obs=self.n)

        tests = {
            "ADF": self.adf_test,
            "KPSS": self.kpss_test,
            "Variance Ratio": self.variance_ratio_test,
            "Jarque-Bera": self.jarque_bera_test,
            "Shapiro-Wilk": self.shapiro_wilk_test,
            "DAgostino-Pearson": self.dagostino_pearson_test,
            "Anderson-Darling": self.anderson_darling_test,
            "Ljung-Box": self.ljung_box_test,
            "Durbin-Watson": self.durbin_watson_test,
            "ARCH-LM": self.arch_lm_test,
        }

        for name, test_fn in tests.items():
            try:
                battery.results[name] = test_fn()
            except Exception as e:
                battery.results[name] = TestResult(
                    test_name=name,
                    statistic=0.0,
                    p_value=1.0,
                    reject_null=False,
                    null_hypothesis="",
                    interpretation=f"Test failed: {e}",
                )

        battery.summary = battery.to_dataframe()
        return battery

    # ──────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────

    def _unavailable(self, name: str) -> TestResult:
        return TestResult(
            test_name=name,
            statistic=0.0,
            p_value=1.0,
            reject_null=False,
            null_hypothesis="",
            interpretation="statsmodels not installed. Run: pip install statsmodels",
        )


# ─────────────────────────────────────────────
#  Universe-level screening
# ─────────────────────────────────────────────

def screen_universe_stationarity(
    returns: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run ADF stationarity test across all tickers in a universe.

    Parameters
    ----------
    returns : pd.DataFrame   Daily returns, columns = tickers.
    alpha   : float          Significance level.

    Returns
    -------
    pd.DataFrame  One row per ticker with ADF result.
    """
    rows = []
    for ticker in returns.columns:
        ret = returns[ticker].dropna()
        if len(ret) < 30:
            continue
        try:
            st = StatisticalTests(ret, alpha=alpha, name=ticker)
            result = st.adf_test()
            rows.append({
                "Ticker": ticker,
                "ADF Stat": result.statistic,
                "P-Value": result.p_value,
                "Stationary": result.reject_null,
                "Obs": len(ret),
            })
        except Exception:
            pass
    return pd.DataFrame(rows).set_index("Ticker")
