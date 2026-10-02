"""
markov_regime_switching.py
==========================
Hamilton (1989) Markov Regime Switching model for return series.

Unlike HMM (which uses EM), this model estimates regime parameters
via maximum likelihood directly, producing regime-conditional
return distributions and transition probabilities.

Complements hidden_markov_models.py — use both and compare results.

Usage
-----
    from src.regime_detection.markov_regime_switching import MarkovRegimeSwitching

    mrs = MarkovRegimeSwitching(returns, n_regimes=2)
    mrs.fit()
    regimes = mrs.get_regimes()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    HAS_STATSMODELS_MRS = True
except ImportError:
    HAS_STATSMODELS_MRS = False


@dataclass
class MRSResult:
    """Container for Markov Regime Switching results."""
    n_regimes: int = 2
    regime_sequence: pd.Series = field(default_factory=pd.Series)
    regime_labels: pd.Series = field(default_factory=pd.Series)
    smoothed_probs: pd.DataFrame = field(default_factory=pd.DataFrame)
    filtered_probs: pd.DataFrame = field(default_factory=pd.DataFrame)
    transition_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    regime_means: np.ndarray = field(default_factory=lambda: np.array([]))
    regime_stds: np.ndarray = field(default_factory=lambda: np.array([]))
    log_likelihood: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    regime_stats: pd.DataFrame = field(default_factory=pd.DataFrame)


class MarkovRegimeSwitching:
    """
    Hamilton (1989) Markov Regime Switching model.

    Parameters
    ----------
    returns   : pd.Series   Daily return series.
    n_regimes : int         Number of regimes (2 or 3 recommended).
    """

    LABEL_MAP = {
        2: {0: "Bear", 1: "Bull"},
        3: {0: "Bear", 1: "Sideways", 2: "Bull"},
    }

    def __init__(
        self,
        returns: pd.Series,
        n_regimes: int = 2,
    ) -> None:
        self.returns = returns.dropna().copy()
        self.n_regimes = n_regimes
        self.label_map = self.LABEL_MAP.get(
            n_regimes, {i: f"Regime {i}" for i in range(n_regimes)}
        )
        self._result: Optional[MRSResult] = None

    def fit(self) -> MRSResult:
        """Fit the Markov Regime Switching model."""
        if HAS_STATSMODELS_MRS:
            result = self._fit_statsmodels()
        else:
            result = self._fit_manual()

        result = self._sort_by_mean(result)
        result.regime_labels = result.regime_sequence.map(self.label_map)
        result.regime_stats = self._compute_stats(result)
        self._result = result
        return result

    def _fit_statsmodels(self) -> MRSResult:
        """Fit using statsmodels MarkovRegression."""
        try:
            model = MarkovRegression(
                self.returns,
                k_regimes=self.n_regimes,
                trend="c",
                switching_variance=True,
            )
            fit = model.fit(disp=False, maxiter=200)

            # Extract smoothed probabilities
            smoothed = pd.DataFrame(
                fit.smoothed_marginal_probabilities,
                index=self.returns.index,
                columns=[f"Regime_{i}" for i in range(self.n_regimes)],
            )
            filtered = pd.DataFrame(
                fit.filtered_marginal_probabilities,
                index=self.returns.index,
                columns=[f"Regime_{i}" for i in range(self.n_regimes)],
            )

            regime_seq = pd.Series(
                smoothed.values.argmax(axis=1),
                index=self.returns.index,
                name="Regime",
            )

            # Extract parameters
            means = np.array([
                float(fit.params[f"[{i}]const"])
                for i in range(self.n_regimes)
            ])
            stds = np.array([
                np.sqrt(float(fit.params[f"[{i}]sigma2"]))
                for i in range(self.n_regimes)
            ])

            # Transition matrix
            trans = fit.regime_transition.T

            n = len(self.returns)
            k = self.n_regimes * 2 + self.n_regimes * (self.n_regimes - 1)
            ll = float(fit.llf)
            aic = -2 * ll + 2 * k
            bic = -2 * ll + k * np.log(n)

            return MRSResult(
                n_regimes=self.n_regimes,
                regime_sequence=regime_seq,
                smoothed_probs=smoothed,
                filtered_probs=filtered,
                transition_matrix=trans,
                regime_means=means,
                regime_stds=stds,
                log_likelihood=ll,
                aic=aic,
                bic=bic,
                label_map=self.label_map.copy(),
            )

        except Exception as e:
            print(f"  [MRS] statsmodels failed: {e}. Using manual MLE.")
            return self._fit_manual()

    def _fit_manual(self) -> MRSResult:
        """
        Manual 2-regime Hamilton filter via maximum likelihood.
        """
        obs = self.returns.values
        n = len(obs)
        K = self.n_regimes

        def neg_log_likelihood(params):
            # Unpack parameters
            means = params[:K]
            log_stds = params[K:2 * K]
            stds = np.exp(log_stds)

            # Transition probabilities (logit parameterisation)
            # p11 = P(stay in regime 0), p22 = P(stay in regime 1)
            logit_p = params[2 * K: 2 * K + K]
            stay_probs = 1 / (1 + np.exp(-logit_p))

            # Build transition matrix
            trans = np.full((K, K), (1 - stay_probs[0]) / (K - 1))
            for k in range(K):
                trans[k, k] = stay_probs[k]
                row_sum = trans[k].sum() - trans[k, k]
                off_diag = 1 - trans[k, k]
                if K > 2:
                    for j in range(K):
                        if j != k:
                            trans[k, j] = off_diag / (K - 1)

            # Hamilton filter
            probs = np.full(K, 1 / K)
            log_ll = 0.0

            for t in range(n):
                emit = np.array([
                    stats.norm.pdf(obs[t], means[k], stds[k]) + 1e-300
                    for k in range(K)
                ])
                joint = probs @ trans
                joint_weighted = joint * emit
                total = joint_weighted.sum()
                if total <= 0:
                    return 1e10
                log_ll += np.log(total)
                probs = joint_weighted / total

            return -log_ll

        # Initial parameters
        sorted_obs = np.sort(obs)
        chunk = len(sorted_obs) // K
        init_means = [sorted_obs[i * chunk: (i + 1) * chunk].mean() for i in range(K)]
        init_log_stds = [np.log(obs.std())] * K
        init_logit = [2.0] * K  # high probability of staying

        x0 = init_means + init_log_stds + init_logit

        result = minimize(neg_log_likelihood, x0, method="Nelder-Mead",
                          options={"maxiter": 2000, "xatol": 1e-6})

        params = result.x
        means = params[:K]
        stds = np.exp(params[K:2 * K])
        stay = 1 / (1 + np.exp(-params[2 * K:]))

        trans = np.full((K, K), 0.05)
        for k in range(K):
            trans[k, k] = stay[k]
            off = (1 - stay[k]) / (K - 1)
            for j in range(K):
                if j != k:
                    trans[k, j] = off

        # Run filter for state probabilities
        probs_hist = []
        probs = np.full(K, 1 / K)
        for t in range(n):
            emit = np.array([
                stats.norm.pdf(obs[t], means[k], stds[k]) + 1e-300
                for k in range(K)
            ])
            joint = (probs @ trans) * emit
            total = joint.sum()
            probs = joint / total if total > 0 else np.full(K, 1 / K)
            probs_hist.append(probs.copy())

        probs_arr = np.array(probs_hist)
        smoothed = pd.DataFrame(
            probs_arr,
            index=self.returns.index,
            columns=[f"Regime_{i}" for i in range(K)],
        )
        regime_seq = pd.Series(
            probs_arr.argmax(axis=1),
            index=self.returns.index,
            name="Regime",
        )

        ll = -result.fun
        k_params = K * 2 + K
        aic = -2 * ll + 2 * k_params
        bic = -2 * ll + k_params * np.log(n)

        return MRSResult(
            n_regimes=self.n_regimes,
            regime_sequence=regime_seq,
            smoothed_probs=smoothed,
            filtered_probs=smoothed,
            transition_matrix=trans,
            regime_means=means,
            regime_stds=stds,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            label_map=self.label_map.copy(),
        )

    def _sort_by_mean(self, result: MRSResult) -> MRSResult:
        """Sort regimes so lowest mean = regime 0 (Bear)."""
        order = np.argsort(result.regime_means)
        mapping = {old: new for new, old in enumerate(order)}
        result.regime_sequence = result.regime_sequence.map(mapping)
        result.regime_means = result.regime_means[order]
        result.regime_stds = result.regime_stds[order]
        P = result.transition_matrix
        if P.size > 0:
            result.transition_matrix = P[np.ix_(order, order)]
        return result

    def _compute_stats(self, result: MRSResult) -> pd.DataFrame:
        """Per-regime statistics."""
        rows = []
        for state in range(self.n_regimes):
            mask = result.regime_sequence == state
            ret = self.returns[mask]
            label = self.label_map.get(state, f"Regime {state}")
            rows.append({
                "Regime": label,
                "Ann. Return": float(ret.mean() * 252),
                "Ann. Volatility": float(ret.std() * np.sqrt(252)),
                "Sharpe": (
                    float(ret.mean() / ret.std() * np.sqrt(252))
                    if ret.std() > 0 else 0
                ),
                "% of Time": float(mask.mean()),
                "Stay Probability": (
                    float(result.transition_matrix[state, state])
                    if result.transition_matrix.size > 0 else np.nan
                ),
                "Obs": int(mask.sum()),
            })
        return pd.DataFrame(rows)

    def get_regimes(self) -> pd.Series:
        """Return the regime label series."""
        if self._result is None:
            self.fit()
        return self._result.regime_labels

    def get_result(self) -> MRSResult:
        """Return full result object."""
        if self._result is None:
            self.fit()
        return self._result

    def current_regime(self) -> dict:
        """Most recent regime and its probability."""
        if self._result is None:
            self.fit()
        latest_state = int(self._result.regime_sequence.iloc[-1])
        latest_probs = self._result.smoothed_probs.iloc[-1]
        label = self.label_map.get(latest_state, f"Regime {latest_state}")
        return {
            "regime": label,
            "state": latest_state,
            "probability": float(latest_probs.iloc[latest_state]),
        }
