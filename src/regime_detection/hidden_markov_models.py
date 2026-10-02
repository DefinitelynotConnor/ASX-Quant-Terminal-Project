"""
hidden_markov_models.py
=======================
Hidden Markov Model (HMM) regime detection for ASX equity markets.

The HMM treats market regimes as hidden (unobservable) states that drive
the observed return distribution. The model learns which regime is most
likely at each point in time from the return data alone.

Model
-----
    States    : N hidden regimes (e.g. Bull, Bear, Sideways)
    Emissions : Gaussian distribution per regime
                (each regime has its own mean and variance)
    Transition: Markov transition matrix P[i,j] = P(regime j | regime i)

Implementation
--------------
    Uses Baum-Welch (EM algorithm) for parameter estimation.
    Viterbi algorithm for most likely state sequence.
    Forward-backward algorithm for smoothed state probabilities.

Usage
-----
    from src.regime_detection.hidden_markov_models import HiddenMarkovModel

    hmm = HiddenMarkovModel(returns, n_states=3)
    hmm.fit()
    regimes = hmm.predict_regimes()
    probs = hmm.state_probabilities()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

warnings.filterwarnings("ignore")

try:
    from hmmlearn import hmm as hmmlearn_hmm
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False


# ─────────────────────────────────────────────
#  Regime labels
# ─────────────────────────────────────────────

REGIME_LABELS_2 = {0: "Bear", 1: "Bull"}
REGIME_LABELS_3 = {0: "Bear", 1: "Sideways", 2: "Bull"}
REGIME_LABELS_4 = {0: "Bear", 1: "High Vol", 2: "Sideways", 3: "Bull"}

REGIME_COLOURS = {
    "Bull": "#00C853",
    "Sideways": "#FFD600",
    "High Vol": "#FF6B35",
    "Bear": "#FF1744",
}


# ─────────────────────────────────────────────
#  Result Container
# ─────────────────────────────────────────────

@dataclass
class HMMResult:
    """Container for fitted HMM results."""
    n_states: int = 2
    regime_sequence: pd.Series = field(default_factory=pd.Series)
    regime_labels: pd.Series = field(default_factory=pd.Series)
    state_probs: pd.DataFrame = field(default_factory=pd.DataFrame)
    transition_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    state_means: np.ndarray = field(default_factory=lambda: np.array([]))
    state_stds: np.ndarray = field(default_factory=lambda: np.array([]))
    state_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    log_likelihood: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    regime_durations: pd.DataFrame = field(default_factory=pd.DataFrame)
    label_map: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
#  HiddenMarkovModel
# ─────────────────────────────────────────────

class HiddenMarkovModel:
    """
    Gaussian Hidden Markov Model for market regime detection.

    Parameters
    ----------
    returns   : pd.Series   Daily return series.
    n_states  : int         Number of hidden regimes (2, 3, or 4).
    n_iter    : int         Maximum EM iterations.
    tol       : float       Convergence tolerance.
    random_state : int      Reproducibility seed.
    """

    def __init__(
        self,
        returns: pd.Series,
        n_states: int = 3,
        n_iter: int = 200,
        tol: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.returns = returns.dropna().copy()
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self._result: Optional[HMMResult] = None

        # Choose label map based on n_states
        self.label_map = {
            2: REGIME_LABELS_2,
            3: REGIME_LABELS_3,
            4: REGIME_LABELS_4,
        }.get(n_states, {i: f"State {i}" for i in range(n_states)})

    # ──────────────────────────────────────
    #  Fitting
    # ──────────────────────────────────────

    def fit(self) -> HMMResult:
        """
        Fit the HMM using Baum-Welch (EM algorithm).
        Falls back to manual Gaussian HMM if hmmlearn is unavailable.

        Returns
        -------
        HMMResult with fitted parameters and regime assignments.
        """
        if HAS_HMMLEARN:
            result = self._fit_hmmlearn()
        else:
            result = self._fit_manual()

        # Re-label states by mean return (state 0 = lowest mean = Bear)
        result = self._sort_states_by_mean(result)
        result.regime_labels = result.regime_sequence.map(result.label_map)
        result.state_stats = self._compute_state_stats(result)
        result.regime_durations = self._compute_regime_durations(result)

        self._result = result
        return result

    def _fit_hmmlearn(self) -> HMMResult:
        """Fit using the hmmlearn library — most robust implementation."""
        np.random.seed(self.random_state)
        X = self.returns.values.reshape(-1, 1)

        model = hmmlearn_hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        model.fit(X)

        states = model.predict(X)
        state_probs = pd.DataFrame(
            model.predict_proba(X),
            index=self.returns.index,
            columns=[f"State_{i}" for i in range(self.n_states)],
        )
        log_ll = model.score(X)
        n = len(X)
        k = self.n_states ** 2 + 2 * self.n_states   # params
        aic = -2 * log_ll + 2 * k
        bic = -2 * log_ll + k * np.log(n)

        means = model.means_.flatten()
        stds = np.sqrt(model.covars_.flatten())

        return HMMResult(
            n_states=self.n_states,
            regime_sequence=pd.Series(states, index=self.returns.index, name="Regime"),
            state_probs=state_probs,
            transition_matrix=model.transmat_,
            state_means=means,
            state_stds=stds,
            log_likelihood=log_ll,
            aic=aic,
            bic=bic,
            label_map=self.label_map.copy(),
        )

    def _fit_manual(self) -> HMMResult:
        """
        Manual Gaussian HMM via Baum-Welch EM algorithm.
        Used when hmmlearn is not installed.
        """
        np.random.seed(self.random_state)
        obs = self.returns.values
        n = len(obs)
        K = self.n_states

        # Initialise parameters using k-means style split
        sorted_obs = np.sort(obs)
        chunk = len(sorted_obs) // K
        means = np.array([
            sorted_obs[i * chunk: (i + 1) * chunk].mean()
            for i in range(K)
        ])
        stds = np.full(K, obs.std())
        trans = np.full((K, K), 1 / K)
        init_probs = np.full(K, 1 / K)

        log_ll_prev = -np.inf

        for iteration in range(self.n_iter):
            # E-step: forward-backward
            log_emit = np.column_stack([
                stats.norm.logpdf(obs, means[k], stds[k])
                for k in range(K)
            ])

            alpha = self._forward(log_emit, np.log(init_probs + 1e-300),
                                  np.log(trans + 1e-300))
            beta = self._backward(log_emit, np.log(trans + 1e-300))

            log_gamma = alpha + beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # Xi (joint probability of consecutive states)
            log_xi = np.zeros((n - 1, K, K))
            for t in range(n - 1):
                for i in range(K):
                    for j in range(K):
                        log_xi[t, i, j] = (
                            alpha[t, i]
                            + np.log(trans[i, j] + 1e-300)
                            + log_emit[t + 1, j]
                            + beta[t + 1, j]
                        )
                log_xi[t] -= logsumexp(log_xi[t].flatten())

            xi = np.exp(log_xi)

            # M-step: update parameters
            init_probs = gamma[0] + 1e-300
            init_probs /= init_probs.sum()

            for k in range(K):
                g_k = gamma[:, k]
                means[k] = np.dot(g_k, obs) / (g_k.sum() + 1e-300)
                stds[k] = np.sqrt(
                    np.dot(g_k, (obs - means[k]) ** 2) / (g_k.sum() + 1e-300)
                )
                stds[k] = max(stds[k], 1e-6)

            trans = xi.sum(axis=0)
            trans_sum = trans.sum(axis=1, keepdims=True)
            trans = trans / (trans_sum + 1e-300)

            # Check convergence
            log_ll = logsumexp(alpha[-1])
            if abs(log_ll - log_ll_prev) < self.tol:
                break
            log_ll_prev = log_ll

        # Viterbi decoding for most likely state sequence
        states = self._viterbi(obs, means, stds, trans, init_probs)

        state_probs_arr = gamma
        state_probs = pd.DataFrame(
            state_probs_arr,
            index=self.returns.index,
            columns=[f"State_{i}" for i in range(K)],
        )

        k_params = K ** 2 + 2 * K
        aic = -2 * log_ll + 2 * k_params
        bic = -2 * log_ll + k_params * np.log(n)

        return HMMResult(
            n_states=self.n_states,
            regime_sequence=pd.Series(states, index=self.returns.index, name="Regime"),
            state_probs=state_probs,
            transition_matrix=trans,
            state_means=means,
            state_stds=stds,
            log_likelihood=float(log_ll),
            aic=float(aic),
            bic=float(bic),
            label_map=self.label_map.copy(),
        )

    def _forward(
        self,
        log_emit: np.ndarray,
        log_init: np.ndarray,
        log_trans: np.ndarray,
    ) -> np.ndarray:
        """Forward pass (log-scale for numerical stability)."""
        n, K = log_emit.shape
        alpha = np.full((n, K), -np.inf)
        alpha[0] = log_init + log_emit[0]

        for t in range(1, n):
            for j in range(K):
                alpha[t, j] = (
                    logsumexp(alpha[t - 1] + log_trans[:, j]) + log_emit[t, j]
                )
        return alpha

    def _backward(
        self,
        log_emit: np.ndarray,
        log_trans: np.ndarray,
    ) -> np.ndarray:
        """Backward pass (log-scale)."""
        n, K = log_emit.shape
        beta = np.zeros((n, K))

        for t in range(n - 2, -1, -1):
            for i in range(K):
                beta[t, i] = logsumexp(
                    log_trans[i] + log_emit[t + 1] + beta[t + 1]
                )
        return beta

    def _viterbi(
        self,
        obs: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
        trans: np.ndarray,
        init_probs: np.ndarray,
    ) -> np.ndarray:
        """Viterbi algorithm — most likely state sequence."""
        n = len(obs)
        K = len(means)

        log_emit = np.column_stack([
            stats.norm.logpdf(obs, means[k], stds[k]) for k in range(K)
        ])
        log_trans = np.log(trans + 1e-300)

        viterbi = np.full((n, K), -np.inf)
        psi = np.zeros((n, K), dtype=int)

        viterbi[0] = np.log(init_probs + 1e-300) + log_emit[0]

        for t in range(1, n):
            for j in range(K):
                scores = viterbi[t - 1] + log_trans[:, j]
                psi[t, j] = np.argmax(scores)
                viterbi[t, j] = scores[psi[t, j]] + log_emit[t, j]

        # Backtrack
        states = np.zeros(n, dtype=int)
        states[-1] = np.argmax(viterbi[-1])
        for t in range(n - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    # ──────────────────────────────────────
    #  Post-processing
    # ──────────────────────────────────────

    def _sort_states_by_mean(self, result: HMMResult) -> HMMResult:
        """
        Re-order states so state 0 = lowest mean return (Bear)
        and state N-1 = highest mean return (Bull).
        """
        order = np.argsort(result.state_means)
        mapping = {old: new for new, old in enumerate(order)}

        result.regime_sequence = result.regime_sequence.map(mapping)
        result.state_means = result.state_means[order]
        result.state_stds = result.state_stds[order]

        # Reorder transition matrix
        P = result.transition_matrix
        P_sorted = P[np.ix_(order, order)]
        result.transition_matrix = P_sorted

        # Reorder state probs columns
        old_cols = [f"State_{i}" for i in range(self.n_states)]
        new_cols = [f"State_{mapping[i]}" for i in range(self.n_states)]
        result.state_probs = result.state_probs.rename(
            columns=dict(zip(old_cols, new_cols))
        )
        result.state_probs = result.state_probs[
            [f"State_{i}" for i in range(self.n_states)]
        ]

        return result

    def _compute_state_stats(self, result: HMMResult) -> pd.DataFrame:
        """Per-regime descriptive statistics."""
        rows = []
        for state in range(self.n_states):
            mask = result.regime_sequence == state
            regime_ret = self.returns[mask]
            label = result.label_map.get(state, f"State {state}")

            rows.append({
                "State": state,
                "Label": label,
                "Mean Return (ann.)": float(regime_ret.mean() * 252),
                "Volatility (ann.)": float(regime_ret.std() * np.sqrt(252)),
                "Sharpe": float(
                    regime_ret.mean() / regime_ret.std() * np.sqrt(252)
                ) if regime_ret.std() > 0 else 0,
                "% of Time": float(mask.mean()),
                "Obs Count": int(mask.sum()),
                "Avg Duration (days)": self._avg_duration(result.regime_sequence, state),
                "Transition Prob (stay)": float(result.transition_matrix[state, state]),
            })
        return pd.DataFrame(rows).set_index("State")

    def _avg_duration(self, sequence: pd.Series, state: int) -> float:
        """Average consecutive days in a given regime."""
        durations = []
        count = 0
        for val in sequence:
            if val == state:
                count += 1
            else:
                if count > 0:
                    durations.append(count)
                count = 0
        if count > 0:
            durations.append(count)
        return float(np.mean(durations)) if durations else 0.0

    def _compute_regime_durations(self, result: HMMResult) -> pd.DataFrame:
        """
        Identify all distinct regime episodes with start, end, duration.
        """
        rows = []
        seq = result.regime_sequence
        prev_state = seq.iloc[0]
        start = seq.index[0]

        for date, state in seq.iloc[1:].items():
            if state != prev_state:
                label = result.label_map.get(int(prev_state), f"State {prev_state}")
                rows.append({
                    "Start": start,
                    "End": date,
                    "Label": label,
                    "Duration (days)": (date - start).days,
                    "State": int(prev_state),
                })
                start = date
                prev_state = state

        # Last regime
        label = result.label_map.get(int(prev_state), f"State {prev_state}")
        rows.append({
            "Start": start,
            "End": seq.index[-1],
            "Label": label,
            "Duration (days)": (seq.index[-1] - start).days,
            "State": int(prev_state),
        })

        return pd.DataFrame(rows)

    # ──────────────────────────────────────
    #  Prediction
    # ──────────────────────────────────────

    def predict_regimes(self) -> pd.Series:
        """Return the regime label series (e.g. 'Bull', 'Bear', 'Sideways')."""
        if self._result is None:
            self.fit()
        return self._result.regime_labels

    def current_regime(self) -> dict:
        """Return the current (most recent) regime and its probability."""
        if self._result is None:
            self.fit()
        latest_state = int(self._result.regime_sequence.iloc[-1])
        latest_probs = self._result.state_probs.iloc[-1]
        label = self._result.label_map.get(latest_state, f"State {latest_state}")
        return {
            "regime": label,
            "state": latest_state,
            "probability": float(latest_probs.iloc[latest_state]),
            "all_probabilities": {
                self._result.label_map.get(i, f"State {i}"): float(p)
                for i, p in enumerate(latest_probs)
            },
        }

    def state_probabilities(self) -> pd.DataFrame:
        """Return smoothed state probability time series."""
        if self._result is None:
            self.fit()
        df = self._result.state_probs.copy()
        df.columns = [
            self._result.label_map.get(int(c.split("_")[1]), c)
            for c in df.columns
        ]
        return df

    def get_result(self) -> HMMResult:
        """Return the full fitted result object."""
        if self._result is None:
            self.fit()
        return self._result

    # ──────────────────────────────────────
    #  Model selection
    # ──────────────────────────────────────

    @classmethod
    def select_n_states(
        cls,
        returns: pd.Series,
        max_states: int = 5,
    ) -> pd.DataFrame:
        """
        Fit HMMs with 2 to max_states regimes and compare by AIC/BIC.
        Helps select the optimal number of regimes.

        Returns
        -------
        pd.DataFrame  rows = n_states, columns = [log_likelihood, AIC, BIC]
        """
        rows = []
        for n in range(2, max_states + 1):
            try:
                model = cls(returns, n_states=n)
                result = model.fit()
                rows.append({
                    "N States": n,
                    "Log Likelihood": result.log_likelihood,
                    "AIC": result.aic,
                    "BIC": result.bic,
                })
            except Exception as e:
                print(f"  [HMM] n={n} failed: {e}")

        df = pd.DataFrame(rows).set_index("N States")
        optimal_aic = int(df["AIC"].idxmin())
        optimal_bic = int(df["BIC"].idxmin())
        print(f"  [HMM] Optimal n_states: AIC→{optimal_aic}, BIC→{optimal_bic}")
        return df
