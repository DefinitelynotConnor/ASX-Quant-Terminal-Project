"""
src/regime_detection
====================
Market regime identification for the ASX Quant Terminal.

Modules
-------
hidden_markov_models     : Gaussian HMM with Baum-Welch / Viterbi
markov_regime_switching  : Hamilton (1989) MRS model
volatility_regimes       : Rule-based vol and trend regime detection
                           + RegimeEngine consensus classifier

Usage
-----
    # Quick consensus regime
    from src.regime_detection.volatility_regimes import RegimeEngine
    engine = RegimeEngine(returns, prices)
    print(engine.current_regime())

    # Full HMM
    from src.regime_detection.hidden_markov_models import HiddenMarkovModel
    hmm = HiddenMarkovModel(returns, n_states=3)
    result = hmm.fit()
    print(result.state_stats)
"""

from .hidden_markov_models import HiddenMarkovModel, HMMResult
from .markov_regime_switching import MarkovRegimeSwitching, MRSResult
from .volatility_regimes import (
    VolatilityRegimeDetector,
    TrendRegimeDetector,
    RegimeEngine,
)

__all__ = [
    "HiddenMarkovModel",
    "HMMResult",
    "MarkovRegimeSwitching",
    "MRSResult",
    "VolatilityRegimeDetector",
    "TrendRegimeDetector",
    "RegimeEngine",
]
