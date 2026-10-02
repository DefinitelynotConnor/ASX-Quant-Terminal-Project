"""
src/strategies
==============
Trading strategy library for the ASX Quant Terminal.

All strategies produce signals compatible with the Backtester engine.

Modules
-------
momentum_strategy       : CrossSectional, TimeSeries, DualMomentum, RiskAdjusted
mean_reversion_strategy : BollingerBand, RSI mean reversion
pairs_trading           : PairsTrading with OLS/Kalman hedge ratio
factor_strategy         : FactorStrategy, RegimeAdaptive, FactorTiming
"""

from .momentum_strategy import (
    CrossSectionalMomentum,
    TimeSeriesMomentum,
    DualMomentumStrategy,
    RiskAdjustedMomentum,
)
from .mean_reversion_strategy import (
    BollingerBandReversion,
    RSIMeanReversion,
    PairsTrading,
)
from .factor_strategy import (
    FactorStrategy,
    RegimeAdaptiveStrategy,
    FactorTimingStrategy,
)

__all__ = [
    "CrossSectionalMomentum",
    "TimeSeriesMomentum",
    "DualMomentumStrategy",
    "RiskAdjustedMomentum",
    "BollingerBandReversion",
    "RSIMeanReversion",
    "PairsTrading",
    "FactorStrategy",
    "RegimeAdaptiveStrategy",
    "FactorTimingStrategy",
]
