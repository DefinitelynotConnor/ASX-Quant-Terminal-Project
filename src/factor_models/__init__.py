"""
src/factor_models
=================
Factor model library for the ASX Quant Terminal.

Modules
-------
capm             : Capital Asset Pricing Model (single factor)
fama_french      : Fama-French 3-Factor Model (market, size, value)
momentum_factor  : Cross-sectional momentum signals
value_factor     : Cross-sectional value signals (within momentum_factor.py)
quality_factor   : Cross-sectional quality signals (within momentum_factor.py)

The FactorLibrary class in momentum_factor.py combines all three
cross-sectional factors into a single unified interface.
"""

from .capm import CAPM, CAPMResult
from .fama_french import FamaFrench, FamaFrenchResult
from .momentum_factor import (
    MomentumFactor,
    ValueFactor,
    QualityFactor,
    FactorLibrary,
)

__all__ = [
    "CAPM",
    "CAPMResult",
    "FamaFrench",
    "FamaFrenchResult",
    "MomentumFactor",
    "ValueFactor",
    "QualityFactor",
    "FactorLibrary",
]
