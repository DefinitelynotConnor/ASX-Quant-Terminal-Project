"""
src/analytics
=============
Market analytics layer for the ASX Quant Terminal.

Modules
-------
technical_indicators : Price-based technical indicators (RSI, MACD, Bollinger etc.)
volatility_models    : GARCH, realised vol, volatility surface (next module)
statistical_tests    : Stationarity, cointegration, normality tests (next module)
"""

from .technical_indicators import TechnicalIndicators, compute_universe_indicators
from .statistical_tests import StatisticalTests, screen_universe_stationarity
from .volatility_models import VolatilityModels

__all__ = [
    "TechnicalIndicators",
    "compute_universe_indicators",
    "StatisticalTests",
    "screen_universe_stationarity",
    "VolatilityModels",
]
