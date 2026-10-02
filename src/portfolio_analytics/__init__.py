"""
Portfolio Analytics Module
==========================
Institutional-grade portfolio diagnostics for ASX securities.

Modules:
    portfolio_metrics     - Performance and risk statistics
    exposure_analysis     - Factor, sector, and beta exposures
    performance_attribution - Return decomposition and attribution
    risk_decomposition    - Marginal and component risk contribution
    portfolio_dashboard   - Streamlit interactive dashboard
"""

from .portfolio_metrics import PortfolioMetrics
from .exposure_analysis import ExposureAnalyser
from .performance_attribution import PerformanceAttributor
from .risk_decomposition import RiskDecomposer

__all__ = [
    "PortfolioMetrics",
    "ExposureAnalyser",
    "PerformanceAttributor",
    "RiskDecomposer",
]
