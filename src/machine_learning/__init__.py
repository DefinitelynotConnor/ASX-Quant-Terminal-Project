"""
src/machine_learning
====================
Machine learning alpha generation for the ASX Quant Terminal.

Modules
-------
feature_engineering : Build ML feature matrix from prices, technicals, macro
alpha_models        : Random Forest, Gradient Boosting, XGBoost, Ensemble
"""

from .feature_engineering import FeatureEngineer
from .alpha_models import (
    RandomForestAlpha,
    GradientBoostingAlpha,
    XGBoostAlpha,
    MLEnsemble,
    AlphaModelPipeline,
)

__all__ = [
    "FeatureEngineer",
    "RandomForestAlpha",
    "GradientBoostingAlpha",
    "XGBoostAlpha",
    "MLEnsemble",
    "AlphaModelPipeline",
]
