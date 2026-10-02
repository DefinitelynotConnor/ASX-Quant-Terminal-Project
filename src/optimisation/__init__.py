"""
src/optimisation
================
Portfolio optimisation models for the ASX Quant Terminal.

Modules
-------
markowitz_optimizer : Mean-variance, efficient frontier, risk parity
black_litterman     : BL model with investor views and equilibrium prior
"""

from .markowitz_optimizer import MarkowitzOptimizer, OptimisationResult
from .black_litterman import BlackLitterman, BLResult
from .parameter_optimisation import (
    GridSearch,
    RandomSearch,
    BayesianOptimisation,
    WalkForwardOptimisation,
    parameter_sensitivity,
)

__all__ = [
    "MarkowitzOptimizer",
    "OptimisationResult",
    "BlackLitterman",
    "BLResult",
    "GridSearch",
    "RandomSearch",
    "BayesianOptimisation",
    "WalkForwardOptimisation",
    "parameter_sensitivity",
]
