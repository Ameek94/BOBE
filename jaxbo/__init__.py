"""
JaxBO - Bayesian Optimization for Expensive Likelihoods using JAX

JaxBO is a package for performing Bayesian model comparison for expensive 
likelihood functions, developed for applications to cosmology. It uses 
Bayesian Optimization to train Gaussian process surrogates and runs 
nested sampling/MCMC on the surrogate instead of the underlying expensive 
likelihood.

Main Components:
- BOBE: Main Bayesian Optimization class
- GP: Gaussian Process implementation
- GPwithClassifier: GP with SVM/NN/Ellipsoid classifier for filtering
- Likelihood classes: Likelihood, CobayaLikelihood
- Acquisition functions: EI, LogEI, WIPV, WIPStd
"""

__version__ = "0.1.0"

# Core classes
from .bo import BOBE, load_gp
from .gp import GP
from .clf_gp import GPwithClassifier

# Likelihood interfaces
from .likelihood import Likelihood

try:
    from .likelihood import CobayaLikelihood
    _COBAYA_AVAILABLE = True
except ImportError:
    _COBAYA_AVAILABLE = False
    CobayaLikelihood = None

# Acquisition functions
from .acquisition import EI, LogEI, WIPV, WIPStd

# Utilities (commonly used)
from .utils import (
    BOBEResults,
    BOBESummaryPlotter,
    get_logger,
    scale_to_unit,
    scale_from_unit,
)

# High-level interface
from .run import run_bobe

# Build __all__ dynamically based on available optional dependencies
__all__ = [
    # Version
    "__version__",
    # Core classes
    "BOBE",
    "GP",
    "GPwithClassifier",
    "load_gp",
    # Likelihood classes
    "Likelihood",
    # Acquisition
    "EI",
    "LogEI",
    "WIPV",
    "WIPStd",
    # Utilities
    "BOBEResults",
    "BOBESummaryPlotter",
    "get_logger",
    "scale_to_unit",
    "scale_from_unit",
    # High-level
    "run_bobe",
]

# Add optional dependencies to __all__ if available
if _COBAYA_AVAILABLE:
    __all__.append("CobayaLikelihood")
