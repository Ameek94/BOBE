"""
Utilities package for JaxBo.

This package contains utility modules for results management, plotting, 
logging, timing, and other helper functions.
"""

# Import key utilities for easy access
from .results import BOBEResults, load_bobe_results, create_resumable_results
from .summary_plots import BOBESummaryPlotter, plot_final_samples
from .core_utils import (
    suppress_stdout_stderr, 
    split_vmap, 
    scale_to_unit, 
    scale_from_unit, 
    renormalise_log_weights, 
    resample_equal
)
from .logging_utils import get_logger, setup_logging
from .seed_utils import get_numpy_rng, get_jax_key, set_global_seed
from .timing import BOBETimingCollector

__all__ = [
    # Results management
    'BOBEResults', 'load_bobe_results',
    # Plotting
    'BOBESummaryPlotter', 'plot_final_samples', 
    # Core utilities
    'suppress_stdout_stderr', 'split_vmap', 'scale_to_unit', 'scale_from_unit',
    'renormalise_log_weights', 'resample_equal',
    # Logging
    'get_logger', 'setup_logging',
    # Random number generation
    'get_numpy_rng', 'get_jax_rng', 'set_global_seed',
    # Timing
    'TimingManager'
]
