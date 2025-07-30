"""
JaxBo: A Bayesian Optimization package for cosmology using JAX
"""

# Import seed utilities
from .seed_utils import (
    set_global_seed,
    get_global_seed,
    get_jax_key,
    get_new_jax_key,
    split_jax_key,
    ensure_reproducibility,
    with_seed
)

# Import logging utilities
from .logging_utils import (
    get_logger,
    setup_logger,
    configure_package_logging,
    set_global_log_level,
    enable_debug_logging,
    disable_logging
)

# Import main classes and functions
from .bo import BOBE
from .loglike import cobaya_likelihood, external_likelihood
from .acquisition import WIPV, EI
from .gp import DSLP_GP, SAAS_GP
from .nested_sampler import nested_sampling_Dy, nested_sampling_jaxns
from .optim import optimize

__version__ = "0.1.0"

__all__ = [
    # Seed utilities
    "set_global_seed",
    "get_global_seed", 
    "get_jax_key",
    "get_new_jax_key",
    "split_jax_key",
    "ensure_reproducibility",
    "with_seed",
    # Logging utilities
    "get_logger",
    "setup_logger", 
    "configure_package_logging",
    "set_global_log_level",
    "enable_debug_logging",
    "disable_logging",
    # Main classes and functions
    "BOBE",
    "external_loglike",
    "external_likelihood", 
    "WIPV",
    "EI",
    "DSLP_GP",
    "SAAS_GP", 
    "nested_sampling_Dy",
    "nested_sampling_jaxns",
    "optimize",
]