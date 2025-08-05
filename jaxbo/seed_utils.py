"""
Utility functions for managing global random seeds across the JaxBo package.
"""

import os
import random
import numpy as np
import jax
import jax.random as jax_random
import logging

log = logging.getLogger(__name__)

# Global variables
_global_seed = None
_jax_key = None

def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducible results."""
    global _global_seed, _jax_key

    if seed is None:
        seed = random.randint(0, 2**31 - 1)  # Generate a random seed if none is provided
    
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a non-negative integer")
    
    random.seed(seed)
    np.random.seed(seed)
    _jax_key = jax_random.PRNGKey(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    _global_seed = seed
    
    log.info(f"Global random seed set to {seed}")

def get_global_seed() -> int:
    """Get the current global seed value."""
    return _global_seed

def get_jax_key() -> jax_random.PRNGKey:
    """Get the current JAX random key."""
    if _jax_key is None:
        raise RuntimeError("No global seed has been set. Call set_global_seed() first.")
    return _jax_key

def split_jax_key() -> tuple[jax_random.PRNGKey, jax_random.PRNGKey]:
    """Split the current JAX random key and update the global key."""
    global _jax_key
    
    if _jax_key is None:
        raise RuntimeError("No global seed has been set. Call set_global_seed() first.")
    
    _jax_key, use_key = jax_random.split(_jax_key)
    return _jax_key, use_key

def get_new_jax_key() -> jax_random.PRNGKey:
    """Get a new JAX random key by splitting the current global key."""
    _, use_key = split_jax_key()
    return use_key

def ensure_reproducibility(seed: int = 42) -> None:
    """Ensure reproducibility by setting seeds and JAX configurations."""
    set_global_seed(seed)
    jax.config.update("jax_enable_x64", True)
    log.info(f"Reproducibility ensured with seed {seed}")