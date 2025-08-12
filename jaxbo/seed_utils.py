"""
Utility functions for managing global random seeds across the JaxBo package.
"""

import os
import random
import numpy as np
import jax
import jax.random as jax_random
from .utils.logging_utils import get_logger
log = get_logger("[seed_utils]")

# Global variables
_global_seed = None
_jax_key = None
_np_rng = None  # Will hold np.random.Generator instance


def set_global_seed(seed: int | None = None) -> int:
    """Set global random seed for reproducible results.

    Args:
        seed: The random seed to use. If None, a random seed is generated.

    Returns:
        The seed used (useful when auto-generated).
    """
    global _global_seed, _jax_key, _np_rng

    # Generate a random seed if not provided
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
        log.info(f"No seed provided. Generated seed: {seed}")
    else:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("Seed must be a non-negative integer or None")

    # Set Python built-in random seed
    random.seed(seed)

    # Initialize NumPy Generator with the seed
    _np_rng = np.random.default_rng(seed)

    # Set JAX PRNG key
    _jax_key = jax_random.PRNGKey(seed)

    # Set environment for hash randomness
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Store globally
    _global_seed = seed

    log.info(f"Global random seed set to {seed}")
    return seed


def get_global_seed() -> int:
    """Get the current global seed value."""
    if _global_seed is None:
        raise RuntimeError("No global seed has been set. Call set_global_seed() first.")
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


def get_numpy_rng() -> np.random.Generator:
    """Get the global NumPy random generator."""
    if _np_rng is None:
        raise RuntimeError("No global seed has been set. Call set_global_seed() first.")
    return _np_rng

def ensure_reproducibility(seed: int | None = None) -> int:
    """Ensure reproducibility by setting seeds and JAX configurations.

    Args:
        seed: Seed to use. If None, a random seed is generated.

    Returns:
        The seed used.
    """
    used_seed = set_global_seed(seed)
    jax.config.update("jax_enable_x64", True)
    log.info(f"Reproducibility ensured with seed {used_seed}")
    return used_seed