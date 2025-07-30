"""
Utility functions for managing global random seeds across the JaxBo package.

This module provides centralized seed management for reproducible results across
NumPy, JAX, and Python's built-in random module.
"""

import os
import random
import numpy as np
import jax
import jax.random as jax_random
import logging

log = logging.getLogger("[SEED]")

# Global variables to track the current seeds
_global_seed = None
_jax_key = None


def set_global_seed(seed: int) -> None:
    """
    Set the global random seed for reproducible results across all random number generators.
    
    This function sets the seed for:
    - Python's built-in random module
    - NumPy random number generator
    - JAX random number generator
    - Environment variable for any other libraries that respect PYTHONHASHSEED
    
    Args:
        seed (int): The seed value to use for all random number generators.
                   Should be a non-negative integer.
    
    Example:
        >>> from jaxbo import set_global_seed
        >>> set_global_seed(42)
    """
    global _global_seed, _jax_key
    
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a non-negative integer")
    
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set JAX random seed
    _jax_key = jax_random.PRNGKey(seed)
    
    # Set environment variable for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Store the current global seed
    _global_seed = seed
    
    log.info(f"Global random seed set to {seed}")


def get_global_seed() -> int:
    """
    Get the current global seed value.
    
    Returns:
        int: The current global seed, or None if no seed has been set.
    """
    return _global_seed


def get_jax_key() -> jax_random.PRNGKey:
    """
    Get the current JAX random key.
    
    Returns:
        jax.random.PRNGKey: The current JAX random key.
        
    Raises:
        RuntimeError: If no global seed has been set.
    """
    if _jax_key is None:
        raise RuntimeError("No global seed has been set. Call set_global_seed() first.")
    return _jax_key


def split_jax_key() -> tuple[jax_random.PRNGKey, jax_random.PRNGKey]:
    """
    Split the current JAX random key and update the global key.
    
    Returns:
        tuple: A tuple containing (new_key, old_key) where new_key becomes 
               the new global key and old_key can be used for operations.
    
    Raises:
        RuntimeError: If no global seed has been set.
    """
    global _jax_key
    
    if _jax_key is None:
        raise RuntimeError("No global seed has been set. Call set_global_seed() first.")
    
    new_key, use_key = jax_random.split(_jax_key)
    _jax_key = new_key
    
    return new_key, use_key


def get_new_jax_key() -> jax_random.PRNGKey:
    """
    Get a new JAX random key by splitting the current global key.
    
    Returns:
        jax.random.PRNGKey: A new JAX random key for use in operations.
        
    Raises:
        RuntimeError: If no global seed has been set.
    """
    _, use_key = split_jax_key()
    return use_key


def ensure_reproducibility(seed: int = 42) -> None:
    """
    Ensure reproducibility by setting all random seeds and JAX configurations.
    
    This function not only sets the global seed but also configures JAX for
    deterministic behavior where possible.
    
    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    # Set the global seed
    set_global_seed(seed)
    
    # Configure JAX for deterministic behavior
    jax.config.update("jax_enable_x64", True)
    
    # Note: Some JAX operations may still be non-deterministic on GPU
    # due to hardware-level optimizations
    log.info(f"Reproducibility ensured with seed {seed}")


def with_seed(seed: int):
    """
    Decorator to temporarily set a specific seed for a function.
    
    Args:
        seed (int): The seed to use for the decorated function.
        
    Returns:
        Callable: The decorated function.
        
    Example:
        >>> @with_seed(123)
        ... def my_random_function():
        ...     return np.random.random()
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            global _global_seed, _jax_key
            # Save current seed state
            old_seed = _global_seed
            old_jax_key = _jax_key
            old_np_state = np.random.get_state()
            old_random_state = random.getstate()
            
            try:
                # Set temporary seed
                set_global_seed(seed)
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore previous seed state
                _global_seed = old_seed
                _jax_key = old_jax_key
                np.random.set_state(old_np_state)
                random.setstate(old_random_state)
        
        return wrapper
    return decorator
