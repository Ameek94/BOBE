#!/usr/bin/env python3
"""
Example demonstrating how to use JaxBo's global random seed system.

This example shows how to set a global seed for reproducible results across
all random number generators used in JaxBo (Python, NumPy, and JAX).
"""

import numpy as np
import jax.numpy as jnp
import jax.random as jax_random

# Import JaxBo's seed utilities
from jaxbo import set_global_seed, get_global_seed, get_new_jax_key, ensure_reproducibility

def demonstrate_seed_usage():
    """Demonstrate various ways to use the global seed system."""
    
    print("=== JaxBo Global Random Seed Demonstration ===\n")
    
    # Method 1: Basic global seed setting
    print("1. Setting global seed to 42:")
    set_global_seed(42)
    print(f"   Current global seed: {get_global_seed()}")
    
    # Generate some random numbers to show reproducibility
    print("\n   Generating random numbers with seed 42:")
    print(f"   Python random: {np.random.random():.6f}")
    print(f"   NumPy random:  {np.random.random():.6f}")
    
    # Get JAX random key from the global system
    jax_key = get_new_jax_key()
    jax_random_val = jax_random.normal(jax_key)
    print(f"   JAX random:    {float(jax_random_val):.6f}")
    
    print("\n2. Resetting with same seed - should get identical results:")
    set_global_seed(42)
    print(f"   Python random: {np.random.random():.6f}")
    print(f"   NumPy random:  {np.random.random():.6f}")
    
    jax_key = get_new_jax_key()
    jax_random_val = jax_random.normal(jax_key)
    print(f"   JAX random:    {float(jax_random_val):.6f}")
    
    # Method 2: Using ensure_reproducibility for maximum determinism
    print("\n3. Using ensure_reproducibility() for complete setup:")
    ensure_reproducibility(seed=123)
    print(f"   Current global seed: {get_global_seed()}")
    print("   This also configures JAX for deterministic behavior")
    
    # Method 3: Temporary seed context using decorator
    print("\n4. Using @with_seed decorator for temporary seed context:")
    
    from jaxbo import with_seed
    
    @with_seed(999)
    def temporary_seed_function():
        return np.random.random(), float(jax_random.normal(get_new_jax_key()))
    
    # Current seed is still 123
    print(f"   Before function: seed={get_global_seed()}")
    val1, val2 = temporary_seed_function()
    print(f"   Inside @with_seed(999): Python={val1:.6f}, JAX={val2:.6f}")
    print(f"   After function: seed={get_global_seed()}")  # Should still be 123
    
    # Show that calling the function again gives same results
    val1_repeat, val2_repeat = temporary_seed_function()
    print(f"   Second call (same results): Python={val1_repeat:.6f}, JAX={val2_repeat:.6f}")
    print(f"   Same as first call: {val1 == val1_repeat and val2 == val2_repeat}")

def demonstrate_bayesian_optimization_seed():
    """Show how to use seeds with JaxBo's Bayesian Optimization."""
    
    print("\n=== Using Seeds with Bayesian Optimization ===\n")
    
    # Import JaxBo components
    from jaxbo import external_likelihood, BOBE
    
    # Example objective function (2D banana function)
    def banana_function(x):
        """Rosenbrock banana function"""
        x = np.atleast_2d(x)
        return -((1 - x[:, 0])**2 + 100 * (x[:, 1] - x[:, 0]**2)**2)
    
    # Create likelihood with parameter bounds
    likelihood = external_likelihood(
        loglikelihood=banana_function,
        ndim=2,
        param_list=['x1', 'x2'],
        param_bounds=np.array([[-2, 2], [-1, 3]]).T,
        vectorized=True
    )
    
    print("Setting up Bayesian Optimization with reproducible seed:")
    
    # Create BOBE instance with specific random seed
    bo = BOBE(
        loglikelihood=likelihood,
        maxiters=5,
        random_seed=42,  # This will call set_global_seed(42) internally
        verbose=True
    )
    
    print(f"Global seed after BOBE initialization: {get_global_seed()}")
    print("All random operations in the optimization will now be reproducible!")
    
    # The optimization would be run with: bo.run()
    # (commented out to keep this example simple)

if __name__ == "__main__":
    demonstrate_seed_usage()
    demonstrate_bayesian_optimization_seed()
    
    print("\n=== Summary ===")
    print("JaxBo provides comprehensive random seed management:")
    print("• set_global_seed(seed) - Sets seed for all RNG systems")
    print("• ensure_reproducibility(seed) - Complete deterministic setup")
    print("• @with_seed(seed) - Temporary seed context")
    print("• BOBE(random_seed=seed) - Automatic seed management in BO")
    print("• get_new_jax_key() - Get JAX keys from global system")
