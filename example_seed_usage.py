#!/usr/bin/env python3
"""
Example showing how to use JaxBo's global seed system
"""

import jaxbo
import numpy as np
import jax
import jax.numpy as jnp

# Set global seed for reproducibility
jaxbo.set_global_seed(42)

print(f"Current seed: {jaxbo.get_global_seed()}")

# All random operations will now be reproducible
random_numpy = np.random.random(3)
random_jax = jax.random.normal(jaxbo.get_new_jax_key(), (3,))

print(f"NumPy random: {random_numpy}")
print(f"JAX random: {random_jax}")

# Reset and verify reproducibility
jaxbo.set_global_seed(42)
random_numpy2 = np.random.random(3)
random_jax2 = jax.random.normal(jaxbo.get_new_jax_key(), (3,))

print(f"Same NumPy: {np.allclose(random_numpy, random_numpy2)}")
print(f"Same JAX: {np.allclose(random_jax, random_jax2)}")
