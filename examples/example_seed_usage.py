#!/usr/bin/env python3
"""
Example showing how to use JaxBo's global seed system
"""

import jaxbo
import numpy as np
import jax
import jax.numpy as jnp
from jaxbo.logging_utils import get_logger

log = get_logger("[Seed Example]")

# Set global seed for reproducibility
jaxbo.set_global_seed(42)

log.info(f"Current seed: {jaxbo.get_global_seed()}")

# All random operations will now be reproducible
random_numpy = np.random.random(3)
random_jax = jax.random.normal(jaxbo.get_new_jax_key(), (3,))

log.info(f"NumPy random: {random_numpy}")
log.info(f"JAX random: {random_jax}")

# Reset and verify reproducibility
jaxbo.set_global_seed(42)
random_numpy2 = np.random.random(3)
random_jax2 = jax.random.normal(jaxbo.get_new_jax_key(), (3,))

log.info(f"Same NumPy: {np.allclose(random_numpy, random_numpy2)}")
log.info(f"Same JAX: {np.allclose(random_jax, random_jax2)}")
