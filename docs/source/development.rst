Development Guide
=================

This guide covers advanced development topics for BOBE contributors.

Architecture Overview
---------------------

BOBE is structured around several key components:

Core Components
~~~~~~~~~~~~~~~

- **BOBE Class** (``bo.py``): Main optimization engine
- **Gaussian Processes** (``gp.py``, ``clf_gp.py``): Surrogate models
- **Acquisition Functions** (``acquisition.py``): Sampling strategies
- **Likelihood Interfaces** (``likelihood.py``): Model evaluation
- **Utilities** (``utils/``): Supporting functionality

Data Flow
~~~~~~~~~

1. **Initialization**: Generate initial sample points
2. **Evaluation**: Compute log-likelihood via likelihood interface
3. **GP Fitting**: Train GP on accumulated data
4. **Acquisition**: Find next evaluation point
5. **Iteration**: Repeat until convergence

Adding New Features
-------------------

New Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

To add a new acquisition function:

.. code-block:: python

   # In acquisition.py
   class NewAcquisition(BaseAcquisition):
       def __init__(self, gp, **kwargs):
           super().__init__(gp, **kwargs)
           # Initialize acquisition-specific parameters
       
       def __call__(self, x):
           """Compute acquisition value at point x"""
           # Implementation here
           return acquisition_value
       
       def grad(self, x):
           """Compute gradient (optional for optimization)"""
           # Implementation here
           return gradient

Register in ``bo.py``:

.. code-block:: python

   _acq_funcs = {
       "wipv": WIPV, 
       "ei": EI, 
       "logei": LogEI,
       "new_acq": NewAcquisition  # Add here
   }

New GP Models
~~~~~~~~~~~~~

To add a new GP variant:

.. code-block:: python

   # In gp.py
   class NewGP(GP):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           # Initialize GP-specific components
       
       def fit(self, X, y):
           """Fit GP to data"""
           # Implementation here
       
       def predict(self, X):
           """Make predictions"""
           # Return mean, variance
           return mean, variance

Register in ``bo.py``:

.. code-block:: python

   def _get_gp(gp_type, **kwargs):
       if gp_type == "dslp":
           return DSLP_GP(**kwargs)
       elif gp_type == "saas":
           return SAAS_GP(**kwargs)
       elif gp_type == "new_gp":
           return NewGP(**kwargs)  # Add here

New Likelihood Interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~

To support a new likelihood interface:

.. code-block:: python

   # In likelihood.py
   class NewLikelihood(Likelihood):
       def __init__(self, config, **kwargs):
           super().__init__(**kwargs)
           # Initialize interface
       
       def log_likelihood(self, theta):
           """Evaluate log-likelihood"""
           # Implementation here
           return loglike

JAX Integration
---------------

BOBE leverages JAX for performance. Key principles:

Pure Functions
~~~~~~~~~~~~~~

Write functions that are JAX-compatible:

.. code-block:: python

   import jax.numpy as jnp
   from jax import jit, grad, vmap
   
   @jit  # JIT compile for speed
   def my_function(x):
       # Use jnp instead of np
       return jnp.sum(x**2)
   
   # Vectorize over batch dimension
   batched_function = vmap(my_function)

Random Number Generation
~~~~~~~~~~~~~~~~~~~~~~~~

Use JAX's stateful random number generation:

.. code-block:: python

   from jax import random
   
   # Get key from seed utilities
   key = get_jax_key()
   
   # Split key for multiple random operations
   key, subkey = random.split(key)
   samples = random.normal(subkey, shape=(100,))

Gradient Computation
~~~~~~~~~~~~~~~~~~~~

JAX provides automatic differentiation:

.. code-block:: python

   from jax import grad, jacfwd, jacrev
   
   # Gradient of scalar function
   grad_fn = grad(my_scalar_function)
   
   # Jacobian
   jac_fn = jacfwd(my_vector_function)  # Forward mode
   jac_fn = jacrev(my_vector_function)  # Reverse mode

Performance Optimization
------------------------

Memory Management
~~~~~~~~~~~~~~~~~

- Use ``max_gp_size`` to limit memory usage
- Implement data subset selection for large datasets
- Consider using ``jax.device_put`` for GPU memory management

Computational Efficiency
~~~~~~~~~~~~~~~~~~~~~~~~

- JIT compile critical functions with ``@jit``
- Use ``vmap`` for batch operations
- Profile code to identify bottlenecks
- Consider using ``pmap`` for multi-device parallelization

GPU/TPU Support
~~~~~~~~~~~~~~~

BOBE automatically uses available accelerators:

.. code-block:: python

   import jax
   print(f"Available devices: {jax.devices()}")
   print(f"Default backend: {jax.default_backend()}")

Testing Guidelines
------------------

Unit Tests
~~~~~~~~~~

Write tests for individual components:

.. code-block:: python

   # tests/test_gp.py
   import pytest
   from jaxbo.gp import DSLP_GP
   
   def test_gp_fitting():
       gp = DSLP_GP(input_dim=2)
       X = jnp.array([[0, 0], [1, 1]])
       y = jnp.array([0.0, 1.0])
       
       gp.fit(X, y)
       mean, var = gp.predict(X)
       
       assert mean.shape == (2,)
       assert var.shape == (2,)

Integration Tests
~~~~~~~~~~~~~~~~~

Test complete workflows:

.. code-block:: python

   def test_bobe_integration():
       likelihood = ToyLikelihood()
       bobe = BOBE(likelihood, max_eval_budget=50)
       results = bobe.run()
       
       assert results.n_evaluations <= 50
       assert results.log_evidence is not None

Performance Tests
~~~~~~~~~~~~~~~~~

Monitor performance regressions:

.. code-block:: python

   def test_performance():
       # Time critical operations
       import time
       start = time.time()
       # ... operation ...
       duration = time.time() - start
       assert duration < threshold

Documentation Standards
-----------------------

Docstring Format
~~~~~~~~~~~~~~~~

Use NumPy-style docstrings:

.. code-block:: python

   def my_function(x, y=None):
       """
       Brief description.
       
       Longer description explaining the function's purpose,
       algorithm, and any important details.
       
       Parameters
       ----------
       x : array_like
           Description of parameter x.
       y : float, optional
           Description of parameter y. Default is None.
       
       Returns
       -------
       result : ndarray
           Description of return value.
       
       Raises
       ------
       ValueError
           When invalid input is provided.
       
       Examples
       --------
       >>> result = my_function([1, 2, 3])
       >>> print(result)
       6
       """

Type Hints
~~~~~~~~~~

Use type hints for better code clarity:

.. code-block:: python

   from typing import Optional, Union, Tuple
   import jax.numpy as jnp
   
   def my_function(
       x: jnp.ndarray, 
       y: Optional[float] = None
   ) -> Tuple[jnp.ndarray, float]:
       pass

Release Process
---------------

Version Numbering
~~~~~~~~~~~~~~~~~

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Release Checklist
~~~~~~~~~~~~~~~~~

1. Update version in ``pyproject.toml``
2. Update ``CHANGELOG.md``
3. Run full test suite
4. Build and test documentation
5. Create release tag
6. Update documentation deployment

Debugging Tips
--------------

Common Issues
~~~~~~~~~~~~~

- **JAX array device mismatch**: Use ``jax.device_put()``
- **Gradient computation fails**: Check for non-differentiable operations
- **Memory errors**: Reduce batch sizes or use gradient checkpointing
- **NaN values**: Add numerical stability checks

Development Tools
~~~~~~~~~~~~~~~~~

Useful tools for debugging:

.. code-block:: python

   # JAX debugging
   from jax.config import config
   config.update("jax_debug_nans", True)
   config.update("jax_debug_infs", True)
   
   # Print intermediate values
   from jax import debug
   debug.print("Value: {}", x)
