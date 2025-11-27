Quick Start Guide
=================

JaxBO estimates Bayesian evidence using Gaussian Process surrogates of expensive likelihoods.

Simple Example
--------------

Here's a minimal example using a test function:

.. code-block:: python

   import numpy as np
   from jaxbo.run import run_bobe
   
   # Define a simple 2D likelihood
   def my_likelihood(X):
       x, y = X[0], X[1]
       return -0.5 * (x**2 + y**2)
   
   # Run BOBE
   results = run_bobe(
       likelihood=my_likelihood,
       likelihood_kwargs={
           'param_list': ['x', 'y'],
           'param_bounds': np.array([[-3, 3], [-3, 3]]).T,
           'name': 'test',
       },
       max_evals=100,
       seed=42,
   )
   
   # Get results
   print(f"Log Evidence: {results['logz']['logz']:.2f}")

For detailed examples, see:

- :doc:`examples/banana` - 2D test function example
- :doc:`examples/cosmology` - Cosmological likelihood with Cobaya

**Expected Output:**

The code will print progress information and converge to a log-evidence estimate. 
The banana function has a complex posterior shape, demonstrating JaxBO's ability 
to handle non-trivial likelihood surfaces.

Understanding the Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Key parameters to tune:

- ``n_sobol_init``: Number of initial space-filling points (more for higher dimensions)
- ``min_evals``, ``max_evals``: Budget for likelihood evaluations
- ``mc_points_method``: 'NUTS' for good exploration, 'uniform' for simpler problems
- ``use_clf``: Enable classifier for high-dimensional or expensive likelihoods
- ``acq``: Acquisition function - 'wipv' (recommended), 'ei', or 'logei'
