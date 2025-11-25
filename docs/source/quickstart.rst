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

Example 2: Cosmological Likelihood
-----------------------------------

Now let's look at a more realistic example using a cosmological likelihood through Cobaya.

.. note::
   This example requires the optional Cobaya dependency. Install with:
   ``pip install 'jaxbo[cobaya]'``

.. code-block:: python

   from jaxbo.run import run_bobe
   
   # Path to Cobaya input file defining the cosmological model
   cobaya_input_file = './cosmo_input/LCDM_Planck_DESI_Omk.yaml'
   
   # Run BOBE with cosmological likelihood
   results = run_bobe(
       likelihood=cobaya_input_file,  # Cobaya YAML file
       likelihood_kwargs={
           'confidence_for_unbounded': 0.9999995,
           'minus_inf': -1e5,
           'noise_std': 0.0,
           'name': 'Planck_DESI_LCDM',
       },
       verbosity='INFO',
       # Initial sampling
       n_cobaya_init=32,        # Points from Cobaya reference distribution
       n_sobol_init=64,         # Additional Sobol points
       
       # Budget
       min_evals=800,
       max_evals=2500,
       max_gp_size=1500,
       
       # Acquisition settings
       acq=['wipv'],
       convergence_n_iters=2,   # Require 2 consecutive convergence checks
       
       # Step settings
       fit_step=5,              # Fit GP every 5 evaluations
       wipv_batch_size=5,       # Evaluate 5 points per acquisition

