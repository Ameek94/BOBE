Quick Start Guide
=================

JaxBo estimates Bayesian evidence for expensive likelihood functions using Gaussian Process (GP) surrogates. The GP is trained using Bayesian optimisation with an acquisition function.

Simple Example
--------------

Here's a minimal example using a test function:

.. code-block:: python

   import numpy as np
   from jaxbo import BOBE
   
   # Define a simple 2D likelihood
   def my_likelihood(X):
       x, y = X[0], X[1]
       return -0.5 * (x**2 + y**2)
   
   # Initialize BOBE with setup parameters
   bobe = BOBE(
       loglikelihood=my_likelihood,
       param_bounds=np.array([[-3, 3], [-3, 3]]).T,
       likelihood_name='test',
       n_sobol_init=4,
       seed=42,
   )
   
   # Run optimization with convergence and run settings
   results = bobe.run(
       acqs='wipv',
       max_evals=100,
   )
   
   # Get results
   print(f"Log Evidence: {results['logz']['mean']:.2f}")

For detailed examples, see:

- :doc:`examples/banana` - 2D test function example
- :doc:`examples/cosmology` - Cosmological likelihood with Cobaya

Cosmology Example
~~~~~~~~~~~~~~~~~

For Cobaya cosmological likelihoods, simply pass the YAML file path:

.. code-block:: python

   from jaxbo import BOBE
   
   # Initialize BOBE with Cobaya YAML - CobayaLikelihood created internally
   bobe = BOBE(
       loglikelihood='planck_lcdm.yaml',
       likelihood_name='planck_lcdm',
       confidence_for_unbounded=0.9999995,
   )
   
   # Run with optimization settings
   results = bobe.run(
       acqs='wipv',
       max_evals=1000,
   )
   print(f"Log Evidence: {results['logz']['mean']:.2f}")

**Expected Output:**

The code will print progress information and converge to a log-evidence estimate. 
The banana function has a complex posterior shape, demonstrating JaxBO's ability 
to handle non-trivial likelihood surfaces.

Understanding the Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Key parameters are split between initialization (``__init__``) and execution (``run()``):

**Initialization parameters** (passed to ``BOBE()``):

- ``loglikelihood``: Your likelihood function or Cobaya YAML file path
- ``param_bounds``: Parameter bounds as array of shape (2, ndim)
- ``n_sobol_init``: Number of initial space-filling points (more for higher dimensions)
- ``n_cobaya_init``: Number of initial points from Cobaya reference distribution
- ``use_clf``: Enable classifier for high-dimensional or expensive likelihoods
- ``gp_kwargs``: GP configuration (kernel, priors, bounds)

**Execution parameters** (passed to ``run()``):

- ``acqs``: Acquisition function - 'wipv' (recommended), 'ei', or 'logei'
- ``min_evals``, ``max_evals``: Budget for likelihood evaluations
- ``mc_points_method``: 'NS' (nested sampling) or 'HMC' for GP posterior sampling
- ``fit_step``: How often to refit GP hyperparameters
- ``logz_threshold``: Convergence threshold for log-evidence
