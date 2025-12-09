Quick Start
=================

BOBE estimates Bayesian evidence for expensive likelihood functions using Gaussian Process (GP) surrogates. The GP is trained using Bayesian optimisation with an acquisition function.

Simple Example
--------------

Here's a minimal example using a test function:

.. code-block:: python

   # Define your likelihood function
   def my_likelihood(X):
      x, y = X[0], X[1]
      logpdf = -0.25 * (5 * (0.2 - x))**2 - (20 * (y/4 - x**4))**2 # Example: a likelihood with a curved degeneracy
      return logpdf 


   # Initialize BOBE with setup parameters
   sampler = BOBE(
      loglikelihood=my_likelihood,
      param_list=['x1', 'x2'], # list of parameter names
      param_bounds=np.array([[-1, 1], [-1, 2]]).T, # lower and upper bounds for parameters (2, ndim) shaped
      n_sobol_init=2, # number of initial Sobol samples to start the run from
      save_dir='./results',

   )

   # Run optimization with convergence and run settings
   results = sampler.run(
      min_evals=10, # do at least 20 evaluations
      max_evals=100, # max evaluation budget
      batch_size=2, # acquisition function batch size
      fit_n_points=4, # fit gp after every 4 likelihood evaluations
      ns_n_points=4, # run nested sampling after every 4 likelihood evaluations
      logz_threshold=0.1, # target logz uncertainty from GP
   )

   # Access the evidence and posterior samples
   print(f"Log Evidence: {results['logz']['mean']}")
   samples = results['samples'] # dictionary containing keys 'x', 'logl', 'weights'

For detailed examples, see:

- :doc:`examples/banana` - 2D test function example
- :doc:`examples/cosmology` - Cosmological likelihood with Cobaya

Cosmology Example
~~~~~~~~~~~~~~~~~

For Cobaya cosmological likelihoods, simply pass the YAML file path:

.. code-block:: python

   from BOBE import BOBE
   
   # Initialize BOBE with Cobaya YAML file - CobayaLikelihood created internally
   sampler = BOBE(
       loglikelihood='path/to/cobaya_input.yaml',
       likelihood_name='CobayaLikelihood',
       n_sobol_init=4,
       n_cobaya_init=4,
       save_dir='./results',
   )
   
   # Run with optimization settings
   results = sampler.run(
       min_evals=10,
       max_evals=1000,
       batch_size=2,
       fit_n_points=4,
       ns_n_points=4,
       logz_threshold=0.1,
   )
   
   print(f"Log Evidence: {results['logz']['mean']}")

**Expected Output:**

The code will print progress information and converge to a log-evidence estimate. 
The banana function has a complex posterior shape, demonstrating BOBE's ability 
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
- ``mc_points_method``: 'NUTS' (default), 'NS' (nested sampling), or 'uniform' for GP posterior sampling
- ``fit_step``: How often to refit GP hyperparameters
- ``logz_threshold``: Convergence threshold for log-evidence
