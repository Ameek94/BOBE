Quick Start
=================

For users in a hurry, here's a minimal example using a test function:

For detailed examples, see:

- :doc:`detailed_usage` - Himmelblau function example, visualisations and comparison with dynesty
- :doc:`cosmology` - LCDM model with Planck likelihood through the Cobaya interface


Basic Example
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from BOBE import BOBE

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
      likelihood_name='quickstart_example', # name for output files
      save_dir='./results', # directory to save results

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
   print(f"Log Evidence mean: {results['logz']['mean']}")
   samples = results['samples'] # dictionary containing keys 'x', 'logl', 'weights'


**Expected Output:**

The code will print progress information and converge to a log-evidence estimate. 
The banana function has a complex posterior shape, demonstrating BOBE's ability 
to handle non-trivial likelihood surfaces.


Cosmology Example
~~~~~~~~~~~~~~~~~

For Cobaya cosmological likelihoods, simply pass the YAML file path instead of specifying parameter names and bounds:

.. code-block:: python

   from BOBE import BOBE
   
   # Initialize BOBE with Cobaya YAML file - CobayaLikelihood created internally
   sampler = BOBE(
       loglikelihood='path/to/cobaya_input.yaml',
       likelihood_name='CobayaLikelihood',
       n_sobol_init=4,
       n_cobaya_init=4,  # We can also specify reference dists in the Cobaya yaml file to generate additional initial points
       likelihood_name='quickstart_cobaya_example', # name for output files
       save_dir='./results',
      use_clf=True
   )
   
   # Run with optimization settings
   results = sampler.run(
       min_evals=10,
       max_evals=1000, # adjust according to your evaluation budget
       batch_size=5,
       fit_n_points=10,
       ns_n_points=10,
       logz_threshold=0.5,
   )
   
   # Access the evidence and posterior samples
   print(f"Log Evidence mean: {results['logz']['mean']}")
   samples = results['samples'] # dictionary containing keys 'x', 'logl', 'weights'

Understanding the Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's what the parameters in the examples above do:

**Initialization parameters** (passed to ``BOBE()``):

- ``loglikelihood``: Your likelihood function or Cobaya YAML file path
- ``param_list``: List of parameter names (required for callable functions)
- ``param_bounds``: Parameter bounds as array of shape (2, ndim) (required for callable functions)
- ``likelihood_name``: Name for output files
- ``n_sobol_init``: Number of initial space-filling Sobol points
- ``n_cobaya_init``: Number of initial points from Cobaya reference distribution (only for Cobaya likelihoods)
- ``use_clf``: Enable classifier to filter low-likelihood regions (recommended for cosmological examples where likelihood can return -inf values)
- ``save_dir``: Directory for saving results

**Execution parameters** (passed to ``run()``):

- ``min_evals``: Minimum likelihood evaluations before checking convergence
- ``max_evals``: Maximum likelihood evaluation budget
- ``batch_size``: Number of points to acquire per acquisition function iteration
- ``fit_n_points``: Refit GP hyperparameters after adding this many new points
- ``ns_n_points``: Run nested sampling on GP after adding this many new points
- ``logz_threshold``: Convergence threshold for the uncertainty on the log of the Bayesian evidence

For a complete list of parameters and advanced options, see the :doc:`../api/index`.
