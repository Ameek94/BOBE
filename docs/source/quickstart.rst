Quick Start Guide
=================

This guide will help you get up and running with JaxBo quickly.

Basic Concepts
--------------

JaxBo implements **Bayesian Optimization for Bayesian Evidence (BOBE)**, which uses 
Gaussian Processes to efficiently estimate the Bayesian evidence (marginal likelihood) 
of cosmological models.

Key Components:

- **BOBE**: The main optimization class
- **Likelihood**: Interface to your cosmological model
- **Gaussian Process**: Models the log-evidence surface
- **Acquisition Function**: Determines where to sample next

Simple Example
--------------

Here's a minimal example using a toy likelihood:

.. code-block:: python

   import numpy as np
   from jaxbo import BOBE
   from jaxbo.likelihood import BaseLikelihood
   
   # Define a simple 2D likelihood
   class ToyLikelihood(BaseLikelihood):
       def __init__(self):
           # Define parameter bounds
           self.bounds = np.array([[-2, 2], [-2, 2]])
           self.param_names = ['x', 'y']
       
       def log_likelihood(self, theta):
           x, y = theta
           # Simple bivariate normal
           return -0.5 * (x**2 + y**2)
   
   # Create and run BOBE
   likelihood = ToyLikelihood()
   bobe = BOBE(
       loglikelihood=likelihood,
       max_eval_budget=100,
       acquisition_func="wipv"
   )
   
   results = bobe.run()
   print(f"Log evidence estimate: {results.log_evidence:.3f}")

Cosmology Example with Cobaya
------------------------------

For real cosmological applications, JaxBo integrates with Cobaya:

.. code-block:: python

   from jaxbo import BOBE
   from jaxbo.likelihood import CobayaLikelihood
   
   # Cobaya model configuration
   info = {
       'params': {
           'omega_b': {'prior': {'min': 0.02, 'max': 0.025}},
           'omega_cdm': {'prior': {'min': 0.10, 'max': 0.15}},
           'H0': {'prior': {'min': 60, 'max': 80}},
           'tau_reio': 0.06,  # Fixed parameter
           'A_s': {'prior': {'min': 1.8e-9, 'max': 3.0e-9}},
           'n_s': {'prior': {'min': 0.9, 'max': 1.1}},
       },
       'likelihood': {
           'planck_2018_lowl.TT': None,
           'planck_2018_lowl.EE': None,
       },
       'theory': {
           'camb': {'extra_args': {'num_massive_neutrinos': 1}}
       }
   }
   
   # Create likelihood
   likelihood = CobayaLikelihood(info)
   
   # Run BOBE
   bobe = BOBE(
       loglikelihood=likelihood,
       max_eval_budget=1000,
       acquisition_func="wipv",
       gp_type="saas",  # Use sparse GP for high dimensions
       verbose=True
   )
   
   results = bobe.run()

Configuration Options
---------------------

The BOBE class accepts many configuration options:

Core Parameters
~~~~~~~~~~~~~~~

.. code-block:: python

   bobe = BOBE(
       loglikelihood=likelihood,
       
       # Budget control
       max_eval_budget=1500,      # Maximum function evaluations
       min_evals=200,             # Minimum evaluations before stopping
       
       # Initialization
       n_sobol_init=32,           # Initial Sobol sequence points
       n_cobaya_init=4,           # Initial Cobaya samples (if using Cobaya)
       
       # Gaussian Process
       gp_type="dslp",            # "dslp", "saas", or "standard"
       max_gp_size=1200,          # Maximum GP training set size
       
       # Acquisition function
       acquisition_func="wipv",    # "wipv", "ei", or "logei"
       
       # Optimization
       lr_gp=0.01,               # GP learning rate
       num_gp_epochs=500,        # GP training epochs
       
       # Convergence
       convergence_check=True,    # Enable convergence checking
       patience=50,              # Patience for early stopping
       
       # Output
       verbose=True,             # Enable verbose output
       output_dir="./results/",  # Output directory
   )

GP Model Types
~~~~~~~~~~~~~~

- **"dslp"**: Deep Sigmoidal Location Process - Good for smooth functions
- **"saas"**: Sparse Axis-Aligned Subspace - Better for high-dimensional problems  
- **"standard"**: Standard GP - Simple but can be slow for large datasets

Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~

- **"wipv"**: Weighted Integrated Posterior Variance - Balances exploration/exploitation
- **"ei"**: Expected Improvement - Classic acquisition function
- **"logei"**: Log Expected Improvement - More stable for small improvements

Working with Results
--------------------

The ``run()`` method returns a ``BOBEResults`` object:

.. code-block:: python

   results = bobe.run()
   
   # Access key results
   print(f"Log evidence: {results.log_evidence:.3f} ± {results.log_evidence_error:.3f}")
   print(f"Function evaluations: {results.n_evaluations}")
   print(f"Runtime: {results.total_time:.1f} seconds")
   
   # Access sample data
   samples = results.get_samples()  # Parameter samples
   log_likes = results.get_log_likelihoods()  # Log-likelihood values
   
   # Generate plots
   results.plot_convergence()
   results.plot_corner()
   results.plot_acquisition_evolution()

Parallel Computing
------------------

For large problems, use MPI parallelization:

.. code-block:: python

   from jaxbo.utils.pool import MPI_Pool
   
   # In your script
   if __name__ == "__main__":
       with MPI_Pool() as pool:
           bobe = BOBE(
               loglikelihood=likelihood,
               max_eval_budget=2000,
               pool=pool  # Pass the MPI pool
           )
           results = bobe.run()

Run with MPI:

.. code-block:: bash

   mpirun -n 4 python your_script.py

Next Steps
----------

- Read the :doc:`tutorials/index` for detailed examples
- Explore the :doc:`examples/index` for real-world applications  
- Check the :doc:`api/core` for detailed API documentation
- Learn about advanced features in the user guide
