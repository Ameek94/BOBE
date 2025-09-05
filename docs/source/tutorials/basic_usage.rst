Basic Usage Tutorial
====================

This tutorial covers the fundamental concepts and usage patterns of JaxBo.

Understanding Bayesian Optimization for Evidence
-------------------------------------------------

JaxBo implements Bayesian Optimization for Bayesian Evidence (BOBE), which aims to 
efficiently estimate the marginal likelihood (evidence) of a model:

.. math::

   Z = \int p(D|\theta) p(\theta) d\theta

Where:
- :math:`D` is the observed data
- :math:`\theta` represents the model parameters  
- :math:`p(D|\theta)` is the likelihood
- :math:`p(\theta)` is the prior

The challenge is that this integral is typically intractable, especially in 
high-dimensional parameter spaces common in cosmology.

The BOBE Approach
~~~~~~~~~~~~~~~~~

BOBE uses a Gaussian Process (GP) to model the log-evidence surface and an 
acquisition function to decide where to evaluate next. The process is:

1. **Initialize**: Start with a few random evaluations
2. **Model**: Fit a GP to the log-evidence evaluations  
3. **Acquire**: Use an acquisition function to find the next evaluation point
4. **Evaluate**: Compute the log-evidence at the new point
5. **Repeat**: Until convergence or budget exhausted

Setting Up Your First BOBE Run
-------------------------------

Step 1: Define Your Likelihood
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, you need to define your likelihood function. JaxBo provides a base class:

.. code-block:: python

   import numpy as np
   from jaxbo.likelihood import BaseLikelihood
   
   class MyLikelihood(BaseLikelihood):
       def __init__(self):
           # Define parameter bounds and names
           self.bounds = np.array([
               [0.02, 0.025],    # omega_b
               [0.10, 0.15],     # omega_cdm  
               [60, 80],         # H0
           ])
           self.param_names = ['omega_b', 'omega_cdm', 'H0']
       
       def log_likelihood(self, theta):
           """Compute log-likelihood for parameter vector theta"""
           omega_b, omega_cdm, H0 = theta
           
           # Your likelihood computation here
           # This is a toy example - replace with real physics!
           loglike = -0.5 * np.sum((theta - 0.5)**2)
           
           return loglike

Step 2: Create and Configure BOBE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jaxbo import BOBE
   
   # Create likelihood instance
   likelihood = MyLikelihood()
   
   # Configure BOBE
   bobe = BOBE(
       loglikelihood=likelihood,
       
       # Budget settings
       max_eval_budget=500,        # Maximum function evaluations
       min_evals=100,              # Minimum before checking convergence
       
       # Initialization
       n_sobol_init=20,            # Initial space-filling design
       
       # GP configuration  
       gp_type="dslp",             # GP variant to use
       max_gp_size=300,            # Maximum GP training set size
       
       # Acquisition function
       acquisition_func="wipv",     # Acquisition strategy
       
       # Optimization
       lr_gp=0.01,                 # GP learning rate
       num_gp_epochs=200,          # GP training epochs
       
       # Output
       verbose=True,               # Show progress
       output_dir="./my_results/"  # Save results here
   )

Step 3: Run the Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run BOBE
   results = bobe.run()
   
   # Print summary
   print(f"Log evidence: {results.log_evidence:.3f}")
   print(f"Uncertainty: ±{results.log_evidence_error:.3f}")
   print(f"Evaluations used: {results.n_evaluations}")

Understanding the Results
-------------------------

The ``results`` object contains comprehensive information about the optimization:

Key Attributes
~~~~~~~~~~~~~~

.. code-block:: python

   # Evidence estimate
   log_Z = results.log_evidence
   log_Z_err = results.log_evidence_error
   
   # Optimization info
   n_evals = results.n_evaluations
   runtime = results.total_time
   converged = results.converged
   
   # Sample data
   params = results.get_samples()          # Parameter samples
   log_likes = results.get_log_likelihoods()  # Log-likelihood values
   log_evidence_est = results.get_log_evidence_estimates()  # Evidence evolution

Visualization
~~~~~~~~~~~~~

JaxBo provides built-in plotting functions:

.. code-block:: python

   # Plot convergence
   results.plot_convergence()
   
   # Corner plot of samples
   results.plot_corner()
   
   # Acquisition function evolution
   results.plot_acquisition_evolution()
   
   # GP predictions (for 1D/2D problems)
   results.plot_gp_surface()

Saving and Loading Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save results
   results.save("my_bobe_run.pkl")
   
   # Load results later
   from jaxbo.utils.results import BOBEResults
   loaded_results = BOBEResults.load("my_bobe_run.pkl")

Configuration Best Practices
-----------------------------

Choosing GP Type
~~~~~~~~~~~~~~~~

- **"dslp"**: Good for smooth, well-behaved functions. Works well for most cosmology problems.
- **"saas"**: Better for high-dimensional problems (>10 parameters) with sparse structure.
- **"standard"**: Simple GP, good for small problems but doesn't scale well.

Setting Budget
~~~~~~~~~~~~~~

The evaluation budget depends on your problem:

- **Simple problems (2-5 dimensions)**: 100-500 evaluations
- **Moderate problems (5-10 dimensions)**: 500-1500 evaluations  
- **Complex problems (>10 dimensions)**: 1000-3000+ evaluations

Rule of thumb: Start with 50-100 evaluations per dimension.

Acquisition Function Choice
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **"wipv"**: Generally recommended. Balances exploration and exploitation well.
- **"ei"**: Classic choice, works well for smooth functions.
- **"logei"**: More stable version of EI, good when improvements are small.

Convergence Monitoring
~~~~~~~~~~~~~~~~~~~~~~

Enable convergence checking to stop early when the evidence estimate stabilizes:

.. code-block:: python

   bobe = BOBE(
       # ... other parameters ...
       convergence_check=True,
       patience=50,  # Stop if no improvement for 50 evaluations
       rel_tol=0.01,  # Stop if relative change < 1%
   )

Common Issues and Solutions
---------------------------

GP Training Fails
~~~~~~~~~~~~~~~~~~

If GP training fails or gives poor results:

- Reduce ``lr_gp`` (try 0.001 or 0.005)
- Increase ``num_gp_epochs``
- Check if your likelihood function has numerical issues
- Try a different GP type

Slow Convergence
~~~~~~~~~~~~~~~~

If optimization is slow to converge:

- Increase initial samples (``n_sobol_init``)
- Try a different acquisition function
- Check if your problem has multiple modes
- Consider using constraints if parts of parameter space are invalid

Memory Issues
~~~~~~~~~~~~~

For large problems:

- Reduce ``max_gp_size``
- Use "saas" GP type for better scaling
- Enable ``gp_subset_selection`` for automatic data selection

Next Steps
----------

- Learn about :doc:`gaussian_processes` for advanced GP configuration
- Explore :doc:`acquisition_functions` for different sampling strategies  
- See :doc:`cosmology_applications` for real-world examples
- Check :doc:`../examples/index` for complete working examples
