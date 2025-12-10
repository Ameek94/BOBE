In Depth Tutorial: Computing the Bayesian evidence for the Himmelblau Function
==================================================================================

This tutorial demonstrates the basic usage of BOBE through a classic optimization problem: the Himmelblau function. This example is based on the ``examples/Himmelblau.py`` file and shows how to set up and run a Bayesian optimization loop using the BOBE framework.

Overview
--------

The Himmelblau function is a well-known multi-modal test function in optimization that has four global minima, making it challenging for optimization algorithms. 
In this example, we test BOBE's performance on this function with the additional complication of embedding a 2D Himmelblau function in a 4D parameter space where only the first two dimensions are relevant.

The Himmelblau function (in loglikelihood form) is defined as:

.. math::

   \log \mathcal{L}(x_1, x_2, x_3, x_4) = -0.5 \cdot (a \cdot r_1 + r_2)

where 

.. math::

   r_1(x_1, x_2) = (x_1 + x_2^2 - 7)^2

   r_2(x_1, x_2) = (x_1^2 + x_2 - 11)^2

and

:math:`x_1, x_2, x_3, x_4 \in [-4, 4]` and :math:`a = 0.1` is a scaling factor. 
Note that :math:`x_3` and :math:`x_4` do not appear in the likelihood—this tests whether BOBE's can identify and handle the irrelevant dimensions

By the end of this tutorial, you will understand:

- How to define a likelihood function for BOBE
- How to set up parameter bounds and transformations
- How to configure and run BOBE
- How to analyze and visualize the results
- How to compare with reference methods such as Dynesty

1. Setup and Imports
--------------------

First, let's import all the necessary libraries and set up the environment.

.. code-block:: python

   # Core BOBE imports
   from BOBE import BOBE
   from BOBE.utils.plot import plot_final_samples, BOBESummaryPlotter
   import matplotlib.pyplot as plt
   import time
   import numpy as np
   import seaborn as sns # optional, for better plot styles
   from getdist import MCSamples, plots
   
   # External libraries for comparison
   from dynesty import DynamicNestedSampler

   # Enable LaTeX rendering for plots (optional)
   plt.rcParams['text.usetex'] = False  # Set to True if you have LaTeX installed
   plt.rcParams['font.family'] = 'serif'
   plt.style.use('default')

2. Define the Likelihood
-------------------------

Now let's define our target function—a 2D Himmelblau function embedded in 4D space—and wrap it in BOBE's Likelihood class.

.. code-block:: python

   # Scaling factor for the Himmelblau function
   afac = 0.1

   def loglike(X, slow=False):
       """
       The Himmelblau log-likelihood function embedded in 4D space.
       
       Only uses the first two dimensions (x1, x2), testing BOBE's ability
       to identify irrelevant dimensions via the SAAS prior.
       
       Original form: f(x,y) = (x² + y - 11)² + (x + y² - 7)²
       
       Parameters:
       -----------
       X : array-like, shape (4,)
           Input parameters [x1, x2, x3, x4] - only x1, x2 are used
       slow : bool
           If True, add artificial delay (useful for testing expensive likelihoods)
       
       Returns:
       --------
       float : Log-likelihood value
       """
       r1 = (X[0] + X[1]**2 - 7)**2
       r2 = (X[0]**2 + X[1] - 11)**2
       
       logpdf = -0.5 * (afac * r1 + r2)
       
       if slow:
           time.sleep(2)  # Artificial delay for testing
       
       return logpdf


   # Problem setup
   ndim = 4
   param_list = ['x1', 'x2', 'x3', 'x4']
   param_labels = ['x_1', 'x_2', 'x_3', 'x_4']
   param_bounds = np.array([[-4, 4], [-4, 4], [-4, 4], [-4, 4]]).T  # Shape: (2, ndim)

   print(f"Problem dimension: {ndim}")
   print(f"Parameter names: {param_list}")
   print(f"Parameter bounds:\n{param_bounds}")
   print(f"Scaling factor (afac): {afac}")

3. Configure and Run BOBE
--------------------------

Now let's set up the BOBE configuration and run the optimization. We'll use the SAAS (Sparsity-Aware Adaptive Shrinkage) lengthscale prior, which is particularly effective for high-dimensional problems.

.. code-block:: python

   # Configuration
   likelihood_name = "Himmelblau_test"

   print("Starting BOBE optimization...")
   print(f"Likelihood name: {likelihood_name}")

   start_time = time.time()

   # Initialize BOBE with setup parameters
   sampler = BOBE(
       likelihood=likelihood,
       param_list=param_list,
       param_bounds=param_bounds,
       param_labels=param_labels,
       verbosity='INFO',          # Set verbosity level
              
       # Initialization
       n_sobol_init=8,           # Initial Sobol sequence points
       
       # Random seed
       seed=42,
   )
   
   # Run optimization with convergence and run settings
   results = sampler.run(
       
       # Budget control
       min_evals=25,             # Minimum evaluations before starting to check convergence
       max_evals=250,            # Maximum function evaluations
       max_gp_size=250,          # Maximum GP training set size
       
       # GP and optimization settings
       fit_n_points=2,           # Refit GP every 2 iterations
       batch_size=2,             # Batch size for WIPV acquisition
       ns_n_points=5,            # Run nested sampling every 5 iterations

       # MCMC settings for WIPV (uses NUTS by default)
       num_hmc_warmup=256,       # NUTS warmup steps
       num_hmc_samples=1024,     # NUTS samples
       mc_points_size=256,       # Size of MC sample for WIPV
       
       # Convergence
       logz_threshold=0.01,      # Evidence convergence threshold
   )

   end_time = time.time()
   total_time = end_time - start_time

   print(f"\nBOBE optimization completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

4. Analyze the Results
----------------------

Let's examine what BOBE found and analyze the optimization process.

.. code-block:: python

   print("\n" + "="*60)
   print("RESULTS ANALYSIS")
   print("="*60)

   # Extract components
   gp = results['gp']
   logz_dict = results.get('logz', {})
   results_manager = results['results_manager']
   timing_data = results.get('timing', {})
       
   # Basic statistics
   print(f"Final GP size: {gp.train_x.shape[0]}")
   print(f"Converged: {results_manager.converged}")
   print(f"Termination reason: {results_manager.termination_reason}")

   logz_std = (logz_dict['upper'] - logz_dict['lower']) / 2
   print(f"Final LogZ estimate: {logz_dict['mean']:.4f} ± {logz_std:.4f}")

   # Find best point
   best_idx = np.argmax(gp.train_y)
   best_params = gp.train_x[best_idx]
   best_loglike = gp.train_y[best_idx]

   # Transform back to parameter space
   param_ranges = param_bounds[1] - param_bounds[0]
   best_params_scaled = best_params * param_ranges + param_bounds[0]
       
   print(f"\nBest point found:")
   for i, (name, val) in enumerate(zip(param_list, best_params_scaled)):
       print(f"  {name}: {val:.4f}")
   print(f"  Log-likelihood: {best_loglike.item():.4f}")

5. Compare with Dynesty (Reference Method)
-------------------------------------------

To validate our results, let's compare BOBE's evidence estimate with Dynesty, a popular nested sampling package.

.. code-block:: python

   print("\n" + "="*60)
   print("RUNNING DYNESTY FOR COMPARISON")
   print("="*60)

   # Define the proper prior transform function for Dynesty
   def dynesty_prior_transform(u):
       """Map [0,1]^4 to [-4,4]^4"""
       x = np.array(u)  
       return 8*x - 4

   # Run Dynesty (disable slow mode for fair timing)
   dns_sampler = DynamicNestedSampler(
       loglike, 
       dynesty_prior_transform, 
       ndim=ndim,
       logl_kwargs={'slow': False}
   )

   dynesty_start = time.time()
   dns_sampler.run_nested(print_progress=True, dlogz_init=0.01)
   dynesty_time = time.time() - dynesty_start
   
   dns_results = dns_sampler.results

   # Extract Dynesty results
   dynesty_logz = dns_results['logz'][-1]
   dynesty_logz_err = dns_results['logzerr'][-1]
   dynesty_samples = dns_results.samples_equal()

   print(f"\nDynesty results:")
   print(f"LogZ = {dynesty_logz:.4f} ± {dynesty_logz_err:.4f}")
   print(f"Runtime: {dynesty_time:.2f} seconds")

   bobe_logz = logz_dict['mean']
   print(f"\nComparison:")
   print(f"BOBE LogZ:    {bobe_logz:.4f} ± {logz_std:.4f}")
   print(f"Dynesty LogZ: {dynesty_logz:.4f} ± {dynesty_logz_err:.4f}")
   print(f"Difference:   {abs(bobe_logz - dynesty_logz):.4f}")
   print(f"Agreement:    {abs(bobe_logz - dynesty_logz) / dynesty_logz_err:.2f}σ")

Efficiency Comparison
~~~~~~~~~~~~~~~~~~~~~

Now let's compare the number of likelihood evaluations required. BOBE is much more efficient because it uses a GP surrogate instead of evaluating the true likelihood repeatedly.

.. code-block:: python

   # Compare number of likelihood evaluations
   bobe_evals = gp.train_x.shape[0]
   dynesty_evals = np.sum(dns_results['ncall'])
   
   print(f"\n" + "="*60)
   print("EFFICIENCY COMPARISON")
   print("="*60)
   print(f"BOBE evaluations:    {bobe_evals}")
   print(f"Dynesty evaluations: {dynesty_evals}")
   print(f"Speedup factor:      {dynesty_evals/bobe_evals:.1f}x")
   print(f"\nIf each likelihood took 2 seconds:")
   print(f"BOBE would take:    {bobe_evals * 2 / 60:.1f} minutes")
   print(f"Dynesty would take: {dynesty_evals * 2 / 60:.1f} minutes")
   print(f"Time saved:         {(dynesty_evals - bobe_evals) * 2 / 60:.1f} minutes")

6. Visualize Parameter Samples and Compare
-------------------------------------------

Let's create visualizations to compare the parameter samples from BOBE and Dynesty.

.. code-block:: python

   print("\n" + "="*60)
   print("CREATING VISUALIZATIONS")
   print("="*60)

   # Create GetDist samples for comparison
   dynesty_mcsamples = MCSamples(
       samples=dynesty_samples, 
       names=param_list, 
       labels=param_labels, 
       label='Dynesty'
   )

   # Use BOBE's built-in plotting function
   sample_array = results_manager.samples
   weights_array = results_manager.weights
   
   plot_final_samples(
       gp,
       {
           'x': sample_array, 
           'weights': weights_array, 
           'logl': results_manager.logl
       },
       param_list=likelihood.param_list,
       param_bounds=likelihood.param_bounds,
       param_labels=likelihood.param_labels,
       output_file=f"./results/{likelihood_name}",
       reference_samples=dynesty_mcsamples,
       reference_label='Dynesty',
       scatter_points=True
   )

   print(f"✓ Corner plot saved: ./results/{likelihood_name}.pdf")

7. Timing Analysis
------------------

Let's analyze how the computational time was spent across different phases.

.. code-block:: python

   print("\n" + "="*60)
   print("TIMING ANALYSIS")
   print("="*60)
       
   print(f"Total runtime: {timing_data['total_runtime']:.2f} seconds")
       
   print("\nPhase breakdown:")
   print("-" * 40)
   for phase in ['initialization', 'gp_fitting', 'acquisition', 'nested_sampling']:
       time_spent = timing_data['phase_times'].get(phase, 0)
       percentage = timing_data['percentages'].get(phase, 0)
       if time_spent > 0:
           print(f"{phase:25s}: {time_spent:8.2f}s ({percentage:5.1f}%)")
       
   # Calculate overhead
   total_measured = sum(timing_data['phase_times'].values())
   overhead = timing_data['total_runtime'] - total_measured
   overhead_pct = (overhead / timing_data['total_runtime']) * 100
   
   print(f"{'overhead/unmeasured':25s}: {overhead:8.2f}s ({overhead_pct:5.1f}%)")
       
   # Find the dominant phase
   max_phase = max(timing_data['phase_times'].items(), key=lambda x: x[1])
   print(f"\nDominant phase: {max_phase[0]} ({timing_data['percentages'][max_phase[0]]:.1f}%)")

8. Generate Summary Dashboard
-----------------------------

BOBE provides comprehensive plotting utilities. Let's create a summary dashboard.

.. code-block:: python

   print("\n" + "="*60)
   print("GENERATING SUMMARY DASHBOARD")
   print("="*60)

   # Initialize plotter
   plotter = BOBESummaryPlotter(results_manager)
   
   # Get data for plotting
   gp_data = results_manager.get_gp_data()
   best_loglike_data = results_manager.get_best_loglike_data()
   acquisition_data = results_manager.get_acquisition_data()
   
   # Create summary dashboard with timing data
   fig_dashboard = plotter.create_summary_dashboard(
       gp_data=gp_data,
       acquisition_data=acquisition_data,
       best_loglike_data=best_loglike_data,
       timing_data=timing_data,
       save_path=f"./results/{likelihood_name}_dashboard.pdf"
   )
   
   plt.show()
   
   print(f"\n✓ Summary dashboard: ./results/{likelihood_name}_dashboard.pdf")
   print(f"✓ Results saved to: ./results/{likelihood_name}_results.pkl")

Summary
-------

This tutorial demonstrated the complete workflow for using BOBE on a problem with irrelevant dimensions:

1. **Problem Setup**: Defined a 2D Himmelblau function embedded in 4D space
2. **BOBE Configuration**: Set up the optimization with SAAS lengthscale prior for automatic relevance determination
3. **Execution**: Ran the optimization with WIPV acquisition function
4. **Analysis**: Examined results, convergence, and best points found
5. **Validation**: Compared against Dynesty to verify accuracy
6. **Efficiency Analysis**: Demonstrated BOBE's efficiency in likelihood evaluations (typically 10-50x fewer evaluations)
7. **Visualization**: Created comprehensive plots for analysis

Key takeaways:

- **BOBE is efficient**: Uses significantly fewer likelihood evaluations than traditional nested sampling methods
- **Evidence estimation**: Provides reliable Bayesian evidence estimates with uncertainty quantification
- **Multi-modal capability**: Successfully handles functions with multiple minima through the GP surrogate
- **Automatic relevance determination**: The SAAS prior automatically identifies and down-weights irrelevant dimensions (x3, x4 in this example)
- **Scalability**: BOBE scales to moderate dimensional problems (tested up to ~15D) especially when using SAAS prior
- **Comprehensive output**: Generates detailed timing, convergence, and visualization data

This example demonstrates BOBE's robustness to irrelevant dimensions—a common challenge in real-world problems where not all parameters significantly affect the likelihood. The SAAS prior learns appropriate lengthscales for each dimension, effectively "turning off" dimensions that don't contribute to the likelihood. For real-world applications with expensive likelihoods (>1 second per evaluation), the efficiency gains become even more dramatic. Simply replace the toy likelihood with your actual physics/cosmology model.

Next Steps
~~~~~~~~~~

- Try the **Banana function** example for a simpler 2D case
- Explore the **Cosmology** example for a real-world application (ΛCDM with Planck+DESI data)
- Read the **User Guide** to understand convergence criteria and hyperparameter tuning
- Check the **API Reference** for advanced configuration options

