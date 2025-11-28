JaxBo Basic Tutorial: Computing the Bayesian evidence for the Himmelblau Function
====================================================================================

This tutorial demonstrates the basic usage of JaxBo through a classic optimization problem: the Himmelblau function. This example is based on the ``examples/Himmelblau.py`` file and shows how to set up and run a Bayesian optimization using the BOBE (Bayesian Optimization for Bayesian Evidence) framework.

Overview
--------

The Himmelblau function is a well-known multi-modal test function in optimization that has four global minima, making it challenging for optimization algorithms. It's defined as:

.. math::

   f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2

For our Bayesian optimization, we'll use the log-likelihood form:

.. math::

   \log \mathcal{L}(x_1, x_2) = -0.5 \cdot (a \cdot (x_1 + x_2^2 - 7)^2 + (x_1^2 + x_2 - 11)^2)

where :math:`x_1, x_2 \in [-4, 4]` and :math:`a = 0.1` is a scaling factor.

The function has four global minima at approximately:

- :math:`(3.0, 2.0)`
- :math:`(-2.8, 3.1)` 
- :math:`(-3.8, -3.3)`
- :math:`(3.6, -1.8)`

By the end of this tutorial, you will understand:

- How to define a likelihood function for JaxBo
- How to set up parameter bounds and transformations
- How to configure and run the BOBE optimizer
- How to analyze and visualize the results
- How to compare with reference methods like Dynesty

1. Setup and Imports
--------------------

First, let's import all the necessary libraries and set up the environment.

.. code-block:: python

   # Core JaxBo imports
   from jaxbo import BOBE
   from jaxbo.utils.plot import plot_final_samples, BOBESummaryPlotter
   import matplotlib.pyplot as plt
   import time
   import numpy as np
   from getdist import MCSamples, plots, loadMCSamples
   # External libraries for comparison
   from dynesty import DynamicNestedSampler

   # Enable LaTeX rendering for plots
   plt.rcParams['text.usetex'] = False  # Set to True if you have LaTeX installed
   plt.rcParams['font.family'] = 'serif'
   plt.style.use('default')

2. Define the Likelihood
-------------------------

Now let's define our target function (the Himmelblau function) and the prior transformation.

.. code-block:: python

   # Scaling factor for the Himmelblau function
   afac = 0.1

   def loglike(X, slow=True):
       """
       The Himmelblau log-likelihood function.
       
       The Himmelblau function is a multi-modal function with four global minima.
       Original form: f(x,y) = (x² + y - 11)² + (x + y² - 7)²
       
       Parameters:
       -----------
       X : array-like
           Input parameters [x1, x2]
       slow : bool
           If True, add artificial delay (useful for testing)
       
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


   def prior_transform(x):
       """
       Transform unit cube samples to parameter space.
       This function maps [0,1]^2 to our parameter bounds [-4,4]^2.
       
       Parameters:
       -----------
       x : array-like
           Unit cube coordinates [0,1]^2
       
       Returns:
       --------
       array : Transformed parameters
       """
       # Both x1 and x2: [0,1] -> [-4,4]
       return 8*x - 4


   # Problem setup
   ndim = 2
   param_list = ['x1', 'x2']
   param_labels = ['x_1', 'x_2']
   param_bounds = np.array([[-4, 4], [-4, 4]]).T  # Shape: (2, ndim)

   print(f"Problem dimension: {ndim}")
   print(f"Parameter names: {param_list}")
   print(f"Parameter bounds:\n{param_bounds}")
   print(f"Scaling factor (afac): {afac}")

3. Configure and Run BOBE
--------------------------

Now let's set up the BOBE configuration and run the optimization. We'll use a smaller evaluation budget for this tutorial.

.. code-block:: python

   # Configuration
   likelihood_name = f'Himmelblau_tutorial'

   print("Starting BOBE optimization...")
   print(f"Likelihood name: {likelihood_name}")

   start_time = time.time()

   # Run BOBE optimization
   bobe = BOBE(
       loglikelihood=loglike,
       param_list=param_list,
       param_bounds=param_bounds,
       param_labels=param_labels,
       likelihood_name=likelihood_name,
       verbosity='INFO',
       
       # Initialization
       n_sobol_init=4,       # Initial Sobol sequence points
       
       # Budget control
       min_evals=20,         # Minimum evaluations before starting to check convergence
       max_evals=250,        # Maximum function evaluations
       max_gp_size=250,      # Maximum GP training set size
       
       # GP and optimization settings
       fit_step=2,           # Refit GP every 2 iterations
       wipv_batch_size=2,    # Batch size for WIPV acquisition
       ns_step=5,            # Run nested sampling every 5 iterations

       # MCMC settings for WIPV
       num_hmc_warmup=256,   # HMC warmup steps
       num_hmc_samples=1024, # HMC samples
       mc_points_size=256,   # Size of MC sample for WIPV
       
       # Convergence
       logz_threshold=0.001, # Evidence convergence threshold
       minus_inf=-1e5,
       
       # Reproducibility
       seed=42,
   )
   
   results = bobe.run(['wipv'])

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
   likelihood = results['likelihood']
   comprehensive_results = results['comprehensive']
   timing_data = comprehensive_results['timing']
       
   # Basic statistics
   print(f"Final GP size: {gp.train_x.shape[0]}")
   print(f"Converged: {comprehensive_results['converged']}")
   print(f"Termination reason: {comprehensive_results['termination_reason']}")

   logz_std = logz_dict['std']
   print(f"Final LogZ estimate: {logz_dict.get('mean', 'N/A'):.4f} ± {logz_std:.4f}")

   # Find best point
   best_idx = np.argmax(gp.train_y)
   best_params = gp.train_x[best_idx]
   best_loglike = gp.train_y[best_idx]

   # Transform back to parameter space
   param_ranges = param_bounds[1] - param_bounds[0]
   best_params_scaled = best_params * param_ranges + param_bounds[0]
       
   print(f"\nBest point found:")
   print(f"  Parameters: [{best_params_scaled[0]:.4f}, {best_params_scaled[1]:.4f}]")
   print(f"  Log-likelihood: {best_loglike.item():.4f}")

5. Compare with Dynesty (Reference Method)
-------------------------------------------

To validate our results, let's compare BOBE's evidence estimate with Dynesty, a popular nested sampling package.

.. code-block:: python

   print("Running Dynesty for comparison...")

   # Define the proper prior transform function for Dynesty
   def dynesty_prior_transform(u):
       x = np.array(u)  
       return 8*x - 4   # Transform [0,1]^2 to [-4,4]^2

   # Run Dynesty
   dns_sampler = DynamicNestedSampler(
       loglike, 
       dynesty_prior_transform, 
       ndim=ndim,
       logl_kwargs={'slow': False}
   )

   dns_sampler.run_nested(print_progress=True, dlogz_init=0.01)
   dns_results = dns_sampler.results

   # Extract Dynesty results
   dynesty_logz = dns_results['logz'][-1]
   dynesty_logz_err = dns_results['logzerr'][-1]
   dynesty_samples = dns_results.samples_equal()

   print(f"\nDynesty results:")
   print(f"LogZ = {dynesty_logz:.4f} ± {dynesty_logz_err:.4f}")

   bobe_logz = logz_dict.get('mean', np.nan)
   print(f"\nComparison:")
   print(f"BOBE LogZ:    {bobe_logz:.4f}")
   print(f"Dynesty LogZ: {dynesty_logz:.4f} ± {dynesty_logz_err:.4f}")
   print(f"Difference:   {abs(bobe_logz - dynesty_logz):.4f}")

Efficiency Comparison
~~~~~~~~~~~~~~~~~~~~~

Now let us compare the number of likelihood evaluations required to get these results. We will see that BOBE is much more efficient in terms of the number of true likelihood evaluations. If we had run dynesty with the 'slow' version of the likelihood (which took 2s/eval) the run would have taken far longer. This is the exact situation BOBE was designed for, slow likelihoods of low to moderate number of dimensions.

.. code-block:: python

   # Compare number of likelihood evaluations
   bobe_evals = gp.train_x.shape[0]
   dynesty_evals = np.sum(dns_results['ncall'])
   print(f"\nNumber of likelihood evaluations:")
   print(f"BOBE evaluations:    {bobe_evals}")
   print(f"Dynesty evaluations: {dynesty_evals}")
   print(f"BOBE used {int(dynesty_evals/bobe_evals)} times fewer evaluations than Dynesty.")

Visualize Parameter Samples and Compare
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's create visualizations to compare the parameter samples from BOBE and Dynesty.

.. code-block:: python

   # BOBE samples get saved in getdist compatible format
   gp_samples = loadMCSamples(f'./{likelihood_name}')
   dynesty_samples_getdist = MCSamples(samples=dynesty_samples, names=param_list, labels=param_labels, label='Dynesty Samples')
   g = plots.get_subplot_plotter()
   g.triangle_plot([gp_samples, dynesty_samples_getdist], filled=[True,False],contour_colors=['blue','red'], )
   ax = g.subplots[1,0]
   training_points = gp.train_x * (param_bounds[1] - param_bounds[0]) + param_bounds[0]
   ax.scatter(training_points[:,0], training_points[:,1], c='green', s=10, marker='.', zorder=5,alpha=0.5)

6. Timing Analysis
------------------

Let's analyze how the computational time was spent across different phases.

.. code-block:: python

   print("TIMING ANALYSIS")
       
   print(f"Total runtime: {timing_data['total_runtime']:.2f} seconds ({timing_data['total_runtime']/60:.2f} minutes)")
       
   print("\nPhase breakdown:")
   print("-" * 40)
   for phase, time_spent in timing_data['phase_times'].items():
       if time_spent > 0:
           percentage = timing_data['percentages'].get(phase, 0)
           print(f"{phase:25s}: {time_spent:8.2f}s ({percentage:5.1f}%)")
       
   # Find the dominant phase
   if any(t > 0 for t in timing_data['phase_times'].values()):
       max_phase = max(timing_data['phase_times'].items(), key=lambda x: x[1])
       print(f"\nDominant phase: {max_phase[0]} ({timing_data['percentages'][max_phase[0]]:.1f}%)")
       
   # Create timing visualization
   phases = []
   times = []
   for phase, time_spent in timing_data['phase_times'].items():
       if time_spent > 0:
           phases.append(phase.replace(' ', '\n'))  # Break long names
           times.append(time_spent)
       
   plt.figure(figsize=(8, 4))
   bars = plt.bar(phases, times, color='skyblue', alpha=0.7, edgecolor='navy')
   plt.xlabel('Optimization Phase')
   plt.ylabel('Time (seconds)')
   plt.title('BOBE Timing Breakdown')
   plt.xticks(rotation=45, ha='right')
           
   # Add value labels on bars
   for bar, time_val in zip(bars, times):
       plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{time_val:.1f}s', ha='center', va='bottom')
           
   plt.tight_layout()
   plt.show()

7. Generate Summary Dashboard
-----------------------------

BOBE provides comprehensive plotting utilities. Let's create a summary dashboard.

.. code-block:: python

   if results is not None:
       print("Creating summary plots...")
       
       # Use BOBE's built-in plotting function
       plot_final_samples(
           gp,
           {'x': sample_array, 'weights': weights_array, 'logl': samples.get('logl', [])},
           param_list=likelihood.param_list,
           param_bounds=likelihood.param_bounds,
           param_labels=likelihood.param_labels,
           output_file=likelihood_name,
           reference_samples=dynesty_mcsamples,
           reference_file=None,
           reference_label='Dynesty',
           scatter_points=True
       )
       
       # Create comprehensive dashboard
       plotter = BOBESummaryPlotter(results['results_manager'])
       
       # Get data for plotting
       gp_data = results['results_manager'].get_gp_data()
       best_loglike_data = results['results_manager'].get_best_loglike_data()
       acquisition_data = results['results_manager'].get_acquisition_data()
       
       # Create summary dashboard
       fig_dashboard = plotter.create_summary_dashboard(
           gp_data=gp_data,
           acquisition_data=acquisition_data,
           best_loglike_data=best_loglike_data,
           timing_data=timing_data,
           save_path=f"{likelihood_name}_dashboard.pdf"
       )
       
       plt.show()
       
       print(f"\nPlots saved:")
       print(f"✓ Parameter samples: {likelihood_name}_samples.pdf")
       print(f"✓ Summary dashboard: {likelihood_name}_dashboard.pdf")

Summary
-------

This tutorial demonstrated the complete workflow for using JaxBo's BOBE framework:

1. **Problem Setup**: Defined the Himmelblau function as a multi-modal test case
2. **BOBE Configuration**: Set up the optimization with appropriate parameters
3. **Execution**: Ran the optimization and monitored progress  
4. **Analysis**: Examined results, convergence, and best points found
5. **Validation**: Compared against Dynesty to verify accuracy
6. **Efficiency Analysis**: Demonstrated BOBE's efficiency in likelihood evaluations
7. **Visualization**: Created comprehensive plots for analysis

Key takeaways:

- **BOBE is efficient**: Uses significantly fewer likelihood evaluations than traditional methods
- **Evidence estimation**: Provides reliable Bayesian evidence estimates with uncertainty quantification
- **Multi-modal capability**: Successfully handles functions with multiple minima
- **Comprehensive output**: Generates detailed timing, convergence, and visualization data

The Himmelblau function serves as an excellent test case demonstrating BOBE's capabilities on a challenging multi-modal optimization problem. For real-world applications, you would replace the toy likelihood with your actual physics/cosmology model.
