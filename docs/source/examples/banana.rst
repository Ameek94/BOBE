Banana Function Example
=======================

The Rosenbrock "banana" function is a classic test problem for optimization algorithms. 
Its banana-shaped contours make it challenging for many methods, providing a good test 
of JaxBO's ability to handle non-trivial likelihood surfaces.

Problem Description
-------------------

The banana function is defined as:

.. math::

   \log \mathcal{L}(x_1, x_2) = -0.25 (5(0.2 - x_1))^2 - (20(x_2/4 - x_1^4))^2

This creates a narrow, curved valley that is difficult to explore efficiently.

Parameter Space
~~~~~~~~~~~~~~~

- :math:`x_1 \in [-1, 1]`
- :math:`x_2 \in [-1, 2]`

Complete Example Code
----------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from jaxbo.run import run_bobe
   from jaxbo.utils.log import get_logger
   from jaxbo.utils.plot import BOBESummaryPlotter
   from getdist import MCSamples
   
   def loglike_banana(X):
       """Rosenbrock banana function likelihood."""
       x, y = X[0], X[1]
       logpdf = -0.25 * (5 * (0.2 - x))**2 - (20 * (y/4 - x**4))**2
       return logpdf
   
   def main():
       # Problem setup
       param_list = ['x1', 'x2']
       param_labels = [r'x_1', r'x_2']
       param_bounds = np.array([[-1, 1], [-1, 2]]).T
       
       print("Starting BOBE run...")
       
       results = run_bobe(
           likelihood=loglike_banana,
           likelihood_kwargs={
               'param_list': param_list,
               'param_bounds': param_bounds,
               'param_labels': param_labels,
               'name': 'banana_test',
               'minus_inf': -1e5,
           },
           verbosity='INFO',
           use_gp_pool=True,
           n_sobol_init=8,
           min_evals=20,
           max_evals=100,
           max_gp_size=200,
           fit_step=1,
           wipv_batch_size=2,
           ns_step=5,
           optimizer='scipy',
           mc_points_method='NUTS',
           num_hmc_warmup=256,
           num_hmc_samples=4000,
           mc_points_size=128,
           thinning=4,
           num_chains=4,
           use_clf=False,  # Simple 2D problem doesn't need classifier
           minus_inf=-1e5,
           logz_threshold=1e-3,
           seed=42,
           save_dir='./results/',
           save=True,
           acq=['wipv'],
           ei_goal=1e-5,
           do_final_ns=True,
       )
       
       if results is not None:
           log = get_logger("main")
           
           # Extract results
           gp = results['gp']
           samples = results['samples']
           logz_dict = results.get('logz', {})
           likelihood = results['likelihood']
           
           log.info("\n" + "="*60)
           log.info("RUN COMPLETED")
           log.info("="*60)
           log.info(f"Log Evidence: {logz_dict.get('logz', 'N/A'):.4f}")
           log.info(f"Log Evidence Error: {logz_dict.get('logzerr', 0):.4f}")
           log.info(f"Number of evaluations: {gp.train_x.shape[0]}")
           
           # Create GetDist samples for plotting
           sample_array = samples['x']
           sample_weights = samples.get('weights', None)
           
           mcs = MCSamples(
               samples=sample_array,
               weights=sample_weights,
               names=param_list,
               labels=param_labels,
               ranges=dict(zip(param_list, param_bounds.T))
           )
           
           # Triangle plot
           import getdist.plots as gdplt
           g = gdplt.get_subplot_plotter()
           g.triangle_plot([mcs], filled=True)
           plt.savefig('./results/banana_triangle.pdf', bbox_inches='tight')
           log.info("Saved triangle plot to ./results/banana_triangle.pdf")
           
           # Summary plots
           plotter = BOBESummaryPlotter(results)
           plotter.plot_gp_training_evolution()
           plotter.plot_logz_evolution()
           plotter.plot_acquisition_evolution()
           
           log.info("Analysis complete!")
   
   if __name__ == '__main__':
       main()

Expected Results
----------------

Running this example should produce:

1. **Log Evidence**: Approximately -1.5 to -2.0 (depends on random seed)
2. **Number of Evaluations**: 100 (as specified by max_evals)
3. **Convergence**: Should converge within 60-80 evaluations

Visualization
~~~~~~~~~~~~~

The code generates several plots:

- **Triangle plot**: Shows the 2D posterior distribution in the banana shape
- **GP training evolution**: Shows how the GP improves over iterations
- **Log evidence evolution**: Shows convergence of the evidence estimate
- **Acquisition evolution**: Shows how the acquisition function selects points

Key Takeaways
-------------

1. **No Classifier Needed**: Simple 2D problems don't need a classifier
2. **Fast Convergence**: The WIPV acquisition function efficiently explores the banana valley
3. **Accurate Evidence**: JaxBO accurately estimates the evidence with ~100 evaluations

Parameter Tuning Tips
---------------------

For similar test functions:

- ``n_sobol_init``: 5-10 for 2D, scale with dimension
- ``mc_points_method='NUTS'``: Best for complex posteriors
- ``use_clf=False``: Only enable for high-D or expensive likelihoods
- ``acq='wipv'``: Generally the best choice for evidence estimation

Next Steps
----------

- Try modifying the banana function parameters
- Experiment with different acquisition functions (ei, logei)
- Test with your own 2D test functions
- Move on to :doc:`cosmology` for realistic applications
