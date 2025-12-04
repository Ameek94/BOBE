Banana Function Example
=======================

The Rosenbrock "banana" function is a classic test problem for optimization algorithms. 
Its banana-shaped contours make it challenging for many methods, providing a good test 
of BOBE's ability to handle non-trivial likelihood surfaces.

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
   from BOBE import BOBE
   
   def my_likelihood(X):
       """Rosenbrock banana function likelihood."""
       x, y = X[0], X[1]
       logpdf = -0.25 * (5 * (0.2 - x))**2 - (20 * (y/4 - x**4))**2
       return logpdf
   
   # Initialize BOBE with setup parameters
   sampler = BOBE(
       loglikelihood=my_likelihood,
       param_list=['x1', 'x2'],
       param_bounds=np.array([[-1, 1], [-1, 2]]).T,
       n_sobol_init=2,
       save_dir='./results',
   )
   
   # Run optimization with convergence and run settings
   results = sampler.run(
       min_evals=10,
       max_evals=100,
       batch_size=2,
       fit_n_points=4,
       ns_n_points=4,
       logz_threshold=0.1,
   )
   
   # Access results
   print(f\"Log Evidence: {results['logz']['mean']}\")\n   samples = results['samples']  # dictionary containing keys 'x', 'logl', 'weights'

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
3. **Accurate Evidence**: BOBE accurately estimates the evidence with ~100 evaluations

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
