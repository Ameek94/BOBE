Cosmological Likelihoods (through Cobaya)
======================================

This example demonstrates how to use BOBE with realistic cosmological likelihoods interfaced
through the Cobaya package. We'll estimate the Bayesian evidence for the 
standard LCDM model using Planck and DESI data.

Prerequisites
-------------

.. note::
   This example requires Cobaya. Install with:
   
   .. code-block:: bash
   
      pip install -e '.[cobaya]'
   
   Run this from the BOBE source directory.

You'll also need the Planck and DESI data files. Follow the 
`Cobaya installation guide <https://cobaya.readthedocs.io/>`_ for data setup.

Problem Description
-------------------

We're testing a cosmological model with the following:

- **Likelihood**: Planck 2018 (lowl TT + highl Camspec TTTEEE + lensing) + DESI DR2 BAO
- **Model**: Standard LCDM (flat universe)
- **Parameters**: 6 cosmological parameters

  - :math:`\Omega_b h^2`: Baryon density
  - :math:`\Omega_c h^2`: Cold dark matter density  
  - :math:`H_0`: Hubble parameter
  - :math:`\log(10^{10} A_s)`: Primordial amplitude
  - :math:`n_s`: Scalar spectral index
  - :math:`\tau`: Optical depth to reionization

Cobaya Input File
-----------------

First, create a YAML file defining your cosmological model:

.. code-block:: yaml

   # LCDM_Planck_DESI.yaml
   
   likelihood:
     # Planck 2018 likelihoods
     planck_2018_lowl.TT: null
     planck_2018_highl_plik.TTTEEE: null
     
     # DESI 2024 BAO
     bao.desi_2024_bao_all: null
   
   params:
     # Baryon density
     omegabh2:
       prior:
         min: 0.019
         max: 0.026
       ref:
         dist: norm
         loc: 0.02237
         scale: 0.00015
       latex: \\Omega_b h^2
     
     # Cold dark matter density
     omegach2:
       prior:
         min: 0.08
         max: 0.2
       ref:
         dist: norm
         loc: 0.1200
         scale: 0.0014
       latex: \\Omega_c h^2
     
     # Hubble parameter
     H0:
       prior:
         min: 40
         max: 100
       latex: H_0
     
     # Primordial amplitude
     logA:
       prior:
         min: 2.0
         max: 4.0
       ref:
         dist: norm
         loc: 3.05
         scale: 0.02
       latex: \\log(10^{10} A_s)
     
     # Spectral index
     ns:
       prior:
         min: 0.8
         max: 1.2
       ref:
         dist: norm
         loc: 0.965
         scale: 0.005
       latex: n_s
     
     # Optical depth
     tau:
       prior:
         min: 0.01
         max: 0.8
       ref:
         dist: norm
         loc: 0.055
         scale: 0.01
       latex: \\tau
   
   theory:
     camb:
       extra_args:
         lens_potential_accuracy: 1
         num_massive_neutrinos: 1
         nnu: 3.046

Complete Python Code
--------------------

.. code-block:: python

   from BOBE import BOBE
   
   # Initialize BOBE with Cobaya YAML file - CobayaLikelihood created internally
   sampler = BOBE(
       loglikelihood='./cosmo_input/LCDM_Planck_DESI.yaml',
       likelihood_name='Planck_DESI_LCDM',
       n_cobaya_init=32,
       n_sobol_init=64,
       save_dir='./results/',
   )
   
   # Run optimization with convergence and run settings
   results = sampler.run(
       min_evals=800,
       max_evals=2500,
       batch_size=5,
       fit_n_points=5,
       ns_n_points=5,
       logz_threshold=0.01,
   )
   
   # Access results
   print(f\"Log Evidence: {results['logz']['mean']} ± {results['logz']['err']}\")\n   samples = results['samples']

Running with MPI
           loglikelihood=cobaya_input_file,
           likelihood_name=likelihood_name,
           confidence_for_unbounded=0.9999995,
           minus_inf=-1e5,
           
           # Output
           save=True,
           save_dir='./results/',
           seed=42,
           verbosity='INFO',
           
           # Initial sampling
           n_cobaya_init=32,   # Points from Cobaya reference distribution
           n_sobol_init=64,    # Additional Sobol points
           
           # Budget
           min_evals=800,      # Minimum evaluations before convergence check
           max_evals=2500,     # Maximum likelihood evaluations
           max_gp_size=1500,   # Maximum GP training set size
           
           # Step settings
           fit_step=5,             # Fit GP every 5 evaluations
           wipv_batch_size=5,      # Evaluate 5 points per acquisition
           ns_step=5,              # Run nested sampling every 5 iterations
           optimizer='scipy',      # 'scipy' or 'optax'
           
           # HMC/MC settings for acquisition
           num_hmc_warmup=512,
           num_hmc_samples=12000,
           mc_points_size=512,
           num_chains=4,           # Parallel NUTS chains
           
           # GP settings
           gp_kwargs={
               'lengthscale_prior': None,
               'kernel_variance_prior': None,
               'lengthscale_bounds': [1e-2, 5.],
           },
           
           # Classifier settings (HIGHLY RECOMMENDED for cosmology)
           use_clf=True,
           clf_type='svm',              # 'svm' (always available) or 'nn' (requires [nn])
           clf_nsigma_threshold=20,     # Filter points below -20 sigma
           clf_update_step=1,           # Update classifier every iteration
           
           # Convergence
           logz_threshold=0.01,
           convergence_n_iters=2,  # Require 2 consecutive convergence checks
           do_final_ns=True,
       )
       
       # Run optimization
       results = bobe.run(acqs='wipv')
       
       end = time.time()
       
       if results is not None:
           log = get_logger("main")
           
           # Extract results
           gp = results['gp']
           samples = results['samples']
           logz_dict = results.get('logz', {})
           likelihood = results['likelihood']
           results_manager = results['results_manager']
           
           log.info("\n" + "="*80)
           log.info("RUN COMPLETED")
           log.info("="*80)
           log.info(f"Total time: {(end - start)/60:.2f} minutes")
           log.info(f"Log Evidence: {logz_dict.get('logz', 'N/A'):.4f} ± {logz_dict.get('logzerr', 0):.4f}")
           log.info(f"Number of likelihood evaluations: {gp.train_x.shape[0]}")
           log.info("="*80)
           
           # Create plots
           plot_results(results, likelihood_name)
   
   def plot_results(results, name):
       """Generate diagnostic and posterior plots."""
       
       samples = results['samples']
       likelihood = results['likelihood']
       
       # Get parameter names from likelihood
       param_list = likelihood.param_list
       param_labels = likelihood.param_labels
       param_bounds = likelihood.param_bounds
       
       # Create GetDist samples
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
       g = gdplt.get_subplot_plotter(width_inch=8)
       g.triangle_plot([mcs], filled=True, 
                      title_limit=1)
       plt.savefig(f'./results/{name}_triangle.pdf', bbox_inches='tight', dpi=150)
       print(f"Saved triangle plot to ./results/{name}_triangle.pdf")
       
       # Summary plots
       plotter = BOBESummaryPlotter(results)
       plotter.plot_gp_training_evolution()
       plotter.plot_logz_evolution()
       plotter.plot_acquisition_evolution()
       
       # 1D marginalized posteriors for key parameters
       g = gdplt.get_subplot_plotter()
       g.plots_1d([mcs], params=['H0', 'omegabh2', 'ns'])
       plt.savefig(f'./results/{name}_1d_marginals.pdf', bbox_inches='tight')
       print(f"Saved 1D marginals to ./results/{name}_1d_marginals.pdf")
   
   if __name__ == '__main__':
       main()

Running with MPI
----------------

For expensive cosmological likelihoods, use MPI parallelization:

.. code-block:: bash

   # Install MPI support (from BOBE source directory)
   pip install -e '.[mpi]'
   
   # Run with 4 processes
   mpirun -n 4 python cosmology_example.py

The code automatically detects MPI and distributes likelihood evaluations 
across processes.

Expected Runtime
----------------

- **Without MPI**: ~10-20 hours for 2500 evaluations
- **With MPI (4 cores)**: ~3-5 hours
- **With MPI (8+ cores)**: ~1-3 hours

Expected Results
----------------

For standard LCDM with Planck+DESI:

- **Log Evidence**: Approximately -5800 to -5850
- **Convergence**: Typically requires 1200-1800 evaluations

Posterior Constraints
~~~~~~~~~~~~~~~~~~~~~

Typical 68% confidence intervals:

- :math:`H_0 = 67.4 \pm 0.5` km/s/Mpc  
- :math:`\Omega_b h^2 = 0.02237 \pm 0.00015`
- :math:`\Omega_c h^2 = 0.1200 \pm 0.0012`
- :math:`n_s = 0.965 \pm 0.004`

Important Notes
---------------

Classifier is Essential
~~~~~~~~~~~~~~~~~~~~~~~

For cosmological likelihoods, **always use** ``use_clf=True``. The classifier:

- Filters out parameter regions with extremely low likelihoods
- Reduces wasted evaluations by ~50-70%
- Focuses the GP on the posterior region

The default SVM classifier (``clf_type='svm'``) works well and is always available.

Memory Considerations
~~~~~~~~~~~~~~~~~~~~~

For very long runs:

- Set ``max_gp_size`` to limit memory (1000-2000 is typical)
- The GP uses the most informative points when this limit is reached
- Save intermediate results with ``save=True``

Resume from Checkpoint
~~~~~~~~~~~~~~~~~~~~~~

To resume an interrupted run:

.. code-block:: python

   bobe = BOBE(
       loglikelihood=cobaya_input_file,
       likelihood_name=likelihood_name,
       resume=True,
       resume_file='./results/Planck_DESI_LCDM',
       # ... other settings ...
   )
   results = bobe.run(acqs='wipv')

Troubleshooting
---------------

Slow Convergence
~~~~~~~~~~~~~~~~

If convergence is slow:

1. Increase ``n_cobaya_init`` to 64-128 for better initialization
2. Try ``optimizer='optax'`` (requires ``[nn]`` install)
3. Increase ``mc_points_size`` to 1024 for better acquisition
4. Lower ``fit_step`` to 2-3 for more frequent GP updates

Likelihood Failures
~~~~~~~~~~~~~~~~~~~

If you see many ``-1e5`` values:

1. Check your Cobaya YAML file is correct
2. Verify parameter priors are reasonable
3. Increase ``clf_nsigma_threshold`` from 20 to 30

Next Steps
----------

- Compare evidence to alternative models (e.g., LCDM+Omk, CPL dark energy)
- Experiment with different acquisition functions
- Try neural network classifiers: ``clf_type='nn'`` (requires ``[nn]``)
- See :doc:`../user guide/advanced` for model comparison workflows
