"""
Test script demonstrating GP parameter rotation with covariance from reference chains.

This script:
1. Loads reference MCMC chains with GetDist
2. Extracts covariance matrix and best-fit point
3. Runs BOBE with parameter rotation enabled
4. Compares results with and without rotation
"""

import os
import sys
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    os.cpu_count()
)
from BOBE import BOBE
from BOBE.transforms import RotationTransform
from BOBE.utils.core import scale_from_unit
import time
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
import seaborn as sns
from getdist import MCSamples, plots, loadMCSamples
import numpy as np

def main():

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42

    # Set up the cosmological likelihood
    cobaya_input_file = './cosmo_input/CPL_Planck_DESI.yaml'
    likelihood_name = f'Planck_DESI_CPL_{seed}'

    # Load reference samples to extract covariance and best-fit point
    print("Loading reference samples...")
    reference_samples = loadMCSamples(
            './cosmo_input/chains/union3_CPL',
            settings={'ignore_rows': 0.3, 'label': 'MCMC'}
        )
    
    # cov_matrix = np.loadtxt('./results/CPL/Planck_DESI_CPL_Fisher_100_cov_matrix.txt')
    # center_point = np.loadtxt('./results/CPL/Planck_DESI_CPL_Fisher_100_cov_peak.txt')
    # If an initial covariance is available, pass it via:
    #   transform=RotationTransform(param_bounds, covariance=cov_matrix, center=center_point)
    # param_bounds can be obtained from a CobayaLikelihood instance before constructing BOBE.
    
    start = time.time()
    print("\n" + "="*80)
    print("Starting BOBE run WITH parameter rotation...")
    print("="*80)

    # Pass Cobaya YAML file path directly to BOBE with covariance rotation
    bobe = BOBE(
        loglikelihood=cobaya_input_file,
        likelihood_name=likelihood_name,
        confidence_for_unbounded=0.9999995,
        resume=True,
        resume_file=f'./results/CPL/{likelihood_name}',
        save_dir='./results/CPL/',
        save=True,
        verbosity='INFO',
        n_cobaya_init=16,
        n_sobol_init=64,
        use_clf=True,
        clf_type='svm',
        minus_inf=-1e5,
        seed=seed,
        gp_kwargs={'lengthscale_bounds': (0.01, 100), 'kernel_variance_bounds': (1e-4, 1e4)},
        # RotationTransform passed as a class: BOBE instantiates it with the
        # resolved param_bounds. Supply an instance instead to use an initial
        # covariance, e.g.: RotationTransform(param_bounds, covariance=cov_matrix, center=center_point)
        transform=None,
    )

    results = bobe.run(
        acq='logei',
        min_evals=100, 
        max_evals=250,
        max_gp_size=1500,
        fit_n_points=25, 
        ns_n_points=25,
        batch_size=5,
        do_final_ns=False,
    )
    
    results = bobe.run(
        acq='wipstd',
        min_evals=600, 
        max_evals=3000,
        max_gp_size=1600,
        fit_n_points=25, 
        ns_n_points=25,
        batch_size=5,
        convergence_n_iters=2,
        num_hmc_warmup=512,
        num_hmc_samples=8000, 
        mc_points_size=512,
        logz_threshold=0.4,
        num_chains=8,
        do_final_ns=False,
    )

    end = time.time()

    if results is not None:  # when running in MPI mode, only rank 0 returns results, rest return None

        gp = results['gp']
        logz_dict = results.get('logz', {})
        likelihood = results['likelihood']
        results_manager = results['results_manager']
        samples = results['samples']
        param_bounds = likelihood.param_bounds
        param_list = likelihood.param_list
        param_labels = likelihood.param_labels
        ndim = len(param_list)

        manual_timing = end - start

        print("\n" + "="*80)
        print("RUN COMPLETED WITH ROTATION")
        print("="*80)
        # print(f"Final LogZ: {logz_dict.get('mean', 'N/A'):.4f}")
        # if 'upper' in logz_dict and 'lower' in logz_dict:
        #     print(f"LogZ uncertainty: ±{(logz_dict['upper'] - logz_dict['lower'])/2:.4f}")
        print(f"Total runtime: {manual_timing:.2f} seconds ({manual_timing/60:.2f} minutes)")
        print(f"Number of GP training points: {gp.train_x.shape[0]}")
        
        # if gp.use_rotation:
        #     print(f"\nRotation details:")
        #     print(f"  Center (unit cube): {gp.center_unit}")
        #     print(f"  L_rotation shape: {gp.L_rotation.shape}")
        #     print(f"  Condition number of L: {np.linalg.cond(gp.L_rotation):.2e}")

        # Create MCSamples from BOBE results
        sample_array = samples['x']
        weights_array = samples['weights']
        BOBE_Samples = MCSamples(
            samples=sample_array, 
            names=param_list, 
            labels=param_labels,
            weights=weights_array, 
            ranges=dict(zip(param_list, param_bounds.T))
        )

        # Gaussian_samples = np.random.multivariate_normal(mean=center_point, cov=cov_matrix, size=16000)
        # Gaussian_samples = np.clip(Gaussian_samples, param_bounds[0, :], param_bounds[1, :])  # Ensure samples are within bounds
        # Gaussian_MCSamples = MCSamples(
        #     samples=Gaussian_samples,
        #     names=param_list,
        #     labels=param_labels,
        #     ranges=dict(zip(param_list, param_bounds.T))
        # )

        # Create parameter samples plot
        print("\nCreating parameter samples plot...")
        sns.set_theme('notebook', 'ticks', palette='husl')
        plt.rcParams['text.usetex'] = False  # Disable LaTeX for compatibility
        plt.rcParams['font.family'] = 'sans-serif'

        g = plots.get_subplot_plotter(subplot_size=2.5, subplot_size_ratio=1)
        g.settings.legend_fontsize = 16
        g.settings.axes_fontsize = 16
        g.settings.axes_labelsize = 16
        g.triangle_plot(
            [BOBE_Samples, reference_samples], 
            params=['w','wa','ombh2', 'omch2', 'H0', 'ns', 'logA', 'tau'],
            filled=[True, False],
            contour_colors=['#006FED', 'black'], 
            contour_lws=[1, 1.5],
            legend_labels=['BOBE (rotated)', 'MCMC'],
        ) 
        
        # # Add scatter points for GP training data
        # points = scale_from_unit(gp.train_x_original, param_bounds)
        # for i in range(ndim):
        #     for j in range(i+1, ndim):
        #         ax = g.subplots[j, i]
        #         ax.scatter(points[:, i], points[:, j], alpha=0.75, color='red', s=4)
        
        g.export(f'./results/CPL/{likelihood_name}_cosmo_samples.pdf')
        print(f"Saved plot to ./results/CPL/{likelihood_name}_cosmo_samples.pdf")

        g = plots.get_subplot_plotter(subplot_size=2.5, subplot_size_ratio=1)
        g.settings.legend_fontsize = 16
        g.settings.axes_fontsize = 16
        g.settings.axes_labelsize = 16
        g.triangle_plot(
            [BOBE_Samples, reference_samples], 
            params=param_list,
            filled=[True, False],
            contour_colors=['#006FED', 'black'], 
            contour_lws=[1, 1.5],
            legend_labels=['BOBE (rotated)', 'MCMC'],
        ) 
        
        # # Add scatter points for GP training data
        # points = scale_from_unit(gp.train_x_original, param_bounds)
        # for i in range(ndim):
        #     for j in range(i+1, ndim):
        #         ax = g.subplots[j, i]
        #         ax.scatter(points[:, i], points[:, j], alpha=0.75, color='red', s=4)
        
        g.export(f'./results/CPL/{likelihood_name}_full_samples.pdf')
        print(f"Saved plot to ./results/CPL/{likelihood_name}_full_samples.pdf")


        # Print timing analysis
        print("\n" + "="*80)
        print("DETAILED TIMING ANALYSIS")
        print("="*80)

        timing_data = results_manager.get_timing_summary()

        print(f"Total runtime: {timing_data['total_runtime']:.2f} seconds ({timing_data['total_runtime']/60:.2f} minutes)")
        print("\nPhase Breakdown:")
        print("-" * 50)  
        for phase, time_spent in timing_data['phase_times'].items():
            if time_spent > 0:
                percentage = timing_data['percentages'].get(phase, 0)
                print(f"{phase:30s}: {time_spent:8.2f}s ({percentage:5.1f}%)")

        # Plot acquisition data
        acquisition_data = results_manager.get_acquisition_data()
        iterations = np.array(acquisition_data['iterations'])
        values = np.array(acquisition_data['values'])
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(iterations, values, linestyle='-', marker='o', markersize=3)
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Acquisition Value')
        ax.set_title('Acquisition Function Values (with rotation)')
        ax.grid(True, alpha=0.3)
        plt.savefig(f"./results/CPL/{likelihood_name}_acquisition.pdf", bbox_inches='tight')
        print(f"Saved acquisition plot to ./results/CPL/{likelihood_name}_acquisition.pdf")

        # Compare lengthscales before/after rotation
        print("\n" + "="*80)
        print("GP HYPERPARAMETERS")
        print("="*80)
        print(f"Lengthscales: {gp.lengthscales}")
        print(f"Kernel variance: {gp.kernel_variance:.4f}")
        print(f"Noise: {gp.noise:.2e}")
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)

if __name__ == "__main__":
    main()
