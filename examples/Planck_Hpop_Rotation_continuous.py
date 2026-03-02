import os
import sys
import time
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from getdist import MCSamples, plots, loadMCSamples

# --- Command line arguments ---
# Arg 1: Number of devices for XLA
num_devices = int(sys.argv[1]) if len(sys.argv) > 1 else 8
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices}"

# Arg 2: Classifier type ('svm' or 'gp')
clf_type = str(sys.argv[2]) if len(sys.argv) > 2 else 'svm'

# Arg 3: Random seed
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

# --- Imports ---
from BOBE import BOBE
from BOBE.utils.log import get_logger
from BOBE.utils.plot import plot_final_samples, BOBESummaryPlotter

def load_covariance_and_center(covmat_file, minimum_file, param_names):
    """
    Load covariance matrix and best fit point from Cobaya/GetDist output files.
    
    Parameters
    ----------
    covmat_file : str
        Path to .covmat file (covariance matrix)
    minimum_file : str
        Path to .minimum file (best fit point)
    param_names : list
        List of parameter names in the order expected by BOBE
        
    Returns
    -------
    cov_matrix : np.ndarray, shape (n_params, n_params)
        Covariance matrix
    center : np.ndarray, shape (n_params,)
        Best fit parameter values
    """
    # Load covariance matrix
    cov_matrix = np.loadtxt(covmat_file)
    
    # Load best fit point
    with open(minimum_file, 'r') as f:
        lines = f.readlines()
    
    # Parse the minimum file to extract parameter values
    # Format: index value name label
    best_fit_dict = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-log'):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                idx = int(parts[0])
                value = float(parts[1])
                name = parts[2]
                best_fit_dict[name] = value
            except ValueError:
                continue
    
    # Extract values in the order specified by param_names
    center = np.array([best_fit_dict[name] for name in param_names])
    
    return cov_matrix, center

def main():
    """
    Main function to configure and run the Bayesian optimization with rotation.
    """
    # Load rotation matrix from existing MCMC results
    # covmat_file = './cosmo_input/chains/Hpop.covmat'
    # min_file = './cosmo_input/chains/Hpop.minimum'
    # paramnames = ['omch2', 'logA', 'ns', 'H0', 'ombh2', 'tau', 'A_planck', 'cal100A', 'cal100B', 'cal143B', 'cal217A', 'cal217B', 'Aradio', 'Adusty', 'AdustT', 'beta_dustT', 'Acib', 'beta_cib', 'Atsz', 'Aksz', 'xi', 'AdustP', 'beta_dustP']
    
    # cov, center = load_covariance_and_center(covmat_file, min_file, paramnames)

    # Set up the cosmological likelihood
    cobaya_input_file = './cosmo_input/Hpop.yaml'
    
    start = time.time()
    print("Starting BOBE run with automatic timing measurement...")

    likelihood_name = f'Planck_Hpop_rotation_continuous_{seed}'

    # Pass Cobaya YAML file path directly to BOBE
    bobe = BOBE(
        loglikelihood=cobaya_input_file,
        likelihood_name=likelihood_name,
        confidence_for_unbounded=0.9999995,
        resume=True,
        resume_file=f'./results/LCDM/{likelihood_name}',
        save_dir='./results/LCDM/',
        save=True,
        verbosity='INFO',
        n_cobaya_init=32,
        n_sobol_init=64,
        optimizer='scipy',
        gp_kwargs={'lengthscale_prior': None, 'lengthscale_bounds': [1e-2, 4.]},
        use_clf=True,
        clf_type=clf_type,
        seed=seed,
    )

    results = bobe.run(
        acq= 'logei',
        min_evals=500,
        max_evals=500,
        max_gp_size=1600,
        convergence_n_iters=2,
        fit_n_points=25,
    )
    
    results = bobe.run(
        acq= 'wipstd',
        min_evals=750,
        max_evals=3000,
        max_gp_size=1600,
        convergence_n_iters=2,
        fit_n_points=25,
        batch_size=5,
        ns_n_points=25,
        num_hmc_warmup=512,
        num_hmc_samples=4096,
        mc_points_size=512,
        num_chains=8,
        logz_threshold=0.8,
        do_final_ns=True,
        max_rotation_updates=20,
        rotation_update_step=25,
        rotation_kl_threshold=0.5    
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

        print("\n" + "="*60)
        print("RUN COMPLETED")
        print(f"Final LogZ: {logz_dict.get('mean', 'N/A'):.4f}")
        if 'upper' in logz_dict and 'lower' in logz_dict:
            print(f"LogZ uncertainty: ±{(logz_dict['upper'] - logz_dict['lower'])/2:.4f}")

        print("="*60)
        print(f"Manual timing: {manual_timing:.2f} seconds ({manual_timing/60:.2f} minutes)")

        reference_samples = loadMCSamples(
            './cosmo_input/chains/Hpop',
            settings={'ignore_rows': 0.3, 'label': 'MCMC'}
        )

        # Create MCSamples from BOBE results
        sample_array = samples['x']
        weights_array = samples['weights']
        BOBE_Samples = MCSamples(samples=sample_array, names=param_list, labels=param_labels,
                                    weights=weights_array, 
                                    ranges=dict(zip(param_list, param_bounds.T)))

        # Create parameter samples plot - cosmology parameters only
        print("Creating cosmology parameter samples plot...")
        # sns.set_theme('notebook', 'ticks', palette='husl')
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'

        param_list_cosmo = ['omch2', 'ombh2', 'H0', 'logA', 'ns', 'tau']
        g = plots.get_subplot_plotter(subplot_size=2.5, subplot_size_ratio=1)
        g.settings.legend_fontsize = 18
        g.settings.axes_fontsize = 18
        g.settings.axes_labelsize = 18
        g.triangle_plot([BOBE_Samples, reference_samples], filled=[True, False],
                    contour_colors=['#006FED', 'black'], contour_lws=[1, 1.5],
                    params=param_list_cosmo,
                    legend_labels=['BOBE', 'MCMC']) 
        g.export(f'./results/LCDM/{likelihood.name}_cosmo_posteriors.pdf')

        # Create parameter samples plot - all parameters
        print("Creating full parameter samples plot...")
        g = plots.get_subplot_plotter(subplot_size=2.5, subplot_size_ratio=1)
        g.settings.legend_fontsize = 22
        g.settings.axes_fontsize = 22
        g.settings.axes_labelsize = 22
        g.triangle_plot([BOBE_Samples, reference_samples], filled=[True, False],
                    contour_colors=['#006FED', 'black'], contour_lws=[1, 1.5],
                    legend_labels=['BOBE', 'MCMC']) 
        g.export(f'./results/LCDM/{likelihood.name}_full_posteriors.pdf')

        # # Compare BOBE posterior vs covariance-based Gaussian approximation
        # print("Creating Gaussian comparison plot...")
        # n_gauss = 100_000
        # gauss_samples = np.random.default_rng(42).multivariate_normal(center, cov, size=n_gauss)
        # # Clip to physical bounds so GetDist doesn't complain
        # lower, upper = param_bounds[0], param_bounds[1]
        # mask = np.all((gauss_samples >= lower) & (gauss_samples <= upper), axis=1)
        # gauss_samples = gauss_samples[mask]
        # Gauss_Samples = MCSamples(samples=gauss_samples, names=param_list, labels=param_labels,
        #                           label='Gaussian approx',
        #                           ranges=dict(zip(param_list, param_bounds.T)))

        # g2 = plots.get_subplot_plotter(subplot_size=2.5, subplot_size_ratio=1)
        # g2.settings.legend_fontsize = 18
        # g2.settings.axes_fontsize = 18
        # g2.settings.axes_labelsize = 18
        # g2.triangle_plot([BOBE_Samples, Gauss_Samples, reference_samples], params=param_list_cosmo,
        #                  filled=[True, False, False],
        #                  contour_colors=['#006FED', '#E6550D', 'black'], contour_lws=[1, 1.5, 1.5],
        #                  legend_labels=['BOBE', 'Covariance approx', 'MCMC'])
        # g2.export(f'./results/LCDM/{likelihood.name}_gauss_comparison.pdf')
        # print(f"Gaussian comparison plot saved ({gauss_samples.shape[0]} samples after clipping)")

        # Print timing analysis
        print("DETAILED TIMING ANALYSIS")

        timing_data = results_manager.get_timing_summary()

        print(f"Automatic timing: {timing_data['total_runtime']:.2f} seconds ({timing_data['total_runtime']/60:.2f} minutes)")
        print("Phase Breakdown:")
        print("-" * 40)  
        for phase, time_spent in timing_data['phase_times'].items():
            if time_spent > 0:
                percentage = timing_data['percentages'].get(phase, 0)
                print(f"{phase:25s}: {time_spent:8.2f}s ({percentage:5.1f}%)")

        # Plot acquisition data
        acquisition_data = results_manager.get_acquisition_data()
        iterations = np.array(acquisition_data['iterations'])
        values = np.array(acquisition_data['values'])
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(iterations, values, linestyle='-')
        ax.set_yscale('log')
        ax.set_xlabel(r'Iteration')
        ax.set_ylabel(r'Acquisition Value')
        plt.savefig(f"./results/LCDM/{likelihood.name}_acquisition.pdf", bbox_inches='tight')

if __name__ == "__main__":
    main()
