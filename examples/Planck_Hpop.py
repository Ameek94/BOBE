import os
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
# --- Command line arguments ---
# Arg 1: Number of devices for XLA
num_devices = int(sys.argv[1]) if len(sys.argv) > 1 else 8
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices}"


# --- Command line arguments ---
# Arg 1: Classifier type ('svm' or 'gp')
clf_type = str(sys.argv[2]) if len(sys.argv) > 2 else 'svm'

# Arg 3: MPI method ('serial' or 'mpi')
mpi_method = str(sys.argv[3]) if len(sys.argv) > 3 else 'serial'

# --- Imports ---
from jaxbo.run import run_bobe
from jaxbo.utils.logging_utils import get_logger
from jaxbo.utils.summary_plots import plot_final_samples, BOBESummaryPlotter

def main():
    """
    Main function to configure and run the Bayesian optimization for Planck_Hpop.
    """
    # Determine classifier update step based on type
    clf_update_step = 1 if clf_type == 'svm' else 2

    # Set up the cosmological likelihood
    cobaya_input_file = './cosmo_input/LCDM_new_CMB.yaml'
    likelihood_name = f'Planck_Hpop_{clf_type}_{mpi_method}'

    start = time.time()
    print("Starting BOBE run with automatic timing measurement...")

    # --- Run BOBE with combined settings ---
    results = run_bobe(
        # Likelihood settings
        likelihood=cobaya_input_file,
        likelihood_kwargs={
            'confidence_for_unbounded': 0.9999995,
            'minus_inf': -1e6,
            'noise_std': 0.0,
            'name': likelihood_name,
        },

        # General run settings
        verbosity='INFO',
        seed=200,

        # Iteration and budget settings
        n_log_ei_iters=400,
        n_cobaya_init=32,
        n_sobol_init=128,
        min_iters=600,
        max_eval_budget=5000,
        max_gp_size=2100,

        # Step settings
        fit_step=10,
        update_mc_step=1,
        ns_step=5,
        wipv_batch_size=5,

        # HMC/MC settings
        num_hmc_warmup=512,
        num_hmc_samples=8000,
        mc_points_size=768,

        # GP settings
        lengthscale_priors='DSLP',
        noise=1e-6,

        # resume
        resume=False,
        resume_file=f'{likelihood_name}',

        # Classifier settings
        use_clf=True,
        clf_type=clf_type,
        clf_use_size=100,
        clf_update_step=clf_update_step,
        clf_threshold=350,
        gp_threshold=1000,

        # Convergence and other settings
        minus_inf=-1e6,
        logz_threshold=0.05,
        do_final_ns=True,
    )

    end = time.time()

    # --- Post-processing ---
    if results is not None:
        manual_timing = end - start
        log = get_logger("main")

        log.info("\n" + "="*60)
        log.info("RUN COMPLETED")
        log.info("="*60)
        log.info(f"Manual timing: {manual_timing:.2f} seconds ({manual_timing/60:.2f} minutes)")

        # Extract components
        gp = results['gp']
        samples = results['samples']
        likelihood = results['likelihood']
        logz_dict = results.get('logz', {})
        comprehensive_results = results['comprehensive']
        timing_data = comprehensive_results['timing']

        log.info("Creating parameter samples plot...")
        if hasattr(samples, 'samples'):
            sample_array = samples.samples
            weights_array = samples.weights
        else:
            sample_array = samples['x']
            weights_array = samples['weights']

        plt.style.use('default')

        param_list_LCDM = ['omch2','H0','ombh2','logA','ns','tau']
        plot_final_samples(
            gp, 
            {'x': sample_array, 'weights': weights_array, 'logl': samples.get('logl', [])},
            param_list=likelihood.param_list,
            param_bounds=likelihood.param_bounds,
            param_labels=likelihood.param_labels,
            plot_params=param_list_LCDM,
            output_file=f'{likelihood.name}_cosmo',
            reference_file='./cosmo_input/chains/Hpop',
            reference_ignore_rows=0.3,
            reference_label='MCMC',
            scatter_points=False,
        )


        plot_final_samples(
            gp, 
            {'x': sample_array, 'weights': weights_array, 'logl': samples.get('logl', [])},
            param_list=likelihood.param_list,
            param_bounds=likelihood.param_bounds,
            param_labels=likelihood.param_labels,
            output_file=f'{likelihood.name}_full',
            reference_file='./cosmo_input/chains/Hpop',
            reference_ignore_rows=0.3,
            reference_label='MCMC',
            scatter_points=False,
        )

        # --- Timing Analysis ---
        log.info("\n" + "="*60)
        log.info("DETAILED TIMING ANALYSIS")
        log.info("="*60)
        log.info(f"Automatic timing: {timing_data['total_runtime']:.2f} seconds ({timing_data['total_runtime']/60:.2f} minutes)")
        log.info(f"Timing difference: {abs(manual_timing - timing_data['total_runtime']):.2f} seconds")
        log.info("\nPhase Breakdown:")
        log.info("-" * 40)
        for phase, time_spent in timing_data['phase_times'].items():
            if time_spent > 0:
                percentage = timing_data['percentages'].get(phase, 0)
                log.info(f"{phase:25s}: {time_spent:8.2f}s ({percentage:5.1f}%)")

        # --- Convergence Analysis ---
        log.info("\n" + "="*60)
        log.info("CONVERGENCE ANALYSIS")
        log.info("="*60)
        log.info(f"Converged: {comprehensive_results['converged']}")
        log.info(f"Termination reason: {comprehensive_results['termination_reason']}")
        log.info(f"Final GP size: {gp.train_x.shape[0]}")
        if logz_dict:
            log.info(f"Final LogZ: {logz_dict.get('mean', 'N/A'):.4f}")
            if 'upper' in logz_dict and 'lower' in logz_dict:
                log.info(f"LogZ uncertainty: ±{(logz_dict['upper'] - logz_dict['lower'])/2:.4f}")

        # --- Plotting ---
        log.info("\n" + "="*60)
        log.info("GENERATING PLOTS")
        log.info("="*60)

        sns.set_theme('notebook','ticks',palette='Paired',)

        plotter = BOBESummaryPlotter(results['results_manager'])
        gp_data = results['results_manager'].get_gp_data()
        best_loglike_data = results['results_manager'].get_best_loglike_data()
        acquisition_data = results['results_manager'].get_acquisition_data()

        log.info("Creating summary dashboard...")
        plotter.create_summary_dashboard(
            gp_data=gp_data,
            acquisition_data=acquisition_data,
            best_loglike_data=best_loglike_data,
            timing_data=timing_data,
            save_path=f"{likelihood.name}_dashboard.pdf"
        )



if __name__ == "__main__":
    main()