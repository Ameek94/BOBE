import os
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns

# --- Command line arguments ---
# Arg 1: Number of devices for XLA
num_devices = int(sys.argv[1]) if len(sys.argv) > 1 else 8
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices}"

# Arg 2: Classifier type ('svm' or 'gp')
clf_type = str(sys.argv[2]) if len(sys.argv) > 2 else 'svm'

# --- Imports ---
from jaxbo.run import run_bobe
from jaxbo.utils.logging_utils import get_logger
from jaxbo.utils.summary_plots import plot_final_samples, BOBESummaryPlotter

def main():
    """
    Main function to configure and run the Bayesian optimization.
    """

    # Set up the cosmological likelihood
    cobaya_input_file = './cosmo_input/LCDM_Planck_DESIDr2.yaml'
    
    start = time.time()
    print("Starting BOBE run with automatic timing measurement...")

    likelihood_name = f'LCDM_Planck_DESIDr2_{clf_type}_uniform'

    # --- Run BOBE with combined settings ---
    results = run_bobe(
        # Likelihood settings
        likelihood=cobaya_input_file,
        likelihood_kwargs={
            'confidence_for_unbounded': 0.9999995,
            'minus_inf': -1e5,
            'noise_std': 0.0,
            'name': likelihood_name,
        },
        
        # General run settings
        resume=False,
        resume_file=f'./results/{likelihood_name}',
        save_dir='./results',
        verbosity='INFO',
        seed=1500,

        n_cobaya_init=16,
        n_sobol_init=32,
        min_evals=500,
        max_evals=2500,
        max_gp_size=1250,
        
        # Step settings
        fit_step=5,
        wipv_batch_size=5,
        ns_step=5,
                
        # HMC/MC settings
        num_hmc_warmup=512,
        num_hmc_samples=6000,
        mc_points_size=512,
        num_chains = 6,
        thinning = 4,
        
        # GP settings
        gp_kwargs={'lengthscale_prior': None, 'kernel_variance_prior': None},
        
        # Classifier settings
        use_clf=True,
        clf_type=clf_type,
        
        # Convergence and other settings
        minus_inf=-1e5,
        logz_threshold=0.01,
        do_final_ns=True, 
        convergence_n_iters=2,
    )

    end = time.time()

    # --- Post-processing (runs only on the master process in MPI) ---
    if results is not None:
        log = get_logger("main")
        manual_timing = end - start

        log.info("\n" + "="*60)
        log.info("RUN COMPLETED")
        log.info("="*60)
        log.info(f"Manual timing: {manual_timing:.2f} seconds ({manual_timing/60:.2f} minutes)")

        # Extract components for backward compatibility
        gp = results['gp']
        samples = results['samples']
        logz_dict = results.get('logz', {})
        likelihood = results['likelihood']
        results_manager = results['results_manager']

        plt.style.use('default')
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'

        # Create parameter samples plot
        log.info("Creating parameter samples plot...")
        sample_array = samples['x']
        weights_array = samples['weights']

        param_list_LCDM = ['omch2','ombh2','H0','logA','ns','tau']
        plot_final_samples(
            gp, 
            {'x': sample_array, 'weights': weights_array, 'logl': samples.get('logl', [])},
            param_list=likelihood.param_list,
            param_bounds=likelihood.param_bounds,
            param_labels=likelihood.param_labels,
            plot_params=param_list_LCDM,
            output_file=f'{likelihood.name}_cosmo',
            output_dir='./results/',
            reference_file='./cosmo_input/chains/Planck_DESIDr2_LCDM_MCMC',
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
            output_dir='./results/',
            reference_file='./cosmo_input/chains/Planck_DESIDr2_LCDM_MCMC',
            reference_ignore_rows=0.3,
            reference_label='MCMC',
            scatter_points=False,
        )

 
        # Print detailed timing analysis
        log.info("\n" + "="*60)
        log.info("DETAILED TIMING ANALYSIS")
        log.info("="*60)

        timing_data = results_manager.get_timing_summary()

        log.info(f"Automatic timing: {timing_data['total_runtime']:.2f} seconds ({timing_data['total_runtime']/60:.2f} minutes)")
        log.info(f"Timing difference: {abs(manual_timing - timing_data['total_runtime']):.2f} seconds")

        log.info("\nPhase Breakdown:")
        log.info("-" * 40)
        for phase, time_spent in timing_data['phase_times'].items():
            if time_spent > 0:
                percentage = timing_data['percentages'].get(phase, 0)
                log.info(f"{phase:25s}: {time_spent:8.2f}s ({percentage:5.1f}%)")

        # Analyze timing efficiency
        log.info("\nTiming Efficiency Analysis:")
        log.info("-" * 40)
        total_measured = sum(t for t in timing_data['phase_times'].values() if t > 0)
        overhead = timing_data['total_runtime'] - total_measured
        overhead_pct = (overhead / timing_data['total_runtime']) * 100 if timing_data['total_runtime'] > 0 else 0

        log.info(f"Total measured phases: {total_measured:.2f}s ({(total_measured/timing_data['total_runtime']*100):.1f}%)")
        log.info(f"Overhead/unmeasured: {overhead:.2f}s ({overhead_pct:.1f}%)")

        # Find dominant phase
        if any(t > 0 for t in timing_data['phase_times'].values()):
            max_phase = max(timing_data['phase_times'].items(), key=lambda x: x[1])
            log.info(f"Dominant phase: {max_phase[0]} ({timing_data['percentages'][max_phase[0]]:.1f}%)")

        sns.set_theme('notebook', 'ticks', palette='husl')

        # Print convergence info
        log.info("\n" + "="*60)
        log.info("CONVERGENCE ANALYSIS")
        log.info("="*60)
        log.info(f"Converged: {results_manager.converged}")
        log.info(f"Termination reason: {results_manager.termination_reason}")
        log.info(f"Final GP size: {gp.train_x.shape[0]}")

        if logz_dict:
            log.info(f"Final LogZ: {logz_dict.get('mean', 'N/A'):.4f}")
            if 'upper' in logz_dict and 'lower' in logz_dict:
                log.info(f"LogZ uncertainty: ±{(logz_dict['upper'] - logz_dict['lower'])/2:.4f}")

        # Create comprehensive plots
        log.info("\n" + "="*60)
        log.info("GENERATING PLOTS")
        log.info("="*60)

        # Initialize plotter
        plotter = BOBESummaryPlotter(results_manager)

        # Get GP and best loglike evolution data
        gp_data = results_manager.get_gp_data()
        best_loglike_data = results_manager.get_best_loglike_data()
        acquisition_data = results_manager.get_acquisition_data()

        # Create summary dashboard with timing data
        log.info("Creating summary dashboard...")
        fig_dashboard = plotter.create_summary_dashboard(
            gp_data=gp_data,
            acquisition_data=acquisition_data,
            best_loglike_data=best_loglike_data,
            timing_data=timing_data,
            save_path=f"{likelihood.name}_dashboard.pdf"
        )

if __name__ == "__main__":
    main()