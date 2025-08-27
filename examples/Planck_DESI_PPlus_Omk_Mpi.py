import os
import sys
num_devices = int(sys.argv[1]) if len(sys.argv) > 1 else 8
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    num_devices
)
from jaxbo.utils.summary_plots import plot_final_samples, BOBESummaryPlotter
import time
import matplotlib.pyplot as plt
import seaborn as sns   
from jaxbo.run import run_bobe
from jaxbo.utils.logging_utils import get_logger

def main():

    # Set up the cosmological likelihood
    cobaya_input_file = './cosmo_input/LCDM_Planck_DESI_Omk_PPlus.yaml'
    
    start = time.time()
    print("Starting BOBE run with automatic timing measurement...")

    name = str(sys.argv[3]) if len(sys.argv) > 3 else 'serial'

    clf_type = str(sys.argv[2]) if len(sys.argv) > 2 else 'svm'

    likelihood_name = f'Planck_DESI_PP_Omk_{name}_{clf_type}'
    
    results = run_bobe(
        likelihood=cobaya_input_file,
        likelihood_kwargs={
            'confidence_for_unbounded': 0.9999995,
            'minus_inf': -1e5,
            'noise_std': 0.0,
            'name': likelihood_name,
        },
        verbosity='INFO',
        n_log_ei_iters=300,
        n_cobaya_init=8,
        n_sobol_init=32,
        min_iters=400,
        max_eval_budget=2500,
        max_gp_size=1800,
        fit_step=10, 
        zeta_ei=0.1,
        update_mc_step=1, 
        wipv_batch_size=5,
        ns_step=5,
        num_hmc_warmup=512,
        num_hmc_samples=5000, 
        mc_points_size=512,
        lengthscale_priors='DSLP', 
        use_clf=True,
        clf_use_size=10,
        clf_threshold=350,
        gp_threshold=500,
        clf_update_step=1,  # SVM update step
        clf_type=clf_type,  # Using SVM for classification
        minus_inf=-1e5,
        logz_threshold=0.02,
        seed=10,  # For reproducibility
        do_final_ns=False,
    )

    end = time.time()

    # The rest of the script runs only on the master process
    if results is not None:
        # Run BOBE with automatic timing collection
        manual_timing = end - start
        log = get_logger("[main]")

        log.info("\n" + "="*60)
        log.info("RUN COMPLETED")
        log.info("="*60)
        log.info(f"Manual timing: {manual_timing:.2f} seconds ({manual_timing/60:.2f} minutes)")

        # Extract components for backward compatibility
        gp = results['gp']
        samples = results['samples']
        likelihood = results['likelihood']
        logz_dict = results.get('logz', {})
        comprehensive_results = results['comprehensive']
        timing_data = comprehensive_results['timing']


        plt.style.use('default')

        # Enable LaTeX rendering for mathematical expressions
        plt.rcParams['text.usetex'] = True 
        plt.rcParams['font.family'] = 'serif'

        param_list_LCDM = ['omk','omch2','ombh2','logA','ns','H0','tau']
        plot_final_samples(
            gp, 
            {'x': sample_array, 'weights': weights_array, 'logl': samples.get('logl', [])},
            param_list=likelihood.param_list,
            param_bounds=likelihood.param_bounds,
            param_labels=likelihood.param_labels,
            plot_params=param_list_LCDM,
            output_file=f'{likelihood.name}_cosmo',
            reference_file='./cosmo_input/chains/PPlus_curved_LCDM',
            reference_ignore_rows=0.3,
            reference_label='MCMC',
            scatter_points=False
        )


        plot_final_samples(
            gp, 
            {'x': sample_array, 'weights': weights_array, 'logl': samples.get('logl', [])},
            param_list=likelihood.param_list,
            param_bounds=likelihood.param_bounds,
            param_labels=likelihood.param_labels,
            output_file=f'{likelihood.name}_full',
            reference_file='./cosmo_input/chains/PPlus_curved_LCDM',
            reference_ignore_rows=0.3,
            reference_label='MCMC',
            scatter_points=False
        )

        # Print detailed timing analysis
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

        # Analyze timing efficiency
        log.info("\nTiming Efficiency Analysis:")
        log.info("-" * 40)
        total_measured = sum(t for t in timing_data['phase_times'].values() if t > 0)
        overhead = timing_data['total_runtime'] - total_measured
        overhead_pct = (overhead / timing_data['total_runtime']) * 100

        log.info(f"Total measured phases: {total_measured:.2f}s ({(total_measured/timing_data['total_runtime']*100):.1f}%)")
        log.info(f"Overhead/unmeasured: {overhead:.2f}s ({overhead_pct:.1f}%)")

        # Find dominant phase
        max_phase = max(timing_data['phase_times'].items(), key=lambda x: x[1])
        log.info(f"Dominant phase: {max_phase[0]} ({timing_data['percentages'][max_phase[0]]:.1f}%)")

        # Print convergence info
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

        # Create comprehensive plots
        log.info("\n" + "="*60)
        log.info("GENERATING PLOTS")
        log.info("="*60)

        # Initialize plotter
        plotter = BOBESummaryPlotter(results['results_manager'])

        # Get GP and best loglike evolution data
        gp_data = results['results_manager'].get_gp_data()
        best_loglike_data = results['results_manager'].get_best_loglike_data()
        acquisition_data = results['results_manager'].get_acquisition_data()

        # Create summary dashboard with timing data
        log.info("Creating summary dashboard...")
        fig_dashboard = plotter.create_summary_dashboard(
            gp_data=gp_data,
            acquisition_data=acquisition_data,
            best_loglike_data=best_loglike_data,
            timing_data=timing_data,
            save_path=f"{likelihood.name}_dashboard.pdf"
        )
        # plt.show()

        # Create individual timing plot
        log.info("Creating detailed timing plot...")
        fig_timing, ax_timing = plt.subplots(1, 1, figsize=(10, 6))
        plotter.plot_timing_breakdown(timing_data=timing_data, ax=ax_timing)
        ax_timing.set_title(f"Timing Breakdown - {likelihood.name}")
        plt.tight_layout()
        plt.savefig(f"{likelihood.name}_timing_detailed.pdf", bbox_inches='tight')
        # plt.show()

        # Create evidence evolution plot if available
        if comprehensive_results.get('logz_history'):
            log.info("Creating evidence evolution plot...")
            fig_evidence, ax_evidence = plt.subplots(1, 1, figsize=(10, 6))
            plotter.plot_evidence_evolution(ax=ax_evidence)
            ax_evidence.set_title(f"Evidence Evolution - {likelihood.name}")
            plt.tight_layout()
            plt.savefig(f"{likelihood.name}_evidence.pdf", bbox_inches='tight')
            # plt.show()

        # Create acquisition function evolution plot
        log.info("Creating acquisition function evolution plot...")
        acquisition_data = results['results_manager'].get_acquisition_data()
        if acquisition_data and acquisition_data.get('iterations'):
            fig_acquisition, ax_acquisition = plt.subplots(1, 1, figsize=(10, 6))
            plotter.plot_acquisition_evolution(acquisition_data=acquisition_data, ax=ax_acquisition)
            ax_acquisition.set_title(f"Acquisition Function Evolution - {likelihood.name}")
            plt.tight_layout()
            plt.savefig(f"{likelihood.name}_acquisition_evolution.pdf", bbox_inches='tight')
            # plt.show()
        else:
            log.info("No acquisition function data available for plotting.")

        # Create parameter samples plot
        log.info("Creating parameter samples plot...")
        if hasattr(samples, 'samples'):  # GetDist samples
            sample_array = samples.samples
            weights_array = samples.weights
        else:  # Dictionary format
            sample_array = samples['x']
            weights_array = samples['weights']




if __name__ == "__main__":
    # Run the analysis
    main()