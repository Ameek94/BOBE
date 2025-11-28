import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    os.cpu_count()
)
from jaxbo.utils.plot import plot_final_samples, BOBESummaryPlotter
import time
import matplotlib.pyplot as plt
import seaborn as sns
from jaxbo.utils.log import get_logger
from jaxbo import BOBE

def main():
    # Set up the cosmological likelihood
    cobaya_input_file = './cosmo_input/LCDM_lite.yaml'
    ls_priors = None
    likelihood_name = f'Planck_lite_uniform'

    start = time.time()
    print("Starting BOBE run...")

    # Pass Cobaya YAML file path directly to BOBE
    bobe = BOBE(
        loglikelihood=cobaya_input_file,  # BOBE handles CobayaLikelihood internally
        likelihood_name=likelihood_name,
        confidence_for_unbounded=0.9999995,
        noise_std=0.0,
        resume=False,
        resume_file=f'{likelihood_name}',
        save_dir='./results/',
        save=True,
        verbosity='INFO',
        n_cobaya_init=4, 
        n_sobol_init=8, 
        min_evals=25, 
        max_evals=250,
        max_gp_size=200,
        fit_step=4, 
        ns_step=4,
        wipv_batch_size=4,
        num_hmc_warmup=256,
        num_hmc_samples=2048, 
        mc_points_size=256,
        gp_kwargs={'lengthscale_prior': ls_priors,}, 
        use_clf=True,
        clf_type='svm',
        minus_inf=-1e5,
        logz_threshold=0.001,
        seed=10,
        do_final_ns=False,
    )
    
    results = bobe.run(['wipv'])

    end = time.time()

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

        plot_final_samples(
            gp, 
            {'x': sample_array, 'weights': weights_array, 'logl': samples.get('logl', [])},
            param_list=likelihood.param_list,
            param_bounds=likelihood.param_bounds,
            param_labels=likelihood.param_labels,
            output_file=f'./results/{likelihood.name}',
            reference_file='./cosmo_input/chains/Planck_lite_mcmc',
            reference_ignore_rows=0.3,
            reference_label='MCMC',
            scatter_points=True
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
            save_path=f"./results/{likelihood.name}_dashboard.pdf"
        )

        # Save comprehensive results
        log.info("\n" + "="*60)
        log.info("SAVING RESULTS")
        log.info("="*60)

        # Results are automatically saved by BOBE, but let's summarize what was saved
        log.info(f"✓ Main results: {likelihood_name}_results.pkl")
        log.info(f"✓ Timing data: {likelihood_name}_timing.json")
        log.info(f"✓ Legacy samples: {likelihood_name}_samples.npz")
        log.info(f"✓ Summary dashboard: {likelihood_name}_dashboard.pdf")
        # log.info(f"✓ Detailed timing: {likelihood_name}_timing_detailed.pdf")
        # log.info(f"✓ Evidence evolution: {likelihood_name}_evidence.pdf")
        # log.info(f"✓ Acquisition evolution: {likelihood_name}_acquisition_evolution.pdf")
        # log.info(f"✓ Parameter samples: {likelihood_name}_samples.pdf")

        log.info("\n" + "="*60)
        log.info("ANALYSIS COMPLETE")
        log.info("="*60)
        log.info("Check the generated plots and saved files for detailed analysis.")

if __name__ == "__main__":
    # Run the analysis
    main()