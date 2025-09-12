import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    os.cpu_count()
)
from jaxbo.utils.summary_plots import plot_final_samples, BOBESummaryPlotter
import time
import matplotlib.pyplot as plt
import seaborn as sns
from jaxbo.utils.logging_utils import get_logger
from jaxbo.utils.core_utils import renormalise_log_weights
from jaxbo.run import run_bobe
from getdist import MCSamples
import numpy as np
from dynesty import DynamicNestedSampler

def loglike(X, slow=False):
    logpdf = -0.25 * (5 * (0.2 - X[0])) ** 2 - (20 * (X[1] / 4 - X[0] ** 4)) ** 2
    if slow:
        time.sleep(2)
    return logpdf

def prior_transform(x):
    x[0] = x[0]*2 - 1 #x[0] * (param_bounds[0,1] - param_bounds[0,0]) + param_bounds[0,0]
    x[1] = x[1]*3 - 1 #x[1] * (param_bounds[1,1] - param_bounds[1,0]) + param_bounds[1,0]
    return x

def main():
    # Problem setup
    ndim = 2
    param_list = ['x1', 'x2']
    param_labels = ['x_1', 'x_2']
    param_bounds = np.array([[-1, 1], [-1, 2]]).T
    ls_priors = 'DSLP'
    likelihood_name = f'banana_test_{ls_priors}'

    start = time.time()
    print("Starting BOBE run...")

    results = run_bobe(
        likelihood=loglike,
        likelihood_kwargs={
            'param_list': param_list,
            'param_bounds': param_bounds,
            'param_labels': param_labels,
            'name': likelihood_name,
            'minus_inf': -1e5,
        },
        verbosity='INFO',
        n_cobaya_init=4,
        n_sobol_init=8,
        min_evals=18,
        max_evals=100,
        max_gp_size=200,
        fit_step=1,
        wipv_batch_size=2,
        ns_step=3,
        num_hmc_warmup=256,
        num_hmc_samples=1024,
        mc_points_size=128,
        thinning=4,
        num_chains=4,
        use_clf=False,
        minus_inf=-1e5,
        logz_threshold=1e-3,
        seed=42,
        save_dir='./results/',
        save=True,
        acq = ['logei','wipv'],
        ei_goal = 1e-5,
        do_final_ns=False,
    )

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

        dns_sampler =  DynamicNestedSampler(loglike,prior_transform,ndim=ndim,
                                               sample='rwalk',logl_kwargs={'slow': False})

        dns_sampler.run_nested(print_progress=True,dlogz_init=0.01) 
        res = dns_sampler.results  
        mean = res['logz'][-1]
        logz_err = res['logzerr'][-1]
        print(f"Mean logz from dynesty = {mean:.4f} +/- {logz_err:.4f}")

        dns_samples = res['samples']
        weights = renormalise_log_weights(res['logwt'])

        reference_samples = MCSamples(samples=dns_samples, names=param_list, labels=param_labels,
                                    weights=weights, 
                                    ranges= dict(zip(param_list,param_bounds.T)))      

        plot_final_samples(
            gp,
            {'x': sample_array, 'weights': weights_array, 'logl': samples.get('logl', [])},
            param_list=likelihood.param_list,
            param_bounds=likelihood.param_bounds,
            param_labels=likelihood.param_labels,
            output_file=likelihood.name,
            output_dir='./results/',
            reference_samples=reference_samples,
            reference_file=None,
            reference_label='Dynesty',
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
        # plt.show()

        # # Create individual timing plot
        # log.info("Creating detailed timing plot...")
        # fig_timing, ax_timing = plt.subplots(1, 1, figsize=(10, 6))
        # plotter.plot_timing_breakdown(timing_data=timing_data, ax=ax_timing)
        # ax_timing.set_title(f"Timing Breakdown - {likelihood.name}")
        # plt.tight_layout()
        # plt.savefig(f"{likelihood.name}_timing_detailed.pdf", bbox_inches='tight')
        # # plt.show()

        # # Create evidence evolution plot if available
        # if comprehensive_results.get('logz_history'):
        #     log.info("Creating evidence evolution plot...")
        #     fig_evidence, ax_evidence = plt.subplots(1, 1, figsize=(10, 6))
        #     plotter.plot_evidence_evolution(ax=ax_evidence)
        #     ax_evidence.set_title(f"Evidence Evolution - {likelihood.name}")
        #     plt.tight_layout()
        #     plt.savefig(f"{likelihood.name}_evidence.pdf", bbox_inches='tight')
        #     # plt.show()

        # # Create acquisition function evolution plot
        # log.info("Creating acquisition function evolution plot...")
        # acquisition_data = results['results_manager'].get_acquisition_data()
        # if acquisition_data and acquisition_data.get('iterations'):
        #     fig_acquisition, ax_acquisition = plt.subplots(1, 1, figsize=(10, 6))
        #     plotter.plot_acquisition_evolution(acquisition_data=acquisition_data, ax=ax_acquisition)
        #     ax_acquisition.set_title(f"Acquisition Function Evolution - {likelihood.name}")
        #     plt.tight_layout()
        #     plt.savefig(f"{likelihood.name}_acquisition_evolution.pdf", bbox_inches='tight')
        #     # plt.show()
        # else:
        #     log.info("No acquisition function data available for plotting.")

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
    main()
