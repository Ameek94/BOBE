from jaxbo.utils.summary_plots import plot_final_samples, BOBESummaryPlotter 
from jaxbo.utils.logging_utils import get_logger
from jaxbo.utils.core_utils import renormalise_log_weights
from jaxbo.run import run_bobe
import matplotlib.pyplot as plt
import time
import sys
from getdist import MCSamples
from dynesty import DynamicNestedSampler
import numpy as np

ndim = 2
param_bounds = np.array([[-4,4],[-4,4]]).T
param_list = ['x1','x2']
param_labels = ['x_1','x_2']
likelihood_name = 'Himmelblau'

afac= 0.1

def prior_transform(x):
    return 8*x - 4

def loglike(X):
    r1 = (X[0] + X[1]**2 -7)**2
    r2 = (X[0]**2 + X[1]-11)**2
    return -0.5*(afac*r1 + r2)

def main():
    start = time.time()
    print("Starting BOBE run...")

    # Run BOBE with the new interface
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
        n_sobol_init=4,
        min_iters=10,
        n_log_ei_iters=15,
        max_eval_budget=250,
        max_gp_size=250,
        fit_step=2,
        wipv_batch_size=3,
        ns_step=2,
        num_hmc_warmup=256,
        num_hmc_samples=1024,
        mc_points_size=128,
        lengthscale_priors='DSLP',
        use_clf=False,
        minus_inf=-1e5,
        logz_threshold=0.001,
        seed=42,
        do_final_ns=True,
    )

    end = time.time()

    if results is not None:
        log = get_logger("[main]")
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
        comprehensive_results = results['comprehensive']
        timing_data = comprehensive_results['timing']

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
        overhead_pct = (overhead / timing_data['total_runtime']) * 100 if timing_data['total_runtime'] > 0 else 0

        log.info(f"Total measured phases: {total_measured:.2f}s ({(total_measured/timing_data['total_runtime']*100):.1f}%)")
        log.info(f"Overhead/unmeasured: {overhead:.2f}s ({overhead_pct:.1f}%)")

        # Find dominant phase
        if any(t > 0 for t in timing_data['phase_times'].values()):
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

        # Create parameter samples plot
        log.info("Creating parameter samples plot...")
        if hasattr(samples, 'samples'):
            sample_array = samples.samples
            weights_array = samples.weights
        else:
            sample_array = samples['x']
            weights_array = samples['weights']

        # Run Dynesty for comparison
        log.info("Running Dynesty for comparison...")
        dns_sampler = DynamicNestedSampler(loglike, prior_transform, ndim=ndim,
                                          sample='rwalk')

        dns_sampler.run_nested(print_progress=True, dlogz_init=0.01)
        res = dns_sampler.results
        mean = res['logz'][-1]
        logz_err = res['logzerr'][-1]
        log.info(f"Mean logz from dynesty = {mean:.4f} +/- {logz_err:.4f}")

        dns_samples = res['samples']
        weights = renormalise_log_weights(res['logwt'])

        reference_samples = MCSamples(samples=dns_samples, names=param_list, labels=param_labels,
                                    weights=weights,
                                    ranges=dict(zip(param_list, param_bounds.T)))

        plot_final_samples(
            gp,
            {'x': sample_array, 'weights': weights_array, 'logl': samples.get('logl', [])},
            param_list=likelihood.param_list,
            param_bounds=likelihood.param_bounds,
            param_labels=likelihood.param_labels,
            output_file=likelihood.name,
            reference_samples=reference_samples,
            reference_file=None,
            reference_label='Dynesty',
            scatter_points=True
        )

        # Save comprehensive results
        log.info("\n" + "="*60)
        log.info("SAVING RESULTS")
        log.info("="*60)

        # Results are automatically saved by BOBE, but let's summarize what was saved
        log.info(f"✓ Main results: {likelihood_name}_results.pkl")
        log.info(f"✓ Timing data: {likelihood_name}_timing.json")
        log.info(f"✓ Summary dashboard: {likelihood_name}_dashboard.pdf")
        log.info(f"✓ Parameter samples: {likelihood_name}_param_posteriors.pdf")

        log.info("\n" + "="*60)
        log.info("ANALYSIS COMPLETE")
        log.info("="*60)
        log.info("Check the generated plots and saved files for detailed analysis.")

if __name__ == "__main__":
    main()
