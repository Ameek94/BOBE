from BOBE.utils.plot import plot_final_samples, BOBESummaryPlotter
from BOBE.utils.log import get_logger
from BOBE.utils.core import renormalise_log_weights
from BOBE import BOBE
from getdist import MCSamples
from dynesty import DynamicNestedSampler
import numpy as np
import time
import matplotlib.pyplot as plt

ndim = 2
param_list = ['x1','x2']
param_labels = ['x_1','x_2']
param_bounds = np.array([[0,1],[0,1]]).T
likelihood_name = 'GaussianRing'

mean_r = 0.2
scale = 0.02

def loglike(X):
    r2 = (X[0]-0.5)**2 + (X[1]-0.5)**2
    r = np.sqrt(r2)
    return -0.5*((r-mean_r)/scale)**2

def prior_transform(x):
    return x

def main():
    start = time.time()
    print("Starting BOBE run...")

    # Run BOBE with simplified interface
    gp_kwargs = {'lengthscale_prior': 'DSLP'}
    
    bobe = BOBE(
        loglikelihood=loglike,
        param_list=param_list,
        param_bounds=param_bounds,
        param_labels=param_labels,
        likelihood_name=likelihood_name,
        gp_kwargs=gp_kwargs,
        verbosity='INFO',
        n_cobaya_init=4,
        n_sobol_init=8,
        use_clf=False,
        minus_inf=-1e5,
        seed=42,
    )
    
    results = bobe.run(
        acq='wipv',
        min_evals=30,
        max_evals=200,
        max_gp_size=200,
        fit_n_points=2,
        ns_n_points=4,
        batch_size=2,
        num_hmc_warmup=512,
        num_hmc_samples=512,
        mc_points_size=128,
        mc_points_method='NS',
        logz_threshold=0.001,
        do_final_ns=False,
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