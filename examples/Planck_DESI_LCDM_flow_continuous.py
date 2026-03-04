import os
import sys

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    os.cpu_count()
)

from BOBE import BOBE
from BOBE.utils.core import renormalise_log_weights, scale_from_unit
import time
import matplotlib.pyplot as plt
import seaborn as sns
from getdist import MCSamples, plots, loadMCSamples
import numpy as np

def main():
    # Set up the cosmological likelihood
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    cobaya_input_file = './cosmo_input/LCDM_Planck_DESI.yaml'
    likelihood_name = f'Planck_DESI_LCDM_rotation_flowdiag_{seed}'
    
    start = time.time()
    print("Starting BOBE run with rotation transform + flow diagnostic…")

    # Use the rotation-based approach (covariance whitening).  Within BOBE, each
    # time the rotation is updated, a diagnostic normalising flow is also trained
    # on the same MC samples and the mean log-prob under both models is compared
    # (see BOBE._compute_flow_rotation_kl_diag).  This helps diagnose whether the
    # posterior is well approximated by a Gaussian or needs a full flow.
    bobe = BOBE(
        loglikelihood=cobaya_input_file,
        likelihood_name=likelihood_name,
        confidence_for_unbounded=0.9999995,
        resume=False,
        resume_file=f'./results/LCDM/{likelihood_name}',
        save_dir='./results/LCDM/',
        save=True,
        verbosity='INFO',
        n_cobaya_init=4,
        n_sobol_init=32,
        optimizer='scipy',
        gp_kwargs={'lengthscale_prior': None, 'lengthscale_bounds': [1e-2, 4.]},
        use_clf=True,
        clf_type='svm',
        seed=seed,
        # No use_flow_transform — use the default rotation-based approach.
    )
    
    results = bobe.run(
        acq='wipstd',
        min_evals=400,
        max_evals=2500,
        max_gp_size=1500,
        convergence_n_iters=2,
        fit_n_points=20,
        batch_size=5,
        ns_n_points=20,
        num_hmc_warmup=512,
        num_hmc_samples=4096,
        mc_points_size=512,
        num_chains=8,
        thinning=1,
        logz_threshold=1/3,
        do_final_ns=True,
        # Rotation update settings: update every 20 iterations once we have
        # >= min_evals samples.  Each update also trains a diagnostic flow and
        # logs E[log q_flow] - E[log q_gauss] to compare both approximations.
        rotation_update_step=20,
        max_rotation_updates=5,
        rotation_logz_threshold=4.0,
    )

    end = time.time()

    if results is not None:  # when running in MPI mode, only rank 0 returns results, rest return None

        gp = results['gp']
        logz_dict = results.get('logz', {})
        likelihood = results['likelihood']
        results_manager = results['results_manager']
        samples = results['samples']
        flow_samples_phys = results.get('flow_samples', None)
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
            './cosmo_input/chains/Planck_DESIDr2_LCDM_MCMC',
            settings={'ignore_rows': 0.3, 'label': 'MCMC'}
        )

        # Create MCSamples from BOBE results
        sample_array = samples['x']
        weights_array = samples['weights']
        BOBE_Samples = MCSamples(samples=sample_array, names=param_list, labels=param_labels,
                                    weights=weights_array, 
                                    ranges= dict(zip(param_list,param_bounds.T)))

        # Create MCSamples from final flow samples (equal-weight draws from the trained flow)
        plot_sets = [BOBE_Samples, reference_samples]
        legend_labels = ['BOBE (nested)', 'MCMC']
        contour_colors = ['#006FED', 'black']
        filled = [True, False]
        contour_lws = [1, 1.5]
        if flow_samples_phys is not None:
            Flow_Samples = MCSamples(samples=flow_samples_phys, names=param_list, labels=param_labels,
                                     ranges=dict(zip(param_list, param_bounds.T)),
                                     label='Flow')
            plot_sets.append(Flow_Samples)
            legend_labels.append('Flow samples')
            contour_colors.append('#E8000B')
            filled.append(False)
            contour_lws.append(1.5)

        # Create parameter samples plot - cosmology parameters only
        print("Creating cosmology parameter samples plot...")
        sns.set_theme('notebook', 'ticks', palette='husl')
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'

        param_list_LCDM = ['omch2','ombh2','H0','logA','ns','tau']
        g = plots.get_subplot_plotter(subplot_size=2.5, subplot_size_ratio=1)
        g.settings.legend_fontsize = 18
        g.settings.axes_fontsize = 18
        g.settings.axes_labelsize = 18
        g.triangle_plot(plot_sets, filled=filled,
                    contour_colors=contour_colors, contour_lws=contour_lws,
                    params=param_list_LCDM,
                    legend_labels=legend_labels) 
        g.export(f'./results/LCDM/{likelihood.name}_cosmo_posteriors.pdf')

        # Create parameter samples plot - all parameters
        print("Creating full parameter samples plot...")
        g = plots.get_subplot_plotter(subplot_size=2.5, subplot_size_ratio=1)
        g.settings.legend_fontsize = 22
        g.settings.axes_fontsize = 22
        g.settings.axes_labelsize = 22
        g.triangle_plot(plot_sets, filled=filled,
                    contour_colors=contour_colors, contour_lws=contour_lws,
                    legend_labels=legend_labels) 
        g.export(f'./results/LCDM/{likelihood.name}_full_posteriors.pdf')

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
        ax.plot(iterations, values,  linestyle='-')
        ax.set_yscale('log')
        ax.set_xlabel(r'Iteration')
        ax.set_ylabel(r'Acquisition Value')
        plt.savefig(f"./results/LCDM/{likelihood.name}_acquisition.pdf", bbox_inches='tight')

if __name__ == "__main__":
    main()
