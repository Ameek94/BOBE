"""
Planck + DESI LCDM run with continuous Normalising Flow transform.

A Real-NVP flow is trained on HMC samples once the acquisition value drops
below ``transform_acq_threshold`` and then refreshed every ``update_step``
iterations, progressively warping the GP's unit cube to match the posterior.
"""

import os
import sys

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    os.cpu_count()
)

from BOBE import BOBE
from BOBE.transforms import NormalisingFlowTransform
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

    cobaya_input_file = './cosmo_input/LCDM_Planck_DESI.yaml'
    likelihood_name = f'Planck_DESI_LCDM_Flow_continuous_{seed}'

    print("Loading reference samples...")
    reference_samples = loadMCSamples(
        './cosmo_input/chains/Planck_DESIDr2_LCDM_MCMC',
        settings={'ignore_rows': 0.3, 'label': 'MCMC'},
    )

    start = time.time()
    print("\n" + "=" * 80)
    print("Starting BOBE run WITH Normalising Flow transform...")
    print("=" * 80)

    bobe = BOBE(
        loglikelihood=cobaya_input_file,
        likelihood_name=likelihood_name,
        confidence_for_unbounded=0.9999995,
        resume=True,
        resume_file=f'./results/LCDM_Flow/{likelihood_name}',
        save_dir='./results/LCDM_Flow/',
        save=True,
        verbosity='INFO',
        n_cobaya_init=8,
        n_sobol_init=32,
        use_clf=True,
        clf_type='svm',
        minus_inf=-1e5,
        seed=seed,
        # Pass transform as (class, kwargs): BOBE resolves param_bounds from the
        # likelihood and calls NormalisingFlowTransform(param_bounds, **kwargs).
        transform=(NormalisingFlowTransform, {
            'kl_threshold': 0.5,   # min KL between old/new posteriors to trigger update
            'max_updates': 5,       # stop after 5 flow re-fits
            'update_step': 50,      # minimum BO iterations between consecutive re-fits
            'n_layers': 8,
            'hidden_dim': 64,
            'flow_n_epochs': 2000,
        }),
    )

    results = bobe.run(
        acq='wipstd',
        min_evals=400,
        max_evals=1500,
        max_gp_size=1000,
        fit_n_points=25,
        ns_n_points=25,
        batch_size=5,
        num_hmc_warmup=512,
        num_hmc_samples=8000,
        mc_points_size=512,
        logz_threshold=0.2,
        num_chains=8,
        do_final_ns=False,
        # acquisition value below which a flow update is attempted
        transform_acq_threshold=1.0,
    )

    end = time.time()

    if results is not None:
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

        print("\n" + "=" * 80)
        print("RUN COMPLETED WITH NORMALISING FLOW TRANSFORM")
        print("=" * 80)
        print(f"Total runtime: {manual_timing:.2f}s ({manual_timing / 60:.2f} min)")
        print(f"Number of GP training points: {gp.train_x.shape[0]}")
        print(f"Flow update count: {bobe.transform.update_count}")

        sample_array = samples['x']
        weights_array = samples['weights']
        BOBE_Samples = MCSamples(
            samples=sample_array,
            names=param_list,
            labels=param_labels,
            weights=weights_array,
            ranges=dict(zip(param_list, param_bounds.T)),
        )

        print("\nCreating cosmology parameter samples plot...")
        sns.set_theme('notebook', 'ticks', palette='husl')
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'sans-serif'

        param_list_LCDM = ['omch2', 'ombh2', 'H0', 'logA', 'ns', 'tau']
        g = plots.get_subplot_plotter(subplot_size=2.5, subplot_size_ratio=1)
        g.settings.legend_fontsize = 16
        g.settings.axes_fontsize = 16
        g.settings.axes_labelsize = 16
        g.triangle_plot(
            [BOBE_Samples, reference_samples],
            params=param_list_LCDM,
            filled=[True, False],
            contour_colors=['#006FED', 'black'],
            contour_lws=[1, 1.5],
            legend_labels=['BOBE (flow)', 'MCMC'],
        )
        g.export(f'./results/LCDM_Flow/{likelihood_name}_cosmo_samples.pdf')

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
            legend_labels=['BOBE (flow)', 'MCMC'],
        )
        g.export(f'./results/LCDM_Flow/{likelihood_name}_full_samples.pdf')

        print("\n" + "=" * 80)
        print("DETAILED TIMING ANALYSIS")
        print("=" * 80)
        timing_data = results_manager.get_timing_summary()
        print(f"Total: {timing_data['total_runtime']:.2f}s ({timing_data['total_runtime'] / 60:.2f} min)")
        for phase, time_spent in timing_data['phase_times'].items():
            if time_spent > 0:
                pct = timing_data['percentages'].get(phase, 0)
                print(f"  {phase:30s}: {time_spent:8.2f}s ({pct:5.1f}%)")

        acquisition_data = results_manager.get_acquisition_data()
        iterations = np.array(acquisition_data['iterations'])
        values = np.array(acquisition_data['values'])
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(iterations, values, linestyle='-', marker='o', markersize=3)
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Acquisition Value')
        ax.set_title('Acquisition Function Values (with flow transform)')
        ax.grid(True, alpha=0.3)
        plt.savefig(f'./results/LCDM_Flow/{likelihood_name}_acquisition.pdf', bbox_inches='tight')

        print(f"\nGP Lengthscales: {gp.lengthscales}")
        print(f"Kernel variance: {gp.kernel_variance:.4f}")

        print("\n" + "=" * 80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)


if __name__ == "__main__":
    main()
