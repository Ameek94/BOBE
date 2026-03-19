"""
Planck-lite LCDM run using a Normalising Flow transform.

The flow is trained on HMC samples once the acquisition value drops below
``transform_acq_threshold`` and then every ``update_step`` iterations after
that, remapping the GP's unit cube to track the posterior manifold.
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
import seaborn as sns
from getdist import MCSamples, plots, loadMCSamples
import numpy as np


def main():

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42

    cobaya_input_file = 'cosmo_input/LCDM_lite.yaml'
    likelihood_name = f'Planck_lite_LCDM_Flow_{seed}'

    start = time.time()
    print("Starting BOBE run with Normalising Flow transform...")

    bobe = BOBE(
        loglikelihood=cobaya_input_file,
        likelihood_name=likelihood_name,
        confidence_for_unbounded=0.9999995,
        resume=False,
        resume_file=f'./results/LCDM_Lite/{likelihood_name}',
        save_dir='./results/LCDM_Lite/',
        save=True,
        verbosity='INFO',
        n_cobaya_init=4,
        n_sobol_init=8,
        use_clf=True,
        clf_type='svm',
        minus_inf=-1e5,
        seed=seed,
        # Pass transform as (class, kwargs): BOBE resolves param_bounds from the
        # likelihood and calls NormalisingFlowTransform(param_bounds, **kwargs).
        transform=(NormalisingFlowTransform, {
            'kl_threshold': 0.5,   # min KL between old/new posteriors to trigger update
            'max_updates': 3,       # stop after 3 flow re-fits
            'update_step': 5,      # minimum BO iterations between re-fits
            'n_layers': 8,
            'hidden_dim': 64,
            'flow_n_epochs': 2000,
        }),
    )

    results = bobe.run(
        acq='wipstd',
        min_evals=25,
        max_evals=200,
        max_gp_size=150,
        fit_n_points=8,
        ns_n_points=4,
        batch_size=2,
        num_hmc_warmup=256,
        num_hmc_samples=5000,
        mc_points_size=512,
        logz_threshold=0.025,
        do_final_ns=False,
        # acquisition value below which a flow update is attempted
        transform_acq_threshold=4.,
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

        print("\n" + "=" * 60)
        print("RUN COMPLETED")
        # print(f"Final LogZ: {logz_dict.get('mean', 'N/A'):.4f}")
        # if 'upper' in logz_dict and 'lower' in logz_dict:
        #     print(f"LogZ uncertainty: ±{(logz_dict['upper'] - logz_dict['lower']) / 2:.4f}")
        print("=" * 60)
        print(f"Runtime: {manual_timing:.2f}s ({manual_timing / 60:.2f} min)")

        reference_samples = loadMCSamples(
            './cosmo_input/chains/Planck_lite_LCDM',
        )

        sample_array = samples['x']
        weights_array = samples['weights']
        BOBE_Samples = MCSamples(
            samples=sample_array,
            names=param_list,
            labels=param_labels,
            weights=weights_array,
            ranges=dict(zip(param_list, param_bounds.T)),
        )

        print("Creating parameter samples plot...")
        sns.set_theme('notebook', 'ticks', palette='husl')
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'

        g = plots.get_subplot_plotter(subplot_size=2.5, subplot_size_ratio=1)
        g.settings.legend_fontsize = 16
        g.settings.axes_fontsize = 16
        g.settings.axes_labelsize = 16
        g.triangle_plot(
            [BOBE_Samples, reference_samples],
            params=['ombh2', 'omch2', 'H0', 'ns', 'logA', 'tau'],
            filled=[True, False],
            contour_colors=['#006FED', 'black'],
            contour_lws=[1, 1.5],
            legend_labels=['BOBE (flow)', 'Nested Sampling'],
        )
        # scatter GP training points
        points = scale_from_unit(gp.train_x, param_bounds)
        for i in range(ndim):
            for j in range(i + 1, ndim):
                ax = g.subplots[j, i]
                ax.scatter(points[:, i], points[:, j], alpha=0.75, color='red', s=4)
        g.export(f'./results/LCDM_Lite/{likelihood_name}_samples.pdf')

        print("\nDETAILED TIMING ANALYSIS")
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
        ax.plot(iterations, values, linestyle='-')
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Acquisition Value')
        plt.savefig(f'./results/LCDM_Lite/{likelihood_name}_acquisition.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()
