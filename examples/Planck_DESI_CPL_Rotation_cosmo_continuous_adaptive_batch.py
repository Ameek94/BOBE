"""
CPL script with continuous GP parameter rotation applied to cosmological
parameters only.  Planck + DESI + Union3 nuisance parameters (CamSpec
calibration) are kept in a simple linear (axis-aligned) unit-cube mapping.

Uses adaptive batch sizing: effective batch grows from min_batch_size toward
batch_size as the acquisition value approaches the convergence threshold.

The active cosmological parameters are:
    w, wa, omch2, ombh2, H0, logA, ns, tau
All remaining sampled parameters use linear scaling.
"""

import os
import sys

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    os.cpu_count()
)

from BOBE import BOBE, CobayaLikelihood
from BOBE.transforms import RotationTransform
from BOBE.utils.core import scale_from_unit
import time
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
import seaborn as sns
from getdist import MCSamples, plots, loadMCSamples
import numpy as np

# Cosmological parameters to which the rotation is applied.
# All other sampled parameters (nuisance / calibration) remain linear.
CPL_COSMO_PARAMS = ['w', 'wa', 'omch2', 'ombh2', 'H0', 'logA', 'ns', 'tau']


def main():

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42

    cobaya_input_file = './cosmo_input/CPL_Planck_DESI.yaml'
    likelihood_name = f'Planck_DESI_CPL_Rotation_cosmo_continuous_adaptive_batch_{seed}'

    print("Loading reference samples...")
    reference_samples = loadMCSamples(
        './cosmo_input/chains/union3_CPL',
        settings={'ignore_rows': 0.3, 'label': 'MCMC'}
    )

    # Build the likelihood first so we can read param_list and param_bounds
    # before constructing the transform.
    print("Building Cobaya likelihood...")
    likelihood = CobayaLikelihood(
        cobaya_input_file,
        confidence_for_unbounded=0.9999995,
        minus_inf=-1e5,
        name=likelihood_name,
    )

    active_dims = [likelihood.param_list.index(p)
                   for p in CPL_COSMO_PARAMS if p in likelihood.param_list]
    print(f"param_list      : {likelihood.param_list}")
    print(f"active dims     : {active_dims}")
    print(f"active params   : {[likelihood.param_list[i] for i in active_dims]}")
    print(f"inactive params : {[p for p in likelihood.param_list if p not in CPL_COSMO_PARAMS]}")

    transform = RotationTransform(
        likelihood.param_bounds,
        active_dims=active_dims,
        kl_threshold=0.5,
        max_updates=15,
        update_step=25,
    )

    start = time.time()
    print("\n" + "="*80)
    print("Starting BOBE run WITH cosmological-parameter rotation (adaptive batch)...")
    print("="*80)

    bobe = BOBE(
        loglikelihood=likelihood,
        resume=True,
        resume_file=f'./results/CPL/{likelihood_name}',
        save_dir='./results/CPL/',
        save=True,
        verbosity='INFO',
        n_cobaya_init=16,
        n_sobol_init=32,
        use_clf=True,
        clf_type='svm',
        seed=seed,
        gp_kwargs = {'lengthscale_bounds': [0.01, 100.0], 'kernel_variance_bounds': [1e-4, 1e4]},
        transform=transform,
    )

    results = bobe.run(
        acq='logei',
        min_evals=100,
        max_evals=300,
        max_gp_size=1500,
        fit_n_points=25,
        ns_n_points=25,
        batch_size=5,
        do_final_ns=False,
    )

    results = bobe.run(
        acq='wipstd',
        min_evals=600,
        max_evals=4000,
        max_gp_size=1600,
        fit_n_points=30,
        ns_n_points=30,
        batch_size=8,
        convergence_n_iters=2,
        num_hmc_warmup=512,
        num_hmc_samples=8000,
        mc_points_size=512,
        logz_threshold=0.4,
        num_chains=8,
        do_final_ns=False,
        transform_acq_threshold=4.0,
        adaptive_batch=True,
        min_batch_size=2,
    )

    end = time.time()

    if results is not None:

        gp = results['gp']
        results_manager = results['results_manager']
        samples = results['samples']
        param_bounds = likelihood.param_bounds
        param_list = likelihood.param_list
        param_labels = likelihood.param_labels

        manual_timing = end - start

        print("\n" + "="*80)
        print("RUN COMPLETED WITH COSMO ROTATION (ADAPTIVE BATCH)")
        print("="*80)
        print(f"Total runtime: {manual_timing:.2f} seconds ({manual_timing/60:.2f} minutes)")
        print(f"Number of GP training points: {gp.train_x.shape[0]}")

        sample_array = samples['x']
        weights_array = samples['weights']
        BOBE_Samples = MCSamples(
            samples=sample_array,
            names=param_list,
            labels=param_labels,
            weights=weights_array,
            ranges=dict(zip(param_list, param_bounds.T))
        )

        print("\nCreating parameter samples plot...")
        sns.set_theme('notebook', 'ticks', palette='husl')
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'sans-serif'

        g = plots.get_subplot_plotter(subplot_size=2.5, subplot_size_ratio=1)
        g.settings.legend_fontsize = 16
        g.settings.axes_fontsize = 16
        g.settings.axes_labelsize = 16
        g.triangle_plot(
            [BOBE_Samples, reference_samples],
            params=['w', 'wa', 'ombh2', 'omch2', 'H0', 'ns', 'logA', 'tau'],
            filled=[True, False],
            contour_colors=['#006FED', 'black'],
            contour_lws=[1, 1.5],
            legend_labels=['BOBE (cosmo rotation, adaptive batch)', 'MCMC'],
        )
        g.export(f'./results/CPL/{likelihood_name}_cosmo_samples.pdf')
        print(f"Saved plot to ./results/CPL/{likelihood_name}_cosmo_samples.pdf")

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
            legend_labels=['BOBE (cosmo rotation, adaptive batch)', 'MCMC'],
        )
        g.export(f'./results/CPL/{likelihood_name}_full_samples.pdf')
        print(f"Saved plot to ./results/CPL/{likelihood_name}_full_samples.pdf")

        print("\n" + "="*80)
        print("DETAILED TIMING ANALYSIS")
        print("="*80)

        timing_data = results_manager.get_timing_summary()
        print(f"Total runtime: {timing_data['total_runtime']:.2f} seconds ({timing_data['total_runtime']/60:.2f} minutes)")
        print("\nPhase Breakdown:")
        print("-" * 50)
        for phase, time_spent in timing_data['phase_times'].items():
            if time_spent > 0:
                percentage = timing_data['percentages'].get(phase, 0)
                print(f"{phase:30s}: {time_spent:8.2f}s ({percentage:5.1f}%)")

        acquisition_data = results_manager.get_acquisition_data()
        iterations = np.array(acquisition_data['iterations'])
        values = np.array(acquisition_data['values'])

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(iterations, values, linestyle='-', marker='o', markersize=3)
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Acquisition Value')
        ax.set_title('Acquisition Function Values (cosmo rotation, adaptive batch)')
        ax.grid(True, alpha=0.3)
        plt.savefig(f"./results/CPL/{likelihood_name}_acquisition.pdf", bbox_inches='tight')
        print(f"Saved acquisition plot to ./results/CPL/{likelihood_name}_acquisition.pdf")

        print("\n" + "="*80)
        print("GP HYPERPARAMETERS")
        print("="*80)
        print(f"Lengthscales: {gp.lengthscales}")
        print(f"Kernel variance: {gp.kernel_variance:.4f}")
        print(f"Noise: {gp.noise:.2e}")

        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)


if __name__ == "__main__":
    main()
