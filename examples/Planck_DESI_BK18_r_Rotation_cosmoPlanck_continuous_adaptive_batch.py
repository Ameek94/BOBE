"""
Planck (CamSpec TTTEEE) + Planck lensing + BICEP/Keck 2018 + DESI DR2 LCDM+r
script with continuous GP parameter rotation applied to cosmological and Planck
nuisance parameters.  Only the tensor-to-scalar ratio r is kept in a simple
linear (axis-aligned) unit-cube mapping.

Uses adaptive batch sizing: effective batch grows from min_batch_size toward
batch_size as the acquisition value approaches the convergence threshold.

The active parameters (rotation applied) are:
    omch2, ombh2, H0, logA, ns, tau,
    A_planck, amp_143, amp_217, amp_143x217,
    n_143, n_217, n_143x217, calTE, calEE
Only r uses linear scaling.
"""

import os
import sys

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    os.cpu_count()
)

from BOBE import BOBE, CobayaLikelihood
from BOBE.transforms import RotationTransform
import time
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
import seaborn as sns
from getdist import MCSamples, plots, loadMCSamples
import numpy as np

# Parameters to which the rotation is applied: all cosmo + Planck nuisance.
# Only r (BK18) remains linear.
PLANCK_DESI_BK18_R_COSMO_PARAMS = [
    'omch2', 'ombh2', 'H0', 'logA', 'ns', 'tau',
    'A_planck', 'amp_143', 'amp_217', 'amp_143x217',
    'n_143', 'n_217', 'n_143x217', 'calTE', 'calEE',
]


def main():

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42

    cobaya_input_file = './cosmo_input/Planck_DESI_LCDM_r.yaml'
    likelihood_name = f'Planck_DESI_BK18_r_Rotation_cosmoPlanck_continuous_adaptive_batch_{seed}'

    print("Loading reference samples...")
    reference_samples = loadMCSamples(
        './cosmo_input/chains/Planck_DESI_BK18_r/Planck_DESI_BK18',
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
                   for p in PLANCK_DESI_BK18_R_COSMO_PARAMS if p in likelihood.param_list]
    print(f"param_list      : {likelihood.param_list}")
    print(f"active dims     : {active_dims}")
    print(f"active params   : {[likelihood.param_list[i] for i in active_dims]}")
    print(f"inactive params : {[p for p in likelihood.param_list if p not in PLANCK_DESI_BK18_R_COSMO_PARAMS]}")

    transform = RotationTransform(
        likelihood.param_bounds,
        active_dims=active_dims,
        kl_threshold=0.5,
        max_updates=10,
        update_step=25,
    )

    start = time.time()
    print("\n" + "="*80)
    print("Starting BOBE run WITH cosmological-parameter rotation (adaptive batch)...")
    print("="*80)

    bobe = BOBE(
        loglikelihood=likelihood,
        resume=True,
        resume_file=f'./results/Planck_DESI_BK18_r/{likelihood_name}',
        save_dir='./results/Planck_DESI_BK18_r/',
        save=True,
        verbosity='INFO',
        n_cobaya_init=16,
        n_sobol_init=64,
        use_clf=True,
        clf_type='svm',
        seed=seed,
        gp_kwargs={'lengthscale_bounds': (0.01, 100), 'kernel_variance_bounds': (1e-4, 1e4)},
        transform=transform,
    )

    bobe.run(
        acq='logei',
        min_evals=50,
        max_evals=200,
    )

    results = bobe.run(
        acq='wipstd',
        min_evals=400,
        max_evals=4000,
        max_gp_size=1800,
        fit_n_points=40,
        ns_n_points=25,
        batch_size=8,
        num_hmc_warmup=512,
        num_hmc_samples=8000,
        mc_points_size=512,
        logz_threshold=0.5,
        num_chains=8,
        convergence_n_iters=2,
        do_final_ns=False,
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

        print("\nCreating cosmology parameter samples plot...")
        sns.set_theme('notebook', 'ticks', palette='husl')
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'sans-serif'

        cosmo_params = ['r', 'omch2', 'ombh2', 'H0', 'logA', 'ns', 'tau']
        g = plots.get_subplot_plotter(subplot_size=2.5, subplot_size_ratio=1)
        g.settings.legend_fontsize = 16
        g.settings.axes_fontsize = 16
        g.settings.axes_labelsize = 16
        g.triangle_plot(
            [BOBE_Samples, reference_samples],
            params=cosmo_params,
            filled=[True, False],
            contour_colors=['#006FED', 'black'],
            contour_lws=[1, 1.5],
            legend_labels=['BOBE (cosmo rotation, adaptive batch)', 'MCMC'],
        )
        g.export(f'./results/Planck_DESI_BK18_r/{likelihood_name}_cosmo_samples.pdf')
        print(f"Saved plot to ./results/Planck_DESI_BK18_r/{likelihood_name}_cosmo_samples.pdf")

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
        g.export(f'./results/Planck_DESI_BK18_r/{likelihood_name}_full_samples.pdf')
        print(f"Saved plot to ./results/Planck_DESI_BK18_r/{likelihood_name}_full_samples.pdf")

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
        plt.savefig(f"./results/Planck_DESI_BK18_r/{likelihood_name}_acquisition.pdf", bbox_inches='tight')
        print(f"Saved acquisition plot to ./results/Planck_DESI_BK18_r/{likelihood_name}_acquisition.pdf")

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
