"""
Planck + DESI LCDM run with a *pretrained* Normalising Flow transform.

A Masked Autoregressive Flow (MAF) is trained on reference MCMC chains
*before* the BOBE run starts.  The pretrained flow is used as a fixed
transform throughout the entire run — no in-run updates are performed.

Workflow
--------
1. Load reference MCMC samples.
2. Build param_bounds from the Cobaya YAML (via ``CobayaLikelihood``).
3. Create a ``NormalisingFlowTransform`` with ``max_updates=0``.
4. Call ``transform.pretrain(ref_phys_samples)`` to train and activate the flow.
5. Pass the pre-built transform directly to ``BOBE``.
"""

import os
import sys

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    os.cpu_count()
)

from BOBE import BOBE
from BOBE.likelihood import CobayaLikelihood
from BOBE.transforms import NormalisingFlowTransform
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
    likelihood_name = f'Planck_DESI_LCDM_Flow_pretrained_{seed}'
    confidence_for_unbounded = 0.9999995

    # ------------------------------------------------------------------
    # 1. Load reference MCMC samples
    # ------------------------------------------------------------------
    print("Loading reference samples...")
    reference_samples = loadMCSamples(
        './cosmo_input/chains/Planck_DESIDr2_LCDM_MCMC',
        settings={'ignore_rows': 0.3, 'label': 'MCMC'},
    )

    # ------------------------------------------------------------------
    # 2. Extract param_bounds and param ordering from the Cobaya YAML.
    #    CobayaLikelihood is used here only for its metadata; the same
    #    object is NOT reused so that BOBE can manage its own instance.
    # ------------------------------------------------------------------
    print("Extracting parameter metadata from Cobaya YAML...")
    _meta = CobayaLikelihood(
        cobaya_input_file,
        confidence_for_unbounded=confidence_for_unbounded,
        name='meta',
    )
    param_list_ordered = _meta.param_list
    param_bounds = _meta.param_bounds     # shape (2, D)
    del _meta                             # free Cobaya resources

    # ------------------------------------------------------------------
    # 3. Align reference samples with the BOBE parameter ordering and
    #    extract physical (non-standardised) values.
    # ------------------------------------------------------------------
    ref_param_names = reference_samples.getParamNames().list()
    try:
        col_indices = [ref_param_names.index(p) for p in param_list_ordered]
    except ValueError as exc:
        raise RuntimeError(
            f"Parameter mismatch between Cobaya YAML and reference chains: {exc}"
        ) from exc

    ref_phys = reference_samples.samples[:, col_indices]   # (N, D), physical space
    print(f"Reference samples shape: {ref_phys.shape}")
    thinned_ref_phys = ref_phys[::5]  # thin by factor of 5 for faster pretraining

    # ------------------------------------------------------------------
    # 4. Build and pretrain the NormalisingFlowTransform.
    #    max_updates=0 ensures no further updates during the BOBE run.
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Pre-training normalising flow on reference samples...")
    print("=" * 80)
    transform = NormalisingFlowTransform(
        param_bounds,
        n_sigma=5.0,
        max_updates=0,          # lock: no updates during BOBE run
        n_layers=8,
        hidden_dim=64,
        flow_n_epochs=2000,
        use_rotation_precon=False,
        seed=seed,
    )
    transform.pretrain(thinned_ref_phys)

    # ------------------------------------------------------------------
    # 5. Run BOBE with the pretrained transform.
    # ------------------------------------------------------------------
    start = time.time()
    print("\n" + "=" * 80)
    print("Starting BOBE run WITH pretrained Normalising Flow transform...")
    print("=" * 80)

    bobe = BOBE(
        loglikelihood=cobaya_input_file,
        likelihood_name=likelihood_name,
        confidence_for_unbounded=confidence_for_unbounded,
        resume=True,
        resume_file=f'./results/LCDM/{likelihood_name}',
        save_dir='./results/LCDM/',
        save=True,
        verbosity='INFO',
        n_cobaya_init=8,
        n_sobol_init=32,
        use_clf=True,
        clf_type='svm',
        minus_inf=-1e5,
        seed=seed,
        # Pass the pre-built, pre-trained transform instance directly.
        # BOBE will use it as-is without any further updates.
        transform=transform,
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
        print("RUN COMPLETED WITH PRETRAINED NORMALISING FLOW TRANSFORM")
        print("=" * 80)
        print(f"Total runtime: {manual_timing:.2f}s ({manual_timing / 60:.2f} min)")
        print(f"Number of GP training points: {gp.train_x.shape[0]}")
        print(f"Flow update count (expected 0): {bobe.transform.update_count}")

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
            legend_labels=['BOBE (pretrained flow)', 'MCMC'],
        )
        g.export(f'./results/LCDM/{likelihood_name}_cosmo_samples.pdf')

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
            legend_labels=['BOBE (pretrained flow)', 'MCMC'],
        )
        g.export(f'./results/LCDM/{likelihood_name}_full_samples.pdf')

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
        ax.set_title('Acquisition Function Values (pretrained flow transform)')
        ax.grid(True, alpha=0.3)
        plt.savefig(f'./results/LCDM/{likelihood_name}_acquisition.pdf', bbox_inches='tight')

        print(f"\nGP Lengthscales: {gp.lengthscales}")
        print(f"Kernel variance: {gp.kernel_variance:.4f}")

        print("\n" + "=" * 80)
        print("RUN COMPLETED SUCCESSFULLY")
        print("=" * 80)


if __name__ == "__main__":
    main()
