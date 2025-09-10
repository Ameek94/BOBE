import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    os.cpu_count()
)
import sys
from jaxbo.utils.summary_plots import plot_final_samples, BOBESummaryPlotter
import time
import matplotlib.pyplot as plt
import seaborn as sns
from jaxbo.utils.logging_utils import get_logger
from jaxbo.run import run_bobe

def main():
    # Set up the cosmological likelihood
    cobaya_input_file = './cosmo_input/CPL_Planck_DESI_Union3.yaml'
    ei_method = str(sys.argv[1]) if len(sys.argv) > 1 else 'logei'
    clf_type = str(sys.argv[2]) if len(sys.argv) > 2 else 'svm'
    ls_prior = str(sys.argv[3]) if len(sys.argv) > 3 else 'DSLP'
    kernel_var_prior = str(sys.argv[4]) if len(sys.argv) > 4 else {'name': 'LogNormal', 'loc': 0.0, 'scale': 0.1}
    kernel_var_prior_str = kernel_var_prior['name'] if isinstance(kernel_var_prior, dict) else str(kernel_var_prior)
    likelihood_name = f'Planck_DESI_U3_CPL_{ei_method}_{ls_prior}_{kernel_var_prior_str}_{clf_type}'

    start = time.time()
    print("Starting BOBE run...")

    results = run_bobe(
        likelihood=cobaya_input_file,
        likelihood_kwargs={
            'confidence_for_unbounded': 0.9999995,
            'minus_inf': -1e5,
            'noise_std': 0.0,
            'name': likelihood_name,
        },
        gp_kwargs={'lengthscale_prior': ls_prior, 'kernel_variance_prior': kernel_var_prior},
        acq=ei_method,
        verbosity='INFO',
        n_cobaya_init=0,
        n_sobol_init=64,
        min_evals=200,
        max_evals=750,
        max_gp_size=750,
        fit_step=5,
        use_clf=True,
        clf_type=clf_type,
        minus_inf=-1e5,
        ei_goal = 1e-20,
        convergence_n_iters=2,
        zeta_ei = 0.01,
        logz_threshold=1e-3,
        seed=100000,
        resume=False,
        resume_file=f'{likelihood_name}',
        save_dir='./results',
        save=True,
    )


    end = time.time()

    if results is not None:
        log = get_logger("main")
        manual_timing = end - start

        log.info("\n" + "="*60)
        log.info("RUN COMPLETED")
        log.info("="*60)
        log.info(f"Manual timing: {manual_timing:.2f} seconds ({manual_timing/60:.2f} minutes)")

        results_manager = results['results_manager']

        best_loglike_data = results_manager.get_best_loglike_data()
        acquisition_data = results_manager.get_acquisition_data()

        plotter = BOBESummaryPlotter(results_manager)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        plotter.plot_best_loglike_evolution(best_loglike_data=best_loglike_data, ax=ax[0])
        plotter.plot_acquisition_evolution(
            acquisition_data=acquisition_data, ax=ax[1]
        )
        # print(acquisition_data)
        # plt.show()
        plt.savefig(f'{likelihood_name}_run_summary.pdf',bbox_inches='tight')

if __name__ == "__main__":
    # Run the analysis
    main()