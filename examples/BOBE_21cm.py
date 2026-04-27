import os

#-------------------------#
#--- Environment Setup ---#
#-------------------------#

# Tell JAX about host CPU cores
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(os.cpu_count())

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from getdist import MCSamples, plots
from mpi4py import MPI
from likelihood_21cmFAST import Likelihood21cmFAST

def main():
    # ----------------------------
    # MPI / runtime setup
    # ----------------------------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    fiducial_path = os.environ.get(
        "FIDUCIAL_LC_PATH",
        "cache/21cm_test_runs_lcs/fiducial-lightcone.h5",
    )
    sensitivity_path = os.environ.get(
        "SKA_SENSE_PATH",
        "sensitivities/ska_sense.txt",
    )
    job_id = os.environ.get("PBS_JOBID", "nojid")
    
    # Rank-specific cache directory so concurrent evaluations do not clash
    cache_dir = f"/srv/scratch/cosmo/21CmTests/cache/21cm_{job_id}_rank{rank}"
    os.makedirs(cache_dir, exist_ok=True)

    # Optional debug output per rank
    debug_dir = "/srv/scratch/cosmo/21CmTests/ps_compare_outputs"
    os.makedirs(debug_dir, exist_ok=True)
    debug_output_file = os.path.join(
        debug_dir,
        f"ps_compare_{job_id}_rank{rank}.txt"
    )

    print(f"[rank {rank}/{size}] starting run", flush=True)
    print(f"[rank {rank}/{size}] OMP_NUM_THREADS = {n_threads}", flush=True)
    print(f"[rank {rank}/{size}] fiducial_path = {fiducial_path}", flush=True)
    print(f"[rank {rank}/{size}] sensitivity_path = {sensitivity_path}", flush=True)
    print(f"[rank {rank}/{size}] cache_dir = {cache_dir}", flush=True)

    #-----------------------------#
    # Construct likelihood object #
    #-----------------------------#
    like = Likelihood21cmFAST(
        fiducial_path=fiducial_path,
        sensitivity_path=sensitivity_path,
        cache_dir=cache_dir,
        n_threads=n_threads,
        random_seed=1234,
        include_norm=False,
        debug_output_file=None,   # Set to an output file to record components that go into calculating the likelihood
        param_bounds=None,        # pass custom bounds here
        nsigma_bounds=5.0,
    )

    loglike = like.loglike
    param_names = like.param_names
    param_labels = like.param_labels
    param_bounds = like.param_bounds
    fiducial_theta = like.fiducial_theta
    likelihood_name = f"21cmTest_{ndim}D_z01_{nsigma}_sigma_learn_noise_clf_no_norm_{size}Ranks_{n_threads}Threads"
    start = time.time()
    print(f"[rank {rank}/{size}] Starting BOBE run...", flush=True)

    #----------------------------------------#
    #--- Step 10: Initialise and run BOBE ---#
    #----------------------------------------#
    
    bobe = BOBE(
        loglikelihood=loglike,
        likelihood_name=likelihood_name,
        param_bounds=param_bounds,
        param_list=varying_names,
        param_labels=param_labels,
        confidence_for_unbounded=0.9999995,
        resume=False,
        resume_file=f'./results/Learned_noise_BO_prior_range/{likelihood_name}',
        save_dir='./results/Learned_noise_BO_prior_range/',
        save=True,
        save_step=1,
        verbosity='INFO',
        n_sobol_init=8,
        use_clf=True,
        clf_use_size=50,
        clf_nsigma_threshold=20,
        clf_type='svm',
        minus_inf=-1e10,
        seed=10,
        gp_kwargs={'noise_prior': 'learn'},
        #init_train_x =[ -1.3, 0.5, -1., -0.5, 8.7, 0.5, 40.5, 500.],
        #init_train_y =[-1279.3320449933374],
    )
    
    results = bobe.run(
        acq='wipstd',
        min_evals=50, 
        max_evals=500,
        max_gp_size=500,
        fit_n_points=1, 
        ns_n_points=size,
        batch_size=size,
        num_hmc_warmup=256,
        num_hmc_samples=1024, 
        mc_points_size=512,
        num_chains=8,   
        logz_threshold=1e-1,
        do_final_ns=True,
    )

    end = time.time()

    if results is not None:  # when running in MPI mode, only rank 0 returns results, rest return None

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

        print("\n" + "="*60)
        print("RUN COMPLETED")
        print(f"Final LogZ: {logz_dict.get('mean', 'N/A'):.4f}")
        if 'upper' in logz_dict and 'lower' in logz_dict:
            print(f"LogZ uncertainty: ±{(logz_dict['upper'] - logz_dict['lower'])/2:.4f}")

        print("="*60)
        print(f"Manual timing: {manual_timing:.2f} seconds ({manual_timing/60:.2f} minutes)")

        # Create MCSamples from BOBE results
        sample_array = samples['x']
        weights_array = samples['weights']
        BOBE_Samples = MCSamples(samples=sample_array, names=param_list, labels=param_labels,
                                    weights=weights_array, 
                                    ranges= dict(zip(param_list,param_bounds.T)))

        # Create parameter samples plot
        print("Creating parameter samples plot...")
        sns.set_theme('notebook', 'ticks', palette='husl')
        plt.rcParams['text.usetex'] = True # optional for LaTeX-style text rendering
        plt.rcParams['font.family'] = 'serif'

        g = plots.get_subplot_plotter(subplot_size=2.5, subplot_size_ratio=1)
        g.settings.legend_fontsize = 16
        g.settings.axes_fontsize = 16
        g.settings.axes_labelsize = 16
        g.triangle_plot([BOBE_Samples], params=param_list,
                        filled=[True, False],
                    contour_colors=['#006FED', 'black'], contour_lws=[1, 1.5],
                    legend_labels=['BOBE', 'Nested Sampling'],) 
        # add scatter points for gp training data
        points = scale_from_unit(gp.train_x, param_bounds)
        for i in range(ndim):
            for j in range(i+1, ndim):
                ax = g.subplots[j, i]
                ax.scatter(points[:, i], points[:, j], alpha=0.75, color='red', s=4)
        g.export(f'./results/Learned_noise_BO_prior_range/{likelihood.name}_samples.pdf')

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
        plt.savefig(f"./results/Learned_noise_BO_prior_range/{likelihood.name}_acquisition.pdf", bbox_inches='tight')

if __name__ == "__main__":
    main()