import os

#-------------------------#
#--- Environment Setup ---#
#-------------------------#

# Tell JAX about host CPU cores
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(os.cpu_count())
# Force LaTeX to be available for plotting
TEXBIN = "/apps/z_install_tree/linux-rocky8-ivybridge/gcc-12.2.0/texlive-20220321-7ejwbhyks4jxvs4cg6cddeczjlnf2fhi/bin/x86_64-linux/"
os.environ["PATH"] = TEXBIN + ":" + os.environ["PATH"]

import time
import numpy as np
import py21cmfast as p21c
import matplotlib.pyplot as plt
import seaborn as sns
from getdist import MCSamples, plots
from mpi4py import MPI

from likelihood_21cm import (
    PARAMETER_NAMES,
    PARAMETER_LABELS,
    FIDUCIAL_THETA,
    build_fiducial_dataset,
    make_base_inputs,
    make_loglike_function,
    get_param_bounds,
    get_varying_indices,
)

from BOBE import BOBE
from BOBE.utils.core import renormalise_log_weights, scale_from_unit

def main():
    #----------------------------#
    #--- MPI / Runtime config ---#
    #----------------------------#

    # Set up MPI communicator and identify this rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Read runtime settings from the environemt
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    ndim = int(os.environ.get("NDIM", "2"))
    obs_noise = float(os.environ.get("OBS_NOISE", "1e-8"))

    # Paths to staged static input files. These are typically copied to node-local scratch by the submission script
    fiducial_path = os.environ.get(
        "FIDUCIAL_LC_PATH",
        "cache/21cm_test_runs_lcs/fiducial-lightcone.h5",
    )
    ska_path = os.environ.get(
        "SKA_SENSE_PATH",
        "sensitivities/ska_sense.txt",
    )

    print(f"[rank {rank}/{size}] starting BO run", flush=True)
    print(f"[rank {rank}/{size}] OMP_NUM_THREADS = {n_threads}", flush=True)
    print(f"[rank {rank}/{size}] n_concurrent_evals = {size}", flush=True)
    print(f"[rank {rank}/{size}] fiducial_path = {fiducial_path}", flush=True)
    print(f"[rank {rank}/{size}] ska_path = {ska_path}", flush=True)
    #-----------------------------------------------------------------------------#
    #--- Step 1: Define redshift chunks used for the power spectrum likelihood ---#
    #-----------------------------------------------------------------------------#
    
    chunk_z_list_HERA = [
        27.4, 23.4828, 20.5152, 18.1892, 16.3171, 14.7778, 13.4898, 12.3962,
        11.4561, 10.6393, 9.92308, 9.28986, 8.72603, 8.22078, 7.76543,
        7.35294, 6.97753, 6.63441, 6.31959, 6.0297, 5.7619, 5.51376
    ]
    #-----------------------------------------------------------------#
    #--- Step 2: Build the fiducial dataset used by the likelihood ---#
    #-----------------------------------------------------------------#
    # This helper:
    #     - Loads the saved fiducial lightcone
    #     - Computes the fiducial chunked power spectra
    #     - Loads the sensitivity file
    #     - Applies the z/k masking used by the likelihood
    setup = build_fiducial_dataset(
        fiducial_path=fiducial_path,
        sensitivity_path=ska_path,
        chunk_z_list=chunk_z_list_HERA,
        n_psbins=47,
        k_min_ps=3.337118317301632e-02,
        k_max_ps=2.675685850887854e+00,
        z_min=6.0,
        z_max=30.0,
        k_min=0.1,
        k_max=1.0,
    )

    dataset = setup["dataset"]
    chunk_indices_HERA = setup["chunk_indices"]

    #------------------------------------------------------------#
    #--- Step 3: Construct the baseline 21cmFAST input object ---#
    #------------------------------------------------------------#
    # This uses:
    #    - The Park19 template
    #    - The Park et al. fiducial parameter values
    #    - A fixed random seed for reproducibility
    #    - The OpenMP thread count from the environment
    inputs = make_base_inputs(n_threads=n_threads, random_seed=1234)
    
    #-----------------------------------------------------#
    #--- Step 4: Build a rank-specific cache directory ---#
    #-----------------------------------------------------#
    # Each MPI rank gets its own cache directory so that concurrent evaluations do not clash
    
    job_id = os.environ.get("PBS_JOBID", "nojid")
    cache_dir = f"/srv/scratch/cosmo/21CmTests/cache/21cm_test_runs_lcs_{job_id}_rank{rank}"
    os.makedirs(cache_dir, exist_ok=True)
    cache = p21c.OutputCache(cache_dir)
    print(f"[rank {rank}/{size}] using cache dir: {cache_dir}", flush=True)
    
    #--------------------------------------------------#
    #--- Step 5: Define the full parameter metadata ---#
    #--------------------------------------------------#
    # These are imported from likelihood_21cm.py so that the analysis choices all live in the same place
    
    parameter_names = PARAMETER_NAMES
    parameter_labels = PARAMETER_LABELS
    fiducial_theta = FIDUCIAL_THETA
    
    # Build the full parameter bounds as fiducial ± nsimga
    # Here nsigma=1 corresponds to a 1-sigma box around the Park et al. fiducial point (Table 2 of the paper)
    nsigma = 3.0
    param_bounds = get_param_bounds(nsigma)

    #--------------------------------------------------#
    #--- Step 6: Build the full likelihood function ---#
    #--------------------------------------------------#
    #This returns a function of the full 8D parameter vector

    loglike = make_loglike_function(
        dataset=dataset,
        base_inputs=inputs,
        cache=cache,
        lightcone_quantities=("brightness_temp",),
        chunk_indices=chunk_indices_HERA,
        n_psbins=47,
        k_min_ps=3.337118317301632e-02,
        k_max_ps=2.675685850887854e00,
        parameter_names=parameter_names,
    )

    #-------------------------------------------------------------#
    #--- Step 7: Choose which parameters to vary based on NDIM ---#
    #-------------------------------------------------------------#
    # The convention used here is to work backwards through the parameter list
    # Doesn't really matter because we plan on going 1D -> 2D -> 8D
    
    varying_indices = get_varying_indices(ndim, parameter_names)
    varying_names = [parameter_names[i] for i in varying_indices]
    varying_labels = [parameter_labels[i] for i in varying_indices]
    param_bounds_nd = param_bounds[:, varying_indices]
    fiducial_nd = fiducial_theta[varying_indices]

    #-------------------------------------------------------------------------#
    #--- Step 8: Wrap the full likelihood as an NDIM-restricted likelihood ---#
    #-------------------------------------------------------------------------#
    # BOBE only sees the active n-D subspace. The inactive parameters are held fixed at their fiducial values.
    
    def expand_theta_nd(theta_nd):
        theta_full = fiducial_theta.copy()
        theta_full[varying_indices] = theta_nd
        return theta_full
    
    def loglike_nd(theta_nd):
        return loglike(expand_theta_nd(theta_nd))
    
    print(f"[rank {rank}/{size}] varying_names = {varying_names}", flush=True)
    print(f"[rank {rank}/{size}] varying_indices = {varying_indices}", flush=True)
    print(f"[rank {rank}/{size}] param_bounds_2d =\n{param_bounds_nd}", flush=True)
    print(f"[rank {rank}/{size}] fiducial 2D point = {fiducial_theta[varying_indices]}", flush=True)

    #--------------------------------------------------------#
    #--- Step 9: Define a descriptive likelihood/run name ---#
    #--------------------------------------------------------#

    likelihood_name = f"21cmTest_{ndim}D_z01_{int(nsigma)}sigma_{float(obs_noise)}noise_{size}Ranks_{n_threads}Threads"

    start = time.time()
    print(f"[rank {rank}/{size}] Starting BOBE run...", flush=True)

    #----------------------------------------#
    #--- Step 10: Initialise and run BOBE ---#
    #----------------------------------------#
    
    bobe = BOBE(
        loglikelihood=loglike_nd,
        likelihood_name=likelihood_name,
        param_bounds=param_bounds_nd,
        param_list=varying_names,
        param_labels=varying_labels,
        confidence_for_unbounded=0.9999995,
        resume=False,
        resume_file=f'./results/{likelihood_name}',
        save_dir='./results/',
        save=True,
        verbosity='INFO',
        n_sobol_init=0,
        use_clf=False,
        clf_type='svm',
        minus_inf=-1e10,
        seed=10,
        gp_kwargs={'noise': obs_noise}
    )
    
    results = bobe.run(
        acq='wipstd',
        min_evals=10, 
        max_evals=50,
        max_gp_size=50,
        fit_n_points=size, 
        ns_n_points=size,
        batch_size=size,
        num_hmc_warmup=512,
        num_hmc_samples=2048, 
        mc_points_size=1024,
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
        g.export(f'./results/{likelihood.name}_samples.pdf')

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
        plt.savefig(f"./results/{likelihood.name}_acquisition.pdf", bbox_inches='tight')

if __name__ == "__main__":
    main()
