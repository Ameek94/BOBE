import os
# Tell JAX about host CPU cores
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    os.cpu_count()
)

import time
import numpy as np
import py21cmfast as p21c

# Import likelihood functions
from likelihood_21cm import (
    chunk_indices,
    powerspectra_chunks,
    build_mock_dataset,
    make_loglike_function,
)

from BOBE import BOBE
from BOBE.utils.core import renormalise_log_weights, scale_from_unit
import time
import matplotlib.pyplot as plt
import seaborn as sns # optional for improved plot aesthetics
from getdist import MCSamples, plots
import numpy as np
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get number of threads from environemnt (requires it to be set in the submission script).
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    # Get the path of the fiducial lightcone (requires it to be set in the submission script).
    fiducial_path = os.environ.get(
        "FIDUCIAL_LC_PATH",
        "cache/21cm_test_runs_lcs/fiducial-lightcone.h5",
    )
    # Get the path of the SKA sensitivities (requires it to be set in the submission script).
    ska_path = os.environ.get(
        "SKA_SENSE_PATH",
        "sensitivities/ska_sense.txt",
    )

    print(f"[rank {rank}/{size}] starting BO run", flush=True)
    print(f"[rank {rank}/{size}] OMP_NUM_THREADS = {n_threads}", flush=True)
    print(f"[rank {rank}/{size}] n_concurrent_evals = {size}", flush=True)
    print(f"[rank {rank}/{size}] fiducial_path = {fiducial_path}", flush=True)
    print(f"[rank {rank}/{size}] ska_path = {ska_path}", flush=True)

    # 1. Read saved fiducial lightcone
    lightcone_fiducial = p21c.LightCone.from_file(path=fiducial_path)
    
    # 2. Define redshift chunks
    chunk_z_list_HERA = [
        27.4, 23.4828, 20.5152, 18.1892, 16.3171, 14.7778, 13.4898, 12.3962,
        11.4561, 10.6393, 9.92308, 9.28986, 8.72603, 8.22078, 7.76543,
        7.35294, 6.97753, 6.63441, 6.31959, 6.0297, 5.7619, 5.51376
    ]
    chunk_indices_HERA = chunk_indices(lightcone_fiducial, chunk_z_list_HERA)
    
    # 3. Compute fiducial power spectrum
    chunk_redshifts_fiducial, data_fiducial, _ = powerspectra_chunks(
        lightcone_fiducial,
        chunk_indices=chunk_indices_HERA,
        n_psbins=47,
        k_min=3.337118317301632e-02,
        k_max=2.675685850887854e+00,
        remove_nans=False,
    )
    
    fiducial_ps = np.array([chunk["delta"] for chunk in data_fiducial])
    fiducial_k_list = data_fiducial[0]["k"]
    fiducial_ps_z = chunk_redshifts_fiducial
    
    # 4. Load sensitivity
    ska_sensitivity = np.loadtxt(ska_path)
    ska_sensitivity = ska_sensitivity[:-2, :]
    
    # 5. Build dataset
    dataset = build_mock_dataset(
        fiducial_ps=fiducial_ps,
        redshifts=fiducial_ps_z,
        k_values=fiducial_k_list,
        sensitivity=ska_sensitivity,
        z_min=6.0,
        z_max=30.0,
        k_min=0.1,
        k_max=1.0,
    )
    
    # 6. Base inputs
    inputs = p21c.InputParameters.from_template("Park19", random_seed=1234)
    inputs = inputs.evolve_input_structs(
        F_STAR10=-1.3,
        ALPHA_STAR=0.5,
        F_ESC10=-1.0,
        ALPHA_ESC=-0.5,
        M_TURN=8.7,
        t_STAR=0.5,
        L_X=40.5,
        NU_X_THRESH=500.0,
        N_THREADS=n_threads,
    )
    
    # 7. Per-rank cache - we do this to mitigate slowdowns from all ranks trying to read the same file in at the same time
    job_id = os.environ.get("PBS_JOBID", "nojid")
    cache_dir = f"/srv/scratch/cosmo/21CmTests/cache/21cm_test_runs_lcs_{job_id}_rank{rank}"
    os.makedirs(cache_dir, exist_ok=True)
    cache = p21c.OutputCache(cache_dir)
    print(f"[rank {rank}/{size}] using cache dir: {cache_dir}", flush=True)
        
    # 8. Parameter names and bounds
    parameter_names = [
    "F_STAR10",
    "ALPHA_STAR",
    "F_ESC10",
    "ALPHA_ESC",
    "M_TURN",
    "t_STAR",
    "L_X",
    "NU_X_THRESH", #eV in code, keV in paper
    ]

    parameter_labels = [
    r"\log_{10}(f_{*,10})",
    r"\alpha_*",
    r"\log_{10}(f_{\mathrm{esc},10})",
    r"\alpha_{\mathrm{esc}}",
    r"\log_{10}(M_{\mathrm{turn}})",
    r"t_*",
    r"\log_{10}\!\left(\frac{L_{X<2\,\mathrm{keV}}}{\mathrm{SFR}}\right)",
    r"E_0",
    ]
    # 21cm only Fiducial value ± 5 sigma (Fiducial value from Table 2,  sigma from row 3: https://arxiv.org/pdf/1809.08995)
    param_bounds = np.array([
    [-2.35, -1.05, -2.05, -1.85,  7.40, -0.20, 40.15, 300.0],
    [-0.40,  1.65,  0.20,  0.80, 10.05,  1.35, 40.85, 700.0],
    ])
    # 9. Likelihood lightcone quantities: keep minimal
    lightcone_quantities_like = ("brightness_temp",)
    
    # 10. Build loglike
    loglike = make_loglike_function(
        dataset=dataset,
        base_inputs=inputs,
        cache=cache,
        lightcone_quantities=lightcone_quantities_like,
        chunk_indices=chunk_indices_HERA,
        n_psbins=47,
        k_min_ps=3.337118317301632e-02,
        k_max_ps=2.675685850887854e+00,
        parameter_names=parameter_names,
    )

    fiducial_theta = np.array([
        -1.3,   # F_STAR10
         0.5,   # ALPHA_STAR
        -1.0,   # F_ESC10
        -0.5,   # ALPHA_ESC
         8.7,   # M_TURN
         0.5,   # t_STAR
        40.5,   # L_X
       500.0,   # NU_X_THRESH [eV]
    ])
    
    # Select a subset of the parameter space to test on.
    varying_names = ["L_X", "NU_X_THRESH"]
    varying_indices = [parameter_names.index(name) for name in varying_names]
    
    param_bounds_2d = param_bounds[:, varying_indices]
    param_labels_2d = [parameter_labels[i] for i in varying_indices]
    
    def expand_theta_2d(theta_2d, fiducial_theta, varying_indices):
        theta_full = fiducial_theta.copy()
        theta_full[varying_indices] = theta_2d
        return theta_full
    
    def loglike_2d(theta_2d):
        theta_full = expand_theta_2d(theta_2d, fiducial_theta, varying_indices)
        return loglike(theta_full)
    
    print(f"[rank {rank}/{size}] varying_names = {varying_names}", flush=True)
    print(f"[rank {rank}/{size}] varying_indices = {varying_indices}", flush=True)
    print(f"[rank {rank}/{size}] param_bounds_2d =\n{param_bounds_2d}", flush=True)
    print(f"[rank {rank}/{size}] fiducial 2D point = {fiducial_theta[varying_indices]}", flush=True)

    # Set up the cosmological likelihood
    likelihood_name = f'21cmTest_2D_z01_5sigma'

    start = time.time()
    print(f"[rank {rank}/{size}] Starting BOBE run...", flush=True)

    # Run BO Loop
    bobe = BOBE(
        loglikelihood=loglike_2d,
        likelihood_name=likelihood_name,
        param_bounds=param_bounds_2d,
        param_list=varying_names,
        param_labels=param_labels_2d,
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
