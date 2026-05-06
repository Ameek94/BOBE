import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from getdist import MCSamples, plots
from mpi4py import MPI

from likelihood_21cmFAST import (
    FIDUCIAL_THETA,
    LIGHTCONE_QUANTITIES,
    PARAMETER_NAMES,
    Likelihood21cmFAST,
)


FULL_BOUNDS = np.array(
    [
        [-2.0, -0.5],
        [0.0, 1.0],
        [-2.0, -0.5],
        [-1.0, -0.3],
        [8.2, 9.0],
        [0.01, 1.0],
        [40.25, 40.75],
        [475.0, 525.0],
    ],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_fiducial = script_dir / "21cm" / "fiducial-lightcone.h5"
    default_cache_root = script_dir / "21cm" / "cache"
    default_results_dir = script_dir / "21cm" / "results"
    print(f"Script directory: {script_dir}")
    print(f"Default fiducial path: {default_fiducial}")
    print(f"Default cache root: {default_cache_root}")
    print(f"Default results dir: {default_results_dir}")

    parser = argparse.ArgumentParser(
        description=(
            "Run 1D BOBE 21cm analysis varying only ALPHA_STAR. "
            "If the fiducial lightcone is missing, it is generated first."
        )
    )
    parser.add_argument("--fiducial-path", type=str, default=str(default_fiducial))
    parser.add_argument("--sensitivity-path", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="SKA", choices=["SKA", "HERA"])
    parser.add_argument("--cache-root", type=str, default=str(default_cache_root))
    parser.add_argument("--results-dir", type=str, default=str(default_results_dir))
    parser.add_argument("--run-tag", type=str, default="run")
    parser.add_argument("--n-threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "1")))
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=10, help="BOBE global seed")
    parser.add_argument("--min-evals", type=int, default=10)
    parser.add_argument("--max-evals", type=int, default=30)
    parser.add_argument("--max-gp-size", type=int, default=30)
    parser.add_argument("--fit-n-points", type=int, default=1)
    parser.add_argument("--num-hmc-warmup", type=int, default=256)
    parser.add_argument("--num-hmc-samples", type=int, default=512)
    parser.add_argument("--mc-points-size", type=int, default=32)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--logz-threshold", type=float, default=1e-1)
    parser.add_argument("--disable-clf", action="store_true", help="Disable classifier filtering")
    return parser.parse_args()


def generate_fiducial_lightcone(
    out_path: str,
    cache_dir: str,
    n_threads: int,
    random_seed: int,
) -> None:
    """Generate the fiducial lightcone with the same path as Likelihood21cmFAST."""
    import py21cmfast as p21c

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    cache = p21c.OutputCache(cache_dir)
    inputs = p21c.InputParameters.from_template("Park19", random_seed=random_seed)
    inputs = inputs.evolve_input_structs(
        N_THREADS=n_threads,
        **dict(zip(PARAMETER_NAMES, [float(v) for v in FIDUCIAL_THETA])),
    )

    t0 = time.time()
    ic = p21c.compute_initial_conditions(inputs=inputs, cache=cache, write=False)
    print(f"[fiducial] initial_conditions: {time.time() - t0:.1f} s", flush=True)

    lightconer = p21c.RectilinearLightconer.between_redshifts(
        min_redshift=min(inputs.node_redshifts) + 0.1,
        max_redshift=max(inputs.node_redshifts) - 0.1,
        quantities=LIGHTCONE_QUANTITIES,
        resolution=inputs.simulation_options.cell_size,
    )
    lc = p21c.run_lightcone(
        lightconer=lightconer,
        inputs=inputs,
        initial_conditions=ic,
        cache=cache,
        write=False,
        progressbar=True,
    )
    print(f"[fiducial] run_lightcone: {time.time() - t0:.1f} s total", flush=True)

    lc.save(out_path)
    print(f"[fiducial] saved lightcone to {out_path}", flush=True)


def make_reduced_likelihood(fast_like, likelihood_cls, varying_names, name, minus_inf=-1e100):
    """Wrap the 8D 21cmFAST likelihood to expose only the requested subset."""
    all_names = list(fast_like.param_names)
    all_labels = list(fast_like.param_labels)
    all_bounds = np.asarray(fast_like.param_bounds)
    fiducial = np.asarray(fast_like.fiducial_theta, dtype=float)

    missing = [n for n in varying_names if n not in all_names]
    if missing:
        raise ValueError(f"Unknown parameter(s) in varying_names: {missing}")

    varying_idx = np.array([all_names.index(n) for n in varying_names], dtype=int)
    param_list = [all_names[i] for i in varying_idx]
    param_labels = [all_labels[i] for i in varying_idx]
    param_bounds = all_bounds[:, varying_idx]

    def loglike(theta_reduced):
        theta_full = fiducial.copy()
        theta_full[varying_idx] = np.asarray(theta_reduced, dtype=float)
        return fast_like.loglike(theta_full)

    like = likelihood_cls(
        loglikelihood=loglike,
        param_list=param_list,
        param_labels=param_labels,
        param_bounds=param_bounds,
        name=name,
        minus_inf=minus_inf,
    )
    like.varying_idx = varying_idx
    like.fiducial_full = fiducial
    return like


def save_gp_1d_plot(gp, scale_from_unit_fn, param_bounds, param_name, param_label, output_file):
    """Save GP mean +/- 1 sigma and training points for a 1D model."""
    if gp.train_x.shape[1] != 1:
        raise ValueError("GP plotting helper expects a 1D GP.")

    x_grid_unit = np.linspace(0.0, 1.0, 50).reshape(-1, 1)
    pred_mean = np.asarray(gp.predict_mean_batched(x_grid_unit)).reshape(-1)
    pred_var = np.asarray(gp.predict_var_batched(x_grid_unit)).reshape(-1)
    pred_std = np.sqrt(np.clip(pred_var, 1e-16, None))

    x_grid_phys = np.asarray(scale_from_unit_fn(x_grid_unit, param_bounds)).reshape(-1)
    sort_idx = np.argsort(x_grid_phys)

    train_x_phys = np.asarray(scale_from_unit_fn(np.asarray(gp.train_x), param_bounds)).reshape(-1)
    train_y = np.asarray(gp.train_y).reshape(-1) * float(gp.y_std) + float(gp.y_mean)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(x_grid_phys[sort_idx], pred_mean[sort_idx], color="blue", lw=2, label="GP mean")
    ax.fill_between(
        x_grid_phys[sort_idx],
        pred_mean[sort_idx] - pred_std[sort_idx],
        pred_mean[sort_idx] + pred_std[sort_idx],
        color="blue",
        alpha=0.25,
        label="GP mean ± 1σ",
    )
    ax.scatter(train_x_phys, train_y, s=24, color="red", alpha=0.7, label="Training points")
    ax.set_xlabel(param_label)
    ax.set_ylabel("log-likelihood")
    ax.set_title(f"GP posterior for {param_name}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    # Configure threading/XLA before importing BOBE (which imports JAX).
    os.environ["OMP_NUM_THREADS"] = str(args.n_threads)
    os.environ.setdefault(
        "XLA_FLAGS",
        f"--xla_force_host_platform_device_count={args.n_threads}",
    )

    from BOBE import BOBE, Likelihood, scale_from_unit

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    fiducial_path = Path(args.fiducial_path).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()

    if args.sensitivity_path is not None:
        sensitivity_path = Path(args.sensitivity_path).expanduser().resolve()
    else:
        sensitivity_path = (
            Path(__file__).resolve().parent
            / "21cm"
            / "sensitivities"
            / f"{args.experiment.lower()}_sense.txt"
        ).resolve()

    rank_cache_dir = cache_root / f"{args.run_tag}_rank{rank}"
    fiducial_cache_dir = cache_root / f"{args.run_tag}_fiducial"

    results_dir.mkdir(parents=True, exist_ok=True)
    rank_cache_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        if fiducial_path.exists():
            print(f"[fiducial] reusing existing lightcone at {fiducial_path}", flush=True)
        else:
            print(f"[fiducial] missing file; generating at {fiducial_path}", flush=True)
            generate_fiducial_lightcone(
                out_path=str(fiducial_path),
                cache_dir=str(fiducial_cache_dir),
                n_threads=args.n_threads,
                random_seed=args.random_seed,
            )

    if size > 1:
        comm.Barrier()

    varying_names = ["ALPHA_STAR"]
    clf_type = "svm"
    clf_used = args.disable_clf == False
    likelihood_name = (
        f"21cmFAST_alpha_star_1D_{clf_type}_{clf_used}_"
        f"{size}R_{args.n_threads}T_{args.run_tag}"
    )

    print(f"[rank {rank}/{size}] starting run", flush=True)
    print(f"[rank {rank}/{size}] OMP_NUM_THREADS = {args.n_threads}", flush=True)
    print(f"[rank {rank}/{size}] fiducial_path = {fiducial_path}", flush=True)
    print(f"[rank {rank}/{size}] sensitivity_path = {sensitivity_path}", flush=True)
    print(f"[rank {rank}/{size}] cache_dir = {rank_cache_dir}", flush=True)

    fast_like = Likelihood21cmFAST(
        fiducial_path=str(fiducial_path),
        sensitivity_path=str(sensitivity_path),
        cache_dir=str(rank_cache_dir),
        n_threads=args.n_threads,
        random_seed=args.random_seed,
        include_norm=False,
        debug_output_file=None,
        param_bounds=FULL_BOUNDS.T,
    )

    like = make_reduced_likelihood(
        fast_like=fast_like,
        likelihood_cls=Likelihood,
        varying_names=varying_names,
        name=likelihood_name,
    )

    start = time.time()
    bobe = BOBE(
        loglikelihood=like.logl,
        likelihood_name=likelihood_name,
        param_bounds=like.param_bounds,
        param_list=like.param_list,
        param_labels=like.param_labels,
        confidence_for_unbounded=0.9999995,
        resume=False,
        resume_file=str(results_dir / likelihood_name),
        save_dir=str(results_dir),
        save=True,
        save_step=1,
        verbosity="INFO",
        n_sobol_init=4,
        use_clf=not args.disable_clf,
        clf_use_size=5,
        clf_nsigma_threshold=20,
        clf_type=clf_type,
        minus_inf=-1e100,
        seed=args.seed,
        gp_kwargs={"noise_prior": "learn"},
    )

    results = bobe.run(
        acq="wipstd",
        min_evals=args.min_evals,
        max_evals=args.max_evals,
        max_gp_size=args.max_gp_size,
        fit_n_points=args.fit_n_points,
        ns_n_points=max(1, size),
        batch_size=max(1, size),
        num_hmc_warmup=args.num_hmc_warmup,
        num_hmc_samples=args.num_hmc_samples,
        mc_points_size=args.mc_points_size,
        num_chains=args.num_chains,
        logz_threshold=args.logz_threshold,
        do_final_ns=True,
    )

    end = time.time()

    if results is None:
        return

    gp = results["gp"]
    logz_dict = results.get("logz", {})
    likelihood = results["likelihood"]
    results_manager = results["results_manager"]
    samples = results["samples"]

    print("\n" + "=" * 60)
    print("RUN COMPLETED")
    if "mean" in logz_dict:
        print(f"Final LogZ: {logz_dict['mean']:.4f}")
    else:
        print("Final LogZ: N/A")
    if "upper" in logz_dict and "lower" in logz_dict:
        print(f"LogZ uncertainty: +/- {(logz_dict['upper'] - logz_dict['lower']) / 2:.4f}")
    print("=" * 60)
    print(f"Manual timing: {end - start:.2f} seconds ({(end - start) / 60:.2f} minutes)")

    gp_plot_path = results_dir / f"{likelihood.name}_gp_1d_mean_std.pdf"
    save_gp_1d_plot(
        gp=gp,
        scale_from_unit_fn=scale_from_unit,
        param_bounds=likelihood.param_bounds,
        param_name=likelihood.param_list[0],
        param_label=likelihood.param_labels[0],
        output_file=str(gp_plot_path),
    )
    print(f"Saved GP summary plot: {gp_plot_path}")

    sample_array = samples["x"]
    weights_array = samples["weights"]
    ranges = dict(zip(likelihood.param_list, likelihood.param_bounds.T))
    bobe_samples = MCSamples(
        samples=sample_array,
        names=likelihood.param_list,
        labels=likelihood.param_labels,
        weights=weights_array,
        ranges=ranges,
    )

    print("Creating parameter samples plot...")
    sns.set_theme("notebook", "ticks", palette="husl")
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    g = plots.get_subplot_plotter(subplot_size=3.0, subplot_size_ratio=1)
    g.settings.legend_fontsize = 14
    g.settings.axes_fontsize = 14
    g.settings.axes_labelsize = 14
    g.plot_1d(
        [bobe_samples],
        param=likelihood.param_list[0],
        filled=[True],
        contour_colors=["#006FED"],
        contour_lws=[1],
        legend_labels=["BOBE"],
    )
    sample_plot_path = results_dir / f"{likelihood.name}_samples.pdf"
    g.export(str(sample_plot_path))

    timing_data = results_manager.get_timing_summary()
    print("DETAILED TIMING ANALYSIS")
    print(
        f"Automatic timing: {timing_data['total_runtime']:.2f} seconds "
        f"({timing_data['total_runtime'] / 60:.2f} minutes)"
    )
    print("Phase Breakdown:")
    print("-" * 40)
    for phase, time_spent in timing_data["phase_times"].items():
        if time_spent > 0:
            percentage = timing_data["percentages"].get(phase, 0)
            print(f"{phase:25s}: {time_spent:8.2f}s ({percentage:5.1f}%)")

    acquisition_data = results_manager.get_acquisition_data()
    iterations = np.array(acquisition_data["iterations"])
    values = np.array(acquisition_data["values"])
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(iterations, values, linestyle="-")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Acquisition Value")
    acq_plot_path = results_dir / f"{likelihood.name}_acquisition.pdf"
    fig.tight_layout()
    fig.savefig(str(acq_plot_path), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()