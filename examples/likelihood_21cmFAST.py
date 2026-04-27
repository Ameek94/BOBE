import gc
import os
import time
from dataclasses import dataclass

import numpy as np
import psutil
import py21cmfast as p21c
from powerbox.tools import get_power


def rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


#----------------------------#
# Fixed analysis assumptions #
#----------------------------#

PARAMETER_NAMES = [
    "F_STAR10",
    "ALPHA_STAR",
    "F_ESC10",
    "ALPHA_ESC",
    "M_TURN",
    "t_STAR",
    "L_X",
    "NU_X_THRESH",
]

PARAMETER_LABELS = [
    r"\log_{10}(f_{*,10})",
    r"\alpha_*",
    r"\log_{10}(f_{\mathrm{esc},10})",
    r"\alpha_{\mathrm{esc}}",
    r"\log_{10}(M_{\mathrm{turn}})",
    r"t_*",
    r"\log_{10}\left(\frac{L_{X<2\,\mathrm{keV}}}{\mathrm{SFR}}\right)",
    r"E_0",
]

# Park et al. fiducial point used as the baseline model point.
# NU_X_THRESH is stored in eV because that is what 21cmFAST expects.
FIDUCIAL_THETA = np.array([
    -1.30,
     0.50,
    -1.00,
    -0.50,
     8.70,
     0.50,
    40.50,
   500.0,
], dtype=float)

# Default local widths used to define a practical box around the fiducial point.
SIGMA_LO = np.array([0.02, 0.02, 0.02, 0.015, 0.02, 0.02, 0.0015, 0.95], dtype=float)
SIGMA_HI = np.array([0.02, 0.02, 0.02, 0.015, 0.02, 0.02, 0.0015, 1.40], dtype=float)

# Fixed redshift chunks used throughout this likelihood.
CHUNK_Z_LIST = [
    27.4, 23.4828, 20.5152, 18.1892, 16.3171, 14.7778, 13.4898, 12.3962,
    11.4561, 10.6393, 9.92308, 9.28986, 8.72603, 8.22078, 7.76543,
    7.35294, 6.97753, 6.63441, 6.31959, 6.0297, 5.7619, 5.51376,
]

# Masking and power-spectrum settings.
N_PSBINS = 47
K_MIN_PS = 3.337118317301632e-02
K_MAX_PS = 2.675685850887854e+00
Z_MIN = 6.0
Z_MAX = 30.0
K_MIN = 0.1
K_MAX = 1.0
LIGHTCONE_QUANTITIES = ("brightness_temp",)


def default_param_bounds(nsigma: float = 5.0) -> np.ndarray:
    return np.array([
        FIDUCIAL_THETA - nsigma * SIGMA_LO,
        FIDUCIAL_THETA + nsigma * SIGMA_HI,
    ])


def compute_power(
    box,
    length,
    n_psbins,
    log_bins=True,
    k_min=None,
    k_max=None,
    ignore_kperp_zero=True,
    ignore_kpar_zero=False,
    ignore_k_zero=False,
):
    """Compute the power spectrum for one lightcone chunk."""
    k_weights = np.ones(box.shape, dtype=int)
    n0 = k_weights.shape[0]
    n1 = k_weights.shape[-1]

    if ignore_kperp_zero:
        k_weights[n0 // 2, n0 // 2, :] = 0
    if ignore_kpar_zero:
        k_weights[:, :, n1 // 2] = 0
    if ignore_k_zero:
        k_weights[n0 // 2, n0 // 2, n1 // 2] = 0

    if k_min is None and k_max is None:
        bins = n_psbins
    else:
        bins = (
            np.logspace(np.log10(k_min), np.log10(k_max), n_psbins)
            if log_bins
            else np.linspace(k_min, k_max, n_psbins)
        )

    power, k_edges, variance = get_power(
        box,
        boxlength=length,
        bins=bins,
        bin_ave=False,
        get_variance=True,
        log_bins=log_bins,
        k_weights=k_weights,
    )

    if log_bins:
        k = np.exp((np.log(k_edges[1:]) + np.log(k_edges[:-1])) / 2)
    else:
        k = 0.5 * (k_edges[1:] + k_edges[:-1])

    return power, k, variance


def chunk_indices(lightcone, chunk_z_list):
    """Return slice indices closest to the requested chunk redshifts."""
    lc_redshifts = lightcone.lightcone_redshifts
    return [np.argmin(np.abs(lc_redshifts - z)) for z in chunk_z_list][::-1]


def powerspectra_chunks(
    lightcone,
    chunk_indices_list,
    n_psbins=N_PSBINS,
    k_min=K_MIN_PS,
    k_max=K_MAX_PS,
    logk=True,
    remove_nans=False,
):
    """
    Compute chunked Delta^2(k,z) from a lightcone.
    - brightness_temp is the only field used
    - chunk boundaries are fixed externally
    """
    data = []
    nchunks = len(chunk_indices_list) - 1
    lc_redshifts = lightcone.lightcone_redshifts
    chunk_redshift = np.zeros(nchunks)

    for i in range(nchunks):
        start = chunk_indices_list[i]
        end = chunk_indices_list[i + 1]
        chunklen = (end - start) * lightcone.cell_size
        chunk_redshift[i] = np.median(lc_redshifts[start:end])

        power, k, variance = compute_power(
            lightcone.lightcones["brightness_temp"][:, :, start:end],
            (lightcone.lightcone_dimensions[0], lightcone.lightcone_dimensions[0], chunklen),
            n_psbins=n_psbins,
            log_bins=logk,
            k_min=k_min,
            k_max=k_max,
        )

        if remove_nans:
            good = ~np.isnan(power)
            power, k, variance = power[good], k[good], variance[good]
        else:
            variance[np.isnan(power)] = np.inf

        data.append({
            "k": k,
            "delta": power * k**3 / (2 * np.pi**2),
            "err_delta": np.sqrt(variance) * k**3 / (2 * np.pi**2),
        })

    return chunk_redshift, data


def save_ps_comparison_txt(filename, z, k, data_ps, sigma_ps, model_ps, theta=None):
    """Write the masked data/model comparison table used for debugging."""
    residual = model_ps - data_ps
    pull = np.full_like(residual, np.nan, dtype=float)

    good = np.isfinite(model_ps) & np.isfinite(data_ps) & np.isfinite(sigma_ps) & (sigma_ps > 0)
    pull[good] = residual[good] / sigma_ps[good]

    Z, K = np.meshgrid(z, k, indexing="ij")
    out = np.column_stack([
        Z.ravel(),
        K.ravel(),
        data_ps.ravel(),
        sigma_ps.ravel(),
        model_ps.ravel(),
        residual.ravel(),
        pull.ravel(),
    ])

    header = [
        "Power spectrum comparison table",
        "Columns: z  k  data_ps  sigma_ps  model_ps  residual  pull",
    ]
    if theta is not None:
        header.append("theta = " + np.array2string(np.asarray(theta), precision=8))

    np.savetxt(filename, out, header="\n".join(header), fmt="%.10e")


@dataclass
class Likelihood21cmFAST:
    """
    21cm power-spectrum likelihood built from 21cmFAST:
    - 8 Dimensional with parameters in the order given by PARAMETER_NAMES
    - fiducial data vector is built from a stored fiducial lightcone
    - sensitivity is read from a fixed text file and treated as fixed noise
    - Gaussian likelihood with diagonal covariance
    - by default the likelihood uses only the chi^2 term, not the Gaussian normalization
    - masking in z and k is fixed and built once during initialization
    """
    fiducial_path: str
    sensitivity_path: str
    cache_dir: str
    n_threads: int = 1
    random_seed: int = 1234
    include_norm: bool = False
    debug_output_file: str | None = None
    param_bounds: np.ndarray | None = None
    nsigma_bounds: float = 5.0

    def __post_init__(self):
        self.param_names = PARAMETER_NAMES
        self.param_labels = PARAMETER_LABELS
        self.fiducial_theta = FIDUCIAL_THETA.copy()
        #self.param_bounds = get_param_bounds()
        if self.param_bounds is None:
            self.param_bounds = default_param_bounds(self.nsigma_bounds)
        else:
            self.param_bounds = np.asarray(self.param_bounds, dtype=float)
            if self.param_bounds.shape != (2, 8):
                raise ValueError(f"param_bounds must have shape (2, 8), got {self.param_bounds.shape}")

        self._build_dataset()
        self._build_base_inputs()
        self.cache = p21c.OutputCache(self.cache_dir)

    def _build_dataset(self):
        """Build the fixed masked data vector from the stored fiducial lightcone."""
        lightcone_fiducial = p21c.LightCone.from_file(path=self.fiducial_path)
        self.chunk_idx = chunk_indices(lightcone_fiducial, CHUNK_Z_LIST)

        redshifts, data_fid = powerspectra_chunks(
            lightcone_fiducial,
            chunk_indices_list=self.chunk_idx,
            n_psbins=N_PSBINS,
            k_min=K_MIN_PS,
            k_max=K_MAX_PS,
            remove_nans=False,
        )

        fiducial_ps = np.array([chunk["delta"] for chunk in data_fid])
        fiducial_k = data_fid[0]["k"]
        sensitivity = np.loadtxt(self.sensitivity_path)[:-2, :]

        redshift_mask = (redshifts > Z_MIN) & (redshifts < Z_MAX)
        k_mask = (fiducial_k > K_MIN) & (fiducial_k < K_MAX)

        self.z = redshifts[redshift_mask]
        self.k = fiducial_k[k_mask]
        self.data_ps = fiducial_ps[np.ix_(redshift_mask, k_mask)]
        self.sigma_ps = sensitivity[np.ix_(redshift_mask, k_mask)]
        self.mask = np.isfinite(self.data_ps) & np.isfinite(self.sigma_ps) & (self.sigma_ps > 0)

        self.redshift_mask = redshift_mask
        self.k_mask = k_mask

    def _build_base_inputs(self):
        """Build the fixed baseline 21cmFAST input object."""
        inputs = p21c.InputParameters.from_template("Park19", random_seed=self.random_seed)
        self.base_inputs = inputs.evolve_input_structs(
            F_STAR10=-1.3,
            ALPHA_STAR=0.5,
            F_ESC10=-1.0,
            ALPHA_ESC=-0.5,
            M_TURN=8.7,
            t_STAR=0.5,
            L_X=40.5,
            NU_X_THRESH=500.0,
            N_THREADS=self.n_threads,
        )

    def _compute_model_ps(self, theta: np.ndarray) -> np.ndarray:
        """Compute the masked model power spectrum."""
        params = {name: value for name, value in zip(self.param_names, theta)}

        t0 = time.time()
        print(f"[model] start RSS = {rss_gb():.2f} GB", flush=True)

        inputs = self.base_inputs.evolve_input_structs(**params)

        initial_conditions = p21c.compute_initial_conditions(
            inputs=inputs,
            cache=self.cache,
            write=False,
        )
        print(f"[model] after IC RSS = {rss_gb():.2f} GB, dt = {time.time()-t0:.1f}s", flush=True)

        lightconer = p21c.RectilinearLightconer.between_redshifts(
            min_redshift=min(inputs.node_redshifts) + 0.1,
            max_redshift=max(inputs.node_redshifts) - 0.1,
            quantities=LIGHTCONE_QUANTITIES,
            resolution=inputs.simulation_options.cell_size,
        )

        lightcone = p21c.run_lightcone(
            lightconer=lightconer,
            inputs=inputs,
            initial_conditions=initial_conditions,
            cache=self.cache,
            write=False,
            progressbar=False,
        )
        print(f"[model] after run_lightcone RSS = {rss_gb():.2f} GB, dt = {time.time()-t0:.1f}s", flush=True)

        _, data_model = powerspectra_chunks(
            lightcone,
            chunk_indices_list=self.chunk_idx,
            n_psbins=N_PSBINS,
            k_min=K_MIN_PS,
            k_max=K_MAX_PS,
            remove_nans=False,
        )

        model_ps = np.array([chunk["delta"] for chunk in data_model])
        model_ps = model_ps[np.ix_(self.redshift_mask, self.k_mask)]

        del initial_conditions, lightcone, data_model
        gc.collect()

        print(f"[model] before return RSS = {rss_gb():.2f} GB, dt = {time.time()-t0:.1f}s", flush=True)
        return model_ps

    def gaussian_loglike(self, model_ps: np.ndarray) -> float:
        """Evaluate the Gaussian log-likelihood for a model power spectrum."""
        if model_ps.shape != self.data_ps.shape or model_ps.shape != self.sigma_ps.shape:
            raise ValueError(
                f"Shape mismatch: model {model_ps.shape}, data {self.data_ps.shape}, sigma {self.sigma_ps.shape}"
            )

        mask = self.mask & np.isfinite(model_ps)
        resid = model_ps[mask] - self.data_ps[mask]
        var = self.sigma_ps[mask] ** 2
        chi2 = np.sum((resid ** 2) / var)

        if self.include_norm:
            return -0.5 * (chi2 + np.sum(np.log(2.0 * np.pi * var)))
        return -0.5 * chi2

    def __call__(self, theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (8,):
            raise ValueError(f"Expected theta with shape (8,), got {theta.shape}")

        model_ps = self._compute_model_ps(theta)

        if self.debug_output_file:
            save_ps_comparison_txt(
                self.debug_output_file,
                z=self.z,
                k=self.k,
                data_ps=self.data_ps,
                sigma_ps=self.sigma_ps,
                model_ps=model_ps,
                theta=theta,
            )

        return self.gaussian_loglike(model_ps)

    def loglike(self, theta: np.ndarray) -> float:
        """Alias for __call__."""
        return self(theta)