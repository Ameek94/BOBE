import numpy as np
import py21cmfast as p21c
from powerbox.tools import get_power 

import os
import time
import gc
import psutil

def rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


#--------------------------#
#--- Parameter Metadata ---#
#--------------------------#

PARAMETER_NAMES = [
    "F_STAR10",
    "ALPHA_STAR",
    "F_ESC10",
    "ALPHA_ESC",
    "M_TURN",
    "t_STAR",
    "L_X",
    "NU_X_THRESH", #eV in code, keV in paper
    ]

PARAMETER_LABELS = [
    r"\log_{10}(f_{*,10})",
    r"\alpha_*",
    r"\log_{10}(f_{\mathrm{esc},10})",
    r"\alpha_{\mathrm{esc}}",
    r"\log_{10}(M_{\mathrm{turn}})",
    r"t_*",
    r"\log_{10}\!\left(\frac{L_{X<2\,\mathrm{keV}}}{\mathrm{SFR}}\right)",
    r"E_0",
    ]
# Park et al. fiducial values.
# Note: NU_X_THRESH / E0 is stored in eV here as that's what 21cmFAST wants
FIDUCIAL_THETA = np.array([
    -1.30,   # F_STAR10
     0.50,   # ALPHA_STAR
    -1.00,   # F_ESC10
    -0.50,   # ALPHA_ESC
     8.70,   # M_TURN
     0.50,   # t_STAR
    40.50,   # L_X
   500.0,    # NU_X_THRESH [eV]  <- paper quotes 0.50 keV
], dtype=float)

# 1-sigma lower errors from the "21-cm only" row of Park et al. Table 2.
SIGMA_LO = np.array([
    0.21,
    0.31,
    0.21,
    0.27,
    0.26,
    0.14,
    0.07,
    0.04 * 1000.0,   # E0: 0.04 keV -> 40 eV
], dtype=float)    

# 1-sigma upper errors from the "21-cm only" row of Park et al. Table 2.
SIGMA_HI = np.array([
    0.18,
    0.23,
    0.24,
    0.26,
    0.27,
    0.17,
    0.07,
    0.04 * 1000.0,   # E0: 0.04 keV -> 40 eV
], dtype=float)

def get_param_bounds(nsigma=5.0):
    """
    Construct parameter bounds as fiducial +/- nsigma * sigma.

    Uses the Park et al. fiducial values and the 1-sigma bands from the
    '21-cm only' row of Table 2.

    Parameters
    ----------
    nsigma : float
        Number of sigma for the lower/upper bounds.

    Returns
    -------
    param_bounds : ndarray, shape (2, ndim_full)
        Lower and upper bounds for all parameters.
    """
    return np.array([
        FIDUCIAL_THETA - nsigma * SIGMA_LO,
        FIDUCIAL_THETA + nsigma * SIGMA_HI,
    ])

def get_varying_indices(ndim, parameter_names=None):
    """
    Choose the last `ndim` parameters from the full parameter list.

    Convention used here:
    - NDIM = 1 -> ['NU_X_THRESH']
    - NDIM = 2 -> ['L_X', 'NU_X_THRESH']
    - NDIM = 3 -> ['t_STAR', 'L_X', 'NU_X_THRESH']
    - etc., working backwards through the list.

    Parameters
    ----------
    ndim : int
        Number of active/varying parameters.

    parameter_names : list[str], optional
        Full ordered parameter list. Defaults to PARAMETER_NAMES.

    Returns
    -------
    varying_indices : list[int]
        Indices of active parameters in the full parameter list.
    """
    if parameter_names is None:
        parameter_names = PARAMETER_NAMES

    if not 1 <= ndim <= len(parameter_names):
        raise ValueError(f"NDIM must be in [1, {len(parameter_names)}], got {ndim}")

    return list(range(len(parameter_names) - ndim, len(parameter_names)))

def get_varying_metadata(ndim,
                         parameter_names=None,
                         parameter_labels=None,
                         fiducial_theta=None,
                         param_bounds=None):
    """
    Return the active parameter names, labels, fiducial values, and bounds
    for a given NDIM.

    Parameters
    ----------
    ndim : int
        Number of active/varying parameters.

    parameter_names, parameter_labels, fiducial_theta, param_bounds
        Full metadata arrays/lists. Defaults use the module-level values.

    Returns
    -------
    meta : dict
        Dictionary containing:
        - varying_indices
        - varying_names
        - varying_labels
        - fiducial_nd
        - param_bounds_nd
    """
    if parameter_names is None:
        parameter_names = PARAMETER_NAMES
    if parameter_labels is None:
        parameter_labels = PARAMETER_LABELS
    if fiducial_theta is None:
        fiducial_theta = FIDUCIAL_THETA
    if param_bounds is None:
        param_bounds = get_param_bounds()

    varying_indices = get_varying_indices(ndim, parameter_names)

    return {
        "varying_indices": varying_indices,
        "varying_names": [parameter_names[i] for i in varying_indices],
        "varying_labels": [parameter_labels[i] for i in varying_indices],
        "fiducial_nd": fiducial_theta[varying_indices],
        "param_bounds_nd": param_bounds[:, varying_indices],
    }

def expand_theta_nd(theta_nd, fiducial_theta=None, varying_indices=None):
    """
    Insert an NDIM parameter vector into the full fiducial parameter vector.

    Parameters
    ----------
    theta_nd : ndarray, shape (ndim,)
        Active parameter values.

    fiducial_theta : ndarray, optional
        Full fiducial parameter vector. Defaults to FIDUCIAL_THETA.

    varying_indices : list[int]
        Indices of active parameters within the full vector.

    Returns
    -------
    theta_full : ndarray, shape (ndim_full,)
        Full parameter vector.
    """
    if fiducial_theta is None:
        fiducial_theta = FIDUCIAL_THETA
    if varying_indices is None:
        raise ValueError("varying_indices must be provided")

    theta_full = fiducial_theta.copy()
    theta_full[varying_indices] = theta_nd
    return theta_full
#---------------------------------------------#
#--- Power spectrum and likelihood helpers ---#
#---------------------------------------------#

def compute_power(box,
                   length,
                   n_psbins,
                   log_bins=True,
                   k_min=None,
                   k_max=None,
                   ignore_kperp_zero=True,
                   ignore_kpar_zero=False,
                   ignore_k_zero=False):
    """
    Calculate power spectrum for a redshift chunk
    """
    
    # Determine the weighting function required from ignoring k's.
    k_weights = np.ones(box.shape, dtype=int)
    n0 = k_weights.shape[0]
    n1 = k_weights.shape[-1]

    if ignore_kperp_zero:
        k_weights[n0 // 2, n0 // 2, :] = 0
    if ignore_kpar_zero:
        k_weights[:, :, n1 // 2] = 0
    if ignore_k_zero:
        k_weights[n0 // 2, n0 // 2, n1 // 2] = 0

    # Define k bins
    if k_min is None and k_max is None:
        bins = n_psbins
    else:
        if log_bins:
            bins = np.logspace(np.log10(k_min), np.log10(k_max), n_psbins)
        else:
            bins = np.linspace(k_min, k_max, n_psbins)

    res = get_power(
        box,
        boxlength=length,
        bins=bins,
        bin_ave=False,
        get_variance=True,
        log_bins=log_bins,
        k_weights=k_weights,
    )

    res = list(res)
    k = res[1]
    if log_bins:
        k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
    else:
        k = (k[1:] + k[:-1]) / 2

    res[1] = k

    return res

def powerspectra_chunks(lightcone, nchunks=10,
                        chunk_indices=None,
                        n_psbins=50,
                        k_min=0.1,
                        k_max=1.0,
                        logk=True,
                        ignore_kperp_zero=True,
                        ignore_kpar_zero=False,
                        ignore_k_zero=False,
                        remove_nans=True,
                        vb=False):

    """
    Make power spectra for given number of equally spaced redshift chunks OR list of redshift chunk lightcone indices
    """
    data = []

    # Create lightcone redshift chunks
    # If chunk indices not given, divide lightcone into nchunks equally spaced redshift chunks
    if chunk_indices is None:
        chunk_indices = list(range(0,lightcone.n_slices,round(lightcone.n_slices / nchunks),))
        print(f'Chunk indices: {chunk_indices}', flush=True)
        if len(chunk_indices) > nchunks:
            chunk_indices = chunk_indices[:-1]

        chunk_indices.append(lightcone.n_slices)

    else:
        nchunks = len(chunk_indices) - 1

    chunk_redshift = np.zeros(nchunks)

    lc_redshifts = lightcone.lightcone_redshifts
    redshift_medians = np.zeros(nchunks)
    # Calculate PS in each redshift chunk
    for i in range(nchunks):
        if vb:
            print(f'Chunk {i}/{nchunks}...', flush=True)
        start    = chunk_indices[i]
        end      = chunk_indices[i + 1]
        chunklen = (end - start) * lightcone.cell_size

        chunk_redshift[i] = np.median(lc_redshifts[start:end])
        redshift_medians[i] = chunk_redshift[i]
        if chunklen == 0:
            print(f'Chunk size = 0 for z = {lc_redshifts[start]}-{lc_redshifts[end]}', flush=True)
        else:
            power, k, variance = compute_power(
                    lightcone.lightcones['brightness_temp'][:, :, start:end],
                    (lightcone.lightcone_dimensions[0], lightcone.lightcone_dimensions[0], chunklen),
                    n_psbins,
                    log_bins=logk,
                    k_min=k_min,
                    k_max=k_max,
                    ignore_kperp_zero=ignore_kperp_zero,
                    ignore_kpar_zero=ignore_kpar_zero,
                    ignore_k_zero=ignore_k_zero,)

            if remove_nans:
                power, k, variance = power[~np.isnan(power)], k[~np.isnan(power)], variance[~np.isnan(power)]
            else:
                variance[np.isnan(power)] = np.inf

            data.append({"k": k, "delta": power * k ** 3 / (2 * np.pi ** 2), "err_delta": np.sqrt(variance) * k ** 3 / (2 * np.pi ** 2)})

    return chunk_redshift, data, redshift_medians


def chunk_indices(lightcone, chunk_z_list):
    """
    Get indices of lightcone redshifts that are closest to the given chunk redshifts.
    """
    lc_redshifts = lightcone.lightcone_redshifts
    chunk_indices = [np.argmin(np.abs(lc_redshifts - z_HERA)) for z_HERA in chunk_z_list][::-1]
    return chunk_indices

def build_mock_dataset(
    fiducial_ps,
    redshifts,
    k_values,
    sensitivity,
    z_min=6.0,
    z_max=30.0,
    k_min=0.1,
    k_max=1.0):
    """
    Build a clean mock dataset for the 21cm power spectrum likelihood.
    """

    if fiducial_ps.shape != (len(redshifts), len(k_values)):
        raise ValueError(
            f"fiducial_ps shape {fiducial_ps.shape} does not match " 
            f"(Nz, Nk)=({len(redshifts)}, {len(k_values)})"
        )
    # Apply z/k cuts to fiducial quantities first
    redshift_mask = (redshifts > z_min) & (redshifts < z_max)
    k_mask = (k_values > k_min) & (k_values < k_max)

    z_like = redshifts[redshift_mask]
    k_like = k_values[k_mask]
    ps_like = fiducial_ps[np.ix_(redshift_mask, k_mask)]

    if sensitivity.shape != fiducial_ps.shape:
        raise ValueError(f"Shape mismatch between sensitivity ({sensitivity.shape}) and fiducial power spectrum ({fiducial_ps.shape}) ")
    
    sigma_like = sensitivity[np.ix_(redshift_mask, k_mask)]
    valid_mask = np.isfinite(ps_like) & np.isfinite(sigma_like) & (sigma_like > 0)

    return {
        "redshift_mask": redshift_mask,
        "k_mask": k_mask,
        "z": z_like,
        "k": k_like,
        "ps_fid": ps_like,
        "sigma": sigma_like,
        "mask": valid_mask
    }

def gaussian_loglike(model_ps, data_ps, sigma_ps, mask=None, include_norm=True):
    """
    Gaussian log-likelihood for Delta^2(k, z) assuming diagonal covariance.
    """ 

    if model_ps.shape != data_ps.shape or model_ps.shape != sigma_ps.shape:
        raise ValueError(f"Shape mismatch: model {model_ps.shape}, data {data_ps.shape}, sigma {sigma_ps.shape}")

    if mask is None:
        mask = np.isfinite(model_ps) & np.isfinite(data_ps) & np.isfinite(sigma_ps) & (sigma_ps > 0)

    resid = model_ps[mask] - data_ps[mask]
    var = sigma_ps[mask] ** 2

    chi2 = np.sum((resid ** 2) / var)

    if include_norm:
        return -0.5 * (chi2 + np.sum(np.log(2.0 * np.pi * var)))
    else:
        return -0.5 * chi2

def make_base_inputs(n_threads=1, random_seed=1234):
    """
    Construct the baseline 21cmFAST inputs.

    NOTE: this currently uses the Park19 template and the Park et al. fiducial values.
    Those are analysis choices, not generic 21cmFAST defaults.
    """
    inputs = p21c.InputParameters.from_template("Park19", random_seed=random_seed)
    return inputs.evolve_input_structs(
        F_STAR10=-1.3,
        ALPHA_STAR=0.5,
        F_ESC10=-1.0,
        ALPHA_ESC=-0.5,
        M_TURN=8.7,
        t_STAR=0.5,
        L_X=40.5,
        NU_X_THRESH=500.0,   # eV in code
        N_THREADS=n_threads,
    )


def build_fiducial_dataset(fiducial_path,
                           sensitivity_path,
                           chunk_z_list,
                           n_psbins=47,
                           k_min_ps=3.337118317301632e-02,
                           k_max_ps=2.675685850887854e+00,
                           z_min=6.0,
                           z_max=30.0,
                           k_min=0.1,
                           k_max=1.0):
    """
    Build the fiducial dataset used by the likelihood.

    NOTE: the default chunk redshifts, k-bin settings, and mask cuts below
    are the specific analysis choices currently used in this project.
    """
    lightcone_fiducial = p21c.LightCone.from_file(path=fiducial_path)

    chunk_idx = chunk_indices(lightcone_fiducial, chunk_z_list)

    chunk_redshifts_fid, data_fid, _ = powerspectra_chunks(
        lightcone_fiducial,
        chunk_indices=chunk_idx,
        n_psbins=n_psbins,
        k_min=k_min_ps,
        k_max=k_max_ps,
        remove_nans=False,
    )

    fiducial_ps = np.array([chunk["delta"] for chunk in data_fid])
    fiducial_k = data_fid[0]["k"]

    sensitivity = np.loadtxt(sensitivity_path)[:-2, :]

    dataset = build_mock_dataset(
        fiducial_ps=fiducial_ps,
        redshifts=chunk_redshifts_fid,
        k_values=fiducial_k,
        sensitivity=sensitivity,
        z_min=z_min,
        z_max=z_max,
        k_min=k_min,
        k_max=k_max,
    )

    return {
        "dataset": dataset,
        "lightcone_fiducial": lightcone_fiducial,
        "chunk_indices": chunk_idx,
        "chunk_redshifts_fid": chunk_redshifts_fid,
        "fiducial_ps": fiducial_ps,
        "fiducial_k": fiducial_k,
    }


def compute_model_ps_from_params(
    params,
    base_inputs,
    cache,
    lightcone_quantities,
    chunk_indices,
    n_psbins,
    k_min_ps,
    k_max_ps,
    z_mask,
    k_mask):
    """
    Recompute the 21cmFAST lightcone and chunked Delta^2(k,z) for one parameter point.

    Parameters
    ----------
    params: dict
        Dictionary of input parameter updates for inputs.evolve_input_structs(...)

    base_inputs: py21cmfast InputParameters
        Baseline input object to evolve from.

    Returns
    -------
    model_ps_masked: array, shape (Nz_like, Nk_like)
    """

    t0 = time.time()
    print(f"[model] start RSS = {rss_gb():.2f} GB", flush=True)
    
    inputs = base_inputs.evolve_input_structs(**params)

    initial_conditions = p21c.compute_initial_conditions(
        inputs=inputs,
        cache=cache,
        write=False
    )

    print(f"[model] after IC RSS = {rss_gb():.2f} GB, dt = {time.time()-t0:.1f} s", flush=True)
    
    lcn = p21c.RectilinearLightconer.between_redshifts(
        min_redshift=min(inputs.node_redshifts) + 0.1,
        max_redshift=max(inputs.node_redshifts) - 0.1,
        quantities=lightcone_quantities,
        resolution=inputs.simulation_options.cell_size,
    )

    print(f"[model] after lightconer RSS = {rss_gb():.2f} GB, dt = {time.time()-t0:.1f} s", flush=True)
    
    lightcone = p21c.run_lightcone(
        lightconer=lcn,
        inputs=inputs,
        initial_conditions=initial_conditions,
        cache=cache,
        write=False,
        progressbar=False,
    )

    print(f"[model] after run_lightcone RSS = {rss_gb():.2f} GB, dt = {time.time()-t0:.1f} s", flush=True)
    
    _, data_model, _ = powerspectra_chunks(
        lightcone,
        chunk_indices=chunk_indices,
        n_psbins=n_psbins,
        k_min=k_min_ps,
        k_max=k_max_ps,
        remove_nans=False,
    )

    print(f"[model] after powerspectra_chunks RSS = {rss_gb():.2f} GB, dt = {time.time()-t0:.1f} s", flush=True)

    
    model_ps = np.array([chunk["delta"] for chunk in data_model])

    # Sanity: assume same native grid as fiducial for now
    model_ps_masked = model_ps[np.ix_(z_mask, k_mask)]

    del initial_conditions
    del lightcone
    del data_model
    del model_ps
    gc.collect()
    

    print(f"[model] before return RSS = {rss_gb():.2f} GB, dt = {time.time()-t0:.1f} s", flush=True)
    
    return model_ps_masked



def make_loglike_function(
    dataset,
    base_inputs,
    cache,
    lightcone_quantities,
    chunk_indices,
    n_psbins,
    k_min_ps,
    k_max_ps,
    parameter_names):
    """
    Return a loglike(theta) function for BOBE to use
    """

    def loglike(theta):
        params = {name: value for name, value in zip(parameter_names, theta)}

        model_ps = compute_model_ps_from_params(
            params=params,
            base_inputs=base_inputs,
            cache=cache,
            lightcone_quantities=lightcone_quantities,
            chunk_indices=chunk_indices,
            n_psbins=n_psbins,
            k_min_ps=k_min_ps,
            k_max_ps=k_max_ps,
            z_mask=dataset["redshift_mask"],
            k_mask=dataset["k_mask"],
        )

        return gaussian_loglike(
            model_ps=model_ps,
            data_ps=dataset['ps_fid'],
            sigma_ps=dataset['sigma'],
            mask=dataset['mask'],
            include_norm=True,
        )
    return loglike


def make_nd_loglike(dataset,
                    base_inputs,
                    cache,
                    ndim,
                    nsigma=5.0,
                    lightcone_quantities=("brightness_temp",),
                    chunk_indices=None,
                    n_psbins=47,
                    k_min_ps=3.337118317301632e-02,
                    k_max_ps=2.675685850887854e+00,
                    parameter_names=None,
                    parameter_labels=None,
                    fiducial_theta=None):
    """
    Build an NDIM-restricted likelihood by varying only the last `ndim`
    parameters of the full parameter list.

    Returns
    -------
    loglike_nd : callable
        Function of theta_nd only.

    meta : dict
        Useful metadata for plotting/running:
        - varying_indices
        - varying_names
        - varying_labels
        - fiducial_nd
        - param_bounds_nd
        - fiducial_theta
        - parameter_names
        - parameter_labels
    """
    if parameter_names is None:
        parameter_names = PARAMETER_NAMES
    if parameter_labels is None:
        parameter_labels = PARAMETER_LABELS
    if fiducial_theta is None:
        fiducial_theta = FIDUCIAL_THETA

    if chunk_indices is None:
        raise ValueError("chunk_indices must be provided")

    param_bounds = get_param_bounds(nsigma)
    meta = get_varying_metadata(
        ndim=ndim,
        parameter_names=parameter_names,
        parameter_labels=parameter_labels,
        fiducial_theta=fiducial_theta,
        param_bounds=param_bounds,
    )

    loglike_full = make_loglike_function(
        dataset=dataset,
        base_inputs=base_inputs,
        cache=cache,
        lightcone_quantities=lightcone_quantities,
        chunk_indices=chunk_indices,
        n_psbins=n_psbins,
        k_min_ps=k_min_ps,
        k_max_ps=k_max_ps,
        parameter_names=parameter_names,
    )

    def loglike_nd(theta_nd):
        theta_full = expand_theta_nd(
            theta_nd,
            fiducial_theta=fiducial_theta,
            varying_indices=meta["varying_indices"],
        )
        return loglike_full(theta_full)

    meta.update({
        "fiducial_theta": fiducial_theta,
        "parameter_names": parameter_names,
        "parameter_labels": parameter_labels,
        "param_bounds": param_bounds,
    })

    return loglike_nd, meta
