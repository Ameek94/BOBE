# This module manages the samplers used to run HMC/Nested sampling using the GP model as a surrogate for the objective function
# It contains two functions, one for the Dynesty nested sampler and the other for the HMC sampler using NUTS from numpyro
import time
from typing import Any, List, Optional, Dict, Union
import jax.numpy as jnp
import jax.random as random
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from numpyro.util import enable_x64
enable_x64()
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from .gp import GP
from .clf_gp import GPwithClassifier
from .utils.log import get_logger
from .utils.seed import get_new_jax_key, get_numpy_rng
from .utils.core import is_cluster_environment, renormalise_log_weights, resample_equal
log = get_logger("sampler")

from dynesty import NestedSampler as StaticNestedSampler, DynamicNestedSampler
import math

# dynesty utility function for computing evidence
def compute_integrals(logl=None, logvol=None, reweight=None,squared=False):
    assert logl is not None
    assert logvol is not None
    loglstar_pad = np.concatenate([[-1.e300], logl])
    # we want log(exp(logvol_i)-exp(logvol_(i+1)))
    # assuming that logvol0 = 0
    # log(exp(LV_{i})-exp(LV_{i+1})) =
    # = LV{i} + log(1-exp(LV_{i+1}-LV{i}))
    # = LV_{i+1} - (LV_{i+1} -LV_i) + log(1-exp(LV_{i+1}-LV{i}))
    dlogvol = np.diff(logvol, prepend=0)
    logdvol = logvol - dlogvol + np.log1p(-np.exp(dlogvol))
    if squared:
        logdvol = 2 * logdvol
    # logdvol is log(delta(volumes)) i.e. log (X_i-X_{i-1})
    logdvol2 = logdvol + math.log(0.5)
    # These are log(1/2(X_(i+1)-X_i))
    dlogvol = -np.diff(logvol, prepend=0)
    # this are delta(log(volumes)) of the run
    # These are log((L_i+L_{i_1})*(X_i+1-X_i)/2)
    saved_logwt = np.logaddexp(loglstar_pad[1:], loglstar_pad[:-1]) + logdvol2
    if reweight is not None:
        saved_logwt = saved_logwt + reweight
    saved_logz = np.logaddexp.accumulate(saved_logwt)
    return saved_logz

def nested_sampling_Dy(gp: GP,
                       mode: str = 'acq',
                       ndim: int = 1,
                       dlogz: float = 0.1,
                       dynamic: bool = False,
                       maxcall: Optional[int] = int(5e6),
                       print_progress: Optional[bool] = True,
                       equal_weights: bool = False,
                       sample_method: str = 'rwalk',
                       rng=None,
                       param_bounds=None,
                       transform=None,
                       ) -> tuple[np.ndarray,Dict,bool]:
    """
    Nested Sampling using Dynesty, always running in physical parameter space.

    The prior transform maps the dynesty unit cube to
    ``[param_bounds[0], param_bounds[1]]``.  The log-likelihood converts
    physical coordinates to the GP unit cube via ``transform.to_unit_jax()``
    inside a JIT-compiled function, returning -1e300 for points outside
    [0, 1]^D.

    After sampling, NS samples are resampled to equal weights in physical
    space, mapped to the GP unit cube (no clipping), and samples outside
    [0, 1]^D are discarded.  ``samples_dict['x']`` is always in GP unit-cube
    space so downstream code remains unchanged.

    Arguments
    ---------
    gp : GP or GPwithClassifier
    ndim : int
        GP training dimension (used for nlive / n_init scaling).
    dlogz : float
        Log-evidence convergence goal.
    dynamic : bool
        Unused; kept for API compatibility.
    maxcall : int
        Maximum log-likelihood calls.
    print_progress : bool, optional
        Print dynesty progress. Auto-disabled in cluster environments.
    equal_weights : bool
        Unused; always returns equal-weighted in-bounds samples.
    sample_method : str
        Dynesty sampling method (default ``'rwalk'``).
    rng : numpy Generator, optional
        Random number generator.
    param_bounds : array-like (2, D)
        Physical parameter bounds. Required.
    transform : BaseTransform
        Parameter-space transform with a JIT-compatible ``to_unit_jax()``
        method. Required.

    Returns
    -------
    samples_dict : dict
        ``x`` in GP unit-cube space (equal-weighted, in-bounds), ``weights``
        (all ones), ``logl``, ``best``.
    logz_dict : dict
        ``mean``, ``upper``, ``lower``, ``dlogz_sampler``, ``var``, ``std``.
    success : bool
        False when all logl values are identical (degenerate run).
    """
    if param_bounds is None or transform is None:
        raise ValueError("nested_sampling_Dy requires param_bounds and transform.")

    log.info("Running Nested Sampling using Dynesty...")

    if is_cluster_environment():
        print_progress = False

    lb = np.asarray(param_bounds[0], dtype=np.float64)
    ub = np.asarray(param_bounds[1], dtype=np.float64)
    phys_range = ub - lb
    ndim_phys = int(len(lb))

    # Prior transform: dynesty unit cube [0,1]^D → physical space [lb, ub]
    def prior_transform(u):
        return lb + np.asarray(u, dtype=np.float64) * phys_range

    # JIT-compiled log-likelihood: physical → unit cube → GP surrogate.
    # transform.to_unit accepts JAX arrays and is fully JIT-traceable.
    @jax.jit
    def loglike(theta):
        u = transform.to_unit(theta)
        in_bounds = jnp.all((u >= 0.0) & (u <= 1.0))
        return jnp.where(in_bounds, gp.predict_mean_single(u), jnp.float64(-1e300))

    # ------------------------------------------------------------------
    # nlive / budget settings (scaled by GP ndim)
    # ------------------------------------------------------------------
    start = time.time()

    if mode == 'acq':
        nlive = max(100, min(500, 20 * ndim))
        dlogz = 0.1
        maxcall = int(2e6)
    else:
        nlive = max(500, 40 * ndim)

    rng = rng if rng is not None else get_numpy_rng()

    # ------------------------------------------------------------------
    # Initial live points
    # Sample in unit-cube space and map to physical via transform.from_unit.
    # This guarantees all initial points lie within the GP's valid domain,
    # even when the physical bounding box (param_bounds) is much larger than
    # the effective region (e.g. with RotationTransform or NormalisingFlow).
    # ------------------------------------------------------------------
    if isinstance(gp, GPwithClassifier):
        n_init = 5000 * ndim
        init_u = rng.uniform(0.0, 1.0, size=(n_init, ndim))
        init_theta = transform.from_unit(init_u)
        init_logl = np.array(
            jax.lax.map(loglike, jnp.array(init_theta, dtype=jnp.float64), batch_size=200)
        )
        live_idx = rng.choice(n_init, size=nlive, replace=False)
        success = False
        for i in range(1000):
            sel_logl = init_logl[live_idx]
            if not np.all(sel_logl == sel_logl[0]):
                log.debug(f" Successful live points on try {i+1}/1000.")
                success = True
                break
            live_idx = rng.choice(n_init, size=nlive, replace=False)
        if not success:
            log.debug(" Could not find diverse live points; injecting GP fallback point.")
            valid_unit = gp.get_random_point(rng=rng, nstd=1.0)
            valid_phys = transform.from_unit(valid_unit)
            init_theta[live_idx[0]] = valid_phys
            init_logl[live_idx[0]] = float(loglike(jnp.array(valid_phys, dtype=jnp.float64)))
        live_theta = init_theta[live_idx]
        live_logl = init_logl[live_idx]
    else:
        init_u = rng.uniform(0.0, 1.0, size=(nlive, ndim))
        live_theta = transform.from_unit(init_u)
        live_logl = np.array(
            jax.lax.map(loglike, jnp.array(live_theta, dtype=jnp.float64), batch_size=200)
        )

    live_u = (live_theta - lb) / phys_range

    # ------------------------------------------------------------------
    # Run dynesty static nested sampler in physical space
    # ------------------------------------------------------------------
    sampler = StaticNestedSampler(
        loglike, prior_transform, ndim=ndim_phys, blob=False,
        live_points=[live_u, live_theta, live_logl],
        sample=sample_method, nlive=nlive, rstate=rng,
    )
    sampler.run_nested(print_progress=print_progress, dlogz=dlogz, maxcall=maxcall)

    res = sampler.results
    mean = res['logz'][-1]
    logz_err = res['logzerr'][-1]
    logl = res['logl']
    logvol = res['logvol']
    success = ~np.all(logl == logl[0])
    log.debug(f" Nested Sampling took {time.time() - start:.2f}s")
    log.debug(" Log Z evaluated using {} points".format(np.shape(logl)))
    log.debug(f" Dynesty made {np.sum(res['ncall'])} function calls, max value of logl = {np.max(logl):.4f}")

    samples_phys_all = res['samples']

    # Resample to equal weights in physical space, convert to unit cube,
    # discard (do not clip) samples that land outside [0, 1]^D.
    weights_all = renormalise_log_weights(res['logwt'])
    eq_phys, eq_logl = resample_equal(samples_phys_all, logl, weights=weights_all)
    eq_unit = np.asarray(transform.to_unit(eq_phys, clip=False))
    in_bounds = np.all((eq_unit >= 0.0) & (eq_unit <= 1.0), axis=1)
    n_discarded = int((~in_bounds).sum())
    if n_discarded > 0:
        log.debug(f" Discarded {n_discarded}/{len(eq_unit)} equal-weighted samples outside unit cube.")
    eq_unit = eq_unit[in_bounds]
    eq_logl = eq_logl[in_bounds]

    # Best point: highest logl among equal-weighted in-bounds samples.
    best_pt = eq_unit[np.argmax(eq_logl)]

    # logZ uncertainty: mean GP predictive std over ≤512 posterior samples.
    n_std = min(512, len(eq_unit))
    gp_var = np.asarray(jax.lax.map(
        gp.predict_var_single, jnp.array(eq_unit[:n_std], dtype=jnp.float64), batch_size=100))
    mean_gp_std = float(np.mean(np.sqrt(np.clip(gp_var, 1e-12, None))))

    logz_dict = {
        'mean': mean, 'dlogz_sampler': logz_err,
        'upper': mean + mean_gp_std,
        'lower': mean - mean_gp_std,
        'var': mean_gp_std ** 2,
        'std': mean_gp_std,
    }

    samples_dict = {}

    if equal_weights:
        samples_dict['x'] = eq_unit
        samples_dict['weights'] = np.ones(len(eq_unit))
        samples_dict['logl'] = eq_logl
    else:
        samples_dict['x'] = samples_phys_all
        samples_dict['weights'] = weights_all
        samples_dict['logl'] = logl

    samples_dict['best'] = best_pt
    samples_dict['method'] = 'nested'
    
    return (samples_dict, logz_dict, success)

def sample_GP_NUTS(gp: Union[GP, GPwithClassifier], 
                   np_rng=None, 
                   rng_key=None, 
                   num_chains=4, 
                   temp=1., 
                   **kwargs):
    """
    Obtain samples from the posterior represented by the GP mean as the logprob.
    This is a unified function that works for both GP and GPwithClassifier.
    
    Parameters
    ----------
    gp : Union[GP, GPwithClassifier]
        The Gaussian Process model to sample from.
    np_rng : np.random.Generator, optional
        NumPy random number generator. Default is None.
    rng_key : jax.random.PRNGKey, optional
        JAX random key. Default is None.
    num_chains : int, optional
        Number of parallel HMC chains. Default is 4.
    temp : float, optional
        Temperature parameter for tempering. Default is 1.0.
    **kwargs : dict
        Additional keyword arguments. Can include:
        - warmup_steps : int, optional
            Number of warmup steps for HMC. If not provided, defaults based on dimensionality.
        - num_samples : int, optional
            Number of samples to draw from each chain. If not provided, defaults based on dimensionality.
        - thinning : int, optional
            Thinning factor for samples. If not provided, defaults to 4.
        - dense_mass : bool, optional
            Whether to use dense mass matrix in NUTS. Default is True.
        - max_tree_depth : int, optional
            Maximum tree depth for NUTS. Default is 6.
            
    Returns
    -------
    samples_dict : dict
        Dictionary containing:
        - 'x': samples array of shape (num_chains * num_samples / thinning, ndim)
        - 'logp': log probabilities for each sample
        - 'best': best sample found
        - 'method': 'MCMC'
    """
    # Extract HMC settings from kwargs with simple fallback defaults
    # Note: Dimension-based defaults are now handled centrally in bo.py
    warmup_steps = kwargs.get('warmup_steps', 512)
    num_samples = kwargs.get('num_samples', 1024)
    thinning = kwargs.get('thinning', 4)
    dense_mass = kwargs.get('dense_mass', True)
    max_tree_depth = kwargs.get('max_tree_depth', 6)
    

    shape = gp.train_x.shape[1]
    
    def model():
        x = numpyro.sample('x', dist.Uniform(
            low=jnp.zeros(shape),
            high=jnp.ones(shape)
        ))
        
        mean = gp.predict_mean_batched(x)
        numpyro.factor('y', mean/temp)
        numpyro.deterministic('logp', mean)
    
    @jax.jit
    def run_single_chain(rng_key,init_x):
        init_strategy = init_to_value(values={'x': init_x})
        kernel = NUTS(model, dense_mass=dense_mass, max_tree_depth=max_tree_depth, 
                        init_strategy=init_strategy)
        mcmc = MCMC(kernel, num_warmup=warmup_steps, num_samples=num_samples,
                    num_chains=1, progress_bar=False, thinning=thinning)
        mcmc.run(rng_key)
        samples_x = mcmc.get_samples()['x']
        logps = mcmc.get_samples()['logp']
        return samples_x, logps
    
    num_devices = jax.device_count()
    
    rng_key = rng_key if rng_key is not None else get_new_jax_key()
    rng_keys = jax.random.split(rng_key, num_chains)
    
    # Generate initialization points if needed
    if num_chains == 1: 
        inits = jnp.array([gp.get_random_point(rng=np_rng)])
    else:
        inits = jnp.vstack([gp.get_random_point(rng=np_rng) for _ in range(num_chains-1)])
        inits = jnp.vstack([inits, gp.train_x[jnp.argmax(gp.train_y)]])  # Add best training point as one init

    log.debug(f"Running MCMC with {num_chains} chains on {num_devices} devices.")

    # Adaptive method selection based on device/chain configuration
    if num_devices == 1:
        # Sequential method for single device
        log.debug("Using sequential method (single device)")
        samples_x = []
        logps = []
        for i in range(num_chains):
            samples_x_i, logps_i = run_single_chain(rng_keys[i], inits[i])
            samples_x.append(samples_x_i)
            logps.append(logps_i)
        samples_x = jnp.concatenate(samples_x)
        logps = jnp.concatenate(logps)
        
    elif num_devices >= num_chains and num_chains > 1:
        # Direct pmap method when devices >= chains
        log.debug("Using direct pmap method (devices >= chains)")
        pmapped = jax.pmap(run_single_chain, in_axes=(0, 0), out_axes=(0, 0))
        samples_x, logps = pmapped(rng_keys, inits)
        samples_x = jnp.concatenate(samples_x, axis=0)
        logps = jnp.concatenate(logps, axis=0)
        logps = jnp.reshape(logps, (samples_x.shape[0],))
        
    elif 1 < num_devices < num_chains:
        # Chunked method when devices < chains (but > 1 device)
        log.debug(f"Using chunked pmap method ({num_devices} devices < {num_chains} chains)")
        
        # Process chains in chunks of device count using the existing run_single_chain
        pmapped_chunked = jax.pmap(run_single_chain, in_axes=(0, 0), out_axes=(0, 0))
        
        all_samples = []
        all_logps = []
        
        for i in range(0, num_chains, num_devices):
            end_idx = min(i + num_devices, num_chains)
            chunk_keys = rng_keys[i:end_idx]
            chunk_inits = inits[i:end_idx]
            
            # Run chunk (pmap handles variable chunk sizes automatically)
            chunk_samples, chunk_logps = pmapped_chunked(chunk_keys, chunk_inits)
            
            all_samples.append(chunk_samples)
            all_logps.append(chunk_logps)
        
        # Concatenate all chunks
        samples_x = jnp.concatenate([jnp.concatenate(chunk, axis=0) for chunk in all_samples], axis=0)
        logps = jnp.concatenate([jnp.concatenate(chunk, axis=0) for chunk in all_logps], axis=0)

    samples_dict = {
        'x': samples_x,
        'logp': logps,
        'best': samples_x[jnp.argmax(logps)],
        'method': "MCMC"
    }

    log.debug(f"Max logl found in HMC = {np.max(logps):.4f}")

    return samples_dict