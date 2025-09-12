# This module manages the nested samplers used to compute the Bayesian evidence with the GP model as a surrogate for the objective function
# The module contains two functions, one for the Dynesty sampler (preferred) and the other for the JaxNS sampler 
import time
from typing import Any, List, Optional, Dict, Union
import jax.numpy as jnp
import jax.random as random
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from .gp import GP
from .clf_gp import GPwithClassifier
from .utils.logging_utils import get_logger
from .utils.seed_utils import get_numpy_rng
from .utils.core_utils import is_cluster_environment
from scipy.special import logsumexp
log = get_logger("ns")

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

def renormalise_log_weights(log_weights):
    log_total = logsumexp(log_weights)
    normalized_weights = np.exp(log_weights - log_total)
    return normalized_weights

def resample_equal(samples, aux, weights=None, logwts=None, rng = None):
    rng = get_numpy_rng() if rng is None else rng
    # Resample samples to obtain equal weights. Taken from jaxns
    if logwts is not None:
        wts = renormalise_log_weights(logwts)
    else:
        wts = weights
    weights = wts / wts.sum()
    cumulative_sum = np.cumsum(weights)
    cumulative_sum /= cumulative_sum[-1]
    nsamples = len(weights)
    positions = (rng.random() + np.arange(nsamples)) / nsamples
    idx = np.zeros(nsamples, dtype=int)
    i, j = 0, 0
    while i < nsamples:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    perm = rng.permutation(nsamples)
    resampled_samples = samples[idx][perm]
    resampled_aux = aux[idx][perm]
    return resampled_samples, resampled_aux


def prior_transform(x):
    return x
        

def nested_sampling_Dy(gp: GP
                       ,ndim: int = 1
                       ,dlogz: float = 0.1
                       ,dynamic: bool = False
                       ,maxcall: Optional[int] = int(5e6)
                        ,print_progress : Optional[bool] = True
                        ,equal_weights: bool = False
                        ,sample_method='rwalk',
                        rng=None,
                       ) -> tuple[np.ndarray,Dict,bool]:
    """
    Nested Sampling using Dynesty

    Arguments
    ---------
    gp : saas_fbgp
        Gaussian Process model
    ndim : int
        Number of dimensions
    dlogz : float
        Log evidence goal
    dynamic : bool
        Use dynamic nested sampling, see Dynesty documentation for more details
    logz_std : bool
        Compute the upper and lower bounds on logZ using the GP uncertainty
    maxcall : int
        Maximum number of function calls
    boost_maxcall : int
        Boost the maximum number of function calls
    print_progress : bool, optional
        Print progress of the nested sampling run. If None, automatically disables 
        progress printing in cluster environments and enables it otherwise.
    equal_weights : bool
        Resample to obtain equal weights
    sample_method : str
        Sampling method for dynesty
    rng : random number generator
        Random number generator

    Returns
    -------
    samples : ndarray
        Equally weighted samples from the nested sampler
    logz_dict : dict
        Dictionary containing the mean, upper and lower bounds on logZ and the logZ error from the nested sampler
    success : bool
        Whether the nested sampling run was successful
    """

    # Auto-detect cluster environment if print_progress not explicitly set
    if is_cluster_environment():
        print_progress = False

    start = time.time()

    nlive = 500 if ndim <= 10 else 750

    success = False
    max_tries = 1000 # we can do lots since this is very fast in case of failure
    n_tried = 0
    while not success:  # loop in case of failure
        if dynamic:
            sampler = DynamicNestedSampler(gp.predict_mean_single,prior_transform,ndim=ndim,blob=False,
                                       sample=sample_method,nlive=nlive,rstate=rng)
            sampler.run_nested(print_progress=print_progress,dlogz_init=dlogz,maxcall=maxcall)
        else:
            sampler = StaticNestedSampler(gp.predict_mean_single,prior_transform,ndim=ndim,blob=False,
                                      sample=sample_method,nlive=nlive,rstate=rng)
            sampler.run_nested(print_progress=print_progress,dlogz=dlogz,maxcall=maxcall)
        res = sampler.results  # type: ignore # grab our results
        logl = res['logl']
        n_tried += 1
        # add check for all same logl values in case of initial plateau
        if np.all(logl == logl[0]):
            success = False
            log.debug(f" All logl values are the same on try {n_tried+1}/{max_tries}. Retrying...")
        else:
            success = True
            log.info(f" Successful result on try {n_tried+1}/{max_tries}.")
        if n_tried >= max_tries:
            log.warning("Nested sampling failed after maximum retries. Exiting.")
            break
        if n_tried==50 or n_tried==100 or n_tried==500:
            nlive = 2*nlive
            log.warning(f"Unable to get non-constant logl values in {n_tried} tries. Retrying with increased nlive.")

        
        # nlive = 2*nlive
        # log.warning("All initial logl values are the same. Retrying with the dynamic nested sampler and increased nlive.")
        # sampler = DynamicNestedSampler(loglike,prior_transform,ndim=ndim,blob=False,
        #                                sample=sample_method,nlive=nlive)
        # sampler.run_nested(print_progress=print_progress,dlogz_init=dlogz,maxcall=maxcall)     
        #     # need a better method to guarantee that at least one finite logl present.
        
    res = sampler.results
    mean = res['logz'][-1]
    logz_err = res['logzerr'][-1]
    logz_dict = {'mean': mean}
    logz_dict['dlogz_sampler'] = logz_err
    samples_x = res['samples']
    logl = res['logl']
    success = ~np.all(logl == logl[0]) # in case of failure do not check convergence
    log.info(f" Nested Sampling took {time.time() - start:.2f}s")
    log.info(" Log Z evaluated using {} points".format(np.shape(logl))) 
    log.info(f" Dynesty made {np.sum(res['ncall'])} function calls, max value of logl = {np.max(logl):.4f}")

    var = jax.lax.map(gp.predict_var_single,samples_x,batch_size=100)
    std = np.sqrt(var)
    logl_lower,logl_upper = logl - std, logl + std
    logvol = res['logvol']
    upper = compute_integrals(logl=logl_upper,logvol=logvol)
    lower = compute_integrals(logl=logl_lower,logvol=logvol)

    var = np.clip(var,a_min=1e-8,a_max=1e6)
    varintegrand = 2*logl + np.log(var) #+ np.log1p(var)
    log_var_delta = compute_integrals(logl=varintegrand,logvol=logvol,squared=True)[-1]
    log_var_logz = log_var_delta - 2*mean 
    log_var_logz = np.clip(log_var_logz, a_min=-100, a_max=100)  # Avoid numerical issues with very small or large variances
    var_logz = np.exp(log_var_logz)
    logz_dict['upper'] = upper[-1]
    logz_dict['lower'] = lower[-1]
    logz_dict['var'] = var_logz
    logz_dict['std'] = 2*np.sqrt(var_logz) # 2 sigma
    samples_dict = {}
    best_pt = samples_x[np.argmax(logl)]
    samples_dict['best'] = best_pt
    weights = renormalise_log_weights(res['logwt'])
    if equal_weights: #for MC points
        samples_x, logl = resample_equal(samples_x, logl, weights=weights,rng=rng)
        weights = np.ones(samples_x.shape[0])  # Equal weights after resampling
    samples_dict['x'] = samples_x
    samples_dict['weights'] = weights    
    samples_dict['logl'] = logl
    samples_dict['logl_upper'] = logl_upper
    samples_dict['logl_lower'] = logl_lower
    samples_dict['logvol'] = logvol
    samples_dict['method']= 'nested'
    return (samples_dict, logz_dict, success)