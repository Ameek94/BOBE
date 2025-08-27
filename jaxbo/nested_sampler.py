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
from .utils.logging_utils import get_logger
from .utils.seed_utils import get_numpy_rng
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
                       ,dynamic: bool = True
                       ,logz_std: bool = True
                       ,maxcall: Optional[int] = None
                        ,boost_maxcall: Optional[int] = 1
                        ,print_progress : bool = True
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
    progress : bool
        Print progress of the nested sampling run

    Returns
    -------
    samples : ndarray
        Equally weighted samples from the nested sampler
    logz_dict : dict
        Dictionary containing the mean, upper and lower bounds on logZ and the logZ error from the nested sampler
    """ 

    if maxcall is None:
        if ndim<=4:
            maxcall = int(5000*ndim*boost_maxcall) # type: ignore
        else:
            maxcall = max(int(10000*ndim*boost_maxcall),int(1e6)*boost_maxcall) # type: ignore
    else:
         maxcall = int(maxcall)

    @jax.jit
    def loglike(x):
        mu = gp.predict_mean_single(x) 
        return mu 

    start = time.time()

    success = True

    nlive = 500 if ndim <= 12 else 750

    if dynamic:
        sampler = DynamicNestedSampler(loglike,prior_transform,ndim=ndim,blob=False,
                                       sample=sample_method,nlive=nlive,rstate=rng)
        sampler.run_nested(print_progress=print_progress,dlogz_init=dlogz,maxcall=maxcall)
    else:
        sampler = StaticNestedSampler(loglike,prior_transform,ndim=ndim,blob=False,
                                      sample=sample_method,nlive=nlive,rstate=rng)
        sampler.run_nested(print_progress=print_progress,dlogz=dlogz,maxcall=maxcall)
        res = sampler.results  # type: ignore # grab our results
        logl = res['logl']
        # add check for all same logl values in case of initial plateau
        if np.all(logl == logl[0]):
            success = False
            print("All logl values are the same, this may indicate a problem with the model or the data. Retrying with the dynamic nested sampler.")
            sampler = DynamicNestedSampler(loglike,prior_transform,ndim=ndim,blob=False,
                                       sample=sample_method,nlive=nlive)
            sampler.run_nested(print_progress=print_progress,dlogz_init=dlogz,maxcall=maxcall)     
            # need a better method to guarantee that at least one finite logl present.
        
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

    var = np.clip(var,a_min=1e-6,a_max=1e2)
    E_Z = compute_integrals(logl=logl + 0.5*var ,logvol=logvol)[-1] 
    varintegrand = 2*logl + np.log(var) + np.log1p(1+var) # var + np.log(np.expm1(var)) #+ 
    log_var_delta = compute_integrals(logl=varintegrand,logvol=logvol,squared=True)[-1]
    log_var_logz = log_var_delta - 2*mean #2*E_Z 
    log_var_logz = np.clip(log_var_logz, a_min=-100, a_max=100)  # Avoid numerical issues with very small variances
    log.info(f"Log variance of logZ: {log_var_logz:.4f}, log_var_delta: {log_var_delta:.4f}, E_Z: {E_Z:.4f}")
    var_logz = np.exp(log_var_logz)
    logz_dict['upper'] = upper[-1]
    logz_dict['lower'] = lower[-1]
    logz_dict['var'] = var_logz
    logz_dict['std'] = np.sqrt(var_logz)
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
    samples_dict['method']= 'NS'
    return (samples_dict, logz_dict, success)