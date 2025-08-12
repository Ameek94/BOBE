# This module manages the nested samplers used to compute the Bayesian evidence with the GP model as a surrogate for the objective function
# The module contains two functions, one for the Dynesty sampler (preferred) and the other for the JaxNS sampler 

import os
# print(f"Setting XLA flags for JAX: {os.cpu_count()} CPU cores")
# os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={os.cpu_count()}'
import time
from typing import Any, List, Optional, Dict, Union
import jax.numpy as jnp
import jax.random as random
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from .gp import GP
from .logging_utils import get_logger
from .seed_utils import get_numpy_rng
from .utils import renormalise_log_weights, resample_equal
from scipy.special import logsumexp
log = get_logger("[ns]")

try:
    from dynesty import NestedSampler as StaticNestedSampler,DynamicNestedSampler, pool
except ModuleNotFoundError:
    print("Proceeding without dynesty since not installed")
import math

import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions
from jaxns.framework.model import Model
from jaxns.framework.prior import Prior
from jaxns import NestedSampler, TerminationCondition, resample


#-------------Dynesty functions---------------------
def prior_transform(x):
    return x

# dynesty utility function for computing evidence
def compute_integrals(logl=None, logvol=None, reweight=None):
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


def nested_sampling_Dy(gp: GP
                       ,ndim: int = 1
                       ,dlogz: float = 0.1
                       ,dynamic: bool = True
                       ,logz_std: bool = True
                       ,maxcall: Optional[int] = None
                        ,boost_maxcall: Optional[int] = 1
                        ,print_progress : bool = True
                        ,equal_weights: bool = False
                        ,sample_method='rwalk'
                       ,) -> tuple[np.ndarray,Dict]:
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
        mu = gp.predict_mean(x) 
        # var = gp.predict_var(x) 
        # std = jnp.sqrt(var)
        # mu = mu 
        return jnp.reshape(mu,()) #, jnp.reshape(std,()) 

    # loglike = gp.jitted_single_predict_mean

    start = time.time()

    success = True

    nlive = 500 if ndim <= 20 else 750

    if dynamic:
        sampler = DynamicNestedSampler(loglike,prior_transform,ndim=ndim,blob=False,
                                       sample=sample_method,nlive=nlive)
        sampler.run_nested(print_progress=print_progress,dlogz_init=dlogz,maxcall=maxcall)
    else:
        sampler = StaticNestedSampler(loglike,prior_transform,ndim=ndim,blob=False,
                                      sample=sample_method,nlive=nlive) 
        sampler.run_nested(print_progress=print_progress,dlogz=dlogz,maxcall=maxcall)
        res = sampler.results  # type: ignore # grab our results
        logl = res['logl']
        # add check for all same logl values in case of
        if np.all(logl == logl[0]):
            success = False
            log.warning("All logl values are the same, this may indicate a problem with the model or the data. Retrying with the dynamic nested sampler.")
            sampler = DynamicNestedSampler(loglike,prior_transform,ndim=ndim,blob=False,
                                       sample=sample_method,nlive=nlive)
            sampler.run_nested(print_progress=print_progress,dlogz_init=dlogz,maxcall=maxcall)     
        
    res = sampler.results
    mean = res['logz'][-1]
    logz_err = res['logzerr'][-1]
    logz_dict = {'mean': mean}
    logz_dict['dlogz sampler'] = logz_err
    samples_x = res['samples']
    logl = res['logl']
    success = ~np.all(logl == logl[0]) # in case of failure do not check convergence
    log.info(f" Nested Sampling took {time.time() - start:.2f}s")
    log.info(" Log Z evaluated using {} points".format(np.shape(logl))) 
    log.info(f" Dynesty made {np.sum(res['ncall'])} function calls, max value of logl = {np.max(logl):.4f}")
    if logz_std:
        var = jax.lax.map(gp.predict_var,samples_x,batch_size=100)
        std = np.sqrt(var.squeeze(-1))
        logl_lower,logl_upper = logl - std, logl + std
        logvol = res['logvol']
        print(f"shapes: logl: {logl.shape}, std: {std.shape}, logl_lower: {logl_lower.shape}, logl_upper: {logl_upper.shape}, logvol: {logvol.shape}")
        upper = compute_integrals(logl=logl_upper,logvol=logvol)
        lower = compute_integrals(logl=logl_lower,logvol=logvol)
        logz_dict['upper'] = upper[-1]
        logz_dict['lower'] = lower[-1]
    samples_dict = {}
    best_pt = samples_x[np.argmax(logl)]
    samples_dict['best'] = best_pt
    # if equal_weights:
    #     equal_samples, equal_logl = resample_equal(res['samples'], logl, res['logwt'], get_numpy_rng())
    #     samples['x'] = equal_samples
    #     samples['logl'] = equal_logl
    #     samples['weights'] = np.ones(len(equal_samples))
    # else:
    samples_dict['x'] = samples_x
    weights = renormalise_log_weights(res['logwt'])
    samples_dict['weights'] = weights    
    samples_dict['logl'] = logl
    return (samples_dict, logz_dict, success)

#-------------JAXNS functions---------------------

def nested_sampling_jaxns(gp
                          ,ndim: int = 1
                          ,dlogZ: float = 0.1
                          ,evidence_uncert: float = 0.1
                          ,logz_std: bool = True
                          ,maxcall: int = 1e6 # type: ignore
                          ,boost_maxcall: int = 1
                          ,batch_size = 100 # what is the optimal size?
                          ,parameter_estimation = False
                          ,difficult_model = False
                        ,equal_weights: bool = False):
    """
    Nested Sampling using JaxNS

    Arguments
    ---------
    gp : saas_fbgp
        Gaussian Process model
    ndim : int
        Number of dimensions
    dlogz : float
        Log evidence goal
    logz_std : bool
        Compute the upper and lower bounds on logZ using the GP uncertainty
    maxcall : int
        Maximum number of function calls
    boost_maxcall : int
        Boost the maximum number of function calls
    batch_size : int
        Batch size for computing the upper and lower bounds on logZ, used to manage memory
    parameter_estimation : bool
        Jaxns settings to get robust parameter estimation, see Jaxns documentation for more details
    difficult_model : bool  
        Jaxns settings to handle difficult models, see Jaxns documentation for more details

    Returns
    -------
    samples : ndarray
        Equally weighted samples from the nested sampler
    logz_dict : dict
        Dictionary containing the mean, upper and lower bounds on logZ and the logZ error from the nested sampler
    """

    success = True

    @jax.jit
    def log_likelihood(x):
        return  gp.predict_mean(x) 
        
    def prior_model():
        x = yield Prior(tfpd.Uniform(low=jnp.zeros(ndim), high= jnp.ones(ndim)), name='x') # type: ignore
        return x
    
    model_mean = Model(prior_model=prior_model,
              log_likelihood=log_likelihood)
    
    term_cond = TerminationCondition(evidence_uncert=evidence_uncert,dlogZ=dlogZ
                                     ,max_num_likelihood_evaluations=int(maxcall*boost_maxcall)) 
    
    start = time.time()
    log.info(" Running Jaxns for logZ computation")
    ns_mean = NestedSampler(model=model_mean,
                        max_samples=maxcall*boost_maxcall,
                        parameter_estimation=parameter_estimation,
                        difficult_model=difficult_model,)
                        #num_parallel_workers=10)
     # Run the sampler
    termination_reason, state = ns_mean(jax.random.PRNGKey(42),term_cond=term_cond)
    # Get the results
    results = ns_mean.to_results(termination_reason=termination_reason, state=state)

    # ns_mean.plot_cornerplot(results)
    
    mean = results.log_Z_mean
    logz_err = results.log_Z_uncert

    # Upper and Lower bound calculation
    logvol = results.log_X_mean

    # variance needs to be computed in batches
    f = jax.jit(lambda x: (gp.predict_var(x),))
    # num_inputs = len(results.samples['x'])
    # log.info(f" Computing upper and lower logZ using {num_inputs} points")
    # # batch_size = batch_size
    # num_batches = (num_inputs + batch_size - 1 ) // batch_size
    # input_arrays = (results.samples['x'],)
    # batch_idxs = [np.arange( i*batch_size, min( (i+1)*batch_size,num_inputs  )) for i in range(num_batches)]
    # res = [f(*tuple([arr[idx] for arr in input_arrays])) for idx in batch_idxs]
    # nres = len(res[0])
    # # now combine results across batches and function outputs to return a tuple (num_outputs, num_inputs, ...)
    # logl_var = tuple(np.concatenate([x[i] for x in res]) for i in range(nres))[0]
    
    # can we use map instead?
    logl_var = jax.lax.map(gp.predict_var,results.samples['x'],batch_size=100)
     #jax.lax.map(f,results.samples['x'],batch_size=batch_size) # type: ignore
    logl_std = np.sqrt(logl_var.squeeze(-1))

    logl_upper = results.log_L_samples + logl_std
    logl_lower = results.log_L_samples - logl_std

    print(f"shapes logvol {logvol.shape}, logvar {logl_var.shape}, logl_upper {logl_upper.shape}, logl_lower {logl_lower.shape}")

    upper =  compute_integrals(logl=logl_upper, logvol=logvol)[-1]
    lower = compute_integrals(logl=logl_lower, logvol=logvol)[-1]
    
    #Log evidence estimates
    log.info(f" Nested Sampling took {time.time() - start:.2f}s")
    log.info(f" jaxns did {results.total_num_likelihood_evaluations} likelihood evaluations") #, terminated due to {termination_reasons[results.termination_reason]}")
    # log.info(f" Mean LogZ: {mean}, Upper LogZ: {upper}, Lower LogZ: {lower}, Internal dLogZ: {logz_err}")
    logz_dict = {'upper': upper, 'mean': mean.item(), 'lower': lower,'dlogz sampler': logz_err.item()}


    ns_samples = {}
    if equal_weights:
        samples = resample(key=jax.random.PRNGKey(0),
                    samples=results.samples,
                    log_weights=results.log_dp_mean, # type: ignore
                    replace=True,) 
        weights = np.ones(samples.shape[0])

    else:    
        samples = results.samples
        logwts = results.log_dp_mean
        weights = renormalise_log_weights(logwts)
    
    ns_samples['x'] = samples
    ns_samples['weights'] = weights

    return (ns_samples, logz_dict, success)