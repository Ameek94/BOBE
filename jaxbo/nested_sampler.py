# This module manages the nested samplers used to compute the Bayesian evidence with the GP model as a surrogate for the objective function
# The module contains two functions, one for the Dynesty sampler and the other for the JaxNS sampler (preferred)

import time
from typing import Any, List, Optional, Dict, Union
import jax.numpy as jnp
import jax.random as random
import numpy as np
import jax
from jax import config, vmap, jit
config.update("jax_enable_x64", True)
from .gp import GP
from scipy.special import logsumexp

try:
    from dynesty import NestedSampler as StaticNestedSampler,DynamicNestedSampler
except ModuleNotFoundError:
    print("Proceeding without dynesty since not installed")
import math

try:
    from nautilus import Sampler as NautilusSampler
except ModuleNotFoundError:
    print("Proceeding without nautilus since not installed")

import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions
from jaxns.framework.model import Model
from jaxns.framework.prior import Prior
from jaxns import NestedSampler, TerminationCondition, resample
import logging
log = logging.getLogger("[NS]")

def renormalise_log_weights(log_weights):
    log_total = logsumexp(log_weights)
    normalized_weights = np.exp(log_weights - log_total)
    return normalized_weights

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
                       ,dlogz: float = 0.2
                       ,dynamic: bool = True
                       ,logz_std: bool = True
                       ,maxcall: Optional[int] = None
                        ,boost_maxcall: Optional[int] = 1
                        ,progress : bool = True
                        ,equal_weights: bool = False
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
            maxcall = int(3000*ndim*boost_maxcall) # type: ignore
        else:
            maxcall = max(int(6000*ndim*boost_maxcall),60000*boost_maxcall) # type: ignore
    else:
         maxcall = int(maxcall)
         
    # def loglike(x,logz_std=logz_std) -> Any:
    #     mu, var = gp.posterior(x,single=True,unstandardize=True)
    #     mu = mu.squeeze(-1)
    #     var = var.squeeze(-1)
    #     # print(mu.shape)
    #     if logz_std:
    #         blob = np.zeros(2)
    #         std = np.sqrt(var)
    #         ul = mu + std
    #         ll = mu - std
    #         blob[0] = ll
    #         blob[1] = ul
    #         return mu, blob
    #     else:
    #         return mu     

    @jit #partial(jit,static_argnums=())
    def loglike(x):
        mu = gp.predict_mean(x) # vmap(f,in_axes=(0),out_axes=(0,0))(x)
        var = gp.predict_var(x) # vmap(f,in_axes=(0),out_axes=(0,0))(x)
        std = jnp.sqrt(var)
        mu = mu #.squeeze(-1)
        # print(f"mu shape: {mu.shape}, std shape: {std.shape}")
        return jnp.reshape(mu,()), jnp.reshape(std,()) #(mu - std, mu + std) # type: ignore

    start = time.time()
    if dynamic:
        sampler = DynamicNestedSampler(loglike,prior_transform,ndim=ndim,blob=logz_std,
                                       sample='rwalk') #,logl_args={'logz_std': logz_std})
        sampler.run_nested(print_progress=progress,dlogz_init=dlogz,maxcall=maxcall) #tune? ,maxcall=20000
    else:
        sampler = StaticNestedSampler(loglike,prior_transform,ndim=ndim,blob=logz_std,
                                      sample='rwalk') #,logl_args={'logz_std': logz_std}) # type: ignore
        sampler.run_nested(print_progress=progress,dlogz=dlogz,maxcall=maxcall) # type: ignore #tune? ,maxcall=20000
    log.info(f" Nested Sampling took {time.time() - start:.2f}s")
    res = sampler.results  # type: ignore # grab our results
    logl = res['logl']
    log.info(" Log Z evaluated using {} points".format(np.shape(logl))) 
    log.info(f" Dynesty made {np.sum(res['ncall'])} function calls")
    mean = res['logz'][-1]
    logz_err = res['logzerr'][-1]
    logz_dict = {'mean': mean}
    logz_dict['dlogz sampler'] = logz_err
    if logz_std:
        logl_lower,logl_upper = logl - res['blob'], logl + res['blob'] #res['blob'].T
        logvol = res['logvol']
        logl = res['logl']
        upper = compute_integrals(logl=logl_upper,logvol=logvol)
        lower = compute_integrals(logl=logl_lower,logvol=logvol)
        logz_dict = {'upper': upper[-1], 'mean': mean, 'lower': lower[-1],'dlogz sampler': logz_err}
    # samples = res.samples_equal()
    samples = {}
    if equal_weights:
        samples['x'] = res.samples_equal()
        weights = np.ones(samples['x'].shape[0])
        samples['weights'] = weights
    else:
        samples['x'] = res['samples']
        weights = renormalise_log_weights(res['logwt'])
        samples['weights'] = weights    
        samples['logl'] = res['logl']
    # samples['x'] = res['samples'] #res.samples_equal() #res['samples']
    # samples['logl'] = res['logl']
    # print(f"LogZ info: "+"".join(f"{key} = {value:.4f}, " for key, value in logz_dict.items()))
    return samples, logz_dict

#-------------JAXNS functions---------------------

def nested_sampling_jaxns(gp
                          ,ndim: int = 1
                          ,dlogZ: float = 0.1
                          ,evidence_uncert: float = 0.1
                          ,logz_std: bool = True
                          ,maxcall: int = 1e6 # type: ignore
                          ,boost_maxcall: int = 1
                          ,batch_size = 25 # what is the optimal size?
                          ,parameter_estimation = False
                          ,difficult_model = False
                          ,return_samples=False):
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
        
    @jit
    def log_likelihood(x):
        mu = gp.predict_mean(x) # vmap(f,in_axes=(0),out_axes=(0,0))(x)
        mu = mu #.squeeze(-1)
        return mu
        
    # @jit
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
    f = jit(lambda x: (gp.predict_var(x),))
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
    logl_var = jax.lax.map(f,results.samples['x'],batch_size=batch_size) # type: ignore

    logl_upper = results.log_L_samples + np.sqrt(logl_var)
    logl_lower = results.log_L_samples - np.sqrt(logl_var)
    
    upper =  compute_integrals(logl=logl_upper, logvol=logvol)[-1]
    lower = compute_integrals(logl=logl_lower, logvol=logvol)[-1]
    
    #Log evidence estimates
    log.info(f" Nested Sampling took {time.time() - start:.2f}s")
    log.info(f" jaxns did {results.total_num_likelihood_evaluations} likelihood evaluations") #, terminated due to {termination_reasons[results.termination_reason]}")
    # log.info(f" Mean LogZ: {mean}, Upper LogZ: {upper}, Lower LogZ: {lower}, Internal dLogZ: {logz_err}")
    logz_dict = {'upper': upper, 'mean': mean.item(), 'lower': lower,'dlogz sampler': logz_err.item()}

    #Get uniform samples for getdist
    # samples = resample(key=jax.random.PRNGKey(0),
    #                 samples=results.samples,
    #                 log_weights=results.log_dp_mean, # type: ignore
    #                 replace=True,) 
    

    samples = results.samples
    logwts = results.log_dp_mean
    weights = renormalise_log_weights(logwts)
    samples['weights'] = weights
    samples['logl'] = results.log_L_samples

    if return_samples:
        return samples, logz_dict
    else:
        return logz_dict
    # return np.array(samples['x']), logz_dict/

#-------------Nautilus functions---------------------

def nested_sampling_nautilus(gp,frac_remain: float = 0.01,n_like_max=1e6,return_samples=True) -> None:

    @jit
    def log_likelihood(x):
        mu = gp.predict_mean(x) # vmap(f,in_axes=(0),out_axes=(0,0))(x)
        mu = mu #.squeeze(-1)
        return mu
    
    def prior(x):
        return x
    
    ndim = gp.train_x.shape[1]
    sampler = NautilusSampler(prior, log_likelihood, ndim, vectorized=True, pool=(None,4))
    start = time.time()
    sampler.run(verbose=True, f_live=frac_remain, n_like_max=n_like_max)#, n_eff=2000*ndim)
    samples, logl, logwt = sampler.posterior()
    weights = renormalise_log_weights(logwt)
    end = time.time()
    print('Time taken: {:.4f} s'.format(end - start))
    print('log Z: {:.2f}'.format(sampler.log_z))
    logz_dict = { 'mean': sampler.log_z}
    samples_dict = {'x': samples, 'logl': logl, 'weights': weights}
    if return_samples:
        return samples_dict, logz_dict
    else:
        return logz_dict
