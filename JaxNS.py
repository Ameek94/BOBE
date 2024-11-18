import time
from typing import Any, List, Optional, Dict, Union
from kiwisolver import Term
from numpyro.util import enable_x64
import jax.numpy as jnp
import jax.random as random
import numpy as np
import jax
from jax import config, vmap
config.update("jax_enable_x64", True)
from numpyro.util import enable_x64
enable_x64()
from JaxFBGP import saas_fbgp
from GPUtils import split_vmap
import functools

try:
    from dynesty import NestedSampler,DynamicNestedSampler
except ModuleNotFoundError:
    print("Proceeding without dynesty since not installed")
import math

import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions
from jaxns.framework.model import Model
from jaxns.framework.prior import Prior
from jaxns import NestedSampler, TerminationCondition, resample
import logging
log = logging.getLogger("[NS]")


class gp_likelihood:

    def __init__(self,
                 gp: saas_fbgp,) -> None:
        self.gp = gp
        pass

    def __call__(self, X,logz_std=True,) -> Any:
        mu, var = self.gp.posterior(X,single=True,unstandardize=True)
        mu = mu.squeeze(-1)
        var = var.squeeze(-1)
        # print(mu.shape)
        blob = np.zeros(2)
        if logz_std:
            std = jnp.sqrt(var)
            ul = mu + std
            ll = mu - std
            blob[0] = ll
            blob[1] = ul
            return mu, blob
        else:
            return mu 

#-------------Dynesty functions---------------------
def prior_transform(x):
    return x

# dynesty utility for computing evidence
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


def nested_sampling_Dy(gp: saas_fbgp
                       ,ndim: int = 1
                       ,dlogz: float = 0.1
                       ,dynamic: bool = True
                       ,logz_std: bool = True
                       ,maxcall: Optional[int] = None
                        ,boost_maxcall: Optional[int] = 1
                       ) -> tuple[np.ndarray,Dict]:
    if maxcall is None:
        if ndim<=4:
            maxcall = int(3000*ndim*boost_maxcall) # type: ignore
        else:
            maxcall = max(int(6000*ndim*boost_maxcall),60000*boost_maxcall) # type: ignore
    else:
         maxcall = int(maxcall)
         
    #loglike = lambda x: gplikelihood_FB(model=model,temp=temp,interp_logp=interp_logp,blob=blob,x=x)
    loglike =  gp_likelihood(gp=gp)
    start = time.time()
    if dynamic:
        sampler = DynamicNestedSampler(loglike,prior_transform,ndim=ndim,blob=logz_std,logl_args={'logz_std': logz_std})
        sampler.run_nested(print_progress=False,dlogz_init=dlogz,maxcall=maxcall) #tune? ,maxcall=20000
    else:
        sampler = NestedSampler(loglike,prior_transform,ndim=ndim,blob=logz_std,logl_args={'logz_std': logz_std}) # type: ignore
        sampler.run_nested(print_progress=False,dlogz=dlogz,maxcall=maxcall) # type: ignore #tune? ,maxcall=20000
    log.info(f" Nested Sampling took {time.time() - start:.2f}s")
    res = sampler.results  # type: ignore # grab our results
    logl = res['logl']
    log.info(" Log Z evaluated using {} points".format(np.shape(logl))) 
    log.info(f" Dynesty made {np.sum(res['ncall'])} function calls")
    logl_lower,logl_upper = res['blob'].T
    logvol = res['logvol']
    logl = res['logl']
    upper = compute_integrals(logl=logl_upper,logvol=logvol)
    lower = compute_integrals(logl=logl_lower,logvol=logvol)
    mean = np.float64(res['logz'][-1])
    logz_err = res['logzerr'][-1]
    logz_dict = {'mean': mean,'upper': upper[-1], 'lower': lower[-1],'dlogz sampler': logz_err}
    samples = res.samples_equal()
    print(logl, logvol)
    return samples, logz_dict

#-------------JAXNS functions---------------------

# currently there is no simple way to get the GP upper and lower limits as we do in dynesty, need to implement
def nested_sampling_jaxns(gp: saas_fbgp
                          ,ndim: int = 1
                          ,dlogz: float = 0.1
                          ,logz_std: bool = True
                          ,maxcall: int = 1e4 # type: ignore
                          ,post_prior_ratio: int = 1 #Ratio of posterior density to prior volume [1, 2]
                          ,boost_maxcall: int = 1
                          ,boost_livepoints: int = 100
                          ,num_samples_equal=1000
                          ,difficult_model = False) -> tuple[np.ndarray,Dict]:
        
    def log_likelihood(x):
        mu, var = gp.posterior(x,single=True,unstandardize=True) # vmap(f,in_axes=(0),out_axes=(0,0))(x)
        mu = mu.squeeze(-1)
        return mu
        
    def prior_model():
        x = yield Prior(tfpd.Uniform(low=jnp.zeros(ndim), high= jnp.ones(ndim)), name='x') # type: ignore
        return x


    model_mean = Model(prior_model=prior_model,
              log_likelihood=log_likelihood)
    
    term_cond = TerminationCondition(evidence_uncert=1e-2) 

    if ndim < 6:
        num_live_points = np.sqrt(ndim)/(0.01**2)*boost_livepoints 
        max_samples = np.sqrt(ndim)/(0.01**2)*(post_prior_ratio*ndim)
    elif ndim < 10 and ndim >= 6:
        num_live_points =  ndim/(0.01**2)*boost_livepoints
        max_samples = ndim/(0.01**2)*(post_prior_ratio*ndim)
    if ndim >=10:
        num_live_points = (ndim**(3/2))/(0.01**2)*boost_livepoints
        max_samples = (ndim**(3/2))/(0.01**2)*(post_prior_ratio*ndim)
    
    
    start = time.time()
    ns_mean = NestedSampler(model=model_mean,
                            parameter_estimation=True,
                            difficult_model=difficult_model,
                            num_live_points=num_live_points,
                            max_samples=max_samples)
     # Run the sampler
    termination_reason, state = ns_mean(jax.random.PRNGKey(42),term_cond=term_cond)
    # Get the results
    results = ns_mean.to_results(termination_reason=termination_reason, state=state)
    #ns_mean.summary(results)
    ns_mean.plot_cornerplot(results)
    
    mean = results.log_Z_mean
    logz_err = results.log_Z_uncert
    logvol = results.log_X_mean


    '''vmap_func = lambda x: gp.posterior(x, single=True)
    _, logl_var = split_vmap(vmap_func, results.samples['x'], 10)'''
    
    ### Attempt to use vmap for var and mean ###
    num_inputs = len(results.samples['x'])
    batch_size = 1000
    num_batches = (num_inputs + batch_size - 1 ) // batch_size
    input_arrays = (results.samples['x'],)
    batch_idxs = [jnp.arange( i*batch_size, min( (i+1)*batch_size,num_inputs  )) for i in range(num_batches)]
    f = lambda x: gp.posterior(x, single=True)
    res = [f(*tuple([arr[idx] for arr in input_arrays])) for idx in batch_idxs]
    nres = len(res[0])
    vmap_results = tuple(jnp.concatenate([x[i] for x in res]) for i in range(nres))
    _, logl_var = vmap_results
    
    '''# Upper and Lower bound calculation
    logvol = results.log_X_mean
    #log.info("Getting variance at live points")
    vmap_func = lambda x :  gp.posterior(x, single=True)   
    vmap_arrays = ([results.samples['x']])
    vmap_size = 8

    _, logl_var = split_vmap(vmap_func,vmap_arrays,batch_size=vmap_size)'''

    
    logl_std = jnp.sqrt(logl_var)

    #log.info("Calculating upper and lower logl values")
    logl_upper = results.log_L_samples + logl_std
    
    logl_lower = results.log_L_samples - logl_std
    #log.info("Calculating upper and lower integral bounds")
    
    upper =  compute_integrals(logl=np.array(logl_upper), logvol=logvol)[-1]
    lower = compute_integrals(logl=np.array(logl_lower), logvol=logvol)[-1]

    termination_reasons = ['Reached max samples',
                           'Evidence uncertainty low enough',
                            'Small remaining evidence',
                            'Reached ESS',
                            "Used max num likelihood evaluations",
                            'Likelihood contour reached',
                            'Sampler efficiency too low',
                            'All live-points are on a single plateau (sign of possible precision error)',
                            'relative spread of live points < rtol',
                            'absolute spread of live points < atol',
                            'no seed points left (consider decreasing shell_fraction)']
    
    #Log evidence estimates
    log.info(f" Mean LogZ: {mean}, Upper LogZ: {upper}, Lower LogZ: {lower}, Internal dLogZ: {logz_err}")
    log.info(f" Nested Sampling took {time.time() - start:.2f}s")
    log.info(results.termination_reason)
    log.info(f" jaxns did {results.total_num_likelihood_evaluations} likelihood evaluations, terminated due to {termination_reasons[results.termination_reason-1]}")
    logz_dict = {'mean': mean,'upper': upper, 'lower': lower,'dlogz sampler': logz_err}

    #Get uniform samples for acq function
    samples = resample(key=jax.random.PRNGKey(0),
                    samples=results.samples,
                    log_weights=results.log_dp_mean, # type: ignore
                    S=num_samples_equal*ndim, # type: ignore # check with effective sample size...
                    replace=True,) 
    
    return np.array(samples['x']), logz_dict