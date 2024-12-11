import time
from typing import Any, List, Optional, Dict, Union
from kiwisolver import Term
from numpyro.util import enable_x64
import jax.numpy as jnp
import jax.random as random
import numpy as np
import jax
from jax import config, vmap, jit
config.update("jax_enable_x64", True)
from numpyro.util import enable_x64
enable_x64()
from fb_gp import saas_fbgp

try:
    from dynesty import NestedSampler,DynamicNestedSampler
except ModuleNotFoundError:
    print("Proceeding without dynesty since not installed")
import math

import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions
from jaxns.framework.model import Model
from jaxns.framework.prior import Prior
from jaxns import NestedSampler as JaxNestedSampler
from jaxns import TerminationCondition, resample

try:
    from dynesty import NestedSampler as DynestyNestedSampler
    from dynesty import DynamicNestedSampler as DynestyDynamicNestedSampler
except ModuleNotFoundError:
    print("Proceeding without dynesty since not installed")

import logging
log = logging.getLogger("[NS]")



class NestedSampler():
    
    def __init__(self
                ,gp: saas_fbgp
                ,ndim: int
                ,ns_kwargs: dict[str,Any]
                ,name = "") -> None:
        self.gp = gp
        self.ndim = ndim
        self.ns_kwargs = ns_kwargs
        self.dlogz_goal = self.ns_kwargs['dlogz_goal']
        self.final_ns_dlogz = self.ns_kwargs['final_ns_dlogz']
        self.max_call = self.ns_kwargs['max_call']
        #self.ns_method = ns_settings['method']
        #self.ns_kwargs = ns_settings[self.ns_method]
        self.name = name
        log.info(f" Initialised {name} as Nested Sampler with settings {self.ns_kwargs}")
        pass

    def run(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def compute_integrals(self, logl=None, logvol=None, reweight=None):
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

#-------------JAXNS functions---------------------

class JaxNS(NestedSampler):
    
    def __init__(self
                ,gp: saas_fbgp
                ,ns_kwargs: dict[str,Any]
                #,dlogz_goal: float
                #,final_ns_dlogz: float
                ,ndim: int
                #,max_call: int
                #,difficult_model: bool
                ,batch_size: int = 50) -> None:
        
        super().__init__(gp, ndim, ns_kwargs, name="JaxNS")
        self.difficult_model=ns_kwargs['difficult_model']
        self.batch_size = batch_size
    
    def log_likelihood(self, x):
        mu, var = self.gp.posterior(x,single=True,unstandardize=True)
        mu = mu.squeeze(-1)
        return mu
        
    def prior_model(self):
        x = yield Prior(tfpd.Uniform(low=jnp.zeros(self.ndim), high= jnp.ones(self.ndim)), name='x') # type: ignore
        return x


    def run(self, final_run: bool, boost_maxcall: int = 1):
        
        
        model = Model(prior_model=self.prior_model, log_likelihood=self.log_likelihood)
        if final_run:
            difficult_model = True
            term_cond = TerminationCondition(evidence_uncert=self.final_ns_dlogz, max_samples=self.max_call*boost_maxcall)
        else:
            difficult_model = self.difficult_model
            term_cond = TerminationCondition(evidence_uncert=self.dlogz_goal, max_samples=self.max_call*boost_maxcall)
        
        start = time.time()
        log.info(" Running Jaxns for logZ computation")
        ns = JaxNestedSampler(model=model,
                            parameter_estimation=True,
                            difficult_model=difficult_model)
                
        # Run the sampler
        termination_reason, state = ns(jax.random.PRNGKey(42),term_cond=term_cond)
        # Get the results
        results = ns.to_results(termination_reason=termination_reason, state=state)
        
        mean = results.log_Z_mean
        logz_err = results.log_Z_uncert
    
        # Upper and Lower bound calculation
        logvol = results.log_X_mean
    
        # variance needs to be computed in batches
        num_inputs = len(results.samples['x'])
        log.info(f" Computing upper and lower logZ using {num_inputs} points")
        #batch_size = batch_size
        num_batches = (num_inputs + self.batch_size - 1 ) // self.batch_size
        f = lambda x: self.gp.posterior(x,single=True)
        input_arrays = (results.samples['x'],)
        batch_idxs = [np.arange( i*self.batch_size, min( (i+1)*self.batch_size,num_inputs  )) for i in range(num_batches)]
        res = [f(*tuple([arr[idx] for arr in input_arrays])) for idx in batch_idxs]
        nres = len(res[0])
        # now combine results across batches and function outputs to return a tuple (num_outputs, num_inputs, ...)
        logl_mean, logl_var = tuple(np.concatenate([x[i] for x in res]) for i in range(nres))
        
        logl_upper = results.log_L_samples + np.sqrt(logl_var)
        logl_lower = results.log_L_samples - np.sqrt(logl_var)
        
        upper =  self.compute_integrals(logl=logl_upper, logvol=logvol)[-1]
        lower = self.compute_integrals(logl=logl_lower, logvol=logvol)[-1]
        
        #Log evidence estimates
        log.info(f" Nested Sampling took {time.time() - start:.2f}s")
        log.info(f" JaxNS did {results.total_num_likelihood_evaluations} likelihood evaluations")
        log.info(f" Log Z evaluated using {np.shape(logl_var)} points")
        # log.info(f" Mean LogZ: {mean}, Upper LogZ: {upper}, Lower LogZ: {lower}, Internal dLogZ: {logz_err}")
        logz_dict = {'upper': upper, 'mean': mean, 'lower': lower,'dlogz sampler': logz_err}
    
        #Get uniform samples for acq function
        samples = resample(key=jax.random.PRNGKey(0),
                        samples=results.samples,
                        log_weights=results.log_dp_mean, # type: ignore
                        replace=True,) 
        
        return np.array(samples['x']), logz_dict

#-------------Dynesty functions---------------------
class DynestyNS(NestedSampler):
    def __init__(self
                 ,gp: saas_fbgp
                 ,ns_kwargs: dict[str,Any]
                 ,ndim: int):
        super().__init__(gp, ndim, ns_kwargs, name="Dynesty")
        self.dynamic = ns_kwargs['dynamic']
        
    def prior_transform(self, x):
        return x
        
    def loglike(self, x) -> Any:
        mu, var = self.gp.posterior(x,single=True,unstandardize=True)
        mu = mu.squeeze(-1)
        var = var.squeeze(-1)
        # print(mu.shape)
        blob = np.zeros(2)
        std = np.sqrt(var)
        ul = mu + std
        ll = mu - std
        blob[0] = ll
        blob[1] = ul
        return mu, blob
          

    def run(self, final_run: bool, boost_maxcall: int = 1):
        if final_run:
            dlogz_goal = self.final_ns_dlogz
        else:
            dlogz_goal = self.dlogz_goal
        log.info(" Running Dynesty for logZ computation")
        start = time.time()
        if self.dynamic:
            sampler = DynestyDynamicNestedSampler(self.loglike,self.prior_transform,ndim=self.ndim,blob=True)
            sampler.run_nested(print_progress=False,dlogz_init=dlogz_goal,maxcall=self.max_call) #tune? ,maxcall=20000
        else:
            sampler = DynestyNestedSampler(self.loglike,self.prior_transform,ndim=self.ndim,blob=True) # type: ignore
            sampler.run_nested(print_progress=False,dlogz=self.dlogz_goal,maxcall=self.max_call) # type: ignore #tune? ,maxcall=20000
        res = sampler.results  # type: ignore # grab our results
        logl = res['logl']
        mean = res['logz'][-1]
        logz_err = res['logzerr'][-1]
        logz_dict = {'mean': mean}
        logz_dict['dlogz sampler'] = logz_err
        logl_lower,logl_upper = res['blob'].T
        logvol = res['logvol']
        logl = res['logl']
        upper = self.compute_integrals(logl=logl_upper,logvol=logvol)
        lower = self.compute_integrals(logl=logl_lower,logvol=logvol)
        logz_dict = {'mean': mean,'upper': upper[-1], 'lower': lower[-1],'dlogz sampler': logz_err}
        samples = res.samples_equal()
        log.info(f" Nested Sampling took {time.time() - start:.2f}s")
        log.info(f" Dynesty did {np.sum(res['ncall'])} likelihood evaluations")
        log.info(f" Log Z evaluated using {np.shape(logl)} points") 
        return samples, logz_dict