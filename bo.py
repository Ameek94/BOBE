# The main sampler class

import pandas as pd
import numpy as np
from typing import Callable, List,Optional, Tuple, Dict
from fb_gp import saas_fbgp
import time
import jax.numpy as jnp
from jax import random,vmap, grad
from acquisition import EI, IPV, optim_scipy_bh
from scipy.stats import qmc
from jaxns import NestedSampler
from nested_sampler import nested_sampling_jaxns, nested_sampling_Dy
from utils import input_standardize,input_unstandardize, output_standardize, output_unstandardize, plot_gp
import logging
log = logging.getLogger("[BO]")



def test_function(x):
    return -0.5*jnp.sum((x-0.5)**2/0.04,axis=-1,keepdims=True)


class sampler:

    def __init__(self,
            ndim: int,
            loglike =  None, # for external functions this should be the logposterior 
            param_list: Optional[List] = None,
            param_bounds: Optional[List] = None, # shape 2 x D
            param_labels: Optional[List] = None,
            gp_kwargs: Optional[dict] = None,
            gp: str = 'default',
            noise: float = 1e-4,
            gpfit_step: int = 1, 
            nstart: int = 8,
            max_steps: int = 4,
            acq_goal: float = 1e-6, # currently arbitrary...
            save: Optional[bool] = False,
            nested_sampler: str = "default", #jaxns 
            ns_step: int = 1, 
            mc_points_size: int = 16,
            nested_sampler_kwargs: Optional[dict] = None,
            final_ns_dlogz: float = 0.01,
            feedback_lvl: int = 1,
            seed: int = 0,) -> None:
        
        self.feedback = feedback_lvl

        self.max_steps = max_steps
        self.final_dlogz = final_ns_dlogz # the precision for the final run of the nested sampler
        self.nested_sampler_kwargs = nested_sampler_kwargs # not used yet
        self.ns_step = ns_step
        self.curr_step = 0
        self.gpfit_step = gpfit_step
        self.acq_goal  = acq_goal
        self.acq_val = 1e100
        self.num_step = 0

        self.mc_points_size = mc_points_size

        self.loglike = loglike if loglike is not None else test_function # assuming loglike returns N x 1 shape for input N x d
        self.nstart = nstart
        self.ndim = ndim # check consistency of provided quantities
        self.param_list = param_list if param_list is not None else ['x_%i'%(i+1) for i in range(ndim)]
        self.param_labels = param_labels if param_labels is not None else ['x_%i'%(i+1) for i in range(ndim)]
        self.param_bounds = param_bounds if param_bounds is not None else np.array(ndim*[[0,1]]).T
        log.info(f"\tRunning the sampler with params {self.param_labels}")
        log.info(f"\tParameter bounds {self.param_bounds.T}")

        #output and timing dataframes


        #initialize train_x, train_y and run GP
        self.train_x = qmc.Sobol(ndim, scramble=True).random(nstart)
        self.train_y = self.loglike(self.train_x)
        self.train_x = input_standardize(self.train_x,self.param_bounds)
        # train_y = output_standardize(train_y)
        self.gp_kwargs = gp_kwargs
        self.noise = noise
        self.gp = saas_fbgp(self.train_x,self.train_y,noise=self.noise)
        rng_key, _ = random.split(random.PRNGKey(seed), 2)
        self.gp.fit(rng_key,warmup_steps=256,num_samples=256,thinning=16,verbose=False) # input settings


    def run(self):
        self.converged = False
        while not self.converged:
            if (self.num_step%self.ns_step==0):
                samples, logz_dict = samples, logz_dict = nested_sampling_jaxns(self.gp,ndim=self.ndim,dlogz=0.1)
                log.info(f"\tLogZ info :"+"".join(f"{key}: = {value:.4f}, " for key, value in logz_dict.items()))
                size = len(samples)
                mc_points = samples[::int(size/self.mc_points_size),:]
            acq_func = IPV(self.gp,mc_points)
            grad_fn = grad(acq_func)
            x0 =  np.random.uniform(0,1,self.ndim)
            results = optim_scipy_bh(acq_func,x0=x0,stepsize=1/4,
                                      niter=15,minimizer_kwargs={'jac': grad_fn, 'bounds': self.ndim*[(0,1)] })
            self.acq_val = abs(results.fun)
            next_x = jnp.atleast_2d(results.x)
            log.info(f"\tNext point at x = {results.x} with acquisition function value = {results.fun:.4e}")
            next_y = self.loglike(next_x)
            self.train_x = jnp.concatenate([self.train_x,next_x])
            self.train_y = jnp.concatenate([self.train_y,next_y])
            seed = self.num_step
            rng_key, _ = random.split(random.PRNGKey(seed), 2)
            if (self.num_step%self.gpfit_step==0):
                self.gp = saas_fbgp(self.train_x,self.train_y,noise=self.noise)
                self.gp.fit(rng_key,warmup_steps=256,num_samples=256,thinning=16,verbose=False) # change rng key?
            else:
                self.gp.quick_update(next_x,next_y)
            log.info(f"\t----------------------Step {self.num_step+1} complete----------------------\n")
            self.num_step+=1
            self.converged = self.check_converged()
            if self.converged:
                log.info("\tRun Complete")
        samples, logz_dict = samples, logz_dict = nested_sampling_jaxns(self.gp,ndim=self.ndim,dlogz=0.01,difficult_model=True)
        log.info(f"\tFinal LogZ info :"+"".join(f"{key}: = {value:.4f}, " for key, value in logz_dict.items()))
        # plot_gp(samples,self.gp,self.param_list,self.param_labels,self.param_bounds)

    def check_converged(self):
        acq = (self.acq_val < self.acq_goal)
        steps = (self.num_step >= self.max_steps)
        if acq:
            log.info("\tAcquisition goal reached")
        if steps:
            log.info("\tMax steps reached")
        return acq or steps
