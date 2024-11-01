# The main sampler class

import functools
import pandas as pd
import numpy as np
from typing import Callable, List,Optional, Tuple, Dict
from fb_gp import saas_fbgp, sample_GP_NUTS
import time
import jax.numpy as jnp
from jax import random,vmap, grad
from acquisition import EI, IPV, optim_scipy_bh
from scipy.stats import qmc
from jaxns import NestedSampler
from nested_sampler import nested_sampling_jaxns, nested_sampling_Dy
from bo_utils import input_standardize,input_unstandardize, output_standardize, output_unstandardize, plot_gp
import logging
log = logging.getLogger("[BO]")



def test_function(x):
    return -0.5*jnp.sum((x-0.5)**2/0.04,axis=-1,keepdims=True)


class sampler:

    def __init__(self,
            ndim = None,
            cobaya_model = False,
            input_file = None,
            loglike =  None, # for external functions this should be the logposterior 
            param_list: Optional[List] = None,
            param_bounds: Optional[List] = None, # shape 2 x D
            param_labels: Optional[List] = None,
            gp_kwargs: Optional[dict] = None,
            gp: str = 'default',
            noise: float = 1e-4,
            gpfit_step: int = 1, 
            nstart: int = 4,
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

        if cobaya_model:
            self._cobaya_init(input_file)
        else:
            assert loglike is not None
            self.logp = functools.partial(self._ext_logp,loglike=loglike) # assuming loglike returns N x 1 shape for input N x d, eventually add option to define loglike model in external file
            self.param_list = param_list if param_list is not None else ['x_%i'%(i+1) for i in range(ndim)]
            self.param_labels = param_labels if param_labels is not None else ['x_%i'%(i+1) for i in range(ndim)]
            self.param_bounds = np.array(param_bounds) if param_bounds is not None else np.array(ndim*[[0,1]]).T
        self.ndim = len(self.param_list)
        self.bounds_dict = dict(zip(self.param_list,self.param_bounds.T)) # type: ignore
        self.nstart = nstart
        log.info(f" Running the sampler with the following params {self.param_list}")
        log.info(f" Parameter bounds \n{self.param_bounds.T}")
        log.info(f" Parameter labels \n{self.param_labels}")
        #output and timing dataframes
        # self.unit_transform = functools.partial(input_standardize,param_bounds=self.param_bounds)

        #initialize train_x, train_y and run GP
        self.train_x = qmc.Sobol(self.ndim, scramble=True,seed=seed).random(nstart)
        log.info(f" Initial values to evaluate \n{self.print_point(self.train_x)}")
        self.train_y = self.logp(self.train_x)
        log.info(f" Initial loglikes \n{self.train_y}")
        # self.train_x = input_standardize(self.train_x,self.param_bounds)
        # train_y = output_standardize(train_y)
        # self.gp_kwargs = gp_kwargs
        self.noise = noise
        self.gp = saas_fbgp(self.train_x,self.train_y,noise=self.noise)
        rng_key, _ = random.split(random.PRNGKey(seed), 2)
        self.gp.fit(rng_key,warmup_steps=256,num_samples=256,thinning=16,verbose=False) # input settings

    def run(self):
        start = time.time()
        self.converged = False
        while not self.converged:
            mc_points = self.get_mc_points(self.num_step)
            acq_func = IPV(self.gp,mc_points)
            grad_fn = grad(acq_func)
            x0 =  np.random.uniform(0,1,self.ndim)
            results = optim_scipy_bh(acq_func,x0=x0,stepsize=1/4,
                                      niter=15,minimizer_kwargs={'jac': grad_fn, 'bounds': self.ndim*[(0,1)] })
            self.acq_val = abs(results.fun)
            next_x = jnp.atleast_2d(results.x)
            log.info(f" Next point at x = {self.print_point(results.x)} with acquisition function value = {results.fun:.4e}")
            next_y = self.logp(next_x)
            max_idx = jnp.argmax(self.train_y)
            log.info(f" Loglike at new point = {next_y}, current best loglike = {self.train_y[max_idx]} at {self.print_point(self.train_x[max_idx])} ")
            self.train_x = jnp.concatenate([self.train_x,next_x])
            self.train_y = jnp.concatenate([self.train_y,next_y])
            seed = self.num_step
            rng_key, _ = random.split(random.PRNGKey(seed), 2)
            if (self.num_step%self.gpfit_step==0):
                # self.gp = saas_fbgp(self.train_x,self.train_y,noise=self.noise)
                # self.gp.fit(rng_key,warmup_steps=256,num_samples=256,thinning=16,verbose=False)
                self.gp.update(next_x,next_y,rng_key,warmup_steps=256,num_samples=256,thinning=16,verbose=False)
            else:
                self.gp.quick_update(next_x,next_y)
            # np.savetxt('train_x.txt',input_unstandardize(self.train_x,self.param_bounds))
            log.info(f" ----------------------Step {self.num_step+1} complete----------------------\n")
            self.num_step+=1
            self.converged = self.check_converged()
        samples, logz_dict =  nested_sampling_jaxns(self.gp,ndim=self.ndim,dlogz=0.01,difficult_model=True)
        log.info(f" Final LogZ info :"+"".join(f"{key}: = {value:.4f}, " for key, value in logz_dict.items()))
        if self.converged:
            log.info(" Run Complete")
        log.info(f" BO took {time.time() - start:.2f}s")

        # plot_gp(samples,self.gp,self.param_list,self.param_labels,self.param_bounds)

    def check_converged(self):
        acq = (self.acq_val < self.acq_goal)
        steps = (self.num_step >= self.max_steps)
        if acq:
            log.info(" Acquisition goal reached")
        if steps:
            log.info(" Max steps reached")
        return acq or steps

    def _ext_logp(self,loglike,x):
        x  = input_unstandardize(x,self.param_bounds)
        return jnp.atleast_2d(loglike(x))

    def _cobaya_logp(self, x): #logposterior for cobaya likelihoods
        # X should be a N x DIM  with parameters in the same order as the param_list in range [0,1]^DIM
        pdf = []
        x =  input_unstandardize(x,self.param_bounds)
        for point in x: # can parallelize evaluation of likelihood by splitting x into nproc parts
            logpost = self.cobaya_model.logpost(point,make_finite=True) 
            # if logpost == -np.inf:
            #     logpost = -1e100  # or whatever the machine precision allows for
            pdf.append(logpost)
        return np.array(pdf).reshape(-1,1)

    def print_point(self,x):
        x = input_unstandardize(x,self.param_bounds)
        # if len(x==1):
        #     x = x.item()
        return dict(zip(self.param_list,x))

    def _cobaya_init(self,input_file):
        try:
            from cobaya.yaml import yaml_load
            from cobaya.model import get_model
            self.logp = self._cobaya_logp
            assert input_file is not None
            info = yaml_load(input_file)
            self.cobaya_model = get_model(info) # type: ignore  #note that model is already a cobaya object
            rootlogger = logging.getLogger() 
            rootlogger.handlers.pop()    
            self.param_list = list(self.cobaya_model.parameterization.sampled_params()) # how do we deal with derived parameters
            self.param_bounds = np.array(self.cobaya_model.prior.bounds(confidence_for_unbounded=0.95)).T
            self.param_labels = [self.cobaya_model.parameterization.labels()[key] for key in self.param_list]         
            #get_valid_point(max_tries, ignore_fixed_ref=False, logposterior_as_dict=False, random_state=None)
        except ModuleNotFoundError:
            log.error(" Cobaya not found")

    def get_mc_points(self,step):
        # if (step%self.ns_step==0):
        #     samples, logz_dict = samples, logz_dict = nested_sampling_jaxns(self.gp,ndim=self.ndim,dlogz=0.1)
        #     log.info(f" LogZ info :"+"".join(f"{key}: = {value:.4f}, " for key, value in logz_dict.items()))                
        #     size = len(samples)
        #     mc_points = samples[::int(size/self.mc_points_size),:]        
        seed = step
        rng_key, _ = random.split(random.PRNGKey(seed), 2)
        samples = sample_GP_NUTS(gp = self.gp,rng_key=rng_key,num_warmup=1024,num_samples=1024,thinning=32)
        # size = len(samples)
        # mc_points = samples[::int(size/self.mc_points_size),:]                
        return samples

    def _gp_init(self):
        # noise, kernel, NUTS, mc_points size
        pass

    def _acq_init(self):
        # which acq, needs mc_points or best_val
        pass

    def _bo_init(self):
        # skip steps for fit, nested sampler, acq_goal, upper-lower goal...
        pass

    def _params_init(self):
        # param names, bounds, labels
        pass

