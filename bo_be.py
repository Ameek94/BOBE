# The main sampler class

import functools
import pandas as pd
import numpy as np
from typing import Any, Callable, List,Optional, Tuple, Dict
from fb_gp import saas_fbgp, sample_GP_NUTS
import time
import jax
import jax.numpy as jnp
from jax import random,vmap, grad
from acquisition import EI, WIPV, optim_bobyqa, optim_scipy_bh, optimize_acq
from scipy.stats import qmc
from jaxns import NestedSampler
from nested_sampler import nested_sampling_jaxns, nested_sampling_Dy
from bo_utils import input_standardize,input_unstandardize, output_standardize, output_unstandardize, plot_gp
import logging
from init import input_settings
log = logging.getLogger("[BO]")


# todo
# 1. save and resume
# 2. initialize everything from input yaml file and clean up this module
# 3. ?

def test_function(x):
    return -0.5*jnp.sum((x-0.5)**2/0.04,axis=-1,keepdims=True)

class sampler:

    def __init__(self,
            ndim = None,
            input_file: Optional[str] = None,
            cobaya_model = False,
            cobaya_start: int = 8,
            cobaya_input_file: Optional[str] = None,
            resume_from_file=False,
            resume_file = None,
            loglike =  None, # for external functions this should be the logposterior 
            param_list: Optional[List] = None,
            param_bounds: Optional[List] = None, # shape 2 x D
            param_labels: Optional[List] = None,
            gp_kwargs: dict[str,Any] = {'kernel_func':"rbf", 'vmap_size': 8},
            noise: float = 1e-8,
            gpfit_step: int = 1, 
            nstart: int = 4,
            max_steps: int = 4,
            acq_strat: Optional[str] = 'WIPV',
            acq_goal: float = 1e-6, # currently arbitrary...
            save: Optional[bool] = True,
            save_step: int = 5,
            save_file: str = 'run_data',
            nested_sampler: str = "jaxns", #jaxns 
            ns_step: int = 1, 
            mc_points_size: int = 16,
            nested_sampler_kwargs: Optional[dict] = None,
            final_ns_dlogz: float = 0.01,
            feedback_lvl: int = 1,
            seed: int = 0,) -> None:
        
        if input_file is not None:
            set_from_file=True
        else:
            set_from_file = False
        inputs = input_settings(set_from_file=set_from_file
                                ,file=input_file
                                ,input_dict=None)
        self.settings = inputs.settings
        
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
        self.save_step = save_step
        self.save = save
        self.save_file = save_file
        self.mc_points_size = mc_points_size

        if cobaya_model:
            points, lls = self._cobaya_init(cobaya_input_file,cobaya_start=cobaya_start) # type: ignore
        else:
            assert loglike is not None
            self.logp = functools.partial(self._ext_logp,loglike=loglike) # assuming loglike returns N x 1 shape for input N x d, eventually add option to define loglike model in external file
            self.param_list = param_list if param_list is not None else ['x_%i'%(i+1) for i in range(ndim)] # type: ignore
            self.param_labels = param_labels if param_labels is not None else ['x_%i'%(i+1) for i in range(ndim)] # type: ignore
            self.param_bounds = np.array(param_bounds) if param_bounds is not None else np.array(ndim*[[0,1]]).T # type: ignore
        self.ndim = len(self.param_list)
        self.bounds_dict = dict(zip(self.param_list,self.param_bounds.T)) # type: ignore
        log.info(f" Running the sampler with the following params {self.param_list}")
        log.info(f" Parameter bounds \n{self.param_bounds.T}")
        log.info(f" Parameter labels \n{self.param_labels}")

        self.noise = noise
        if resume_from_file:
            assert resume_file is not None
            with np.load(resume_file) as run_data:
                self.train_x = jnp.array(run_data['train_x'])
                self.train_y = jnp.array(run_data['train_y'])
                self.param_bounds = run_data['param_bounds']
                lengthscales = jnp.array(run_data["lengthscales"])
                outputscales = jnp.array(run_data["outputscales"])
            self.gp = saas_fbgp(self.train_x,self.train_y,noise=self.noise,**gp_kwargs,sample_lengthscales=lengthscales,sample_outputscales=outputscales) 
            log.info(f"Resuming from file {resume_file} with {self.train_x.shape[0]} previous points")
        else:
            self.train_x = qmc.Sobol(self.ndim, scramble=True,seed=seed).random(nstart)
            print(self.train_x)
            self.train_y = self.logp(self.train_x)
            print(self.train_y)
            if cobaya_model:
                for pt,ll in zip(points,lls): 
                    print("adding cobaya points")
                    print(dict(zip(self.param_list,pt)))
                    pt = input_standardize(np.reshape(pt,(1,self.ndim)),self.param_bounds)
                    if pt not in self.train_x:
                        print(pt,ll)
                        self.train_x = np.concatenate([self.train_x,pt])
                        self.train_y = np.concatenate([self.train_y,np.atleast_2d(ll)])
            log.info(f" Initial loglikes \n{self.train_y}")
            log.info(f" Sampler will start with {len(self.train_y)} points and run for a maximum of {self.max_steps} steps")
            self.gp = saas_fbgp(self.train_x,self.train_y,noise=self.noise,**gp_kwargs) 
            rng_key, _ = random.split(random.PRNGKey(seed), 2)
            self.gp.fit(rng_key,warmup_steps=512,num_samples=512,thinning=16,verbose=False) # input settings

    def run(self):
        """
        Run the BO loop until convergence or max steps reached. Here all x are standarised to [0,1]
        """
        start = time.time()
        self.converged = False
        while not self.converged:
            seed = self.num_step
            rng_key, _ = random.split(random.PRNGKey(seed), 2)
            # mc_points = self.get_mc_points(self.num_step)
            # acq_func = WIPV(self.gp,mc_points)
            # grad_fn = grad(acq_func)
            x0 =  np.random.uniform(0,1,self.ndim) # get better initial point
            pt, val  = optimize_acq(rng_key=rng_key,
                                    gp=self.gp,
                                    x0=x0,
                                    ndim=self.ndim,
                                    step=self.num_step,
                                    )
            #optim_bobyqa(acq_func,x0,ndim=self.ndim) 
            #optim_scipy_bh(acq_func,x0=x0,stepsize=1/5,ndim=self.ndim
                                      #niter=20,minimizer_kwargs={'jac': grad_fn, 'bounds': self.ndim*[(0,1)] })
            self.acq_val = abs(val)
            next_x = jnp.atleast_2d(pt)
            log.info(f" Next point at x = {self._print_point(pt)} \nwith acquisition function value = {val:.4e}")
            next_y = self.logp(next_x)
            max_idx = jnp.argmax(self.train_y)
            self.train_x = jnp.concatenate([self.train_x,next_x])
            self.train_y = jnp.concatenate([self.train_y,next_y])
            log.info(f" Loglike at new point = {next_y}, \ncurrent best loglike = {self.train_y[max_idx]} at \n{self._print_point(self.train_x[max_idx])} ")
            if (self.num_step%5==0 and self.num_step>0):
                jax.clear_caches() # hack for managing memory, is there a better way?
            if (self.num_step%self.gpfit_step==0):
                self.gp.update(next_x,next_y,rng_key,warmup_steps=512,num_samples=512,thinning=16,verbose=False)
            else:
                self.gp.quick_update(next_x,next_y)
            log.info(f" ----------------------Step {self.num_step+1} complete----------------------\n")
            self.num_step+=1
            self.converged = self._check_converged()
            if ((self.num_step%self.save_step==0 and self.save) or self.converged):
                log.info(f"Run training data and hyperparameters saved at step {self.num_step}")
                self.gp.save(self.save_file)
        samples, logz_dict =  nested_sampling_jaxns(self.gp,ndim=self.ndim,dlogz=0.01,difficult_model=True)
        log.info(f" Final LogZ info :"+"".join(f"{key}: = {value:.4f}, " for key, value in logz_dict.items()))
        if self.converged:
            log.info(" Run Completed")
        log.info(f" BO took {time.time() - start:.2f}s")

    def _check_converged(self):
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
            pdf.append(logpost)
        return np.array(pdf).reshape(-1,1)

    def _cobaya_init(self,input_file,cobaya_start=8):
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
            points = []
            logpost = []
            for i in range(cobaya_start):
                res = self.cobaya_model.get_valid_point(100, ignore_fixed_ref=False
                                                             , logposterior_as_dict=True
                                                             , random_state=None)
                points.append(res[0])
                logpost.append(res[1]['logpost']) # type: ignore
            return np.array(points), np.array(logpost)
        except ModuleNotFoundError:
            log.error(" Cobaya not found")

    
    def _print_point(self,x):
        x = input_unstandardize(x,self.param_bounds)
        return dict(zip(self.param_list,x))

    def get_mc_points(self,step):
        if (step%self.ns_step==0):
        #     samples, logz_dict = samples, logz_dict = nested_sampling_jaxns(self.gp,ndim=self.ndim,dlogz=0.1)
        #     log.info(f" LogZ info :"+"".join(f"{key}: = {value:.4f}, " for key, value in logz_dict.items()))                
        #     size = len(samples)
        #     mc_points = samples[::int(size/self.mc_points_size),:]        
            seed = step
            rng_key, _ = random.split(random.PRNGKey(seed), 2)
            samples = sample_GP_NUTS(gp = self.gp,rng_key=rng_key,warmup_steps=512,num_samples=512,thinning=8)
        # size = len(samples)
        # mc_points = samples[::int(size/self.mc_points_size),:]                
        return samples



