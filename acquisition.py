import time
from typing import Any,List, Optional
import jax.numpy as jnp
import jax.random as random
import numpy as np
import jax
from jax import jit, vmap, grad, value_and_grad
from jax.lax import scan
from jax.scipy.stats import norm
# from util import chunk_vmap
from jax import config
from fb_gp import saas_fbgp, sample_GP_NUTS
config.update("jax_enable_x64", True)
from numpyro.util import enable_x64
enable_x64()
from functools import partial
import scipy.optimize
import optax, pybobyqa
import logging
log = logging.getLogger("[AQ]")

import matplotlib.pyplot as plt

#------------------The acqusition functions-------------------------

# @jit
def Z_EI(mean, sigma, best_f, zeta,):
    """Returns `z(x) = (mean(x) - best_obs) / sigma(x)`"""
    z = (mean - best_f-zeta) / sigma
    return z # 

class Acquisition():

    def __init__(self
                 ,gp: saas_fbgp
                 ,ndim: int
                 ,optimizer_settings: dict[str,Any]
                 ,name = "") -> None:
        self.gp = gp
        self.ndim = ndim
        self.optimizer_method = optimizer_settings['method']
        self.optimizer_kwargs = optimizer_settings[self.optimizer_method]
        self.name = name
        log.info(f" Initialized {name} Acquisition function")
        log.info(f" Acquisition optimizer is {self.optimizer_method} with settings {self.optimizer_kwargs}")
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def optimize(self,x0,):
        acq_func = self.__call__
        optimizer = optimzer_functions[self.optimizer_method]
        start = time.time()
        pt, val = optimizer(acq_func,x0,self.ndim,**self.optimizer_kwargs)
        log.info(f" {self.optimizer_method} took {time.time()-start:.4f}s")
        return pt, val
    
    def update_mc_points(self,rng_key):
        pass

class EI(Acquisition):
    """
    The classic expected improvement acquisition function
    """

    def __init__(self,
                 gp: saas_fbgp,
                 rng_key,               
                 ndim: int,
                 optimizer_settings: dict[str,Any],
                 ei_kwargs: dict[str,Any] = {'zeta': 0.},
                 batch_size=1, # EI batch_size must always be 1
                 ) -> None:
        super().__init__(gp,ndim,optimizer_settings,name="EI")
        self.best_f =  gp.train_y.max() #best_f
        self.zeta = ei_kwargs['zeta']

    def __call__(self, X) -> Any:
        mu, var = self.gp.predict(X)
        mu  = mu.mean(axis=0)
        var = var.mean(axis=0)
        sigma = jnp.sqrt(var)
        u = Z_EI(mu, sigma, self.best_f,self.zeta) # type: ignore
        ei = sigma * (u*norm.cdf(u) + norm.pdf(u))
        # print(mu.shape,ei.shape)
        return np.reshape(-ei,()) # ei # can we make it directly a scalar without reshape
        # -ve so that we can minimize it

# todo implement the more stable logEI 
# stable implementation of logEI from Ament et al. (2023) arxiv: 2310.20708 as implemented in BoTorch: https://github.com/pytorch/botorch/blob/main/botorch/acquisition/analytic.py#L355
# class logEI(EI):

#     def __init__(self, 
#                  gp: saas_fbgp, 
#                  rng_key, ndim: int, 
#                  optimizer_settings: dict[str, Any], 
#                  ei_kwargs: dict[str, Any] = { 'zeta': 0. }
#                  , batch_size=1) -> None:
#         super().__init__(gp, rng_key, ndim, optimizer_settings, ei_kwargs, batch_size)

#     def __call__(self, X) -> Any:
#         mu, var = self.gp.predict(X)
#         mu  = mu.mean(axis=0)
#         var = var.mean(axis=0)
#         sigma = jnp.sqrt(var)
#         u = Z_EI(mu, sigma, self.best_f,self.zeta)
#         log_ei = jnp.log(sigma) + self.log_ei_helper(u)
#         return jnp.reshape(-log_ei,())
    
#     def log_ei_helper(self,u):
#         bound = -1
#         u_upper = jnp.where(u<bound,bound,u) # u.masked_fill(u < bound, bound)  # mask u to avoid NaNs in gradients
#         log_ei_upper =  jnp.log(u_upper*norm.cdf(u_upper) + norm.pdf(u_upper)) # type: ignore
#         neg_inv_sqrt_eps = -1e6 
#         u_lower = jnp.where(u > bound, bound, u)
#         u_eps = jnp.where(u < neg_inv_sqrt_eps, neg_inv_sqrt_eps, u_lower) # type: ignore
#         w = _log_abs_u_Phi_div_phi(u_eps)
#         log_ei_lower = log_phi(u) + (
#             jnp.where(
#                 u > neg_inv_sqrt_eps,
#                 log1mexp(w),
#                 -2 * jnp.log(jnp.abs(u_lower)), # type: ignore
#                 )
#             )
#         return jnp.where(u > bound, log_ei_upper, log_ei_lower)

class WIPV(Acquisition):
    """
    Integrated (mean) posterior variance over the over a set of test points (mc_points). 
    Can do joint optimization here with batch size > 1
    """

    def __init__(self
                ,gp: saas_fbgp
                ,rng_key
                ,ndim: int
                ,batch_size: int
                ,optimizer_settings: dict[str,Any]
                ,mcmc_kwargs: dict[str,Any] = {'num_samples': 512, 'warmup_steps': 512, 'thinning': 16, 'progress_bar': False, 'verbose': False}
                )-> None:
        super().__init__(gp,ndim,optimizer_settings,name="WIPV")
        self.mcmc_kwargs = mcmc_kwargs
        self.mc_points = sample_GP_NUTS(gp,rng_key,**mcmc_kwargs)
        log.info(" MC points generated")
        self.batch_size = batch_size
        self.ndim = self.gp.train_x.shape[1]
    
    def __call__(self, X) -> Any:
        X = jnp.reshape(X,(self.batch_size,self.ndim)) # input X is 1D array (x1_1,x1_2,...x1_d,x2_1,x2_2,...,x2_d) for d-dim x
        # X = jnp.atleast_2d(X) # new_x
        return self.variance(X)
    
    def variance(self,X):
        #var = self.gp.fantasy_var_fb(X,self.mc_points)
        var = self.gp.fantasy_var_fb_acq(X, self.mc_points)
        #var = var.mean(axis=-1)
        var = var.mean(axis=-1)
        return var
        # +ve can be directly minimized

    def update_mc_points(self,rng_key):
        self.mc_points = sample_GP_NUTS(self.gp,rng_key,**self.mcmc_kwargs)
        log.info(" Updated acquisition MC points")
        # print(self.gp.train_x.shape[0])
        # plt.scatter(self.mc_points[:,0],self.mc_points[:,1],alpha=0.3)
        # plt.show()


#------------------Optimizers for the acqusition functions-------------------------

# effectively a local optimizer with multiple restarts
def optim_scipy_bh(acq_func,x0,ndim,minimizer_kwargs={'method': 'L-BFGS-B'  },stepsize=1/4,niter=15):
    acq_grad = grad(acq_func)
    # ideally stepsize should be ~ max(delta,distance between sampled points)
    # with delta some small number to ensure that step size does not become too small
    minimizer_kwargs['jac'] = acq_grad
    minimizer_kwargs['bounds'] = ndim*[(0,1)]
    results = scipy.optimize.basinhopping(acq_func,
                                        x0=x0,
                                        stepsize=stepsize,
                                        niter=niter,
                                        minimizer_kwargs=minimizer_kwargs) 
    # minimizer_kwargs is for the choice of the local optimizer, bounds and to provide gradient if necessary
    return results.x, results.fun

# add here jax based optax or optuna optimizers

def optim_optax(acq_func,x0: np.ndarray,ndim: int
                ,steps=100,start_learning_rate=1e-2, n_restarts = 4,jump_sdev = 0.1): # needs more work
    optimizer = optax.adam(start_learning_rate)
    params = jnp.array(x0)
    opt_state = optimizer.init(params)
    max_iters = ndim*steps

    @jit
    def step(carry,xs):
        params, opt_state = carry
        acqval, gradval = value_and_grad(acq_func)(params) 
        updates, opt_state = optimizer.update(gradval, opt_state)
        params = optax.apply_updates(params, updates)
        params = optax.projections.projection_hypercube(params)
        carry = params, opt_state
        return carry, acqval
    @jit
    def findoptim(x0):
        params = x0
        opt_state = optimizer.init(params)
        (params, _ ), acqvals = scan(step,(params,opt_state),length=max_iters) # scan is much faster but more complicated to terminate early
        return (params,acqvals[-1])
        
    
    log.info(f" Acquisition Function Optimisation running on {jax.device_count()} devices")
    #if jax.device_count()>1:
    #    xi = x0 + jump_sdev*np.random.randn(jax.device_count() ,ndim)
    #    res = jax.pmap(findoptim,devices=jax.devices())(xi)
    #else:
    xi = x0 + jump_sdev*np.random.randn(n_restarts ,ndim)
    res = jax.lax.map(findoptim, xi) # or jax.vmap(findoptim,)(xi)
    best_val, idx = np.min(res[1]), np.argmin(res[1])
    best_params = res[0][idx]


    return best_params, best_val


# some gradient free optimizers (e.g. from iminuit or pybobyqa)

def optim_cma(acq_func,x0,):
    pass

def optim_bobyqa(acq_func
                 ,x0: np.ndarray
                 ,ndim: int = 1
                 ,batch_size: int = 1
                 ,seek_global_minimum: bool = False):
    # set up bounds for the solver

    upper = np.ones(ndim*batch_size)
    lower = np.zeros_like(upper)

    x0 = np.atleast_1d(x0) # tweak for batched acquisition
    soln = pybobyqa.solve(acq_func,x0,bounds=(lower,upper)
                          ,seek_global_minimum=seek_global_minimum,print_progress=False,do_logging=False)
    return soln.x, soln.f

#------------------Optimizing the acquisition-------------------------

optimzer_functions = {"scipy_bh": optim_scipy_bh, "optax": optim_optax, "bobyqa": optim_bobyqa, "cma": optim_cma}

# def optimize_acq(rng_key
#                  ,gp: saas_fbgp
#                  ,x0: np.ndarray
#                  ,ndim: int
#                  ,step: int
#                  ,acquistion: str = "WIPV"
#                  ,method: str = "bobyqa"
#                  ,acq_kwargs: dict[str,Any] = {}
#                  ,optimizer_kwargs: dict[str,Any] = {}
#                  ):
    
#     if acquistion=="WIPV":
#         acq_func = WIPV(gp,rng_key,**acq_kwargs)
#     else:
#         acq_func = EI(gp,**acq_kwargs)

#     optimizer = optimzer_functions[method]

#     pt, val = optimizer(acq_func,x0,ndim,**optimizer_kwargs)

#     return pt, val
