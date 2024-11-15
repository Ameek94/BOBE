import time
from typing import Any,List, Optional
import numpyro
import numpyro.distributions as dist
from numpyro.util import enable_x64
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import jit, vmap, grad
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
from jax.scipy.stats import norm
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS
# from util import chunk_vmap
from jax import config
from pyro import sample
from fb_gp import saas_fbgp, sample_GP_NUTS
config.update("jax_enable_x64", True)
from numpyro.util import enable_x64
enable_x64()
from functools import partial
import scipy.optimize
import optax, pybobyqa
import logging
log = logging.getLogger("[AQ]")



# Implement optimizer call within the acquisition class?


#------------------The acqusition functions-------------------------

# @jit
def Z_EI(mean, sigma, best_f, zeta,):
    """Returns `z(x) = (mean(x) - best_obs) / sigma(x)`"""
    z = (mean - best_f-zeta) / sigma
    return z # 

class Acquisition():

    def __init__(self,gp: saas_fbgp,) -> None:
        self.gp = gp
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
        

class EI(Acquisition):
    """
    The classic expected improvement acquisition function
    """

    def __init__(self,
                 gp: saas_fbgp,
                 zeta: float = 0.,
                 batch_size=1, # EI batch_size must always be 1
                 ) -> None:
        super().__init__(gp)
        self.best_f =  gp.train_y.max() #best_f
        self.zeta = zeta

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

# implement the more stable logEI 
    

class WIPV(Acquisition):
    """
    Integrated (mean) posterior variance over the over a set of test points (mc_points). 
    Can do joint optimization here with batch size > 1
    """

    def __init__(self, gp: saas_fbgp,rng_key
                 ,batch_size=1
                 ,wipv_kwargs: dict[str,Any] = {}) -> None:
        super().__init__(gp)
        self.mc_points = sample_GP_NUTS(gp,rng_key,**wipv_kwargs)
        self.batch_size = batch_size
        self.ndim = self.gp.train_x.shape[1]
    
    def __call__(self, X) -> Any:
        X = jnp.reshape(X,(self.batch_size,self.ndim)) # input X is 1D array (x1_1,x1_2,...x1_d,x2_1,x2_2,...,x2_d) for d-dim x
        # X = jnp.atleast_2d(X) # new_x
        return self.variance(X)
    
    def variance(self,X):
        var = self.gp.fantasy_var_fb(X,self.mc_points)
        var = var.mean(axis=-1)
        var = var.mean(axis=-1)
        return var #np.reshape(var,()) #*mu
        # +ve can be directly minimized



#------------------Optimizers for the acqusition functions-------------------------

# effectively a local optimizer with multiple restarts
def optim_scipy_bh(acq_func,x0,ndim,minimizer_kwargs,stepsize=1/4,niter=15):
    start = time.time()
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
    log.info(f" Acquisition optimization took {time.time() - start:.2f} s")
    return results.x, results.fun

# add here jax based optax or optuna optimizers

def optim_optax(acq_func,x0,ndim,iters=100,start_learning_rate=1e-2): # needs more work
    acq_grad = grad(acq_func)
    optimizer = optax.adam(start_learning_rate)
    params = jnp.array(x0)
    opt_state = optimizer.init(params)
    start = time.time()
    iters = ndim*iters
    for _ in range(iters):
        grads = acq_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        params = optax.projections.projection_hypercube(params)
    print(f"Optax optimizer took {time.time() - start:.4f}s")
    # print(params)
    return params, acq_func(params)

# some gradient free optimizers (e.g. from iminuit or pybobyqa)

def optim_cma(acq_func,x0,):
    pass

def optim_bobyqa(acq_func
                 ,x0: np.ndarray
                 ,ndim: int = 1
                 ,batch_size: int = 1):
    # set up bounds for the solver

    upper = np.ones(ndim*batch_size)
    lower = np.zeros_like(upper)

    start = time.time()
    x0 = np.atleast_1d(x0) # tweak for batched acquisition
    soln = pybobyqa.solve(acq_func,x0,bounds=(lower,upper)
                          ,seek_global_minimum=True,print_progress=False,do_logging=False)

    print(f"Py-Bobyqa took {time.time()-start:.4f} s")
    return soln.x, soln.f

#------------------Optimizing the acquisition-------------------------

optimzer_functions = {"scipy": optim_scipy_bh, "optax": optim_optax, "bobyqa": optim_bobyqa, "cma": optim_cma}

def optimize_acq(rng_key
                 ,gp: saas_fbgp
                 ,x0: np.ndarray
                 ,ndim: int
                 ,step: int
                 ,acquistion: str = "WIPV"
                 ,method: str = "bobyqa"
                 ,acq_kwargs: dict[str,Any] = {}
                 ,optimizer_kwargs: dict[str,Any] = {}
                 ):
    
    if acquistion=="WIPV":
        acq_func = WIPV(gp,rng_key,**acq_kwargs)
    else:
        acq_func = EI(gp,**acq_kwargs)

    optimizer = optimzer_functions[method]

    pt, val = optimizer(acq_func,x0,ndim,**optimizer_kwargs)

    return pt, val