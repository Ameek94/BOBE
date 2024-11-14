import time
from typing import Any,List
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
from fb_gp import saas_fbgp
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
                 best_f,
                 zeta: float = 0.1,
                 batch_size=1, # EI batch_size must always be 1
                 ) -> None:
        super().__init__(gp)
        self.best_f =  gp.train_y.max() #best_f
        self.zeta = zeta

    # @jit ?
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
    

class IPV(Acquisition):
    """
    Integrated (mean) posterior variance over the over a set of test points (mc_points). 
    Can do joint optimization here with batch size > 1
    """

    def __init__(self, gp: saas_fbgp,
                 mc_points,batch_size=1) -> None:
        super().__init__(gp)
        self.mc_points = mc_points
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
    minimizer_kwargs['bounds': self.ndim*[(0,1)]']
    results = scipy.optimize.basinhopping(acq_func,
                                        x0=x0,
                                        stepsize=stepsize,
                                        niter=niter,
                                        minimizer_kwargs=minimizer_kwargs) 
    # minimizer_kwargs is for the choice of the local optimizer, bounds and to provide gradient if necessary
    log.info(f" Acquisition optimization took {time.time() - start:.2f} s")
    return results.x, results.fun

# add here jax based optax optimizers - stochastic gradient descent

def optim_optax(acq_func,x0,iters=100): # needs more work
    acq_grad = grad(acq_func)
    start_learning_rate = 1e-1
    optimizer = optax.adam(start_learning_rate)
    params = jnp.array(x0)
    opt_state = optimizer.init(params)
    start = time.time()
    for _ in range(iters):
        grads = acq_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        params = optax.projections.projection_hypercube(params)
    print(f"Optax optimizer took {time.time() - start:.4f}s")
    # print(params)
    return params


# some gradient free optimizers (e.g. from iminuit or pybobyqa)

def optim_cma(acq_func,x0,):
    pass


def optim_bobyqa(acq_func
                 ,x0: np.ndarray
                 ,dim: int = 1
                 ,batch_size: int = 1):
    # set up bounds for the solver
    upper = np.ones(dim*batch_size)
    lower = np.zeros_like(upper)
    start = time.time()
    x0 = np.atleast_1d(x0) # tweak for batched acquisition
    soln = pybobyqa.solve(acq_func,x0,bounds=(lower,upper)
                          ,seek_global_minimum=True,print_progress=False,do_logging=False)

    # # convert acq_func output to numpy output
    # def wrap_acq_func(x):
    #     with torch.no_grad():
    #         x = np.array(x).reshape(batch_size,dim) # optimizer cannot do joint optimization but can get batched points by converting to batch_size*ndim shaped arrays
    #         X = torch.tensor(x,**tkwargs)        
    #         Y = -acq_func(X)
    #         del X
    #         y = Y.view(-1).double().numpy()
    #         return y
    # run the solver
    # xs = soln.x # xs is 1D array of size dim*batch_size
    # best_x  = np.array(xs).reshape(batch_size,dim)
    # best_x = torch.tensor(best_x,**tkwargs)
    # best_val = torch.tensor(soln.f,**tkwargs)
    print(f"Py-Bobyqa took {time.time()-start:.4f} s")
    return soln.x, soln.f