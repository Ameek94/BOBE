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



#------------------The acqusition functions-------------------------

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
                 zeta: float = 0.1,) -> None:
        super().__init__(gp)
        self.best_f = best_f
        self.zeta = zeta

    # @jit needs static argnums or similar
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

    # def grad_fn(self,X):
    #     return grad(self.__call__(X))
    

class IPV(Acquisition):
    """
    Integrated (mean) posterior variance over the over a set of test points (mc_points).
    """

    def __init__(self, gp: saas_fbgp,
                 mc_points,) -> None:
        super().__init__(gp)
        self.mc_points = mc_points

    def __call__(self, X) -> Any:
        # return super().__call__(*args, **kwds)
        X = jnp.atleast_2d(X)
        return self.variance(X)

    def variance(self,X):
        var = self.gp.fantasy_var_fb(X,self.mc_points)
        var = var.mean(axis=-1)
        var = var.mean(axis=-1)
        # mu, _ = self.gp.predict(X,standardize=False)
        return np.reshape(var,()) #*mu
        # +ve can be directly minimized



#------------------Optimizers for the acqusition functions-------------------------

# effectively a local optimizer with multiple restarts
def optim_scipy_bh(acq_func,x0,minimizer_kwargs,stepsize=1/4,niter=15):
    start = time.time()
    # ideally stepsize should be ~ max(delta,distance between sampled points)
    # with delta some small number to ensure that step size does not become too large
    results = scipy.optimize.basinhopping(acq_func,
                                        x0=x0,
                                        stepsize=stepsize,
                                        niter=niter,
                                        minimizer_kwargs=minimizer_kwargs) # minimizer_kwargs is for the choice of the local optimizer, bounds and to provide gradient if necessary
    print(f"Acquisition optimization took {time.time() - start:.2f} s")
    return results

# add here jax based optax optimizers - stochastic gradient descent

def optim_adam():
    pass

# some gradient free optimizer (e.g. from iminuit or pybobyqa)

def optim_cma(acq_func,x0,):
    pass


def optim_bobyqa(acq_func,x0,):
    pass