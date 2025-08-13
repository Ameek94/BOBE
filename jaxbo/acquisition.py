import time
from typing import Any, List, Optional, Dict, Tuple
import jax.numpy as jnp
from jax import lax
import numpy as np
from scipy.stats import qmc
from jax.scipy.stats import norm
from jax import config
from jax.scipy.special import erfc, logsumexp
import tensorflow_probability.substrates.jax as tfp
import logging
from .optim import optimize
from .utils.logging_utils import get_logger
from .utils.seed_utils import get_numpy_rng
from .nested_sampler import nested_sampling_Dy
config.update("jax_enable_x64", True)
log = get_logger("[acq]")

#------------------Helper functions-------------------------
# These are jax versions of the BoTorch functions.

# def WIPV(x, gp, mc_points=None):
#     """
#     Computes the Weighted Integrated Posterior Variance acquisition function.
    
#     Args:
#         x: Input points (shape: [n, ndim])
#         gp: Gaussian process model
#         mc_points: Optional Monte Carlo points for fantasy variance computation
    
#     Returns:
#         Mean of the posterior variance at the input points.
#     """
#     var = gp.fantasy_var(x, mc_points=mc_points)
#     return jnp.mean(var)



def WIPV(x, gp, mc_points=None):
    """
    Computes the Weighted Integrated Posterior Variance acquisition function.
    
    Args:
        x: Input points (shape: [n, ndim])
        gp: Gaussian process model
        mc_points: Optional Monte Carlo points for fantasy variance computation
    
    Returns:
        Mean of the posterior variance at the input points.
    """
    var = gp.fantasy_var(x, mc_points=mc_points)
    return jnp.mean(var)



def get_mc_samples(gp,warmup_steps=512, num_samples=512, thinning=4,method="NUTS",init_params=None):
    if method=='NUTS':
        try:
            mc_samples = gp.sample_GP_NUTS(warmup_steps=warmup_steps,
            num_samples=num_samples, thinning=thinning
            )
        except Exception as e:
            log.error(f"Error in sampling GP NUTS: {e}")
            mc_samples, logz, success = nested_sampling_Dy(gp, gp.ndim, maxcall=int(2e6)
                                            , dynamic=False, dlogz=0.5,equal_weights=True,
            )
    elif method=='NS':
        mc_samples, logz, success = nested_sampling_Dy(gp, gp.ndim, maxcall=int(2e6)
                                            , dynamic=False, dlogz=0.5,equal_weights=True,
        )
    elif method=='uniform':
        mc_samples = {}
        points = qmc.Sobol(gp.ndim, scramble=True).random(num_samples)
        mc_samples['x'] = points
    else:
        raise ValueError(f"Unknown method {method} for sampling GP")
    return mc_samples


def get_mc_points(mc_samples, mc_points_size=64):
    mc_size = max(mc_samples['x'].shape[0], mc_points_size)
    idxs = np.random.choice(mc_size, size=mc_points_size, replace=False)
    return mc_samples['x'][idxs]

# -------------------------
def _scaled_improvement(mu, sigma, best_f):
    """u = (mu - best_f) / sigma, safe for sigma=0."""
    return (mu - best_f) / sigma

def _log_phi(u):
    """log of standard normal PDF"""
    return -0.5 * (u**2 + jnp.log(2 * jnp.pi))

def _ei_helper(u):
    """EI = phi(u) + u * Phi(u), stable for large |u|."""
    return norm.pdf(u) + u * norm.cdf(u)

def _log_abs_u_Phi_div_phi(u):
    """
    log(|u| * Phi(u) / phi(u)), valid for u < 0.
    Uses erfcx for numerical stability in tail.
    """
    neg_inv_sqrt2 = -1.0 / jnp.sqrt(2.0)
    log_sqrt_pi_div_2 = 0.5 * jnp.log(jnp.pi / 2.0)

    erfcx_val = tfp.math.erfcx(neg_inv_sqrt2 * u)
    return jnp.log(jnp.abs(u) * erfcx_val) + log_sqrt_pi_div_2

def _log_ei_helper(u):
    """
    Accurately computes log(phi(u) + u * Phi(u)).
    Matches BoTorch branching for stability.
    """
    if u.dtype not in [jnp.float32, jnp.float64]:
        raise TypeError(
            f"LogExpectedImprovement only supports float32 and float64, got {u.dtype}."
        )

    bound = -1.0
    neg_inv_sqrt_eps = -1e6 if u.dtype == jnp.float64 else -1e3

    # u > bound: directly log(EI)
    u_upper = jnp.where(u < bound, bound, u)
    log_ei_upper = jnp.log(_ei_helper(u_upper))

    # u <= bound: use asymptotic expansion
    u_lower = jnp.where(u > bound, bound, u)
    u_eps = jnp.where(u_lower < neg_inv_sqrt_eps, neg_inv_sqrt_eps, u_lower)

    w = _log_abs_u_Phi_div_phi(u_eps)
    log_phi_u = _log_phi(u)

    second_term = jnp.where(
        u > neg_inv_sqrt_eps,
        tfp.math.log1mexp(w),
        -2.0 * jnp.log(jnp.abs(u_lower))
    )
    log_ei_lower = log_phi_u + second_term

    return jnp.where(u > bound, log_ei_upper, log_ei_lower)

# -------------------------
# Main EI and LogEI
# -------------------------

def EI(x, gp, best_y=0.,zeta=0.):
    """
    Expected Improvement in pure JAX.
    """
    mu, var = gp.predict_single(x)
    var = jnp.clip(var, a_min=1e-18)  # prevent zero variance
    sigma = jnp.sqrt(var)

    u = _scaled_improvement(mu , sigma, best_y)
    ei = _ei_helper(u) * sigma

    return jnp.reshape(-ei, ())  # optimizer minimizes this

def LogEI(x, gp, best_y=0.,zeta=0.):
    """
    Log Expected Improvement in pure JAX.
    Returns *positive* log-EI, so you can maximize directly.
    """
    mu, var = gp.predict_single(x)
    var = jnp.clip(var, a_min=1e-18)  # prevent zero variance
    sigma = jnp.sqrt(var)

    u = _scaled_improvement(mu - zeta, sigma, best_y)
    log_ei = _log_ei_helper(u) + jnp.log(sigma)

    return jnp.reshape(-log_ei, ())  # optimizer minimizes this


#------------------The acquisition function classes-------------------------

# class AcquisitionFunction:
#     """Base class for acquisition functions"""
#     def __init__(self,gp,optimizer_kwargs: Optional[Dict[str, Any]] = {}):
#         self.gp = gp
#         self.optimizer_kwargs = optimizer_kwargs

#     def fun(self, x):
#         raise NotImplementedError

#     def get_next(self, fun_args: Tuple = (), fun_kwargs: Dict[str, Any] = {}, step: int = 0) -> Tuple[np.ndarray, float]:
#         """
#         Optimize the acquisition function to obtain the next point to sample.
        
#         Args:
#             gp: Gaussian process model
#             ndim: Number of dimensions (inferred from bounds if not provided)
#             optimizer_name: Name of optimizer to use
#             **kwargs: Additional arguments passed to optimize()
        
#         Returns:
#             Tuple of (best_x, best_value)
#         """

#         return optimize(func = self.fun, fun_args = fun_args, fun_kwargs = fun_kwargs, ndim=self.gp.ndim, bounds=None, 
#                         **self.optimizer_kwargs)

# class EI(AcquisitionFunction):
#     """Expected Improvement acquisition function"""
#     def __init__(self, gp, zeta: float = 0.0, optimizer_kwargs: Optional[Dict[str, Any]] = {}):
#         super().__init__(gp, optimizer_kwargs=optimizer_kwargs)
#         self.zeta = zeta

#     def fun(self, x):
#         mu, var = self.gp.predict(x)
#         std = jnp.sqrt(var)
#         best_f = self.gp.train_y.max() - self.zeta
#         z = (mu - best_f) / std
#         ei = std * (z * norm.cdf(z) + norm.pdf(z))
#         return jnp.reshape(-ei, ()) # EI is maximized, so we return -EI

# class LogEI(EI):
#     """Log Expected Improvement acquisition function"""
#     def __init__(self, gp, zeta: float = 0.0, optimizer_kwargs: Optional[Dict[str, Any]] = {}):
#         super().__init__(gp, zeta=zeta, optimizer_kwargs=optimizer_kwargs)

#     def fun(self, x):
#         mu, var = self.gp.predict(x)
#         # if jnp.any(var <= 0):
#         #     log.warning(f"Non-positive variance detected: {var}")
#         var = jnp.clip(var, min=1e-8)  # Avoid division by zero
#         sigma = jnp.sqrt(var)
#         best_f = self.gp.train_y.max() - self.zeta

#         # Compute scaled improvement
#         u = _scaled_improvement(mu, sigma, best_f)
        
#         # Compute log EI
#         log_ei = _log_ei_helper(u) + jnp.log(sigma)
        
#         return jnp.reshape(-log_ei, ())

# class MonteCarloAcquisition(AcquisitionFunction):
#     """Monte Carlo acquisition function base class"""

#     def __init__(self, gp, mc_kwargs: Optional[Dict[str, Any]] = {}, optimizer_kwargs: Optional[Dict[str, Any]] = {}):

#         super().__init__(gp, optimizer_kwargs)
#         self.mc_points_method = mc_kwargs.get('mc_points_method', "NUTS")
#         self.mc_points_size = mc_kwargs.get('mc_points_size', 64)
#         self.warmup_steps = mc_kwargs.get('warmup_steps', 512)
#         self.num_samples = mc_kwargs.get('num_samples', 512)
#         self.thinning = mc_kwargs.get('thinning', 4)
#         self.mc_update_step = mc_kwargs.get('mc_update_step', 5)
#         self.num_chains = mc_kwargs.get('num_chains', 1)
#         self.ns_maxcall = mc_kwargs.get('ns_maxcall', int(1e6))
#         self.ns_dynamic = mc_kwargs.get('ns_dynamic', False)
#         self.ns_dlogz = mc_kwargs.get('ns_dlogz', 0.25)
#         self.mc_samples = None
#         self.mc_points = None
#         self.rng = get_numpy_rng()

#     def generate_mc_samples(self):
#         if self.mc_points_method=='NUTS':
#             try:
#                 self.mc_samples = self.gp.sample_GP_NUTS(warmup_steps=self.warmup_steps,
#                 num_samples=self.num_samples, thinning=self.mc_update_step
#                 )
#             except Exception as e:
#                 log.error(f"Error in sampling GP NUTS: {e}")
#                 self.mc_samples, logz, success = nested_sampling_Dy(self.gp, self.gp.ndim, maxcall=self.ns_maxcall
#                                                 , dynamic=True, dlogz=self.ns_dlogz,equal_weights=True,
#                         )
#         elif self.mc_points_method=='NS':
#             self.mc_samples, logz, success = nested_sampling_Dy(self.gp, self.gp.ndim, maxcall=self.ns_maxcall
#                                             , dynamic=self.ns_dynamic, dlogz=self.ns_dlogz,equal_weights=True,
#             )
#         elif self.mc_points_method=='uniform':
#             self.mc_samples = {}
#             points = qmc.Sobol(self.gp.ndim, scramble=True).random(self.mc_points_size)
#             self.mc_samples['x'] = points
#             self.mc_samples['weights'] = jnp.ones(self.mc_points_size)
#         else:
#             raise ValueError(f"Unknown method {self.mc_points_method} for sampling GP")
     
#     def get_mc_points(self):
#         mc_size = max(self.mc_samples['x'].shape[0], self.mc_points_size)
#         idxs = self.rng.choice(mc_size, size=self.mc_points_size, replace=False)
#         return self.mc_samples['x'][idxs]
    
    
#     def get_next(self, **kwargs):
#         # Here add logic to regenerate and retrieve mc_points based on step
#         update_mc = kwargs.get('update_mc', False)
#         if update_mc or self.mc_samples is None:
#             log.info("Generating new Monte Carlo samples")
#             self.generate_mc_samples()
#         mc_points = self.get_mc_points()
#         return super().get_next(fun_kwargs={'mc_points': mc_points}, **kwargs)


# class WIPV(MonteCarloAcquisition):
#     """Weighted Integrated Posterior Variance acquisition function"""
#     def __init__(self, gp, mc_kwargs: Optional[dict] = {}, optimizer_kwargs: Optional[dict] = {}):
#         super().__init__(gp, mc_kwargs=mc_kwargs, optimizer_kwargs=optimizer_kwargs)

#     def fun(self, x, mc_points=None):
#         var = self.gp.fantasy_var(new_x=x, mc_points=mc_points)
#         return jnp.mean(var)
    
#     def get_next(self, **kwargs):
#         return super().get_next(**kwargs)

# class MaxVar(MonteCarloAcquisition):
#     """Maximum Variance acquisition function"""

#     def __init__(self, gp, mc_kwargs: Optional[dict] = {}, optimizer_kwargs: Optional[dict] = {}):
#         super().__init__(gp, mc_kwargs=mc_kwargs, optimizer_kwargs=optimizer_kwargs)

#     def __call__(self, x, gp, **kwargs):
#         mc_points = kwargs.get('mc_points', self.mc_points)
#         var = gp.fantasy_var(x, mc_points=mc_points)

#         return jnp.max(var)
    
# def get_acquisition_function(name: str, gp, **kwargs) -> AcquisitionFunction:
#     """Factory function to get an acquisition function by name."""
#     if name == "EI":
#         return EI(gp, **kwargs)
#     elif name == "LogEI":
#         return LogEI(gp, **kwargs)
#     elif name == "WIPV":
#         return WIPV(gp, **kwargs)
#     else:
#         raise ValueError(f"Unknown acquisition function: {name}")
