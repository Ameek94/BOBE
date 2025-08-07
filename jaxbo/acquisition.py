import time
from typing import Any, List, Optional, Dict, Tuple
import jax.numpy as jnp
from jax import lax
import numpy as np
from scipy.stats import qmc
from jax.scipy.stats import norm
from jax import config
from jax.scipy.special import erfc, logsumexp
import logging
from .optim import optimize
from .logging_utils import get_logger
from .seed_utils import get_numpy_rng
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


def _scaled_improvement(mean, sigma, best_f):
    """Returns `u = (mean - best_f) / sigma`"""
    u = (mean - best_f) / sigma
    return u 

def _ei_helper(u):
    """Computes phi(u) + u * Phi(u), where phi and Phi are the standard normal
    pdf and cdf, respectively. This is used to compute Expected Improvement.
    """
    phi_u = norm.pdf(u)
    Phi_u = norm.cdf(u)
    return phi_u + u * Phi_u

def log1mexp(x):
    """Computes log(1 - exp(x)) numerically stable for x < 0"""
    return jnp.where(x < jnp.log(jnp.finfo(x.dtype).eps), 
                     x,  # For very small x, log(1-exp(x)) ≈ x
                     jnp.log(-jnp.expm1(x)))  # More stable computation

def _log_phi(u):
    """Computes log of standard normal pdf"""
    return -0.5 * (u**2 + jnp.log(2 * jnp.pi))

def _log_abs_u_Phi_div_phi(u):
    """Computes log(abs(u) * Phi(u) / phi(u)), where phi and Phi are the normal pdf
    and cdf, respectively. The function is valid for u < 0.
    """
    # Constants
    neg_inv_sqrt2 = -1.0 / jnp.sqrt(2.0)
    log_sqrt_pi_div_2 = 0.5 * jnp.log(jnp.pi / 2.0)
    
    # Compute erfcx(-u/sqrt(2)) = exp((u/sqrt(2))^2) * erfc(-u/sqrt(2))
    # Using the relation: erfcx(x) = 2*exp(x^2)/sqrt(pi) * int_x^inf exp(-t^2) dt
    # And Phi(u) = 0.5 * erfc(-u/sqrt(2))
    a = neg_inv_sqrt2
    erfcx_val = jnp.exp((a * u)**2) * erfc(a * u)
    
    return jnp.log(erfcx_val * jnp.abs(u)) + log_sqrt_pi_div_2

def _log_ei_helper(u):
    """Accurately computes log(phi(u) + u * Phi(u)) in a differentiable manner"""
    
    # Check dtype
    if u.dtype not in [jnp.float32, jnp.float64]:
        raise TypeError(
            f"LogExpectedImprovement only supports float32 and float64 "
            f"dtypes, but received {u.dtype}."
        )
    
    # The function has two branching decisions. The first is u < bound
    bound = -1.0
    neg_inv_sqrt_eps = -1e6 if u.dtype == jnp.float64 else -1e3
    
    # First branch: u > bound
    u_upper = jnp.where(u < bound, bound, u)  # mask u to avoid issues
    log_ei_upper = jnp.log(_ei_helper(u_upper))
    
    # Second branch: u <= bound
    u_lower = jnp.where(u > bound, bound, u)
    u_eps = jnp.where(u < neg_inv_sqrt_eps, neg_inv_sqrt_eps, u_lower)
    
    # compute the logarithm of abs(u) * Phi(u) / phi(u) for moderately large negative u
    w = _log_abs_u_Phi_div_phi(u_eps)
    
    # Compute log_ei_lower
    log_phi_u = _log_phi(u)
    second_term = jnp.where(
        u > neg_inv_sqrt_eps,
        log1mexp(w),
        # The contribution vanishes when w << eps but captures leading order
        -2 * jnp.log(jnp.abs(u_lower))
    )
    log_ei_lower = log_phi_u + second_term
    
    return jnp.where(u > bound, log_ei_upper, log_ei_lower)

#------------------The acquisition function classes-------------------------

class AcquisitionFunction:
    """Base class for acquisition functions"""
    def __init__(self,gp,optimizer_kwargs: Optional[Dict[str, Any]] = {}):
        self.gp = gp
        self.optimizer_kwargs = optimizer_kwargs

    def fun(self, x):
        raise NotImplementedError

    def get_next(self, fun_args: Tuple = (), fun_kwargs: Dict[str, Any] = {}, step: int = 0) -> Tuple[np.ndarray, float]:
        """
        Optimize the acquisition function to obtain the next point to sample.
        
        Args:
            gp: Gaussian process model
            ndim: Number of dimensions (inferred from bounds if not provided)
            optimizer_name: Name of optimizer to use
            **kwargs: Additional arguments passed to optimize()
        
        Returns:
            Tuple of (best_x, best_value)
        """

        return optimize(func = self.fun, fun_args = fun_args, fun_kwargs = fun_kwargs, ndim=self.gp.ndim, bounds=None, 
                        **self.optimizer_kwargs)

class EI(AcquisitionFunction):
    """Expected Improvement acquisition function"""
    def __init__(self, gp, zeta: float = 0.0, optimizer_kwargs: Optional[Dict[str, Any]] = {}):
        super().__init__(gp, optimizer_kwargs=optimizer_kwargs)
        self.zeta = zeta

    def fun(self, x):
        mu, var = self.gp.predict(x)
        std = jnp.sqrt(var)
        best_f = self.gp.train_y.max() - self.zeta
        z = (mu - best_f) / std
        ei = std * (z * norm.cdf(z) + norm.pdf(z))
        return jnp.reshape(-ei, ()) # EI is maximized, so we return -EI

class LogEI(EI):
    """Log Expected Improvement acquisition function"""
    def __init__(self, gp, zeta: float = 0.0, optimizer_kwargs: Optional[Dict[str, Any]] = {}):
        super().__init__(gp, zeta=zeta, optimizer_kwargs=optimizer_kwargs)

    def fun(self, x):
        mu, var = self.gp.predict(x)
        sigma = jnp.sqrt(var)
        best_f = self.gp.train_y.max() - self.zeta

        # Compute scaled improvement
        u = _scaled_improvement(mu, sigma, best_f, self.maximize)
        
        # Compute log EI
        log_ei = _log_ei_helper(u) + jnp.log(sigma)
        
        return jnp.reshape(-log_ei, ())

class MonteCarloAcquisition(AcquisitionFunction):
    """Monte Carlo acquisition function base class"""

    def __init__(self, gp, mc_kwargs: Optional[Dict[str, Any]] = {}, optimizer_kwargs: Optional[Dict[str, Any]] = {}):

        super().__init__(gp, optimizer_kwargs)
        self.mc_points_method = mc_kwargs.get('mc_points_method', "NUTS")
        self.mc_points_size = mc_kwargs.get('mc_points_size', 64)
        self.warmup_steps = mc_kwargs.get('warmup_steps', 512)
        self.num_samples = mc_kwargs.get('num_samples', 1024)
        self.mc_update_step = mc_kwargs.get('mc_update_step', 5)
        self.num_chains = mc_kwargs.get('num_chains', 1)
        self.ns_maxcall = mc_kwargs.get('ns_maxcall', int(1e6))
        self.ns_dynamic = mc_kwargs.get('ns_dynamic', False)
        self.ns_dlogz = mc_kwargs.get('ns_dlogz', 0.25)
        self.mc_samples = None
        self.mc_points = None
        self.rng = get_numpy_rng()

    def generate_mc_samples(self):
        if self.mc_points_method=='NUTS':
            try:
                self.mc_samples = self.gp.sample_GP_NUTS(warmup_steps=self.warmup_steps,
                num_samples=self.num_samples, thinning=self.mc_update_step
                )
            except Exception as e:
                log.error(f"Error in sampling GP NUTS: {e}")
                self.mc_samples, logz, success = nested_sampling_Dy(self.gp, self.gp.ndim, maxcall=self.ns_maxcall
                                                , dynamic=True, dlogz=self.ns_dlogz,equal_weights=True,
                        )
        elif self.mc_points_method=='NS':
            self.mc_samples, logz, success = nested_sampling_Dy(self.gp, self.gp.ndim, maxcall=self.ns_maxcall
                                            , dynamic=self.ns_dynamic, dlogz=self.ns_dlogz,equal_weights=True,
            )
        elif self.mc_points_method=='uniform':
            self.mc_samples = {}
            points = qmc.Sobol(self.gp.ndim, scramble=True).random(self.mc_points_size)
            self.mc_samples['x'] = points
            self.mc_samples['weights'] = jnp.ones(self.mc_points_size)
        else:
            raise ValueError(f"Unknown method {self.mc_points_method} for sampling GP")
     
    def get_mc_points(self):
        mc_size = max(self.mc_samples['x'].shape[0], self.mc_points_size)
        idxs = self.rng.choice(mc_size, size=self.mc_points_size, replace=False)
        return self.mc_samples['x'][idxs]
    
    
    def get_next(self, **kwargs):
        # Here add logic to regenerate and retrieve mc_points based on step
        update_mc = kwargs.get('update_mc', False)
        if update_mc or self.mc_samples is None:
            log.info("Generating new Monte Carlo samples")
            self.generate_mc_samples()
        mc_points = self.get_mc_points()
        return super().get_next(fun_kwargs={'mc_points': mc_points}, **kwargs)


class WIPV(MonteCarloAcquisition):
    """Weighted Integrated Posterior Variance acquisition function"""
    def __init__(self, gp, mc_kwargs: Optional[dict] = {}, optimizer_kwargs: Optional[dict] = {}):
        super().__init__(gp, mc_kwargs=mc_kwargs, optimizer_kwargs=optimizer_kwargs)

    def fun(self, x, mc_points=None):
        var = self.gp.fantasy_var(new_x=x, mc_points=mc_points)
        return jnp.mean(var)
    
    def get_next(self, **kwargs):
        return super().get_next(**kwargs)

class MaxVar(MonteCarloAcquisition):
    """Maximum Variance acquisition function"""

    def __init__(self, gp, mc_kwargs: Optional[dict] = {}, optimizer_kwargs: Optional[dict] = {}):
        super().__init__(gp, mc_kwargs=mc_kwargs, optimizer_kwargs=optimizer_kwargs)

    def __call__(self, x, gp, **kwargs):
        mc_points = kwargs.get('mc_points', self.mc_points)
        var = gp.fantasy_var(x, mc_points=mc_points)

        return jnp.max(var)