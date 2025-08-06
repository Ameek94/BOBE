import time
from typing import Any, List, Optional
import jax.numpy as jnp
from jax import lax
import numpy as np
from jax.scipy.stats import norm
from jax import config
from jax.scipy.special import erfc, logsumexp
import logging
from .optim import FunctionOptimizer

config.update("jax_enable_x64", True)
log = logging.getLogger("[ACQ]")

#------------------Helper functions-------------------------
# These are jax versions of the BoTorch functions.

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


def _scaled_improvement(mean, sigma, best_f, maximize=True):
    """Returns `u = (mean - best_f) / sigma`, -u if maximize == True."""
    u = (mean - best_f) / sigma
    return u if maximize else -u

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
    def __init__(self,gp):
        self.gp = gp
    
    def __call__(self, x, gp, **kwargs):
        raise NotImplementedError
    
    def optimize(self, gp, bounds=None, ndim=None, **kwargs):
        """
        Optimize the acquisition function.
        
        Args:
            gp: Gaussian process model
            bounds: Parameter bounds as [(low1, high1), (low2, high2), ...] or None
            ndim: Number of dimensions (inferred from bounds if not provided)
            **kwargs: Additional arguments passed to optimizer.optimize()
        
        Returns:
            Tuple of (best_x, best_value)
        """
        if bounds is None:
            raise ValueError("Bounds must be provided for optimization")
        
        if ndim is None:
            ndim = len(bounds)
        
        # Create objective function that closes over the GP
        def objective(x):
            return self(x, gp)
        
        return self.optimizer.optimize(objective, ndim=ndim, bounds=bounds, **kwargs)

class EI(AcquisitionFunction):
    """Expected Improvement acquisition function"""
    def __init__(self, zeta: float = 0.0):
        super().__init__()
        self.zeta = zeta
    
    def __call__(self, x, gp, **kwargs):
        mu, var = gp.predict(x)
        std = jnp.sqrt(var)
        best_f = gp.train_y.max() - self.zeta
        z = (mu - best_f) / std
        ei = std * (z * norm.cdf(z) + norm.pdf(z))
        return jnp.reshape(-ei, ())

class LogEI(EI):
    """Log Expected Improvement acquisition function"""
    def __init__(self, zeta: float = 0.0, maximize: bool = True):
        super().__init__()
        self.zeta = zeta
        self.maximize = maximize
    
    def __call__(self, x, gp, **kwargs):
        mu, var = gp.predict(x)
        sigma = jnp.sqrt(var)
        best_f = gp.train_y.max() - self.zeta
        
        # Compute scaled improvement
        u = _scaled_improvement(mu, sigma, best_f, self.maximize)
        
        # Compute log EI
        log_ei = _log_ei_helper(u) + jnp.log(sigma)
        
        return jnp.reshape(-log_ei, ())

class MCAcquisition(AcquisitionFunction):
    """Monte Carlo acquisition function base class"""
    def __init__(self, mc_points: Optional[Any] = None):
        super().__init__()
        self.mc_points = mc_points
    
    def __call__(self, x, gp, **kwargs):
        raise NotImplementedError("Subclasses must implement __call__ method")


# class WIPV(MCAcquisition):
#     """Weighted Integrated Posterior Variance acquisition function"""
#     def __init__(self, mc_points: Optional[Any] = None):
#         super().__init__()
#         self.mc_points = mc_points
    
#     def __call__(self, x, gp, **kwargs):
#         mc_points = kwargs.get('mc_points', self.mc_points)
#         var = gp.fantasy_var(x, mc_points=mc_points)

#         return jnp.mean(var)
    
class MaxVar(MCAcquisition):
    """Maximum Variance acquisition function"""
    def __init__(self, mc_points: Optional[Any] = None):
        super().__init__()
        self.mc_points = mc_points
    
    def __call__(self, x, gp, **kwargs):
        mc_points = kwargs.get('mc_points', self.mc_points)
        var = gp.fantasy_var(x, mc_points=mc_points)

        return jnp.max(var)