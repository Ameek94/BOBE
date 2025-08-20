from typing import Any, List, Optional, Dict, Tuple
import jax.numpy as jnp
from jax import lax
import numpy as np
from scipy.stats import qmc
from jax.scipy.stats import norm
from jax import config
import tensorflow_probability.substrates.jax as tfp
from .optim import optimize
from .utils.logging_utils import get_logger
from .utils.seed_utils import get_numpy_rng
from .nested_sampler import nested_sampling_Dy
config.update("jax_enable_x64", True)
log = get_logger("[acq]")

#------------------Helper functions-------------------------
# These are jax versions of the BoTorch functions.

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
    Matches BoTorch branching for stability, based on Ament et al., [arxiv: 2310.20708].
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

#------------------The acquisition function classes-------------------------

class AcquisitionFunction:
    """Base class for acquisition functions"""

    name: str = "BaseAcquisitionFunction"

    def __init__(self, optimizer_kwargs: Optional[Dict[str, Any]] = {}):
        self.optimizer_kwargs = optimizer_kwargs

    def fun(self, x):
        raise NotImplementedError

    def get_next_point(self, fun_args: Tuple = (), fun_kwargs: Dict[str, Any] = {}, step: int = 0) -> Tuple[np.ndarray, float]:
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

        raise NotImplementedError("Base class get_next() not implemented")

    def get_next_batch(self, n_batch: int = 1, fun_args: Tuple = (), fun_kwargs: Dict[str, Any] = {}, step: int = 0) -> Tuple[np.ndarray, float]:

        """
        Get the next batch of points to sample.
        """
        raise NotImplementedError("Base class get_next_batch() not implemented")

class EI(AcquisitionFunction):
    """Expected Improvement acquisition function"""

    name: str = "EI"

    def __init__(self, zeta: float = 0.2, optimizer_kwargs: Optional[Dict[str, Any]] = {}):
        super().__init__(optimizer_kwargs=optimizer_kwargs)
        self.zeta = zeta

    def fun(self, x, gp, best_y, zeta):
        """
        Expected Improvement in pure JAX.
        """
        mu, var = gp.predict_single(x)
        var = jnp.clip(var, a_min=1e-18)  # prevent zero variance
        sigma = jnp.sqrt(var)

        u = _scaled_improvement(mu - zeta, sigma, best_y)
        ei = _ei_helper(u) * sigma

        return jnp.reshape(-ei, ())  # optimizer minimizes this
    
    def get_next_point(self, gp, 
                       acq_kwargs,
                 optimizer_name: str = "adam",
                 lr: float = 0.001,
                 optimizer_kwargs: dict | None = {},
                 maxiter: int = 500,
                 n_restarts: int = 8,
                 verbose: bool = True,
                 early_stop_patience: int = 25,):
        zeta = acq_kwargs.get('zeta', self.zeta)
        best_y = acq_kwargs.get('best_y', max(gp.train_y.flatten()))
        fun_args = (gp, best_y, zeta)
        fun_kwargs = {}
        best_x = gp.train_x[jnp.argmax(gp.train_y)]
        if n_restarts > 1:
            x0_acq = jnp.vstack([gp.get_random_point() for _ in range(n_restarts-1)])
            x0_acq = jnp.vstack([x0_acq, best_x])
        else:
            x0_acq = best_x
        jitter = np.random.normal(0.,0.001,size=x0_acq.shape)
        x0_acq = jnp.clip(x0_acq + jitter, 0., 1.)
        return optimize(fun=self.fun,
                        fun_args=fun_args,
                        fun_kwargs=fun_kwargs,
                        ndim=gp.ndim,
                        x0=x0_acq,
                        lr=lr,
                        optimizer_name=optimizer_name,
                        optimizer_kwargs=optimizer_kwargs,
                        maxiter=maxiter,
                        n_restarts=n_restarts,
                        verbose=verbose,
                        early_stop_patience=early_stop_patience)

class LogEI(EI):
    """Log Expected Improvement acquisition function. Better numerical stability compared to EI."""

    name: str = "LogEI"

    def __init__(self, zeta: float = 0.2, optimizer_kwargs: Optional[Dict[str, Any]] = {}):
        super().__init__(zeta=zeta, optimizer_kwargs=optimizer_kwargs)

    def fun(self, x, gp, best_y, zeta):
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


class WIPV(AcquisitionFunction):
    """Weighted Integrated Posterior Variance acquisition function"""

    name: str = "WIPV"

    def __init__(self,
                 optimizer_kwargs: Optional[Dict[str, Any]] = {}):

        super().__init__(optimizer_kwargs=optimizer_kwargs)
        self.rng = get_numpy_rng()

    def fun(self, x, gp,  mc_points=None, k_train_mc = None):
        var = gp.fantasy_var(new_x=x, mc_points=mc_points,k_train_mc=k_train_mc)
        return jnp.mean(var)

    def get_mc_points(self,mc_samples):
        mc_size = max(mc_samples['x'].shape[0], self.mc_points_size)
        idxs = self.rng.choice(mc_size, size=self.mc_points_size, replace=False)
        return mc_samples['x'][idxs]


    def get_next_point(self, gp,
                 acq_kwargs,
                 optimizer_name: str = "adam",
                 lr: float = 0.001,
                 optimizer_kwargs: dict | None = {},
                 maxiter: int = 200,
                 n_restarts: int = 4,
                 verbose: bool = True,
                 early_stop_patience: int = 25,):
        
        mc_samples = acq_kwargs.get('mc_samples')
        mc_points_size = acq_kwargs.get('mc_points_size', 256)
        mc_points = get_mc_points(mc_samples, mc_points_size=mc_points_size)
        k_train_mc = gp.kernel(gp.train_x, mc_points, gp.lengthscales, gp.outputscale, gp.noise, include_noise=False)

        acq_vals = lax.map(lambda x: self.fun(x, gp, mc_points=mc_points, k_train_mc=k_train_mc), mc_points)
        acq_val_min = jnp.min(acq_vals)
        best_x = mc_points[jnp.argmin(acq_vals)]

        return best_x, float(acq_val_min)
        
        # mc_samples = acq_kwargs.get('mc_samples')
        # mc_points_size = acq_kwargs.get('mc_points_size', 128)
        # mc_points = get_mc_points(mc_samples, mc_points_size=mc_points_size)
        # x0_acq1 = mc_samples['best']
        # vars = lax.map(gp.predict_var,mc_points,batch_size=25)
        # x0_acq2 = mc_points[jnp.argmax(vars)]
        # x0_acq3 = gp.train_x[jnp.argmax(gp.train_y)]
        # x0_acq = jnp.vstack([x0_acq1, x0_acq2, x0_acq3])
        # if n_restarts > 3:
        #     x0_acq = jnp.vstack([x0_acq, [gp.get_random_point() for _ in range(n_restarts - 3)]])
        # else:
        #     x0_acq = x0_acq[:n_restarts]
        # k_train_mc = gp.kernel(gp.train_x, mc_points, gp.lengthscales, gp.outputscale, gp.noise, include_noise=False)
        # fun_kwargs = {'mc_points': mc_points, 'k_train_mc': k_train_mc}

        # return optimize(fun=self.fun,
        #                 fun_args=(gp,),
        #                 fun_kwargs=fun_kwargs,
        #                 ndim=gp.ndim,
        #                 x0=x0_acq,
        #                 lr=lr,
        #                 optimizer_name=optimizer_name,
        #                 optimizer_kwargs=optimizer_kwargs,
        #                 maxiter=maxiter,
        #                 n_restarts=n_restarts,
        #                 verbose=verbose,
        #                 early_stop_patience=early_stop_patience)

def get_mc_samples(gp,warmup_steps=512, num_samples=512, thinning=4,method="NUTS",init_params=None):
    if method=='NUTS':
        try:
            mc_samples = gp.sample_GP_NUTS(warmup_steps=warmup_steps,
            num_samples=num_samples, thinning=thinning
            )
        except Exception as e:
            log.error(f"Error in sampling GP NUTS: {e}")
            mc_samples, logz, success = nested_sampling_Dy(gp, gp.ndim, maxcall=int(2e6)
                                            , dynamic=False, dlogz=0.05,equal_weights=True,
            )
    elif method=='NS':
        mc_samples, logz, success = nested_sampling_Dy(gp, gp.ndim, maxcall=int(2e6)
                                            , dynamic=False, dlogz=0.05,equal_weights=True,
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
    
def get_acquisition_function(name: str, gp, **kwargs) -> AcquisitionFunction:
    """Factory function to get an acquisition function by name."""
    if name == "EI":
        return EI(gp, **kwargs)
    elif name == "LogEI":
        return LogEI(gp, **kwargs)
    elif name == "WIPV":
        return WIPV(gp, **kwargs)
    else:
        raise ValueError(f"Unknown acquisition function: {name}")
