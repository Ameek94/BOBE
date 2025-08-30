from typing import Any, List, Optional, Dict, Tuple
import jax
import jax.numpy as jnp
from jax import lax,jit
import numpy as np
from scipy.stats import qmc
from jax.scipy.stats import norm
from jax import config
import tensorflow_probability.substrates.jax as tfp
from .optim import optimize_optax, optimize_scipy
from .utils.logging_utils import get_logger
from .utils.seed_utils import get_numpy_rng
from .nested_sampler import nested_sampling_Dy
from .gp import GP
config.update("jax_enable_x64", True)
log = get_logger("acq")

#------------------Helper functions-------------------------
# These are jax versions of the BoTorch functions.

# def log1mexp(x):
#     """
#     Compute `log(1 - exp(-|x|))` elementwise in a numerically stable way.

#     Args:
#         x: Array-like input.

#     Returns:
#         Array of log(1 - exp(-|x|)) values.

#     #### References

#     [1]: Machler, Martin. Accurately computing log(1 - exp(-|a|))
#          https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
#     """
#     x = jnp.asarray(x, dtype=jnp.float64)  # or jnp.float32
#     x = jnp.abs(x)
    
#     # Use tf.math.expm1 equivalent: expm1(-x) = exp(-x) - 1
#     # So log(-expm1(-x)) = log(1 - exp(-x))
#     # This is more stable for small x
    
#     # Use log1p equivalent: log1p(-exp(-x)) = log(1 - exp(-x))
#     # This is more stable for large x
    
#     # Switching point recommended in [1]
#     return jnp.where(
#         x < jnp.log(2), 
#         jnp.log(-jnp.expm1(-x)),      # More stable for x < log(2)
#         jnp.log1p(-jnp.exp(-x))       # More stable for x >= log(2)
#     )


# def erfcx(x):
#     """Compute erfcx using a Chebyshev expansion (JAX-compatible)."""
#     x = jnp.asarray(x, dtype=jnp.float32)
#     x_abs = jnp.abs(x)

#     # Shift parameter
#     y = (x_abs - 3.75) / (x_abs + 3.75)

#     # Chebyshev coefficients (from Shepherd & Laframboise, 1981)
#     coeff = jnp.array([
#         3e-21,
#         9.7e-20,
#         2.7e-20,
#         -2.187e-18,
#         -2.237e-18,
#         5.0681e-17,
#         7.4182e-17,
#         -1.250795e-15,
#         -1.864563e-15,
#         3.33478119e-14,
#         3.2525481e-14,
#         -9.65469675e-13,
#         1.94558685e-13,
#         2.8687950109e-11,
#         -6.3180883409e-11,
#         -7.75440020883e-10,
#         4.521959811218e-09,
#         1.0764999465671e-08,
#         -2.18864010492344e-07,
#         7.74038306619849e-07,
#         4.139027986073010e-06,
#         -6.9169733025012064e-05,
#         4.90775836525808632e-04,
#         -2.413163540417608191e-03,
#         9.074997670705265094e-03,
#         -2.6658668435305752277e-02,
#         5.9209939998191890498e-02,
#         -8.4249133366517915584e-02,
#         -4.590054580646477331e-03,
#         1.177578934567401754080,
#     ], dtype=jnp.float32)

#     # Clenshaw recurrence for Chebyshev expansion
#     result = -4e-21
#     previous_result = 0.0
#     for c in coeff[:-1]:
#         result, previous_result = (2 * y * result - previous_result + c, result)
#     result = y * result - previous_result + coeff[-1]

#     result = result / (1.0 + 2.0 * x_abs)

#     # Flip approximation for negative x
#     result = jnp.where(x < 0.0, 2.0 * jnp.exp(x**2) - result, result)
#     result = jnp.where(jnp.isinf(x), jnp.array(1.0, dtype=result.dtype), result)

#     return result

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

    def __init__(self, optimizer="optax", optimizer_kwargs: Optional[Dict[str, Any]] = {'name': 'adam', 'lr': 1e-3}):
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        if self.optimizer == "scipy":
            self.acq_optimize = optimize_scipy
        else:
            self.acq_optimize = optimize_optax

    def fun(self, x):
        raise NotImplementedError

    def get_next_point(self, gp: GP,
                       acq_kwargs: Dict[str, Any] = {},
                       maxiter: int = 500,
                       n_restarts: int = 8,
                       verbose: bool = True,
                       early_stop_patience: int = 25,
                       rng=None) -> Tuple[np.ndarray, float]:
        """
        Optimize the acquisition function to obtain the next point to sample.

        Args:
            gp: Gaussian process model
            ndim: Number of dimensions (inferred from bounds if not provided)
            optimizer_name: Name of optimizer to use
            **kwargs: Additional arguments passed to optimize()

        """

        raise NotImplementedError("Base class get_next() not implemented")

    def get_next_batch(self, gp: GP,
                       n_batch: int = 1,
                       acq_kwargs: Dict[str, Any] = {},
                       maxiter: int = 500,
                       n_restarts: int = 8,
                       verbose: bool = True,
                       early_stop_patience: int = 25,
                       rng=None) -> Tuple[np.ndarray, float]:

        """
        Get the next batch of points to sample.
        """

        rng = rng if rng is not None else get_numpy_rng()

        x_batch, acq_vals = [], []

        if n_batch == 1:
            x_next, acq_val_next = self.get_next_point(gp, acq_kwargs=acq_kwargs,
                                        maxiter=maxiter,
                                        n_restarts=n_restarts,
                                        verbose=verbose,
                                        early_stop_patience=early_stop_patience,
                                        rng=rng)
            x_batch.append(x_next)
            acq_vals.append(acq_val_next)

        else:
            if hasattr(gp,'gp'):
                dummy_gp = gp.gp.copy()
            else:
                dummy_gp = gp.copy()

            for i in range(n_batch):
                x_next, acq_val_next = self.get_next_point(dummy_gp, acq_kwargs=acq_kwargs,
                                                        maxiter=maxiter,
                                                        n_restarts=n_restarts,
                                                        verbose=verbose,
                                                        early_stop_patience=early_stop_patience,
                                                        rng=rng)
                x_batch.append(x_next)
                acq_vals.append(acq_val_next)

                mu = dummy_gp.predict_mean_single(x_next)
                dummy_gp.update(x_next, mu,refit=False)

        return np.array(x_batch), np.array(acq_vals)

        # raise NotImplementedError("Base class get_next_batch() not implemented")

class EI(AcquisitionFunction):
    """Expected Improvement acquisition function"""

    name: str = "EI"

    def __init__(self, zeta: float = 0.1, 
                 optimizer: str = "optax", optimizer_kwargs: Optional[Dict[str, Any]] = {'name': 'adam', 'lr': 1e-3}):
        super().__init__(optimizer=optimizer, optimizer_kwargs=optimizer_kwargs)
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
                 maxiter: int = 250,
                 n_restarts: int = 20,
                 verbose: bool = True,
                 early_stop_patience: int = 25,
                 rng=None):

        rng = rng if rng is not None else get_numpy_rng()
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
        jitter = rng.normal(0.,0.001,size=x0_acq.shape)
        x0_acq = jnp.clip(x0_acq + jitter, 0., 1.)
        return self.acq_optimize(fun=self.fun,
                            fun_args=fun_args,
                            fun_kwargs=fun_kwargs,
                            ndim=gp.ndim,
                            x0=x0_acq,
                            optimizer_kwargs=self.optimizer_kwargs,
                            maxiter=maxiter,
                            n_restarts=n_restarts,
                            verbose=verbose)

class LogEI(EI):
    """Log Expected Improvement acquisition function. Better numerical stability compared to EI."""

    name: str = "LogEI"

    def __init__(self, zeta: float = 0.2, optimizer = "optax", optimizer_kwargs: Optional[Dict[str, Any]] = {'name': 'adam', 'lr': 1e-3}):
        super().__init__(zeta=zeta, optimizer=optimizer, optimizer_kwargs=optimizer_kwargs)

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
                 optimizer: str = "optax", optimizer_kwargs: Optional[Dict[str, Any]] = {'name': 'adam', 'lr': 1e-3}):

        super().__init__(optimizer=optimizer, optimizer_kwargs=optimizer_kwargs)

    def fun(self, x, gp,  mc_points=None, k_train_mc = None):
        var = gp.fantasy_var(new_x=x, mc_points=mc_points,k_train_mc=k_train_mc)
        return jnp.mean(var)


    def get_next_point(self, gp,
                 acq_kwargs,
                 maxiter: int = 200,
                 n_restarts: int = 4,
                 verbose: bool = True,
                 early_stop_patience: int = 25,
                 rng=None):
        
        mc_samples = acq_kwargs.get('mc_samples')
        mc_points_size = acq_kwargs.get('mc_points_size', 128)
        mc_points = get_mc_points(mc_samples, mc_points_size=mc_points_size, rng=rng)
        k_train_mc = gp.kernel(gp.train_x, mc_points, gp.lengthscales, gp.outputscale, gp.noise, include_noise=False)

        # @jax.jit
        def mapped_fn(x):
            return self.fun(x, gp, mc_points=mc_points, k_train_mc=k_train_mc)
        acq_vals = lax.map(mapped_fn, mc_points)
        acq_val_min = jnp.min(acq_vals)
        log.info(f"WIPV acquisition min value on MC points: {float(acq_val_min):.4e}")
        best_x = mc_points[jnp.argmin(acq_vals)]
        x0_acq = best_x

        return self.acq_optimize(fun=self.fun,
                                  fun_args=(gp,),
                                  fun_kwargs={'mc_points': mc_points, 'k_train_mc': k_train_mc},
                                  ndim=gp.ndim,
                                  x0=x0_acq,
                                  optimizer_kwargs=self.optimizer_kwargs,
                                  maxiter=maxiter,
                                  n_restarts=n_restarts,
                                  verbose=verbose)

def get_mc_samples(gp,warmup_steps=512, num_samples=512, thinning=4,method="NUTS",init_params=None,np_rng=None,rng_key=None):
    if method=='NUTS':
        try:
            mc_samples = gp.sample_GP_NUTS(warmup_steps=warmup_steps,
            num_samples=num_samples, thinning=thinning
            )
        except Exception as e:
            log.error(f"Error in sampling GP NUTS: {e}")
            mc_samples, logz, success = nested_sampling_Dy(gp, gp.ndim, maxcall=int(2e6)
                                            , dynamic=False, dlogz=0.05,equal_weights=True,rng=np_rng
            )
    elif method=='NS':
        mc_samples, logz, success = nested_sampling_Dy(gp, gp.ndim, maxcall=int(2e6)
                                            , dynamic=False, dlogz=0.05,equal_weights=True,rng=np_rng
        )
    elif method=='uniform':
        mc_samples = {}
        points = qmc.Sobol(gp.ndim, scramble=True, rng=np_rng).random(num_samples)
        mc_samples['x'] = points
    else:
        raise ValueError(f"Unknown method {method} for sampling GP")
    return mc_samples


def get_mc_points(mc_samples, mc_points_size=128, rng=None):
    mc_size = max(mc_samples['x'].shape[0], mc_points_size)
    rng = rng if rng is not None else get_numpy_rng()   
    idxs = rng.choice(mc_size, size=mc_points_size, replace=False)
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
