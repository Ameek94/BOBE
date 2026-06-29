"""
Kernel implementations for Gaussian Process models.

All kernels inherit from the base Kernel class and implement the covariance() method.
JAX JIT compilation is handled at higher levels (acquisition functions, optimization).
"""

from abc import ABC, abstractmethod
from math import sqrt, pi
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular
from .priors import build_prior_state, DSLPPrior
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Any, Dict
from .transforms import PrincipalAxesTransform

from .utils.log import get_logger

log = get_logger("kernels")

jax.config.update("jax_enable_x64", True)

# Constants for Matérn kernel
sqrt2 = sqrt(2.)
sqrt3 = sqrt(3.)
sqrt5 = sqrt(5.)

safe_noise_floor = 1e-12

def _as_f64(x):
    return jnp.array(x, dtype=jnp.float64)

def _to_1d(x: jnp.ndarray) -> jnp.ndarray:
    return x.reshape(-1,)

@jax.jit
def gp_mll(k,train_y,num_points):
    """
    Computes the negative marginal log likelihood of the GP
    """
    L = jnp.linalg.cholesky(k)
    alpha = cho_solve((L,True),train_y)
    mll = -0.5*jnp.einsum("ij,ji",train_y.T,alpha) - jnp.sum(jnp.log(jnp.diag(L))) - 0.5*num_points*jnp.log(2*pi)
    return mll

@jax.jit
def fast_update_cholesky(L: jnp.ndarray, k: jnp.ndarray, k_self: float):
    # solve L v = k  -> v has shape (n,)
    v = solve_triangular(L, k, lower=True)

    # new diagonal entry
    diag = jnp.sqrt(k_self - jnp.dot(v, v))

    # print(f"Shapes L: {L.shape}, k: {k.shape}, k_self: {k_self}, v: {v.shape}, diag: {diag.shape}")

    # build a zero (n+1)x(n+1) and fill blocks
    n = L.shape[0]
    new_L = jnp.zeros((n+1, n+1), dtype=L.dtype)
    new_L = new_L.at[:n, :n].set(L)      # top-left
    new_L = new_L.at[n, :n].set(v)       # bottom-left
    new_L = new_L.at[n, n].set(diag)     # bottom-right
    return new_L

@jax.jit
def woodbury_solve(L, y, noise):
    # L: (n,k), y: (n,), noise: scalar
    # Returns alpha = (L L^T + noise I)^{-1} y
    inv_noise = 1.0 / noise

    Lt_y = L.T @ y                                         # (k,)
    B = jnp.eye(L.shape[1]) + inv_noise * (L.T @ L)        # (k,k) the plus sign here is different from gpytorch which is not the standard form of the woodbury 
    cholB = jnp.linalg.cholesky(B)
    tmp = cho_solve((cholB, True), Lt_y)
    alpha = inv_noise * y - (inv_noise**2) * (L @ tmp)
    return alpha, cholB

@jax.jit
def woodbury_logdet(L, noise, cholB):
    # log| noise I + L L^T | = n log noise + log|B|
    n = L.shape[0]
    logdetB = 2.0 * jnp.sum(jnp.log(jnp.diag(cholB)))
    return n * jnp.log(noise) + logdetB

def _hp(hp_init: dict, key: str, default):
    """
    Fetch a hyperparam for hp_init, falling back to default when None
    """
    if hp_init is None:
        return default
    v = hp_init.get(key, default)
    return default if v is None else v


@dataclass(frozen=True)
class HyperParam:
    name: str
    size: int
    bounds_key: str
    transform: str = "log" # "log" or "none"

class HyperParamLayout:
    """
    Handles packing/unpacking optimiser vectors for a list of HyperParam fields.
    Assumes optimser spacing is log-space if transform == "log"
    """

    def __init__(self, fields: List[HyperParam]):
        self.fields = list(fields)

    def names(self) -> Tuple[str, ...]:
        names = []
        for f in self.fields:
            if f.size == 1:
                names.append(f.name)
            else:
                names += [f"{f.name}_{i}" for i in range(f.size)]
        return tuple(names)
    
    def pack_from_kernel(self, kernel) -> jnp.ndarray:
        """
        Returns optimiser-space vector (log-space for log params)
        """
        parts = []
        for f in self.fields:
            v = getattr(kernel, f.name, None)
            if v is None:
                raise ValueError(f"Kernel missing hyperparameter value for '{f.name}'")
            v = jnp.ravel(v)
            if v.shape[0] != f.size:
                raise ValueError(f"Hyperparameter '{f.name}' expected size {f.size}, got {v.shape[0]}")
            parts.append(v)
        #Apply transforms
        out = []
        for f, block in zip(self.fields, parts):
            if f.transform == "log":
                out.append(jnp.log(block))
            elif f.transform == "none":
                out.append(block)
            else:
                raise ValueError(f"Unsupported transform '{f.transform}' for hyperparameter '{f.name}'")
        
        return jnp.concatenate(out, axis=0) if out else jnp.zeros((0,), dtype=jnp.float64)
        
    def unpack_to_dict(self, params_opt: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Parses optimiser-space vector into dict {name: value_in_natural_space}
        """
        params_opt = params_opt.reshape(-1,)
        out: Dict[str, jnp.ndarray] = {}
        idx = 0
        for f in self.fields:
            block = params_opt[idx:idx + f.size]
            if f.transform == "log":
                block = jnp.exp(block)
            elif f.transform == "none":
                pass
            else:
                raise ValueError(f"Unsupported transform '{f.transform}' for hyperparameter '{f.name}'")
            out[f.name] = block[0] if f.size == 1 else block
            idx += f.size
        return out
    
    def bounds_opt(self, kernel) -> jnp.ndarray:
        """
        Return optimiser-space bounds array shaped (2, P)
        Supports bounds as:
        - (2,) global -> replicated to size
        - (size, 2) per-dim
        """
        blocks = []
        for f in self.fields:
            b = kernel.bounds_spec.get(f.bounds_key, None) #kernel-level hook for transforms
            if b is None:
                raise ValueError(f"Kernel requires bounds for '{f.bounds_key}' (for param '{f.name}')")
            b = jnp.array(b)
            if b.shape == (2,):
                block = jnp.repeat(b.reshape(2, 1), f.size, axis=1) #(2, size)
            elif b.shape == (f.size, 2):
                block = b.T #(2, size)
            else:
                raise ValueError(f"Bounds for '{f.bounds_key}' must be shape (2,) or ({f.size}, 2); got {b.shape}")
            
            if f.transform == "log":
                block = jnp.log(block)
            elif f.transform == "none":
                pass
            else:
                raise ValueError(f"Unsupported transform '{f.transform}' for  '{f.name}'")

            blocks.append(block)
        
        return jnp.concatenate(blocks, axis=1) if blocks else jnp.zeros((2, 0), dtype=jnp.float64)



class Kernel(ABC):
    """
    Abstract base class for all kernels in BOBE.
    
    Attributes
    ----------
    lengthscales : jnp.ndarray
        Lengthscale parameters for each dimension, shape (D,)
    kernel_variance : float
        Overall variance/amplitude of the kernel
    noise : float
        Observation noise level
    """
    
    def __init__(self, hp_init: dict, noise=1e-8):
        """
        Initialize kernel with hyperparameters.
        
        Parameters
        ----------
        hp_init: dict, optional
            Dictionary with all user parsed hyperparameter initial values.
        noise : float, optional
            Noise level added to diagonal. Default is 1e-8.
        """
        
        # Get hyperparameter bounds and prior settings
        self.hp_init = hp_init or {}

        self.bounds_spec = self.hp_init.get("bounds", {}) or {}
        self.prior_spec = self.hp_init.get("priors", {}) or {}

        # Get hyperparameter initial values or set to default
        self.lengthscales = _hp(hp_init, "lengthscales", None)
        if self.lengthscales is None:
            raise ValueError("Kernel requires 'lengthscales' initialisation")
        self.lengthscales = jnp.array(self.lengthscales, dtype=jnp.float64)
        self.ndim = self.lengthscales.shape[0]

        self.kernel_variance = _hp(hp_init, "kernel_variance", None)
        self.kernel_variance = jnp.array(self.kernel_variance, dtype=jnp.float64) if self.kernel_variance is not None else jnp.array(1.0, dtype=jnp.float64)
        
        self.tausq = _hp(hp_init, "tausq", None)
        if self.tausq is not None:
            self.tausq = jnp.array(self.tausq, dtype=jnp.float64)

        self.noise = noise

        # Prior flags (set later by configure_priors)
        self.fixed_kernel_variance = False
        self.tausq_enabled = False

        # Prior objects to be configured
        self.prior = None

        # GP Posterior State
        self._is_fit = False
        self.cholesky = None
        self.alphas = None

        self.train_x = None
        self.train_y = None

        # Hyperparameter optimisation metadata
        self.hp_layout = None
        self.hyperparam_names = None
        self.hyperparam_bounds = None
        self.num_hyperparams = None 

        # Input transform hooks (initialialised as identity)
        self.input_transform = None

    
    def set_input_transform(self, tform):
        self.input_transform = tform

    def clear_input_transform(self):
        self.input_transform = None
    
    def _x(self, x):
        if self.input_transform is None or not self.input_transform.is_active:
            return x
        return self.input_transform.forward(x)
    
    def configure_hyperparam_optimisation(self):
        # If using a transform-aware default, refresh bounds here
        # if "lengthscales" not in self.bounds_spec or self.input_transform is not None:
        #     self.bounds_spec["lengthscales"] = self.default_lengthscale_bounds()
        # else:
        # # only refresh automatically if the transform actually changes scale
        #     t = self.input_transform
        #     if (
        #         t is not None
        #         and getattr(t, "is_active", False)
        #         and isinstance(t, PrincipalAxesTransform)
        #         and t.mode == "whiten"
        #     ):
        #         self.bounds_spec["lengthscales"] = self.default_lengthscale_bounds()

        fields = self.base_hyperparams() + (self.prior.extra_hyperparams(self) if self.prior is not None else [])
        self.hp_layout = HyperParamLayout(fields)
        self.hyperparam_names = self.hp_layout.names()
        self.hyperparam_bounds = self.hp_layout.bounds_opt(self)
        self.num_hyperparams = self.hyperparam_bounds.shape[1]
    
    def configure_priors(self):
        ps = self.prior_spec
        name = ps.get("name", "dslp").lower()

        if name == "dslp":
            self.prior = DSLPPrior().configure(self)
        else:
            raise ValueError(f"Unkown Prior '{name}'")

    
    def update_hyperparams(self, parsed=None, **kwargs):
        if parsed is not None:
            kwargs.update(parsed)

        if "lengthscales" in kwargs:
            self.lengthscales = kwargs["lengthscales"]
        if "kernel_variance" in kwargs:
            self.kernel_variance = kwargs["kernel_variance"]
        if "tausq" in kwargs and self.tausq is not None:
            self.tausq = kwargs["tausq"]
    
    def logprior(self, **parsed) -> jnp.ndarray:
        if self.prior is None:
            return 0.0
        return self.prior.logprior(**parsed)

    # def default_lengthscale_bounds(self, lower_factor=0.1, upper_factor=5.0):
    #     """
    #     Returns natural-space bounds for lengthscales.
    #     If an active PrincipalAxesTransform is attached, define bounds in the
    #     transformed coordinate system using its axis scales.
    #     Otherwise fall back to existing scalar bounds if present.
    #     """
    #     t = self.input_transform
    #     if t is not None and getattr(t, "is_active", False):
    #         if isinstance(t, PrincipalAxesTransform): 
    #             if t.mode == "whiten":
    #                 return t.recommended_lengthscale_bounds(
    #                     lower_factor=lower_factor,
    #                     upper_factor=upper_factor,
    #                 )
    #     # fallback to user-provided/global bounds
    #     b = self.bounds_spec.get("lengthscales", None)
    #     if b is None:
    #         raise ValueError("No bounds provided for 'lengthscales'")
    #     return jnp.array(b)
        
    
    def build_posterior_cache(self, train_x: jnp.ndarray, train_y: jnp.ndarray) -> None:
        """
        Build and store linear algebra objected neede for prediction / fantasy
        self.cholesky (lower-triangle)
        self.alphas   (K^{-1} y)
        """
        K = self.covariance(train_x, train_x, include_noise=True)
        L = jnp.linalg.cholesky(K)
        alpha = cho_solve((L, True), train_y)

        self.cholesky = L
        self.alphas = alpha

        self.train_x = train_x
        self.train_y = train_y
        self._is_fit = True
    
    def sq_dist(self, xa, xb):
        """
        Compute squared Euclidean distance between two sets of points.
        
        This utility method is used by many kernel implementations.
        
        Parameters
        ----------
        xa : jnp.ndarray
            First set of points, shape (n1, D)
        xb : jnp.ndarray
            Second set of points, shape (n2, D)
            
        Returns
        -------
        sq_dist : jnp.ndarray
            Squared distances, shape (n1, n2)
        """
        return jnp.sum(jnp.square(xa[:, None, :] - xb[None, :, :]), axis=-1)
    
    @abstractmethod
    def covariance(self, xa, xb, include_noise=True):
        """
        Compute covariance matrix between two sets of points.
        
        Parameters
        ----------
        xa : jnp.ndarray
            First set of points, shape (n1, D)
        xb : jnp.ndarray
            Second set of points, shape (n2, D)
        include_noise : bool, optional
            Whether to add noise to diagonal (only when xa is xb). Default is True.
            
        Returns
        -------
        K : jnp.ndarray
            Covariance matrix of shape (n1, n2)
        """
        pass
    
    def diagonal(self, x, include_noise=True):
        """
        Compute only the diagonal of the kernel matrix K(x,x).
        
        For stationary kernels, the diagonal is constant: kernel_variance (+ noise).
        Override this method if your kernel has a non-constant diagonal.
        
        Parameters
        ----------
        x : jnp.ndarray
            Points at which to compute diagonal, shape (n, D)
        include_noise : bool, optional
            Whether to include noise in diagonal. Default is True.
            
        Returns
        -------
        diag : jnp.ndarray
            Diagonal values, shape (n,)
        """
        x = self._x(x)
        diag = self.kernel_variance * jnp.ones(x.shape[0])
        if include_noise:
            diag += self.noise
        return diag

    def predict_mean_single(self, x: jnp.ndarray, y_mean: float, y_std: float) -> jnp.ndarray:
        """
        Single point prediction of mean
        """
        if self.alphas is None:
            raise ValueError("Kernel posteror cache missing")
        
        x = jnp.atleast_2d(x)
        k12 = self.covariance(self.train_x, x, include_noise=False) # shape (N,1)
        mean = jnp.einsum('ij,ji', k12.T, self.alphas)
        return mean*y_std + y_mean 
    
    def predict_var_single(self, x: jnp.ndarray, y_std: float) -> jnp.ndarray:
        if self.cholesky is None:
            raise ValueError("Kernel posteror cache missing")
        
        x = jnp.atleast_2d(x)
        k12 = self.covariance(self.train_x, x, include_noise=False) # shape (N,1)
        vv = solve_triangular(self.cholesky, k12, lower=True) # shape (N,1)
        k22 = self.diagonal(x, include_noise=True) # shape (1,) for x (1,ndim)
        var = k22 - jnp.sum(vv*vv,axis=0) 
        var = jnp.clip(var, safe_noise_floor, None)
        return y_std**2 * var.squeeze()
    
    def fantasy_var(self, new_x: jnp.ndarray,  mc_points: jnp.ndarray, k_train_mc: jnp.ndarray, y_std: float) -> jnp.ndarray:
        """
        Computes the variance of the GP at the mc_points assuming a single point new_x is added to the training set
        """
        if self.cholesky is None:
            raise ValueError("Kernel posteror cache missing")
        
        new_x = jnp.atleast_2d(new_x)
        # new_train_x = jnp.concatenate([self.train_x,new_x])
        k = self.covariance(self.train_x, new_x, include_noise=False).flatten()           # shape (n,)
        k_self = self.diagonal(new_x, include_noise=True)[0]  # scalar
        k11_cho = fast_update_cholesky(self.cholesky,k,k_self)

        # Compute only the extra row for new_x
        k_new_mc = self.covariance(new_x, mc_points, include_noise=False)  # shape (1, n_mc)
        k12 = jnp.vstack([k_train_mc,k_new_mc])
        k22 = self.diagonal(mc_points, include_noise=True) # (N_mc,)
        vv = solve_triangular(k11_cho, k12, lower=True) # shape (N_train,N_mc)
        var = k22 - jnp.sum(vv*vv,axis=0) 
        # handle nans and negative variances due to numerical issues
        var = jnp.where(jnp.isnan(var),safe_noise_floor,var)
        var = jnp.where(var<safe_noise_floor,safe_noise_floor,var)
        return var * y_std**2 
    
    def mll(self, params_opt: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the negative log marginal likelihood for the GP with given hyperparameters.
        """
        if not self._is_fit:
            raise ValueError("Kernel posteror cache missing")
        
        parsed = self.hp_layout.unpack_to_dict(params_opt)
        self.update_hyperparams(parsed=parsed)
        
        K = self.covariance(self.train_x, self.train_x, include_noise=True)
        mll = gp_mll(K, self.train_y, self.train_y.shape[0])
        
        # Add prior
        mll += self.logprior(**parsed)
        
        return -mll
    
    def __call__(self, xa, xb, include_noise=True):
        """Convenience method - same as covariance()"""
        return self.covariance(xa, xb, include_noise=include_noise)

class RBFKernel(Kernel):
    """
    Radial Basis Function (RBF) / Squared Exponential kernel.
    
    k(x, x') = σ² * exp(-0.5 * ||x - x'||²/ℓ²)
    
    where σ² is kernel_variance and ℓ is lengthscale.
    """

    def base_hyperparams(self) -> List[HyperParam]:
        """
        Returns list of required hyperparams for this kernel
        """
        fields = [
            HyperParam(name="lengthscales", size=self.ndim, bounds_key="lengthscales", transform="log"),
        ]
        if not getattr(self, "fixed_kernel_variance", False):
            fields.append(
                HyperParam(name="kernel_variance", size=1, bounds_key="kernel_variance", transform="log")
            )
        return tuple(fields)

    
    def covariance(self, xa, xb, include_noise=True):
        """
        Compute RBF covariance matrix.
        
        Parameters
        ----------
        xa : jnp.ndarray
            First set of input points, shape (n1, d).
        xb : jnp.ndarray
            Second set of input points, shape (n2, d).
        include_noise : bool, optional
            Whether to include noise on diagonal. Default is True.
            
        Returns
        -------
        jnp.ndarray
            Kernel matrix of shape (n1, n2).
        """
        xa = self._x(xa)
        xb = self._x(xb)

        # Scale inputs by lengthscales
        xa_scaled = xa / self.lengthscales
        xb_scaled = xb / self.lengthscales
        
        # Compute squared distances
        sq_dist = self.sq_dist(xa_scaled, xb_scaled)
        
        # Apply RBF kernel
        K = self.kernel_variance * jnp.exp(-0.5 * sq_dist)
        
        # Add noise to diagonal if needed
        if include_noise:
            K += self.noise * jnp.eye(K.shape[0])
        
        return K



class MaternKernel(Kernel):
    """
    Matérn-5/2 kernel.
    
    k(x, x') = σ² * (1 + √5*d + 5*d²/3) * exp(-√5*d)
    
    where d = ||x - x'||/ℓ, σ² is kernel_variance, and ℓ is lengthscale.
    """
    
    def covariance(self, xa, xb, include_noise=True):
        """
        Compute Matérn-5/2 covariance matrix.
        
        Parameters
        ----------
        xa : jnp.ndarray
            First set of input points, shape (n1, d).
        xb : jnp.ndarray
            Second set of input points, shape (n2, d).
        include_noise : bool, optional
            Whether to include noise on diagonal. Default is True.
            
        Returns
        -------
        jnp.ndarray
            Kernel matrix of shape (n1, n2).
        """
        # Scale inputs by lengthscales
        xa_scaled = xa / self.lengthscales
        xb_scaled = xb / self.lengthscales
        
        # Compute squared distances
        dsq = self.sq_dist(xa_scaled, xb_scaled)
        
        # Safe sqrt to avoid division by zero
        d = jnp.sqrt(jnp.where(dsq < 1e-30, 1e-30, dsq))
        
        # Matérn-5/2 formula
        exp_term = jnp.exp(-sqrt5 * d)
        poly_term = 1. + d * (sqrt5 + d * 5. / 3.)
        K = self.kernel_variance * poly_term * exp_term
        
        # Add noise to diagonal if needed
        if include_noise:
            K += self.noise * jnp.eye(K.shape[0])
        
        return K


class SphericalLinearKernel(Kernel):
    """
    Spherical Linear kernel.
    
    k(x, x') = b_0 + b_1 * <P(z), P(z')>
    where P is the inverse sterographic projection onto the unit sphere.
    """

    def __init__(self, hp_init, noise=1e-8):
        super().__init__(hp_init, noise)
        # Posterior state (set by fit)

        self.kernel_variance = None
        self.fixed_kernel_variance = False
        self.tausq_enabled = False

        self.raw_coeffs = jnp.array(_hp(hp_init, "raw_coeffs", [0.0, 0.0]), dtype=jnp.float64)
        self.raw_global_lengthscale = jnp.array(_hp(hp_init, "raw_global_lengthscale", 0.0), dtype=jnp.float64)

        # self.raw_coeffs = jnp.array([0.0, 0.0], dtype=jnp.float64)         # default -> softmax = (0.5, 0.5)
        # self.raw_global_lengthscale = jnp.array(0.0, dtype=jnp.float64)      # default -> sigmoid(0) = 0.5

        # bounds
        self.mins = None
        self.maxs = None
        self.centres = None
        self.ndim = self.lengthscales.shape[0]
        input_bounds = self.hp_init.get("input_bounds", (0.0, 1.0))
        self.mins, self.maxs, self.centres = self.ensure_bounds(input_bounds, self.ndim, dtype=jnp.float64)

        # posterior state (set by fit)
        self.Phi = None
        self.alpha= None
        self.cholB = None
        # self.L = None
        # self.mu_w = None

    def configure_hyperparam_optimisation(self):
        lengthscale_bounds = self.bounds_spec["lengthscales"]
        raw_coeffs_bounds = self.bounds_spec['raw_coeffs']
        raw_global_lengthscale_bounds = self.bounds_spec['raw_global_lengthscale']
        # if raw_global_lengthscale_bounds is None:
        #     raw_global_lengthscale_bounds = [0.1*jnp.sqrt(self.ndim), 10*jnp.sqrt(self.ndim)]

        if lengthscale_bounds is None or raw_coeffs_bounds is None or raw_global_lengthscale_bounds is None:
            raise ValueError(f"Spherical Linear Kernel requires bounds for lengthscales, raw coeffs and raw global lengthscale: {lengthscale_bounds=}, {raw_coeffs_bounds=}, {raw_global_lengthscale_bounds=}")


        self.hyperparam_names = tuple(
            [f"log_lengthscales_{i}" for i in range(self.ndim)] + ["raw_coeffs", "raw_global_lengthscale"]
        )
        
        ls_b = jnp.log(jnp.array(lengthscale_bounds, dtype=jnp.float64)) # (2,)
        ls_block = jnp.stack(
            [
                jnp.full((self.ndim,), ls_b[0]), 
                jnp.full((self.ndim,), ls_b[1]),
            ], 
        axis=0
        ) # (2, D)
        
        rc_b = jnp.array(raw_coeffs_bounds, dtype=jnp.float64)
        rg_b = jnp.array(raw_global_lengthscale_bounds, dtype=jnp.float64)

        rc_block = jnp.stack(
            [
                jnp.full((2,), rc_b[0]), 
                jnp.full((2,), rc_b[1])
            ], 
            axis=0
            ) # (2, 2)
        rg_block = jnp.stack(
            [
                jnp.full((1,), rg_b[0]), 
                jnp.full((1,), rg_b[1])
            ], 
            axis=0) # (2, 1)

        self.hyperparam_bounds = jnp.concatenate([ls_block, rc_block, rg_block], axis=1) # (2, D+3)
        self.num_hyperparams = int(self.hyperparam_bounds.shape[1])


    def logprior(self, lengthscales, raw_coeffs, raw_global_lengthscale):
        raise NotImplementedError
    

    def update_hyperparams(self, parsed=None, lengthscales=None, raw_coeffs=None, raw_global_lengthscale=None, noise=None):
        if parsed is not None:
            if isinstance(parsed, dict):
                lengthscales = parsed.get("lengthscales", lengthscales)
                raw_coeffs = parsed.get("raw_coeffs", raw_coeffs)
                raw_global_lengthscale = parsed.get("raw_global_lengthscale", raw_global_lengthscale)
            else:
                if len(parsed) != 3:
                    raise ValueError(f"Expected (lengthscales, raw_coeffs, raw_global_lengthscale), got {len(parsed)}")
                lengthscales, raw_coeffs, raw_global_lengthscale = parsed

        if lengthscales is not None:
            self.lengthscales = jnp.array(lengthscales)
        if raw_coeffs is not None:
            self.raw_coeffs = jnp.array(raw_coeffs)
        if raw_global_lengthscale is not None:
            self.raw_global_lengthscale = jnp.array(raw_global_lengthscale)
        if noise is not None:
            self.noise = noise

    def parse_hyperparams(self, log_params):
        log_params = log_params.reshape(-1,)
        lengthscales = jnp.exp(log_params[:self.ndim])
        raw_coeffs = log_params[self.ndim:self.ndim+2]
        raw_global_lengthscale = log_params[self.ndim+2]
        return {
            "lengthscales": lengthscales, 
            "raw_coeffs": raw_coeffs, 
            "raw_global_lengthscale": raw_global_lengthscale,
        }
    def get_hyperparams(self):
        return jnp.concatenate(
            [
                jnp.log(self.lengthscales),
                self.raw_coeffs.reshape(-1),
                jnp.atleast_1d(self.raw_global_lengthscale),
            ],
            axis=0,
        )
    
    def initial_log_params(self):
        return self.get_hyperparams()
    
    def build_posterior_cache(self, train_x, train_y):
        Phi = self.spherical_linear_features(train_x)
        y = jnp.squeeze(train_y)

        alpha, cholB = woodbury_solve(Phi, y, self.noise)

        self.Phi = Phi
        self.alpha = alpha
        self.cholB = cholB

        self.logdetKy = woodbury_logdet(Phi, self.noise, cholB)

        self.train_x = train_x
        self.train_y = train_y
        self._is_fit = True

    def project_onto_unit_sphere(self, x: jnp.ndarray) -> jnp.ndarray:
        x_sq_norm = jnp.sum(x * x, axis=-1, keepdims=True)
        x_ = jnp.concatenate([2 * x, (x_sq_norm - 1.0)], axis=-1) 
        x_ *= 1.0 / (1.0 + x_sq_norm)
        return x_
    
    def ensure_bounds(self, bounds, D: int, dtype=jnp.float64):
        """
        bounds: either (min, max) or sequence of (min_d, max_d)
        returns: mins, maxs, centres as (D,)
        """
        if isinstance(bounds, (float, int)):
            raise TypeError("bounds must be (min, max) or array-like of shape (D,2); got scalar.")

        if isinstance(bounds, (tuple, list)) and len(bounds) == 2 and isinstance(bounds[0], (float, int)):
            mins = jnp.full((D,), bounds[0], dtype=dtype)
            maxs = jnp.full((D,), bounds[1], dtype=dtype)
        else:
            b = jnp.asarray(bounds, dtype=dtype)
            if b.shape != (D, 2):
                raise ValueError(f"bounds must have shape (D,2) with D={D}; got {b.shape}.")
            mins = b[:, 0]
            maxs = b[:, 1]
        centres = 0.5 * (mins + maxs)
        return mins, maxs, centres

    
    def spherical_linear_features(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        X: (N, D) assumed within [mins, maxs]
        returns Phi: (N, D+2) where kernel is Phi @ Phi.T
        """
        # constants
        lengthscale = self.lengthscales
        max_sq_norm = jnp.sum(((self.maxs - self.mins) / (2.0 * lengthscale))**2, keepdims=True)
        global_ls = jax.nn.sigmoid(self.raw_global_lengthscale)
        global_ls_eff = jnp.sqrt(global_ls * max_sq_norm)

        # centre and scale
        X1 = (X - self.centres) / lengthscale
        X1 = X1 / global_ls_eff

        # project onto sphere
        S = self.project_onto_unit_sphere(X1)

        # weighted concat
        terms = jax.nn.softmax(self.raw_coeffs)
        term0_sqrt = jnp.sqrt(terms[0])
        term1_sqrt = jnp.sqrt(terms[1])

        Phi = jnp.concatenate(
            [
                S * term1_sqrt,
                jnp.full((X.shape[0], 1), term0_sqrt, dtype=X.dtype)
            ],
            axis=-1
        )

        return Phi

    
    def covariance(self, xa, xb, include_noise=True):
        """
        Feature-space kernel: K = Phi(xa) Phi(xb)^T
        """
        Phi_a = self.spherical_linear_features(xa)
        Phi_b = self.spherical_linear_features(xb)
        K = Phi_a @ Phi_b.T

        if include_noise:
            K += self.noise * jnp.eye(K.shape[0], dtype=K.dtype)
        return K
    
    def diagonal(self, x, include_noise=True):
        Phi = self.spherical_linear_features(x)
        diag = jnp.sum(Phi * Phi, axis=-1)
        if include_noise:
            diag += self.noise
        return diag
    

    def mll(self, log_params) -> jnp.ndarray:
        """
        Feature space log marginal likelihood analogue.
        """

        if not self._is_fit:
            raise ValueError("Kernel posteror cache missing")
        
        parsed = self.parse_hyperparams(log_params)
        self.update_hyperparams(parsed=parsed)

        X = self.train_x
        y = jnp.squeeze(self.train_y)

        Phi = self.spherical_linear_features(X)
        alpha, cholB = woodbury_solve(Phi, y, self.noise)
        logdetKy = woodbury_logdet(Phi, self.noise, cholB)

        N = Phi.shape[0]
        mll_val = -0.5 * (y @ alpha + logdetKy + N * jnp.log(2.0 * jnp.pi))

        mll_val += self.logprior(
            lengthscales=parsed.get("lengthscales", self.lengthscales), 
            raw_coeffs=parsed.get("raw_coeffs", self.raw_coeffs), 
            raw_global_lengthscale=parsed.get("raw_global_lengthscale", self.raw_global_lengthscale)) 

        return -(mll_val)
    

    def predict_mean_single(self, x: jnp.ndarray, y_mean: float, y_std: float) -> jnp.ndarray:
        if self.alpha is None or self.Phi is None:
            raise ValueError("Kernel not fit")
        
        phi_x = jnp.squeeze(self.spherical_linear_features(jnp.atleast_2d(x)), axis=0)
        k_xX = phi_x @ self.Phi.T
        mu_std = k_xX @ self.alpha
        return mu_std * y_std + y_mean
    
    
    def predict_var_single(self, x: jnp.ndarray, y_std: int) -> jnp.ndarray:
        phi_x = jnp.squeeze(self.spherical_linear_features(jnp.atleast_2d(x)), axis=0)

        k_xx = phi_x @ phi_x
        k_xX = phi_x @ self.Phi.T

        Ky_inv_kXx, _ = woodbury_solve(self.Phi, k_xX, self.noise)
        var_latent = k_xx - (k_xX @ Ky_inv_kXx)

        var_latent = jnp.maximum(var_latent, safe_noise_floor)

        return var_latent * (y_std**2) 
    
    def fantasy_var(self, new_x: jnp.ndarray, mc_points: jnp.ndarray, k_train_mc: jnp.ndarray, y_std: int) -> jnp.ndarray:
        """
        Woodbury version of fantasy variance
        Computes posterior variance at MC points after adding a fantasy observation at new_x
        """

        # Features
        Phi_train = self.Phi                                                                 # (N, k)
        Phi_mc = self.spherical_linear_features(mc_points)                                   # (M, k)
        phi_new = jnp.squeeze(self.spherical_linear_features(jnp.atleast_2d(new_x)), axis=0) # (k,)

        # Kernel blocks in feature space
        K_mX = Phi_mc @ Phi_train.T                                                          # (M, N)
        k_Xx = Phi_train @ phi_new                                                           # (N,)
        K_mm_diag = jnp.sum(Phi_mc * Phi_mc, axis=1)                                         # (M,)

        # Woodbury Solves
        Ky_inv_kXx, _ = woodbury_solve(Phi_train, k_Xx, self.noise)                          # (N,)
        Ky_inv_KXm, _ = woodbury_solve(Phi_train, Phi_train @ Phi_mc.T, self.noise)          # (N, M)

        # Base posterior variance at MC points
        proj_diag = jnp.sum(K_mX * Ky_inv_KXm.T, axis=1)                                     # diag(K_mX Ky^{-1} K_Xm)
        s = K_mm_diag - proj_diag

        # Rank-1 fantasy update
        c = K_mX @ Ky_inv_kXx
        denom = 1.0 + k_Xx @ Ky_inv_kXx

        var = s - (c * c) / denom

        var = jnp.maximum(var, safe_noise_floor)

        return var * (y_std**2)
    

class AdditiveKernel(Kernel):
    """ 
    Additive RBF kernel over groups of dimensions.

    K(x, x') = sum_{g=1..G} s_g * exp(-0.5 ||(x_g - x'_g) / ell_g||^2)
    - Always has per dim lengthscales (ell_j, j=1..D)
    - If enable_group_outputscale=False -> Single global outputscale:
        K(x, x') = kernel_variance * sum_g exp(-0.5 * dist_g^2)
    - If enable_group_outputscale=True -> Outputscale per kernel
        K(x, x') = sum_g group_outputscales[g] * exp(-0.5 * dist_g^2)

    """
    def __init__(self, hp_init, noise=1e-8):
        super().__init__(hp_init, noise)

        groups = hp_init.get("groups", None)

        if groups is None or len(groups) == 0:
            raise ValueError(f"AdditiveKernel requires groups > 0, got {groups}")

        self.groups = [jnp.array(g, dtype=jnp.int32) for g in groups]
        self.num_groups = len(self.groups)

        self.enable_group_outputscale = bool(hp_init.get("enable_group_outputscale", False))

        if self.enable_group_outputscale:
            g_os = hp_init.get("group_outputscales", None) if hp_init is not None else None
            if g_os is None:
                g_os = jnp.ones((self.num_groups,), dtype=jnp.float64)
            self.group_outputscales = jnp.array(g_os, dtype=jnp.float64)
            if self.group_outputscales.shape != (self.num_groups,):
                raise ValueError(f"group_outputscales must have shape ({self.num_groups},) got {self.group_outputscales.shape}")
        else:
            self.group_outputscales = None
        
        all_idx = jnp.concatenate(self.groups, axis=0)
        if jnp.any(all_idx < 0) or jnp.any(all_idx >= self.ndim):
            raise ValueError(f"group indices must be in [0, {self.ndim-1}]. Got min={int(all_idx.min())}, max={int(all_idx.max())}")
        sorted_idx = jnp.sort(all_idx)
        if sorted_idx.shape[0] != jnp.unique(sorted_idx).shape[0]:
            raise ValueError("AdditiveKernel groups overlap (duplicate dimension indices).")
        if sorted_idx.shape[0] != self.ndim:
            present = set(map(int, np.array(sorted_idx)))
            missing = [d for d in range(self.ndim) if d not in present]
            raise ValueError(
                f"AdditiveKernel groups must form a partition of all D={self.ndim} dims. "
                f"Missing dims: {missing}"
            )


    def param_spec(self):
        base = list(super().param_spec())

        if self.enable_group_outputscale:
            new = []
            for ps in base:
                if ps.name == "kernel_variance":
                    new.append(ps._replace(enabled_fn=lambda _self: False))
                else:
                    new.append(ps)
            def g_os_enabled(_self): 
                return True
            def g_os_default(_self):
                v = getattr(_self, "group_outputscales", None)
                if v is None:
                    v = jnp.ones((self.num_groups,), dtype=jnp.float64)
                    _self.group_outputscales = v
                return _as_f64(v)
            new.append(
                ParamSpec(
                    names="group_outputscales",
                    size=int(self.num_groups),
                    bounds_key="kernel_variance",
                    transform="log",
                    enabled_fn=g_os_enabled,
                    default_fn=g_os_default,
                    label_prefix="log_group_outputscales",
                )
            )
            return tuple(new)
        return tuple(base)
    
    # def configure_hyperparam_optimisation(self):
    #     """
    #     Optimiser-space params are log-space (consistent with base kernel)
    #     Always has:
    #         log_lengthscales[0:D]
    #     If enable_group_outputscale==False;
    #         log_kernel_variance (unless fixed)
    #     If enabled_group_outputscale==True:
    #         log_group_outputscales[0:G]
    #     If tausq_enabled:
    #         log_tausq
    #     """

    #     lengthscale_bounds = self.bounds_spec.get("lengthscales", None)
    #     kernel_variance_bounds = self.bounds_spec.get("kernel_variance", None)
    #     tausq_bounds = self.bounds_spec.get("tausq", None)

    #     if lengthscale_bounds is None:
    #         raise ValueError("AdditiveKernel requires bounds for lengthscales")
        
    #     names = [f"log_lengthscales_{i}" for i in range(self.ndim)]
    #     bounds_list = [lengthscale_bounds] * self.ndim

    #     if self.enable_group_outputscale:
    #         if kernel_variance_bounds is None:
    #             raise ValueError("AdditiveKernel (group outputscales) requires kernel_variance bounds to use as group-scale bounds")
    #         names += [f"log_group_outputscales_{g}" for g in range(self.num_groups)]
    #         bounds_list += [kernel_variance_bounds]*self.num_groups
    #     else:
    #         if not getattr(self, "fixed_kernel_variance", False):
    #             if kernel_variance_bounds is None:
    #                 raise ValueError("AdditiveKernel requires bounds for kernel_variance")
    #             names.append("log_kernel_variance")
    #             bounds_list.append(kernel_variance_bounds)
    #     if getattr(self, "tausq_enabled", False):
    #         if tausq_bounds is None:
    #             raise ValueError("AdditiveKernel requires bounds for tausq when enabled")
    #         names.append("log_tausq")
    #         bounds_list.append(tausq_bounds)

    #     self.hyperparam_names = tuple(names)
        
    #     b = jnp.array(bounds_list, dtype=jnp.float64).T    # (2, P)
    #     self.hyperparam_bounds = jnp.log(b)
    #     self.num_hyperparams = int(self.hyperparam_bounds.shape[1])
        
    # def parse_hyperparams(self, log_params):
    #     log_params = log_params.reshape(-1,)
    #     hyp = jnp.exp(log_params)

    #     idx = 0
    #     lengthscales = hyp[idx:idx + self.ndim]
    #     idx += self.ndim

    #     if self.enable_group_outputscale:
    #         group_outputscales = hyp[idx:idx + self.num_groups]
    #         idx += self.num_groups
    #         if getattr(self, "tausq_enabled", False):
    #             tausq = hyp[idx]
    #         else:
    #             tausq = getattr(self, "tausq", 1.0)
    #         return lengthscales, group_outputscales, tausq
        
    #     if getattr(self, "fixed_kernel_variance", False):
    #         kernel_variance = self.kernel_variance
    #     else:
    #         kernel_variance = hyp[idx]
    #         idx += 1
    #     if getattr(self, "tausq_enabled", False):
    #         tausq = hyp[idx]
    #     else:
    #         tausq = getattr(self, "tausq", 1.0)
    #     return lengthscales, kernel_variance, tausq
    
    # def get_hyperparams(self):
    #     """
    #     Return hyperparams in non-log space
    #     """
    #     parts = [self.lengthscales]

    #     if self.enable_group_outputscale:
    #         parts.append(self.group_outputscales)
    #     else:
    #         if not getattr(self, "fixed_kernel_variance", False):
    #             parts.append(jnp.array([self.kernel_variance], dtype=jnp.float64))
    #     if getattr(self, "tausq_enabled", False):
    #         parts.append(jnp.array([getattr(self, "tausq", 1.0)], dtype=jnp.float64))
    #     return jnp.concatenate(parts, axis=0)
    
    # def initial_log_params(self):
    #     return jnp.log(self.get_hyperparams())
    
    # def update_hyperparams(self, *parsed, lengthscales=None, kernel_variance=None, group_outputscales=None, tausq=None, noise=None):
    #     if parsed:
    #         if self.enable_group_outputscale:
    #             if len(parsed) != 3:
    #                 raise ValueError(f"Expected lengthscales, group_outputscales and tausq, only got {len(parsed)} hyperparams")
    #             lengthscales, group_outputscales, tausq = parsed
    #         else:
    #             if len(parsed) == 3:
    #                 lengthscales, kernel_variance, tausq = parsed
    #             elif len(parsed) == 2:
    #                 lengthscales, kernel_variance = parsed
    #             elif len(parsed) == 1:
    #                 (lengthscales,) = parsed
    #             else:
    #                 raise ValueError(f"Unexpected parsed hyperparam number: {len(parsed)}")
    #     if lengthscales is not None:
    #         self.lengthscales = jnp.array(lengthscales, dtype=jnp.float64)
    #     if self.enable_group_outputscale:
    #         if group_outputscales is not None:
    #             g_os = jnp.array(group_outputscales, dtype=jnp.float64)
    #             if g_os.shape != (self.num_groups,):
    #                 raise ValueError(f"Group Outputscales must have shape ({self.num_groups},)")
    #             self.group_outputscales = g_os
    #     else:
    #         if kernel_variance is not None:
    #             self.kernel_variance = jnp.array(kernel_variance, dtype=jnp.float64)
    #     if getattr(self, "tausq_enabled", False) and (tausq is not None):
    #         self.tausq = jnp.array(tausq, dtype=jnp.float64)
        
    #     if noise is not None:
    #         self.noise = noise

    # def _group_sq_dist(self, xa: jnp.ndarray, xb: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    #     """
    #     Squared distance restricted to a subset of dimensions, with per-dim lengthscales
    #     xa: (n1, D), xb: (n2, D), idx: (d_g,)
    #     returns (n1, n2)
    #     """
    #     xa_g = xa[:, idx] / self.lengthscales[idx]
    #     xb_g = xb[:, idx] / self.lengthscales[idx]
    #     return self.sq_dist(xa_g, xb_g)
    
    def covariance(self, xa, xb, include_noise=True):
        """
        K = sum_g w_g * exp(-0.5 * sqdist_g)
        """
        n1, n2 = xa.shape[0], xb.shape[0]
        K_sum = jnp.zeros((n1, n2), dtype=jnp.float64)

        for g, idx in enumerate(self.groups):
            xa_g = xa[:, idx] / self.lengthscales[idx]
            xb_g = xb[:, idx] / self.lengthscales[idx]
            sq = self.sq_dist(xa_g, xb_g)
            Kg = jnp.exp(-0.5 * sq)

            if self.enable_group_outputscale:
                Kg *= self.group_outputscales[g]
            K_sum += Kg 

        if not self.enable_group_outputscale:
            K_sum *= self.kernel_variance
        
        if include_noise:
            K_sum += self.noise * jnp.eye(K_sum.shape[0], dtype=K_sum.dtype)
        
        return K_sum
    
    def diagonal(self, x, include_noise=True):
        """
        Diagonal for additive GP
            each exp(-0.5*0) = 1
            so diag = sum_g w_g (or kernel_variance * sum_g 1) [+ noise]
        """
        n = x.shape[0]
        if self.enable_group_outputscale:
            diag = jnp.sum(self.group_outputscales) 
        else:
            diag = self.kernel_variance * float(self.num_groups)
        
        diag *= jnp.ones((n,), dtype=jnp.float64)
        
        if include_noise:
            diag += self.noise
        return diag

