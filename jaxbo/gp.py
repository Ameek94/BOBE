from math import sqrt,pi
import time
from typing import Any,List
import jax.numpy as jnp
import numpy as np
import jax
from jax import jit
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.nn import softplus
from .utils.core_utils import scale_to_unit, scale_from_unit
jax.config.update("jax_enable_x64", True)
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.util import enable_x64
enable_x64()
from functools import partial
from .utils.logging_utils import get_logger
log = get_logger("gp")
from .optim import optimize_optax, optimize_scipy
from .utils.seed_utils import get_new_jax_key, get_numpy_rng
import numpyro.distributions as dist


safe_noise_floor = 1e-12

sqrt2 = sqrt(2.)
sqrt3 = sqrt(3.)
sqrt5 = sqrt(5.)


def softplus_inv(y, eps=1e-12):
    return jnp.log(jnp.expm1(y) + eps)

class DummyDistribution:
    """A dummy distribution that always returns log_prob = 0.0"""
    def log_prob(self, x):
        return 0.0

def make_distribution(spec: dict) -> dist.Distribution:
    """
    Turn a dictionary specification into a NumPyro distribution.
    
    Example spec:
    {"name": "Normal", "loc": 0.0, "scale": 1.0}
    {"name": "Gamma", "concentration": 2.0, "rate": 1.0}
    {"name": "LogNormal", "loc": 0.0, "scale": 1.0}
    """
    # Ensure distribution exists
    dist_class = getattr(dist, spec["name"], None)
    if dist_class is None:
        raise ValueError(f"Distribution {spec['name']} not found in numpyro.distributions.")
    
    # Remove "name"
    kwargs = {k: v for k, v in spec.items() if k != "name"}
    return dist_class(**kwargs)

def saas_prior_logprob(lengthscales, kernel_variance, tausq):
    """
    Compute SAAS prior log probability.
    
    Arguments
    ---------
    lengthscales : jnp.ndarray
        Lengthscale parameters
    kernel_variance : float
        Kernel variance parameter  
    tausq : float
        SAAS tausq parameter
        
    Returns
    -------
    logprob : float
        Log probability under SAAS priors
    """
    logprior = dist.LogNormal(0., 1.).log_prob(kernel_variance)
    logprior += dist.HalfCauchy(0.1).log_prob(tausq)
    inv_lengthscales_sq = 1 / (tausq * lengthscales**2)
    logprior += jnp.sum(dist.HalfCauchy(1.).log_prob(inv_lengthscales_sq))
    return logprior

def dist_sq(x, y):
    """
    Compute squared Euclidean distance between two points x, y. 
    If x is n1 x d and y is n2 x d returns a n1 x n2 matrix of distancess.
    """
    return jnp.sum(jnp.square(x[:,None,:] - y[None,:,:]),axis=-1) 

@partial(jax.jit, static_argnames='include_noise')
def kernel_diag(x, kernel_variance, noise, include_noise=True):
    """
    Computes only the diagonal of the kernel matrix K(x,x).
    """
    diag = kernel_variance * jnp.ones(x.shape[0]) # The diagonal is just the kernel_variance
    if include_noise:
        diag += noise
    return diag

@partial(jax.jit,static_argnames='include_noise')
def rbf_kernel(xa,xb,lengthscales,kernel_variance,noise,include_noise=True): 
    """
    The RBF kernel
    """
    sq_dist = dist_sq(xa/lengthscales,xb/lengthscales) 
    sq_dist = jnp.exp(-0.5*sq_dist)
    k = kernel_variance*sq_dist
    if include_noise:
        k+= noise*jnp.eye(k.shape[0])
    return k

@partial(jax.jit,static_argnames='include_noise')
def matern_kernel(xa,xb,lengthscales,kernel_variance,noise,include_noise=True):
    """
    The Matern-5/2 kernel
    """
    dsq = dist_sq(xa/lengthscales,xb/lengthscales)
    d = jnp.sqrt(jnp.where(dsq<1e-30,1e-30,dsq))
    exp = jnp.exp(-sqrt5*d)
    poly = 1. + d*(sqrt5 + d*5./3.)
    k = kernel_variance*poly*exp
    if include_noise:
        k+= noise*jnp.eye(k.shape[0])
    return k

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

class GP:
    
    def __init__(self,train_x,train_y,noise=1e-8,kernel="rbf",optimizer="scipy",optimizer_options={},
                 kernel_variance_bounds = [1e-4, 1e8],lengthscale_bounds = [0.01,10],lengthscales=None,kernel_variance=None,
                 kernel_variance_prior=None, lengthscale_prior=None, tausq=None, tausq_bounds=[1e-4,1e4],param_names: List[str] = None):
        """
        Initializes the Gaussian Process model.

        Arguments
        ---------
        train_x : jnp.ndarray
            Training inputs, shape (N, D).
        train_y : jnp.ndarray
            Objective function values at training points, shape (N, 1).
        noise : float, optional
            Noise parameter added to the diagonal of the kernel. Defaults to 1e-8.
        kernel : str, optional
            Kernel to use. Only "rbf" is supported in this implementation. Defaults to "rbf".
        optimizer : str, optional
            Optimizer to use for hyperparameter tuning. Defaults to "scipy".
        optimizer_options : dict, optional
            Keyword arguments for the optimizer. Defaults to {'method': 'L-BFGS-B'}.
        kernel_variance_bounds : list, optional
            Bounds for the kernel variance (in log10 space). Defaults to [-4, 8].
        lengthscale_bounds : list, optional
            Bounds for the lengthscales (in log10 space). Defaults to [log10(0.05), 2].
        lengthscales : jnp.ndarray, optional
            Initial lengthscale values. If None, defaults to ones. Defaults to None.
        kernel_variance : float, optional
            Initial kernel variance. If None, defaults to 1.0. Defaults to None.
        kernel_variance_prior : dict or str, optional
            Specification for the kernel variance prior. 
            If None, defaults to `{'name': 'LogNormal', 'loc': 0.0, 'scale': 1.0}`.
            If 'fixed', the kernel variance will be fixed to the initial value and not optimized.
            Defaults to None.
        lengthscale_prior : str or dict, optional
            Specification for the lengthscale prior. 
            If 'DSLP' or None, uses the DSLP prior. 
            If 'SAAS', uses the SAAS prior with tausq parameter.
            Otherwise, uses the provided distribution spec. Defaults to None.
        tausq : float, optional
            Initial tausq parameter for SAAS prior. Only used when lengthscale_prior='SAAS'. 
            If None, defaults to 1.0. Defaults to None.
        tausq_bounds : list, optional
            Bounds for the tausq parameter (in log10 space). Only used when lengthscale_prior='SAAS'.
            Defaults to [-4, 4].
        """
        # Setup and validate training data
        self._setup_training_data(train_x, train_y)
        # print(f"shapes train_x: {self.train_x.shape}, train_y: {self.train_y.shape}")

        # Setup kernel and initial hyperparameters (RBF only)
        self.kernel_name = "rbf"
        self.kernel = rbf_kernel
        self.lengthscales = lengthscales if lengthscales is not None else jnp.ones(self.ndim)
        self.kernel_variance = kernel_variance if kernel_variance is not None else 1.0
        self.noise = noise
        
        # Compute initial kernel matrices
        K = self.kernel(self.train_x, self.train_x, self.lengthscales, self.kernel_variance, noise=self.noise, include_noise=True)
        self.cholesky = jnp.linalg.cholesky(K)
        self.alphas = cho_solve((self.cholesky, True), self.train_y)

        # Create JIT-compiled prediction methods
        self._update_jit_functions()

        # Setup optimizer
        self.optimizer_method = optimizer
        if optimizer == "scipy":
            self.mll_optimize = optimize_scipy
        else:
            self.mll_optimize = optimize_optax
        self.optimizer_options = optimizer_options
    

        # Store bounds
        self.lengthscale_bounds = lengthscale_bounds
        self.kernel_variance_bounds = kernel_variance_bounds
        # Always store tausq for convenience even though it is only used for SAAS
        self.tausq = tausq if tausq is not None else 1.0
        self.tausq_bounds = tausq_bounds

        # Setup priors and optimization parameters
        self._setup_kernel_variance_prior(kernel_variance_prior)
        self._setup_lengthscale_prior(lengthscale_prior)
        self._setup_optimization_parameters()

        self.param_names = param_names if param_names is not None else [f'x_{i}' for i in range(self.ndim)]


    def _setup_training_data(self, train_x, train_y):
        """Setup and validate training data, compute standardization parameters."""
        # Check x and y sizes
        if train_x.shape[0] != train_y.shape[0]:
            raise ValueError("train_x and train_y must have the same number of points")
        if train_y.ndim != 2:
            train_y = train_y.reshape(-1, 1)
        if train_x.ndim != 2:
            raise ValueError("train_x must be 2D")

        self.ndim = train_x.shape[1]
        
        # Compute standardization parameters
        self.y_mean = jnp.mean(train_y)
        self.y_std = jnp.std(train_y)
        
        # Handle edge case where std is zero (all values identical or only 1 point)
        if self.y_std == 0:
            log.warning("Training targets have zero variance. Setting std to 1.0 to avoid division by zero.")
            self.y_std = 1.0

        # Store standardized training data
        self.train_x = jnp.array(train_x)
        self.train_y = (train_y - self.y_mean) / self.y_std
        log.debug(f"GP training size = {self.train_x.shape[0]}")

    def _setup_kernel_variance_prior(self, kernel_variance_prior):
        """Setup kernel variance prior and determine if it should be fixed."""
        self.kernel_variance_prior_spec = kernel_variance_prior
        if self.kernel_variance_prior_spec is None:
            self.kernel_variance_prior_spec = {'name': 'Uniform', 'low': self.kernel_variance_bounds[0], 'high': self.kernel_variance_bounds[1]}
        
        # Check if kernel variance should be fixed
        self.fixed_kernel_variance = (self.kernel_variance_prior_spec == 'fixed')
        if not self.fixed_kernel_variance:
            self.kernel_variance_prior_dist = make_distribution(self.kernel_variance_prior_spec)
        else:
            # Use dummy distribution that always returns log_prob = 0
            self.kernel_variance_prior_dist = DummyDistribution()

    def _setup_lengthscale_prior(self, lengthscale_prior):
        """Setup lengthscale prior and determine prior function."""
        self.lengthscale_prior_spec = lengthscale_prior
        if self.lengthscale_prior_spec is None:
            self.lengthscale_prior_spec = {'name': 'Uniform', 'low': self.lengthscale_bounds[0], 'high': self.lengthscale_bounds[1]}

        # Set up lengthscale priors and prior function
        if self.lengthscale_prior_spec == 'DSLP':
            self.lengthscale_prior_dist = dist.LogNormal(loc=sqrt2 + 0.5*jnp.log(self.ndim), scale=sqrt3)
            self.prior_func = self._standard_prior_logprob
        elif self.lengthscale_prior_spec == 'SAAS':
            self.lengthscale_prior_dist = None
            self.prior_func = self._saas_prior_logprob  
        else:
            self.lengthscale_prior_dist = make_distribution(self.lengthscale_prior_spec)
            self.prior_func = self._standard_prior_logprob

    def _setup_optimization_parameters(self):
        """Setup parameter names and bounds for optimization."""
        # Build parameter names and bounds based on what's being optimized
        self.param_names = ['lengthscales']
        self.hyperparam_bounds = [self.lengthscale_bounds] * self.ndim
        
        if not self.fixed_kernel_variance:
            self.param_names.append('kernel_variance')
            self.hyperparam_bounds.append(self.kernel_variance_bounds)
            
        if self.lengthscale_prior_spec == 'SAAS':
            self.param_names.append('tausq')
            self.hyperparam_bounds.append(self.tausq_bounds)

        self.hyperparam_bounds = jnp.array(self.hyperparam_bounds).T
        self.num_hyperparams = self.hyperparam_bounds.shape[1]
        log.debug(f" Hyperparameter bounds =  {self.hyperparam_bounds}")

    def _predict_mean_single(self, x):
        """Implementation of single point mean prediction for RBF kernel."""
        x = jnp.atleast_2d(x)
        # Optimized RBF kernel computation
        inv_ls = 1.0 / self.lengthscales
        train_x_scaled = self.train_x * inv_ls
        x_scaled = x * inv_ls
        
        D = (
            jnp.sum(train_x_scaled ** 2, axis=1)[:, None]
            + jnp.sum(x_scaled ** 2, axis=1)[None, :]
            - 2 * train_x_scaled @ x_scaled.T
        )
        k12 = self.kernel_variance * jnp.exp(-0.5 * D)  # (n_train, n_test)
        mean_std = jnp.dot(k12.T, self.alphas).squeeze(-1)
        mean = mean_std * self.y_std + self.y_mean
        return mean.squeeze()

    def _predict_var_single(self, x):
        """Implementation of single point variance prediction for RBF kernel."""
        x = jnp.atleast_2d(x)
        # Optimized RBF kernel computation
        inv_ls = 1.0 / self.lengthscales
        train_x_scaled = self.train_x * inv_ls
        x_scaled = x * inv_ls
        
        D = (
            jnp.sum(train_x_scaled ** 2, axis=1)[:, None]
            + jnp.sum(x_scaled ** 2, axis=1)[None, :]
            - 2 * train_x_scaled @ x_scaled.T
        )
        k12 = self.kernel_variance * jnp.exp(-0.5 * D)  # (n_train, n_test)
        
        V = solve_triangular(self.cholesky, k12, lower=True)
        k22 = self.kernel_variance * jnp.ones(x.shape[0])
        var_std = k22 - jnp.sum(V ** 2, axis=0)
        var_std = jnp.maximum(var_std, safe_noise_floor)
        
        var = var_std * (self.y_std ** 2)
        return var.squeeze()

    def _predict_single(self, x):
        """Implementation of single point prediction (mean and variance) - returns standardized values."""
        x = jnp.atleast_2d(x)
        # Optimized RBF kernel computation
        inv_ls = 1.0 / self.lengthscales
        train_x_scaled = self.train_x * inv_ls
        x_scaled = x * inv_ls
        
        D = (
            jnp.sum(train_x_scaled ** 2, axis=1)[:, None]
            + jnp.sum(x_scaled ** 2, axis=1)[None, :]
            - 2 * train_x_scaled @ x_scaled.T
        )
        k12 = self.kernel_variance * jnp.exp(-0.5 * D)  # (n_train, n_test)
        
        mean_std = jnp.dot(k12.T, self.alphas).squeeze(-1)  # Keep standardized for EI
        V = solve_triangular(self.cholesky, k12, lower=True)
        k22 = self.kernel_variance * jnp.ones(x.shape[0])
        var_std = k22 - jnp.sum(V ** 2, axis=0)
        var_std = jnp.maximum(var_std, safe_noise_floor)
        
        return mean_std.squeeze(), var_std.squeeze()

    def _update_jit_functions(self):
        """Create JIT-compiled prediction functions with current GP state."""
        # Create JIT-compiled versions of the implementation functions
        self.predict_mean_single = jax.jit(self._predict_mean_single)
        self.predict_var_single = jax.jit(self._predict_var_single)
        self.predict_single = jax.jit(self._predict_single)
        
        # Create batched versions using vmap
        self.predict_mean_batched = jax.jit(jax.vmap(self._predict_mean_single, in_axes=0))
        self.predict_var_batched = jax.jit(jax.vmap(self._predict_var_single, in_axes=0))
        self.predict_batched = jax.jit(jax.vmap(self._predict_single, in_axes=0))

        # Warm up JIT compilation with a dummy point
        dummy_x = jnp.zeros(self.ndim)
        dummy_batch = jnp.zeros((2, self.ndim))
        
        _ = self.predict_mean_single(dummy_x)
        _ = self.predict_var_single(dummy_x)
        _ = self.predict_single(dummy_x)
        _ = self.predict_mean_batched(dummy_batch)
        _ = self.predict_var_batched(dummy_batch)
        _ = self.predict_batched(dummy_batch)
        
        log.debug("JIT-compiled prediction functions updated and warmed up")

    def _standard_prior_logprob(self, lengthscales, kernel_variance, tausq=None):
        """Standard prior log probability for DSLP and custom priors."""
        logprior = self.kernel_variance_prior_dist.log_prob(kernel_variance)
        if self.lengthscale_prior_dist is not None:
            logprior += self.lengthscale_prior_dist.log_prob(lengthscales).sum()
        return logprior
    
    def _saas_prior_logprob(self, lengthscales, kernel_variance, tausq):
        """SAAS prior log probability."""
        return saas_prior_logprob(lengthscales, kernel_variance, tausq)
    
    def _parse_hyperparams(self, unconstrained_params):
        """
        Parse unconstrained parameters into lengthscales, kernel_variance, and optionally tausq.
        Using the softplus transform to enforce positivity.
        """
        hyperparams = softplus(unconstrained_params)

        lengthscales = hyperparams[:self.ndim]

        if self.fixed_kernel_variance:
            kernel_variance = self.kernel_variance
            if 'tausq' in self.param_names:
                tausq = hyperparams[self.ndim] if len(hyperparams) > self.ndim else self.tausq
            else:
                tausq = self.tausq
        else:
            kernel_variance = hyperparams[self.ndim]
            tausq = hyperparams[self.ndim + 1] if len(hyperparams) > self.ndim + 1 else self.tausq

        return lengthscales, kernel_variance, tausq

    def neg_mll(self, unconstrained_params):
        """
        Computes the negative log marginal likelihood for the GP with given hyperparameters.
        Parameters are in unconstrained space; we transform them with softplus.
        """
        lengthscales, kernel_variance, tausq = self._parse_hyperparams(unconstrained_params)

        K = self.kernel(
            self.train_x, self.train_x, lengthscales, kernel_variance,
            noise=self.noise, include_noise=True
        )
        mll = gp_mll(K, self.train_y, self.train_y.shape[0])

        # Add prior in original space
        mll += self.prior_func(lengthscales, kernel_variance, tausq)

        return -mll

    def fit(self, maxiter=1000, n_restarts=4, rng=None):
        """ 
        Fits the GP using maximum likelihood hyperparameters with the chosen optimizer.
        """
        rng = rng if rng is not None else get_numpy_rng()

        # Prepare initial parameters from current hyperparameters
        init_params = jnp.array(self.lengthscales)
        if not self.fixed_kernel_variance:
            init_params = jnp.concatenate([init_params, jnp.array([self.kernel_variance])])
        if 'tausq' in self.param_names:
            init_params = jnp.concatenate([init_params, jnp.array([self.tausq])])

        # Logging initial MLL
        lengthscales_str = {n: f"{float(v):.4f}" for n, v in zip(self.param_names, self.lengthscales.tolist())}
        msg = f"Fitting GP: lengthscales={lengthscales_str}, kernel_variance={self.kernel_variance:.4f}"
        if self.fixed_kernel_variance:
            msg += " (fixed kernel_variance)"
        if 'tausq' in self.param_names:
            msg += f", tausq={self.tausq:.4f}"
        current_mll = -self.neg_mll(softplus_inv(init_params))
        msg += f", current MLL={current_mll:.4f}"
        log.info(msg)

        # Warm start in unconstrained space
        init_params_u = softplus_inv(init_params)

        # Add random restarts directly in unconstrained space
        if n_restarts > 1:
            addn_init_params_u = rng.normal(size=(n_restarts-1, init_params.shape[0]))
            init_params_u = np.vstack([init_params_u, addn_init_params_u])
        x0 = init_params_u

        optimizer_options = self.optimizer_options.copy()

        best_params, best_f = self.mll_optimize(
            fun=self.neg_mll,
            num_params=self.num_hyperparams,
            bounds=None,  # no need for bounds in unconstrained space
            x0=x0,
            maxiter=maxiter,
            n_restarts=n_restarts,
            optimizer_options=optimizer_options
        )

        # Transform to +ve constrained space
        lengthscales, kernel_variance, tausq = self._parse_hyperparams(best_params)

        # Final hard clipping to bounds and update GP
        lengthscales = jnp.clip(lengthscales, min=self.hyperparam_bounds[0,:self.ndim], max=self.hyperparam_bounds[1,:self.ndim])
        self.lengthscales = lengthscales
        if not self.fixed_kernel_variance:
            kernel_variance = jnp.clip(kernel_variance, min=self.hyperparam_bounds[0,self.ndim], max=self.hyperparam_bounds[1,self.ndim])
            self.kernel_variance = kernel_variance
        if 'tausq' in self.param_names:
            tausq = jnp.clip(tausq, min=self.hyperparam_bounds[0,self.ndim + 1], max=self.hyperparam_bounds[1,self.ndim + 1])
            self.tausq = tausq

        # Final log
        lengthscales_str = {n: f"{float(v):.4f}" for n, v in zip(self.param_names, self.lengthscales.tolist())}
        msg = f"Final hyperparams: lengthscales={lengthscales_str}, kernel_variance={self.kernel_variance:.4f}"
        if self.fixed_kernel_variance:
            msg += " (fixed kernel_variance)"
        if 'tausq' in self.param_names:
            msg += f", tausq={self.tausq:.4f}"
        msg += f", final MLL={-best_f:.4f}"
        log.info(msg)


        self.recompute_cholesky_alphas()

    def update(self,new_x,new_y,refit=True,maxiter=1000,n_restarts=4):
        """
        Updates the GP with new training points and refits the GP if refit is True.

        Arguments
        ---------        
        refit: bool
            Whether to refit the GP hyperparameters. Default is True.
        maxiter: int
            The maximum number of iterations for the optimizer. Default is 1000.
        n_restarts: int
            The number of restarts for the optax optimizer. Default is 4.
        """
        new_x = jnp.atleast_2d(new_x)
        new_y = jnp.atleast_2d(new_y).reshape(-1, 1)  # Ensure (n, 1) shape

        new_pts_to_add = []
        new_vals_to_add = []
        
        # Check for duplicates and collect new points
        for i in range(new_x.shape[0]):
            if jnp.any(jnp.all(jnp.isclose(self.train_x, new_x[i], atol=1e-6, rtol=1e-4), axis=1)):
                log.debug(f"Point {new_x[i]} already exists in the training set, not updating")
            else:
                new_pts_to_add.append(new_x[i])
                new_vals_to_add.append(new_y[i])

        # Add new points if any
        if new_pts_to_add:
            new_pts_to_add = jnp.array(new_pts_to_add)
            new_vals_to_add = jnp.array(new_vals_to_add).reshape(-1, 1)  # Ensure proper shape
            
            # Add to training data
            self.train_x = jnp.vstack([self.train_x, new_pts_to_add])
            train_y_original = jnp.vstack([self.train_y * self.y_std + self.y_mean, new_vals_to_add])
            
            # Recompute standardization parameters
            self.y_mean = jnp.mean(train_y_original)
            self.y_std = jnp.std(train_y_original)
            
            if self.y_std == 0:
                log.warning("Training targets have zero variance. Setting std to 1.0 to avoid division by zero.")
                self.y_std = 1.0
            
            self.train_y = (train_y_original - self.y_mean) / self.y_std
        
        if refit:
            self.fit(maxiter=maxiter,n_restarts=n_restarts)
        else:
            self.recompute_cholesky_alphas()

        # Update JIT-compiled functions with new state
        self._update_jit_functions()


    def recompute_cholesky_alphas(self):
        """
        Recomputes the Cholesky decomposition and alphas. Useful if hyperparameters are changed manually.
        Also updates JIT-compiled prediction functions.
        """
        K = self.kernel(self.train_x, self.train_x, self.lengthscales, self.kernel_variance, noise=self.noise, include_noise=True)
        self.cholesky = jnp.linalg.cholesky(K)
        self.alphas = cho_solve((self.cholesky, True), self.train_y)
        
    def fantasy_var(self,new_x,mc_points,k_train_mc):
        """
        Fast fantasy variance update: computes posterior variance at mc_points 
        after adding new_x to the training set, using Cholesky rank-1 update.

        Args:
            new_x (jnp.ndarray): New point, shape (d,)
            mc_points (jnp.ndarray): Points to evaluate variance at, shape (M, d)
            k_train_mc (jnp.ndarray): Precomputed k(train_x, mc_points), shape (N, M)

        Returns:
            var (jnp.ndarray): Posterior variances at mc_points, shape (M,), scaled
        """
        # Assume new_x is (d,) — avoid reshaping
        new_x_2d = jnp.expand_dims(new_x, 0)  # (1, d)

        # k(train_x, new_x): (N,)
        k = self.kernel(
            self.train_x, new_x_2d,
            self.lengthscales, self.kernel_variance,
            noise=self.noise, include_noise=False
        ).squeeze()  # (N,)

        # k(new_x, new_x) + noise
        k_self = self.kernel_variance + self.noise

        # Fast Cholesky update: [L   0]
        #                      [v^T  d]
        L_fantasy = fast_update_cholesky(self.cholesky, k, k_self)  # (N+1, N+1)

        # k(new_x, mc_points): (1, M)
        k_new_mc = self.kernel(
            new_x_2d, mc_points,
            self.lengthscales, self.kernel_variance,
            noise=self.noise, include_noise=False
        )  # (1, M)

        # Build [k_train_mc]
        #       [k_new_mc ]  -> (N+1, M) without copying
        k12 = jnp.concatenate([k_train_mc, k_new_mc], axis=0)  # (N+1, M)

        # Solve: L_fantasy @ V = k12  → V = L_fantasy^{-1} @ k12
        V = solve_triangular(L_fantasy, k12, lower=True)  # (N+1, M)

        # Variance: k(x,x) - ||V||^2
        k22 = kernel_diag(mc_points, self.kernel_variance, self.noise, include_noise=True)  # (M,)
        var_std = k22 - jnp.sum(V ** 2, axis=0)  # (M,)
        var_std = jnp.maximum(var_std, safe_noise_floor)

        return var_std * (self.y_std ** 2)  # scale back to original

    def get_random_point(self,rng=None,nstd=None):
        """
        Returns a random point in the unit cube.
        """
        rng = rng if rng is not None else get_numpy_rng()
        pt = rng.uniform(0, 1, size=self.train_x.shape[1])
        return pt

    def sample_GP_NUTS(self,warmup_steps=256,num_samples=512,thinning=8,
                       temp=1.,num_chains=4, np_rng=None, rng_key=None):

        """
        Obtain samples from the posterior represented by the GP mean as the logprob.
        Optionally restarts MCMC if all logp values are the same or if HMC fails.
        """        

        rng_mcmc = np_rng if np_rng is not None else get_numpy_rng()
        prob = rng_mcmc.uniform(0, 1)
        high_temp = rng_mcmc.uniform(1., 2.) ** 2
        temp = np.where(prob < 1/3, 1., high_temp) # Randomly choose temperature either 1 or high_temp
        log.info(f"Running MCMC chains with temperature {temp:.4f}")

        def model():
            x = numpyro.sample('x', dist.Uniform(
                low=jnp.zeros(self.train_x.shape[1]),
                high=jnp.ones(self.train_x.shape[1])
            ))

            mean = self.predict_mean_single(x)
            numpyro.factor('y', mean/temp)
            numpyro.deterministic('logp', mean)

        @jax.jit
        def run_single_chain(rng_key):
                kernel = NUTS(model, dense_mass=False, max_tree_depth=5,)
                mcmc = MCMC(kernel, num_warmup=warmup_steps, num_samples=num_samples,
                        num_chains=1, progress_bar=False, thinning=thinning)
                mcmc.run(rng_key)
                samples_x = mcmc.get_samples()['x']
                logps = mcmc.get_samples()['logp']
                return samples_x,logps


        num_devices = jax.device_count()
        
        rng_key = rng_key if rng_key is not None else get_new_jax_key()
        rng_keys = jax.random.split(rng_key, num_chains)

        log.info(f"Running MCMC with {num_chains} chains on {num_devices} devices.")

        # Adaptive method selection based on device/chain configuration
        if num_devices == 1:
            # Sequential method for single device
            log.info("Using sequential method (single device)")
            samples_x = []
            logps = []
            for i in range(num_chains):
                samples_x_i, logps_i = run_single_chain(rng_keys[i])
                samples_x.append(samples_x_i)
                logps.append(logps_i)
            samples_x = jnp.concatenate(samples_x)
            logps = jnp.concatenate(logps)
            
        elif num_devices >= num_chains and num_chains > 1:
            # Direct pmap method when devices >= chains
            log.info("Using direct pmap method (devices >= chains)")
            pmapped = jax.pmap(run_single_chain, in_axes=(0,),out_axes=(0,0))
            samples_x, logps = pmapped(rng_keys)
            # reshape to get proper shapes
            samples_x = jnp.concatenate(samples_x, axis=0)
            logps = jnp.reshape(logps, (samples_x.shape[0],))
            
        elif 1 < num_devices < num_chains:
            # Chunked method when devices < chains (but > 1 device)
            log.info(f"Using chunked pmap method ({num_devices} devices < {num_chains} chains)")
            
            # Process chains in chunks of device count using the existing run_single_chain
            pmapped_chunked = jax.pmap(run_single_chain, in_axes=(0,), out_axes=(0, 0))
            
            all_samples = []
            all_logps = []
            
            for i in range(0, num_chains, num_devices):
                end_idx = min(i + num_devices, num_chains)
                chunk_keys = rng_keys[i:end_idx]
                
                # Run chunk (pmap handles variable chunk sizes automatically)
                chunk_samples, chunk_logps = pmapped_chunked(chunk_keys)
                
                all_samples.append(chunk_samples)
                all_logps.append(chunk_logps)
            
            # Concatenate all chunks
            samples_x = jnp.concatenate([jnp.concatenate(chunk, axis=0) for chunk in all_samples], axis=0)
            logps = jnp.concatenate([jnp.concatenate(chunk, axis=0) for chunk in all_logps], axis=0)
            
        else:
            # Fallback to sequential (single chain case)
            log.info("Using sequential method (fallback)")
            samples_x = []
            logps = []
            for i in range(num_chains):
                samples_x_i, logps_i = run_single_chain(rng_keys[i])
                samples_x.append(samples_x_i)
                logps.append(logps_i)
            samples_x = jnp.concatenate(samples_x)
            logps = jnp.concatenate(logps)

        samples_dict = {
            'x': samples_x,
            'logp': logps,
            'best': samples_x[jnp.argmax(logps)],
            'method': "MCMC"
        }

        return samples_dict

    def state_dict(self):
        """
        Returns a dictionary containing the complete state of the GP.
        This can be used for saving, loading, or copying the GP.
        
        Returns
        -------
        state: dict
            Dictionary containing all necessary information to reconstruct the GP
        """
        state = {
            # Training data (original, unstandardized)
            'train_x': np.array(self.train_x),
            'train_y': np.array(self.train_y * self.y_std + self.y_mean),  # unstandardize
            
            # Hyperparameters
            'lengthscales': np.array(self.lengthscales),
            'kernel_variance': float(self.kernel_variance),
            'noise': float(self.noise),
            'tausq': float(self.tausq),
            
            # Standardization parameters
            'y_mean': float(self.y_mean),
            'y_std': float(self.y_std),
            
            # Model configuration
            'kernel_name': self.kernel_name,
            'lengthscale_prior_spec': self.lengthscale_prior_spec,
            'kernel_variance_prior_spec': self.kernel_variance_prior_spec,
            'fixed_kernel_variance': self.fixed_kernel_variance,
            'optimizer_method': self.optimizer_method,
            'optimizer_options': self.optimizer_options,
            
            # Bounds
            'lengthscale_bounds': self.lengthscale_bounds,
            'kernel_variance_bounds': self.kernel_variance_bounds,
            'tausq_bounds': self.tausq_bounds,
            
            # Computed state
            'cholesky': np.array(self.cholesky) if hasattr(self, 'cholesky') else None,
            'alphas': np.array(self.alphas) if hasattr(self, 'alphas') else None,
            
            # Dimensions
            'ndim': self.ndim,
            'param_names': self.param_names,
            
            # Class identifier
            'gp_class': 'GP'
        }
        
        return state
    
    @classmethod
    def from_state_dict(cls, state):
        """
        Creates a GP instance from a state dictionary.
        
        Arguments
        ---------
        state: dict
            State dictionary returned by state_dict()
            
        Returns
        -------
        gp: GP
            The reconstructed GP object
        """
        # Create GP instance
        gp = cls(
            train_x=state['train_x'],
            train_y=state['train_y'],
            noise=state['noise'],
            kernel=state['kernel_name'],
            optimizer=state['optimizer_method'],
            optimizer_options=state['optimizer_options'],
            lengthscales=state['lengthscales'],
            kernel_variance=state['kernel_variance'],
            lengthscale_bounds=state['lengthscale_bounds'],
            kernel_variance_bounds=state['kernel_variance_bounds'],
            kernel_variance_prior=state.get('kernel_variance_prior_spec'),
            lengthscale_prior=state.get('lengthscale_prior_spec'),
            tausq=state.get('tausq', 1.0),
            tausq_bounds=state.get('tausq_bounds', [-4, 4]),
            param_names=state.get('param_names', None)
        )
        
        # Restore computed state if available
        if state['cholesky'] is not None:
            gp.cholesky = jnp.array(state['cholesky'])
        if state['alphas'] is not None:
            gp.alphas = jnp.array(state['alphas'])
        
        return gp
    
    @classmethod
    def load(cls, filename, **kwargs):
        """
        Loads a GP from a file
        
        Arguments
        ---------
        filename: str
            The name of the file to load the GP from (with or without .npz extension)
        **kwargs: 
            Additional keyword arguments to pass to the GP constructor
            
        Returns
        -------
        gp: GP
            The loaded GP object
        """
        if not filename.endswith('.npz'):
            filename += '.npz'
            
        try:
            data = np.load(filename, allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file {filename}")
        
        # Convert arrays back to the expected format
        state = {}
        for key in data.files:
            value = data[key]
            if isinstance(value, np.ndarray) and value.shape == ():
                # Handle scalar arrays
                state[key] = value.item()
            else:
                state[key] = value
        
        # Apply any override kwargs
        state.update(kwargs)
        
        # Use from_state_dict for loading
        gp = cls.from_state_dict(state)
        
        log.info(f"Loaded GP from {filename} with {gp.train_x.shape[0]} training points")
        return gp

    def save(self, filename='gp'):
        """
        Save the GP state to a file using state_dict.
        
        Arguments
        ---------
        filename: str
            The filename to save to (with or without .npz extension). Default is 'gp'.
        """
        if not filename.endswith('.npz'):
            filename += '.npz'
        
        state = self.state_dict()
        np.savez(filename, **state)
        log.info(f"Saved GP state to {filename}")


    def copy(self):
        """
        Creates a deep copy of the GP using state_dict.
        
        Returns
        -------
        gp_copy: GP
            A deep copy of the current GP
        """
        state = self.state_dict()
        return self.__class__.from_state_dict(state)
    
    @property
    def npoints(self):
        return self.train_x.shape[0]
