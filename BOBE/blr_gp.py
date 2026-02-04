from math import sqrt,pi
from typing import Any,List, Dict, Optional
import jax.numpy as jnp
import numpy as np
import jax
from jax.scipy.linalg import cho_solve, solve_triangular
jax.config.update("jax_enable_x64", True)
from functools import partial
from .utils.log import get_logger
log = get_logger("gp")
from .optim import optimize_optax, optimize_scipy
from .utils.seed import get_new_jax_key, get_numpy_rng
import numpyro.distributions as dist
from .kernels import SphericalLinearKernel
from .gp import DummyDistribution, make_distribution, GP

safe_noise_floor = 1e-12

# Constants for DSLP prior
sqrt2 = sqrt(2.)
sqrt3 = sqrt(3.)



class GPwithBLR(GP):
    """
    Feature-space Bayesian linear regression GP for finite feature kernels.
    Speciifcally intended for SphericalLinearKernel with kernel._features(x)
    """

    def __init__(
            self,
            train_x,
            train_y,
            noise: float = 1e-6,
            kernel: str = "spherical_linear",
            optimizer: str = "scipy",
            optimizer_options: Dict[str, Any] = {},
            lengthscale_bounds: List[float] = [1.0, 5],
            lengthscales=None,
            lengthscale_prior=None,
            b_bounds: List[float] = [-50, 50],
            a_prior=None,
            param_names: Optional[List[str]] = None,
    ):
        

        # Setup training data
        self._setup_training_data(train_x, train_y)
        self.param_names = param_names if param_names is not None else ['x_'+str(i) for i in range(self.ndim)]

        # Store config
        self.kernel_name = kernel
        if self.kernel_name != "spherical_linear":
            raise ValueError("GPwithBLR currently only support spherical linear kernel")

        # Store Bounds
        self.lengthscale_bounds = lengthscale_bounds
        self.b_bounds = b_bounds

        # Hyperparameter Priors
        self.a_prior = a_prior
        self.lengthscale_prior = lengthscale_prior

        # Fixed Flags
        self.fixed_a = (a_prior == "fixed")

        # Initial hyperparameters
        self.lengthscales = lengthscales if lengthscales is not None else jnp.ones(self.ndim)
        self.kernel_variance = 1.0
        self.noise = noise

        # Defaults for spherical kernel
        self.a = 2.0*jnp.sqrt(self.ndim)
        self.a_bounds = [self.a*1e-3, self.a*1e3]
        self.b_logits = 1.

        self._setup_priors(self.lengthscale_prior)
        

        # Setup kernel
        self.kernel = SphericalLinearKernel(self.lengthscales, self.noise, self.a, self.b_logits)

        # Setup optimizer
        self.optimizer_method = optimizer
        if optimizer == "scipy":
            self.mll_optimize = optimize_scipy
        else:
            self.mll_optimize = optimize_optax
        self.optimizer_options = optimizer_options

        self._setup_optimization_parameters()

        # Feature space caches
        self.Psi = None
        self.PsiTPsi = None
        self.PsiTy = None
        self.yy = None

        self.A_chol = None
        self.mu_w = None

        # Build initial caches
        self._rebuild_feature_cache()
        self._rebuild_posterior_cache()
    


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
        
        # Compute standardization parameters (and handle the case of 0 initialisation points)
        self.y_mean = jnp.mean(train_y) if train_y.size > 0 else 0 
        self.y_std = jnp.std(train_y) if train_y.size > 0 else 1.0
        
        # Handle edge case where std is zero (all values identical or only 1 point)
        if self.y_std == 0:
            log.warning("Training targets have zero variance. Setting std to 1.0 to avoid division by zero.")
            self.y_std = 1.0

        # Store standardized training data
        self.train_x = jnp.array(train_x)
        self.train_y = (train_y - self.y_mean) / self.y_std
        log.debug(f"GP training size = {self.train_x.shape[0]}")

    def _setup_optimization_parameters(self):
        """Setup parameter names and bounds for optimization."""
        # Build parameter names and bounds based on what's being optimized
        self.hyperparam_names = ['lengthscales']
        self.hyperparam_bounds = [self.lengthscale_bounds] * self.ndim
        
            

        if not self.fixed_a:
            self.hyperparam_names.append("a")
            self.hyperparam_bounds.append(self.a_bounds)


        self.hyperparam_bounds = jnp.log(jnp.array(self.hyperparam_bounds).T)


        self.hyperparam_names.append('b_logits')
        self.hyperparam_bounds = jnp.concatenate([self.hyperparam_bounds, jnp.array([self.b_bounds]).T], axis=1)

        self.num_hyperparams = self.hyperparam_bounds.shape[1]
        log.debug(f" Hyperparameter bounds =  {self.hyperparam_bounds}")


    def get_hyperparams(self):
        hp = self.lengthscales

        if not self.fixed_a:
             hp = jnp.hstack([hp, self.a])
        

        hp = jnp.hstack([hp, self.kernel.b_logits])

        return hp
    
    def _parse_hyperparams(self, log_params):
        """Parse log parameters."""
        idx = 0

        lengthscales = jnp.exp(log_params[idx:idx + self.ndim])
        idx += self.ndim
        
        if self.fixed_a:
            a = self.a
        else:
            a = jnp.exp(log_params[idx])
            idx += 1
        
        b_logits = log_params[idx]

        return lengthscales,  b_logits, a 
    

    def _setup_priors(self, lengthscale_prior):
        self.lengthscale_prior_spec = lengthscale_prior
        if self.lengthscale_prior_spec is None:
            # Default to DSLP Prior
            self.lengthscale_prior_dist = dist.LogNormal(loc=sqrt2 + 0.5*jnp.log(self.ndim), scale=sqrt3)
        else:
            # For now do the same thing in either case
            self.lengthscale_prior_dist = dist.LogNormal(loc=sqrt2 + 0.5*jnp.log(self.ndim), scale=sqrt3)
            
        
        if not self.fixed_a:
            if self.a_prior is None:
                self.global_lengthscale_prior_dist = dist.LogNormal(loc=np.log(2*jnp.sqrt(self.ndim)), scale=0.5)
            else:
                self.global_lengthscale_prior_dist = make_distribution(self.a_prior)
        else:
            self.global_lengthscale_prior_dist = DummyDistribution()



    def _prior_logprob(self, lengthscales, a):
        # DSLP Lengthscale Prior
        logprior = self.lengthscale_prior_dist.log_prob(lengthscales).sum()
        # Kernel Variance Prior
    
        if not self.fixed_a:
            logprior += self.global_lengthscale_prior_dist.log_prob(a)

        return logprior
    
    def _noise_std(self):
        return jnp.maximum(self.noise / (self.y_std**2), safe_noise_floor)
    
    def _rebuild_feature_cache(self):
        Psi = self.kernel._features(self.train_x)  # (N, M)
        y = self.train_y.reshape((-1,))

        self.Psi = Psi
        self.PsiTPsi = Psi.T @ Psi
        self.PsiTy = Psi.T @ y
        self.yy = y @ y

    def _rebuild_posterior_cache(self):
        """
        BLR posterior caches
        """
        if self.PsiTPsi is None:
            self._rebuild_feature_cache()

        sigma2 = self._noise_std()
        jitter = safe_noise_floor
        M = self.PsiTPsi.shape[0]

        A = jnp.eye(M, dtype=self.PsiTPsi.dtype) + (self.PsiTPsi / sigma2)
        A += jitter * jnp.eye(M,  dtype=self.PsiTPsi.dtype)

        L = jnp.linalg.cholesky(A)

        alpha = cho_solve((L, True), self.PsiTy)
        mu_w = alpha / sigma2

        self.A_chol = L
        self.mu_w = mu_w
    
    


    def mll_feature_space(self):
        """
        Feature space log marginal likelihood analogue
        """

        if self.Psi is None:
            self._rebuild_feature_cache()

        sigma2 = self._noise_std()
        jitter = safe_noise_floor

        N = self.train_y.shape[0]
        M = self.PsiTPsi.shape[0]


        A = jnp.eye(M,  dtype=self.PsiTPsi.dtype) + (self.PsiTPsi / sigma2)
        A += jitter * jnp.eye(M, dtype=self.PsiTPsi.dtype)

        L = jnp.linalg.cholesky(A)

        logdetA = 2 * jnp.sum(jnp.log(jnp.diag(L)))

        alpha = cho_solve((L, True), self.PsiTy)

        quad = (self.yy / sigma2) - (self.PsiTy @ alpha) / (sigma2**2)

        mll = -0.5 * (
            N * jnp.log(2.0 * jnp.pi) 
            + N * jnp.log(sigma2) 
            + logdetA 
            + quad
        )
        return mll

    def neg_mll(self, log_params):

        lengthscales, b_logits, a = self._parse_hyperparams(log_params)

        self.kernel.update_hyperparams(
            lengthscales=lengthscales,
            b_logits=b_logits,
            a=a
        ) 

        self._rebuild_feature_cache()

        mll = self.mll_feature_space()
        logprior = self._prior_logprob(lengthscales, a)

        return - (mll + logprior)
    
    def fit(self, x0: np.ndarray = None, maxiter: int = 500) -> dict:
        """
        Performs a serial fit for a given batch of starting points (x0).
        x0 must be in optimiser space:
        [log ls..., (option) log kv, (optional) log a, b_logits]
        """
        if x0 is None:
            parts = [jnp.log(self.lengthscales)]
            if not self.fixed_a:
                parts.append(jnp.log(self.a))
            parts.append(self.b_logits)
            x0 = jnp.concatenate([p.reshape(-1,) for p in parts], axis=0)[None, :]

        optimizer_options = self.optimizer_options.copy()

        best_params_log, best_loss = self.mll_optimize(
            fun=self.neg_mll,
            num_params=self.num_hyperparams,
            bounds=self.hyperparam_bounds,
            x0=x0,
            maxiter=maxiter,
            n_restarts=x0.shape[0],
            optimizer_options=optimizer_options,
        )

        log.info(f"Best MLL after fit: {-best_loss}")

        lengthscales, b_logits, a = self._parse_hyperparams(best_params_log)

        self.lengthscales = lengthscales
        self.b_logits = b_logits
        self.a = a

        self.kernel.update_hyperparams(
            lengthscales=self.lengthscales,
            b_logits=self.b_logits,
            a=self.a,
        )

        self._rebuild_feature_cache()
        self._rebuild_posterior_cache()

        return {"mll": -best_loss, "params": best_params_log}
    
    def update_hyperparams(self, hyperparams):
        lengthscales, b_logits, a = self._parse_hyperparams(hyperparams)

        self.lengthscales = lengthscales
        self.a = a
        self.b_logits = b_logits

        self.kernel.update_hyperparams(
            lengthscales=self.lengthscales,
            b_logits=self.b_logits,
            a=self.a,
        )

        self._rebuild_feature_cache()
        self._rebuild_posterior_cache()


    def predict_mean_single(self, x):
        x = jnp.atleast_2d(x)
        Psi = self.kernel._features(x)
        mean_std = (Psi @ self.mu_w).reshape(())
        return mean_std * self.y_std + self.y_mean
    
    def predict_mean_batched(self, x):
        x = jnp.atleast_2d(x)
        return jax.vmap(self.predict_mean_single, in_axes=0)(x)
    
    def predict_var_single(self, x, include_noise=True):
        x = jnp.atleast_2d(x)
        Psi = self.kernel._features(x)

        v = solve_triangular(self.A_chol, Psi.T, lower=True)
        var_std = jnp.sum(v * v, axis=0).squeeze()

        var_std = jnp.where(jnp.isnan(var_std), safe_noise_floor, var_std)
        var_std = jnp.where(var_std < safe_noise_floor, safe_noise_floor, var_std)

        if include_noise:
            var_std += self._noise_std()

        return (self.y_std**2) * var_std
    
    def predict_var_batched(self, x):
        x = jnp.atleast_2d(x)
        return jax.vmap(self.predict_var_single, in_axes=0)(x)
    

    def predict_single(self, x, include_noise=True):
        x = jnp.atleast_2d(x)
        Psi = self.kernel._features(x)

        mean = (Psi @ self.mu_w).reshape(())
        v = solve_triangular(self.A_chol, Psi.T, lower=True)
        var = jnp.sum(v * v, axis=0).squeeze()

        var = jnp.where(jnp.isnan(var), safe_noise_floor, var)
        var = jnp.where(var < safe_noise_floor, safe_noise_floor, var)

        if include_noise:
            var += self._noise_std()

        return mean, var
    
    def predict_batched(self, x):
        x = jnp.atleast_2d(x)
        return jax.vmap(self.predict_single, in_axes=0, out_axes=(0, 0))(x)


    def hyperparams_dict(self):
        ls_str = {name: f"{float(val):.4f}" for name, val in zip(self.param_names, self.lengthscales)}
        param_dict = {
            'lengthscales': ls_str,
        }
        if 'a' in self.hyperparam_names:
            param_dict['a'] = f"{float(self.a):.4f}"
        if 'b_logits' in self.hyperparam_names:
            param_dict['b_logits'] = f"{float(self.b_logits):.4f}"

        return param_dict
    
    def state_dict(self):

        state = {
            # Training data (original, unstandardized)
            'train_x': np.array(self.train_x),
            'train_y': np.array(self.train_y * self.y_std + self.y_mean),  # unstandardize


            # Hyperparameters
            'lengthscales': np.array(self.lengthscales),
            'noise': float(self.noise),
            
            "a": float(self.a),
            "b_logits": float(self.b_logits),

             # Standardization parameters
            'y_mean': float(self.y_mean),
            'y_std': float(self.y_std),
            
            # Model configuration
            'kernel_name': self.kernel_name,
            "fixed_a": self.fixed_a,

            'lengthscale_prior_spec': self.lengthscale_prior_spec,
            "a_prior_spec": self.a_prior,
            
            'optimizer_method': self.optimizer_method,
            'optimizer_options': self.optimizer_options,

            # Bounds
            'lengthscale_bounds': self.lengthscale_bounds,
            "a_bounds": self.a_bounds,
            "b_bounds": self.b_bounds,

            # Dimensions
            'ndim': self.ndim,
            
            # Class identifier
            'gp_class': 'GPwithBLR'
        }

        return state
    

    @classmethod
    def from_state_dict(cls, state):
        gp = cls(
            train_x=state['train_x'],
            train_y=state['train_y'],
            noise=state['noise'],
            kernel=state['kernel_name'],
            optimizer=state['optimizer_method'],
            optimizer_options=state['optimizer_options'],
            lengthscales=state['lengthscales'],
            lengthscale_bounds=state['lengthscale_bounds'],
            lengthscale_prior=state.get('lengthscale_prior_spec'),
            a_prior=state.get("a_prior_spec", None),
            b_bounds=state.get("b_bounds", [-50.0, 50.0]),
            param_names=state.get("param_names", None)
        )

        if "a" in state:
            gp.a = jnp.array(state['a'])
        if "b_logits" in state:
            gp.b_logits = jnp.array(state['b_logits'])

        gp.kernel.update_hyperparams(
            lengthscales=gp.lengthscales,
            a=gp.a,
            b_logits=gp.b_logits
        )

        gp._rebuild_feature_cache()
        gp._rebuild_posterior_cache()
        
        return gp
    
    def copy(self):
        state = self.state_dict()
        return self.__class__.from_state_dict(state)
    
    

    def recompute_cholesky(self):
        """ 
        BLR analogue of recompute_cholesky: rebuild posterior caches.
        """

        self._rebuild_feature_cache()
        self._rebuild_posterior_cache()


    def update(self, new_x, new_y):
        """
        Updates the BLR GP with new training points.
        Mirrors base GP.update duplicate handling + restandardisation
        """
        new_x = jnp.atleast_2d(new_x)
        new_y = jnp.atleast_2d(new_y)

        duplicate = False
        new_pts_to_add = []
        new_vals_to_add = []
        
        # Check for duplicates and collect new points
        for i in range(new_x.shape[0]):
            if jnp.any(jnp.all(jnp.isclose(self.train_x, new_x[i], atol=1e-6, rtol=1e-4), axis=1)):
                log.debug(f"Point {new_x[i]} already exists in the training set, not updating")
            else:
                new_pts_to_add.append(new_x[i])
                new_vals_to_add.append(new_y[i])
        
        if not new_pts_to_add:
            return

        # Add new points if any
        new_pts_to_add = jnp.array(new_pts_to_add)
        new_vals_to_add = jnp.array(new_vals_to_add)
        
        # Add to training data
        self.train_x = jnp.vstack([self.train_x, new_pts_to_add])

        # Rebuild y in original scale then re-standardise
        train_y_original = jnp.vstack([self.train_y * self.y_std + self.y_mean, new_vals_to_add])
        self.y_mean = jnp.mean(train_y_original)
        self.y_std = jnp.std(train_y_original)
        if self.y_std == 0:
            log.warning("Training targets have zero variance. Setting std to 1.0 to avoid division by zero.")
            self.y_std = 1.0
        
        self.train_y = (train_y_original - self.y_mean) / self.y_std

        self.recompute_cholesky()

    
    def fantasy_var(self, new_x, mc_points, k_train_mc=None, include_noise=True):
        """
        BLR fantasy variance at mc_points after adding new_x
        Ignores k_train_mc (dense-GP optimisation input), kept for signature compatibility
        """

        new_x = jnp.atleast_2d(new_x)
        mc_points = jnp.atleast_2d(mc_points)

        if self.PsiTPsi is None:
            self._rebuild_feature_cache()

        sigma2 = self._noise_std()
        jitter = safe_noise_floor

        # Feature for new point
        psi_new = self.kernel._features(new_x) # (1, M)

        # Update sufficient statistics
        PsiTPsi_new = self.PsiTPsi + (psi_new.T @ psi_new) # (M, M)
        # PsiTy would depend on y_new, but variance doesn't need it.
        # We'll just recompute A from PsiTPsi_new.
        M = PsiTPsi_new.shape[0]
        A = jnp.eye(M, dtype=PsiTPsi_new.dtype) + (PsiTPsi_new / sigma2)
        A += jitter * jnp.eye(M, dtype=PsiTPsi_new.dtype)

        L = jnp.linalg.cholesky(A)

        # Predictive variance at mc_points: psi_*^T A^{-1} psi_*
        Psi_mc = self.kernel._features(mc_points)         # (Nmc, M)
        v = solve_triangular(L, Psi_mc.T, lower=True)     # (M, Nmc)
        var_std = jnp.sum(v * v, axis=0)                  # (Nmc,)

        var_std = jnp.where(jnp.isnan(var_std), safe_noise_floor, var_std)
        var_std = jnp.where(var_std < safe_noise_floor, safe_noise_floor, var_std)

        var_std += sigma2 if include_noise else 0.0

        return var_std * (self.y_std **2)
                            


    def init_params_optim_space(self):
        out = jnp.log(self.lengthscales)
        if not self.fixed_a:
            out = jnp.r_[out, jnp.log(self.a)]
        
        out = jnp.r_[out, self.b_logits]

        return out

