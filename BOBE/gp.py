
#from math import sqrt,pi
from typing import List
import jax.numpy as jnp
import numpy as np
import jax
#from jax.scipy.linalg import cho_solve, solve_triangular
jax.config.update("jax_enable_x64", True)
#from functools import partial
from .utils.log import get_logger
log = get_logger("gp")
from .optim import optimize_optax, optimize_scipy
from .utils.seed import get_numpy_rng
#import numpyro.distributions as dist
from .kernels import RBFKernel, MaternKernel, SphericalLinearKernel


safe_noise_floor = 1e-12

class GP:
    
    def __init__(self,train_x,train_y,noise=1e-8,kernel="rbf",optimizer="scipy",optimizer_options={},
                 kernel_variance_bounds = [1e-4, 1e8],lengthscale_bounds = [0.05, 10],lengthscales=None, kernel_variance=None,
                 kernel_variance_prior=None, lengthscale_prior="DSLP", tausq=None, tausq_bounds=[1e-4,1e4], 
                 raw_coeffs=None, raw_coeff_bounds=[-6, 6], raw_global_lengthscale=None, raw_global_lengthscale_bounds=[-1.5, 1.5], 
                 param_names: List[str] = None):
        """
        Initialize the Gaussian Process model.

        Parameters
        ----------
        train_x : jnp.ndarray
            Training inputs, shape (N, D).
        train_y : jnp.ndarray
            Objective function values at training points, shape (N, 1).
        noise : float, optional
            Noise parameter added to the diagonal of the kernel. Default is 1e-8.
        kernel : str, optional
            Kernel to use, either "rbf" or "matern". Default is "rbf".
        optimizer : str, optional
            Optimizer to use for hyperparameter tuning. Default is "scipy".
        optimizer_options : dict, optional
            Keyword arguments for the optimizer. Default is {}.
        kernel_variance_bounds : list, optional
            Bounds for the kernel variance. Default is [1e-4, 1e8].
        lengthscale_bounds : list, optional
            Bounds for the lengthscales. Default is [0.01, 10].
        lengthscales : jnp.ndarray, optional
            Initial lengthscale values. If None, defaults to ones. Default is None.
        kernel_variance : float, optional
            Initial kernel variance. If None, defaults to 1.0. Default is None.
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
            Bounds for the tausq parameter. Only used when lengthscale_prior='SAAS'.
            Defaults to [-4, 4].
        raw_coeffs: list, optional
            Initial raw b0 and b1 coefficient values for the linear spherical kernel. If None, defaults to [0.0, 0.0]
        raw_coeffs_bounds: list, optional
            Bounds for the b0 and b1 coefficients. Default is [-10, 10]
        raw_global_lengthscale: float, optional
            Initial raw global lengthscale parameter a for linear spherical kernel. If None, deftaults to 0.0
        raw_global_lengthscale_bounds: list, optional
            Bounds for the raw global lengthscale parameter a. default is [0.1*sqrt(ndim), 10*sqrt(ndim)]
        """
        # Setup and validate training data
        self._setup_training_data(train_x, train_y)
        self.param_names = param_names if param_names is not None else ['x_'+str(i) for i in range(self.ndim)]

        # Setup kernel and initial hyperparameters
        kernel_classes = {"rbf": RBFKernel, "matern": MaternKernel, 'spherical_linear': SphericalLinearKernel}
        if kernel not in kernel_classes:
            raise ValueError(f"Unknown Kernel '{kernel}'. Available: {list(kernel_classes)}")
        self.kernel_name = kernel #if kernel == "rbf" else "matern"
    

        # Store bounds
        lengthscale_bounds = lengthscale_bounds
        kernel_variance_bounds = kernel_variance_bounds
        # Can store tausq for convenience even though it is only used for SAAS
        tausq = float(tausq) if tausq is not None else 1.0
        tausq_bounds = tausq_bounds

        if self.kernel_name == 'spherical_linear':
            lengthscale_prior = 'dsp_unscaled'
        
        
        
        kernel_init = {
            "lengthscales": lengthscales if lengthscales is not None else jnp.ones(self.ndim),
            "kernel_variance": kernel_variance if kernel_variance is not None else 1.0,
            "tausq": tausq,
            "raw_coeffs": raw_coeffs,
            "raw_global_lengthscale":raw_global_lengthscale,
            "bounds":{
                "lengthscales": lengthscale_bounds,
                "kernel_variance": kernel_variance_bounds,
                "tausq": tausq_bounds,
                "raw_coeffs": raw_coeff_bounds,
                "raw_global_lengthscale": raw_global_lengthscale_bounds,
            },
            "priors":{
                "lengthscales": lengthscale_prior,
                "kernel_variance": kernel_variance_prior,
            },
            "input_bounds": (0.0, 1.0) # or (D,2)
        }
        
        
        # lengthscales = lengthscales if lengthscales is not None else jnp.ones(self.ndim)
        # kernel_variance = kernel_variance if kernel_variance is not None else 1.0
        self.noise = noise
        
        # Instantiate kernel object
        self.kernel = kernel_classes[self.kernel_name](kernel_init, self.noise)

       
        # Setup optimizer
        self.optimizer_method = optimizer
        if optimizer == "scipy":
            self.mll_optimize = optimize_scipy
        else:
            self.mll_optimize = optimize_optax
        self.optimizer_options = optimizer_options

        # Configure Priors
        self.kernel.configure_priors()
        self.kernel.configure_hyperparam_optimisation()

        # Compute initial kernel matrices
        self.kernel.build_posterior_cache(self.train_x, self.train_y)

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


    def fit(self, x0: np.ndarray = None, maxiter: int = 1000) -> dict:
        """
        Performs a serial fit for a given batch of starting points (x0).
        This method is called by each MPI process on its assigned chunk.

        Arguments
        ---------
        x0 : np.ndarray
            Array of shape (n_restarts_chunk, n_params) containing starting points for optimization (in log space).
        maxiter : int
            Maximum number of iterations for the optimizer. Defaults to 500.

        Returns
        -------
        result : dict
            Dictionary containing the best 'mll' and corresponding 'params' (log space) found.
        """

        if x0 is None: # set to current hyperparameters
            #x0 = jnp.log(self.kernel.get_hyperparams())[None, :]
            x0 = self.kernel.initial_log_params()[None, :]

        #log.info(f"Initial Params for restarts: {x0}")

        optimizer_options = self.optimizer_options.copy()

        best_params_log, best_loss = self.mll_optimize(
            fun=self.kernel.mll,
            num_params=self.kernel.num_hyperparams,
            bounds=self.kernel.hyperparam_bounds,
            x0=x0, # Use the chunk of starting points passed in
            maxiter=maxiter,
            n_restarts=x0.shape[0], # The number of restarts is the size of the chunk
            optimizer_options=optimizer_options
        )
        
        parsed = self.kernel.parse_hyperparams(best_params_log)
        self.kernel.update_hyperparams(*parsed)
        self.kernel.build_posterior_cache(self.train_x, self.train_y)

        log.info(f"Best MLL:  {-best_loss}")

        # Return the result in the format the pool expects
        return {
            'mll': -best_loss,
            'params': best_params_log # Optionally return the raw params
        }
    
    def predict_mean_single(self,x):
        """
        Single point prediction of mean
        """
        return self.kernel.predict_mean_single(x, self.y_mean, self.y_std)
    
    def predict_var_single(self,x):
        return self.kernel.predict_var_single(x, self.y_std)
    
    def predict_mean_batched(self,x):
        x = jnp.atleast_2d(x)
        return jax.vmap(self.predict_mean_single, in_axes=0)(x)
    
    def predict_var_batched(self,x):
        x = jnp.atleast_2d(x)
        return jax.vmap(self.predict_var_single, in_axes=0)(x)

    def predict_single(self,x):
        """
        Predicts the mean and variance of the GP at x but does not unstandardize it. To use with EI and the like.
        """
        # x = jnp.atleast_2d(x)
        # k12 = self.kernel.covariance(self.train_x, x, include_noise=False)
        # k22 = self.kernel.diagonal(x, include_noise=True)
        # mean = jnp.einsum('ij,ji', k12.T, self.alphas)
        # vv = solve_triangular(self.cholesky, k12, lower=True) # shape (N,1)
        # var = k22 - jnp.sum(vv*vv,axis=0) 
        # # handle nans and negative variances due to numerical issues
        # var = jnp.where(jnp.isnan(var),safe_noise_floor,var)
        # var = jnp.where(var<safe_noise_floor,safe_noise_floor,var)
        # return mean, var
        raise NotImplementedError
    
    def predict_batched(self,x):
        # x = jnp.atleast_2d(x)
        # return jax.vmap(self.predict_single, in_axes=0,out_axes=(0,0))(x)
        raise NotImplementedError

    def update(self,new_x,new_y):
        """
        Updates the GP with new training points and refits the GP if refit is True.

        Arguments
        ---------        
        refit: bool
            Whether to refit the GP hyperparameters. Default is True.
        maxiter: int
            The maximum number of iterations for the optax optimizer. Default is 200.
        n_restarts: int
            The number of restarts for the optax optimizer. Default is 4.
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

        # Add new points if any
        if new_pts_to_add:
            new_pts_to_add = jnp.array(new_pts_to_add)
            new_vals_to_add = jnp.array(new_vals_to_add)
            
            # Add to training data
            self.train_x = jnp.vstack([self.train_x, new_pts_to_add])
            train_y_original = jnp.vstack([self.train_y * self.y_std + self.y_mean, new_vals_to_add])
            
            self.y_mean = jnp.mean(train_y_original)
            self.y_std = jnp.std(train_y_original)
            
            if self.y_std == 0:
                log.warning("Training targets have zero variance. Setting std to 1.0 to avoid division by zero.")
                self.y_std = 1.0
            
            self.train_y = (train_y_original - self.y_mean) / self.y_std

            #self.recompute_cholesky()
            self.kernel.build_posterior_cache(self.train_x, self.train_y)


    def fantasy_var(self,new_x,mc_points,k_train_mc):
        """
        Computes the variance of the GP at the mc_points assuming a single point new_x is added to the training set
        """
        return self.kernel.fantasy_var(new_x, mc_points, k_train_mc, self.y_std)

    def get_random_point(self,rng=None,nstd=None):
        """
        Returns a random point in the unit cube.
        """
        log.debug(f"Getting random point in unit cube")
        rng = rng if rng is not None else get_numpy_rng()
        pt = rng.uniform(0, 1, size=self.train_x.shape[1])
        return pt

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

            # Standardization parameters
            'y_mean': float(self.y_mean),
            'y_std': float(self.y_std),

            # Model configuration
            'kernel_name': self.kernel_name,
            'noise': float(self.noise),
            'optimizer_method': self.optimizer_method,
            'optimizer_options': self.optimizer_options,

            # Bounds
            'lengthscale_bounds': self.kernel.bounds_spec['lengthscales'],
            'kernel_variance_bounds': self.kernel.bounds_spec['kernel_variance'],
            'tausq_bounds': self.kernel.bounds_spec['tausq'],
            
            # Prior Specs
            'lengthscale_prior_spec': self.kernel.prior_state['lengthscale_prior_spec'],
            'kernel_variance_prior_spec': self.kernel.prior_state['kernel_variance_prior_spec'],
            
            # Kernel hyperparameters in optimiser space
            "kernel_params_log": np.array(self.kernel.initial_log_params()),
    

            # Meta
            'ndim': self.ndim,
            "param_names": list(self.param_names) if getattr(self, "param_names", None) is not None else None,
            # Class identifier
            'gp_class': 'GP'

            # Computed state
            # 'cholesky': np.array(self.cholesky) if hasattr(self, 'cholesky') else None,
            # 'alphas': np.array(self.alphas) if hasattr(self, 'alphas') else None,
            #'fixed_kernel_variance': self.fixed_kernel_variance,
            # Hyperparameters
            # 'lengthscales': np.array(self.lengthscales),
            # 'kernel_variance': float(self.kernel_variance),
            
            # 'tausq': float(self.tausq),
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
            #lengthscales=state['lengthscales'],
            #kernel_variance=state['kernel_variance'],
            lengthscale_bounds=state['lengthscale_bounds'],
            kernel_variance_bounds=state['kernel_variance_bounds'],
            kernel_variance_prior=state.get('kernel_variance_prior_spec'),
            lengthscale_prior=state.get('lengthscale_prior_spec'),
            tausq=state.get('tausq_init', 1.0),
            tausq_bounds=state.get('tausq_bounds', [-4, 4])
        )

        lp = jnp.array(state["kernel_params_log"])
        parsed = gp.kernel.parse_hyperparams(lp)
        gp.kernel.update_hyperparams(*parsed)
        gp.kernel.build_posterior_cache(gp.train_x, gp.train_y)
        
        # Restore computed state if available
        # if state['cholesky'] is not None:
        #     gp.cholesky = jnp.array(state['cholesky'])
        # if state['alphas'] is not None:
        #     gp.alphas = jnp.array(state['alphas'])
        
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
    
    # def get_hyperparams(self):
    #     hp = self.lengthscales
    #     if not self.fixed_kernel_variance:
    #         hp = jnp.hstack([hp, self.kernel_variance])
    #     if self.lengthscale_prior_spec == 'SAAS':
    #         hp = jnp.hstack([hp, self.tausq])
    #     return hp
    
    # def hyperparams_dict(self):
    #     ls_str = {name: f"{float(val):.4f}" for name, val in zip(self.param_names, self.lengthscales)}
    #     param_dict = {
    #         'lengthscales': ls_str,
    #         'kernel_variance': f"{float(self.kernel_variance):.4f}",
    #     }
    #     if 'tausq' in self.hyperparam_names:
    #         param_dict['tausq'] = f"{float(self.tausq):.4f}"
    #     return param_dict

    def hyperparams_dict(self):
        lp = self.kernel.initial_log_params()
        parsed = self.kernel.parse_hyperparams(lp)

        if isinstance(self.kernel, (RBFKernel, MaternKernel)):
            lengthscales, kernel_variance, tausq = parsed
            ls_str = {name: f"{float(val):.4f}" for name, val in zip(self.param_names, np.array(lengthscales))}
            out = {"lengthscales": ls_str}

            if not getattr(self.kernel, "fixed_kernel_variance", False):
                out['kernel_variance'] = f"{float(kernel_variance):.4f}"

            if getattr(self.kernel, "tausq_enabled", False):
                out['tausq'] = f"{float(tausq):.4f}"
            return out
        else:
            lengthscales, raw_coeffs, raw_global_lengthscale = parsed
            ls_str = {name: f"{float(val):.4f}" for name, val in zip(self.param_names, np.array(lengthscales))}

            coeffs = jax.nn.softmax(jnp.array(raw_coeffs))
            out = {
                "lengthscales": ls_str,
                "raw_coeffs": [float(raw_coeffs[0]), float(raw_coeffs[1])],
                "coeffs_softmax": [float(coeffs[0]), float(coeffs[1])],
                "raw_global_lengthscale": float(raw_global_lengthscale),
                "noise": float(self.kernel.noise),
            }
            return out