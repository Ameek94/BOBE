from math import sqrt,pi
import time
from typing import Any,List
import jax.numpy as jnp
import numpy as np
import jax
from jax.scipy.linalg import cho_solve, solve_triangular
from scipy import optimize
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
    
    # Remove "name" and pass the rest as kwargs
    kwargs = {k: v for k, v in spec.items() if k != "name"}
    return dist_class(**kwargs)

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
    
    def __init__(self,train_x,train_y,noise=1e-8,kernel="rbf",optimizer="optax",optimizer_kwargs={'lr': 1e-3, 'name': 'adam'},
                 kernel_variance_bounds = [-4,8],lengthscale_bounds = [np.log10(0.05),2],lengthscales=None,kernel_variance=None,
                 kernel_variance_prior=None, lengthscale_prior=None):
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
            Kernel to use, either "rbf" or "matern". Defaults to "rbf".
        optimizer : str, optional
            Optimizer to use for hyperparameter tuning. Defaults to "optax".
        optimizer_kwargs : dict, optional
            Keyword arguments for the optimizer. Defaults to {'lr': 1e-3, 'name': 'adam'}.
        kernel_variance_bounds : list, optional
            Bounds for the kernel variance (in log10 space). Defaults to [-4, 8].
        lengthscale_bounds : list, optional
            Bounds for the lengthscales (in log10 space). Defaults to [log10(0.05), 2].
        lengthscales : jnp.ndarray, optional
            Initial lengthscale values. If None, defaults to ones. Defaults to None.
        kernel_variance : float, optional
            Initial kernel variance. If None, defaults to 1.0. Defaults to None.
        kernel_variance_prior : dict, optional
            Specification for the kernel variance prior. 
            If None, defaults to `{'name': 'LogNormal', 'loc': 0.0, 'scale': 0.5}`. Defaults to None.
        lengthscale_prior : str or dict, optional
            Specification for the lengthscale prior. 
            If 'DSLP' or None, uses the DSLP prior. Otherwise, uses the provided distribution spec. Defaults to None.
        """
        # check x and y sizes
        if train_x.shape[0] != train_y.shape[0]:
            raise ValueError("train_x and train_y must have the same number of points")
        if train_y.ndim != 2:
            raise ValueError("train_y must be 2D")
        if train_x.ndim != 2:
            raise ValueError("train_x must be 2D")
        
        log.debug(f"GP training size = {self.train_x.shape[0]}")

        self.ndim = train_x.shape[1]
        self.y_mean = jnp.mean(train_y)
        self.y_std = jnp.std(train_y)
        self.train_x = train_x
        self.train_y = (train_y - self.y_mean) / self.y_std

        self.kernel_name = kernel if kernel=="rbf" else "matern"
        self.kernel = rbf_kernel if kernel=="rbf" else matern_kernel
        self.lengthscales = lengthscales if lengthscales is not None else jnp.ones(self.ndim)
        self.kernel_variance = kernel_variance if kernel_variance is not None else 1.0
        self.noise = noise
        K = self.kernel(self.train_x, self.train_x, self.lengthscales, self.kernel_variance, noise=self.noise, include_noise=True)
        self.L = jnp.linalg.cholesky(K)
        self.alphas = cho_solve((self.L, True), self.train_y)

        self.optimizer_method = optimizer
        if optimizer == "scipy":
            self.mll_optimize = optimize_scipy
        else:
            self.mll_optimize = optimize_optax
        self.optimizer_kwargs = optimizer_kwargs
        
        self.kernel_variance_prior_spec = kernel_variance_prior
        if self.kernel_variance_prior_spec is None:
            self.kernel_variance_prior_spec = {'name': 'LogNormal', 'loc': 0.0, 'scale': 0.5}
        self.kernel_variance_prior_dist = make_distribution(self.kernel_variance_prior_spec)
           
        self.lengthscale_prior_spec = lengthscale_prior
        if self.lengthscale_prior_spec is None:
            self.lengthscale_prior_spec = 'DSLP'

        if self.lengthscale_prior_spec == 'DSLP':
            self.lengthscale_prior_dist = dist.LogNormal(loc=sqrt2 + 0.5*jnp.log(self.ndim), scale=sqrt3)
        else:
            self.lengthscale_prior_dist = make_distribution(self.lengthscale_prior_spec)

        self.lengthscale_bounds = lengthscale_bounds
        self.kernel_variance_bounds = kernel_variance_bounds
        self.hyperparam_bounds = [self.lengthscale_bounds]*self.ndim + [self.kernel_variance_bounds]
        self.hyperparam_bounds = jnp.array(self.hyperparam_bounds).T # shape (2, D+1)
        log.debug(f" Hyperparameter bounds (log10) =  {self.hyperparam_bounds}")

    def neg_mll(self,log10_params):
        """
        Computes the negative log marginal likelihood for the GP with given hyperparameters.
        """
        hyperparams = 10**log10_params
        lengthscales = hyperparams[0:-1]
        kernel_variance = hyperparams[-1]
        K = self.kernel(self.train_x, self.train_x, lengthscales, kernel_variance, noise=self.noise, include_noise=True)
        cholesky = jax.scipy.linalg.cho_factor(K, lower=True)
        log_det_K = 2 * jnp.sum(jnp.log(jnp.diag(cholesky[0])))
        mll = -0.5 * self.train_y.T @ jax.scipy.linalg.cho_solve(cholesky, self.train_y) - 0.5 * log_det_K - 0.5 * self.npoints * jnp.log(2 * jnp.pi)
        
        logprior = 0.0
        logprior += self.kernel_variance_prior_dist.log_prob(kernel_variance)
        logprior += self.lengthscale_prior_dist.log_prob(lengthscales).sum()

        return -mll + logprior

    def fit(self, maxiter=200,n_restarts=4):
        """ 
        Fits the GP using maximum likelihood hyperparameters with the chosen optimizer.

        Arguments
        ---------
        maxiter: int
            The maximum number of iterations for the optimizer. Default is 200.
        n_restarts: int
            The number of restarts for the optimizer. Default is 4.
        """
        init_params = jnp.log10(jnp.concatenate([self.lengthscales, jnp.array([self.kernel_variance])]))
        init_params_u = scale_to_unit(init_params, self.hyperparam_bounds)
        if n_restarts>1:
            addn_init_params = init_params_u + 0.25*np.random.normal(size=(n_restarts-1, init_params.shape[0]))
            init_params_u = np.vstack([init_params_u, addn_init_params])
        x0 = jnp.clip(init_params_u, 0.0, 1.0)
        log.info(f"Fitting GP with initial params lengthscales = {self.lengthscales}, kernel_variance = {self.kernel_variance}")

        optimizer_kwargs = self.optimizer_kwargs.copy()

        best_params, best_f = self.mll_optimize(
            fun=self.neg_mll,
            ndim=self.ndim + 1,
            bounds=self.hyperparam_bounds,
            x0=x0,
            maxiter=maxiter,
            n_restarts=n_restarts,
            optimizer_kwargs=optimizer_kwargs
        )

        hyperparams = 10 ** best_params
        self.lengthscales = hyperparams[:-1]
        self.kernel_variance = hyperparams[-1]
        log.info(f"Final hyperparams: lengthscales = {self.lengthscales}, kernel_variance = {self.kernel_variance}, final MLL = {-best_f}")

        K = self.kernel(self.train_x, self.train_x, self.lengthscales, self.kernel_variance, noise=self.noise, include_noise=True)
        self.cholesky = jnp.linalg.cholesky(K)
        self.alphas = cho_solve((self.cholesky, True), self.train_y)

    def predict_mean_single(self,x):
        """
        Single point prediction of mean
        """
        x = jnp.atleast_2d(x)
        k12 = self.kernel(self.train_x,x,self.lengthscales,self.kernel_variance,noise=self.noise,include_noise=False) # shape (N,1)
        mean = jnp.einsum('ij,ji', k12.T, self.alphas)*self.y_std + self.y_mean 
        return mean 
    
    def predict_var_single(self,x):
        x = jnp.atleast_2d(x)
        k12 = self.kernel(self.train_x,x,self.lengthscales,self.kernel_variance,noise=self.noise,include_noise=False) # shape (N,1)
        vv = solve_triangular(self.cholesky, k12, lower=True) # shape (N,1)
        k22 = kernel_diag(x,self.kernel_variance,self.noise,include_noise=True) # shape (1,) for x (1,ndim)
        var = k22 - jnp.sum(vv*vv,axis=0) 
        var = jnp.clip(var, safe_noise_floor, None)
        return self.y_std**2 * var.squeeze()
    
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
        x = jnp.atleast_2d(x)
        k12 = self.kernel(self.train_x,x,self.lengthscales,self.kernel_variance,noise=self.noise,include_noise=False)
        k22 = kernel_diag(x,self.kernel_variance,self.noise,include_noise=True)
        mean = jnp.einsum('ij,ji', k12.T, self.alphas)
        vv = solve_triangular(self.cholesky, k12, lower=True) # shape (N,1)
        var = k22 - jnp.sum(vv*vv,axis=0) 
        var = jnp.clip(var, safe_noise_floor, None)
        return mean, var
    
    def predict_batched(self,x):
        x = jnp.atleast_2d(x)
        return jax.vmap(self.predict_single, in_axes=0,out_axes=(0,0))(x)

    def update(self,new_x,new_y,refit=True,maxiter=200,n_restarts=4):
        """
        Updates the GP with new training points and refits the GP if refit is True.

        Arguments
        ---------        
        refit: bool
            Whether to refit the GP hyperparameters. Default is True.
        lr: float
            The learning rate for the optax optimizer. Default is 1e-2.
        maxiter: int
            The maximum number of iterations for the optax optimizer. Default is 250.
        n_restarts: int
            The number of restarts for the optax optimizer. Default is 2.

        Returns
        -------
        repeat: bool
            Whether the point new_x, new_y already exists in the training set.

        """
        new_x = jnp.atleast_2d(new_x)
        new_y = jnp.atleast_2d(new_y)

        duplicate = False
        for i in range(new_x.shape[0]):
            if jnp.any(jnp.all(jnp.isclose(self.train_x, new_x[i], atol=1e-6,rtol=1e-4), axis=1)):
                log.debug(f"Point {new_x[i]} already exists in the training set, not updating")
                duplicate = True
            else:
                self.add(new_x[i],new_y[i])
        if refit:
            self.fit(maxiter=maxiter,n_restarts=n_restarts)
        else:
            K = self.kernel(self.train_x, self.train_x, self.lengthscales, self.kernel_variance, noise=self.noise, include_noise=True)
            self.cholesky = jnp.linalg.cholesky(K)
            self.alphas = cho_solve((self.cholesky, True), self.train_y)
        return duplicate

    def add(self,new_x,new_y):
        """
        Updates the GP with new training points.
        """
        new_x = jnp.atleast_2d(new_x)
        new_y = jnp.atleast_2d(new_y)
        self.train_x = jnp.concatenate([self.train_x,new_x])
        new_y_scaled = (new_y - self.y_mean) / self.y_std
        self.train_y = jnp.concatenate([self.train_y, new_y_scaled])
        return False

    def __getstate__(self):
        """
        Custom getstate method to pickle the GP object.
        """
        state = self.__dict__.copy()
        # Remove unpicklable attributes
        state.pop("cholesky", None)
        state.pop("alphas", None)
        return state

    def __setstate__(self, state):
        """
        Custom setstate method to unpickle the GP object.
        """
        self.__dict__.update(state)
        # Recompute cholesky and alphas
        self.cholesky = jnp.linalg.cholesky(self.kernel(self.train_x, self.train_x, self.lengthscales, self.kernel_variance, noise=self.noise, include_noise=True))
        self.alphas = cho_solve((self.cholesky, True), self.train_y)

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
            data = np.load(filename)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file {filename}")
        
        init_kwargs = {
            'train_x': jnp.array(data['train_x']),
            'train_y': jnp.array(data['train_y']),
            'noise': float(data['noise']),
            'kernel': str(data['kernel']),
            'optimizer': str(data['optimizer']),
            'kernel_variance_bounds': jnp.array(data['kernel_variance_bounds']),
            'lengthscale_bounds': jnp.array(data['lengthscale_bounds']),
            'lengthscales': jnp.array(data['lengthscales']),
            'kernel_variance': float(data['kernel_variance']),
        }
        if 'kernel_variance_prior' in data.files and data['kernel_variance_prior'] is not None:
            init_kwargs['kernel_variance_prior'] = data['kernel_variance_prior'].item()
        if 'lengthscale_prior' in data.files and data['lengthscale_prior'] is not None:
            init_kwargs['lengthscale_prior'] = data['lengthscale_prior'].item()

        init_kwargs.update(kwargs)
        gp = cls(**init_kwargs)
        
        if 'cholesky' in data.files:
            gp.cholesky = jnp.array(data['cholesky'])
            gp.alphas = cho_solve((gp.cholesky, True), gp.train_y)

        log.info(f"Loaded GP from {filename} with {gp.train_x.shape[0]} training points")
        return gp

    def save(self,outfile='gp'):
        """
        Saves the GP to a file

        Arguments
        ---------
        outfile: str
            The name of the file to save the GP to. Default is 'gp'.
        """
        save_dict = {
            'train_x': self.train_x,
            'train_y': self.train_y * self.y_std + self.y_mean, # unstandardize
            'noise': self.noise,
            'kernel': self.kernel_name,
            'optimizer': self.optimizer_method,
            'kernel_variance_bounds': self.kernel_variance_bounds,
            'lengthscale_bounds': self.lengthscale_bounds,
            'lengthscales': self.lengthscales,
            'kernel_variance': self.kernel_variance,
            'y_mean': self.y_mean,
            'y_std': self.y_std,
            'kernel_variance_prior': self.kernel_variance_prior_spec,
            'lengthscale_prior': self.lengthscale_prior_spec,
        }
        save_dict['cholesky'] = self.cholesky
        save_dict['alphas'] = self.alphas

        np.savez(f'{outfile}.npz', **save_dict)
        log.info(f"Saved GP to {outfile}.npz")

    def fantasy_var(self,new_x,mc_points,k_train_mc):
        """
        Computes the variance of the GP at the mc_points assuming a single point new_x is added to the training set
        """

        new_x = jnp.atleast_2d(new_x)
        # new_train_x = jnp.concatenate([self.train_x,new_x])
        k = self.kernel(self.train_x, new_x,self.lengthscales,self.kernel_variance,
                        noise=self.noise,include_noise=False).flatten()           # shape (n,)
        k_self = kernel_diag(new_x,self.kernel_variance,self.noise,include_noise=True)[0]  # scalar
        k11_cho = fast_update_cholesky(self.cholesky,k,k_self)

        # Compute only the extra row for new_x
        k_new_mc = self.kernel(
            new_x, mc_points,
            self.lengthscales, self.kernel_variance,
        noise=self.noise, include_noise=False)  # shape (1, n_mc)
        k12 = jnp.vstack([k_train_mc,k_new_mc])

        # k12 = self.kernel(new_train_x,mc_points,self.lengthscales,
        #                   self.kernel_variance,noise=self.noise,include_noise=False)
        k22 = kernel_diag(mc_points,self.kernel_variance,self.noise,include_noise=True) # (N_mc,)
        vv = solve_triangular(k11_cho, k12, lower=True) # shape (N_train,N_mc)
        var = k22 - jnp.sum(vv*vv,axis=0) 
        return var * self.y_std**2 # return to physical scale for better interpretability
        
    def get_phys_points(self,x_bounds):
        """
        Returns the physical points
        """
        x = scale_from_unit(self.train_x,x_bounds)
        y = self.train_y*self.y_std + self.y_mean 
        return x,y

    def copy(self):
        """
        Returns a copy of the GP
        """
        new_gp = GP(self.train_x, self.train_y, self.noise, self.kernel_name, self.optimizer_method,
                    self.optimizer_kwargs, self.kernel_variance_bounds, self.lengthscale_bounds,
                    self.lengthscales, self.kernel_variance,
                    self.kernel_variance_prior_spec, self.lengthscale_prior_spec)
        new_gp.cholesky = self.cholesky
        new_gp.alphas = self.alphas

        return new_gp

    @property
    def hyperparams(self):
        """
        Returns the current hyperparameters of the GP.
        """
        return {
            "lengthscales": self.lengthscales,
            "kernel_variance": self.kernel_variance
        }
    
    @property
    def npoints(self):
        return self.train_x.shape[0]