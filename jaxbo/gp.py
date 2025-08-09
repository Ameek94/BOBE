from abc import ABC, abstractmethod
from math import sqrt,pi
import time
from typing import Any,List
import jax.numpy as jnp
import numpy as np
import jax
from jax.scipy.linalg import cho_solve, solve_triangular
from .utils import scale_to_unit, scale_from_unit
jax.config.update("jax_enable_x64", True)
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from numpyro.infer.initialization import init_to_value, init_to_sample
from numpyro.util import enable_x64
enable_x64()
from functools import partial
import logging
from .logging_utils import get_logger
log = get_logger("[gp]")
from optax import adam, apply_updates
from .optim import optimize
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from .seed_utils import get_new_jax_key

# todo
# 
sqrt2 = sqrt(2.)
sqrt3 = sqrt(3.)
sqrt5 = sqrt(5.)

def dist_sq(x, y):
    """
    Compute squared Euclidean distance between two points x, y. 
    If x is n1 x d and y is n2 x d returns a n1 x n2 matrix of distancess.
    """
    return jnp.sum(jnp.square(x[:,None,:] - y[None,:,:]),axis=-1) 

@partial(jax.jit,static_argnames='include_noise')
def rbf_kernel(xa,
               xb,
               lengthscales,
               outputscale,
               noise,include_noise=True): 
    """
    The RBF kernel
    """
    sq_dist = dist_sq(xa/lengthscales,xb/lengthscales) 
    sq_dist = jnp.exp(-0.5*sq_dist)
    k = outputscale*sq_dist
    if include_noise:
        k+= noise*jnp.eye(k.shape[0])
    return k

@partial(jax.jit,static_argnames='include_noise')
def matern_kernel(xa,xb,lengthscales,outputscale,noise,include_noise=True):
    """
    The Matern-5/2 kernel
    """
    dsq = dist_sq(xa/lengthscales,xb/lengthscales)
    d = jnp.sqrt(jnp.where(dsq<1e-30,1e-30,dsq))
    exp = jnp.exp(-sqrt5*d)
    poly = 1. + d*(sqrt5 + d*5./3.)
    k = outputscale*poly*exp
    if include_noise:
        k+= noise*jnp.eye(k.shape[0])
    return k

@jax.jit
def get_var_from_cho(k11_cho,k12,k22):
    vv = solve_triangular(k11_cho,k12,lower=True)
    var = jnp.diag(k22) - jnp.sum(vv*vv,axis=0) # replace k22 computation with single element diagonal
    return var
    
@jax.jit
def get_mean_from_cho(k12,alphas):
    mu = jnp.matmul(jnp.transpose(k12),alphas)
    mean = mu.squeeze(-1)
    return mean

@jax.jit
def gp_predict(k11_cho,alphas,k12,k22):
    """
    Predicts the GP mean and variance at x
    """
    mean = get_mean_from_cho(k12,alphas)
    var = get_var_from_cho(k11_cho,k12,k22)
    return mean, var

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
def fast_update_kernel(K: jnp.ndarray, k: jnp.ndarray, k_self: float):
    n = K.shape[0]
    new_K = jnp.zeros((n+1,n+1),dtype=K.dtype)
    new_K = new_K.at[:n,:n].set(K)
    new_K = new_K.at[:n,n].set(k)
    new_K = new_K.at[n,:n].set(k.T)
    new_K = new_K.at[n,n].set(k_self)
    return new_K

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
def fast_update_kernel_cholesky(K: jnp.ndarray,L: jnp.ndarray,k: jnp.ndarray,k_self: float):
    """
    Fast update of the Kernel and its Cholesky factorization
    """
    K_new = fast_update_kernel(K, k, k_self)
    L_new = fast_update_cholesky(L, k, k_self)
    return K_new, L_new

class GP(ABC):
    """
    Base class for the GP
    """

    def __init__(self,train_x,train_y,noise=1e-8,kernel="rbf",optimizer="adam"
                 ,outputscale_bounds = [-4,4],lengthscale_bounds = [np.log10(0.05),2],lengthscales=None,outputscale=None):
        """
        Arguments
        ---------
        train_x: jnp.ndarray
            Training inputs, size (N x D)
        train_y: jnp.ndarray
            Objective function values at training points size (N x 1)
        noise: float
            Noise parameter to add to the diagonal of the kernel, default 1e-8
        kernel: str
            Kernel to use, either "rbf" or "matern"
        """
        self.train_x = train_x
        self.train_y = train_y
        # check x and y sizes
        if train_x.shape[0] != train_y.shape[0]:
            raise ValueError("train_x and train_y must have the same number of points")
        if train_y.ndim != 2:
            raise ValueError("train_y must be 2D")
        if train_x.ndim != 2:
            raise ValueError("train_x must be 2D")
        
        log.info(f"GP training size = {self.train_x.shape[0]}")

        self.ndim = train_x.shape[1]
        self.npoints = train_x.shape[0]
        self.y_mean = jnp.mean(train_y,axis=0)
        self.y_std = jnp.std(train_y,axis=0)
        self.kernel = rbf_kernel if kernel=="rbf" else matern_kernel
        self.noise = noise
        self.fitted = False
        self.lengthscales = lengthscales if lengthscales is not None else jnp.ones(self.ndim)
        self.outputscale = outputscale if outputscale is not None else 1.0
        self.cholesky = None
        self.alphas = None

        self.train_y = (train_y - self.y_mean) / self.y_std
        self.lengthscales = jnp.ones(self.ndim) if lengthscales is None else jnp.array(lengthscales)
        self.outputscale = 1. if outputscale is None else outputscale
        self.lengthscale_bounds = lengthscale_bounds
        self.outputscale_bounds = outputscale_bounds
        self.hyperparam_bounds = [self.lengthscale_bounds]*self.ndim + [self.outputscale_bounds]
        log.info(f" Hyperparameter bounds (log10) =  {self.hyperparam_bounds}")


    def fit(self,lr=1e-2,maxiter=150,n_restarts=2):
        """
        Fits the GP using maximum likelihood hyperparameters with the optax adam optimizer. Starts from current hyperparameters.

        Arguments
        ---------
        lr: float
            The learning rate for the optax optimizer. Default is 1e-2.
        maxiter: int
            The maximum number of iterations for the optax optimizer. Default is 250.
        n_restarts: int
            The number of restarts for the optax optimizer. Default is 2.

        """
        ndim = self.ndim
        outputscale = jnp.array([self.outputscale])
        init_params = jnp.log10(jnp.concatenate([self.lengthscales,outputscale]))
        log.info(f" Fitting GP with initial params lengthscales = {self.lengthscales}, outputscale = {self.outputscale}")
        bounds = jnp.array(self.hyperparam_bounds)
        mins = bounds[:,0]
        maxs = bounds[:,1]
        scales = maxs - mins
        optimizer = adam(learning_rate=lr)

        # Optimization to be moved, use the method in optim.py
        @jax.jit
        def mll_optim(params):
            hyperparams = 10**params
            lengthscales = hyperparams[0:-1]
            outputscale = hyperparams[-1]
            k = self.kernel(self.train_x,self.train_x,lengthscales,outputscale,noise=self.noise,include_noise=True)
            mll = gp_mll(k,self.train_y,self.train_y.shape[0])
            return -mll    

        @jax.jit
        def step(carry):
            """
            Step function for the optimizer
            """
            u_params, opt_state = carry
            params = scale_from_unit(u_params,bounds.T)
            loss, grads = jax.value_and_grad(mll_optim)(params)
            grads = grads * scales
            updates, opt_state = optimizer.update(grads, opt_state)
            u_params = apply_updates(u_params, updates)
            u_params = jnp.clip(u_params, 0.,1.)
            carry = u_params, opt_state
            return carry, loss
        
        best_f = jnp.inf
        best_params = None

        u_params = scale_to_unit(init_params,bounds.T)

        # display with progress bar
        r = jnp.arange(maxiter)
        for n in range(n_restarts):
            opt_state = optimizer.init(u_params)
            progress_bar = tqdm.tqdm(r,desc=f'Training GP')
            with logging_redirect_tqdm():
                for i in progress_bar:
                    (u_params,opt_state), fval  = step((u_params,opt_state))#,None)
                    progress_bar.set_postfix({"fval": float(fval)})
                    if fval < best_f:
                        best_f = fval
                        best_params = u_params
            u_params = jnp.clip(u_params + 0.25*np.random.normal(size=init_params.shape),0,1)
        params = scale_from_unit(best_params,bounds.T)
        hyperparams = 10 ** params
        self.lengthscales = hyperparams[0:-1]
        self.outputscale = hyperparams[-1]
        log.info(f" Final hyperparams: lengthscales = {self.lengthscales}, outputscale = {self.outputscale}")
        self.K = self.kernel(self.train_x,self.train_x,self.lengthscales,self.outputscale,noise=self.noise,include_noise=True)
        self.cholesky = jnp.linalg.cholesky(self.K)
        self.alphas = cho_solve((self.cholesky, True), self.train_y)
        self.fitted = True

    def predict_mean(self,x):
        """
        Predicts the mean of the GP at x and unstandardizes it
        """
        x = jnp.atleast_2d(x)
        k12 = self.kernel(self.train_x,x,self.lengthscales,self.outputscale,noise=self.noise,include_noise=False)
        mean = get_mean_from_cho(k12,self.alphas) 
        return mean*self.y_std + self.y_mean 

    def predict_var(self,x):
        """
        Predicts the variance of the GP at x and unstandardizes it
        """
        x = jnp.atleast_2d(x)
        k12 = self.kernel(self.train_x,x,self.lengthscales,self.outputscale,noise=self.noise,include_noise=False)
        k22 = self.kernel(x,x,self.lengthscales,self.outputscale,noise=self.noise,include_noise=True)
        var = get_var_from_cho(self.cholesky,k12,k22)
        return var*self.y_std**2

    def predict(self,x):
        """
        Predicts the mean and variance of the GP at x but does not unstandardize it
        """
        x = jnp.atleast_2d(x)
        k12 = self.kernel(self.train_x,x,self.lengthscales,self.outputscale,noise=self.noise,include_noise=False)
        k22 = self.kernel(x,x,self.lengthscales,self.outputscale,noise=self.noise,include_noise=True)
        mean, var = gp_predict(self.cholesky,self.alphas,k12,k22)
        return mean, var

    def update(self,new_x,new_y,refit=True,lr=1e-2,maxiter=150,n_restarts=2,step=0):
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
        if jnp.any(jnp.all(jnp.isclose(self.train_x, new_x, atol=1e-6,rtol=1e-4), axis=1)):
            log.info(f"Point {new_x} already exists in the training set, not updating")
            return True
        else:
            k = self.kernel(self.train_x, new_x,self.lengthscales,self.outputscale,
                        noise=self.noise,include_noise=False).flatten()           # shape (n,)
            k_self = self.kernel(new_x,new_x,self.lengthscales,
                          self.outputscale,noise=self.noise,include_noise=True)[0, 0]  # scalar            
            self.add(new_x,new_y)
            if refit:
                self.fit(lr=lr,maxiter=maxiter,n_restarts=n_restarts)
            else:
            # consider doing rank 1 update of cholesky
                # self.K = self.kernel(self.train_x,self.train_x,self.lengthscales,self.outputscale,noise=self.noise,include_noise=True)
                self.cholesky = fast_update_cholesky(self.cholesky,k,k_self)
                # self.cholesky = jnp.linalg.cholesky(self.K)
                self.alphas = cho_solve((self.cholesky, True), self.train_y)
            return False

    def add(self,new_x,new_y):
        """
        Updates the GP with new training points.
        """
        self.train_x = jnp.concatenate([self.train_x,new_x])
        self.train_y = self.train_y*self.y_std + self.y_mean 
        self.train_y = jnp.concatenate([self.train_y,new_y])
        self.y_mean = jnp.mean(self.train_y,axis=0)
        self.y_std = jnp.std(self.train_y,axis=0)
        self.train_y = (self.train_y - self.y_mean) / self.y_std
        self.npoints = self.train_x.shape[0]
        log.info("Updated GP with new point.")
        log.info(f" GP training size = {self.npoints}")

    def fantasy_var(self,new_x,mc_points):
        """
        Computes the variance of the GP at the mc_points assuming new_x is added to the training set
        """
        new_x = jnp.atleast_2d(new_x)
        new_train_x = jnp.concatenate([self.train_x,new_x])
        k = self.kernel(self.train_x, new_x,self.lengthscales,self.outputscale,
                        noise=self.noise,include_noise=False).flatten()           # shape (n,)
        k_self = self.kernel(new_x,new_x,self.lengthscales,
                          self.outputscale,noise=self.noise,include_noise=True)[0, 0]  # scalar
        k11_cho = fast_update_cholesky(self.cholesky,k,k_self)
        k12 = self.kernel(new_train_x,mc_points,self.lengthscales,
                          self.outputscale,noise=self.noise,include_noise=False)
        k22 = self.kernel(mc_points,mc_points,self.lengthscales,
                          self.outputscale,noise=self.noise,include_noise=True) # precompute k22 instead
        var = get_var_from_cho(k11_cho,k12,k22)
        return var
    
    def get_phys_points(self,x_bounds):
        """
        Returns the physical points
        """
        x = scale_from_unit(self.train_x,x_bounds)
        y = self.train_y*self.y_std + self.y_mean 
        return x,y
    
    def save(self,outfile='gp'):
        """
        Saves the GP to a file
        """
        np.savez(f'{outfile}.npz',train_x=self.train_x,train_y=self.train_y,noise=self.noise,
         y_mean=self.y_mean,y_std=self.y_std,lengthscales=self.lengthscales,outputscale=self.outputscale)

    @classmethod
    def load(cls, filename, kernel="rbf", optimizer="adam", 
             outputscale_bounds=[-4,4], lengthscale_bounds=[np.log10(0.05),2]):
        """
        Loads a GP from a file
        
        Arguments
        ---------
        filename: str
            The name of the file to load the GP from (with or without .npz extension)
        kernel: str
            Kernel type for the GP. Default is 'rbf'. Can be 'rbf' or 'matern'.
        optimizer: str
            Optimizer type for the GP. Default is 'adam'.
        outputscale_bounds: list
            Bounds for the output scale of the GP (in log10 space). Default is [-4,4].
        lengthscale_bounds: list
            Bounds for the length scale of the GP (in log10 space). Default is [np.log10(0.05),2].
            
        Returns
        -------
        gp: GP
            The loaded GP object
        """
        if not filename.endswith('.npz'):
            filename += '.npz'
            
        data = np.load(filename)
        
        # Extract training data
        train_x = jnp.array(data['train_x'])
        train_y = jnp.array(data['train_y'])
        
        # Extract hyperparameters
        noise = float(data['noise'])
        
        # Create GP instance
        gp = cls(train_x, train_y, noise=noise, kernel=kernel, optimizer=optimizer,
                 outputscale_bounds=outputscale_bounds, lengthscale_bounds=lengthscale_bounds)
        
        # Restore saved parameters
        gp.y_mean = jnp.array(data['y_mean'])
        gp.y_std = jnp.array(data['y_std'])
        gp.lengthscales = jnp.array(data['lengthscales'])
        gp.outputscale = float(data['outputscale'])
        
        # Recompute kernel matrix and Cholesky decomposition
        gp.K = gp.kernel(gp.train_x, gp.train_x, gp.lengthscales, gp.outputscale, 
                        noise=gp.noise, include_noise=True)
        gp.cholesky = jnp.linalg.cholesky(gp.K)
        gp.alphas = cho_solve((gp.cholesky, True), gp.train_y)
        gp.fitted = True
        
        log.info(f"Loaded GP from {filename} with {train_x.shape[0]} training points")
        return gp
    
    def sample_GP_NUTS(self, warmup_steps=512, num_samples=512, progress_bar=True, thinning=8, verbose=True,
                       init_params=None, temp=1., rng_key=None):
        """
        Obtain samples from the posterior represented by the GP mean as the logprob.
        Optionally restarts MCMC if all logp values are the same.
        
        Arguments
        ---------
        gp: saas_fbgp
            The GP object
        rng_key: jnp.ndarray
            Random key
        warmup_steps: int
            Number of warmup steps for NUTS, default 512
        num_samples: int
            Number of samples to draw from the posterior, default 512
        progress_bar: bool
            If True, shows a progress bar, default True
        thinning: int
            Thinning factor for the MCMC samples, default 8
        verbose: bool
            If True, prints the MCMC summary, default False
        init_params: dict
            Initial parameters for the MCMC, default None   
        temp: float
            Temperature parameter for the logprob, default 1.0
        """

        rng_key = get_new_jax_key() if rng_key is None else rng_key

        def model():
            x = numpyro.sample('x', dist.Uniform(
                low=jnp.zeros(self.train_x.shape[1]),
                high=jnp.ones(self.train_x.shape[1])
            ))
            mean = self.predict_mean(x)
            numpyro.factor('y', mean / temp)
            numpyro.deterministic('logp', mean)

        # First run
        if init_params is not None:
            params = {'x': init_params}
            init_strategy = init_to_value(values=params)
        else:
            init_strategy = init_to_sample

        start = time.time()
        kernel = NUTS(model, dense_mass=False, max_tree_depth=5, init_strategy=init_strategy)
        mcmc = MCMC(kernel, num_warmup=warmup_steps, num_samples=num_samples,
                    num_chains=1, progress_bar=progress_bar, thinning=thinning)
        mcmc.run(rng_key)
        
        mc_samples = mcmc.get_samples()

        if verbose:
            mcmc.print_summary(exclude_deterministic=False)
        log.info(f"Sampled parameters MCMC took {time.time() - start:.4f} s")

        samples = {
            'x': mc_samples['x'],
            'logp': mc_samples['logp'],
            'best': mc_samples['x'][jnp.argmax(mc_samples['logp'])]
        }

        return samples

class DSLP_GP(GP):

    def __init__(self,train_x,train_y,noise=1e-8,kernel="rbf",optimizer="adam",
                 outputscale_bounds = [-4,4],lengthscale_bounds = [np.log10(0.05),2],lengthscales=None,outputscale=None):
        """
        Class for the Gaussian Process, single output based on maximum likelihood hyperparameters.
        Uses the dimension scaled lengthscale priors from the paper "Vanilla Bayesian Optimization Performs Great in High Dimensions" (2024),
        by Hvarfner, Carl and Hellsten, Erik Orm and Nardi, Luigi

        Arguments
        ---------
        train_x: JAX array of training points, shape (n_samples, n_features)
            The initial training points for the GP.
        train_y: JAX array of training values, shape (n_samples,)
            The initial training values for the GP.
        noise: Scalar noise level for the GP
            Default is 1e-8. This is the noise level for the GP.
        kernel: Kernel type for the GP
            Default is 'rbf'. This is the kernel type for the GP. Can be 'rbf' or 'matern'.
        optimizer: Optimizer type for the GP
            Default is 'adam'. This is the optimizer type for the GP.
        outputscale_bounds: Bounds for the output scale of the GP (in log10 space) 
            Default is [-4,4]. These are the bounds for the output scale of the GP.
        lengthscale_bounds: Bounds for the length scale of the GP (in log10 space) 
            Default is [np.log10(0.05),2]. These are the boundsfor the length scale of the GP.
        """
        super().__init__(train_x,train_y,noise,kernel,optimizer,outputscale_bounds,lengthscale_bounds,lengthscales=lengthscales,outputscale=outputscale)

    # --- Pytree methods ---
    def tree_flatten(self):
        # Choose dynamic (leaf) attributes: these are jax arrays (or None)
        leaves = (self.train_x, self.train_y, self.K, self.cholesky, self.alphas, self.y_mean, self.y_std, self.lengthscales, self.outputscale)
        # The rest are considered static auxiliary data
        aux_data = {
            "ndim": self.ndim,
            "npoints": self.npoints,
            "noise": self.noise,
            "kernel": self.kernel,
            "outputscale_bounds": self.outputscale_bounds,
            "lengthscale_bounds": self.lengthscale_bounds,
            "hyperparam_bounds": self.hyperparam_bounds,
            "fitted": self.fitted,
            "optimizer": self.optimizer,
        }
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        # Unpack dynamic leaves
        train_x, train_y, K, cholesky, alphas, y_mean, y_std, lengthscales, outputscale = leaves
        # Create an instance with minimal initialization
        obj = cls(train_x, train_y, noise=aux_data["noise"], 
                  kernel="rbf" if aux_data["kernel"] == rbf_kernel else "matern", 
                  optimizer=aux_data["optimizer"])
        # Restore static auxiliary data
        obj.ndim = aux_data["ndim"]
        obj.npoints = aux_data["npoints"]
        obj.outputscale_bounds = aux_data["outputscale_bounds"]
        obj.lengthscale_bounds = aux_data["lengthscale_bounds"]
        obj.hyperparam_bounds = aux_data["hyperparam_bounds"]
        obj.fitted = aux_data["fitted"]
        # Restore dynamic (leaf) attributes
        obj.train_x = train_x
        obj.train_y = train_y
        obj.cholesky = cholesky
        obj.y_mean = y_mean
        obj.y_std = y_std
        obj.lengthscales = lengthscales
        obj.outputscale = outputscale
        return obj

    def fit(self,lr=1e-2,maxiter=150,n_restarts=2):
        """
        Fits the GP using maximum likelihood hyperparameters with the optax adam optimizer. Starts from current hyperparameters.

        Arguments
        ---------
        lr: float
            The learning rate for the optax optimizer. Default is 1e-2.
        maxiter: int
            The maximum number of iterations for the optax optimizer. Default is 250.
        n_restarts: int
            The number of restarts for the optax optimizer. Default is 2.

        """
        ndim = self.ndim
        outputscale = jnp.array([self.outputscale])
        init_params = jnp.log10(jnp.concatenate([self.lengthscales,outputscale]))
        log.info(f" Fitting GP with initial params lengthscales = {self.lengthscales}, outputscale = {self.outputscale}")
        bounds = jnp.array(self.hyperparam_bounds)
        mins = bounds[:,0]
        maxs = bounds[:,1]
        scales = maxs - mins
        optimizer = adam(learning_rate=lr)

        # Optimization to be moved, use the method in optim.py
        @jax.jit
        def mll_optim(params):
            hyperparams = 10**params
            lengthscales = hyperparams[0:-1]
            outputscale = hyperparams[-1]
            logprior = dist.Gamma(2.0,0.15).log_prob(outputscale)
            logprior+= dist.LogNormal(loc=sqrt2 + 0.5*jnp.log(ndim) ,scale=sqrt3).expand([ndim]).log_prob(lengthscales).sum()
            k = self.kernel(self.train_x,self.train_x,lengthscales,outputscale,noise=self.noise,include_noise=True)
            mll = gp_mll(k,self.train_y,self.train_y.shape[0])
            return -(mll+logprior)        

        @jax.jit
        def step(carry):
            """
            Step function for the optimizer
            """
            u_params, opt_state = carry
            params = scale_from_unit(u_params,bounds.T)
            loss, grads = jax.value_and_grad(mll_optim)(params)
            grads = grads * scales
            updates, opt_state = optimizer.update(grads, opt_state)
            u_params = apply_updates(u_params, updates)
            u_params = jnp.clip(u_params, 0.,1.)
            carry = u_params, opt_state
            return carry, loss
        
        best_f = jnp.inf
        best_params = None

        u_params = scale_to_unit(init_params,bounds.T)

        # display with progress bar
        r = jnp.arange(maxiter)
        for n in range(n_restarts):
            opt_state = optimizer.init(u_params)
            progress_bar = tqdm.tqdm(r,desc=f'Training GP')
            with logging_redirect_tqdm():
                for i in progress_bar:
                    (u_params,opt_state), fval  = step((u_params,opt_state))#,None)
                    progress_bar.set_postfix({"fval": float(fval)})
                    if fval < best_f:
                        best_f = fval
                        best_params = u_params
            u_params = jnp.clip(u_params + 0.25*np.random.normal(size=init_params.shape),0,1)
        params = scale_from_unit(best_params,bounds.T)
        hyperparams = 10 ** params
        self.lengthscales = hyperparams[0:-1]
        self.outputscale = hyperparams[-1]
        log.info(f" Final hyperparams: lengthscales = {self.lengthscales}, outputscale = {self.outputscale}")
        self.K = self.kernel(self.train_x,self.train_x,self.lengthscales,self.outputscale,noise=self.noise,include_noise=True)
        self.cholesky = jnp.linalg.cholesky(self.K)
        self.alphas = cho_solve((self.cholesky, True), self.train_y)
        self.fitted = True

    # def fit(self,lr=1e-2,maxiter=150,n_restarts=2,optimizer_name="adam"):
    #     """
    #     Fits the GP using maximum likelihood hyperparameters with the optimize function. Starts from current hyperparameters.

    #     Arguments
    #     ---------
    #     lr: float
    #         The learning rate for the optimizer. Default is 1e-2.
    #     maxiter: int
    #         The maximum number of iterations for the optimizer. Default is 150.
    #     n_restarts: int
    #         The number of restarts for the optimizer. Default is 2.
    #     optimizer_name: str
    #         The name of the optimizer to use. Default is "adam".

    #     """
    #     ndim = self.ndim
    #     outputscale = jnp.array([self.outputscale])
    #     init_params = jnp.log10(jnp.concatenate([self.lengthscales,outputscale]))
    #     log.info(f" Fitting GP with initial params lengthscales = {self.lengthscales}, outputscale = {self.outputscale}")
    #     bounds = jnp.array(self.hyperparam_bounds)

    #     # Define MLL objective function
    #     def mll_optim(params):
    #         hyperparams = 10**params
    #         lengthscales = hyperparams[0:-1]
    #         outputscale = hyperparams[-1]
    #         logprior = dist.Gamma(2.0,0.15).log_prob(outputscale)
    #         logprior+= dist.LogNormal(loc=sqrt2 + 0.5*jnp.log(ndim) ,scale=sqrt3).expand([ndim]).log_prob(lengthscales).sum()
    #         k = self.kernel(self.train_x,self.train_x,lengthscales,outputscale,noise=self.noise,include_noise=True)
    #         mll = gp_mll(k,self.train_y,self.train_y.shape[0])
    #         return -(mll+logprior)        

    #     # Use the optimize function directly to minimize negative MLL
    #     best_params, best_f = optimize(
    #         func=mll_optim,
    #         ndim=ndim + 1,  # lengthscales + outputscale
    #         bounds=bounds,
    #         x0=init_params,
    #         lr=lr,
    #         maxiter=maxiter,
    #         n_restarts=n_restarts,
    #         minimize=True,
    #         verbose=True,
    #         optimizer_name=optimizer_name
    #     )
        
    #     # Update GP parameters with optimized values
    #     hyperparams = 10 ** best_params
    #     self.lengthscales = hyperparams[0:-1]
    #     self.outputscale = hyperparams[-1]
    #     log.info(f" Final hyperparams: lengthscales = {self.lengthscales}, outputscale = {self.outputscale}")
    #     self.K = self.kernel(self.train_x,self.train_x,self.lengthscales,self.outputscale,noise=self.noise,include_noise=True)
    #     self.cholesky = jnp.linalg.cholesky(self.K)
    #     self.alphas = cho_solve((self.cholesky, True), self.train_y)
    #     self.fitted = True

    # def update(self,new_x,new_y,refit=True,lr=1e-2,maxiter=200,n_restarts=2):
        # """
        # Updates the GP with new training points and refits the GP if refit is True.

        # Arguments
        # ---------        
        # refit: bool
        #     Whether to refit the GP hyperparameters. Default is True.
        # lr: float
        #     The learning rate for the optax optimizer. Default is 1e-2.
        # maxiter: int
        #     The maximum number of iterations for the optax optimizer. Default is 250.
        # n_restarts: int
        #     The number of restarts for the optax optimizer. Default is 2.

        # Returns
        # -------
        # repeat: bool
        #     Whether the point new_x, new_y already exists in the training set.

        # """
        # if jnp.any(jnp.all(jnp.isclose(self.train_x, new_x, atol=1e-6,rtol=1e-4), axis=1)):
        #     log.info(f"Point {new_x} already exists in the training set, not updating")
        #     return True
        # else:
        #     super().add(new_x,new_y)
        #     if refit:
        #         self.fit(lr=lr,maxiter=maxiter,n_restarts=n_restarts)
        #     else:
        #     # consider doing rank 1 update of cholesky
        #         k = self.kernel(self.train_x,self.train_x,self.lengthscales,self.outputscale,noise=self.noise,include_noise=True)
        #         self.cholesky = jnp.linalg.cholesky(k)
        #     return False

    def fantasy_var(self,new_x,mc_points):
        """
        Computes the variance of the GP at the mc_points assuming new_x is added to the training set
        """
        return super().fantasy_var(new_x,mc_points)

    @classmethod
    def load(cls, filename, kernel="rbf", optimizer="adam", 
             outputscale_bounds=[-4,4], lengthscale_bounds=[np.log10(0.05),2]):
        """
        Loads a DSLP_GP from a file
        
        Arguments
        ---------
        filename: str
            The name of the file to load the GP from (with or without .npz extension)
        kernel: str
            Kernel type for the GP. Default is 'rbf'. Can be 'rbf' or 'matern'.
        optimizer: str
            Optimizer type for the GP. Default is 'adam'.
        outputscale_bounds: list
            Bounds for the output scale of the GP (in log10 space). Default is [-4,4].
        lengthscale_bounds: list
            Bounds for the length scale of the GP (in log10 space). Default is [np.log10(0.05),2].
            
        Returns
        -------
        gp: DSLP_GP
            The loaded DSLP_GP object
        """
        if not filename.endswith('.npz'):
            filename += '.npz'
            
        data = np.load(filename)
        
        # Extract training data
        train_x = jnp.array(data['train_x'])
        train_y = jnp.array(data['train_y'])
        
        # Extract hyperparameters
        noise = float(data['noise'])
        
        # Create DSLP_GP instance
        gp = cls(train_x, train_y, noise=noise, kernel=kernel, optimizer=optimizer,
                 outputscale_bounds=outputscale_bounds, lengthscale_bounds=lengthscale_bounds)
        
        # Restore saved parameters
        gp.y_mean = jnp.array(data['y_mean'])
        gp.y_std = jnp.array(data['y_std'])
        gp.lengthscales = jnp.array(data['lengthscales'])
        gp.outputscale = float(data['outputscale'])
        
        # Recompute kernel matrix and Cholesky decomposition
        gp.K = gp.kernel(gp.train_x, gp.train_x, gp.lengthscales, gp.outputscale, 
                        noise=gp.noise, include_noise=True)
        gp.cholesky = jnp.linalg.cholesky(gp.K)
        gp.alphas = cho_solve((gp.cholesky, True), gp.train_y)
        gp.fitted = True
        
        log.info(f"Loaded DSLP_GP from {filename} with {train_x.shape[0]} training points")
        return gp
        
# Register DSLP_GP as a pytree node with JAX
jax.tree_util.register_pytree_node(
    DSLP_GP,
    DSLP_GP.tree_flatten,
    DSLP_GP.tree_unflatten,
)

class SAAS_GP(DSLP_GP):
    def __init__(self,
                 train_x, train_y, noise=1e-8, kernel="rbf", optimizer="adam",
                 outputscale_bounds = [-4,4],lengthscale_bounds = [np.log10(0.05),2],
                 tausq_bounds = [-4,4]):
        """
        Class for the Gaussian Process with SAAS priors, using maximum likelihood hyperparameters. 
        The implementation is based on the paper "High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces", 2021
        by David Eriksson and Martin Jankowiak.


        Arguments
        ---------
        train_x: JAX array of training points, shape (n_samples, n_features)
            The initial training points for the GP.
        train_y: JAX array of training values, shape (n_samples,)
            The initial training values for the GP.
        noise: Scalar noise level for the GP
            Default is 1e-8. This is the noise level for the GP.
        kernel: Kernel type for the GP
            Default is 'rbf'. This is the kernel type for the GP. Can be 'rbf' or 'matern'.
        optimizer: Optimizer type for the GP
            Default is 'adam'. This is the optimizer type for the GP.
        outputscale_bounds: Bounds for the output scale of the GP (in log10 space) 
            Default is [-4,4]. These are the bounds for the output scale of the GP.
        lengthscale_bounds: Bounds for the length scale of the GP (in log10 space) 
            Default is [np.log10(0.05),2]. These are the bounds for the length scale of the GP.
        tausq_bounds: Bounds for the tausq parameter of the GP (in log10 space)
            Default is [-4,4]. These are the bounds for the tausq parameter of the GP.
        """
        super().__init__(train_x, train_y, noise, kernel, optimizer,outputscale_bounds,lengthscale_bounds)
        self.tausq = 1.
        self.hyperparam_bounds = [tausq_bounds] + self.hyperparam_bounds

    def tree_flatten(self):
        # use the parent class tree_flatten to simplify
        parent_leaves, aux_data = super().tree_flatten()
        # Add dynamic leaves
        leaves = parent_leaves + (self.tausq,)
        # no extra aux data 
        return leaves, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        parent_aux, extra_aux = aux_data
        # Determine how many leaves the parent expected.
        num_parent_leaves = len(DSLP_GP.tree_flatten(cls.__new__(cls))[0])
        parent_leaves = leaves[:num_parent_leaves]
        extra_leaves = leaves[num_parent_leaves:]
        # First, create a parent instance from the parent's leaves.
        obj = DSLP_GP.tree_unflatten(parent_aux, parent_leaves)
        # Convert it into an instance of DerivedGP. One way is to update the class:
        obj.__class__ = cls
        # Now restore the extra fields.
        obj.tausq = extra_leaves
        # No extra static auxiliary data
        return obj

    def fit(self,lr=1e-2,maxiter=250,n_restarts=2):
        """
        Fits the GP using maximum likelihood hyperparameters with the optax adam optimizer. Starts from current hyperparameters.

        Arguments
        ---------
        lr: float
            The learning rate for the optax optimizer. Default is 1e-2.
        maxiter: int
            The maximum number of iterations for the optax optimizer. Default is 250.
        n_restarts: int
            The number of restarts for the optax optimizer. Default is 2.

        """
        # print(f"hyparam shapes {self.lengthscales.shape}")
        tausq = jnp.array([self.tausq])
        outputscale = jnp.array([self.outputscale])
        init_params = jnp.log10(jnp.concatenate([tausq,self.lengthscales,outputscale])) #jnp.zeros(self.ndim+2)
        log.info(f"Fitting GP with initial params tausq = {self.tausq}, lengthscales = {self.lengthscales}, outputscale = {self.outputscale}")
        bounds = jnp.array(self.hyperparam_bounds)
        mins = bounds[:,0]
        maxs = bounds[:,1]
        scales = maxs - mins
        optimizer = adam(learning_rate=lr)

        @jax.jit
        def mll_optim(params):
            hyperparams = 10**params
            tausq = hyperparams[0]
            lengthscales = hyperparams[1:-1]
            outputscale = hyperparams[-1]
            logprior = dist.Gamma(2.0,0.15).log_prob(outputscale)
            tausq = hyperparams[0]
            logprior+= dist.HalfCauchy(0.1).log_prob(tausq)
            lengthscales = hyperparams[1:-1]
            inv_lengthscales_sq = 1/ (tausq * lengthscales**2)
            logprior+= jnp.sum(dist.HalfCauchy(1.).log_prob(inv_lengthscales_sq))
            k = self.kernel(self.train_x,self.train_x,lengthscales,outputscale,noise=self.noise,include_noise=True)
            mll = gp_mll(k,self.train_y,self.train_y.shape[0])
            return -(mll+logprior)        

        @jax.jit
        def step(carry):
            """
            Step function for the optimizer
            """
            u_params, opt_state = carry
            params = scale_from_unit(u_params,bounds.T)
            loss, grads = jax.value_and_grad(mll_optim)(params)
            grads = grads * scales
            updates, opt_state = optimizer.update(grads, opt_state)
            u_params = apply_updates(u_params, updates)
            u_params = jnp.clip(u_params, 0.,1.)
            carry = u_params, opt_state
            return carry, loss
        
        best_f = jnp.inf
        best_params = None
        u_params = scale_to_unit(init_params,bounds.T)
        # display with progress bar
        r = jnp.arange(maxiter)
        for n in range(n_restarts):
            opt_state = optimizer.init(u_params)
            progress_bar = tqdm.tqdm(r,desc=f'Training GP')
            for i in progress_bar:
                (u_params,opt_state), fval  = step((u_params,opt_state))#,None)
                progress_bar.set_postfix({"fval": float(fval)})
            if fval < best_f:
                best_f = fval
                best_params = u_params
            u_params = jnp.clip(u_params + 0.25*np.random.normal(size=init_params.shape),0,1)
        params = scale_from_unit(best_params,bounds.T)
        hyperparams = 10 ** params
        self.tausq = hyperparams[0]
        self.lengthscales = hyperparams[1:-1]
        self.outputscale = hyperparams[-1]
        log.info(f"Final hyperparams: tausq = {self.tausq}, lengthscales = {self.lengthscales}, outputscale = {self.outputscale}")
        k = self.kernel(self.train_x,self.train_x,self.lengthscales,self.outputscale,noise=self.noise,include_noise=True)
        self.cholesky = jnp.linalg.cholesky(k)
        self.fitted = True

    # def update(self,new_x,new_y,refit=True,lr=1e-2,maxiter=200,n_restarts=2):
    #     """
    #     Updates the GP with new training points and refits the GP if refit is True.

    #     Arguments
    #     ---------        
    #     refit: bool
    #         Whether to refit the GP hyperparameters. Default is True.
    #     lr: float
    #         The learning rate for the optax optimizer. Default is 1e-2.
    #     maxiter: int
    #         The maximum number of iterations for the optax optimizer. Default is 250.
    #     n_restarts: int
    #         The number of restarts for the optax optimizer. Default is 2.

    #     Returns
    #     -------
    #     repeat: bool
    #         Whether the point new_x, new_y already exists in the training set.

    #     """
    #     if jnp.any(jnp.all(jnp.isclose(self.train_x, new_x, atol=1e-6,rtol=1e-4), axis=1)):
    #         log.info(f"Point {new_x} already exists in the training set, not updating")
    #         return True
    #     else:
    #         super().add(new_x,new_y)
    #         if refit:
    #             self.fit(lr=lr,maxiter=maxiter,n_restarts=n_restarts)
    #         else:
    #             # consider doing rank 1 update of cholesky
    #             k = self.kernel(self.train_x,self.train_x,self.lengthscales,self.outputscale,noise=self.noise,include_noise=True)
    #             self.cholesky = jnp.linalg.cholesky(k)
    #         return False

    
    def save(self, outfile='gp'):
        """
        Saves the SAAS_GP to a file
        
        Arguments
        ---------
        outfile: str
            The name of the file to save the GP to. Default is 'gp'.
        """
        np.savez(f'{outfile}.npz', train_x=self.train_x, train_y=self.train_y, 
                 noise=self.noise,y_mean=self.y_mean, y_std=self.y_std, 
                 lengthscales=self.lengthscales, outputscale=self.outputscale,
                 tausq=self.tausq)

    @classmethod
    def load(cls, filename, kernel="rbf", optimizer="adam", 
             outputscale_bounds=[-4,4], lengthscale_bounds=[np.log10(0.05),2],
             tausq_bounds=[-4,4]):
        """
        Loads a SAAS_GP from a file
        
        Arguments
        ---------
        filename: str
            The name of the file to load the GP from (with or without .npz extension)
        kernel: str
            Kernel type for the GP. Default is 'rbf'. Can be 'rbf' or 'matern'.
        optimizer: str
            Optimizer type for the GP. Default is 'adam'.
        outputscale_bounds: list
            Bounds for the output scale of the GP (in log10 space). Default is [-4,4].
        lengthscale_bounds: list
            Bounds for the length scale of the GP (in log10 space). Default is [np.log10(0.05),2].
        tausq_bounds: list
            Bounds for the tausq parameter of the GP (in log10 space). Default is [-4,4].
            
        Returns
        -------
        gp: SAAS_GP
            The loaded SAAS_GP object
        """
        if not filename.endswith('.npz'):
            filename += '.npz'
            
        data = np.load(filename)
        
        # Extract training data
        train_x = jnp.array(data['train_x'])
        train_y = jnp.array(data['train_y'])
        
        # Extract hyperparameters
        noise = float(data['noise'])
        
        # Create SAAS_GP instance
        gp = cls(train_x, train_y, noise=noise, kernel=kernel, optimizer=optimizer,
                 outputscale_bounds=outputscale_bounds, lengthscale_bounds=lengthscale_bounds,
                 tausq_bounds=tausq_bounds)
        
        # Restore saved parameters
        gp.y_mean = jnp.array(data['y_mean'])
        gp.y_std = jnp.array(data['y_std'])
        gp.lengthscales = jnp.array(data['lengthscales'])
        gp.outputscale = float(data['outputscale'])
        gp.tausq = float(data['tausq'])
        
        # Recompute kernel matrix and Cholesky decomposition
        gp.K = gp.kernel(gp.train_x, gp.train_x, gp.lengthscales, gp.outputscale, 
                        noise=gp.noise, include_noise=True)
        gp.cholesky = jnp.linalg.cholesky(gp.K)
        gp.alphas = cho_solve((gp.cholesky, True), gp.train_y)
        gp.fitted = True
        
        log.info(f"Loaded SAAS_GP from {filename} with {train_x.shape[0]} training points")
        return gp
    
jax.tree_util.register_pytree_node(
    SAAS_GP,
    SAAS_GP.tree_flatten,
    SAAS_GP.tree_unflatten,
)




def load_gp(filename, gp_type="auto", **kwargs):
    """
    Utility function to load a GP from a file, automatically detecting the GP type if not specified
    
    Arguments
    ---------
    filename: str
        The name of the file to load the GP from (with or without .npz extension)
    gp_type: str
        The type of GP to create. Can be 'auto', 'DSLP', 'SAAS'. If 'auto', attempts to detect
        the type based on the saved parameters. Default is 'auto'.
    **kwargs: 
        Additional keyword arguments to pass to the GP constructor
        
    Returns
    -------
    gp: GP
        The loaded GP object (DSLP_GP or SAAS_GP)
    """
    if not filename.endswith('.npz'):
        filename += '.npz'
        
    try:
        data = np.load(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file {filename}")
    
    if gp_type == "auto":
        # Try to detect GP type based on saved parameters
        if 'tausq' in data.files:
            gp_type = "SAAS"
            log.info("Auto-detected SAAS_GP from saved parameters")
        else:
            gp_type = "DSLP" 
            log.info("Auto-detected DSLP_GP from saved parameters")
    
    if gp_type.upper() == "SAAS":
        return SAAS_GP.load(filename, **kwargs)
    elif gp_type.upper() == "DSLP":
        return DSLP_GP.load(filename, **kwargs)
    else:
        raise ValueError(f"Unknown GP type: {gp_type}. Must be 'DSLP', 'SAAS', or 'auto'")
       