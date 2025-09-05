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
import numpyro.distributions as dist
from numpyro.infer.initialization import init_to_value, init_to_sample
from numpyro.util import enable_x64
enable_x64()
from functools import partial
from .utils.logging_utils import get_logger
log = get_logger("gp")
from .optim import optimize_optax, optimize_scipy
from .utils.seed_utils import get_new_jax_key, get_numpy_rng

safe_noise_floor = 1e-12

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
    """
    Base class for the GP with no hyperparameter priors.
    """
    hyperparam_priors: str = 'uniform'

    def __init__(self,train_x,train_y,noise=1e-6,kernel="rbf"
                 ,optimizer="optax",optimizer_kwargs={'lr': 5e-3, 'name': 'adam'}
                 ,kernel_variance_bounds = [1e-4,1e8],lengthscale_bounds = [0.05,100]
                 ,lengthscales=None,kernel_variance=None):
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
        kernel_variance_bounds: list
            Bounds for the kernel variance in actual space, default [1e-4, 1e8]
        lengthscale_bounds: list  
            Bounds for the lengthscales in actual space, default [0.05, 100]
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
        
        log.debug(f"GP training size = {self.train_x.shape[0]}")

        self.ndim = train_x.shape[1]
        self.y_mean = jnp.mean(train_y)
        self.y_std = jnp.std(train_y)
        self.kernel_name = kernel if kernel=="rbf" else "matern"
        self.kernel = rbf_kernel if kernel=="rbf" else matern_kernel
        self.noise = noise
        self.optimizer_method = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        if optimizer == 'scipy':
            self.mll_optimize = optimize_scipy
        else:
            self.mll_optimize = optimize_optax

        log.debug(f"GP y_mean: {self.y_mean:.4f}, y_std: {self.y_std:.4f}, noise: {self.noise:.2e}")

        self.train_y = (train_y - self.y_mean) / self.y_std
        self.lengthscales = jnp.ones(self.ndim) if lengthscales is None else jnp.array(lengthscales)
        self.kernel_variance = 1. if kernel_variance is None else kernel_variance
        
        # Convert actual bounds to log10 space for internal use
        self.lengthscale_bounds_actual = lengthscale_bounds
        self.kernel_variance_bounds_actual = kernel_variance_bounds
        self.lengthscale_bounds = [jnp.log10(lengthscale_bounds[0]), jnp.log10(lengthscale_bounds[1])]
        self.kernel_variance_bounds = [jnp.log10(kernel_variance_bounds[0]), jnp.log10(kernel_variance_bounds[1])]

        self.hyperparam_bounds = [self.lengthscale_bounds]*self.ndim + [self.kernel_variance_bounds]
        self.hyperparam_bounds = jnp.array(self.hyperparam_bounds).T # shape (2, D+1)
        log.debug(f" Hyperparameter bounds (log10) =  {self.hyperparam_bounds}")
        log.debug(f" Actual lengthscale bounds =  {self.lengthscale_bounds_actual}")
        log.debug(f" Actual kernel variance bounds =  {self.kernel_variance_bounds_actual}")

        self.cholesky = jnp.linalg.cholesky(self.kernel(self.train_x, self.train_x, self.lengthscales, self.kernel_variance, noise=self.noise, include_noise=True))
        self.alphas = cho_solve((self.cholesky, True), self.train_y)
        self.fitted = True

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

    def predict_mean(self,x):
        """
        Predicts the mean of the GP at x and unstandardizes it
        """
        x = jnp.atleast_2d(x)
        k12 = self.kernel(self.train_x,x,self.lengthscales,self.kernel_variance,noise=self.noise,include_noise=False)
        mean = get_mean_from_cho(k12,self.alphas) 
        return mean*self.y_std + self.y_mean 

    def predict_var(self,x):
        """
        Predicts the variance of the GP at x and unstandardizes it
        """
        x = jnp.atleast_2d(x)
        k12 = self.kernel(self.train_x,x,self.lengthscales,self.kernel_variance,noise=self.noise,include_noise=False)
        k22 = self.kernel(x,x,self.lengthscales,self.kernel_variance,noise=self.noise,include_noise=True)
        var = jnp.clip(get_var_from_cho(self.cholesky,k12,k22), safe_noise_floor, None)
        return var*self.y_std**2

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

    def update(self,new_x,new_y,refit=True,maxiter=300,n_restarts=4):
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
        self.train_y = self.train_y*self.y_std + self.y_mean 
        self.train_y = jnp.concatenate([self.train_y,new_y])
        self.y_mean = jnp.mean(self.train_y.flatten(),axis=0)
        self.y_std = jnp.std(self.train_y.flatten(),axis=0)
        self.train_y = (self.train_y - self.y_mean) / self.y_std


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

    def neg_mll(self,log10_params):
        hyperparams = 10**log10_params
        lengthscales = hyperparams[0:-1]
        kernel_variance = hyperparams[-1]
        k = self.kernel(self.train_x,self.train_x,lengthscales,kernel_variance,noise=self.noise,include_noise=True)
        val = gp_mll(k,self.train_y,self.train_y.shape[0])
        return -val

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
            init_params_u = jnp.vstack([init_params_u, addn_init_params])
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
        self.fitted = True

    def save(self,outfile='gp'):
        """
        Saves the GP to a file
        """
        train_y = self.train_y * self.y_std + self.y_mean  # unstandardize the training targets
        np.savez(f'{outfile}.npz',train_x=self.train_x,train_y=train_y,noise=self.noise,
         lengthscales=self.lengthscales,kernel_variance=self.kernel_variance,hyperparam_priors=self.hyperparam_priors,
         lengthscale_bounds_actual=self.lengthscale_bounds_actual,kernel_variance_bounds_actual=self.kernel_variance_bounds_actual)

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
        
        # Extract data from the file
        train_x = jnp.array(data['train_x'])
        train_y = jnp.array(data['train_y'])  # This is unstandardized
        noise = float(data['noise'])
        lengthscales = jnp.array(data['lengthscales']) if 'lengthscales' in data.files else None
        kernel_variance = float(data['kernel_variance']) if 'kernel_variance' in data.files else None
        optimizer = str(data['optimizer']) if 'optimizer' in data.files else "optax"
        optimizer_kwargs = dict(data['optimizer_kwargs']) if 'optimizer_kwargs' in data.files else {"name": "adam", "lr": 1e-3}
        
        # Load bounds - handle both old (log10) and new (actual) formats
        if 'lengthscale_bounds_actual' in data.files:
            lengthscale_bounds = data['lengthscale_bounds_actual'].tolist()
        else:
            # Legacy: convert from log10 bounds to actual bounds
            old_bounds = kwargs.get('lengthscale_bounds', [np.log10(0.05), 2])
            lengthscale_bounds = [10**old_bounds[0], 10**old_bounds[1]]
            
        if 'kernel_variance_bounds_actual' in data.files:
            kernel_variance_bounds = data['kernel_variance_bounds_actual'].tolist()
        else:
            # Legacy: convert from log10 bounds to actual bounds
            old_bounds = kwargs.get('kernel_variance_bounds', [-4, 8])
            kernel_variance_bounds = [10**old_bounds[0], 10**old_bounds[1]]
        
        # Create GP instance - it will automatically standardize train_y and compute cholesky/alphas
        gp = cls(train_x=train_x, train_y=train_y, noise=noise, 
                lengthscales=lengthscales, kernel_variance=kernel_variance, optimizer=optimizer, optimizer_kwargs=optimizer_kwargs, 
                lengthscale_bounds=lengthscale_bounds, kernel_variance_bounds=kernel_variance_bounds, **kwargs)

        log.info(f"Loaded GP from {filename} with {train_x.shape[0]} training points")
        return gp

    def get_random_point(self,rng=None):


        rng = rng if rng is not None else get_numpy_rng()

        pt = rng.uniform(0, 1, size=self.train_x.shape[1])

        return pt

    def sample_GP_NUTS(self,warmup_steps=256,num_samples=512,progress_bar=True,thinning=8,verbose=True,
                       init_params=None,temp=1.,restart_on_flat_logp=True,num_chains=2, np_rng=None, rng_key=None):

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

            mean = self.predict_mean_batched(x)
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
        # num_parallel_chains = min(num_devices,num_chains)

        rng_key = rng_key if rng_key is not None else get_new_jax_key()
        rng_keys = jax.random.split(rng_key, num_chains)

        log.info(f"Running MCMC with {num_chains} chains on {num_devices} devices.")

        if (num_devices >= num_chains) and num_chains > 1:
            # if devices present run with pmap
            pmapped = jax.pmap(run_single_chain, in_axes=(0,),out_axes=(0,0))
            samples_x, logps = pmapped(rng_keys)
            # reshape to get proper shapes
            samples_x = jnp.concatenate(samples_x, axis=0)
            logps = jnp.reshape(logps, (samples_x.shape[0],))
            # log.info(f"Xs shape: {samples_x.shape}, logps shape: {logps.shape}")
        else:
            # run sequentially
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
    
    def copy(self):
        """
        Returns a deep copy of the GP with the same training data, hyperparameters,
        and fitted state. The copy is independent: modifications to the new GP do 
        not affect the original.
        """
        train_y= self.train_y * self.y_std + self.y_mean

        gp_copy = GP(
            train_x=self.train_x,
            train_y=train_y,
            noise=float(self.noise),
            kernel=self.kernel_name,
            optimizer="adam",  # or pass through if you extend GP further
            kernel_variance_bounds=self.kernel_variance_bounds,
            lengthscale_bounds=self.lengthscale_bounds,
            lengthscales=jnp.array(self.lengthscales, copy=True),
            kernel_variance=float(self.kernel_variance)
        )

        gp_copy.alphas = jnp.array(self.alphas, copy=True)
        gp_copy.cholesky = jnp.array(self.cholesky, copy=True)
        gp_copy.fitted = self.fitted

        return gp_copy

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
        """
        Returns the number of training points.
        """
        return self.train_x.shape[0]
    

    


class DSLP_GP(GP):

    hyperparam_priors: str = 'dslp'

    def __init__(self,train_x,train_y,noise=1e-8,kernel="rbf",optimizer="optax",optimizer_kwargs={'lr': 1e-3, 'name': 'adam'},
                 kernel_variance_bounds = [1e-4,1e8],lengthscale_bounds = [0.05,100],lengthscales=None,kernel_variance=None):
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
            Default is 'optax'. This is the optimizer type for the GP.
        kernel_variance_bounds: Bounds for the kernel variance in actual space
            Default is [1e-4, 1e8]. These are the bounds for the kernel variance of the GP.
        lengthscale_bounds: Bounds for the length scale in actual space
            Default is [0.05, 100]. These are the bounds for the length scale of the GP.
        """
        super().__init__(train_x,train_y,noise,kernel,optimizer,optimizer_kwargs,
                         kernel_variance_bounds,lengthscale_bounds,lengthscales=lengthscales,kernel_variance=kernel_variance)

    def neg_mll(self,log10_params):
        hyperparams = 10**log10_params
        lengthscales = hyperparams[0:-1]
        kernel_variance = hyperparams[-1]
        logprior = dist.LogNormal(0.,0.5).log_prob(kernel_variance) #
        # logprior = dist.Gamma(2.0,0.15).log_prob(kernel_variance)
        logprior+= dist.LogNormal(loc=sqrt2 + 0.5*jnp.log(self.ndim) ,scale=sqrt3).expand([self.ndim]).log_prob(lengthscales).sum()
        return super().neg_mll(log10_params) - logprior   

class SAAS_GP(GP):

    hyperparam_priors: str = 'saas'

    def __init__(self,
                 train_x, train_y, noise=1e-8, kernel="rbf", 
                 optimizer="optax",optimizer_kwargs={'lr': 1e-3, 'name': 'adam'},
                 kernel_variance_bounds = [1e-4,1e8],lengthscale_bounds = [0.05,100],
                 tausq_bounds = [1e-4,1e4],lengthscales=None,kernel_variance=None,tausq=None):
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
            Default is 'optax'. This is the optimizer type for the GP.
        kernel_variance_bounds: Bounds for the kernel variance in actual space
            Default is [1e-4, 1e8]. These are the bounds for the kernel variance of the GP.
        lengthscale_bounds: Bounds for the length scale in actual space
            Default is [0.05, 100]. These are the bounds for the length scale of the GP.
        tausq_bounds: Bounds for the tausq parameter in actual space
            Default is [1e-4, 1e4]. These are the bounds for the tausq parameter of the GP.
        """
        super().__init__(train_x, train_y, noise, kernel, optimizer,optimizer_kwargs,kernel_variance_bounds,lengthscale_bounds,lengthscales=lengthscales,kernel_variance=kernel_variance)
        self.tausq = tausq if tausq is not None else 1.0
        
        # Convert actual tausq bounds to log10 space for internal use
        self.tausq_bounds_actual = tausq_bounds
        self.tausq_bounds = [np.log10(tausq_bounds[0]), np.log10(tausq_bounds[1])]
        self.hyperparam_bounds = jnp.vstack([self.hyperparam_bounds.T, self.tausq_bounds]).T # shape (2, D+2)
        log.debug(f'HP bounds shape: {self.hyperparam_bounds.shape}')
        log.debug(f'Actual tausq bounds: {self.tausq_bounds_actual}')

    def neg_mll(self, log10_params):
        hyperparams = 10**log10_params
        lengthscales = hyperparams[:self.ndim]
        kernel_variance = hyperparams[self.ndim]
        tausq = hyperparams[-1]
        logprior = dist.LogNormal(0.,1.).log_prob(kernel_variance)
        # logprior = dist.Gamma(2.0,0.15).log_prob(kernel_variance)
        logprior+= dist.HalfCauchy(0.1).log_prob(tausq)
        inv_lengthscales_sq = 1/ (tausq * lengthscales**2)
        logprior+= jnp.sum(dist.HalfCauchy(1.).log_prob(inv_lengthscales_sq))
        return super().neg_mll(log10_params[:-1]) - logprior

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
        init_params = jnp.log10(jnp.concatenate([self.lengthscales  , jnp.array([self.kernel_variance, self.tausq])]))
        init_params_u = scale_to_unit(init_params, self.hyperparam_bounds)
        if n_restarts>1:
            addn_init_params = init_params_u + np.random.normal(size=(n_restarts-1, init_params.shape[0]))
            init_params_u = np.vstack([init_params_u, addn_init_params])
        x0 = jnp.clip(init_params_u, 0.0, 1.0)
        log.info(f"Fitting GP with initial params lengthscales = {self.lengthscales}, kernel_variance = {self.kernel_variance}, tausq = {self.tausq}")

        optimizer_kwargs = self.optimizer_kwargs.copy()

        best_params, best_f = self.mll_optimize(
            fun=self.neg_mll,
            ndim=self.ndim + 2,
            bounds=self.hyperparam_bounds,
            x0=x0,
            maxiter=maxiter,
            n_restarts=n_restarts,
            optimizer_kwargs=optimizer_kwargs
        )

        hyperparams = 10 ** best_params
        self.lengthscales = hyperparams[:self.ndim]
        self.kernel_variance = hyperparams[self.ndim]
        self.tausq = hyperparams[-1]
        log.info(f"Final hyperparams: lengthscales = {self.lengthscales}, kernel_variance = {self.kernel_variance}, tausq = {self.tausq}, final MLL = {-best_f}")
        K = self.kernel(self.train_x, self.train_x, self.lengthscales, self.kernel_variance, noise=self.noise, include_noise=True)
        self.cholesky = jnp.linalg.cholesky(K)
        self.alphas = cho_solve((self.cholesky, True), self.train_y)
        self.fitted = True

    def save(self,outfile='gp'):
        """
        Saves the SAAS_GP to a file
        """
        train_y = self.train_y * self.y_std + self.y_mean  # unstandardize the training targets
        np.savez(f'{outfile}.npz',train_x=self.train_x,train_y=train_y,noise=self.noise,
         lengthscales=self.lengthscales,kernel_variance=self.kernel_variance,tausq=self.tausq,hyperparam_priors=self.hyperparam_priors,
         lengthscale_bounds_actual=self.lengthscale_bounds_actual,kernel_variance_bounds_actual=self.kernel_variance_bounds_actual,
         tausq_bounds_actual=self.tausq_bounds_actual)



    @classmethod
    def load(cls, filename, **kwargs):
        """
        Loads a SAAS_GP from a file
        
        Arguments
        ---------
        filename: str
            The name of the file to load the GP from (with or without .npz extension)
        **kwargs: 
            Additional keyword arguments to pass to the GP constructor
            
        Returns
        -------
        gp: SAAS_GP
            The loaded SAAS_GP object
        """
        if not filename.endswith('.npz'):
            filename += '.npz'
            
        try:
            data = np.load(filename)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file {filename}")
        
        # Extract data from the file
        train_x = jnp.array(data['train_x'])
        train_y = jnp.array(data['train_y'])  # This is unstandardized
        noise = float(data['noise'])
        lengthscales = jnp.array(data['lengthscales']) if 'lengthscales' in data.files else None
        kernel_variance = float(data['kernel_variance']) if 'kernel_variance' in data.files else None
        tausq = float(data['tausq']) if 'tausq' in data.files else None
        
        # Create SAAS_GP instance - it will automatically standardize train_y and compute cholesky/alphas
        gp = cls(train_x=train_x, train_y=train_y, noise=noise, 
                lengthscales=lengthscales, kernel_variance=kernel_variance, tausq=tausq, **kwargs)
        
        log.info(f"Loaded SAAS_GP from {filename} with {train_x.shape[0]} training points")
        return gp

def load_gp(filename, gp_type="auto", **kwargs):
    """
    Utility function to load a GP from a file, automatically detecting the GP type if not specified
    
    Arguments
    ---------
    filename: str
        The name of the file to load the GP from (with or without .npz extension)
    gp_type: str
        The type of GP to create. Can be 'auto', 'GP', 'DSLP', 'SAAS'. If 'auto', attempts to detect
        the type based on the saved parameters. Default is 'auto'.
    **kwargs: 
        Additional keyword arguments to pass to the GP constructor
        
    Returns
    -------
    gp: GP
        The loaded GP object (GP, DSLP_GP, or SAAS_GP)
    """
    if not filename.endswith('.npz'):
        filename += '.npz'
        
    try:
        data = np.load(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file {filename}")
    
    if gp_type == "auto":
        # Try to detect GP type based on saved parameters
        if 'hyperparam_priors' in data.files:
            hyperparam_priors = str(data['hyperparam_priors'].item()).lower()
            if hyperparam_priors == 'saas':
                gp_type = "SAAS"
                log.info("Auto-detected SAAS_GP from hyperparam_priors")
            elif hyperparam_priors == 'dslp':
                gp_type = "DSLP"
                log.info("Auto-detected DSLP_GP from hyperparam_priors")
            elif hyperparam_priors == 'uniform':
                gp_type = "GP"
                log.info("Auto-detected GP (uniform priors) from hyperparam_priors")
            else:
                gp_type = "GP"
                log.info(f"Unknown hyperparam_priors '{hyperparam_priors}', defaulting to GP")
        elif 'tausq' in data.files:
            # Fallback: if tausq exists but no hyperparam_priors, assume SAAS
            gp_type = "SAAS"
            log.info("Auto-detected SAAS_GP from presence of tausq parameter")
        else:
            # Fallback: no clear indicators, default to DSLP for backward compatibility
            gp_type = "DSLP" 
            log.info("No clear indicators, defaulting to DSLP_GP for backward compatibility")
    
    if gp_type.upper() == "GP":
        return GP.load(filename, **kwargs)
    elif gp_type.upper() == "DSLP":
        return DSLP_GP.load(filename, **kwargs)
    elif gp_type.upper() == "SAAS":
        return SAAS_GP.load(filename, **kwargs)
    else:
        raise ValueError(f"Unknown GP type: {gp_type}. Must be 'GP', 'DSLP', 'SAAS', or 'auto'")
       