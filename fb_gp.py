# This module implements the FullyBayesian Gaussian Process model for Bayesian optimization. 
# The FullyBayesian GP is based on Eriksson, D. and Jankowiak, M., “High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces” (2021), arXiv:2103.00349, 
# doi: 10.1080/00401706.2018.1469433 (see also SAASBO on GitHub) and the BoTorch FullyBayesianGP with minor changes

from math import sqrt
import time
from typing import Any,List
import numpyro
import numpyro.distributions as dist
from numpyro.util import enable_x64
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, util
from numpyro.infer.initialization import init_to_value, init_to_sample
from .bo_utils import split_vmap
from jax import config
from jax.lax import map
from sympy import false
config.update("jax_enable_x64", True)
from numpyro.util import enable_x64
enable_x64()
from functools import partial
import logging
log = logging.getLogger("[FBGP]")
import multiprocessing
from .gp import GP, get_mean_from_cho, get_var_from_cho, gp_predict
global_num_chains = multiprocessing.cpu_count()

# todo
# 1. method to reuse previous mcmc samples if HMC runs into issues
# 2. matern needs to be fixed
# 3. test speed vs standard botorch
# 4. stability of cholesky when points are very close together 
# 5. Pytree for GP?

sqrt5 = jnp.sqrt(5.)

def dist_sq(x, y):
    """
    Compute squared Euclidean distance between two points x, y. 
    If x is n1 x d and y is n2 x d returns a n1 x n2 matrix of distancess.
    """
    return jnp.sum(jnp.square(x[:,None,:] - y),axis=-1) 

@partial(jit,static_argnames='include_noise')
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

@partial(jit,static_argnames='include_noise')
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

@jit
def get_var_from_cho(k11_cho,k12,k22):
    vv = solve_triangular(k11_cho,k12,lower=True)
    var = jnp.diag(k22) - jnp.sum(vv*vv,axis=0) # replace k22 computation with single element diagonal
    return var
    
@jit   # move back to gp and use pytree
def get_mean_from_cho(k11_cho,k12,train_y):
    mu = jnp.matmul(jnp.transpose(k12),cho_solve((k11_cho,True),train_y)) # can also store alphas
    mean = mu[:,0]  
    return mean

def sample_GP_NUTS(gp,rng_key,warmup_steps=512,num_samples=512,progress_bar=True,thinning=2,verbose=False
                   ,init_from_max=True):
    """
    Obtain samples from the posterior represented by the GP mean as the logprob

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
        Thinning factor for the MCMC samples, default 2
    verbose: bool
        If True, prints the MCMC summary, default False
    init_params: dict
        Initial parameters for the MCMC, default None   
    """
    class gp_dist(dist.Distribution):
        support = dist.constraints.real

        def __init__(self,gp):
            super().__init__(batch_shape = (), event_shape=())
            self.gp = gp

        def sample(self, key, sample_shape=()):
            raise NotImplementedError
    
        def log_prob(self,x):
            val = gp.predict_mean(x)
            return val

    def model(train_x):
        x = numpyro.sample('x',dist.Uniform
                           (low=jnp.zeros(train_x.shape[1]),high=jnp.ones(train_x.shape[1]))) # type: ignore
        numpyro.sample('y',gp_dist(gp),obs=x)

    if init_from_max: 
        params = {'x': gp.train_x[jnp.argmax(gp.train_y)]} #, 'y': jnp.max(gp.train_y)} # {'x': 
        init_strategy = init_to_value(values=params)
        # init_params = util.unconstrain_fn(model,model_args=gp.train_x,model_kwargs=None,params=params)
        # print(init_params)
    else:
        init_strategy = init_to_sample
    start = time.time()
    kernel = NUTS(model,dense_mass=False,
                max_tree_depth=6,init_strategy=init_strategy)
    mcmc = MCMC(kernel,num_warmup=warmup_steps,
                num_samples=num_samples,
                num_chains=1,
                progress_bar=progress_bar,
                thinning=thinning,)
    mcmc.run(rng_key,gp.train_x,extra_fields=("potential_energy",))
    if verbose:
        mcmc.print_summary(exclude_deterministic=False)
    log.info(f" Sampled parameters MCMC took {time.time()-start:.4f} s")
    samples = {}
    samples['x'] = mcmc.get_samples()['x'] 
    logp = mcmc.get_extra_fields()['potential_energy']
    samples['logp'] = logp
    return samples


class numpyro_model:
   """
   Class for the numpyro model used within the Fully Bayesian GP, this is only called internally
   """
   
   # Note - train_x and train_y received here are already transformed, npoints x ndim and npoints x 1
   def __init__(self, train_x,  train_y, #train_yvar, 
                kernel_func=rbf_kernel,noise=1e-8) -> None:
      self.train_x = train_x 
      self.train_y = train_y
      self.ndim = train_x.shape[-1]
      self.npoints = train_x.shape[-2]
      self.kernel_func = kernel_func
      self.noise = noise
 
   
   def model(self):
        # add option to use SAAS priors on certain lengthscales only?
        outputscale = numpyro.sample("kernel_var", dist.Gamma(concentration=2.,rate=0.15))
        tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(0.1))
        inv_length_sq = numpyro.sample("_kernel_inv_length_sq",dist.HalfCauchy(jnp.ones(self.ndim))) # type: ignore
        lengthscales = numpyro.deterministic("kernel_length",1/jnp.sqrt(tausq*inv_length_sq)) # type: ignore
        k = self.kernel_func(self.train_x,self.train_x,lengthscales,outputscale,noise=self.noise,include_noise=True) 
        mll = numpyro.sample(
                "Y",
                dist.MultivariateNormal(
                    loc=jnp.zeros(self.train_x.shape[0]), # type: ignore
                    covariance_matrix=k,
                ),
                obs=self.train_y.squeeze(-1),)

    # add method to start from previous samples

    # how can we speed up the MCMC when we already have a large number of samples? main bottleneck -> inversion of kernel
   def run_mcmc(self,rng_key,dense_mass=True,max_tree_depth=6,
                warmup_steps=512,num_samples=512,num_chains=1,thinning=16,
                progbar=True,verbose=False,init_params=None,
                ) -> dict:
        start = time.time()
        kernel = NUTS(self.model,
                dense_mass=dense_mass,
                max_tree_depth=max_tree_depth)
        mcmc = MCMC(
                kernel,
                num_warmup=warmup_steps,
                num_samples=num_samples,
                num_chains=1,
                progress_bar=progbar,
                thinning=thinning,
                )
        #https://forum.pyro.ai/t/initialize-mcmc-chains-from-multiple-predetermined-starting-points/5062
        if init_params is not None:
            init_params = util.unconstrain_fn(self.model,model_args=self.train_x,model_kwargs=None,params=init_params)
        mcmc.run(rng_key,extra_fields=("potential_energy",),init_params=init_params)
        if verbose:
            mcmc.print_summary(exclude_deterministic=False)
        extras = mcmc.get_extra_fields()
        log.info(f" Hyperparameters MCMC elapsed time: {time.time() - start:.2f}s")

        return mcmc.get_samples(), extras # type: ignore

    # add method to start from previous map hyperparams
   def fit_gp_NUTS(self,rng_key,dense_mass=True,max_tree_depth=6,
                warmup_steps=512,num_samples=512,num_chains=1,progbar=True,thinning=16,verbose=False):
        samples, extras = self.run_mcmc(rng_key=rng_key,
                dense_mass=dense_mass,
                max_tree_depth=max_tree_depth,
                warmup_steps=warmup_steps,
                num_samples=num_samples,
                num_chains=num_chains,
                progbar=progbar,
                thinning=thinning,
                verbose=verbose)
        samples["minus_log_prob"] = extras["potential_energy"]  # see also numpyro.infer.util.log_likelihood
        del samples["kernel_tausq"], samples["_kernel_inv_length_sq"]
        # numpyro already thins samples and keeps deterministic params
        return samples
      
   def update(self,train_x,train_y):
       self.train_x = train_x
       self.train_y = train_y


class SAAS_FBGP(GP):
    """
    Main class for the Fully Bayesian GP.

    Arguments
    ---------
    train_x: jnp.ndarray
        Training inputs, size (N x D)
    train_y: jnp.ndarray
        Objective function values at training points size (N x 1)
    standardise_y: bool
        Standardise the output values to have zero mean and unit variance, default True
    noise: float
        Noise parameter to add to the diagonal of the kernel, default 1e-8
    kernel: str
        Kernel to use, either "rbf" or "matern"
    vmap_size: int
        Batch size for vmap, useful for managing memory, default 8
    sample_lengthscales: jnp.ndarray
        Pre-sampled lengthscales, default None
    sample_outputscales: jnp.ndarray
        Pre-sampled output scales, default None
    warmup_steps: int
        Number of warmup steps for NUTS, default 256
    num_samples: int
        Number of samples to draw from the posterior, default 256
    thinning: int
        Thinning factor for the MCMC samples, default 16
    dense_mass: bool
        Use dense mass matrix for NUTS, default True
    max_tree_depth: int
        Maximum tree depth for NUTS, default 6
    num_chains: int
        Number of chains to run in parallel, default 1
    min_lengthscale: float
        Minimum lengthscale value, default 1e-3
    max_lengthscale: float
        Maximum lengthscale value, default 1e2
    """

    def __init__(self
                 ,train_x
                 ,train_y
                 ,noise=1e-8
                 ,kernel="rbf"
                 ,vmap_size=8
                 ,sample_lengthscales=None
                 ,sample_outputscales=None
                 ,warmup_steps:int = 512
                 ,num_samples:int = 512
                 ,thinning:int = 32
                 ,dense_mass: bool = False
                 ,max_tree_depth:int = 6
                 ,num_chains: int = 1
                 ,min_lengthscale: float = 1e-2
                 ,max_lengthscale: float = 1e2) -> None:

        # check train_x and train_y dims are N x D and N x 1, param_bounds are 2 x d
        self.ndim = train_x.shape[-1]
        self.train_x = train_x
        self.y_mean = jnp.mean(train_y,axis=0)
        self.y_std = jnp.std(train_y,axis=0)
        self.train_y = (train_y - self.y_mean) / self.y_std

        self.noise = noise
        self.fitted = False
        self.num_samples = 0
        self.vmap_size = vmap_size # to process vmap in vmap_size batches
        self.kernel_func = rbf_kernel if kernel=="rbf" else matern_kernel
        self.warmup_steps=warmup_steps
        self.num_samples=num_samples
        self.thinning=thinning
        self.dense_mass = dense_mass
        self.max_tree_depth = max_tree_depth
        self.num_chains = num_chains
        self.min_lengthscale = min_lengthscale
        self.max_lengthscale = max_lengthscale

        dummy_lengthscales = jnp.ones((self.num_samples//self.thinning,self.train_x.shape[-1]))
        dummy_outputscales = jnp.ones((self.num_samples//self.thinning,))
        self.samples = {"kernel_length": dummy_lengthscales,
                        "kernel_var": dummy_outputscales}
        if sample_lengthscales is not None and sample_outputscales is not None:
            self.fitted=True
            self.samples["kernel_length"] = sample_lengthscales
            self.samples["kernel_var"] = sample_outputscales
            # self.num_samples = len(sample_outputscales)


    def tree_flatten(self):
        """
        Returns the flattened tree of the GP object
        """
        # dynamic attributes, similar to vanilla GP
        leaves = (self.train_x, self.train_y, self.cholesky, self.y_mean, self.y_std, self.samples)
        # static attributes
        # Collect static (auxiliary) attributes.
        aux_data = {
            "ndim": self.ndim,
            "noise": self.noise,
            "fitted": self.fitted,
            "vmap_size": self.vmap_size,
            "kernel_func": self.kernel_func,
            "warmup_steps": self.warmup_steps,
            "num_samples": self.num_samples,
            "thinning": self.thinning,
            "dense_mass": self.dense_mass,
            "max_tree_depth": self.max_tree_depth,
            "num_chains": self.num_chains,
            "min_lengthscale": self.min_lengthscale,
            "max_lengthscale": self.max_lengthscale,
        }
        return leaves, aux_data


    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        # The leaves are unpacked in the same order they were flattened.
        train_x, train_y, cholesky, y_mean, y_std, samples = leaves
        obj = cls.__new__(cls)
        obj.train_x = train_x
        obj.train_y = train_y
        obj.y_mean = y_mean
        obj.y_std = y_std
        obj.cholesky = cholesky
        obj.samples = samples
        obj.ndim = aux_data["ndim"]
        obj.noise = aux_data["noise"]
        obj.fitted = aux_data["fitted"]
        obj.vmap_size = aux_data["vmap_size"]
        obj.kernel_func = aux_data["kernel_func"]
        obj.warmup_steps = aux_data["warmup_steps"]
        obj.num_samples = aux_data["num_samples"]
        obj.thinning = aux_data["thinning"]
        obj.dense_mass = aux_data["dense_mass"]
        obj.max_tree_depth = aux_data["max_tree_depth"]
        obj.num_chains = aux_data["num_chains"]
        obj.min_lengthscale = aux_data["min_lengthscale"]
        obj.max_lengthscale = aux_data["max_lengthscale"]
        return obj        
    
    def numpyro_model(self):
        outputscale = numpyro.sample("kernel_var", dist.Gamma(concentration=2.,rate=0.15))
        tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(0.1))
        inv_length_sq = numpyro.sample("_kernel_inv_length_sq",dist.HalfCauchy(jnp.ones(self.ndim))) # type: ignore
        lengthscales = numpyro.deterministic("kernel_length",1/jnp.sqrt(tausq*inv_length_sq)) # type: ignore
        k = self.kernel_func(self.train_x,self.train_x,lengthscales,outputscale,noise=self.noise,include_noise=True) 
        mll = numpyro.sample(
                "Y",
                dist.MultivariateNormal(
                    loc=jnp.zeros(self.train_x.shape[0]), # type: ignore
                    covariance_matrix=k,
                ),
                obs=self.train_y.squeeze(-1),)       
    
    def fit(self,rng_key,progbar=True,verbose=False):
        """
        Fits the GP using NUTS
        """
        init_params = None
        start = time.time()
        kernel = NUTS(self.numpyro_model,
                dense_mass=self.dense_mass,
                max_tree_depth=self.max_tree_depth)
        mcmc = MCMC(
                kernel,
                num_warmup=self.warmup_steps,
                num_samples=self.num_samples,
                num_chains=1,
                progress_bar=progbar,
                thinning=self.thinning,
                )
        #https://forum.pyro.ai/t/initialize-mcmc-chains-from-multiple-predetermined-starting-points/5062
        if init_params is not None:
            init_params = util.unconstrain_fn(self.model,model_args=self.train_x,model_kwargs=None,params=init_params)
        mcmc.run(rng_key,extra_fields=("potential_energy",),init_params=init_params)
        if verbose:
            mcmc.print_summary(exclude_deterministic=False)
        extras = mcmc.get_extra_fields()
        log.info(f" Hyperparameters MCMC elapsed time: {time.time() - start:.2f}s")
        self.samples["minus_log_prob"] = extras["potential_energy"]                
        self.samples["kernel_length"] = jnp.clip(self.samples["kernel_length"],self.min_lengthscale,self.max_lengthscale) # best values?
        self.fitted = True
        self.update_choleskys()

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

    def update_choleskys(self):
        """
        Updates the choleskys of the GP using the new training points.
        """
        vmap_func = lambda l,o : (jnp.linalg.cholesky(self.kernel_func(self.train_x,self.train_x,l,o,
                                                                   noise=self.noise,include_noise=True)),)
        vmap_arrays = (self.samples["kernel_length"], self.samples["kernel_var"]) 
        self.cholesky = split_vmap(vmap_func,vmap_arrays,batch_size=self.vmap_size)[0]

    def update(self,x_new,y_new,rng_key,refit=True,progbar=True,verbose=True):
        """
        Updates train_x and train_y, numpyro model and refit the GP using NUTS
        """
        self.add(x_new,y_new)
        if refit:
            self.fit(rng_key=rng_key,progbar=progbar,verbose=verbose)
        else:
            self.update_choleskys()

    def get_kernels(self,x1,x2,include_noise=True):
        f = lambda l,o : (self.kernel_func(x1,x2,l,o,noise=self.noise,include_noise=include_noise),)
        arrays_l_o = (self.samples["kernel_length"], self.samples["kernel_var"])
        kernels = split_vmap(f,arrays_l_o,batch_size=self.vmap_size)[0]
        return kernels

    def predict_mean(self,x):
        x = jnp.atleast_2d(x)
        k12s = self.get_kernels(self.train_x,x,include_noise=False)
        mean_f = lambda cho, k12 : (get_mean_from_cho(cho,k12,self.train_y),)
        means = split_vmap(mean_f,input_arrays=(self.cholesky,k12s),batch_size=self.vmap_size)[0]
        return means.mean(axis=0) * self.y_std + self.y_mean
    
    def predict_var(self,x):
        x = jnp.atleast_2d(x)
        k12s = self.get_kernels(self.train_x,x,include_noise=False)
        k22s = self.get_kernels(x,x,include_noise=True)
        var_f = lambda cho, k12, k22 : (get_var_from_cho(cho,k12,k22),)
        vars = split_vmap(var_f,input_arrays=(self.cholesky,k12s,k22s),batch_size=self.vmap_size)[0]
        return vars.mean(axis=0)*self.y_std**2
    
    def predict(self,x):
        """
        Returns the mean and variance of the unstandardised GP posterior at x, e.g for EI

        Arguments
        ---------
        x: jnp.ndarray
            Input points, size (M x D)
        """
        x = jnp.atleast_2d(x)
        mean = self.predict_mean(x)
        var = self.predict_var(x)
        return mean, var

    def _fantasy_var(self,x_new,lengthscales,outputscales,mc_points):
        x_new = jnp.atleast_2d(x_new)
        new_x = jnp.concatenate([self.train_x,x_new])
        k11 = self.kernel_func(new_x,new_x,lengthscales,outputscales,noise=self.noise,include_noise=True) 
        k11_cho = jnp.linalg.cholesky(k11) # can replace with fast update cholesky!
        k12 = self.kernel_func(new_x,mc_points,lengthscales,outputscales,noise=self.noise,include_noise=False)
        k22 = self.kernel_func(mc_points,mc_points,lengthscales,outputscales,noise=self.noise,include_noise=True)
        var = get_var_from_cho(k11_cho,k12,k22)
        return var
    
    def fantasy_var(self,x_new,mc_points):
        """
        Returns the MAP fantasy variance of the GP at x_new using the MAP hyperparameters
        """
        l, o = self.get_map_hyperparams()
        var = self._fantasy_var(x_new=x_new,lengthscales=l,outputscales=o,mc_points=mc_points)
        return var
    
    def fantasy_var_bayes(self,x_new,mc_points): 
        """
        Returns the fantasy variance of the GP at x_new using the hyperparameter samples
        """
        vmap_func = lambda l,o: (self._fantasy_var(x_new=x_new,lengthscales=l,outputscales=o,mc_points=mc_points),)
        vmap_arrays = (self.samples["kernel_length"], self.samples["kernel_var"])
        var = split_vmap(vmap_func,vmap_arrays,batch_size=self.vmap_size)[0]
        return var
    

    
    def get_map_hyperparams(self):
        """
        Returns the MAP hyperparameters
        """
        map_idx = jnp.argmin(self.samples["minus_log_prob"])
        l = self.samples["kernel_length"][map_idx,:]
        o = self.samples["kernel_var"][map_idx]
        return l,o
        
    def get_median_lengthscales(self):
        """
        Returns the median lengthscales
        """
        lengthscales = self.samples["kernel_length"]
        median_lengths = jnp.median(lengthscales,axis=0)
        return median_lengths
    
    def get_median_outputscales(self):
        """
        Returns the median output scales
        """
        outputscales = self.samples["kernel_var"]
        median_out = jnp.median(outputscales,axis=0)
        return median_out
    
    def get_median_mll(self):
        """
        Returns the median marginal log likelihood
        """
        mlls = self.samples["minus_log_prob"]
        median_mll = jnp.median(mlls)
        return median_mll
    
    def save(self,save_file='saas_fbgp'):
        """
        Saves the GP model as an npz file - training inputs, training outputs, lengthscales and output scales. Note that the training outputs are unstandardized.

        Arguments
        ----------
        save_file: str
            File name to save the GP model
        """
        jnp.savez(save_file+'.npz'
                ,train_x = self.train_x
                ,train_y = self.train_y*self.y_std + self.y_mean
                ,lengthscales=self.samples["kernel_length"]
                ,outputscales=self.samples["kernel_var"])
       
class SVM_SAAS_FBGP(SAAS_FBGP):

    def __init__(self,support_vectors, dual_coef, intercept, gamma_eff
                 ,train_x,train_y,noise=1e-8,kernel="rbf",vmap_size=8
                 ,sample_lengthscales=None,sample_outputscales=None
                 ,warmup_steps:int = 256,num_samples:int = 256,thinning:int = 32
                 ,dense_mass: bool = False,max_tree_depth:int = 6,num_chains: int = 1
                 ,min_lengthscale: float = 1e-2,max_lengthscale: float = 1e2) -> None:
        
        super().__init__(train_x, train_y, noise, kernel,vmap_size,
                         sample_lengthscales, sample_outputscales,
                         warmup_steps, num_samples, thinning,
                         dense_mass, max_tree_depth, num_chains,
                         min_lengthscale, max_lengthscale)
        self.support_vectors = support_vectors
        self.dual_coef = dual_coef
        self.intercept = intercept
        self.gamma_eff = gamma_eff

    # --- Pytree methods ---
    def tree_flatten(self):
        # Choose dynamic (leaf) attributes: these are jax arrays (or None)
        parent_leaves, aux_data = super().tree_flatten()
        # Add dynamic leaves
        leaves = parent_leaves + (self.support_vectors, self.dual_coef, self.intercept, self.gamma_eff)
        # no extra aux data
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        parent_aux, extra_aux = aux_data
        # Determine how many leaves the parent expected.
        num_parent_leaves = len(SAAS_FBGP.tree_flatten(cls.__new__(cls))[0])
        parent_leaves = leaves[:num_parent_leaves]
        extra_leaves = leaves[num_parent_leaves:]
        # First, create a parent instance from the parent's leaves.
        obj = SAAS_FBGP.tree_unflatten(parent_aux, parent_leaves)
        # Convert it into an instance of DerivedGP. One way is to update the class:
        obj.__class__ = cls
        # Now restore the extra fields.
        obj.support_vectors, obj.dual_coef,obj.intercept, obj.gamma_eff = extra_leaves
        # No extra static auxiliary data
        return obj
    
    def fit(self,rng_key,progbar=True,verbose=False):
        """
        Fits the GP using maximum likelihood hyperparameters
        """
        super().fit(rng_key,progbar=progbar,verbose=verbose)

    def update(self,x_new,y_new,rng_key,refit=False,progbar=True,verbose=False):
        super().update(x_new,y_new,rng_key,refit=refit,progbar=progbar,verbose=verbose)

    def update_svm(self,support_vectors, dual_coef, intercept, gamma_eff):
        """
        Updates the SVM parameters
        """
        self.support_vectors = support_vectors
        self.dual_coef = dual_coef
        self.intercept = intercept
        self.gamma_eff = gamma_eff

    def predict_mean(self, x):
        """
        Predicts the mean of the GP at x and unstandardizes it if x is within the boundary of the SVM, else returns -inf
        """
        x = jnp.atleast_2d(x)
        f = lambda single_x: svm_predict(single_x,self.support_vectors, self.dual_coef, self.intercept, self.gamma_eff)
        decision = vmap(f)(x)
        # Here, we choose -inf for points outside the SVM boundary.
        # (Alternatively, you might choose a very low value like -1e3, but -jnp.inf is more explicit.)
        gp_mean = super().predict_mean(x)
        res = jnp.where(decision >= 0, gp_mean, -1e30)
        # f = lambda x: svm_predict(self.support_vectors, self.dual_coef, self.intercept, self.gamma_eff, x)
        # decision = lax.map(f,x,batch_size=25)
        # res = jnp.where(decision > 0, super().predict_mean(x),-1e3)
        return res
    
    def predict_var(self, x):
        """
        Predicts the variance of the GP at x and unstandardizes it if x is within the boundary of the SVM, else returns noise flooer
        """
        x = jnp.atleast_2d(x)
        f = lambda single_x: svm_predict(single_x,self.support_vectors, self.dual_coef, self.intercept, self.gamma_eff)
        decision = vmap(f)(x)
        var =  super().predict_var(x)
        res = jnp.where(decision >= 0, var, self.noise*self.y_std**2)
        return res