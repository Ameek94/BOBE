# The FullyBayesian GP implementation is based on Eriksson, D. and Jankowiak, M., “High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces” (2021), arXiv:2103.00349, 
# doi: 10.1080/00401706.2018.1469433 (see also SAASBO on GitHub) and the BoTorch FullyBayesianGP with minor changes
#

import math
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
from bo_utils import split_vmap
from jax import config
from sympy import false
config.update("jax_enable_x64", True)
from numpyro.util import enable_x64
enable_x64()
from functools import partial
import logging
log = logging.getLogger("[GP]")
import multiprocessing
global_num_chains = multiprocessing.cpu_count()
import jax
#jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
#jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
#jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
#jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


# todo
# 1. method to reuse previous mcmc samples if HMC runs into issues
# 2. matern needs to be fixed
# 3. test speed vs standard botorch
# 4. stability of cholesky when points are very close together 
# 5. Pytree for GP?

sqrt5 = jnp.sqrt(5.)

def dist_sq(x, y):
    """
    Compute squared Euclidean distance between two points x, y. If x is n1 x d and y is n2 x d returns a n1 x n2 matrix of distancess.
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
    k = outputscale*sq_dist #jnp.square(
    if include_noise:
        k+= noise*jnp.eye(k.shape[0])
    return k

@partial(jit,static_argnames='include_noise')
def matern_kernel(xa,xb,lengthscales,outputscale,noise,include_noise=True):
    """
    The Matern-5/2 kernel
    """
    _dist = jnp.sqrt(dist_sq(xa/lengthscales,xb/lengthscales))
    exp = jnp.exp(-sqrt5*_dist)
    poly = 1 + _dist*(sqrt5 + _dist*5/3)
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

def sample_GP_NUTS(gp,rng_key,warmup_steps=512,num_samples=512,progress_bar=False,thinning=2,verbose=False
                   ,init_params=None):
    """
    Sample x using the GP mean as the logprob
    """
    class gp_dist(numpyro.distributions.Distribution):
        support = dist.constraints.real

        def __init__(self,gp: saas_fbgp):
            super().__init__(batch_shape = (), event_shape=())
            self.gp = gp
        def sample(self, key, sample_shape=()):
            raise NotImplementedError
        @jit
        def log_prob(self,x):
            val, _ = gp.posterior(x,single=True,unstandardize=True) #gp.GPSample_posterior(x,single=True,unstandardize=True)
            return val

    def model(train_x):
        x = numpyro.sample('x',numpyro.distributions.Uniform
                           (low=jnp.zeros(train_x.shape[1]),high=jnp.ones(train_x.shape[1]))) # type: ignore
        numpyro.sample('y',gp_dist(gp),obs=x)
    

    start = time.time()
    
    kernel = NUTS(model,dense_mass=False,
                max_tree_depth=6)
    
    num_chains = gp.num_chains
    log.info(f" Posterior Sampling running on {num_chains} devices")
    
    mcmc = MCMC(kernel,num_warmup=warmup_steps,
                num_samples=num_samples,
                num_chains=num_chains,
                chain_method="parallel",
                progress_bar=progress_bar,
                thinning=thinning,
                jit_model_args=True)
    if init_params is not None:
        init_params = util.unconstrain_fn(model,model_args=gp.train_x,model_kwargs=None,params=init_params)
    
    mcmc.run(rng_key,gp.train_x,extra_fields=("potential_energy",),init_params=init_params)
    
    if verbose:
        mcmc.print_summary(exclude_deterministic=False)
    log.info(f" Sampled parameters MCMC took {time.time()-start:.4f} s")
    samples = mcmc.get_samples(group_by_chain=True)
    flattened_samples = {k: v.reshape(-1, *v.shape[2:]) for k, v in samples.items()}
    final_thinned_samples = {k: v[::num_chains] for k, v in flattened_samples.items()}
    print({k: v.shape for k, v in final_thinned_samples.items()})
    #nuts_samples = mcmc.get_samples()['x'] 
    return final_thinned_samples['x']


class numpyro_model:
   # Note - train_x and train_y received here are already transformed, npoints x ndim and npoints x 1
   def __init__(self, train_x,  train_y, #train_yvar, 
                kernel_func=rbf_kernel,noise=1e-4) -> None:
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
                    loc = jnp.zeros(self.train_x.shape[0]), # type: ignore
                    covariance_matrix=k,
                ),
                obs=self.train_y.squeeze(-1),)

    # add method to start from previous samples

    # how can we speed up the MCMC when we already have a large number of samples? main bottleneck -> inversion of kernel
   def run_mcmc(self,rng_key,dense_mass=True,max_tree_depth=6,
                warmup_steps=512,num_samples=512,num_chains=1,thinning=16,
                progbar=True,verbose=True,init_params=None,
                ) -> dict:
        start = time.time()
        kernel = NUTS(self.model,
                dense_mass=dense_mass,
                max_tree_depth=max_tree_depth)
        log.info(f" MCMC Running with {num_chains} chains")
        mcmc = MCMC(
                kernel,
                num_warmup=warmup_steps,
                num_samples=num_samples,
                num_chains=num_chains,
                chain_method="parallel",
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
        samples = mcmc.get_samples(group_by_chain=True)
        flattened_samples = {k: v.reshape(-1, *v.shape[2:]) for k, v in samples.items()}
        final_thinned_samples = {k: v[::num_chains] for k, v in flattened_samples.items()}
        print({k: v.shape for k, v in final_thinned_samples.items()})
        return final_thinned_samples, extras # type: ignore

    # add method to start from previous map hyperparams
   def fit_gp_NUTS(self,rng_key,dense_mass=True,max_tree_depth=6,
                warmup_steps=512,num_samples=512,num_chains=1,progbar=True,thinning=16,verbose=False):
        log.info(f" Hyperparameter Sampling running on {num_chains} devices")
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

class saas_fbgp:

    def __init__(self
                 ,train_x
                 ,train_y
                 ,standardise_y=True
                 ,noise=1e-8
                 ,kernel="rbf"
                 ,vmap_size=8
                 ,sample_lengthscales=None
                 ,sample_outputscales=None
                 ,warmup_steps:int = 256
                 ,num_samples:int = 256
                 ,thinning:int = 16
                 ,dense_mass: bool = True
                 ,max_tree_depth:int = 6
                 ,num_chains: int = 0
                 ,min_lengthscale: float = 1e-3
                 ,max_lengthscale: float = 1e2) -> None:
        """
        train_x: size (N x D), always in [0,1] coming from the sampler module
        train_y: size (N x 1)
        noise
        """

        # check train_x and train_y dims are N x D and N x 1, param_bounds are 2 x d
        
        self.train_x = train_x

        if standardise_y:
            self.y_mean = jnp.mean(train_y,axis=-2)
            self.y_std = jnp.std(train_y,axis=-2)
        else:
            self.y_mean = 0.
            self.y_std = 1.
    
        self.train_y = (train_y - self.y_mean) / self.y_std

        self.noise = noise
        self.fitted = False
        self.num_samples = 0
        self.vmap_size = vmap_size # to process vmap in vmap_size batches
        self.kernel_func = rbf_kernel if kernel=="rbf" else matern_kernel
        
        self.numpyro_model = numpyro_model(self.train_x,self.train_y,
                                           self.kernel_func,noise = self.noise)
        if sample_lengthscales is not None and sample_outputscales is not None:
            self.fitted=True
            self.samples = {}
            self.samples["kernel_length"] = sample_lengthscales
            self.samples["kernel_var"] = sample_outputscales
            # self.num_samples = len(sample_outputscales)
            self.update_choleskys()
        
        self.warmup_steps=warmup_steps
        self.num_samples=num_samples
        self.thinning=thinning
        self.dense_mass = dense_mass
        self.max_tree_depth = max_tree_depth
        self.num_chains = num_chains
    
    def fit(self,rng_key,progbar=True,verbose=False):
        
        self.samples = self.numpyro_model.fit_gp_NUTS(rng_key,dense_mass=self.dense_mass,
                                                   max_tree_depth=self.max_tree_depth,
                                                   warmup_steps=self.warmup_steps,
                                                   num_samples=self.num_samples,
                                                   num_chains=self.num_chains,
                                                   progbar=progbar,
                                                   thinning=self.thinning,
                                                   verbose=verbose)
        
        # self.samples["kernel_length"] = jnp.clip(self.samples["kernel_length"],1e-3,1e2) # best values?
        self.fitted = True
        self.update_choleskys()

    def update(self,x_new,y_new,rng_key,progbar=False,verbose=False):
        """
        Updates train_x and train_y, numpyro model and refit the GP using NUTS
        """
        self.train_x = jnp.concatenate([self.train_x,x_new])
        self.train_y = self.train_y*self.y_std + self.y_mean 
        self.train_y = jnp.concatenate([self.train_y,y_new])
        self.y_mean = jnp.mean(self.train_y,axis=-2)
        self.y_std = jnp.std(self.train_y,axis=-2)
        self.train_y = (self.train_y - self.y_mean) / self.y_std
        self.numpyro_model.update(self.train_x,self.train_y)
        self.fit(rng_key=rng_key
                 ,progbar=progbar
                 ,verbose=verbose)


    # can make this faster with quicker update of cholesky
    def quick_update(self,x_new,y_new):
        """
        Updates train_x and train_y, 
        recomputes the Cholesky matrices but using the same GP hyperparameters from the previous step. 
        We do not refit the GP using NUTS here
        """
        self.train_x = jnp.concatenate([self.train_x,x_new])
        self.train_y = self.train_y*self.y_std + self.y_mean 
        self.train_y = jnp.concatenate([self.train_y,y_new])
        self.y_mean = jnp.mean(self.train_y,axis=-2)
        self.y_std = jnp.std(self.train_y,axis=-2)
        self.train_y = (self.train_y - self.y_mean) / self.y_std
        self.numpyro_model.update(self.train_x,self.train_y)
        self.update_choleskys()
        return None
    
    def update_choleskys(self):
        vmap_func = lambda l,o : (jnp.linalg.cholesky(self.kernel_func(self.train_x,self.train_x,l,o,
                                                                   noise=self.noise,include_noise=True)),)
        vmap_arrays = (self.samples["kernel_length"],self.samples["kernel_var"]) 
        self.cholesky = split_vmap(vmap_func,vmap_arrays,batch_size=self.vmap_size)[0] 

    def get_mean_var(self,X,k11_cho,lengthscales,outputscales):
        """
        Algorithm 2.1 of R&W (2006)
        """
        
        k12 = self.kernel_func(self.train_x,X,
                               lengthscales,
                               outputscales,noise=self.noise,include_noise=False)
        k22 = self.kernel_func(X,X,
                               lengthscales,
                               outputscales,noise=self.noise,include_noise=True)

        var = get_var_from_cho(k11_cho,k12,k22) 
        mean = get_mean_from_cho(k11_cho,k12,self.train_y)
        return mean, var
    
    def predict(self,X):
        X = jnp.atleast_2d(X) # move it to external 
        vmap_func = lambda cho, l, o :  self.get_mean_var(X=X,k11_cho=cho,lengthscales=l,outputscales=o)   
        vmap_arrays = (self.cholesky, 
                       self.samples["kernel_length"],
                       self.samples["kernel_var"])
        mean, var = split_vmap(vmap_func,vmap_arrays,batch_size=self.vmap_size)
        return mean, var
    
    def posterior(self,X,single=False,unstandardize=True): # for use externally
        mean, var = self.predict(X)
        if unstandardize:
            mean = mean*self.y_std + self.y_mean
            var = var*self.y_std**2           
        if single:
            mean = mean.mean(axis=0) 
            var = var.mean(axis=0)    
        return mean, var
    
    def GPSample_predict(self,X):
        X = jnp.atleast_2d(X) # move it to external 
        vmap_func = lambda cho, l, o :  self.get_mean_var(X=X,k11_cho=cho,lengthscales=l,outputscales=o)
        MAP_lengthscale, MAP_outputscale = self.get_map_hyperparams()
        MAP_cholesky = self.get_map_cholesky()
        #vmap_arrays = (MAP_cholesky, MAP_lengthscale, MAP_outputscale)
                       #(self.cholesky,
                       #self.samples["kernel_length"],
                       #self.samples["kernel_var"])
        #mean, var = split_vmap(vmap_func,vmap_arrays,batch_size=self.vmap_size)
        mean, var = vmap_func(MAP_cholesky, MAP_lengthscale, MAP_outputscale)
        return mean, var

    def GPSample_posterior(self,X,single=False,unstandardize=True): # for use externally
        mean, var = self.GPSample_predict(X)
        if unstandardize:
            mean = mean*self.y_std + self.y_mean
            var = var*self.y_std**2           
        if single:
            mean = mean.mean(axis=0) 
            var = var.mean(axis=0)    
        return mean, var

    # think about doing sequential optimization, at the moment joint acquisition is supported
    def _fantasy_var_fb(self,x_new,lengthscales,outputscales,mc_points):
        x_new = jnp.atleast_2d(x_new)
        new_x = jnp.concatenate([self.train_x,x_new])
        k11 = self.kernel_func(new_x,new_x,lengthscales,outputscales,noise=self.noise,include_noise=True) 
        k11_cho = jnp.linalg.cholesky(k11) # can replace with fast update cholesky!
        k12 = self.kernel_func(new_x,mc_points,lengthscales,outputscales,noise=self.noise,include_noise=False)
        k22 = self.kernel_func(mc_points,mc_points,lengthscales,outputscales,noise=self.noise,include_noise=True)
        var = get_var_from_cho(k11_cho,k12,k22)
        return (var,)
    
    # is vmap/map or batching mc_points needed?
    def fantasy_var_fb(self,x_new,mc_points): 
        vmap_func = lambda l,o: self._fantasy_var_fb(x_new=x_new,lengthscales=l,outputscales=o,mc_points=mc_points)
        vmap_arrays = (self.samples["kernel_length"], self.samples["kernel_var"])
        var = split_vmap(vmap_func,vmap_arrays,batch_size=self.vmap_size)[0]
        return var
    def fantasy_var_fb_acq(self, x_new, mc_points):
        l, o = MAP_lengthscale, MAP_outputscale = self.get_map_hyperparams()
        var = self._fantasy_var_fb(x_new=x_new,lengthscales=l,outputscales=o,mc_points=mc_points)[0]
        return var

    def get_map_hyperparams(self):
        map_idx = jnp.argmin(self.samples["minus_log_prob"])
        l = self.samples["kernel_length"][map_idx,:]
        o = self.samples["kernel_var"][map_idx]
        return l, o

    def get_map_cholesky(self):
        map_idx = jnp.argmin(self.samples["minus_log_prob"])
        return self.cholesky[map_idx, :]
    
    def get_median_lengthscales(self):
        lengthscales = self.samples["kernel_length"]
        median_lengths = jnp.median(lengthscales,axis=0)
        return median_lengths
    
    def get_median_outputscales(self):
        outputscales = self.samples["kernel_var"]
        median_out = jnp.median(outputscales,axis=0)
        return median_out
    
    def get_median_mll(self):
        mlls = self.samples["minus_log_prob"]
        median_mll = jnp.median(mlls)
        return median_mll
    
    def save(self,save_file, save_file_path):
        jnp.savez(save_file_path + save_file+'.npz'
                ,train_x = self.train_x
                ,train_y= self.train_y
                ,lengthscales=self.samples["kernel_length"]
                ,outputscales=self.samples["kernel_var"])
       