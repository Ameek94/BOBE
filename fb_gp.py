# Convert to JAX

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
from numpyro.infer import MCMC, NUTS
# from util import chunk_vmap
from jax import config
config.update("jax_enable_x64", True)
from numpyro.util import enable_x64
enable_x64()
from functools import partial

# todo
# 1. check works in n>1 dim cases
# 2. test speed vs standard botorch
# 3. Matern kernal
# 4. methods to load and save from dict
# 5. 

def cdist(x, y):
    # x, y are n1 x d and n2 x d
  return jnp.sum(jnp.square(x[:,None,:] - y),axis=-1)

@jit
def rbf_kernel(xa,
               xb,
               lengthscales,
               outputscale):
    c_dist = cdist(xa/lengthscales,xb/lengthscales) 
    c_dist = jnp.exp(-0.5*c_dist)
    return outputscale*c_dist


def matern_kernel(xa,xb,lengthscales,outputscale):
   return None


class numpyro_model:
   # Note - train_x and train_y received here are already transformed, npoints x ndim and npoints x 1
   def __init__(self, train_x,  train_y, train_yvar, kernel_func) -> None:
      self.train_x = train_x 
      self.train_y = train_y
      self.train_yvar = train_yvar
      self.ndim = train_x.shape[-1]
      self.npoints = train_x.shape[-2]
      self.kernel_func = kernel_func
    #   print(self.npoints,self.ndim)
 
   
   def model(self):
        outputscale = numpyro.sample("kernel_var", dist.Gamma(concentration=2.,rate=0.15))
        tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(0.1))
        inv_length_sq = numpyro.sample("_kernel_inv_length_sq",dist.HalfCauchy(jnp.ones(self.ndim))) # type: ignore
        lengthscales = numpyro.deterministic("kernel_length",1/jnp.sqrt(tausq*inv_length_sq)) # type: ignore
        k = rbf_kernel(self.train_x,self.train_x,lengthscales,outputscale) + jnp.diag(self.train_yvar) #self.train_yvar * jnp.eye(self.train_x.shape[0])
        ll = numpyro.sample(
                "Y",
                dist.MultivariateNormal(
                    loc=jnp.zeros(self.train_x.shape[0]), # type: ignore
                    covariance_matrix=k,
                ),
                obs=self.train_y.squeeze(-1),)
        # loglike = numpyro.deterministic("loglike",jnp.log(ll)) # replace with loglike or use another method for ll

   def run_mcmc(self,rng_key,dense_mass=True,max_tree_depth=6,
                warmup_steps=512,num_samples=512,num_chains=1,thinning=16,progbar=True,
                ) -> dict:
        start = time.time()
        kernel = NUTS(self.model,
                dense_mass=dense_mass,
                max_tree_depth=max_tree_depth)
        mcmc = MCMC(
                kernel,
                num_warmup=warmup_steps,
                num_samples=num_samples,
                num_chains=num_chains,
                progress_bar=progbar,
                thinning=thinning,
                )
        mcmc.run(rng_key,extra_fields=("potential_energy",))
        mcmc.print_summary(exclude_deterministic=False)
        extras = mcmc.get_extra_fields()
        # print(extras.keys())
        print(f"\nMCMC elapsed time: {time.time() - start:.2f}s")
        return mcmc.get_samples(), extras # type: ignore

   def fit_gp_NUTS(self,rng_key,dense_mass=True,max_tree_depth=6,
                warmup_steps=512,num_samples=512,num_chains=1,progbar=True,thinning=16):
        samples, extras = self.run_mcmc(rng_key=rng_key,
                dense_mass=dense_mass,
                max_tree_depth=max_tree_depth,
                warmup_steps=warmup_steps,
                num_samples=num_samples,
                num_chains=num_chains,
                progbar=progbar,
                thinning=thinning)
        samples["minus_log_prob"] = extras["potential_energy"]  # see also numpyro.infer.util.log_likelihood
        # numpyro already thins samples and keeps deterministic params
        # samples["lengthscales"] = (samples["kernel_tausq"].unsqueeze(-1)*samples["_kernel_inv_length_sq"])
        #print(samples["_kernel_inv_length_sq"].size(),samples["lengthscales"].size(),samples["kernel_var"].size())
        return samples

class saas_fbgp: #jit.ScriptModule):

    def __init__(self,train_x,train_y,train_yvar=None,standardise_y=True,noise=1e-6,kernel_func=rbf_kernel) -> None:
        """
        train_x: size (N x D), assumed to be standardised by the main sampler module
        train_y: size (N x 1)
        train_yvar: size (N x 1)
        """

        # check train_x and train_y dims are N x D and N x 1

        self.train_x = train_x

        if standardise_y:
            self.y_mean = jnp.mean(train_y,axis=-2)
            self.y_std = jnp.std(train_y,axis=-2)
        else:
            self.y_mean = 0.
            self.y_std = 1.
    
        self.train_y = (train_y - self.y_mean) / self.y_std

        self.noise = noise
        if train_yvar is None:
            self.train_yvar = self.noise * jnp.ones_like(train_y)
        else:
            self.train_yvar = train_yvar

        self.fitted = False
        self.samples = {}
        self.num_samples = 0
        self.cholesky = []
    
        self.kernel_func = kernel_func
        
        self.numpyro_model = numpyro_model(self.train_x,self.train_y,self.train_yvar,self.kernel_func)

        pass

    def fit(self,rng_key,dense_mass=True,max_tree_depth=6,
                warmup_steps=512,num_samples=512,num_chains=1,progbar=True,thinning=16):
        self.samples = self.numpyro_model.fit_gp_NUTS(rng_key,dense_mass=dense_mass,
                                                   max_tree_depth=max_tree_depth,
                                                   warmup_steps=warmup_steps,
                                                   num_samples=num_samples,
                                                   num_chains=num_chains,
                                                   progbar=progbar,
                                                   thinning=thinning)
        self.num_samples = self.samples["kernel_length"].shape[0]
        lengthscales = self.samples["kernel_length"]
        outputscales = self.samples["kernel_var"]
        # print(lengthscales.size(),outputscales.size())
        kernel = lambda l,o : jnp.linalg.cholesky(self.noise*jnp.eye(self.train_x.shape[0]) + self.kernel_func(self.train_x,self.train_x,l,o))
        self.cholesky = vmap(kernel,in_axes=(0,0),out_axes=(0))(lengthscales,outputscales)
        # print(self.cholesky.size())
        self.map_lengthscales, self.map_outputscales = self.get_map_hyperparams()
        self.fitted = True
    
    def get_mean_var(self,X,k11_cho,lengthscales,outputscales):
        """
        Algorithm 2.1 of R&W (2006)
        """
        if not self.fitted:
            raise RuntimeError("FullyBayesian GP needs to be fitted using model.fit() first")
        
        k12 = self.kernel_func(self.train_x,X,
                               lengthscales,
                               outputscales)
        k22 = self.kernel_func(X,X,
                               lengthscales,
                               outputscales)

        var = self._get_var(k11_cho,k12,k22)
        mean = self._get_mean(k11_cho,k12)
        return mean, var
    
    def _get_var(self,k11_cho,k12,k22):
        vv = solve_triangular(k11_cho,k12,lower=True)
        var = jnp.diag(k22) - jnp.sum(vv*vv,axis=0) 
        return var
    
    def _get_mean(self,k11_cho,k12):
        mu = jnp.matmul(jnp.transpose(k12),cho_solve((k11_cho,True),self.train_y))
        mean = mu[:,0]  
        return mean

    # @jit.script_method
    def predict(self,X):
        X = jnp.atleast_2d(X)
        if not self.fitted:
            raise RuntimeError("FullyBayesian GP needs to be fitted using model.fit() first")
        func = lambda cho, l, o :  self.get_mean_var(X=X,k11_cho=cho,lengthscales=l,outputscales=o)    
        batched_predict = vmap(func,in_axes=(0,0,0),out_axes=(0,0))
        lengthscales = self.samples["kernel_length"]
        outputscales = self.samples["kernel_var"]
        mean, var = batched_predict(self.cholesky,lengthscales,outputscales)
        return mean, var
    
    def posterior(self,X,single=False,unstandardize=True): # for use externally
        mean, var = self.predict(X)
        if unstandardize:
            mean = mean*self.y_std + self.y_mean
            var = var*self.y_std**2           
        if single:
            mean = mean.mean(axis=0) #*self.y_std + self.y_mean
            var = var.mean(axis=0) #* self.y_std**2      
        return mean, var

    def _fantasy_var_fb(self,x_new,lengthscales,outputscales,mc_points):
        x_new = jnp.atleast_2d(x_new)
        new_x = jnp.concatenate([self.train_x,x_new])
        k11 = self.kernel_func(new_x,new_x,lengthscales,outputscales)+self.noise*jnp.eye(new_x.shape[0])
        # k11 = self.kernel_func(new_x,new_x,self.map_lengthscales,self.map_outputscales)+self.noise*jnp.eye(new_x.shape[0])
        k11_cho = jnp.linalg.cholesky(k11) # replace with fast update cholesky!
        # k12 = self.kernel_func(new_x,mc_points,self.map_lengthscales,self.map_outputscales)
        # k22 = self.kernel_func(mc_points,mc_points,self.map_lengthscales,self.map_outputscales)
        # print(np.shape(new_x),np.shape(k11_cho),np.shape(k12),np.shape(k22))
        k12 = self.kernel_func(new_x,mc_points,lengthscales,outputscales)
        k22 = self.kernel_func(mc_points,mc_points,lengthscales,outputscales)
        var = self._get_var(k11_cho,k12,k22)
        return var
    
    def fantasy_var_fb(self,x_new,mc_points):
        func = lambda x,l,o: self._fantasy_var_fb(x_new=x,lengthscales=l,outputscales=o,mc_points=mc_points)
        batched_var_hyperparams = vmap(func,in_axes=(None,0,0))
        batched_var_xnew = vmap(batched_var_hyperparams,in_axes=(0,None,None)) #needed or can be moved to acq?
        lengthscales = self.samples["kernel_length"]
        outputscales = self.samples["kernel_var"]
        # var = batched_fantasy_var(x_new,lengthscales,outputscales)
        var = batched_var_xnew(x_new,lengthscales,outputscales)
        # print(var.shape)
        return var
    
    # maybe not needed?
    def _fantasy_var_map(self,x_new,lengthscales,outputscales,mc_points):
        x_new = jnp.atleast_2d(x_new)
        new_x = jnp.concatenate([self.train_x,x_new])
        k11 = self.kernel_func(new_x,new_x,lengthscales,outputscales)+self.noise*jnp.eye(new_x.shape[0])
        # k11 = self.kernel_func(new_x,new_x,self.map_lengthscales,self.map_outputscales)+self.noise*jnp.eye(new_x.shape[0])
        k11_cho = jnp.linalg.cholesky(k11) # replace with fast update cholesky!
        # k12 = self.kernel_func(new_x,mc_points,self.map_lengthscales,self.map_outputscales)
        # k22 = self.kernel_func(mc_points,mc_points,self.map_lengthscales,self.map_outputscales)
        # print(np.shape(new_x),np.shape(k11_cho),np.shape(k12),np.shape(k22))
        k12 = self.kernel_func(new_x,mc_points,lengthscales,outputscales)
        k22 = self.kernel_func(mc_points,mc_points,lengthscales,outputscales)
        var = self._get_var(k11_cho,k12,k22)
        return var
    
    def fantasy_var_map(self,x_new,mc_points):
        func = lambda x,l,o: self._fantasy_var_fb(x_new=x,lengthscales=l,outputscales=o,mc_points=mc_points)
        batched_var_hyperparams = vmap(func,in_axes=(None,0,0))
        batched_var_xnew = vmap(batched_var_hyperparams,in_axes=(0,None,None))
        lengthscales = self.samples["kernel_length"]
        outputscales = self.samples["kernel_var"]
        # var = batched_fantasy_var(x_new,lengthscales,outputscales)
        var = batched_var_xnew(x_new,lengthscales,outputscales)
        # print(var.shape)
        return var
    
    def get_map_hyperparams(self):
        map_idx = jnp.argmin(self.samples["minus_log_prob"])
        l = self.samples["kernel_length"][map_idx,:]
        o = self.samples["kernel_var"][map_idx]
        return l, o

    # Use JAX NS for this
    def get_evidence(self):
        pass

    def get_param_posteriors(self):
        pass

    def model_dict(self):
        pass

    def load_from_dict(self):
        pass


    


