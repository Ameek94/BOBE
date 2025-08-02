import sys
import numpy as np
from scipy.stats import qmc
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from numpyro.util import enable_x64
enable_x64()
from .acquisition import WIPV, EI #, logEI
from .gp import DSLP_GP, SAAS_GP, sample_GP_NUTS
from .svm_gp import SVM_GP
from .loglike import external_likelihood,   cobaya_likelihood
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from .bo_utils import input_standardize, input_unstandardize
from optax import adam, apply_updates
from .optim import optimize
from .nested_sampler import nested_sampling_Dy, nested_sampling_jaxns
from getdist import plots, MCSamples, loadMCSamples
from .seed_utils import get_new_jax_key, get_global_seed
from .logging_utils import get_logger
import tqdm
import time

# Set up logger using the new logging utilities
log = get_logger("[BO]")

# Acquisition optimizer
def optimize_acq(gp, acq, mc_points, x0=None, lr=5e-3, maxiter=150, n_restarts_optimizer=4, optimizer_name="adam"):
    """
    Optimize acquisition function using the optimize function.
    
    Args:
        gp: Gaussian Process model
        acq: Acquisition function
        mc_points: Monte Carlo points for acquisition evaluation
        x0: Initial guess (optional)
        lr: Learning rate
        maxiter: Maximum iterations
        n_restarts_optimizer: Number of restarts
        optimizer_name: Name of optimizer to use
        
    Returns:
        Tuple of (best_params, best_value)
    """
    ndim = mc_points.shape[1]
    
    # Define acquisition function with proper signature for the optimize function
    def acq_func(x):
        return acq(x=x, gp=gp, mc_points=mc_points)
    
    # Use the optimize function directly to minimize acquisition function (note: most acq functions are minimized)
    best_params, best_fval = optimize(
        func=acq_func,
        ndim=ndim,
        bounds=None,  # Will default to [0,1] bounds
        x0=x0,
        lr=lr,
        maxiter=maxiter,
        n_restarts=n_restarts_optimizer,
        minimize=True,
        verbose=False,  # Set to False to avoid duplicate logging
        optimizer_name=optimizer_name
    )
    
    return jnp.atleast_2d(best_params), best_fval

# Utility functions

# def get_point_with_large_value(train_x,train_y, n_points=1):
#     """
#     Get a point with large value from the training data
#     """
#     idx = jnp.argsort(train_y.flatten())[-n_points:]
#     return train_x[idx].flatten()

def get_mc_samples(gp, rng_key, warmup_steps=512, num_samples=512, thinning=1,method="NUTS",init_params=None):
    if method=='NUTS':
        try:
            mc_samples = sample_GP_NUTS(
            gp=gp, rng_key=rng_key, warmup_steps=warmup_steps,
            num_samples=num_samples, thinning=thinning, init_params=init_params
            )
        except:
            log.error("Error in sampling GP NUTS")
            mc_samples, _ = nested_sampling_Dy(gp, gp.ndim, maxcall=int(5e6)
                                            , dynamic=False, dlogz=0.1,equal_weights=True,
            )
    elif method=='NS':
        mc_samples, _ = nested_sampling_Dy(gp, gp.ndim, maxcall=int(5e6)
                                            , dynamic=False, dlogz=0.1,equal_weights=True,
        )
    elif method=='uniform':
        mc_samples = {}
        points = qmc.Sobol(gp.ndim, scramble=True).random(num_samples)
        mc_samples['x'] = points
    else:
        raise ValueError(f"Unknown method {method} for sampling GP")
    return mc_samples


def get_mc_points(mc_samples, mc_points_size=64):
    mc_size = max(mc_samples['x'].shape[0], mc_points_size)
    # Use numpy random choice - will respect global seed set by set_global_seed()
    idxs = np.random.choice(mc_size, size=mc_points_size, replace=False)
    return mc_samples['x'][idxs]


class BOBE:

    def __init__(self,
                loglikelihood=None,
                 n_cobaya_init=4,
                 n_sobol_init=32,
                 miniters=200,
                 maxiters=1500,
                 max_gp_size=1200,
                 resume=False,
                 resume_file=None,
                 save=True,
                 fit_step=10,
                 update_mc_step=10,
                 ns_step=10,
                 num_hmc_warmup=512,
                 num_hmc_samples=512,
                 mc_points_size=64,
                 mc_points_method='NUTS',
                 lengthscale_priors='DSLP',
                 use_svm=True,
                 svm_use_size = 300,
                 svm_update_step=5,
                 svm_threshold=250,
                 svm_gp_threshold=5000,
                 logz_threshold=1.0,
                 minus_inf=-1e5,
                 do_final_ns=True,
                 return_getdist_samples=False,
                 random_seed=None,):
        """
        Initialize the BOBE sampler class.

        Arguments
        ---------
        loglikelihood : external_loglike
            The loglikelihood function to be used. Must be an instance of external_loglike.
        n_cobaya_init : int
            Number of initial points from the cobaya reference distirbution when starting a run. 
            Is only used when the likelihood is an instance of cobaya_loglike, otherwise ignored.
        n_sobol_init : int
            Number of initial Sobol points for sobol when starting a run. 
        miniters : int
            Minimum number of iterations before checking convergence.
        maxiters : int
            Maximum number of iterations.
        max_gp_size : int
            Maximum number of points used to train the GP. 
            If using SVM, this is not the same as the number of points used to train the SVM.
        resume : bool
            If True, resume from a previous run. The resume_file argument must be provided.
        resume_file : str
            The file to resume from. Must be a .npz file containing the training data.
        save : bool
            If True, save the GP training data to a file so that it can be resumed from later.
        fit_step : int
            Number of iterations between GP refits.
        update_mc_step : int
            Number of iterations between MC point updates.
        ns_step : int
            Number of iterations between nested sampling runs.
        num_hmc_warmup : int
            Number of warmup steps for HMC sampling.
        num_hmc_samples : int
            Number of samples to draw from the GP.
        mc_points_size : int
            Number of points to use for the weighted integrated posterior variance acquisition function.
        mc_points_method : str
            Method to use for generating the MC points. Options are 'NUTS', 'NS', or 'uniform'. 
            Recommend to use 'NUTS' for most cases, 'NS' can be a good choice if the underlying likelihood has a highly complex structure.
        lengthscale_priors : str
            Lengthscale priors to use. Options are 'DSLP' or 'SAAS'. See the GP class for more details.
        use_svm : bool
            If True, use SVM to filter the GP predictions. 
            This is only required for high dimensional problems and when the scale of variation of the likelihood is extremely large. 
            For cosmological likelihoods with nuisance parameters, this is highly recommended.
        svm_use_size : int
            Minimum size of the SVM training set before the SVM filter is used in the GP.
        svm_update_step : int
            Number of iterations between SVM updates.
        logz_threshold : float
            Threshold for convergence of the nested sampling logz. 
            If the difference between the upper and lower bounds of logz is less than this value, the sampling will end.
        minus_inf : float
            Value to use for minus infinity. This is used to set the lower bound of the loglikelihood.
        random_seed : int, optional
            Random seed for reproducible results. If provided, sets the global random seed for
            all random number generators (NumPy, JAX, Python's random module). Default is None.
        """
        if not isinstance(loglikelihood, external_likelihood):
            raise ValueError("loglikelihood must be an instance of external_likelihood")

        # Handle random seed
        self.random_seed = random_seed
        if random_seed is not None:
            from .seed_utils import set_global_seed
            set_global_seed(random_seed)
            log.info(f"Global random seed set to {random_seed}")

        self.loglikelihood = loglikelihood

        self.param_list = self.loglikelihood.param_list
        self.param_bounds = self.loglikelihood.param_bounds
        self.param_labels = self.loglikelihood.param_labels
        self.ndim = len(self.param_list)


        if resume and resume_file is not None:
            # assert resume_file is not None, "resume_file must be provided if resume is True"
            log.info(f" Resuming from file {resume_file}")
            data = np.load(resume_file)
            self.train_x = jnp.array(data['train_x'])
            self.train_y = jnp.array(data['train_y'])
        else:
            init_points, init_vals = self.loglikelihood.get_initial_points(n_cobaya_init=n_cobaya_init,
                                    n_init_sobol=n_sobol_init)
            self.train_x = jnp.array(input_standardize(init_points, self.param_bounds))
            self.train_y = jnp.array(init_vals)

        # Best point so far
        idx_best = jnp.argmax(self.train_y)
        self.best_pt = input_unstandardize(self.train_x[idx_best],self.param_bounds).flatten()
        self.best_f = float(self.train_y.max())
        self.best = {name: f"{float(val):.4f}" for name, val in zip(self.param_list, self.best_pt)}
        log.info(f" Initial best point {self.best} with value = {self.best_f:.4f}")

        # GP setup
        if use_svm:
            gp = SVM_GP(
                train_x=self.train_x, train_y=self.train_y,
                minus_inf=minus_inf, lengthscale_priors=lengthscale_priors,
                kernel='rbf',svm_use_size=svm_use_size,svm_update_step=svm_update_step,
                svm_threshold=svm_threshold,gp_threshold=svm_gp_threshold,
            )
        else:
            gp = {
                'DSLP': DSLP_GP,
                'SAAS': SAAS_GP
            }[lengthscale_priors](
                train_x=self.train_x, train_y=self.train_y,
                noise=1e-8, kernel='rbf'
            )
        self.gp = gp

        # Store settings
        self.maxiters = maxiters
        self.miniters = miniters
        self.max_gp_size = max_gp_size
        self.fit_step = fit_step
        self.update_mc_step = update_mc_step
        self.ns_step = ns_step
        self.num_hmc_warmup = num_hmc_warmup
        self.num_hmc_samples = num_hmc_samples
        self.mc_points_size = mc_points_size
        self.minus_inf = minus_inf
        self.output_file = self.loglikelihood.name
        self.mc_points_method = mc_points_method
        self.save = save
        self.return_getdist_samples = return_getdist_samples
        self.do_final_ns = do_final_ns

        # Convergence control
        self.logz_threshold = logz_threshold
        self.converged = False
        self.termination_reason = "Max iterations reached"



    def check_convergence(self, step, logz_dict, threshold=2.0):
        """
        Check if the nested sampling has converged.
        """
        delta = logz_dict['upper'] - logz_dict['lower']
        mean = logz_dict['mean']
        if (delta < threshold and mean>self.minus_inf+10)  and step > self.miniters:
            log.info(f" Convergence check: delta = {delta:.4f}, step = {step}")
            log.info(" Converged")
            return True
        else:
            return False

    def run(self):
        """
        Run the iterative Bayesian Optimization loop.

        Arguments
        ---------
        None

        Returns
        ---------
        gp : GP object
            The fitted GP object.
        ns_samples : MCSamples | Nested sampling samples
            The samples from the final nested sampling run. This is a either a getdist MCSamples instance or a dictionary with the following keys ['x','weights','logl'].
        logz_dict : dict
            The logz dictionary from the nested sampling run. This contains the upper and lower bounds of the logz.
        """

        # Start by fitting the GP to the initial points
        self.gp.fit()

        # Monte Carlo points for acquisition function
        if get_global_seed() is not None:
            rng_key = get_new_jax_key()
        else:
            rng_key = jax.random.PRNGKey(0)
        x0_hmc = self.gp.get_random_point(npoints=1).squeeze(0)  #self.gp.train_x[jnp.argmax(self.gp.train_y)]
        # print(f"x0_hmc shape = {x0_hmc.shape}")
        self.mc_samples = get_mc_samples(self.gp, rng_key=rng_key,init_params=x0_hmc,
            warmup_steps=self.num_hmc_warmup, num_samples=self.num_hmc_samples, 
            thinning=1,method=self.mc_points_method)
        self.mc_points = get_mc_points(self.mc_samples, self.mc_points_size)

        best_pt_iteration = 0

        ii = 0

        for i in range(self.maxiters):

            ii = i + 1
            refit = (ii % self.fit_step == 0)
            ns_flag = (ii % self.ns_step == 0) and (ii > self.miniters)
            update_mc = (ii % self.update_mc_step == 0) and not ns_flag

            log.info(f" Iteration {ii}/{self.maxiters}, refit={refit}, update_mc={update_mc}, ns={ns_flag}")
            
            # start acq from mc_point with max var or max value 
            # mc_points_var = jax.lax.map(self.gp.predict_var,self.mc_points)
            # x0_acq = self.mc_points[jnp.argmax(mc_points_var)]

            # idxs = np.random.choice(self.gp.train_x.shape[0], size=4, replace=False)
            # idxs = np.random.choice(self.mc_points.shape[0], size=4, replace=False)
            # x0_acq = self.mc_points[idxs] #self.gp.get_random_point(npoints=4) #self.gp.train_x[idxs] #[jnp.argmax(self.gp.train_y[idxs])]
            # x0_acq =  self.gp.train_x[jnp.argmax(self.gp.train_y)]
            x0_acq = self.gp.get_random_point(npoints=4)

            # log.info(f"Acq starting points shape = {x0_acq.shape}")

            new_pt_u, acq_val = optimize_acq(
                self.gp, WIPV, self.mc_points,n_restarts_optimizer=4, x0=x0_acq)
            
            new_pt = input_unstandardize(new_pt_u, self.param_bounds) #.flatten()

            log.info(f" Acquisition value {acq_val:.4e} at new point")
            new_val = self.loglikelihood(
                new_pt, logp_args=(), logp_kwargs={}
            )

            new = {name: f"{float(val):.4f}" for name, val in zip(self.param_list, new_pt.flatten())}
            log.info(f" New point {new} with value = {new_val.item():.4f}, gp prediction = {self.gp.predict_mean(new_pt_u).item():.4f}")

            pt_exists_or_below_threshold = self.gp.update(new_pt_u, new_val, refit=refit)
            # if pt_exists and self.mc_points_method == 'NUTS':
            #     update_mc = True
            if update_mc:
                x0_hmc = self.gp.get_random_point(npoints=1).squeeze(0) #self.gp.train_x[jnp.argmax(self.gp.train_y)]
                print(f"x0_hmc shape = {x0_hmc.shape}")
                if get_global_seed() is not None:
                    rng_key = get_new_jax_key()
                else:
                    rng_key = jax.random.PRNGKey(ii*10)
                self.mc_samples = get_mc_samples(
                    self.gp, rng_key=rng_key,
                    warmup_steps=self.num_hmc_warmup, num_samples=self.num_hmc_samples,
                    thinning=1, method=self.mc_points_method,init_params=x0_hmc
                )
            self.mc_points = get_mc_points(self.mc_samples, self.mc_points_size)

            if float(new_val) > self.best_f:
                self.best_f = float(new_val)
                self.best_pt = new_pt
                self.best = {name: f"{float(val):.4f}" for name, val in zip(self.param_list, self.best_pt.flatten())}
                best_pt_iteration = ii
            log.info(f" Current best point {self.best} with value = {self.best_f:.4f}, found at iteration {best_pt_iteration}")

            if i % 4 == 0 and i > 0:
                jax.clear_caches()

            if (ii % 10 == 0) and self.save:
                log.info(" Saving GP to file")
                self.gp.save(outfile=self.output_file)

            if ns_flag:
                log.info(" Running Nested Sampling")
                ns_samples, logz_dict = nested_sampling_Dy(
                    self.gp, self.ndim, maxcall=int(5e6), dynamic=False, dlogz=0.1
                )
                log.info(" LogZ info: " + ", ".join([f"{k}={v:.4f}" for k,v in logz_dict.items()]))
                if self.check_convergence(i, logz_dict,threshold=self.logz_threshold):
                    self.converged = True
                    self.termination_reason = "LogZ converged"
                    break

            if self.gp.train_x.shape[0] > self.max_gp_size:
                self.termination_reason = "Max GP size exceeded"
                log.info(f" {self.termination_reason}")
                break

        if not self.converged:
            self.gp.fit()

        # Save and final nested sampling
        if self.save:
            self.gp.save(outfile=self.output_file)
        log.info(f" Sampling stopped: {self.termination_reason}")

        log.info(f" Final GP training set size: {self.gp.train_x.shape[0]}, max size: {self.max_gp_size}")
        log.info(f" Number of iterations: {ii}, max iterations: {self.maxiters}")

        if self.do_final_ns and not self.converged:
            log.info(" Final Nested Sampling")
            ns_samples, logz_dict = nested_sampling_Dy(
                self.gp, self.ndim, maxcall=int(1e7), dynamic=True, dlogz=0.01
            )
            log.info(" Final LogZ: " + ", ".join([f"{k}={v:.4f}" for k,v in logz_dict.items()]))
        if self.save:
            np.savez(f'{self.output_file}_samples.npz',ns_samples = input_unstandardize(ns_samples['x'],self.param_bounds),param_bounds = self.param_bounds)

        if self.return_getdist_samples:
            ranges = dict(zip(self.param_list,self.param_bounds.T))
            samples = input_unstandardize(ns_samples['x'],self.param_bounds)
            weights = ns_samples['weights']
            loglikes = ns_samples['logl']
            gd_samples = MCSamples(samples=samples, names=self.param_list, labels=self.param_labels, 
                                ranges=ranges, weights=weights,loglikes=loglikes,label='GP',sampler='nested')
            
            output_samples = gd_samples
        else:
            output_samples = ns_samples
                
        return self.gp, output_samples, logz_dict


