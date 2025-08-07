import sys
import numpy as np
from scipy.stats import qmc
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from numpyro.util import enable_x64
enable_x64()
from typing import Optional, Union, Tuple, Dict, Any
from optax import adam, apply_updates
from getdist import plots, MCSamples, loadMCSamples
import tqdm
import time
# from .acquisition import WIPV, EI #, logEI
from .gp import DSLP_GP, SAAS_GP #, sample_GP_NUTS
from .svm_gp import SVM_GP
from .clf_gp import ClassifierGP
from .loglike import ExternalLikelihood, CobayaLikelihood
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from .utils import scale_from_unit, scale_to_unit
from .seed_utils import set_global_seed, get_global_seed, get_jax_key, split_jax_key, ensure_reproducibility
from .nested_sampler import nested_sampling_Dy, nested_sampling_jaxns
from .optim import optimize
from .logging_utils import get_logger

log = get_logger("[bo]")

# # 1) Filter class: only allow exactly INFO
# class InfoFilter(logging.Filter):
#     """
#     """
#     def filter(self, record):
#         """
#         """
#         return record.levelno == logging.INFO

# # 2) Create and configure the stdout handler
# stdout_handler = logging.StreamHandler(sys.stdout)
# stdout_handler.setLevel(logging.INFO)       # accept INFO and above...
# stdout_handler.addFilter(InfoFilter())      # ...but filter down to only INFO
# stdout_fmt = logging.Formatter('%(asctime)s %(levelname)s:%(name)s: %(message)s')
# stdout_handler.setFormatter(stdout_fmt)

# # 3) Create and configure the stderr handler
# stderr_handler = logging.StreamHandler(sys.stderr)
# stderr_handler.setLevel(logging.WARNING)    # accept WARNING and above
# stderr_fmt = logging.Formatter('%(asctime)s %(levelname)s:%(name)s: %(message)s')
# stderr_handler.setFormatter(stderr_fmt)

# # 4) Get your logger, clear defaults, add both handlers
# log = logging.getLogger(name="[BO]")
# # stop this logger “bubbling” messages up to root
# log.propagate = False  
# log.handlers.clear()
# log.setLevel(logging.INFO)               # ensure INFO+ get processed
# log.addHandler(stdout_handler)
# log.addHandler(stderr_handler)

def WIPV(x, gp, mc_points=None):
    """
    Computes the Weighted Integrated Posterior Variance acquisition function.
    
    Args:
        x: Input points (shape: [n, ndim])
        gp: Gaussian process model
        mc_points: Optional Monte Carlo points for fantasy variance computation
    
    Returns:
        Mean of the posterior variance at the input points.
    """
    var = gp.fantasy_var(x, mc_points=mc_points)
    return jnp.mean(var)

# # Acquisition optimizer
# def optimize_acq(gp, acq, mc_points, x0=None, lr=5e-3, maxiter=200, n_restarts_optimizer=4):
    
#     f = lambda x: acq(x=x, gp=gp, mc_points=mc_points)

#     @jax.jit
#     def acq_val_grad(x):
#         return jax.value_and_grad(f)(x)

#     if x0 is None:
#         x0 = np.random.uniform(size=(n_restarts_optimizer, mc_points.shape[1]))
#     else:
#         x0 = jnp.atleast_2d(x0)  # Ensure x0 is at least 2D
#         # print(f"x0 shape: {x0.shape}, expected shape: {(n_restarts_optimizer, mc_points.shape[1])}")
#         n_x0 = x0.shape[0]
#         if n_x0 < n_restarts_optimizer:
#             needed_x0 = n_restarts_optimizer - n_x0
#             added_x0 = np.random.uniform(size=(needed_x0, mc_points.shape[1]))
#             x0 = jnp.concatenate([x0, added_x0], axis=0)
#     # print(f"x0 shape after processing: {x0.shape}, expected shape: {(n_restarts_optimizer, mc_points.shape[1])}")

#     # params = jnp.array(np.random.uniform(0, 1, size=mc_points.shape[1])) if x0 is None else x0
#     optimizer = adam(learning_rate=lr)

#     @jax.jit
#     def step(carry):
#         params, opt_state = carry
#         (val, grad) = acq_val_grad(params)
#         updates, opt_state = optimizer.update(grad, opt_state)
#         params = apply_updates(params, updates)
#         return (jnp.clip(params, 0., 1.), opt_state), val

#     best_f, best_params = jnp.inf, None
#     r = jnp.arange(maxiter)
#     for n in range(n_restarts_optimizer):
#         params = x0[n]
#         opt_state = optimizer.init(x0[n])
#         progress_bar = tqdm.tqdm(r,desc=f'ACQ Optimization restart {n+1}')
#         for i in progress_bar:
#             (params, opt_state), fval = step((params, opt_state))
#             progress_bar.set_postfix({"fval": float(fval)})
#             if fval < best_f:
#                 best_f, best_params = fval, params
#         # Perturb for next restart
#         # params = jnp.clip(best_params + 0.5 * jnp.array(np.random.normal(size=params.shape)), 0., 1.)

#     # print(f"Best params: {best_params}, fval: {best_f}")
#     return jnp.atleast_2d(best_params), best_f

# Utility functions

# def get_point_with_large_value(train_x,train_y, n_points=1):
#     """
#     Get a point with large value from the training data
#     """
#     idx = jnp.argsort(train_y.flatten())[-n_points:]
#     return train_x[idx].flatten()

def get_mc_samples(gp,warmup_steps=512, num_samples=512, thinning=4,method="NUTS",init_params=None):
    if method=='NUTS':
        try:
            mc_samples = gp.sample_GP_NUTS(warmup_steps=warmup_steps,
            num_samples=num_samples, thinning=thinning
            )
        except Exception as e:
            log.error(f"Error in sampling GP NUTS: {e}")
            mc_samples, logz, success = nested_sampling_Dy(gp, gp.ndim, maxcall=int(2e6)
                                            , dynamic=False, dlogz=0.5,equal_weights=True,
            )
    elif method=='NS':
        mc_samples, logz, success = nested_sampling_Dy(gp, gp.ndim, maxcall=int(2e6)
                                            , dynamic=False, dlogz=0.5,equal_weights=True,
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
                 acq = 'WIPV',
                 use_clf=True,
                 clf_type = "svm",
                 clf_use_size = 300,
                 clf_update_step=5,
                 clf_threshold=250,
                 gp_threshold=5000,
                 logz_threshold=1.0,
                 minus_inf=-1e5,
                 do_final_ns=True,
                 return_getdist_samples=False,
                 seed: Optional[int] = None):
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
        use_clf : bool
            If True, use SVM to filter the GP predictions. 
            This is only required for high dimensional problems and when the scale of variation of the likelihood is extremely large. 
            For cosmological likelihoods with nuisance parameters, this is highly recommended.
        clf_use_size : int
            Minimum size of the classifier training set before the classifier filter is used in the GP.
        clf_update_step : int
            Number of iterations between classifier updates.
        logz_threshold : float
            Threshold for convergence of the nested sampling logz. 
            If the difference between the upper and lower bounds of logz is less than this value, the sampling will end.
        minus_inf : float
            Value to use for minus infinity. This is used to set the lower bound of the loglikelihood.
        """


        set_global_seed(seed)

        if not isinstance(loglikelihood, ExternalLikelihood):
            raise ValueError("loglikelihood must be an instance of ExternalLikelihood")

        self.loglikelihood = loglikelihood
        self.ndim = len(self.loglikelihood.param_list)

        if resume and resume_file is not None:
            # assert resume_file is not None, "resume_file must be provided if resume is True"
            log.info(f" Resuming from file {resume_file}")
            data = np.load(resume_file)
            self.train_x = jnp.array(data['train_x'])
            self.train_y = jnp.array(data['train_y'])
            lengthscales = data['lengthscales']
            outputscale = data['outputscale']
        else:
            init_points, init_vals = self.loglikelihood.get_initial_points(n_cobaya_init=n_cobaya_init,
                                    n_init_sobol=n_sobol_init)
            self.train_x = jnp.array(scale_to_unit(init_points, self.loglikelihood.param_bounds))
            self.train_y = jnp.array(init_vals)
            lengthscales = None
            outputscale = None

        # Best point so far
        idx_best = jnp.argmax(self.train_y)
        self.best_pt = scale_from_unit(self.train_x[idx_best],self.loglikelihood.param_bounds).flatten()
        self.best_f = float(self.train_y.max())
        self.best = {name: f"{float(val):.4f}" for name, val in zip(self.loglikelihood.param_list, self.best_pt)}
        log.info(f" Initial best point {self.best} with value = {self.best_f:.4f}")

        # GP setup
        if use_clf:
            self.gp = ClassifierGP(
                train_x=self.train_x, train_y=self.train_y,
                minus_inf=minus_inf, lengthscale_priors=lengthscale_priors,
                clf_type=clf_type, clf_use_size=clf_use_size, clf_update_step=clf_update_step,
                clf_threshold=clf_threshold, gp_threshold=gp_threshold,
                lengthscales=lengthscales, outputscale=outputscale
            )
            # gp = SVM_GP(
            #     train_x=self.train_x, train_y=self.train_y,
            #     minus_inf=minus_inf, lengthscale_priors=lengthscale_priors,
            #     kernel='rbf',svm_use_size=svm_use_size,svm_update_step=svm_update_step,
            #     svm_threshold=svm_threshold,gp_threshold=svm_gp_threshold,lengthscales=lengthscales,
            #     outputscale=outputscale
            # )
        else:
            self.gp = {
                'DSLP': DSLP_GP,
                'SAAS': SAAS_GP
            }[lengthscale_priors](
                train_x=self.train_x, train_y=self.train_y,
                noise=1e-8, kernel='rbf',lengthscales=lengthscales,outputscale=outputscale
            )


        if resume:
            self.gp.fit(maxiter=100,n_restarts=2) # if resuming need not spend too much time on fitting
        else:
            # Fit the GP to the initial points
            self.gp.fit(maxiter=150,n_restarts=4)


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

        if self.save:
            self.gp.save(outfile=self.output_file)
            log.info(f" Saving GP to file {self.output_file}")

    def check_convergence(self, step, logz_dict, threshold=2.0,ndim=1):
        """
        Check if the nested sampling has converged.
        """
        if ndim > 10:
            delta = logz_dict['upper'] - logz_dict['lower'] # for now just to speed up results
        else:
            delta = logz_dict['upper'] - logz_dict['lower']
        mean = logz_dict['mean']
        if (delta < threshold and mean>self.minus_inf+10)  and step > self.miniters:
            log.info(f" Convergence check: delta = {delta:.4f}, step = {step}")
            log.info(" Converged")
            return True
        else:
            return False

    def run(self,maxsteps=1000,minsteps=0):
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


        results_dict = {}


        # Monte Carlo points for acquisition function
        self.mc_samples = get_mc_samples(self.gp,warmup_steps=self.num_hmc_warmup, num_samples=self.num_hmc_samples, 
                                         thinning=4,method=self.mc_points_method)
        self.mc_samples['method'] = 'MCMC'        
        self.mc_points = get_mc_points(self.mc_samples, self.mc_points_size)
        ns_samples = None
        logz_dict = None

        best_pt_iteration = 0

        ii = 0
        x0_acq =  self.gp.train_x[jnp.argmax(self.gp.train_y)]

        for i in range(self.maxiters):

            ii = i + 1
            refit = (ii % self.fit_step == 0)
            ns_flag = (ii % self.ns_step == 0) and ii >= self.miniters
            update_mc = (ii % self.update_mc_step == 0) and not ns_flag

            print("\n")
            log.info(f" Iteration {ii}/{self.maxiters}, refit={refit}, update_mc={update_mc}, ns={ns_flag}")
            
            # start acq from mc_point with max var or max value 
            # mc_points_var = jax.lax.map(self.gp.predict_var,self.mc_points)
            # x0_acq = self.mc_points[jnp.argmax(mc_points_var)]

            x0_acq1 = self.mc_samples['best']
            vars = jax.lax.map(self.gp.predict_var,self.mc_points,batch_size=10)
            x0_acq2 = self.mc_points[jnp.argmax(vars)]
            x0_acq3 = self.gp.train_x[jnp.argmax(self.gp.train_y)]
            x0_acq = jnp.vstack([x0_acq1, x0_acq2, x0_acq3])
            # print(f"x0_acq shape: {x0_acq.shape}, expected shape: (2, ndim)")
            # new_pt_u, acq_val = optimize_acq(
            #     self.gp, WIPV, self.mc_points, x0=x0_acq)

            new_pt_u, acq_val = optimize(WIPV, 
                                         func_args = (self.gp,), 
                                         func_kwargs = {'mc_points': self.mc_points},
                                         ndim = self.ndim,
                                         x0 = x0_acq,
                                         n_restarts=4,
                                         verbose=True,)
            new_pt_u = jnp.atleast_2d(new_pt_u)  # Ensure new_pt_u is at least 2D
            
            new_pt = scale_from_unit(new_pt_u, self.loglikelihood.param_bounds) #.flatten()

            log.info(f" Acquisition value {acq_val:.4e} at new point")
            new_val = self.loglikelihood(
                new_pt, logp_args=(), logp_kwargs={}
            )

            new_pt_vals = {name: f"{float(val):.4f}" for name, val in zip(self.loglikelihood.param_list, new_pt.flatten())}
            log.info(f" New point {new_pt_vals}")
            log.info(f" Objective function value = {new_val.item():.4f}, GP predicted value = {self.gp.predict_mean(new_pt_u).item():.4f}")

            pt_exists_or_below_threshold = self.gp.update(new_pt_u, new_val, refit=refit,step=ii,n_restarts=4)
            # x0_acq =  self.gp.train_x[jnp.argmax(self.gp.train_y)]

            if (pt_exists_or_below_threshold and self.mc_points_method == 'NUTS') and (self.mc_samples['method'] == 'MCMC'):
                update_mc = True
            if update_mc:
                x0_hmc = self.gp.train_x[jnp.argmax(self.gp.train_y)]
                self.mc_samples = get_mc_samples(
                    self.gp, warmup_steps=self.num_hmc_warmup, num_samples=self.num_hmc_samples,
                    thinning=4, method=self.mc_points_method,init_params=x0_hmc
                )
                self.mc_samples['method'] = 'MCMC'

            if float(new_val) > self.best_f:
                self.best_f = float(new_val)
                self.best_pt = new_pt
                self.best = {name: f"{float(val):.4f}" for name, val in zip(self.loglikelihood.param_list, self.best_pt.flatten())}
                best_pt_iteration = ii
            log.info(f" Current best point {self.best} with value = {self.best_f:.4f}, found at iteration {best_pt_iteration}")

            if i % 4 == 0 and i > 0:
                jax.clear_caches()

            if (ii % 10 == 0) and self.save:
                log.info(" Saving GP to file")
                self.gp.save(outfile=self.output_file)

            if ns_flag:
                log.info(" Running Nested Sampling")
                ns_samples, logz_dict, ns_success = nested_sampling_Dy(
                    self.gp, self.ndim, maxcall=int(5e6), dynamic=False, dlogz=0.1
                )
                log.info(" LogZ info: " + ", ".join([f"{k}={v:.4f}" for k,v in logz_dict.items()]))
                self.converged = self.check_convergence(ii, logz_dict, threshold=self.logz_threshold,ndim=self.ndim)
                if ns_success and self.converged:
                    self.converged = True
                    self.termination_reason = "LogZ converged"
                    results_dict['logz'] = logz_dict
                    results_dict['termination_reason'] = self.termination_reason
                    break

                self.mc_samples = ns_samples
                self.mc_samples['method'] = 'NS'

            self.mc_points = get_mc_points(self.mc_samples, self.mc_points_size)

            if self.gp.train_x.shape[0] >= self.max_gp_size:
                self.termination_reason = "Max GP size reached"
                log.info(f" {self.termination_reason}")
                break
            if self.gp.train_x.shape[0] > 1600:
                self.ns_step = 25

        log.info(f" Sampling stopped: {self.termination_reason}")
        log.info(f" Final GP training set size: {self.gp.train_x.shape[0]}, max size: {self.max_gp_size}")
        log.info(f" Number of iterations: {ii}, max iterations: {self.maxiters}")


        if not self.converged:
            self.gp.fit()


        results_dict['gp'] = self.gp

        # Save and final nested sampling
        if self.save:
            self.gp.save(outfile=self.output_file)

        # Prepare final results 
        if self.do_final_ns and not self.converged:
            log.info(" Final Nested Sampling")
            ns_samples, logz_dict, ns_success = nested_sampling_Dy(
                self.gp, self.ndim, maxcall=int(1e7), dynamic=True, dlogz=0.01
            )
            log.info(" Final LogZ: " + ", ".join([f"{k}={v:.4f}" for k,v in logz_dict.items()]))


        if ns_samples is None:
            log.info("No nested sampling results found, MC samples from HMC/MCMC will be used instead.")
            mc_samples = get_mc_samples(
                    self.gp, warmup_steps=512, num_samples=16384,
                    thinning=4, method="NUTS")
            samples = mc_samples['x']
            weights = mc_samples['weights'] if 'weights' in mc_samples else jnp.ones(mc_samples['x'].shape[0])
            loglikes = mc_samples['logp']
        else:
            samples = ns_samples['x']
            weights = ns_samples['weights']
            loglikes = ns_samples['logl']
            log.info(f"Using nested sampling results")

        samples = scale_from_unit(samples, self.loglikelihood.param_bounds)
        samples_dict = {
            'x': samples,
            'weights': weights,
            'logl': loglikes
        }

        if self.save:
            np.savez(f'{self.output_file}_samples.npz',
                     samples=samples,param_bounds=self.loglikelihood.param_bounds,
                     weights=weights,loglikes=loglikes)

        if self.return_getdist_samples:
            sampler_method = 'nested' if ns_samples is not None else 'mcmc'
            ranges = dict(zip(self.loglikelihood.param_list,self.loglikelihood.param_bounds.T))
            gd_samples = MCSamples(samples=samples, names=self.loglikelihood.param_list, labels=self.loglikelihood.param_labels, 
                                ranges=ranges, weights=weights,loglikes=loglikes,label='GP',sampler=sampler_method)
            output_samples = gd_samples
            log.info(f"Returning getdist samples with method {sampler_method}")
        else:
            output_samples = samples_dict
                

        results_dict['samples'] = output_samples

        return results_dict

