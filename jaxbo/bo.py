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
from scipy import stats
# from .acquisition import WIPV, EI #, logEI
from .gp import GP, DSLP_GP, SAAS_GP, load_gp #, sample_GP_NUTS
from .clf_gp import GPwithClassifier, load_clf_gp
from .loglike import ExternalLikelihood, CobayaLikelihood
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from .utils.core_utils import scale_from_unit, scale_to_unit, renormalise_log_weights, resample_equal, compute_kl_divergences, compute_successive_kl
from .utils.seed_utils import set_global_seed, get_global_seed, get_jax_key, split_jax_key, ensure_reproducibility
from .nested_sampler import nested_sampling_Dy, nested_sampling_jaxns
from .optim import optimize
from .utils.logging_utils import get_logger
from .utils.results import BOBEResults
from .acquisition import *

log = get_logger("[bo]")
log.info(f'JAX using {jax.device_count()} devices.')

_acq_funcs = {"wipv": WIPV, "ei": EI, "logei": LogEI}

import numpy as np
from scipy import stats
import warnings


class BOBE:

    def __init__(self,
                loglikelihood=None,
                 n_cobaya_init=4,
                 n_sobol_init=32,
                 min_iters=200,
                 max_eval_budget=1500,
                 max_gp_size=1200,
                 resume=False,
                 resume_file=None,
                 save=True,
                 noise = 1e-8,
                 fit_step=10,
                 update_mc_step=10,
                 ns_step=10,
                 num_hmc_warmup=512,
                 num_hmc_samples=512,
                 mc_points_size=64,
                 mc_points_method='NUTS',
                 lengthscale_priors='DSLP',
                 acq = 'WIPV',
                 zeta_ei = 0.1,
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
        min_iters : int
            Minimum number of iterations before checking convergence.
        max_eval_budget : int
            Maximum number of true objective function evaluations.
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

        # Store basic settings needed for results manager early
        self.output_file = self.loglikelihood.name
        self.save = save
        self.return_getdist_samples = return_getdist_samples
        self.do_final_ns = do_final_ns
        self.logz_threshold = logz_threshold
        self.converged = False
        self.prev_converged = False
        self.termination_reason = "Max evaluation budget reached"

        # Initialize results manager BEFORE any timing operations
        self.results_manager = BOBEResults(
            output_file=self.output_file,
            param_names=self.loglikelihood.param_list,
            param_labels=self.loglikelihood.param_labels,
            param_bounds=self.loglikelihood.param_bounds,
            settings={
                'n_cobaya_init': n_cobaya_init,
                'n_sobol_init': n_sobol_init,
                'min_iters': min_iters,
                'max_eval_budget': max_eval_budget,
                'max_gp_size': max_gp_size,
                'fit_step': fit_step,
                'update_mc_step': update_mc_step,
                'ns_step': ns_step,
                'num_hmc_warmup': num_hmc_warmup,
                'num_hmc_samples': num_hmc_samples,
                'mc_points_size': mc_points_size,
                'mc_points_method': mc_points_method,
                'lengthscale_priors': lengthscale_priors,
                'acq': acq,
                'use_clf': use_clf,
                'clf_type': clf_type,
                'clf_use_size': clf_use_size,
                'clf_threshold': clf_threshold,
                'clf_update_step': clf_update_step,
                'gp_threshold': gp_threshold,
                'logz_threshold': logz_threshold,
                'minus_inf': minus_inf,
                'do_final_ns': do_final_ns,
                'return_getdist_samples': return_getdist_samples,
                'seed': seed
            },
            likelihood_name=self.loglikelihood.name,
            resume_from_existing=resume
        )

        # Check if we're resuming from a file or from existing results
        if resume and resume_file is not None:
            # Resume from explicit file
            log.info(f" Resuming from file {resume_file}")
            # Use the standard naming convention: add _gp if not present
            if not resume_file.endswith('_gp') and not resume_file.endswith('_gp.npz'):
                gp_file = f"{resume_file}_gp"
            else:
                gp_file = resume_file
            if use_clf:
                self.gp = load_clf_gp(gp_file)
            else:
                self.gp = load_gp(gp_file)                
        else:
            # Fresh start - evaluate initial points
            self.results_manager.start_timing('True Objective Evaluations')
            init_points, init_vals = self.loglikelihood.get_initial_points(n_cobaya_init=n_cobaya_init,
                                    n_init_sobol=n_sobol_init)
            self.results_manager.end_timing('True Objective Evaluations')            
            train_x = jnp.array(scale_to_unit(init_points, self.loglikelihood.param_bounds))
            train_y = jnp.array(init_vals)
            # GP setup
            if use_clf:
                self.gp = GPwithClassifier(
                train_x=train_x, train_y=train_y,noise=noise,
                minus_inf=minus_inf, lengthscale_priors=lengthscale_priors,
                clf_type=clf_type, clf_use_size=clf_use_size, clf_update_step=clf_update_step,
                clf_threshold=clf_threshold, gp_threshold=gp_threshold,)
            else:
                self.gp = {
                    'UNIFORM': GP,
                    'DSLP': DSLP_GP,
                    'SAAS': SAAS_GP
                }[lengthscale_priors.upper()](
                train_x=train_x, train_y=train_y,
                noise=noise, kernel='rbf',)
            self.results_manager.start_timing('GP Training')
            self.gp.fit(maxiter=200,n_restarts=4)
            self.results_manager.end_timing('GP Training')

        idx_best = jnp.argmax(self.gp.train_y)
        self.best_pt = scale_from_unit(self.gp.train_x[idx_best], self.loglikelihood.param_bounds).flatten()
        self.best_f = float(self.gp.train_y.max()) * self.gp.y_std + self.gp.y_mean

        self.best = {name: f"{float(val):.4f}" for name, val in zip(self.loglikelihood.param_list, self.best_pt)}
        log.info(f" Initial best point {self.best} with value = {self.best_f:.4f}")

        # Store remaining settings
        self.max_eval_budget = max_eval_budget
        self.min_iters = min_iters
        self.max_gp_size = max_gp_size
        self.fit_step = fit_step
        self.update_mc_step = update_mc_step
        self.ns_step = ns_step
        self.num_hmc_warmup = num_hmc_warmup
        self.num_hmc_samples = num_hmc_samples
        self.mc_points_size = mc_points_size
        self.minus_inf = minus_inf
        self.mc_points_method = mc_points_method
        self.zeta_ei = zeta_ei

        if self.save:
            self.gp.save(outfile=f"{self.output_file}_gp")
            log.info(f" Saving GP to file {self.output_file}_gp")

        # Initialize KL divergence tracking
        self.prev_samples = None


    def run(self, n_log_ei_iters = 100):
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

        # Initial Monte Carlo points for acquisition function
        self.results_manager.start_timing('MCMC Sampling')
        self.mc_samples = get_mc_samples(self.gp,warmup_steps=self.num_hmc_warmup, num_samples=self.num_hmc_samples, 
                                         thinning=4,method=self.mc_points_method)
        self.results_manager.end_timing('MCMC Sampling')
        self.mc_samples['method'] = 'MCMC'        
        self.mc_points = get_mc_points(self.mc_samples, self.mc_points_size)
        ns_samples = None
        logz_dict = None

        ns_success=False

        best_pt_iteration = 0

        # Check if resuming and adjust starting iteration
        if self.results_manager.is_resuming():
            start_iteration = self.results_manager.get_last_iteration()
            log.info(f"Resuming from iteration {start_iteration}")
            log.info(f"Previous data: {len(self.results_manager.acquisition_values)} acquisition evaluations")
            
            # If we have previous best loglikelihood data, restore the best point info
            if self.results_manager.best_loglike_values:
                self.best_f = max(self.results_manager.best_loglike_values)
                best_loglike_idx = self.results_manager.best_loglike_values.index(self.best_f)
                best_pt_iteration = self.results_manager.best_loglike_iterations[best_loglike_idx]
                log.info(f"Restored best loglikelihood: {self.best_f:.4f} at iteration {best_pt_iteration}")
        else:
            start_iteration = 0
            log.info("Starting fresh optimization")

        ii = start_iteration

        log.info(f"Starting iteration {ii}")

        self.acquisition = LogEI() # start with LogEI

        current_evals = self.gp.npoints  # Number of evaluations so far

        while current_evals < self.max_eval_budget:

            # ideally, we want to decide whether to do the mc_update depending on the results of the previous steps
            #  e.g. if using ns_samples we can stay on it for a bit longer since it explores the space better
            
            ii+=1
            refit = (ii % self.fit_step == 0)
            ns_flag = (ii % self.ns_step == 0) and ii >= self.min_iters
            update_mc = not ns_flag # (ii % self.update_mc_step == 0) and

            if (ii - start_iteration > n_log_ei_iters) and self.acquisition.name in ['EI','LogEI']:
                # change acquisition function to WIPV after a minimum of n_log_ei_iters EI, LogEI
                self.acquisition = WIPV()

            acq_str = self.acquisition.name

            print("\n")
            log.info(f" Iteration {ii}, objective evals {current_evals}/{self.max_eval_budget}, refit={refit}, update_mc={update_mc}, ns={ns_flag}, acq={acq_str}")


            self.results_manager.start_timing('Acquisition Optimization')
            if acq_str == 'WIPV':
                acq_kwargs = {'mc_samples': self.mc_samples, 'mc_points_size': self.mc_points_size}
                n_restarts = 1
                maxiter = 200
                early_stop_patience = 25
                n_batch = self.update_mc_step # since we only need to update true GP before doing the next MCMC
            else:
                acq_kwargs = {'zeta': self.zeta_ei, 'best_y': max(self.gp.train_y.flatten())}
                n_restarts = 10
                maxiter = 500
                early_stop_patience = 50
                n_batch = 1

            new_pts_u, acq_vals = self.acquisition.get_next_batch(gp = self.gp, 
                                                                  n_batch = n_batch,
                                                                  acq_kwargs=acq_kwargs,
                                                                  n_restarts=n_restarts, 
                                                                  maxiter=maxiter, 
                                                                  early_stop_patience=early_stop_patience)
            self.results_manager.end_timing('Acquisition Optimization')
            new_pts_u = jnp.atleast_2d(new_pts_u)  # Ensure new_pt_u is at least 2D
            
            new_pts = scale_from_unit(new_pts_u, self.loglikelihood.param_bounds)

            acq_val = float(np.mean(acq_vals))

            log.info(f" Acquisition value {acq_val:.4e} at new point")
            self.results_manager.update_acquisition(ii, acq_val, acq_str)

            self.results_manager.start_timing('True Objective Evaluations')
            new_vals = self.loglikelihood(
                new_pts, logp_args=(), logp_kwargs={}
            )
            current_evals += n_batch
            self.results_manager.end_timing('True Objective Evaluations')

            for k in range(n_batch):
                new_pt_vals = {name: f"{float(val):.4f}" for name, val in zip(self.loglikelihood.param_list, new_pts[k].flatten())}
                log.info(f" New point {new_pt_vals}, {k+1}/{n_batch}")
                log.info(f" Objective function value = {new_vals[k].item():.4f}, GP predicted value = {self.gp.predict_mean(new_pts_u[k]).item():.4f}")

            # GP Training and timing
            if refit:
                self.results_manager.start_timing('GP Training')
            pt_exists_or_below_threshold = self.gp.update(new_pts_u, new_vals, refit=refit,step=ii,n_restarts=4)
            if refit:
                self.results_manager.end_timing('GP Training')
            log.info(f"New GP y_mean: {self.gp.y_mean:.4f}, y_std: {self.gp.y_std:.4f}")
            log.info("Updated GP with new point.")
            log.info(f" GP training size = {self.gp.npoints}")


            # Extract GP hyperparameters for tracking
            lengthscales = list(self.gp.lengthscales)
            outputscale = float(self.gp.outputscale)
            self.results_manager.update_gp_hyperparams(ii, lengthscales, outputscale)

            # Update results manager with iteration info (simplified)
            self.results_manager.update_iteration(iteration=ii)

            if (pt_exists_or_below_threshold and self.mc_samples['method'] == 'MCMC'):
                update_mc = True
            if update_mc and acq_str == 'WIPV':
                if not refit:
                    self.gp.fit(maxiter=50,n_restarts=1)
                self.results_manager.start_timing('MCMC Sampling')
                self.mc_samples = get_mc_samples(
                    self.gp, warmup_steps=self.num_hmc_warmup, num_samples=self.num_hmc_samples,
                    thinning=4, method=self.mc_points_method)
                self.results_manager.end_timing('MCMC Sampling')


            best_new_idx = np.argmax(new_vals)
            best_new_val = float(np.max(new_vals))
            best_new_pt = new_pts[best_new_idx]
            if float(best_new_val) > self.best_f:
                self.best_f = float(best_new_val)
                self.best_pt = best_new_pt
                self.best = {name: f"{float(val):.4f}" for name, val in zip(self.loglikelihood.param_list, self.best_pt.flatten())}
                best_pt_iteration = ii
            
            # Track best loglikelihood evolution
            self.results_manager.update_best_loglike(ii, self.best_f)
            
            log.info(f" Current best point {self.best} with value = {self.best_f:.4f}, found at iteration {best_pt_iteration}")

            if ii % 4 == 0 and ii > 0:
                jax.clear_caches()
 
            if (ii % 10 == 0) and self.save:
                log.info(" Saving GP to file")
                self.gp.save(outfile=f"{self.output_file}_gp")

            if ns_flag:
                log.info(" Running Nested Sampling")
                self.results_manager.start_timing('Nested Sampling')
                ns_samples, logz_dict, ns_success = nested_sampling_Dy(
                    self.gp, self.ndim, maxcall=int(5e6), dynamic=False, dlogz=0.01,equal_weights=False
                )
                self.results_manager.end_timing('Nested Sampling')

                log.info(f" NS success = {ns_success}, LogZ info: " + ", ".join([f"{k}={v:.4f}" for k,v in logz_dict.items()]))
                # now get equally weighted samples for mc points
                equal_samples, equal_logl = resample_equal(ns_samples['x'], ns_samples['logl'], weights=ns_samples['weights'])
                self.mc_samples = {
                    'x': equal_samples,
                    'logl': equal_logl,
                    'weights': np.ones(equal_samples.shape[0]),
                    'method': 'NS',
                    'best': ns_samples['best']
                }
                if ns_success:
                    self.converged = self.check_convergence(ii, self.gp, logz_dict, ns_samples, threshold=self.logz_threshold)
                    if self.converged:
                        self.termination_reason = "LogZ converged"
                        results_dict['logz'] = logz_dict
                        results_dict['termination_reason'] = self.termination_reason
                        break

            self.mc_points = get_mc_points(self.mc_samples, self.mc_points_size)

            if self.gp.train_x.shape[0] >= self.max_gp_size:
                self.termination_reason = "Max GP size reached"
                log.info(f" {self.termination_reason}")
                break
            if self.gp.train_x.shape[0] > 1800:
                self.ns_step = 25


        #-------End of BO loop-------

        log.info(f" Sampling stopped: {self.termination_reason}")
        log.info(f" Final GP training set size: {self.gp.train_x.shape[0]}, max size: {self.max_gp_size}")
        # log.info(f" Number of iterations: {ii}, max iterations: {self.max_eval_budget}")


        if not self.converged:
            self.results_manager.start_timing('GP Training')
            self.gp.fit()
            self.results_manager.end_timing('GP Training')

        results_dict['gp'] = self.gp

        # Save and final nested sampling
        if self.save:
            self.gp.save(outfile=f"{self.output_file}_gp")

        # Prepare final results 
        if self.do_final_ns and not self.converged:
            log.info(" Final Nested Sampling")
            self.results_manager.start_timing('Nested Sampling')
            ns_samples, logz_dict, ns_success = nested_sampling_Dy(
                self.gp, self.ndim, maxcall=int(1e7), dynamic=True, dlogz=0.01
            )
            self.results_manager.end_timing('Nested Sampling')
            log.info(" Final LogZ: " + ", ".join([f"{k}={v:.4f}" for k,v in logz_dict.items()]))
            if ns_success:
                log.info(f"Using nested sampling results")
                self.check_convergence(ii+1, self.gp, logz_dict, ns_samples, threshold=self.logz_threshold)
                if self.converged:
                    self.termination_reason = "LogZ converged"
                    results_dict['logz'] = logz_dict
                    results_dict['termination_reason'] = self.termination_reason

        samples = ns_samples['x']
        weights = ns_samples['weights']
        loglikes = ns_samples['logl']

        if not ns_success:
        # if not self.do_final_ns:
            log.info("No nested sampling results found, MC samples from HMC/MCMC will be used instead.")
            self.results_manager.start_timing('MCMC Sampling')
            mc_samples = get_mc_samples(
                    self.gp, warmup_steps=512, num_samples=1000*self.ndim,
                    thinning=4, method="NUTS")
            self.results_manager.end_timing('MCMC Sampling')
            samples = mc_samples['x']
            weights = mc_samples['weights'] if 'weights' in mc_samples else np.ones(mc_samples['x'].shape[0])
            loglikes = mc_samples['logp']
                
        samples = scale_from_unit(samples, self.loglikelihood.param_bounds)
        samples_dict = {
            'x': samples,
            'weights': weights,
            'logl': loglikes
        }

        # Extract GP and classifier information
        gp_info = {
            'gp_training_set_size': int(self.gp.train_x.shape[0]),
            'gp_final_best_loglike': float(self.best_f),  # Best value in true physical space
        }
        
        # Add classifier info if using GPwithClassifier
        if hasattr(self.gp, 'clf_flag'):
            gp_info.update({
                'classifier_used': bool(self.gp.clf_flag and self.gp.use_clf),
                'classifier_type': str(self.gp.clf_type) if self.gp.clf_flag else None,
                'classifier_training_set_size': int(self.gp.clf_data_size) if self.gp.clf_flag else 0,
                'classifier_use_threshold': int(self.gp.clf_use_size) if self.gp.clf_flag else None,
                'classifier_probability_threshold': float(self.gp.probability_threshold) if self.gp.clf_flag else None
            })
        else:
            gp_info.update({
                'classifier_used': False,
                'classifier_type': None,
                'classifier_training_set_size': 0
            })

        # Finalize results with comprehensive data
        self.results_manager.finalize(
            samples=samples,
            weights=weights,
            loglikes=loglikes,
            logz_dict=logz_dict,
            converged=self.converged,
            termination_reason=self.termination_reason,
            gp_info=gp_info
        )

        # Print timing summary
        timing_summary = self.results_manager.get_timing_summary()
        log.info(f"\n{'='*50}")
        log.info(f"TIMING SUMMARY")
        log.info(f"{'='*50}")
        log.info(f"Total Runtime: {timing_summary['total_runtime']:.2f} seconds ({timing_summary['total_runtime']/60:.2f} minutes)")
        for phase, time_spent in timing_summary['phase_times'].items():
            if time_spent > 0:
                percentage = timing_summary['percentages'].get(phase, 0)
                log.info(f"{phase}: {time_spent:.2f}s ({percentage:.1f}%)")
        log.info(f"{'='*50}")

        # Legacy save for backward compatibility
        if self.save:
            np.savez(f'{self.output_file}_samples.npz',
                     samples=samples,param_bounds=self.loglikelihood.param_bounds,
                     weights=weights,loglikes=loglikes)

        if self.return_getdist_samples:
            # Use the results manager to create GetDist samples
            output_samples = self.results_manager.get_getdist_samples()
            if output_samples is None:
                # Fallback to manual creation if GetDist not available
                sampler_method = 'nested' if ns_samples is not None else 'mcmc'
                ranges = dict(zip(self.loglikelihood.param_list,self.loglikelihood.param_bounds.T))
                gd_samples = MCSamples(samples=samples, names=self.loglikelihood.param_list, labels=self.loglikelihood.param_labels, 
                                    ranges=ranges, weights=weights,loglikes=loglikes,label='GP',sampler=sampler_method)
                output_samples = gd_samples
            log.info(f"Returning getdist samples")
        else:
            output_samples = samples_dict

        # Get comprehensive results from results manager
        comprehensive_results = self.results_manager.get_results_dict()
        
        # Prepare return dictionary with both legacy and new format
        results_dict['samples'] = output_samples
        results_dict['gp'] = self.gp
        
        # Add comprehensive results
        results_dict['comprehensive'] = comprehensive_results
        results_dict['results_manager'] = self.results_manager
        
        # Add evidence info if available
        if logz_dict:
            results_dict['logz'] = logz_dict

        return results_dict

    def check_convergence(self, step, gp, logz_dict, ns_samples, threshold=1.0):
        """
        Check if the nested sampling has converged and compute KL divergence metrics.
        
        Args:
            step: Current iteration number
            logz_dict: Dictionary with logz bounds and mean
            ns_samples: Nested sampling samples with x, weights, logl
            threshold: LogZ convergence threshold
            
        Returns:
            bool: Whether convergence is achieved based on logz only
        """
        # Standard logz convergence check
        delta = logz_dict['std'] #logz_dict['upper'] - logz_dict['lower'] # # 
        converged = delta < threshold
        
        # Compute KL divergences if we have nested sampling samples
        successive_kl = None
        
        if ns_samples is not None:
            try:
                # Get the three likelihood estimates from logz bounds
                log_weights = np.log(ns_samples['weights'] + 1e-300)  # Avoid log(0)
                logl = ns_samples['logl']

                # Compute successive KL if we have previous samples
                if self.prev_samples is not None:
                    # compare different iterations with the equal weighted samples at the previous iteration.
                    prev_logl = self.prev_samples['logl']
                    prev_samples_x = self.prev_samples['x']
                    new_logl = jax.lax.map(gp.predict_mean_single,prev_samples_x,batch_size=200)
                    log_weights = np.zeros_like(new_logl)
                    successive_kl = compute_successive_kl(prev_logl, new_logl, log_weights)

                # Store current samples for next iteration
                equal_prev_samples, equal_prev_logl = resample_equal(ns_samples['x'], logl, ns_samples['weights'])
                self.prev_samples = {'x': equal_prev_samples, 'logl': equal_prev_logl}

                if successive_kl:
                    log.info(f" Successive KL: symmetric={successive_kl.get('symmetric', 0):.4f}")

            except Exception as e:
                log.warning(f"Could not compute KL divergences: {e}")
                successive_kl = None
        
        # Update results manager with convergence info and KL divergences
        self.results_manager.update_convergence(
            iteration=step,
            logz_dict=logz_dict,
            converged=converged,
            threshold=threshold
        )
        
        # Store KL divergences if computed
        if successive_kl is not None:
            self.results_manager.update_kl_divergences(
                iteration=step,
                successive_kl=successive_kl
            )

        log.info(f" Convergence check: delta = {delta:.4f}, step = {step}, threshold = {threshold}")
        if converged:
            if self.prev_converged:
                log.info(" Convergence achieved after 2 successive iterations")
                return True
            else:
                self.prev_converged = True
                log.info(f" Convergence not yet achieved in successive iterations")
                return False
        else:
            self.prev_converged = False
            return False