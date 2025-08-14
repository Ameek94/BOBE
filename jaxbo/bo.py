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
from .gp import DSLP_GP, SAAS_GP #, sample_GP_NUTS
from .clf_gp import GPwithClassifier
from .loglike import ExternalLikelihood, CobayaLikelihood
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from .utils.core_utils import scale_from_unit, scale_to_unit, renormalise_log_weights, resample_equal
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
            Maximum number of iterations. # Instead change to MAX_EVAL_BUDGET
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
        self.termination_reason = "Max iterations reached"

        # Initialize results manager BEFORE any timing operations
        self.results_manager = BOBEResults(
            output_file=self.output_file,
            param_names=self.loglikelihood.param_list,
            param_labels=self.loglikelihood.param_labels,
            param_bounds=self.loglikelihood.param_bounds,
            settings={
                'n_cobaya_init': n_cobaya_init,
                'n_sobol_init': n_sobol_init,
                'miniters': miniters,
                'maxiters': maxiters,
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
            likelihood_name=self.loglikelihood.name
        )

        if resume and resume_file is not None:
            # assert resume_file is not None, "resume_file must be provided if resume is True"
            log.info(f" Resuming from file {resume_file}")
            data = np.load(resume_file)
            self.train_x = jnp.array(data['train_x'])
            self.train_y = jnp.array(data['train_y'])
            lengthscales = data['lengthscales']
            outputscale = data['outputscale']
        else:
            # Time the initial points evaluation
            self.results_manager.start_timing('True Objective Evaluations')
            init_points, init_vals = self.loglikelihood.get_initial_points(n_cobaya_init=n_cobaya_init,
                                    n_init_sobol=n_sobol_init)
            self.results_manager.end_timing('True Objective Evaluations')
            
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
            self.gp = GPwithClassifier(
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
            self.results_manager.start_timing('GP Training')
            self.gp.fit(maxiter=100,n_restarts=2) # if resuming need not spend too much time on fitting
            self.results_manager.end_timing('GP Training')
        else:
            # Fit the GP to the initial points
            self.results_manager.start_timing('GP Training')
            self.gp.fit(maxiter=150,n_restarts=4)
            self.results_manager.end_timing('GP Training')


        # Store remaining settings
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
        self.mc_points_method = mc_points_method

        if self.save:
            self.gp.save(outfile=self.output_file)
            log.info(f" Saving GP to file {self.output_file}")

    def check_convergence(self, step, logz_dict, threshold=1.0):
        """
        Check if the nested sampling has converged.
        """
        # if ndim > 10:
        #     delta = logz_dict['upper'] - logz_dict['lower'] # for now just to speed up results
        # else:
        delta = logz_dict['upper'] - logz_dict['lower']
        
        converged = delta < threshold
        
        # Update results manager with convergence info
        self.results_manager.update_convergence(
            iteration=step,
            logz_dict=logz_dict,
            converged=converged,
            threshold=threshold
        )
        log.info(f" Convergence check: delta = {delta:.4f}, step = {step}, threshold = {threshold}")
        if converged:
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

        ii = 0
        x0_acq =  self.gp.train_x[jnp.argmax(self.gp.train_y)]

        acqf = WIPV
        acq_str = "WIPV"

        for i in range(self.maxiters):


            # ideally, we want to decide whether to do the mc_update depending on the results of the previous steps
            #  e.g. if using ns_samples we can stay on it for a bit longer since it explores the space better
            ii = i + 1
            # Change acquisition function if no improvement in 100 iterations.
            # if (ii - best_pt_iteration > 40)  and acqf == LogEI:
            #     log.info(f" No improvement in 100 iterations, changing acquisition function from {acq_str} to WIPV")
            #     acq_str = "WIPV"
            #     acqf = WIPV
            #     update_mc = True

            refit = (ii % self.fit_step == 0)
            ns_flag = (ii % self.ns_step == 0) and ii >= self.miniters
            update_mc = (ii % self.update_mc_step == 0) and not ns_flag

            print("\n")
            log.info(f" Iteration {ii}/{self.maxiters}, refit={refit}, update_mc={update_mc}, ns={ns_flag}, acq={acq_str}")


            if acqf == LogEI:
                acq_str = "LogEI"
                n_restarts = 8
                maxiter = 500
                early_stop_patience = 100
                acq_kwargs = {'zeta': 0.05}
                x0_acq = jnp.vstack([self.gp.get_random_point() for _ in range(n_restarts)])
            elif acqf == WIPV: 
                acq_str = "WIPV"
                n_restarts = 6  
                maxiter = 150
                early_stop_patience = 25
                acq_kwargs = {'mc_points': self.mc_points}
                if self.mc_samples is not None:
                    x0_acq1 = self.mc_samples['best']
                    vars = jax.lax.map(self.gp.predict_var,self.mc_points,batch_size=25)
                    x0_acq2 = self.mc_points[jnp.argmax(vars)]
                    x0_acq3 = self.gp.train_x[jnp.argmax(self.gp.train_y)]
                    x0_acq = jnp.vstack([x0_acq1, x0_acq2, x0_acq3])
                    x0_acq = jnp.vstack([x0_acq, [self.gp.get_random_point() for _ in range(n_restarts - 3)] ])
                else:
                    x0_acq = jnp.vstack([self.gp.get_random_point() for _ in range(n_restarts)])


            self.results_manager.start_timing('Acquisition Optimization')
            new_pt_u, acq_val = optimize(acqf, 
                                         fun_args = (self.gp,), 
                                         fun_kwargs = acq_kwargs,
                                         ndim = self.ndim,
                                         x0 = x0_acq,
                                         n_restarts=n_restarts,
                                         maxiter=maxiter,
                                         early_stop_patience=early_stop_patience,
                                         verbose=True,)
            self.results_manager.end_timing('Acquisition Optimization')
            new_pt_u = jnp.atleast_2d(new_pt_u)  # Ensure new_pt_u is at least 2D
            
            new_pt = scale_from_unit(new_pt_u, self.loglikelihood.param_bounds) #.flatten()

            log.info(f" Acquisition value {acq_val:.4e} at new point")
            self.results_manager.update_acquisition(ii, acq_val, acq_str)

            self.results_manager.start_timing('True Objective Evaluations')
            new_val = self.loglikelihood(
                new_pt, logp_args=(), logp_kwargs={}
            )
            self.results_manager.end_timing('True Objective Evaluations')

            new_pt_vals = {name: f"{float(val):.4f}" for name, val in zip(self.loglikelihood.param_list, new_pt.flatten())}
            log.info(f" New point {new_pt_vals}")
            log.info(f" Objective function value = {new_val.item():.4f}, GP predicted value = {self.gp.predict_mean(new_pt_u).item():.4f}")

            # Extract GP hyperparameters for tracking
            try:
                gp_obj = self.gp           

                lengthscales = gp_obj.lengthscales
                outputscale = gp_obj.outputscale
                gp_hyperparams = {
                    'lengthscales': lengthscales.tolist() if hasattr(lengthscales, 'tolist') else float(lengthscales),
                    'outputscale': float(outputscale)
                }
                lengthscales_list = gp_hyperparams['lengthscales'] if isinstance(gp_hyperparams['lengthscales'], list) else [gp_hyperparams['lengthscales']]
                self.results_manager.update_gp_hyperparams(ii, lengthscales_list, gp_hyperparams['outputscale'])
                log.debug(f"Saved GP hyperparameters: {gp_hyperparams}")
        
                # if hasattr(gp_obj, 'lengthscales') and hasattr(gp_obj, 'outputscale'):
                #     gp_hyperparams = {
                #         'lengthscales': gp_obj.lengthscales.tolist() if hasattr(gp_obj.lengthscales, 'tolist') else float(gp_obj.lengthscales),
                #         'outputscale': float(gp_obj.outputscale)
                #     }
                #     if hasattr(gp_obj, 'tausq'):
                #         gp_hyperparams['tausq'] = float(gp_obj.tausq)
                    
                #     # Track GP hyperparameters evolution
                #     lengthscales_list = gp_hyperparams['lengthscales'] if isinstance(gp_hyperparams['lengthscales'], list) else [gp_hyperparams['lengthscales']]
                #     self.results_manager.update_gp_hyperparams(ii, lengthscales_list, gp_hyperparams['outputscale'])
                #     log.info(f"Saved GP hyperparameters: {gp_hyperparams}")
                # else:
                #     gp_hyperparams = None
                #     log.info("GP hyperparameters not available or not in expected format.")
            except:
                gp_hyperparams = None
                log.error("Error extracting GP hyperparameters, they may not be available in this GP implementation.")

            # Update results manager with iteration info (simplified)
            self.results_manager.update_iteration(iteration=ii)

            # GP Training timing
            if refit:
                self.results_manager.start_timing('GP Training')
            pt_exists_or_below_threshold = self.gp.update(new_pt_u, new_val, refit=refit,step=ii,n_restarts=4)
            if refit:
                self.results_manager.end_timing('GP Training')
            # x0_acq =  self.gp.train_x[jnp.argmax(self.gp.train_y)]

            if (pt_exists_or_below_threshold and self.mc_points_method == 'NUTS') and (self.mc_samples['method'] == 'MCMC'):
                update_mc = True
            if update_mc:
                # if not refit:
                #     self.gp.fit(maxiter=75,n_restarts=1)
                x0_hmc = self.gp.train_x[jnp.argmax(self.gp.train_y)]
                self.results_manager.start_timing('MCMC Sampling')
                self.mc_samples = get_mc_samples(
                    self.gp, warmup_steps=self.num_hmc_warmup, num_samples=self.num_hmc_samples,
                    thinning=4, method=self.mc_points_method,init_params=x0_hmc
                )
                self.results_manager.end_timing('MCMC Sampling')
                self.mc_samples['method'] = 'MCMC'

            if float(new_val) > self.best_f:
                self.best_f = float(new_val)
                self.best_pt = new_pt
                self.best = {name: f"{float(val):.4f}" for name, val in zip(self.loglikelihood.param_list, self.best_pt.flatten())}
                best_pt_iteration = ii
            
            # Track best loglikelihood evolution
            self.results_manager.update_best_loglike(ii, self.best_f)
            
            log.info(f" Current best point {self.best} with value = {self.best_f:.4f}, found at iteration {best_pt_iteration}")



            if i % 4 == 0 and i > 0:
                jax.clear_caches()

            if (ii % 10 == 0) and self.save:
                log.info(" Saving GP to file")
                self.gp.save(outfile=self.output_file)

            if ns_flag:
                log.info(" Running Nested Sampling")
                self.results_manager.start_timing('Nested Sampling')
                ns_samples, logz_dict, ns_success = nested_sampling_Dy(
                    self.gp, self.ndim, maxcall=int(5e6), dynamic=False, dlogz=0.01,equal_weights=False
                )
                self.results_manager.end_timing('Nested Sampling')

                log.info(f" NS success = {ns_success}, LogZ info: " + ", ".join([f"{k}={v:.4f}" for k,v in logz_dict.items()]))
                # now get equally weighted samples
                equal_samples, equal_logl = resample_equal(ns_samples['x'], ns_samples['logl'], weights=ns_samples['weights'])
                self.mc_samples = {
                    'x': equal_samples,
                    'logl': equal_logl,
                    'weights': np.ones(equal_samples.shape[0]),
                    'method': 'NS',
                    'best': ns_samples['best']
                }
                if ns_success:
                    self.converged = self.check_convergence(ii, logz_dict, threshold=self.logz_threshold)
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
            if self.gp.train_x.shape[0] > 1600:
                self.ns_step = 25

        log.info(f" Sampling stopped: {self.termination_reason}")
        log.info(f" Final GP training set size: {self.gp.train_x.shape[0]}, max size: {self.max_gp_size}")
        log.info(f" Number of iterations: {ii}, max iterations: {self.maxiters}")


        if not self.converged:
            self.results_manager.start_timing('GP Training')
            self.gp.fit()
            self.results_manager.end_timing('GP Training')


        results_dict['gp'] = self.gp

        # Save and final nested sampling
        if self.save:
            self.gp.save(outfile=self.output_file)

        # Prepare final results 
        if self.do_final_ns and not self.converged:
            log.info(" Final Nested Sampling")
            self.results_manager.start_timing('Nested Sampling')
            ns_samples, logz_dict, ns_success = nested_sampling_Dy(
                self.gp, self.ndim, maxcall=int(1e7), dynamic=True, dlogz=0.01
            )
            self.results_manager.end_timing('Nested Sampling')
            log.info(" Final LogZ: " + ", ".join([f"{k}={v:.4f}" for k,v in logz_dict.items()]))


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

