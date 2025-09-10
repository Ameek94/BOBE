import os
import numpy as np
from scipy.stats import qmc
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from numpyro.util import enable_x64
enable_x64()
from typing import Optional, Union, Tuple, Dict, Any
# from .acquisition import WIPV, EI #, logEI
from .gp import GP
from .clf_gp import GPwithClassifier
from .likelihood import BaseLikelihood, CobayaLikelihood
from .utils.core_utils import scale_from_unit, scale_to_unit, renormalise_log_weights, resample_equal, kl_divergence_gaussian, kl_divergence_samples, get_threshold_for_nsigma
from .utils.seed_utils import set_global_seed, get_jax_key, split_jax_key, ensure_reproducibility, get_numpy_rng
from .nested_sampler import nested_sampling_Dy
from .utils.logging_utils import get_logger
from .utils.results import BOBEResults
from .acquisition import *
from .acquisition import get_mc_samples
from .utils.pool import MPI_Pool

log = get_logger("bo")
log.info(f'JAX using {jax.device_count()} devices.')

_acq_funcs = {"wipv": WIPV, "ei": EI, "logei": LogEI}

import numpy as np
from scipy import stats
import warnings


def load_gp(filename: str, clf: bool) -> Union[GP, GPwithClassifier]:
    """
    Load a GP or GPwithClassifier object from a file.

    Parameters
    ----------
    filename : str
        The path to the file from which to load the GP object.

    Returns
    -------
    Union[GP, GPwithClassifier]
        The loaded GP or GPwithClassifier object.
    """
    if clf:
        gp = GPwithClassifier.load(filename)
    else:
        gp = GP.load(filename)
    return gp

class BOBE:

    def __init__(self,
                loglikelihood=None,
                 gp_kwargs: Dict[str, Any] = {},
                 n_cobaya_init=4,
                 n_sobol_init=32,
                 min_evals=200,
                 max_evals=1500,
                 max_gp_size=1200,
                 resume=False,
                 resume_file=None,
                 save_dir='.',
                 save=True,
                 save_step=5,
                 fit_step=10,
                 wipv_batch_size=4,
                 ns_step=10,
                 num_hmc_warmup=512,
                 num_hmc_samples=512,
                 mc_points_size=64,
                 thinning=4,
                 num_chains=6,
                 mc_points_method='NUTS',
                 acq = 'WIPV',
                 zeta_ei = 0.01,
                 ei_goal = 1e-10,
                 use_clf=True,
                 clf_type = "svm",
                 clf_nsigma_threshold=25.0,
                 clf_use_size = 10,
                 clf_update_step=1,
                 logz_threshold=0.01,
                 convergence_n_iters=1,
                 minus_inf=-1e5,
                 pool: MPI_Pool = None,
                 do_final_ns=False,
                 seed: Optional[int] = None,
                 ):
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
        min_evals : int
            Minimum number of true objective evaluations before checking convergence.
        max_evals : int
            Maximum number of true objective function evaluations.
        max_gp_size : int
            Maximum number of points used to train the GP. 
            If using SVM, this is not the same as the number of points used to train the SVM.
        resume : bool
            If True, resume from a previous run. The resume_file argument must be provided.
        resume_file : str
            The file to resume from. Must be a GP file (without _gp extension).
        save : bool
            If True, save the GP training data to a file so that it can be resumed from later.
        fit_step : int
            Number of iterations between GP refits.
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
        convergence_n_iters : int
            Number of successive iterations the logz threshold must be met for convergence to be declared. Defaults to 2.
        minus_inf : float
            Value to use for minus infinity. This is used to set the lower bound of the loglikelihood.
        optimizer : str
            Optimizer to use for both GP and acquisition function optimization. Options are 'optax' or 'scipy'.
        gp_kwargs : Dict[str, Any], optional
            Additional keyword arguments to pass to GP constructors. These can include:
            - noise: Noise parameter for GP (float, default: 1e-8)
            - kernel: Kernel type ('rbf', 'matern', etc., default: 'rbf')
            - optimizer_kwargs: Dict for optimizer settings (e.g., {'lr': 1e-3, 'name': 'adam'})
            - kernel_variance_bounds: List of [lower, upper] bounds for kernel variance
            - lengthscale_bounds: List of [lower, upper] bounds for lengthscales  
            - lengthscales: Initial lengthscale values (array-like)
            - kernel_variance: Initial kernel variance value (float)
        """

        self.pool = pool


        set_global_seed(seed)
        self.np_rng = get_numpy_rng()

        if not isinstance(loglikelihood, BaseLikelihood):
            raise ValueError("loglikelihood must be an instance of ExternalLikelihood")

        self.loglikelihood = loglikelihood
        self.ndim = len(self.loglikelihood.param_list)

        # Store basic settings needed for results manager early
        self.output_file = self.loglikelihood.name
        self.save = save
        self.save_step = save_step
        self.save_dir = save_dir
        if self.save:
            os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, self.output_file)
        self.do_final_ns = do_final_ns
        self.logz_threshold = logz_threshold
        self.convergence_n_iters = convergence_n_iters
        self.converged = False
        self.prev_converged = False
        self.convergence_counter = 0  # Track successive convergence iterations
        self.min_delta_seen = np.inf  # Track minimum delta for checkpoint saving
        self.termination_reason = "Max evaluation budget reached"
        self.ei_goal_log = np.log(ei_goal)

        self.optimizer = 'scipy' # default
        
        # Initialize results manager BEFORE any timing operations
        self.results_manager = BOBEResults(
            output_file=self.output_file,
            param_names=self.loglikelihood.param_list,
            param_labels=self.loglikelihood.param_labels,
            param_bounds=self.loglikelihood.param_bounds,
            settings={
                'n_cobaya_init': n_cobaya_init,
                'n_sobol_init': n_sobol_init,
                'min_evals': min_evals,
                'max_evals': max_evals,
                'max_gp_size': max_gp_size,
                'fit_step': fit_step,
                'wipv_batch_size': wipv_batch_size,
                'ns_step': ns_step,
                'num_hmc_warmup': num_hmc_warmup,
                'num_hmc_samples': num_hmc_samples,
                'mc_points_size': mc_points_size,
                'mc_points_method': mc_points_method,
                'acq': acq,
                'use_clf': use_clf,
                'clf_type': clf_type,
                'clf_nsigma_threshold': clf_nsigma_threshold,
                'logz_threshold': logz_threshold,
                'convergence_n_iters': convergence_n_iters,
                'minus_inf': minus_inf,
                'do_final_ns': do_final_ns,
                'seed': seed
            },
            likelihood_name=self.loglikelihood.name,
            resume_from_existing=resume
        )

        self.fresh_start = not resume  # Flag to indicate if we are starting fresh or resuming
        
        if resume and resume_file is not None:
            # Resume from explicit file
            try:
                log.info(f" Attempting to resume from file {resume_file}")
                gp_file = resume_file+'_gp'
                self.gp = load_gp(gp_file, use_clf)
                                            
                # Test a simple prediction to ensure everything works
                test_point = self.gp.train_x[0]  # Use first training point as test
                _ = self.gp.predict_mean_single(test_point)
                log.info(f"Loaded GP with {self.gp.train_x.shape[0]} training points")
                                    
                # Check if resuming and adjust starting iteration
                if self.results_manager.is_resuming() and not self.fresh_start:
                    self.start_iteration = self.results_manager.get_last_iteration()
                    log.info(f"Resuming from iteration {self.start_iteration}")
                    log.info(f"Previous data: {len(self.results_manager.acquisition_values)} acquisition evaluations")
                    # If we have previous best loglikelihood data, restore the best point info
                    if self.results_manager.best_loglike_values:
                        self.best_f = max(self.results_manager.best_loglike_values)
                        best_loglike_idx = self.results_manager.best_loglike_values.index(self.best_f)
                        self.best_pt_iteration = self.results_manager.best_loglike_iterations[best_loglike_idx]
                        log.info(f"Restored best loglikelihood: {self.best_f:.4f} at iteration {self.best_pt_iteration}")
                else:
                    self.start_iteration = 0
                    log.info("Starting fresh optimization")
            except Exception as e:
                log.error(f" Failed to load GP from file {gp_file}: {e}")
                log.info(" Starting a fresh run instead.")
                self.fresh_start = True

        if self.fresh_start:
            self.start_iteration = 0
            self.best_pt_iteration = 0
            # Fresh start - evaluate initial points
            self.results_manager.start_timing('True Objective Evaluations')
            if isinstance(self.loglikelihood, CobayaLikelihood):
                init_points, init_vals = self.loglikelihood.get_initial_points(n_cobaya_init=n_cobaya_init,n_sobol_init=n_sobol_init,rng=self.np_rng)
            else:
                init_points, init_vals = self.loglikelihood.get_initial_points(n_sobol_init=n_sobol_init,rng=self.np_rng)
            self.results_manager.end_timing('True Objective Evaluations')
            train_x = jnp.array(scale_to_unit(init_points, self.loglikelihood.param_bounds))
            train_y = jnp.array(init_vals)
        

            gp_kwargs.update({'train_x': train_x, 'train_y': train_y})
            if use_clf:
                # Add clf specific parameters to gp_init_kwargs
                clf_threshold = max(200,get_threshold_for_nsigma(clf_nsigma_threshold,self.ndim))
                gp_kwargs.update({
                    'clf_type': clf_type,
                    'clf_use_size': clf_use_size,
                    'clf_update_step': clf_update_step,
                    'probability_threshold': 0.5,
                    'minus_inf': minus_inf,
                    'clf_threshold': clf_threshold,
                    'gp_threshold': 2 * clf_threshold
                })
                self.gp = GPwithClassifier(**gp_kwargs)
            else:
                self.gp = GP(**gp_kwargs)
            self.results_manager.start_timing('GP Training')
            self.gp.fit(maxiter=500,n_restarts=4)
            self.results_manager.end_timing('GP Training')

        idx_best = jnp.argmax(self.gp.train_y)
        self.best_pt = scale_from_unit(self.gp.train_x[idx_best], self.loglikelihood.param_bounds).flatten()
        best_f_from_gp = float(self.gp.train_y.max()) * self.gp.y_std + self.gp.y_mean

        # Use restored best_f if available and better, otherwise use GP's best
        if not hasattr(self, 'best_f') or best_f_from_gp > getattr(self, 'best_f', -np.inf):
            self.best_f = best_f_from_gp
            # Also update best_pt_iteration if we're using the GP's best point
            if not hasattr(self, 'best_pt_iteration'):
                self.best_pt_iteration = self.start_iteration

        self.best = {name: f"{float(val):.6f}" for name, val in zip(self.loglikelihood.param_list, self.best_pt)}
        log.info(f" Initial best point {self.best} with value = {self.best_f:.6f}")

        # Store remaining settings
        self.min_evals = min_evals
        self.max_evals = max_evals
        self.max_gp_size = max_gp_size
        self.fit_step = fit_step
        self.ns_step = ns_step
        self.wipv_batch_size = wipv_batch_size
        self.num_hmc_warmup = num_hmc_warmup
        self.num_hmc_samples = num_hmc_samples
        self.hmc_thinning = thinning
        self.hmc_num_chains = num_chains
        self.mc_points_size = mc_points_size
        self.minus_inf = minus_inf
        self.mc_points_method = mc_points_method
        self.zeta_ei = zeta_ei


        self.gp.save(filename=f"{self.save_path}_gp")
        log.info(f" Saving GP to file {self.save_path}_gp")

        # Initialize KL divergence tracking
        self.prev_samples = None
    
    def update_gp(self, new_pts_u, new_vals, refit=True, verbose=True):
        """
        Update the GP with new points and values, and track hyperparameters.
        """
        self.results_manager.start_timing('GP Training')
        if self.gp.train_x.shape[0] < 250:
            # Override refit for small training sets
            refit = True
            maxiter = 1000
            n_restarts = 10
        else:
            n_restarts = 4
            maxiter = 500
        self.gp.update(new_pts_u, new_vals, refit=refit, n_restarts=n_restarts, maxiter=maxiter) # add verbose
        self.results_manager.end_timing('GP Training')

        # Extract GP hyperparameters for tracking
        lengthscales = list(self.gp.lengthscales)
        kernel_variance = float(self.gp.kernel_variance)
        self.results_manager.update_gp_hyperparams(self.start_iteration, lengthscales, kernel_variance)

    def get_next_batch(self, acq_kwargs, n_batch, n_restarts, maxiter, early_stop_patience, step, verbose=True):
        """
        Get the next batch of points using the acquisition function, and track acquisition values.
        """
        self.results_manager.start_timing('Acquisition Optimization')
        new_pts_u, acq_vals = self.acquisition.get_next_batch(
            gp=self.gp,
            n_batch=n_batch,
            acq_kwargs=acq_kwargs,
            n_restarts=n_restarts,
            maxiter=maxiter,
            early_stop_patience=early_stop_patience,
        )
        self.results_manager.end_timing('Acquisition Optimization')

        acq_val = float(np.mean(acq_vals))
        if verbose:
            log.info(f"Mean acquisition value {acq_val:.4e} at new points")
        self.results_manager.update_acquisition(step, acq_val, self.acquisition.name)

        return new_pts_u, acq_vals

    def evaluate_likelihood(self, new_pts_u, step, verbose=True):
        """
        Evaluate the likelihood for new points.
        """


        new_pts_u = jnp.atleast_2d(new_pts_u)
        new_pts = scale_from_unit(new_pts_u, self.loglikelihood.param_bounds)

        self.results_manager.start_timing('True Objective Evaluations')
        new_vals = self.pool.run_map(self.loglikelihood, new_pts)
        new_vals = jnp.reshape(new_vals, (len(new_pts), 1))
        self.results_manager.end_timing('True Objective Evaluations')

        best_new_idx = np.argmax(new_vals)
        best_new_val = float(np.max(new_vals))
        best_new_pt = new_pts[best_new_idx]
        if float(best_new_val) > self.best_f:
            self.best_f = float(best_new_val)
            self.best_pt = best_new_pt
            self.best = {name: f"{float(val):.6f}" for name, val in zip(self.loglikelihood.param_list, self.best_pt.flatten())}
            self.best_pt_iteration = step

        for k, new_pt in enumerate(new_pts):
            new_pt_vals = {name: f"{float(val):.4f}" for name, val in zip(self.loglikelihood.param_list, new_pt.flatten())}
            log.info(f" New point {new_pt_vals}, {k+1}/{len(new_pts)}")
            predicted_val = self.gp.predict_mean_single(new_pts_u[k])
            log.info(f" Objective function value = {new_vals[k].item():.4f}, GP predicted value = {predicted_val.item():.4f}")

        return new_vals
    
    def run(self,acqs: Union[str, Tuple[str]]):
        acqs_funcs_available = list(_acq_funcs.keys())

        self.samples_dict = {}
        self.results_dict = {}

        if isinstance(acqs, str):
            acqs = (acqs,)


        self.current_iteration = self.start_iteration

        for acq in acqs:
            if acq.lower() not in acqs_funcs_available:
                raise ValueError(f"Invalid acquisition function '{acq}'. Valid options are: {acqs_funcs_available}")
            
            if acq.lower() == 'wipv':
                self.run_WIPV(ii=self.current_iteration)
            else:
                self.run_EI(acq, ii=self.current_iteration)

        log.info(f" Final best point {self.best} with value = {self.best_f:.6f}, found at iteration {self.best_pt_iteration}")


        #-------End of BO loop-------
        log.info(f" Sampling stopped: {self.termination_reason}")
        log.info(f" Final GP training set size: {self.gp.train_x.shape[0]}, max size: {self.max_gp_size}")

        self.finalise_results()

        return self.results_dict



    def run_EI(self, acq: str, ii = 0):
        """
        Run the optimization loop for EI/LogEI acquisition functions.
        """
        self.acquisition = _acq_funcs[acq.lower()](optimizer=self.optimizer)  # Set acquisition function
        current_evals = self.gp.npoints
        self.convergence_counter = 0  # Track successive convergence iterations
        log.info(f"Starting iteration {ii}")
        while current_evals < self.max_evals:
            ii += 1
            refit = (ii % self.fit_step == 0)
            verbose = True

            if verbose:
                print("\n")
                log.info(f" Iteration {ii} of {self.acquisition.name}, objective evals {current_evals}/{self.max_evals}, refit={refit}")

            acq_kwargs = {'zeta': self.zeta_ei, 'best_y': max(self.gp.train_y.flatten())}
            n_batch = 1
            new_pts_u, acq_vals = self.get_next_batch(acq_kwargs, n_batch = n_batch, n_restarts = 50, maxiter = 2500, early_stop_patience = 50, step = ii, verbose=verbose)
            new_pts_u = jnp.atleast_2d(new_pts_u)

            new_vals = self.evaluate_likelihood(new_pts_u, ii, verbose=verbose)
            current_evals += n_batch

            self.update_gp(new_pts_u, new_vals, refit=refit, verbose=verbose)


            self.results_manager.update_best_loglike(ii, self.best_f)
            if verbose:
                log.info(f" Current best point {self.best} with value = {self.best_f:.6f}, found at iteration {self.best_pt_iteration}")

            # Update results manager with iteration info, also save results and gp if save_step
            self.results_manager.update_iteration(iteration=ii, save_step=self.save_step,gp=self.gp,filepath=self.save_path)

            if current_evals >= self.min_evals:
                self.converged = self.check_convergence_ei(ii,acq_vals)
            if self.converged:
                self.termination_reason = f"{acq.upper()} goal reached"
                self.results_dict['termination_reason'] = self.termination_reason
                break

        # End EI
        self.current_iteration = ii

    def check_convergence_ei(self, step, acq_val):
        """
        Check convergence for EI/LogEI based on the acquisition function value.

        Args:
            step: Current iteration number.
            acq_val: Current acquisition function value.

        Returns:
            bool: Whether convergence is achieved based on acquisition value.
        """
        if self.acquisition.name.lower() == 'ei':
            acq_val = np.log(acq_val + 1e-100)  # Avoid log(0)
        
        converged = acq_val < self.ei_goal_log

        if converged:
            self.convergence_counter += 1
            if self.convergence_counter >= self.convergence_n_iters:
                log.info(f"Convergence achieved after {self.convergence_n_iters} successive iterations")
                return True
            else:
                log.info(f"Convergence iteration {self.convergence_counter}/{self.convergence_n_iters}")
                return False
        else:
            self.convergence_counter = 0  # Reset counter if not converged
            return False

    def run_WIPV(self):
        """
        Run the optimization loop for WIPV acquisition function.
        """
        self.acquisition = WIPV(optimizer=self.optimizer)  # Set acquisition function to WIPV
        ii = self.start_iteration
        current_evals = self.gp.npoints
        self.results_manager.start_timing('MCMC Sampling')
        self.mc_samples = get_mc_samples(
            self.gp,
            warmup_steps=self.num_hmc_warmup,
            num_samples=self.num_hmc_samples,
            thinning=self.hmc_thinning,
            num_chains=self.hmc_num_chains,
            np_rng=self.np_rng,
            rng_key=get_jax_key(),
            method=self.mc_points_method,
        )
        self.results_manager.end_timing('MCMC Sampling')
        self.convergence_counter = 0  # Track successive convergence iterations (should get from results manager if resuming)

        while current_evals < self.max_evals:
            ii += 1
            refit = (ii % self.fit_step == 0)
            ns_flag = (ii % self.ns_step == 0) and current_evals >= self.min_evals
            verbose = True

            if verbose:
                print("\n")
                log.info(f" Iteration {ii} of WIPV, objective evals {current_evals}/{self.max_evals}, refit={refit}, ns={ns_flag}")

            acq_kwargs = {'mc_samples': self.mc_samples, 'mc_points_size': self.mc_points_size}
            new_pts_u, acq_vals = self.get_next_batch(acq_kwargs, n_batch = self.wipv_batch_size, n_restarts = 1, maxiter = 100, early_stop_patience = 10, step = ii, verbose=verbose)
            new_pts_u = jnp.atleast_2d(new_pts_u)
            new_pts = scale_from_unit(new_pts_u, self.loglikelihood.param_bounds)

            acq_val = float(np.mean(acq_vals))
            log.info(f"Mean acquisition value {acq_val:.4e} at new point")
            self.results_manager.update_acquisition(ii, acq_val, self.acquisition.name)

            new_vals = self.evaluate_likelihood(new_pts,ii)
            current_evals += self.wipv_batch_size

            self.update_gp(new_pts_u, new_vals, refit=refit)

            # Check convergence and update MCMC samples
            if ns_flag:
                log.info("Running Nested Sampling")
                self.results_manager.start_timing('Nested Sampling')
                ns_samples, logz_dict, ns_success = nested_sampling_Dy(
                    self.gp, self.ndim, maxcall=int(5e6), dynamic=False, dlogz=0.01, equal_weights=False,
                    rng=self.np_rng
                )
                self.results_manager.end_timing('Nested Sampling')

                log.info(f"NS success = {ns_success}, LogZ info: " + ", ".join([f"{k}={v:.4f}" for k, v in logz_dict.items()]))

                if ns_success:
                    equal_samples, equal_logl = resample_equal(ns_samples['x'], ns_samples['logl'], weights=ns_samples['weights'])
                    self.mc_samples = {
                        'x': equal_samples,
                        'logl': equal_logl,
                        'weights': np.ones(equal_samples.shape[0]),
                        'method': 'NS',
                        'best': ns_samples['best']
                    }
                    self.converged = self.check_convergence(ii, logz_dict, equal_samples, equal_logl)
                    if self.converged:
                        self.termination_reason = "LogZ converged"
                        self.results_dict['logz'] = logz_dict
                        self.results_dict['termination_reason'] = self.termination_reason
                        break
            else:
                self.results_manager.start_timing('MCMC Sampling')
                self.mc_samples = get_mc_samples(
                        self.gp,
                        warmup_steps=self.num_hmc_warmup,
                        num_samples=self.num_hmc_samples,
                        thinning=self.hmc_thinning,
                        num_chains=self.hmc_num_chains,
                        method=self.mc_points_method,
                        np_rng=self.np_rng,
                        rng_key=get_jax_key()
                    )
                self.results_manager.end_timing('MCMC Sampling')

            # Update results manager with iteration info, also save results and gp if save_step
            self.results_manager.update_iteration(iteration=ii, save_step=self.save_step,gp=self.gp, filepath=self.save_path)

            if self.converged:
                break

        # End of main BO loop for WIPV
        self.current_iteration = ii

        # Final nested sampling if not yet converged and do_final_ns is True
        if self.do_final_ns and not self.converged:
            
            self.results_manager.start_timing('GP Training')
            self.gp.fit()
            self.results_manager.end_timing('GP Training')

            log.info(" Final Nested Sampling")
            self.results_manager.start_timing('Nested Sampling')
            ns_samples, logz_dict, ns_success = nested_sampling_Dy(
                self.gp, self.ndim, maxcall=int(1e7), dynamic=True, dlogz=0.01,rng=self.np_rng
            )
            self.results_manager.end_timing('Nested Sampling')
            log.info(" Final LogZ: " + ", ".join([f"{k}={v:.4f}" for k,v in logz_dict.items()]))
            if ns_success:
                equal_samples, equal_logl = resample_equal(ns_samples['x'], ns_samples['logl'], weights=ns_samples['weights'])
                log.info(f"Using nested sampling results")
                self.check_convergence(ii+1, logz_dict, equal_samples, equal_logl)
                if self.converged:
                    self.termination_reason = "LogZ converged"
                    self.results_dict['logz'] = logz_dict
                    self.results_dict['termination_reason'] = self.termination_reason

        if (ns_samples is not None) and ns_success:
            samples = ns_samples['x']
            weights = ns_samples['weights']
            loglikes = ns_samples['logl']
        else:
            log.info("No nested sampling results found or nested sampling unsuccessful, MC samples from HMC/MCMC will be used instead.")
            self.results_manager.start_timing('MCMC Sampling')
            mc_samples = get_mc_samples(
                    self.gp, warmup_steps=512, num_samples=2000*self.ndim,
                    thinning=4, method="NUTS")
            self.results_manager.end_timing('MCMC Sampling')
            samples = mc_samples['x']
            weights = mc_samples['weights'] if 'weights' in mc_samples else np.ones(mc_samples['x'].shape[0])
            loglikes = mc_samples['logp']
                
        samples = scale_from_unit(samples, self.loglikelihood.param_bounds)

        self.samples_dict = {
            'x': samples,
            'weights': weights,
            'logl': loglikes
        }

    def check_convergence_WIPV(self, step, logz_dict, equal_samples, equal_logl, verbose=True):
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
        delta =  logz_dict['std']
        converged = delta < self.logz_threshold
        
        # Compute KL divergences if we have nested sampling samples
        successive_kl = None
        
        equal_samples = scale_from_unit(equal_samples, self.loglikelihood.param_bounds)
        if self.prev_samples is not None:

            prev_samples_x = self.prev_samples['x']
            mu1 = np.mean(prev_samples_x, axis=0)
            cov1 = np.cov(prev_samples_x, rowvar=False)
            mu2 = np.mean(equal_samples, axis=0)
            cov2 = np.cov(equal_samples, rowvar=False)
            successive_kl = kl_divergence_gaussian(mu1, cov1, mu2, cov2)

            # Store current samples for next iteration
            self.prev_samples = {'x': equal_samples, 'logl': equal_logl}

            if successive_kl:
                log.info(f" Successive KL: symmetric={successive_kl.get('symmetric', 0):.4f}")

        # Update results manager with convergence info and KL divergences
        self.results_manager.update_convergence(
            iteration=step,
            logz_dict=logz_dict,
            converged=converged,
            threshold=self.logz_threshold
        )
        
        # Store KL divergences if computed
        if successive_kl is not None:
            self.results_manager.update_kl_divergences(
                iteration=step,
                successive_kl=successive_kl
            )

        log.info(f"Convergence check: delta = {delta:.4f}, step = {step}, threshold = {self.logz_threshold}")
        
        # Check if this is the smallest delta seen so far and save checkpoint, also ensure delta is reasonably good
        if (delta < self.min_delta_seen) and (delta < 0.5):
            self.min_delta_seen = delta

            # Create checkpoint filename with suffix
            checkpoint_filename = f"{self.save_path}_checkpoint"

            # Save GP checkpoint
            self.gp.save(filename=f"{checkpoint_filename}_gp")

            # Save intermediate results checkpoint
            self.results_manager.save_intermediate(gp=self.gp, filename=f"{checkpoint_filename}.json")

            if verbose:
                log.info(f"New minimum delta achieved: {delta:.4f}")
                log.info("Saving checkpoint results for new minimum delta")
                log.info(f"Saved GP checkpoint to {checkpoint_filename}_gp.npz")
                log.info(f"Saved intermediate results checkpoint to {checkpoint_filename}.json")

        if converged:
            self.convergence_counter += 1
            if self.convergence_counter >= self.convergence_n_iters:
                log.info(f"Convergence achieved after {self.convergence_n_iters} successive iterations")
                return True
            else:
                log.info(f"Convergence iteration {self.convergence_counter}/{self.convergence_n_iters}")
                return False
        else:
            self.convergence_counter = 0  # Reset counter if not converged
            return False

    def finalise_results(self):
            # here finalize results
        
        # Prepare return dictionary

        # Extract GP and classifier information
        gp_info = {
            'gp_training_set_size': self.gp.train_x.shape[0],
            'gp_final_best_loglike': float(self.best_f),  # Best value in true physical space
        }
        
        # Add classifier info if using GPwithClassifier
        if isinstance(self.gp, GPwithClassifier):
            gp_info.update({
                'classifier_used': bool(self.gp.use_clf),
                'classifier_type': str(self.gp.clf_type),
                'classifier_training_set_size': int(self.gp.clf_data_size),
                'classifier_use_threshold': int(self.gp.clf_use_size),
                'classifier_probability_threshold': float(self.gp.probability_threshold)
            })
        else:
            gp_info.update({
                'classifier_used': False,
                'classifier_type': None,
                'classifier_training_set_size': 0
            })

        # Add evidence info if available
        logz_dict = self.results_dict.get('logz', {})

        # Finalize results with comprehensive data
        self.results_manager.finalize(
            samples_dict = self.samples_dict,
            logz_dict=logz_dict,
            converged=self.converged,
            termination_reason=self.termination_reason,
            gp_info=gp_info
        )

        self.results_dict['gp'] = self.gp
        self.results_dict['likelihood'] = self.loglikelihood

        # Add results manager info
        self.results_dict['results_manager'] = self.results_manager