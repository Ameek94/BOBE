import os
import numpy as np
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
from .utils.core import scale_from_unit, scale_to_unit,  resample_equal, kl_divergence_gaussian, get_threshold_for_nsigma
from .utils.seed import set_global_seed, get_jax_key,  get_numpy_rng
from .nested_sampler import nested_sampling_Dy
from .utils.log import get_logger
from .utils.results import BOBEResults
from .acquisition import *
from .utils.pool import MPI_Pool

log = get_logger("bo")
log.info(f'JAX using {jax.device_count()} devices.')

_acq_funcs = {"wipv": WIPV, "ei": EI, "logei": LogEI, 'wipstd': WIPStd}



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
                 pool: MPI_Pool = None,
                 use_gp_pool=True,
                 resume=False,
                 resume_file=None,
                 save_dir='.',
                 save=True,
                 save_step=5,
                 optimizer='scipy',
                 fit_step=10,
                 wipv_batch_size=4,
                 ns_step=10,
                 num_hmc_warmup=512,
                 num_hmc_samples=512,
                 mc_points_size=64,
                 thinning=4,
                 num_chains=4,
                 mc_points_method='NUTS',
                 acq = 'WIPV',
                 zeta_ei = 0.01,
                 ei_goal = 1e-10,
                 use_clf=True,
                 clf_type = "svm",
                 clf_nsigma_threshold=20,
                 clf_use_size = 10,
                 clf_update_step=1,
                 logz_threshold=0.01,
                 convergence_n_iters=1,
                 minus_inf=-1e5,
                 do_final_ns=False,
                 seed: Optional[int] = None,
                 ):
        """
        Initialize the BOBE (Bayesian Optimization for Bayesian Evidence) sampler.

        Parameters
        ----------
        loglikelihood : BaseLikelihood
            Likelihood function instance. Must be an instance of BaseLikelihood or its subclasses.
        gp_kwargs : dict, optional
            Additional keyword arguments to pass to GP constructors. Default is {}.
        n_cobaya_init : int, optional
            Number of initial points from Cobaya reference distribution. 
            Only used for CobayaLikelihood instances. Default is 4.
        n_sobol_init : int, optional
            Number of initial Sobol quasi-random points. Default is 32.
        min_evals : int, optional
            Minimum number of likelihood evaluations before checking convergence. Default is 200.
        max_evals : int, optional
            Maximum number of likelihood evaluations. Default is 1500.
        max_gp_size : int, optional
            Maximum number of points used to train the GP. Default is 1200.
        pool : MPI_Pool, optional
            MPI pool for parallel evaluation. Default is None.
        use_gp_pool : bool, optional
            Whether to use MPI pool for GP fitting. Default is True.
        resume : bool, optional
            If True, resume from a previous run. Default is False.
        resume_file : str, optional
            Path to resume from (directory containing GP file). Default is None.
        save_dir : str, optional
            Directory for saving results. Default is '.'.
        save : bool, optional
            Whether to save results periodically. Default is True.
        save_step : int, optional
            Save results every save_step iterations. Default is 5.
        optimizer : str, optional
            Optimizer for GP and acquisition function. Options: 'scipy', 'optax'. Default is 'scipy'.
        fit_step : int, optional
            Fit GP every fit_step iterations. Default is 10.
        wipv_batch_size : int, optional
            Batch size for WIPV acquisition. Default is 4.
        ns_step : int, optional
            Run nested sampling every ns_step iterations. Default is 10.
        num_hmc_warmup : int, optional
            Number of HMC warmup steps. Default is 512.
        num_hmc_samples : int, optional
            Number of HMC samples to draw. Default is 512.
        mc_points_size : int, optional
            Number of MC points for WIPV acquisition. Default is 64.
        thinning : int, optional
            Thinning factor for MC samples. Default is 4.
        num_chains : int, optional
            Number of parallel HMC chains. Default is 4.
        mc_points_method : str, optional
            Method for generating MC points: 'NUTS', 'NS', or 'uniform'. Default is 'NUTS'.
        acq : str, optional
            Acquisition function: 'WIPV', 'EI', 'LogEI', 'WIPStd'. Default is 'WIPV'.
        zeta_ei : float, optional
            Exploration parameter for EI acquisition. Default is 0.01.
        ei_goal : float, optional
            Goal value for EI acquisition. Default is 1e-10.
        use_clf : bool, optional
            Whether to use classifier for GP filtering. Default is True.
        clf_type : str, optional
            Classifier type: 'svm', 'nn', 'ellipsoid'. Default is 'svm'.
        clf_nsigma_threshold : float, optional
            N-sigma threshold for classifier training. Default is 20.
        clf_use_size : int, optional
            Minimum dataset size before using classifier. Default is 10.
        clf_update_step : int, optional
            Update classifier every clf_update_step iterations. Default is 1.
        logz_threshold : float, optional
            Convergence threshold for log evidence change. Default is 0.01.
        convergence_n_iters : int, optional
            Number of successive iterations meeting threshold for convergence. Default is 1.
        minus_inf : float, optional
            Value representing negative infinity for failed evaluations. Default is -1e5.
        do_final_ns : bool, optional
            Whether to run final nested sampling at convergence. Default is False.
        seed : int, optional
            Random seed for reproducibility. Default is None.
        """

        self.pool = pool
        self.use_gp_pool = use_gp_pool


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

        if optimizer.lower() not in ['optax', 'scipy']:
            raise ValueError("optimizer must be either 'optax' or 'scipy'")
        self.optimizer = optimizer
        
        # Initialize results manager BEFORE any timing operations
        self.results_manager = BOBEResults(
            output_file=self.output_file,
            save_dir=self.save_dir,
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
                    if self.results_manager.converged:
                        self.prev_converged = True
                        self.convergence_counter = 1
                        log.info(" Previous run had converged.")
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
            init_points, init_vals = self.loglikelihood.get_initial_points(n_cobaya_init=n_cobaya_init,n_sobol_init=n_sobol_init,rng=self.np_rng)
            self.results_manager.end_timing('True Objective Evaluations')
            train_x = jnp.array(scale_to_unit(init_points, self.loglikelihood.param_bounds))
            train_y = jnp.array(init_vals)
        

            gp_kwargs.update({'train_x': train_x, 'train_y': train_y, 'param_names': self.loglikelihood.param_list, 'optimizer': optimizer})
            if use_clf:
                # Add clf specific parameters to gp_init_kwargs
                clf_threshold = max(100,get_threshold_for_nsigma(clf_nsigma_threshold,self.ndim))
                gp_kwargs.update({
                    'clf_type': clf_type,
                    'clf_use_size': clf_use_size,
                    'clf_update_step': clf_update_step,
                    'probability_threshold': 0.5,
                    'minus_inf': minus_inf,
                    'clf_threshold': clf_threshold,
                    'gp_threshold': 2*clf_threshold
                })
                self.gp = GPwithClassifier(**gp_kwargs)
            else:
                self.gp = GP(**gp_kwargs)
            self.results_manager.start_timing('GP Training')
            log.info(f" Hyperparameters before refit: {self.gp.hyperparams_dict()}")
            self.pool.gp_fit(self.gp, n_restarts=4, maxiters=500,use_pool=self.use_gp_pool,rng=self.np_rng)
            log.info(f" Hyperparameters after refit: {self.gp.hyperparams_dict()}")
            self.results_manager.end_timing('GP Training')
            
        if self.gp.train_y.size > 0: 
            idx_best = jnp.argmax(self.gp.train_y) 
            self.best_pt = scale_from_unit(self.gp.train_x[idx_best], self.loglikelihood.param_bounds).flatten()
            best_f_from_gp = float(self.gp.train_y.max()) * self.gp.y_std + self.gp.y_mean
        else:
            best_f_from_gp = -np.inf
            self.best_pt = None

        # Use restored best_f if available and better, otherwise use GP's best
        if not hasattr(self, 'best_f') or best_f_from_gp > getattr(self, 'best_f', -np.inf):
            self.best_f = best_f_from_gp
            # Also update best_pt_iteration if we're using the GP's best point
            if not hasattr(self, 'best_pt_iteration'):
                self.best_pt_iteration = self.start_iteration

        if self.best_pt != None:
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
    
    def update_gp(self, new_pts_u, new_vals, step = 0, verbose=True):
        """
        Update the GP with new points and values, and track hyperparameters.
        """
        self.results_manager.start_timing('GP Training')
        refit = (step % self.fit_step == 0)
        if self.gp.train_x.shape[0] < 200:
            # Override refit for small training sets to do more frequent fitting
            override_fit_step = min(2, self.fit_step)
            refit = (step % override_fit_step == 0)
            maxiter = 1000
            n_restarts = 8
        elif 200 < self.gp.train_x.shape[0] < 800:
            # for moderate size training sets
            n_restarts = 4
            maxiter = 500
        else:
            # for large training sets we don't need to do too many restarts or frequent fitting
            n_restarts = 4
            maxiter = 200
            override_fit_step = max(10, self.fit_step)
            refit = (step % override_fit_step == 0)  
        self.gp.update(new_pts_u, new_vals, n_restarts=n_restarts, maxiter=maxiter) # add verbose
        if refit:
            self.pool.gp_fit(self.gp, n_restarts=n_restarts, maxiters=maxiter)
        self.results_manager.end_timing('GP Training')

        # Extract GP hyperparameters for tracking
        lengthscales = list(self.gp.lengthscales)
        kernel_variance = float(self.gp.kernel_variance)
        self.results_manager.update_gp_hyperparams(step, lengthscales, kernel_variance)

        if isinstance(self.gp, GPwithClassifier):
            self.results_manager.start_timing('Classifier Training')
            self.gp.train_classifier()
            self.results_manager.end_timing('Classifier Training')

        

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
        new_vals = self.pool.run_map_objective(self.loglikelihood, new_pts)
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

    def check_max_evals_and_gpsize(self,current_evals):
        """
        Check if the maximum evaluations or GP size has been reached.

        Args:
            current_evals: Current number of objective evaluations.
        """
        if current_evals >= self.max_evals:
            self.termination_reason = "Maximum evaluations reached"
            self.results_dict['termination_reason'] = self.termination_reason
            return True
        if self.gp.train_x.shape[0] >= self.max_gp_size:
            self.termination_reason = "Maximum GP size reached"
            self.results_dict['termination_reason'] = self.termination_reason
            return True
        
        return False

    def run(self, acqs: Union[str, Tuple[str]]):
        acqs_funcs_available = list(_acq_funcs.keys())

        self.samples_dict = {}
        self.results_dict = {}

        if isinstance(acqs, str):
            acqs = (acqs,)


        self.current_iteration = self.start_iteration

        for acq in acqs:
            if acq.lower() not in acqs_funcs_available:
                raise ValueError(f"Invalid acquisition function '{acq}'. Valid options are: {acqs_funcs_available}")
            self.acquisition = _acq_funcs[acq.lower()](optimizer=self.optimizer)  # Set acquisition function
            if acq.lower() == 'wipv':
                self.run_WIPV(ii=self.current_iteration)
            elif acq.lower() == 'wipstd':
                self.run_WIPStd(ii=self.current_iteration)
            else:
                self.run_EI(ii=self.current_iteration)

        log.info(f" Final best point {self.best} with value = {self.best_f:.6f}, found at iteration {self.best_pt_iteration}")


        #-------End of BO loop-------
        log.info(f" Sampling stopped: {self.termination_reason}")
        log.info(f" Final GP training set size: {self.gp.train_x.shape[0]}, max size: {self.max_gp_size}")

        self.finalise_results()

        return self.results_dict



    def run_EI(self, ii = 0):
        """
        Run the optimization loop for EI/LogEI acquisition functions.
        """
        current_evals = self.gp.npoints
        log.info(f"Starting iteration {ii}")
        converged=False

        while not converged:
            ii += 1
            verbose = True

            if verbose:
                log.debug("\n")
                log.info(f" Iteration {ii} of {self.acquisition.name}, objective evals {current_evals}/{self.max_evals}")

            acq_kwargs = {'zeta': self.zeta_ei, 'best_y': max(self.gp.train_y.flatten()) if self.gp.train_y.size > 0 else 0.}
            n_batch = 1
            new_pts_u, acq_vals = self.get_next_batch(acq_kwargs, n_batch = n_batch, n_restarts = 50, maxiter = 1000, early_stop_patience = 50, step = ii, verbose=verbose)
            new_pts_u = jnp.atleast_2d(new_pts_u)

            new_vals = self.evaluate_likelihood(new_pts_u, ii, verbose=verbose)
            current_evals += n_batch

            self.update_gp(new_pts_u, new_vals, step = ii, verbose=verbose)

            self.results_manager.update_best_loglike(ii, self.best_f)
            if verbose:
                log.info(f" Current best point {self.best} with value = {self.best_f:.6f}, found at iteration {self.best_pt_iteration}")

            # if current_evals >= self.min_evals:
            converged = self.check_convergence_ei(ii,acq_vals)

            # Update results manager with iteration info, also save results and gp if save_step
            if ii % self.save_step == 0:
                self.results_manager.save_intermediate(gp=self.gp)

            if converged:
                self.termination_reason = f"{self.acquisition.name.upper()} goal reached"
                self.results_dict['termination_reason'] = self.termination_reason
                break
            jax.clear_caches()

            max_evals_or_gpsize_reached = self.check_max_evals_and_gpsize(current_evals)
            if max_evals_or_gpsize_reached:
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
                log.info(f"Convergence for {self.acquisition.name} achieved after {self.convergence_n_iters} successive iterations")
                return True
            else:
                log.info(f"{self.acquisition.name} convergence iteration {self.convergence_counter}/{self.convergence_n_iters}")
                return False
        else:
            self.convergence_counter = 0  # Reset counter if not converged
            return False


    def run_WIPStd(self, ii = 0):
        """
        Run the optimization loop for WIPStd acquisition function.
        """
        self.acquisition = WIPStd(optimizer=self.optimizer)  # Set acquisition function to WIPStd   
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
        self.ns_samples = None

        while not self.converged:
            ii += 1
            ns_flag = (ii % self.ns_step == 0) and current_evals >= self.min_evals
            verbose = True

            if verbose:
                log.debug("\n")
                log.info(f" Iteration {ii} of WIPStd, objective evals {current_evals}/{self.max_evals}")

            acq_kwargs = {'mc_samples': self.mc_samples, 'mc_points_size': self.mc_points_size}
            new_pts_u, acq_vals = self.get_next_batch(acq_kwargs, n_batch = self.wipv_batch_size, n_restarts = 1, maxiter = 100, early_stop_patience = 10, step = ii, verbose=verbose)
            new_pts_u = jnp.atleast_2d(new_pts_u)
            new_vals = self.evaluate_likelihood(new_pts_u, ii, verbose=verbose)
            current_evals += self.wipv_batch_size


            self.update_gp(new_pts_u, new_vals, step = ii)
            self.results_manager.update_best_loglike(ii, self.best_f)

            # Check convergence and update MCMC samples
            if ns_flag:
                log.info("Running Nested Sampling")
                self.results_manager.start_timing('Nested Sampling')
                ns_samples, logz_dict, ns_success = nested_sampling_Dy(mode='convergence',
                    gp=self.gp, ndim=self.ndim, maxcall=int(5e6), dynamic=False, dlogz=0.01, equal_weights=False,
                    rng=self.np_rng
                )
                self.results_manager.end_timing('Nested Sampling')
                if ns_success:
                    self.ns_samples = ns_samples
                    log.info(f"NS success = {ns_success}, LogZ info: " + ", ".join([f"{k}={v:.4f}" for k, v in logz_dict.items()]))
                    equal_samples, equal_logl = resample_equal(self.ns_samples['x'], self.ns_samples['logl'], weights=self.ns_samples['weights'])
                    self.mc_samples = {
                        'x': equal_samples,
                        'logl': equal_logl,
                        'weights': np.ones(equal_samples.shape[0]),
                        'method': 'NS',
                        'best': self.ns_samples['best']
                    }
                    self.converged = self.check_convergence_WIPV(ii, logz_dict, equal_samples, equal_logl)
                    if self.converged:
                        self.termination_reason = "LogZ converged"
                        self.results_dict['logz'] = logz_dict
                        self.results_dict['termination_reason'] = self.termination_reason
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
            
            if verbose:
                log.info(f" Current best point {self.best} with value = {self.best_f:.6f}, found at iteration {self.best_pt_iteration}")

            # Update results manager with iteration info, also save results and gp if save_step
            if ii % self.save_step == 0:
                self.results_manager.save_intermediate(gp=self.gp)

            if self.converged:
                break
            
            jax.clear_caches()

            max_evals_or_gpsize_reached = self.check_max_evals_and_gpsize(current_evals)
            if max_evals_or_gpsize_reached:
                break


        # End of main BO loop for WIPV
        self.current_iteration = ii

        # Final nested sampling if not yet converged and do_final_ns is True
        if self.do_final_ns and not self.converged:
            
            self.results_manager.start_timing('GP Training')
            self.pool.gp_fit(self.gp, n_restarts=4, maxiters=500)
            self.results_manager.end_timing('GP Training')

            log.info(" Final Nested Sampling")
            self.results_manager.start_timing('Nested Sampling')
            self.ns_samples, logz_dict, ns_success = nested_sampling_Dy(mode='convergence',
                gp=self.gp, ndim=self.ndim, maxcall=int(5e6), dynamic=True, dlogz=0.01, rng=self.np_rng
            )
            self.results_manager.end_timing('Nested Sampling')
            if ns_success:
                log.info(" Final LogZ: " + ", ".join([f"{k}={v:.4f}" for k,v in logz_dict.items()]))
                equal_samples, equal_logl = resample_equal(self.ns_samples['x'], self.ns_samples['logl'], weights=self.ns_samples['weights'])
                log.info(f"Using nested sampling results")
                self.check_convergence_WIPV(ii+1, logz_dict, equal_samples, equal_logl)
                if self.converged:
                    self.termination_reason = "LogZ converged"
                    self.results_dict['logz'] = logz_dict
                    self.results_dict['termination_reason'] = self.termination_reason

        if (self.ns_samples is not None) and ns_success:
            samples = self.ns_samples['x']
            weights = self.ns_samples['weights']
            loglikes = self.ns_samples['logl']
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

    def run_WIPV(self, ii = 0):
        """
        Run the optimization loop for WIPV acquisition function.
        """
        self.acquisition = WIPV(optimizer=self.optimizer)  # Set acquisition function to WIPV
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
        self.ns_samples = None

        while not self.converged:
            ii += 1
            refit = (ii % self.fit_step == 0)
            ns_flag = (ii % self.ns_step == 0) and current_evals >= self.min_evals
            verbose = True

            if verbose:
                log.info(f"\nIteration {ii} of WIPV, objective evals {current_evals}/{self.max_evals}, refit={refit}, ns={ns_flag}")

            acq_kwargs = {'mc_samples': self.mc_samples, 'mc_points_size': self.mc_points_size}
            new_pts_u, acq_vals = self.get_next_batch(acq_kwargs, n_batch = self.wipv_batch_size, n_restarts = 1, maxiter = 100, early_stop_patience = 10, step = ii, verbose=verbose)
            new_pts_u = jnp.atleast_2d(new_pts_u)
            new_vals = self.evaluate_likelihood(new_pts_u, ii, verbose=verbose)
            current_evals += self.wipv_batch_size


            self.update_gp(new_pts_u, new_vals, step = ii)
            self.results_manager.update_best_loglike(ii, self.best_f)

            # Check convergence and update MCMC samples
            if ns_flag:
                log.info("Running Nested Sampling")
                self.results_manager.start_timing('Nested Sampling')
                ns_samples, logz_dict, ns_success = nested_sampling_Dy(mode='convergence',
                    gp=self.gp, ndim=self.ndim, maxcall=int(5e6), dynamic=False, dlogz=0.01, equal_weights=False,
                    rng=self.np_rng
                )
                self.results_manager.end_timing('Nested Sampling')

                log.info(f"NS success = {ns_success}, LogZ info: " + ", ".join([f"{k}={v:.4f}" for k, v in logz_dict.items()]))

                if logz_dict['std'] < 0.5:
                    self.ns_samples = ns_samples
                if ns_success:
                    equal_samples, equal_logl = resample_equal(ns_samples['x'], ns_samples['logl'], weights=ns_samples['weights'])
                    self.mc_samples = {
                        'x': equal_samples,
                        'logl': equal_logl,
                        'weights': np.ones(equal_samples.shape[0]),
                        'method': 'NS',
                        'best': ns_samples['best']
                    }
                    self.converged = self.check_convergence_WIPV(ii, logz_dict, equal_samples, equal_logl)
                    if self.converged:
                        self.termination_reason = "LogZ converged"
                        self.results_dict['logz'] = logz_dict
                        self.results_dict['termination_reason'] = self.termination_reason
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
            
            if verbose:
                log.info(f" Current best point {self.best} with value = {self.best_f:.6f}, found at iteration {self.best_pt_iteration}")

            # Update results manager with iteration info, also save results and gp if save_step
            if ii % self.save_step == 0:
                self.results_manager.save_intermediate(gp=self.gp)

            if self.converged:
                break
            
            jax.clear_caches()

            max_evals_or_gpsize_reached = self.check_max_evals_and_gpsize(current_evals)
            if max_evals_or_gpsize_reached:
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
            self.ns_samples, logz_dict, ns_success = nested_sampling_Dy(mode='convergence',
                gp=self.gp, ndim=self.ndim, maxcall=int(5e6), dynamic=True, dlogz=0.01, rng=self.np_rng
            )
            self.results_manager.end_timing('Nested Sampling')
            log.info(" Final LogZ: " + ", ".join([f"{k}={v:.4f}" for k,v in logz_dict.items()]))
            if ns_success:
                equal_samples, equal_logl = resample_equal(self.ns_samples['x'], self.ns_samples['logl'], weights=self.ns_samples['weights'])
                log.info(f"Using nested sampling results")
                self.check_convergence_WIPV(ii+1, logz_dict, equal_samples, equal_logl)
                if self.converged:
                    self.termination_reason = "LogZ converged"
                    self.results_dict['logz'] = logz_dict
                    self.results_dict['termination_reason'] = self.termination_reason

        if (self.ns_samples is not None) and ns_success:
            samples = self.ns_samples['x']
            weights = self.ns_samples['weights']
            loglikes = self.ns_samples['logl']
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
            successive_kl = kl_divergence_gaussian(mu1, np.atleast_2d(cov1), mu2, np.atleast_2d(cov2))

            log.info(f" Successive KL: symmetric={successive_kl.get('symmetric', 0):.4f}")
            # Store KL divergences if computed
            self.results_manager.update_kl_divergences(
                iteration=step,
                successive_kl=successive_kl
            )

        # Store current samples for next iteration
        self.prev_samples = {'x': equal_samples, 'logl': equal_logl}

        # Update results manager with convergence info and KL divergences
        self.results_manager.update_convergence(
            iteration=step,
            logz_dict=logz_dict,
            converged=converged,
            threshold=self.logz_threshold
        )
        

        log.info(f"Convergence check: delta = {delta:.4f}, step = {step}, threshold = {self.logz_threshold}")
        
        # Check if this is the smallest delta seen so far and save checkpoint, also ensure delta is reasonably good
        if (delta < self.min_delta_seen) and (delta < 0.5):
            self.min_delta_seen = delta

            # Create checkpoint filename with suffix
            checkpoint_filename = f"{self.output_file}_checkpoint"

            # Save intermediate results checkpoint
            self.results_manager.save_intermediate(gp=self.gp, filename=f"{checkpoint_filename}")

            # Save getdist chains
            self.results_manager.save_chain_files(samples_dict=self.ns_samples, filename=f"{checkpoint_filename}")

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
        
        # Add classifier info if using GPwithClassifier, this can be done at the start since there are no results here, only settings.
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
        samples_dict = self.samples_dict or {}
        log.debug(f"Samples dict keys: {samples_dict.keys()}")
        logz_dict = self.results_dict.get('logz', {})

        # Finalize results with comprehensive data
        self.results_manager.finalize(
            samples_dict=samples_dict,
            logz_dict=logz_dict,
            converged=self.converged,
            termination_reason=self.termination_reason,
            gp_info=gp_info
        )

        self.results_dict['gp'] = self.gp
        self.results_dict['likelihood'] = self.loglikelihood
        self.results_dict['samples'] = self.samples_dict

        # Add results manager info
        self.results_dict['results_manager'] = self.results_manager