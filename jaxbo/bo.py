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
from .utils.core_utils import scale_from_unit, scale_to_unit,  resample_equal, kl_divergence_gaussian, get_threshold_for_nsigma
from .utils.seed_utils import set_global_seed, get_jax_key,  get_numpy_rng
from .nested_sampler import nested_sampling_Dy
from .utils.logging_utils import get_logger
from .utils.results import BOBEResults
from .acquisition import *
from .mpi import *

log = get_logger("bo")
log.info(f'JAX using {jax.device_count()} devices.')

_acq_funcs = {"wipv": WIPV, "ei": EI, "logei": LogEI}

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
                 loglikelihood: Union[BaseLikelihood, CobayaLikelihood] = None,
                 gp_kwargs: Dict[str, Any] = {},
                 n_cobaya_init=4,
                 n_sobol_init=32,
                 min_evals=200,
                 max_evals=1500,
                 max_gp_size=1200,
                 # pool: MPI_Pool = None,
                 # use_gp_pool=True,
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
            Number of iterations between GP fitting.
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

        if is_main_process():
            seed = set_global_seed(seed)
        else:
            local_seed = seed + get_mpi_rank()
        set_global_seed(local_seed)

        self.np_rng = get_numpy_rng()

        if not isinstance(loglikelihood, BaseLikelihood):
            raise ValueError("loglikelihood must be an instance of BaseLikelihood")
        if optimizer.lower() not in ['optax', 'scipy']:
            raise ValueError("optimizer must be either 'optax' or 'scipy'")

        # Store all parameters as instance attributes
        self.loglikelihood = loglikelihood
        self.n_sobol_init = n_sobol_init
        self.n_cobaya_init = n_cobaya_init
        self.ndim = len(self.loglikelihood.param_list)
        self.save = save
        self.save_step = save_step
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
        self.optimizer = optimizer
        self.do_final_ns = do_final_ns
        self.logz_threshold = logz_threshold
        self.convergence_n_iters = convergence_n_iters
        self.ei_goal_log = np.log(ei_goal)
        
        # Initialize state variables
        self.converged = False
        self.prev_converged = False
        self.convergence_counter = 0
        self.min_delta_seen = np.inf
        self.termination_reason = "Max evaluation budget reached"
        self.prev_samples = None
        self.fresh_start = not resume

        # Initialize results manager and file paths on main process only
        if is_main_process():
            self.output_file = self.loglikelihood.name
            self.save_dir = save_dir
            if save:
                os.makedirs(self.save_dir, exist_ok=True)
            self.save_path = os.path.join(self.save_dir, self.output_file)
            
            self.results_manager = BOBEResults(
                output_file=self.output_file,
                save_dir=self.save_dir,
                param_names=self.loglikelihood.param_list,
                param_labels=self.loglikelihood.param_labels,
                param_bounds=self.loglikelihood.param_bounds,
                settings={k: v for k, v in locals().items() 
                        if k not in ['self', 'loglikelihood', 'gp_kwargs']},
                likelihood_name=self.loglikelihood.name,
                resume_from_existing=resume
            )

        # Handle resume vs fresh start
        if resume and resume_file is not None:
            self._try_resume(resume_file, use_clf, gp_kwargs, optimizer)
        else:
            self._fresh_start(n_sobol_init, n_cobaya_init, use_clf, clf_nsigma_threshold, 
                                clf_type, clf_use_size, clf_update_step, minus_inf, 
                                gp_kwargs, optimizer)

        # Extract best point info and finalize
        self._finalize_initialization()

    def _try_resume(self, resume_file, use_clf, gp_kwargs, optimizer):
        """Try to resume from existing run."""
        # Main process loads GP and resume data
        if is_main_process():
            try:
                log.info(f"Attempting to resume from file {resume_file}")
                gp_file = resume_file + '_gp'
                self.gp = load_gp(gp_file, use_clf)
                
                # Test GP functionality
                _ = self.gp.predict_mean_single(self.gp.train_x[0])
                log.info(f"Loaded GP with {self.gp.train_x.shape[0]} training points")
                
                # Handle resume iteration and best point info
                if self.results_manager.is_resuming():
                    self.start_iteration = self.results_manager.get_last_iteration()
                    log.info(f"Resuming from iteration {self.start_iteration}")
                    
                    if self.results_manager.best_loglike_values:
                        self.best_f = max(self.results_manager.best_loglike_values)
                        best_idx = self.results_manager.best_loglike_values.index(self.best_f)
                        self.best_pt_iteration = self.results_manager.best_loglike_iterations[best_idx]
                        log.info(f"Restored best loglikelihood: {self.best_f:.4f} at iteration {self.best_pt_iteration}")
                    
                    if self.results_manager.converged:
                        self.prev_converged = True
                        self.convergence_counter = 1
                        log.info("Previous run had converged.")
                else:
                    self.start_iteration = 0
                    
                gp_state_data = self.gp.state_dict()

            except Exception as e:
                log.error(f"Failed to load GP from file {gp_file}: {e}")
                log.info("Starting a fresh run instead.")
                self.fresh_start = True
                gp_state_data = None
        else:
            gp_state_data = None
        
        # Share GP state and initialize on workers
        gp_state_data = share(gp_state_data, root=0)
        if not is_main_process() and gp_state_data is not None:
            self.gp = GP.from_state_dict(gp_state_data)
        
        # If resume failed, handle as fresh start
        if self.fresh_start:
            self._fresh_start(self.n_sobol_init, self.n_cobaya_init, use_clf, 20, "svm", 10, 1, -1e5, gp_kwargs, optimizer)

    def _fresh_start(self, n_sobol_init, n_cobaya_init, use_clf, clf_nsigma_threshold,
                        clf_type, clf_use_size, clf_update_step, minus_inf, gp_kwargs, optimizer):
        """Handle fresh start initialization."""
        self.start_iteration = 0
        self.best_pt_iteration = 0
        
        # Generate initial points in parallel
        init_points, init_vals = self._generate_initial_points(n_sobol_init, n_cobaya_init)
        
        # Main process creates GP initialization data
        if is_main_process():
            train_x = jnp.array(scale_to_unit(init_points, self.loglikelihood.param_bounds))
            train_y = jnp.array(init_vals)
            
            # Prepare GP kwargs with classifier settings if needed
            full_gp_kwargs = gp_kwargs.copy()
            full_gp_kwargs.update({
                'train_x': train_x, 'train_y': train_y, 
                'param_names': self.loglikelihood.param_list, 'optimizer': optimizer
            })
            
            if use_clf:
                clf_threshold = max(100, get_threshold_for_nsigma(clf_nsigma_threshold, self.ndim))
                full_gp_kwargs.update({
                    'clf_type': clf_type, 'clf_use_size': clf_use_size, 'clf_update_step': clf_update_step,
                    'probability_threshold': 0.5, 'minus_inf': minus_inf,
                    'clf_threshold': clf_threshold, 'gp_threshold': 2 * clf_threshold
                })
            
            # Create GP and prepare sharing data
            self.gp = GPwithClassifier(**full_gp_kwargs) if use_clf else GP(**full_gp_kwargs)
            gp_state_data = self.gp.state_dict()
        else:
            gp_state_data = None
        
        # Share and initialize GP on all processes
        gp_state_data = share(gp_state_data, root=0)
        if not is_main_process():
            self.gp = GP.from_state_dict(gp_state_data)

        # Distributed GP fitting
        if is_main_process():
            self.results_manager.start_timing('GP Training')
            log.info(f"Hyperparameters before refit: {self.gp.hyperparams_dict()}")
        
        self.distributed_gp_fit(self.gp, n_restarts=4, maxiters=500)
        
        if is_main_process():
            log.info(f"Hyperparameters after refit: {self.gp.hyperparams_dict()}")
            self.results_manager.end_timing('GP Training')

    def _finalize_initialization(self):
        """Extract best point info and save initial state."""
        idx_best = jnp.argmax(self.gp.train_y)
        self.best_pt = scale_from_unit(self.gp.train_x[idx_best], self.loglikelihood.param_bounds).flatten()
        best_f_from_gp = float(self.gp.train_y.max()) * self.gp.y_std + self.gp.y_mean

        if not hasattr(self, 'best_f') or best_f_from_gp > getattr(self, 'best_f', -np.inf):
            self.best_f = best_f_from_gp
            if not hasattr(self, 'best_pt_iteration'):
                self.best_pt_iteration = self.start_iteration

        self.best = {name: f"{float(val):.6f}" for name, val in zip(self.loglikelihood.param_list, self.best_pt)}
        
        if is_main_process():
            log.info(f"Initial best point {self.best} with value = {self.best_f:.6f}")
            if self.save:
                self.gp.save(filename=f"{self.save_path}_gp")
                log.info(f"Saving GP to file {self.save_path}_gp")

    def _generate_initial_points(self, n_sobol_init: int, n_cobaya_init: int):
        """
        Generate and evaluate initial points in parallel using MPI utilities.
        Returns (init_points, init_vals) on main rank; workers return empty arrays.
        """

        if n_sobol_init + n_cobaya_init == 0:
            raise ValueError("At least one of n_sobol_init or n_cobaya_init must be greater than zero.")

        # Generate Sobol points in parallel
        init_points, init_vals = self._generate_sobol_points(n_sobol_init) # this generates a minimum of 2 points
        
        # Generate Cobaya points if needed
        if isinstance(self.loglikelihood, CobayaLikelihood) and n_cobaya_init > 0:
            cobaya_points, cobaya_vals = self._generate_cobaya_points(n_cobaya_init)
            if is_main_process() and cobaya_points.size > 0:
                init_points = np.vstack([init_points, cobaya_points])
                init_vals = np.vstack([init_vals, cobaya_vals])

        return init_points, init_vals
    
        #     # Find the indices of the unique points (rows)
        #     unique_points, unique_indices = np.unique(
        #         init_points, axis=0, return_index=True
        #     )
        
        #     # Check if any duplicates were found
        #     num_original = len(init_points)
        #     num_unique = len(unique_points)
        
        #     if num_unique < num_original:
        #         # 4. Filter both arrays using the same unique indices
        #         final_points = init_points[unique_indices]
        #         final_logpost = init_vals[unique_indices]
        #     else:
        #         # No duplicates found, use the original arrays
        #         final_points = init_points
        #         final_logpost = init_vals

        # return final_points, final_logpost.reshape(-1, 1)

    def _generate_sobol_points(self, n_sobol_init: int):
        """Generate Sobol initial points on main rank and evaluate them in parallel."""
        from scipy.stats import qmc
        
        n_sobol = max(2, n_sobol_init)
        
        # Main process generates Sobol points
        if is_main_process():
            self.results_manager.start_timing('True Objective Evaluations')
            sobol = qmc.Sobol(d=self.ndim, scramble=True, rng=self.np_rng).random(n_sobol)
            sobol_points = scale_from_unit(sobol, self.loglikelihood.param_bounds)
            # Convert to list for scatter
            points_list = [sobol_points[i] for i in range(len(sobol_points))]
        else:
            points_list = None

        # Scatter points across processes
        local_points = scatter(points_list, root=0)
        
        # Each process evaluates its assigned points
        if local_points is not None and len(local_points) > 0:
            local_points = np.atleast_2d(local_points)
            local_vals = np.array([self.loglikelihood._safe_eval(pt) for pt in local_points]).reshape(-1, 1)
        else:
            local_points = np.empty((0, self.ndim))
            local_vals = np.empty((0, 1))

        # Gather results back to main process
        gathered_points = gather(local_points, root=0)
        gathered_vals = gather(local_vals, root=0)

        if is_main_process():
            # Reconstruct full arrays from gathered results
            all_points = np.vstack([pts for pts in gathered_points if len(pts) > 0])
            all_vals = np.vstack([vals for vals in gathered_vals if len(vals) > 0])
            return all_points, all_vals
        else:
            return np.empty((0, self.ndim)), np.empty((0, 1))

    def _generate_cobaya_points(self, n_cobaya_init: int):
        """Generate Cobaya initial points in parallel across ranks."""

        if n_cobaya_init <= 0:
            raise ValueError("n_cobaya_init must be greater than zero to generate Cobaya points.")

        # Distribute requests across processes
        if is_main_process():
            inits_per_process = np.array_split(np.arange(n_cobaya_init), get_mpi_size())
        else:
            inits_per_process = None

        local_inits = scatter(inits_per_process, root=0)

        # Each process generates its assigned points
        local_points = []
        local_vals = [] 
        for _ in range(len(local_inits)):
            pt, lp = self.loglikelihood._get_single_valid_point(rng=self.np_rng)
            local_points.append(pt)
            local_vals.append(lp)

        # Gather all results
        gathered_points = gather(local_points, root=0)
        gathered_vals = gather(local_vals, root=0)

        if is_main_process():
            # Reconstruct full arrays from gathered results
            all_points = np.vstack([pts for pts in gathered_points if len(pts) > 0])
            all_vals = np.vstack([vals for vals in gathered_vals if len(vals) > 0])
            return all_points, all_vals
        else:
            return np.empty((0, self.ndim)), np.empty((0, 1))

    def evaluate_likelihood(self, new_pts_u, step, verbose=True):
        """Evaluate likelihood for new points in parallel."""
        new_pts_u = jnp.atleast_2d(new_pts_u)
        new_pts = scale_from_unit(new_pts_u, self.loglikelihood.param_bounds)
        
        # Distribute points and evaluate
        if is_main_process():
            pts_list = [np.asarray(pt) for pt in new_pts]
            self.results_manager.start_timing('True Objective Evaluations')
        else:
            pts_list = None

        local_pts = scatter(pts_list, root=0)
        local_vals = np.array([self.loglikelihood(pt) for pt in (local_pts or [])]).reshape(-1, 1)
        gathered_vals = gather(local_vals, root=0)

        # Main process: process results and update tracking
        if is_main_process():
            all_vals = np.vstack([vals for vals in gathered_vals if len(vals) > 0])
            new_vals = jnp.array(all_vals)
            self.results_manager.end_timing('True Objective Evaluations')

            # Update best point if improved
            best_idx = int(np.argmax(all_vals))
            best_val = float(all_vals[best_idx])
            if best_val > self.best_f:
                self.best_f = best_val
                self.best_pt = new_pts[best_idx]
                self.best = {name: f"{float(val):.6f}" for name, val in zip(self.loglikelihood.param_list, self.best_pt.flatten())}
                self.best_pt_iteration = step

            # Log results
            if verbose:
                for k, (pt, val) in enumerate(zip(new_pts, all_vals)):
                    pt_dict = {name: f"{float(v):.4f}" for name, v in zip(self.loglikelihood.param_list, pt.flatten())}
                    predicted_val = self.gp.predict_mean_single(new_pts_u[k])
                    log.info(f" Point {k+1}/{len(new_pts)}: {pt_dict}")
                    log.info(f" Objective = {float(val):.4f}, GP predicted = {predicted_val.item():.4f}")
            
            return new_vals
        return None

    def update_gp(self, new_pts_u, new_vals, step=0, verbose=True):
        """Update GP with new points and perform distributed fitting."""
        # Determine fitting schedule
        training_size = self.gp.train_x.shape[0]
        if training_size < 200:
            fit_freq, n_restarts, maxiter = min(2, self.fit_step), 8, 1000
        elif training_size < 800:
            fit_freq, n_restarts, maxiter = self.fit_step, 4, 500
        else:
            fit_freq, n_restarts, maxiter = max(10, self.fit_step), 4, 200

        refit = (step % fit_freq == 0)
        
        if is_main_process():
            self.results_manager.start_timing('GP Training')
            if refit and verbose:
                log.info(f" Hyperparameters before refit: {self.gp.hyperparams_dict()}")

        # Update GP and fit if needed
        if is_main_process():
            self.gp.update(new_pts_u, new_vals)
            gp_state_data = self.gp.state_dict()
        else:
            gp_state_data = None
        
        # Share updated GP state
        gp_state_data = share(gp_state_data, root=0)
        if not is_main_process():
            self.gp = GP.from_state_dict(gp_state_data)

        if refit:
            self.distributed_gp_fit(self.gp, n_restarts=n_restarts, maxiters=maxiter)
            if is_main_process() and verbose:
                log.info(f" Hyperparameters after refit: {self.gp.hyperparams_dict()}")

        if is_main_process():
            self.results_manager.end_timing('GP Training')
            # Track hyperparameters
            self.results_manager.update_gp_hyperparams(step, list(self.gp.lengthscales), float(self.gp.kernel_variance))

        # Train classifier if needed
        if isinstance(self.gp, GPwithClassifier):
            if is_main_process():
                self.results_manager.start_timing('Classifier Training')
            self.gp.train_classifier()
            if is_main_process():
                self.results_manager.end_timing('Classifier Training')

    def distributed_gp_fit(self, gp: GP, n_restarts: int = 4, maxiters: int = 500):
        """Distribute GP fitting across MPI processes."""
        # Generate starting points
        init_params = jnp.log(gp.get_hyperparams())
        n_params = init_params.shape[0]
        
        if is_main_process():
            if n_restarts > 1:
                x0_random = self.np_rng.uniform(gp.hyperparam_bounds[0], gp.hyperparam_bounds[1], 
                                            size=(n_restarts - 1, n_params))
                x0 = np.vstack([np.array(init_params), x0_random])
            else:
                x0 = np.atleast_2d(np.array(init_params))
            x0_list = [x0[i] for i in range(len(x0))]
        else:
            x0_list = None

        # Distribute work and fit
        local_x0 = scatter(x0_list, root=0)
        local_result = {'mll': -np.inf, 'params': None}
        
        if local_x0 and len(local_x0) > 0:
            result = gp.fit(x0=np.array(local_x0), maxiter=maxiters)
            local_result = result if result else local_result

        # Gather and select best result
        all_results = gather(local_result, root=0)
        
        if is_main_process():
            valid_results = [r for r in all_results if r.get('mll', -np.inf) > -np.inf]
            best_result = max(valid_results, key=lambda r: r.get('mll', -np.inf)) if valid_results else {'params': None}
            best_params = best_result.get('params')
        else:
            best_params = None
            best_result = None

        # Share and update
        best_params = share(best_params, root=0)
        if best_params is not None:
            gp.update_hyperparams(best_params)

        return best_result if is_main_process() else None

    def get_next_batch(self, acq_kwargs, n_batch, n_restarts, maxiter, early_stop_patience, step, verbose=True):
        """
        Get the next batch of points using the acquisition function, and track acquisition values.
        """
        if is_main_process():
            self.results_manager.start_timing('Acquisition Optimization')
        new_pts_u, acq_vals = self.acquisition.get_next_batch(
            gp=self.gp,
            n_batch=n_batch,
            acq_kwargs=acq_kwargs,
            n_restarts=n_restarts,
            maxiter=maxiter,
            early_stop_patience=early_stop_patience,
        )
        if is_main_process():
            self.results_manager.end_timing('Acquisition Optimization')

        acq_val = float(np.mean(acq_vals))
        if verbose:
            log.info(f"Mean acquisition value {acq_val:.4e} at new points")
        if is_main_process():
            self.results_manager.update_acquisition(step, acq_val, self.acquisition.name)

        return new_pts_u, acq_vals

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
            else:
                self.run_EI(ii=self.current_iteration)

        if is_main_process():
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
                print("\n")
                log.info(f" Iteration {ii} of {self.acquisition.name}, objective evals {current_evals}/{self.max_evals}")

            acq_kwargs = {'zeta': self.zeta_ei, 'best_y': max(self.gp.train_y.flatten())}
            n_batch = 1
            new_pts_u, acq_vals = self.get_next_batch(acq_kwargs, n_batch = n_batch, n_restarts = 50, maxiter = 1000, early_stop_patience = 50, step = ii, verbose=verbose)
            new_pts_u = jnp.atleast_2d(new_pts_u)

            new_vals = self.evaluate_likelihood(new_pts_u, ii, verbose=verbose)
            current_evals += n_batch

            self.update_gp(new_pts_u, new_vals, step = ii, verbose=verbose)

            if is_main_process():
                self.results_manager.update_best_loglike(ii, self.best_f)
            if verbose:
                log.info(f" Current best point {self.best} with value = {self.best_f:.6f}, found at iteration {self.best_pt_iteration}")

            # if current_evals >= self.min_evals:
            converged = self.check_convergence_ei(ii,acq_vals)

            # Update results manager with iteration info, also save results and gp if save_step
            if ii % self.save_step == 0:
                if is_main_process():
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
            ns_flag = (ii % self.ns_step == 0) and current_evals >= self.min_evals
            verbose = True

            if verbose:
                print("\n")
                log.info(f" Iteration {ii} of WIPV, objective evals {current_evals}/{self.max_evals}, ns={ns_flag}")

            acq_kwargs = {'mc_samples': self.mc_samples, 'mc_points_size': self.mc_points_size}
            new_pts_u, acq_vals = self.get_next_batch(acq_kwargs, n_batch = self.wipv_batch_size, n_restarts = 1, maxiter = 100, early_stop_patience = 10, step = ii, verbose=verbose)
            new_pts_u = jnp.atleast_2d(new_pts_u)
            new_vals = self.evaluate_likelihood(new_pts_u, ii, verbose=verbose)
            current_evals += self.wipv_batch_size


            self.update_gp(new_pts_u, new_vals, step = ii)
            if is_main_process():
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
                if is_main_process():
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
            if is_main_process():
                self.results_manager.start_timing('GP Training')
            self.distributed_gp_fit(self.gp, n_restarts=4, maxiters=500)
            if is_main_process():
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
            if is_main_process():
                self.results_manager.save_intermediate(gp=self.gp, filename=f"{checkpoint_filename}")

            # Save getdist chains
            if is_main_process():
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
        samples_dict = self.samples_dict or {}
        print(samples_dict.keys())
        logz_dict = self.results_dict.get('logz', {})

        # Finalize results with comprehensive data
        if is_main_process():
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