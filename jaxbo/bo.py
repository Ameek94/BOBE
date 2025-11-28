import os
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from typing import Optional, Union, Tuple, Dict, Any, Callable
# from .acquisition import WIPV, EI #, logEI
from .gp import GP
from .clf_gp import GPwithClassifier
from .likelihood import Likelihood, CobayaLikelihood
from .utils.core import scale_from_unit, scale_to_unit,  resample_equal, kl_divergence_gaussian, get_threshold_for_nsigma
from .utils.seed import set_global_seed, get_jax_key,  get_numpy_rng, get_new_jax_key
from .samplers import nested_sampling_Dy, sample_GP_NUTS
from .utils.log import get_logger, update_verbosity
from .utils.results import BOBEResults
from .acquisition import *
from . import mpi

log = get_logger("bo")
log.info(f'JAX using {jax.device_count()} devices.')

_acq_funcs = {"wipv": WIPV, "ei": EI, "logei": LogEI, 'wipstd': WIPStd}


def load_gp_file(filename: str, clf: bool) -> Union[GP, GPwithClassifier]:
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

def load_gp_statedict(state_dict: Dict[str, Any], clf: bool) -> Union[GP, GPwithClassifier]:
    """
    Load a GP or GPwithClassifier object from a state dictionary.

    Parameters
    ----------
    state_dict : dict
        The state dictionary containing the GP parameters.
    clf : bool
        Whether to load a GPwithClassifier (True) or a standard GP (False).

    Returns
    -------
    Union[GP, GPwithClassifier]
        The loaded GP or GPwithClassifier object.
    """
    if clf:
        gp = GPwithClassifier.from_state_dict(state_dict)
    else:
        gp = GP.from_state_dict(state_dict)
    return gp

class BOBE:

    def __init__(self,
                loglikelihood=None,
                 param_list=None,
                 param_bounds=None,
                 param_labels=None,
                 likelihood_name=None,
                 confidence_for_unbounded=0.9999995,
                 gp_kwargs: Dict[str, Any] = {},
                 n_cobaya_init=4,
                 n_sobol_init=32,
                 init_train_x=None,
                 init_train_y=None,
                 min_evals=200,
                 max_evals=1500,
                 max_gp_size=1200,
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
                 minus_inf=-1e10,
                 do_final_ns=False,
                 seed: Optional[int] = None,
                 verbosity: str = 'INFO',
                 ):
        """
        Initialize the BOBE (Bayesian Optimization for Bayesian Evidence) sampler.

        Parameters
        ----------
        loglikelihood : callable, str, dict, or Likelihood
            Log-likelihood specification. Can be:
            - A callable function (requires param_list and param_bounds)
            - A string path to Cobaya YAML file (automatically creates CobayaLikelihood)
            - A dict with Cobaya info (automatically creates CobayaLikelihood)
            - A Likelihood instance (param_list, param_bounds ignored)
        param_list : list of str, optional
            Names of parameters. Required if loglikelihood is a callable.
            Ignored for Cobaya likelihoods (extracted from YAML/dict).
        param_bounds : array-like, optional
            Parameter bounds, shape (2, ndim). Required if loglikelihood is a callable.
            Ignored for Cobaya likelihoods (extracted from priors).
        param_labels : list of str, optional
            LaTeX labels for parameters. If not provided, uses param_list.
            Ignored for Cobaya likelihoods (extracted from YAML/dict).
        likelihood_name : str, optional
            Name for the likelihood (used in output files). If not provided, uses 'likelihood'
            for callables or 'cobaya_model' for Cobaya likelihoods.
        confidence_for_unbounded : float, optional
            Confidence level for unbounded Cobaya priors. Default is 0.9999995.
            Only used when loglikelihood is a Cobaya YAML file or dict.
        noise_std : float, optional
            Noise standard deviation for Cobaya likelihood. Default is 0.0.
            Only used when loglikelihood is a Cobaya YAML file or dict.
        gp_kwargs : dict, optional
            Additional keyword arguments to pass to GP constructors. Default is {}.
        n_cobaya_init : int, optional
            Number of initial points from Cobaya reference distribution. 
            Only used for CobayaLikelihood instances. Default is 4.
        n_sobol_init : int, optional
            Number of initial Sobol quasi-random points. Default is 32.
        init_train_x : array-like, optional
            User-provided initial training points in parameter space, shape (n_points, ndim).
            If provided, these will be added to the initial GP training set. Default is None.
        init_train_y : array-like, optional
            User-provided initial training values (log-likelihood), shape (n_points, 1) or (n_points,).
            Must be provided if init_train_x is given. Default is None.
        min_evals : int, optional
            Minimum number of likelihood evaluations before checking convergence. Default is 200.
        max_evals : int, optional
            Maximum number of likelihood evaluations. Default is 1500.
        max_gp_size : int, optional
            Maximum number of points used to train the GP. Default is 1200.
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
        verbosity : str, optional
            Logging verbosity level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'. Default is 'INFO'.
            
        Notes
        -----
        MPI parallelization is handled automatically and transparently. Users do not
        need to manage MPI processes explicitly in their scripts. When running with
        MPI (e.g., `mpirun -n 4 python script.py`), worker processes automatically
        participate in parallel likelihood evaluations and GP hyperparameter optimization
        via the `mpi` module, while only the main process (rank 0) runs the optimization
        loop and manages results.
        """

        # Update logging verbosity if different from default
        update_verbosity(verbosity=verbosity)
        self.is_main = mpi.is_main_process()
        
        # Convert to Likelihood instance and store for all processes
        self.loglikelihood = self._prepare_likelihood(
            loglikelihood, param_list, param_bounds, param_labels,
            likelihood_name, confidence_for_unbounded, minus_inf
        )
        self.ndim = len(self.loglikelihood.param_list)
        
        # ============================================================================
        # WORKER PROCESS MINIMAL SETUP
        # ============================================================================
        if not self.is_main:
            self._setup_worker(seed, optimizer, resume)
        
        # ============================================================================
        # MAIN PROCESS FULL SETUP
        # ============================================================================
        else:
            self._setup_main_process(
                seed, optimizer, save, save_dir, save_step,
                do_final_ns, logz_threshold, convergence_n_iters, ei_goal,
                n_cobaya_init, n_sobol_init, min_evals, max_evals, max_gp_size,
                fit_step, wipv_batch_size, ns_step, num_hmc_warmup, num_hmc_samples,
                mc_points_size, mc_points_method, acq, use_clf, clf_type,
                clf_nsigma_threshold, minus_inf, resume
            )
        
        # Handle resume - ALL processes participate
        if resume and resume_file is not None:
            self._handle_resume(resume_file, use_clf)

        # Fresh start path - generate and train initial GP
        if self.fresh_start:
            self._handle_fresh_start(
                n_cobaya_init, n_sobol_init, init_train_x, init_train_y,
                use_clf, clf_type, clf_use_size, clf_update_step,
                clf_nsigma_threshold, minus_inf, optimizer, gp_kwargs
            )
        
        # Workers have GP loaded (either from resume or fresh init)
        # Workers now return, they've completed their initialization participation
        if not self.is_main:
            return
        
        # Finalize main process initialization
        self._finalize_main_initialization(
            min_evals, max_evals, max_gp_size, fit_step, ns_step,
            wipv_batch_size, num_hmc_warmup, num_hmc_samples, thinning,
            num_chains, mc_points_size, minus_inf, mc_points_method, zeta_ei
        )
    
    def _get_initial_training_data(self, n_cobaya_init, n_sobol_init, init_train_x=None, init_train_y=None):
        """
        Generate and evaluate initial training points for the GP.
        
        This method:
        1. Generates Sobol initial points in parallel
        2. Generates Cobaya initial points in parallel (if applicable)
        3. Adds user-provided initial points (if given)
        4. Removes duplicates
        5. Returns points and values in unit space for GP training
        
        Parameters
        ----------
        n_cobaya_init : int
            Number of Cobaya initial points (only for CobayaLikelihood).
        n_sobol_init : int
            Number of Sobol initial points.
        init_train_x : array-like, optional
            User-provided initial training points in parameter space.
        init_train_y : array-like, optional
            User-provided initial training values.
            
        Returns
        -------
        train_x : jax.numpy.ndarray
            Training points in unit cube space, shape (n_points, ndim).
        train_y : jax.numpy.ndarray
            Training values, shape (n_points, 1).
        """
        if n_sobol_init + n_cobaya_init == 0:
            raise ValueError("At least one of n_sobol_init or n_cobaya_init must be greater than zero.")
        
        # Generate Sobol points in parallel (generates minimum of 2 points)
        all_points, all_vals = self._generate_sobol_points(n_sobol_init)
        
        # Generate Cobaya points if needed
        if isinstance(self.loglikelihood, CobayaLikelihood) and n_cobaya_init > 0:
            cobaya_points, cobaya_vals = self._generate_cobaya_points(n_cobaya_init)
            if self.is_main and cobaya_points.size > 0:
                all_points = np.vstack([all_points, cobaya_points])
                all_vals = np.vstack([all_vals, cobaya_vals])
        
        # Only main process continues with processing
        if not self.is_main:
            return None, None
        
        # Add user-provided initial training data if available
        if init_train_x is not None and init_train_y is not None:
            init_train_x = np.atleast_2d(init_train_x)
            init_train_y = np.atleast_2d(init_train_y).reshape(-1, 1)
            
            if init_train_x.shape[0] != init_train_y.shape[0]:
                raise ValueError(
                    f"init_train_x and init_train_y must have same number of points. "
                    f"Got {init_train_x.shape[0]} and {init_train_y.shape[0]}"
                )
            if init_train_x.shape[1] != self.ndim:
                raise ValueError(
                    f"init_train_x must have {self.ndim} dimensions. "
                    f"Got {init_train_x.shape[1]}"
                )
            
            log.info(f"Adding {len(init_train_x)} user-provided initial points")
            all_points = np.vstack([all_points, init_train_x])
            all_vals = np.vstack([all_vals, init_train_y])
        elif init_train_x is not None or init_train_y is not None:
            raise ValueError("Both init_train_x and init_train_y must be provided together")
        
        # Remove duplicates
        unique_points, unique_indices = np.unique(all_points, axis=0, return_index=True)
        if len(unique_points) < len(all_points):
            log.warning(
                f"Found and removed {len(all_points) - len(unique_points)} duplicate points "
                f"from the initial set. Final set size: {len(unique_points)}."
            )
            init_points = all_points[unique_indices]
            init_vals = all_vals[unique_indices]
        else:
            init_points = all_points
            init_vals = all_vals
        
        self.results_manager.end_timing('True Objective Evaluations')
        
        # Convert to unit space for GP
        train_x = jnp.array(scale_to_unit(init_points, self.loglikelihood.param_bounds))
        train_y = jnp.array(init_vals)
        
        return train_x, train_y

    def _generate_sobol_points(self, n_sobol_init: int):
        """
        Generate Sobol initial points on main rank and evaluate them in parallel.
        
        Parameters
        ----------
        n_sobol_init : int
            Number of Sobol points to generate.
            
        Returns
        -------
        all_points : np.ndarray
            Sobol points in parameter space, shape (n_points, ndim).
        all_vals : np.ndarray
            Likelihood values, shape (n_points, 1).
        """
        from scipy.stats import qmc
        
        n_sobol = max(2, n_sobol_init)
        
        # Main process generates Sobol points
        if self.is_main:
            self.results_manager.start_timing('True Objective Evaluations')
            sobol = qmc.Sobol(d=self.ndim, scramble=True, rng=self.np_rng).random(n_sobol)
            sobol_points = scale_from_unit(sobol, self.loglikelihood.param_bounds)
            # Convert to list for scatter
            points_list = [sobol_points[i] for i in range(len(sobol_points))]
            log.info(f"Evaluating {len(sobol_points)} Sobol initial points")
        else:
            points_list = None

        # Scatter points across processes
        local_points = mpi.scatter(points_list, root=0)
        
        # Each process evaluates its assigned points
        if local_points is not None and len(local_points) > 0:
            local_points = np.atleast_2d(local_points)
            local_vals = np.array([self.loglikelihood(pt) for pt in local_points]).reshape(-1, 1)
        else:
            local_points = np.empty((0, self.ndim))
            local_vals = np.empty((0, 1))

        # Gather results back to main process
        gathered_points = mpi.gather(local_points, root=0)
        gathered_vals = mpi.gather(local_vals, root=0)

        if self.is_main:
            # Reconstruct full arrays from gathered results
            all_points = np.vstack([pts for pts in gathered_points if len(pts) > 0])
            all_vals = np.vstack([vals for vals in gathered_vals if len(vals) > 0])
            return all_points, all_vals
        else:
            return np.empty((0, self.ndim)), np.empty((0, 1))

    def _generate_cobaya_points(self, n_cobaya_init: int):
        """
        Generate Cobaya initial points in parallel across ranks.
        
        Parameters
        ----------
        n_cobaya_init : int
            Number of Cobaya points to generate.
            
        Returns
        -------
        all_points : np.ndarray
            Cobaya points in parameter space, shape (n_points, ndim).
        all_vals : np.ndarray
            Likelihood values, shape (n_points, 1).
        """
        if n_cobaya_init <= 0:
            raise ValueError("n_cobaya_init must be greater than zero to generate Cobaya points.")

        # Distribute requests across processes
        if self.is_main:
            inits_per_process = np.array_split(np.arange(n_cobaya_init), mpi.get_mpi_size())
        else:
            inits_per_process = None

        local_inits = mpi.scatter(inits_per_process, root=0)

        # Each process generates its assigned points
        local_points = []
        local_vals = [] 
        for _ in range(len(local_inits)):
            pt, lp = self.loglikelihood._get_single_valid_point(rng=self.np_rng)
            local_points.append(pt)
            local_vals.append(lp)

        # Gather all results
        gathered_points = mpi.gather(local_points, root=0)
        gathered_vals = mpi.gather(local_vals, root=0)

        if self.is_main:
            # Reconstruct full arrays from gathered results
            all_points = np.vstack([pts for pts in gathered_points if len(pts) > 0])
            all_vals = np.vstack([vals for vals in gathered_vals if len(vals) > 0])
            return all_points, all_vals
        else:
            return np.empty((0, self.ndim)), np.empty((0, 1))
    
    def _initialize_gp(self, train_x, train_y, use_clf, clf_type, clf_use_size, 
                       clf_update_step, clf_nsigma_threshold, minus_inf, 
                       optimizer, gp_kwargs):
        """
        Initialize and train the GP or GPwithClassifier.
        
        Main process creates GP, workers receive state_dict and reconstruct.
        """
        # Main process creates and trains GP
        if self.is_main:
            # Update GP kwargs with training data
            gp_kwargs.update({
                'train_x': train_x, 
                'train_y': train_y, 
                'param_names': self.loglikelihood.param_list, 
                'optimizer': optimizer
            })
            
            # Create GP or GPwithClassifier
            if use_clf:
                clf_threshold = max(75, get_threshold_for_nsigma(clf_nsigma_threshold, self.ndim))
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
            log.info(f" Hyperparameters before refit: {self.gp.hyperparams_dict()}")
            
            # Get GP state dict to broadcast
            gp_state = self.gp.state_dict()
        else:
            gp_state = None
        
        # Broadcast GP state to all processes
        gp_state = mpi.share(gp_state, root=0)
        
        # Workers reconstruct GP from state dict
        if not self.is_main:
            if 'clf_use_size' in gp_state:
                self.gp = GPwithClassifier.from_state_dict(gp_state)
            else:
                self.gp = GP.from_state_dict(gp_state)
        
        # ALL processes participate in distributed GP fitting
        self.distributed_gp_fit(self.gp, n_restarts=4, maxiters=500)
        
        if self.is_main:
            log.info(f" Hyperparameters after refit: {self.gp.hyperparams_dict()}")
            self.results_manager.end_timing('GP Training')
    
    def distributed_gp_fit(self, gp, n_restarts: int = 4, maxiters: int = 500):
        """
        Distribute GP fitting across MPI processes.
        
        Parameters
        ----------
        gp : GP or GPwithClassifier
            Gaussian Process model to fit.
        n_restarts : int, optional
            Number of random restarts for optimization. Default is 4.
        maxiters : int, optional
            Maximum iterations for each optimization. Default is 500.
            
        Returns
        -------
        dict or None
            Best fit result for main process, None for workers.
        """
        # Generate starting points
        init_params = jnp.log(gp.get_hyperparams())
        n_params = init_params.shape[0]
        
        if mpi.is_main_process():
            if n_restarts > 1:
                x0_random = self.np_rng.uniform(
                    gp.hyperparam_bounds[0], 
                    gp.hyperparam_bounds[1], 
                    size=(n_restarts - 1, n_params)
                )
                x0 = np.vstack([np.array(init_params), x0_random])
            else:
                x0 = np.atleast_2d(np.array(init_params))
            x0_list = [x0[i] for i in range(len(x0))]
        else:
            x0_list = None

        # Distribute work and fit
        local_x0 = mpi.scatter(x0_list, root=0)
        local_result = {'mll': -np.inf, 'params': None}
        
        if local_x0 is not None and len(local_x0) > 0:
            result = gp.fit(x0=np.array(local_x0), maxiter=maxiters)
            local_result = result if result else local_result

        # Gather and select best result
        all_results = mpi.gather(local_result, root=0)
        
        if mpi.is_main_process():
            valid_results = [r for r in all_results if r.get('mll', -np.inf) > -np.inf]
            best_result = max(valid_results, key=lambda r: r.get('mll', -np.inf)) if valid_results else {'params': None}
            best_params = best_result.get('params')
        else:
            best_params = None
            best_result = None

        # Share and update
        best_params = mpi.share(best_params, root=0)
        if best_params is not None:
            gp.update_hyperparams(best_params)

        return best_result if mpi.is_main_process() else None
    
    def update_gp(self, new_pts_u, new_vals, step = 0, verbose=True):
        """
        Update the GP with new points and values, and track hyperparameters.
        
        Main process updates GP with new data, broadcasts state to workers,
        then all processes participate in parallel fitting if needed.
        """
        # Main process updates GP and determines refit parameters
        if self.is_main:
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
            self.gp.update(new_pts_u, new_vals)
            
            # Get updated GP state to broadcast
            gp_state = self.gp.state_dict()
        else:
            refit = False
            n_restarts = maxiter = None
            gp_state = None
        
        # Broadcast updated GP state to all processes
        gp_state = mpi.share(gp_state, root=0)
        
        # Workers reconstruct GP from updated state dict
        if not self.is_main:
            use_clf = 'clf_use_size' in gp_state
            self.gp = load_gp_statedict(gp_state, use_clf)
        
        # Share refit decision and parameters to all processes
        refit, n_restarts, maxiter = mpi.share((refit, n_restarts, maxiter), root=0)
        
        # ALL processes participate in distributed fitting if refitting (synchronization point)
        if refit:
            self.distributed_gp_fit(self.gp, n_restarts=n_restarts, maxiters=maxiter)
        
        # Only main process continues with result processing and tracking
        if not self.is_main:
            return
        
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
        if not self.is_main:
            return None, None
        
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
        Evaluate the likelihood for new points using scatter/gather pattern.
        
        Parameters
        ----------
        new_pts_u : array-like
            Points in unit cube space to evaluate, shape (n_points, ndim).
        step : int
            Current iteration number.
        verbose : bool, optional
            Whether to log detailed information.
            
        Returns
        -------
        new_vals : jax.numpy.ndarray
            Evaluated likelihood values, shape (n_points, 1).
        """
        # Main process prepares points for evaluation
        if self.is_main:
            new_pts_u = jnp.atleast_2d(new_pts_u)
            new_pts = scale_from_unit(new_pts_u, self.loglikelihood.param_bounds)
            self.results_manager.start_timing('True Objective Evaluations')
            # Convert to list for scatter
            points_list = [new_pts[i] for i in range(len(new_pts))]
        else:
            points_list = None

        # Scatter points across processes
        local_points = mpi.scatter(points_list, root=0)
        
        # Each process evaluates its assigned points
        if local_points is not None and len(local_points) > 0:
            local_points = np.atleast_2d(local_points)
            local_vals = np.array([self.loglikelihood(pt) for pt in local_points]).reshape(-1, 1)
        else:
            local_vals = np.empty((0, 1))

        # Gather results back to main process
        gathered_vals = mpi.gather(local_vals, root=0)

        # Only main process continues with result processing
        if not self.is_main:
            return None
        
        # Reconstruct full array from gathered results
        new_vals = np.vstack([vals for vals in gathered_vals if len(vals) > 0])
        new_vals = jnp.array(new_vals)
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
        if not self.is_main:
            return False
        
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
        # Workers don't run the optimization loop
        if not self.is_main:
            return None
        
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
        if not self.is_main:
            return
        
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
        if not self.is_main:
            return False
        
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
        if not self.is_main:
            return
        
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
            self.distributed_gp_fit(self.gp, n_restarts=4, maxiters=500)
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
        if not self.is_main:
            return
        
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
            self.distributed_gp_fit(self.gp, n_restarts=4, maxiters=500)
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
        if not self.is_main:
            return False
        
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
        if not self.is_main:
            return
        
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

    # ============================================================================
    # INITIALIZATION HELPER METHODS
    # ============================================================================
    
    def _prepare_likelihood(self, loglikelihood, param_list, param_bounds, param_labels,
                           likelihood_name, confidence_for_unbounded, minus_inf):
        """Convert input to Likelihood instance if needed."""
        if isinstance(loglikelihood, Likelihood):
            return loglikelihood
        
        if isinstance(loglikelihood, (str, dict)):
            # Cobaya YAML file or info dict
            from .likelihood import CobayaLikelihood
            return CobayaLikelihood(
                input_file_dict=loglikelihood,
                confidence_for_unbounded=confidence_for_unbounded,
                minus_inf=minus_inf,
                name=likelihood_name if likelihood_name is not None else 'cobaya_model',
            )
        
        if callable(loglikelihood):
            # Create Likelihood instance from callable
            return Likelihood(
                loglikelihood=loglikelihood,
                param_list=param_list,
                param_bounds=param_bounds,
                param_labels=param_labels,
                name=likelihood_name,
                minus_inf=minus_inf,
            )
        
        raise ValueError(
            "loglikelihood must be one of: "
            "callable, string (Cobaya YAML path), dict (Cobaya info), or Likelihood instance"
        )
    
    def _setup_worker(self, seed, optimizer, resume):
        """Setup minimal attributes for worker processes."""
        worker_seed = (seed + mpi.rank()) if seed is not None else None
        set_global_seed(worker_seed)
        self.np_rng = get_numpy_rng()
        
        # Minimal attributes for MPI participation
        self.optimizer = optimizer
        self.fresh_start = not resume
        self.gp = None
    
    def _setup_main_process(self, seed, optimizer, save, save_dir, save_step,
                           do_final_ns, logz_threshold, convergence_n_iters, ei_goal,
                           n_cobaya_init, n_sobol_init, min_evals, max_evals, max_gp_size,
                           fit_step, wipv_batch_size, ns_step, num_hmc_warmup, num_hmc_samples,
                           mc_points_size, mc_points_method, acq, use_clf, clf_type,
                           clf_nsigma_threshold, minus_inf, resume):
        """Setup full attributes for main process."""
        set_global_seed(seed)
        self.np_rng = get_numpy_rng()
        
        # File paths and saving
        self.output_file = self.loglikelihood.name
        self.save = save
        self.save_step = save_step
        self.save_dir = save_dir
        if self.save:
            os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, self.output_file)
        
        # Convergence and optimization settings
        self.do_final_ns = do_final_ns
        self.logz_threshold = logz_threshold
        self.convergence_n_iters = convergence_n_iters
        self.converged = False
        self.prev_converged = False
        self.convergence_counter = 0
        self.min_delta_seen = np.inf
        self.termination_reason = "Max evaluation budget reached"
        self.ei_goal_log = np.log(ei_goal)
        
        # Validate optimizer
        if optimizer.lower() not in ['optax', 'scipy']:
            raise ValueError("optimizer must be either 'optax' or 'scipy'")
        self.optimizer = optimizer
        
        # Initialize results manager
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
        
        self.fresh_start = not resume
    
    def _handle_resume(self, resume_file, use_clf):
        """Handle resume from existing run."""
        if self.is_main:
            try:
                log.info(f" Attempting to resume from file {resume_file}")
                gp_file = resume_file + '_gp'
                self.gp = load_gp_file(gp_file, use_clf)
                
                # Test GP functionality
                _ = self.gp.predict_mean_single(self.gp.train_x[0])
                log.info(f"Loaded GP with {self.gp.train_x.shape[0]} training points")
                
                # Restore iteration and best point info
                if self.results_manager.is_resuming():
                    self.start_iteration = self.results_manager.get_last_iteration()
                    log.info(f"Resuming from iteration {self.start_iteration}")
                    log.info(f"Previous data: {len(self.results_manager.acquisition_values)} acquisition evaluations")
                    
                    if self.results_manager.best_loglike_values:
                        self.best_f = max(self.results_manager.best_loglike_values)
                        best_idx = self.results_manager.best_loglike_values.index(self.best_f)
                        self.best_pt_iteration = self.results_manager.best_loglike_iterations[best_idx]
                        log.info(f"Restored best loglikelihood: {self.best_f:.4f} at iteration {self.best_pt_iteration}")
                    else:
                        self.start_iteration = 0
                        self.best_pt_iteration = 0
                    
                    if self.results_manager.converged:
                        self.prev_converged = True
                        self.convergence_counter = 1
                        log.info(" Previous run had converged.")
                else:
                    self.start_iteration = 0
                    self.best_pt_iteration = 0
                    log.info("Starting fresh optimization")
                
                gp_state = self.gp.state_dict()
                resume_success = True
                
            except Exception as e:
                log.error(f" Failed to load GP from file {gp_file}: {e}")
                log.info(" Starting a fresh run instead.")
                self.fresh_start = True
                gp_state = None
                resume_success = False
        else:
            gp_state = None
            resume_success = None
        
        # Broadcast resume success and GP state to all processes
        resume_success = mpi.share(resume_success, root=0)
        
        if resume_success:
            gp_state = mpi.share(gp_state, root=0)
            if not self.is_main:
                self.gp = load_gp_statedict(gp_state, use_clf)
            self.fresh_start = False
        else:
            self.fresh_start = True
    
    def _handle_fresh_start(self, n_cobaya_init, n_sobol_init, init_train_x, init_train_y,
                           use_clf, clf_type, clf_use_size, clf_update_step,
                           clf_nsigma_threshold, minus_inf, optimizer, gp_kwargs):
        """Handle fresh start initialization."""
        if self.is_main:
            self.start_iteration = 0
            self.best_pt_iteration = 0
        
        # Generate and evaluate initial training points
        train_x, train_y = self._get_initial_training_data(
            n_cobaya_init=n_cobaya_init,
            n_sobol_init=n_sobol_init,
            init_train_x=init_train_x,
            init_train_y=init_train_y
        )
        
        # Initialize and train GP
        self._initialize_gp(
            train_x=train_x,
            train_y=train_y,
            use_clf=use_clf,
            clf_type=clf_type,
            clf_use_size=clf_use_size,
            clf_update_step=clf_update_step,
            clf_nsigma_threshold=clf_nsigma_threshold,
            minus_inf=minus_inf,
            optimizer=optimizer,
            gp_kwargs=gp_kwargs
        )
    
    def _finalize_main_initialization(self, min_evals, max_evals, max_gp_size, fit_step, ns_step,
                                     wipv_batch_size, num_hmc_warmup, num_hmc_samples, thinning,
                                     num_chains, mc_points_size, minus_inf, mc_points_method, zeta_ei):
        """Finalize main process initialization."""
        # Extract best point from GP
        if self.gp.train_y.size > 0:
            idx_best = jnp.argmax(self.gp.train_y)
            self.best_pt = scale_from_unit(self.gp.train_x[idx_best], self.loglikelihood.param_bounds).flatten()
            best_f_from_gp = float(self.gp.train_y.max()) * self.gp.y_std + self.gp.y_mean
        else:
            best_f_from_gp = -np.inf
            self.best_pt = None
        
        # Use restored best_f if available and better
        if not hasattr(self, 'best_f') or best_f_from_gp > getattr(self, 'best_f', -np.inf):
            self.best_f = best_f_from_gp
            if not hasattr(self, 'best_pt_iteration'):
                self.best_pt_iteration = self.start_iteration
        
        if self.best_pt is not None:
            self.best = {name: f"{float(val):.6f}" for name, val in zip(self.loglikelihood.param_list, self.best_pt)}
            log.info(f" Initial best point {self.best} with value = {self.best_f:.6f}")
        
        # Store optimization settings
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
        
        # Adjust wipv_batch_size for MPI load balancing
        if mpi.more_than_one_process():
            n_processes = mpi.size()
            original_batch = self.wipv_batch_size
            if self.wipv_batch_size % n_processes != 0:
                self.wipv_batch_size = (self.wipv_batch_size // n_processes) * n_processes
                if self.wipv_batch_size < n_processes:
                    self.wipv_batch_size = n_processes
                log.info(f" Adjusted wipv_batch_size from {original_batch} to {self.wipv_batch_size} "
                        f"(multiple of {n_processes} processes)")
        
        # Save initial GP
        self.gp.save(filename=f"{self.save_path}_gp")
        log.info(f" Saving GP to file {self.save_path}_gp")
        
        # Initialize KL divergence tracking
        self.prev_samples = None