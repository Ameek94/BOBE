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
from .utils.transforms import ParameterTransform
from .utils.flow import FlowTransform
from .utils.seed import set_global_seed, get_jax_key,  get_numpy_rng, get_new_jax_key
from .samplers import nested_sampling_Dy, sample_GP_NUTS
from .utils.log import get_logger, update_verbosity
from .utils.results import BOBEResults
from .acquisition import *
from .pool import MPI_Pool

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

def get_dimension_based_defaults(ndim: int):
    """
    Compute reasonable default values for run() parameters based on problem dimension.
    
    This method provides dimension-scaled defaults for parameters that should adapt
    to the complexity of the problem. Users can override these by providing explicit
    values to the run() method.
    
    Returns
    -------
    dict
        Dictionary of default parameter values keyed by parameter name.
    """
    
    defaults = {
        'min_evals': 8 * ndim,  # scales linearly with dimension
        'max_evals': 200 * ndim,  # more evals for higher dimensions
        'max_gp_size': min(2100, 160 * ndim),  # larger GP for higher dimensions
        'batch_size': 2 if ndim <=6 else min(8,int(2*(ndim/6))),  # 2-8 depending on dimension
        'ns_n_points': min(50, 2*ndim),  # nested sampling frequency, less for higher dimensions
        'num_hmc_warmup': 256 if ndim <= 6 else 512,  # more warmup for higher dimensions
        'num_hmc_samples': min(5000, max(512,int(4096*(ndim/20)))),  # more samples for higher dimensions, capped at 5000
        'mc_points_size': min(512, 32*ndim),  # more MC points for higher dimensions
        'num_chains': min(6, max(3,jax.device_count())),  # 3-6 chains depending on available devices
        'fit_n_points': min(50, 2*ndim),  # refit less often for higher dimensions
        'logz_threshold': 0.01 + 0.01*(ndim/6) if ndim<=6 else min(1.,0.1 + 0.1*(ndim/6)**2),  # looser threshold for higher dimensions
        'rotation_logz_threshold': 4 * (ndim/15)
    }
    return defaults

class BOBE:

    def __init__(self,
                loglikelihood: Union[Callable, str, Dict[str, Any], Likelihood],
                 param_list: List[str] = None,
                 param_bounds=None,
                 param_labels=None,
                 likelihood_name=None,
                 confidence_for_unbounded=0.9999995,
                 gp_kwargs: Dict[str, Any] = {},
                 n_cobaya_init=4,
                 n_sobol_init=4,
                 init_train_x=None,
                 init_train_y=None,
                 resume=False,
                 resume_file=None,
                 save_dir='.',
                 save=True,
                 save_step=5,
                 optimizer='scipy',
                 use_clf=False,
                 clf_type = "svm",
                 clf_nsigma_threshold=20,
                 clf_use_size = 10,
                 clf_update_step=1,
                 minus_inf=-1e10,
                 seed: Optional[int] = None,
                 verbosity: str = 'INFO',
                 rotation_matrix=None,
                 rotation_center=None,
                 rotation_is_fisher=False,
                 use_flow_transform=False,
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
        gp_kwargs : dict, optional
            Additional keyword arguments to pass to GP constructors. Default is {}.
        n_cobaya_init : int, optional
            Number of initial points from Cobaya reference distribution. 
            Only used for CobayaLikelihood instances. Default is 4.
        n_sobol_init : int, optional
            Number of initial Sobol quasi-random points. Default is 4.
        init_train_x : array-like, optional
            User-provided initial training points in parameter space, shape (n_points, ndim).
            If provided, these will be added to the initial GP training set. Default is None.
        init_train_y : array-like, optional
            User-provided initial training values (log-likelihood), shape (n_points, 1) or (n_points,).
            Must be provided if init_train_x is given. Default is None.
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
        minus_inf : float, optional
            Value representing negative infinity for failed evaluations. Default is -1e10.
        seed : int, optional
            Random seed for reproducibility. Default is None.
        verbosity : str, optional
            Logging verbosity level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'. Default is 'INFO'.
        rotation_matrix : array-like, shape (ndim, ndim), optional
            Covariance matrix (or Fisher matrix if rotation_is_fisher=True) for whitening
            the parameter space. Decorrelates parameters to improve GP performance for
            nearly-Gaussian likelihoods. Default is None (no rotation).
        rotation_center : array-like, shape (ndim,), optional
            Center point for the rotation in physical parameter space (e.g., best-fit
            or fiducial parameter values). If None with rotation_matrix given,
            defaults to the midpoint of param_bounds. Default is None.
        rotation_is_fisher : bool, optional
            If True, rotation_matrix is a Fisher matrix (will be inverted to get
            the covariance). Default is False.
        use_flow_transform : bool, optional
            If True, use a normalising flow (flowjax) to map physical parameter
            space to the unit cube instead of the covariance-rotation approach.
            The flow is initially untrained and falls back to linear scaling; it
            is trained automatically during the BO loop whenever ``flow_update_step``
            is set in ``run()``.  Mutually exclusive with ``rotation_matrix``.
            Default is False.
            
        Notes
        -----
        MPI parallelization is handled automatically and transparently. Users do not
        need to manage MPI processes explicitly in their scripts. When running with
        MPI (e.g., `mpirun -n 4 python script.py`), worker processes automatically
        participate in parallel likelihood evaluations and GP hyperparameter optimization
        via the `MPI_Pool` class, while only the main process (rank 0) runs the optimization
        loop and manages results. Worker processes enter a waiting loop after initialization
        and process tasks dispatched by the main process.
        """

        # Update logging verbosity if different from default
        update_verbosity(verbosity=verbosity)
        
        # Initialize MPI pool
        self.pool = MPI_Pool()
        self.is_main = self.pool.is_main_process
        self.is_mpi = self.pool.is_mpi
        
        # Convert to Likelihood instance and store for all processes
        self.loglikelihood = self._prepare_likelihood(
            loglikelihood, param_list, param_bounds, param_labels,
            likelihood_name, confidence_for_unbounded, minus_inf
        )
        self.ndim = len(self.loglikelihood.param_list)
        
        # Create the parameter transform (handles unit-cube scaling and optional rotation)
        if use_flow_transform:
            if rotation_matrix is not None:
                log.warning(
                    "Both use_flow_transform=True and rotation_matrix supplied. "
                    "rotation_matrix will be ignored; the flow transform takes precedence."
                )
            self.transform = FlowTransform(
                param_bounds=self.loglikelihood.param_bounds,
            )
            log.info("Using FlowTransform (will train flow on MC samples during run).")
        else:
            self.transform = ParameterTransform(
                param_bounds=self.loglikelihood.param_bounds,
                rotation_matrix=rotation_matrix,
                rotation_center=rotation_center,
                rotation_is_fisher=rotation_is_fisher,
            )
        
        if not self.is_main:
            # Workers only need likelihood and seed - everything else is handled in worker_wait
            self.pool.worker_wait(likelihood=self.loglikelihood, seed=seed)
            return  # Workers never return from worker_wait until pool.close()
        
        # MAIN PROCESS FULL SETUP
        self._setup_main_process(
            seed, optimizer, save, save_dir, save_step,
            n_cobaya_init, n_sobol_init, use_clf, clf_type,
            clf_nsigma_threshold, minus_inf, resume
        )
        
        # handle resume if needed
        if resume and resume_file is not None:
            self._handle_resume(resume_file, use_clf)

        # Fresh start path - generate and train initial GP (main process only)
        if self.fresh_start:
            self._handle_fresh_start(
                n_cobaya_init, n_sobol_init, init_train_x, init_train_y,
                use_clf, clf_type, clf_use_size, clf_update_step,
                clf_nsigma_threshold, minus_inf, optimizer, gp_kwargs
            )
        
        # Finalize main process initialization

        # Extract best point from GP
        if self.gp.train_y.size > 0:
            idx_best = jnp.argmax(self.gp.train_y)
            self.best_pt = np.asarray(self.transform.from_unit(self.gp.train_x[idx_best])).flatten()
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
            log.info(f"Initial best point {self.best} with value = {self.best_f:.6f}")
        
        # Save initial GP and transform state
        self.gp.save(filename=f"{self.save_path}_gp")
        self._save_transform()
        log.info(f"Saving GP to file {self.save_path}_gp")
        
        # Initialize for KL divergence tracking
        self.prev_samples = None

        # Objective evaluation counter - restored from saved state on resume, or seeded
        # from GP size on fresh start. gp_info is populated on resume by results_manager.
        self.total_objective_evals = self.results_manager.gp_info.get(
            'total_objective_evals',
            self.results_manager.gp_info.get('total_true_evals', self.gp.npoints)  # backward compat
        )

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
                name=likelihood_name if likelihood_name is not None else 'CobayaLikelihood',
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
    
    
    def _setup_main_process(self, seed, optimizer, save, save_dir, save_step,
                           n_cobaya_init, n_sobol_init, use_clf, clf_type,
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
        
        # Validate optimizer
        if optimizer.lower() not in ['optax', 'scipy']:
            raise ValueError("optimizer must be either 'optax' or 'scipy'")
        self.optimizer = optimizer
        self.minus_inf = minus_inf
        
        # Initialize results manager (settings will be updated when run() is called)
        self.results_manager = BOBEResults(
            output_file=self.output_file,
            save_dir=self.save_dir,
            param_names=self.loglikelihood.param_list,
            param_labels=self.loglikelihood.param_labels,
            param_bounds=self.loglikelihood.param_bounds,
            settings={
                'n_cobaya_init': n_cobaya_init,
                'n_sobol_init': n_sobol_init,
                'use_clf': use_clf,
                'clf_type': clf_type,
                'clf_nsigma_threshold': clf_nsigma_threshold,
                'minus_inf': minus_inf,
                'seed': seed,
                'transform_state': self.transform.state_dict(),
            },
            likelihood_name=self.loglikelihood.name,
            resume_from_existing=resume
        )
        
        self.fresh_start = not resume
        # Diagnostic flow instance (trained alongside the rotation for comparison)
        self._diag_flow = None
    
    def _handle_resume(self, resume_file, use_clf):
        """Handle resume from existing run (main process only)."""
        try:
            log.info(f"Attempting to resume from file {resume_file}")
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
                    # Store last convergence info for threshold comparison in run()
                    if self.results_manager.convergence_history:
                        last_conv = self.results_manager.convergence_history[-1]
                        self.prev_convergence_delta = last_conv.delta
                        self.prev_convergence_threshold = last_conv.threshold
                    else:
                        self.prev_convergence_delta = None
                        self.prev_convergence_threshold = None
                else:
                    # Not converged in previous run
                    self.prev_converged = False
                    self.prev_convergence_delta = None
                    self.prev_convergence_threshold = None
            else:
                self.start_iteration = 0
                self.best_pt_iteration = 0
                log.info("Starting fresh optimization")
            
            # Restore the parameter transform from the saved state
            self._load_transform(resume_file)
            
            self.fresh_start = False
            
        except Exception as e:
            log.error(f"Failed to load GP from file {gp_file}: {e}")
            log.info("Starting a fresh run instead.")
            self.fresh_start = True
    
    def _save_transform(self):
        """Save the parameter transform state to disk."""
        transform_file = self.save_path + '_transform.npz'
        state = self.transform.state_dict()
        np.savez(transform_file, **{k: v for k, v in state.items()})
        log.debug(f"Saved transform state to {transform_file}")

        # For FlowTransform, also save the flow model weights via equinox
        if isinstance(self.transform, FlowTransform) and self.transform.is_flow_trained:
            flow_base_path = self.save_path + '_flow_model'
            self.transform.save_flow(flow_base_path)

    def _save_dropped_pool(self):
        """Persist the dropped-point pool to disk so it survives a resume."""
        pool_file = self.save_path + '_dropped_pool.npz'
        np.savez(pool_file,
                 x_phys=self._dropped_pool_x_phys,
                 y_raw=self._dropped_pool_y_raw)
        log.debug(f"Saved dropped pool ({len(self._dropped_pool_x_phys)} pts) to {pool_file}")

    def _load_dropped_pool(self):
        """Restore the dropped-point pool from disk (empty arrays if file absent)."""
        pool_file = self.save_path + '_dropped_pool.npz'
        if os.path.exists(pool_file):
            data = np.load(pool_file)
            self._dropped_pool_x_phys = data['x_phys']
            self._dropped_pool_y_raw  = data['y_raw']
            if self._dropped_pool_x_phys.shape[0] > 0:
                log.info(f"Restored dropped pool with {self._dropped_pool_x_phys.shape[0]} points.")
        else:
            self._dropped_pool_x_phys = np.zeros((0, self.ndim))
            self._dropped_pool_y_raw  = np.zeros(0)

    def _load_transform(self, resume_file):
        """
        Load the parameter transform state from a previous run.
        
        Falls back to the current transform if no saved state exists
        (e.g. resuming from a run that did not use rotation).
        """
        transform_file = resume_file + '_transform.npz'
        if os.path.exists(transform_file) or os.path.exists(transform_file + '.npz'):
            try:
                fname = transform_file if os.path.exists(transform_file) else transform_file + '.npz'
                data = np.load(fname, allow_pickle=True)
                state = {}
                for key in data.files:
                    value = data[key]
                    if isinstance(value, np.ndarray) and value.shape == ():
                        state[key] = value.item()
                    else:
                        state[key] = value

                # Choose the correct class to restore
                transform_type = state.get('transform_type', 'linear')
                if transform_type == 'flow':
                    self.transform = FlowTransform.from_state_dict(state)
                    # Attempt to restore the flow model weights via equinox
                    flow_base_path = resume_file + '_flow_model'
                    if (os.path.exists(flow_base_path + '.eqx') and
                            os.path.exists(flow_base_path + '_arch.json')):
                        loaded = self.transform.load_flow(flow_base_path)
                        if loaded:
                            log.info("Restored flow model from equinox files.")
                    else:
                        log.info(
                            "No flow model files found; FlowTransform will use linear "
                            "fallback until flow is retrained."
                        )
                else:
                    self.transform = ParameterTransform.from_state_dict(state)

                log.info(f"Restored transform state from {fname} (type={transform_type})")
            except Exception as e:
                log.warning(f"Failed to load transform state from {transform_file}: {e}. Using current transform.")
        else:
            log.debug("No saved transform state found, using current transform.")

    def _handle_fresh_start(self, n_cobaya_init, n_sobol_init, init_train_x, init_train_y,
                           use_clf, clf_type, clf_use_size, clf_update_step,
                           clf_nsigma_threshold, minus_inf, optimizer, gp_kwargs):
        """Handle fresh start initialization (main process only)."""
        self.start_iteration = 0
        self.best_pt_iteration = 0
        self.prev_converged = False
        self.prev_convergence_delta = None
        self.prev_convergence_threshold = None
        
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
        train_x = jnp.array(self.transform.to_unit(init_points))
        train_y = jnp.array(init_vals)
        
        return train_x, train_y

    def _generate_sobol_points(self, n_sobol_init: int):
        """
        Generate Sobol initial points on main rank and evaluate them in parallel using pool.
        
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
        
        # Main process generates Sobol points and distributes via pool
        if self.is_main:
            self.results_manager.start_timing('True Objective Evaluations')
            sobol = qmc.Sobol(d=self.ndim, scramble=True, rng=self.np_rng).random(n_sobol)
            sobol_points = np.asarray(self.transform.from_unit(sobol))
            log.info(f"Evaluating {len(sobol_points)} Sobol initial points")
            
            # Use pool to evaluate points in parallel
            all_vals = self.pool.run_map_objective(self.loglikelihood, sobol_points)
            all_vals = np.atleast_2d(all_vals).reshape(-1, 1)
            return sobol_points, all_vals
        else:
            return np.empty((0, self.ndim)), np.empty((0, 1))

    def _generate_cobaya_points(self, n_cobaya_init: int):
        """
        Generate Cobaya initial points in parallel using pool.
        
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

        # Use pool to generate Cobaya points in parallel
        if self.is_main:
            results_tuples = self.pool.get_cobaya_initial_points(
                self.loglikelihood, n_cobaya_init, rng=self.np_rng
            )
            
            # Extract points and values from tuples
            all_points = np.array([pt for pt, _ in results_tuples])
            all_vals = np.array([[lp] for _, lp in results_tuples])
            return all_points, all_vals
        else:
            return np.empty((0, self.ndim)), np.empty((0, 1))
    
    def _initialize_gp(self, train_x, train_y, use_clf, clf_type, clf_use_size, 
                       clf_update_step, clf_nsigma_threshold, minus_inf, 
                       optimizer, gp_kwargs):
        """
        Initialize and train the GP or GPwithClassifier.
        
        Main process creates GP, workers will receive it via pool during fitting.
        """
        # Only main process creates and trains GP
        if not self.is_main:
            return
        
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
        log.info(f"Hyperparameters before refit: {self.gp.hyperparams_dict()}")
        
        # Use pool to fit GP in parallel
        self.pool.gp_fit(self.gp, n_restarts=4, maxiters=500, rng=self.np_rng, use_pool=True)
        
        log.info(f"Hyperparameters after refit: {self.gp.hyperparams_dict()}")
        self.results_manager.end_timing('GP Training')
    

    # ============================================================================
    # RUN HELPER METHODS
    # ============================================================================

    def update_gp(self, new_pts_u, new_vals, step = 0, verbose=True):
        """
        Update the GP with new points and values, and track hyperparameters.
        
        Uses pool for parallel GP fitting when refitting is needed.
        Refits based on number of points added to GP since last fit.
        """
        # Only main process updates GP
        if not self.is_main:
            return
        
        self.results_manager.start_timing('GP Training')
        
        # Track GP size before update
        gp_size_before = self.gp.train_x.shape[0]
        
        # Update GP with new data
        self.gp.update(new_pts_u, new_vals)
        
        # Track actual points added (accounts for filtering by classifier or other mechanisms)
        gp_size_after = self.gp.train_x.shape[0]
        actual_points_added = gp_size_after - gp_size_before
        self.n_points_since_last_fit += actual_points_added
        
        # Determine refit parameters based on training set size and points added
        if gp_size_after < 200:
            # For small training sets, refit more frequently
            refit_threshold = min(4, self.fit_n_points)
            maxiter = 1000
            n_restarts = 8
        elif 200 < gp_size_after < 800:
            # For moderate size training sets
            refit_threshold = self.fit_n_points
            n_restarts = 4
            maxiter = 500
        else:
            # For large training sets, refit less frequently
            refit_threshold = max(50, self.fit_n_points)
            n_restarts = 4
            maxiter = 200
        
        refit = (self.n_points_since_last_fit >= refit_threshold)
        
        # Use pool for parallel GP fitting if refitting
        if refit:
            log.info(f"Refitting GP hyperparameters with {self.gp.train_x.shape[0]} training points ")
            self.pool.gp_fit(self.gp, n_restarts=n_restarts, maxiters=maxiter, rng=self.np_rng, use_pool=True)
            # Reset counter after successful refit
            self.n_points_since_last_fit = 0
        
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
        log.info(f"Optimizing acquisition function '{self.acquisition.name}' to get next {n_batch} points")
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
        Evaluate the likelihood for new points using pool.
        
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
        # Only main process evaluates
        if not self.is_main:
            return None
        
        new_pts_u = jnp.atleast_2d(new_pts_u)
        new_pts = np.asarray(self.transform.from_unit(new_pts_u))
        
        self.results_manager.start_timing('True Objective Evaluations')
        
        # Use pool to evaluate points in parallel
        new_vals = self.pool.run_map_objective(self.loglikelihood, new_pts)
        new_vals = jnp.atleast_2d(new_vals).reshape(-1, 1)
        
        self.results_manager.end_timing('True Objective Evaluations')

        best_new_idx = np.argmax(new_vals)
        best_new_val = float(np.max(new_vals))
        best_new_pt = new_pts[best_new_idx]
        if float(best_new_val) > self.best_f:
            self.best_f = float(best_new_val)
            self.best_pt = best_new_pt
            self.best = {name: f"{float(val):.6f}" for name, val in zip(self.loglikelihood.param_list, self.best_pt.flatten())}
            self.best_pt_iteration = step

        # Increment dedicated objective-eval counter
        self.total_objective_evals += len(new_pts)
        log.info(f"Evaluated objective at {len(new_pts)} new points (total objective evals: {self.total_objective_evals})")
        for k, new_pt in enumerate(new_pts):
            new_pt_vals = {name: f"{float(val):.4f}" for name, val in zip(self.loglikelihood.param_list, new_pt.flatten())}
            log.debug(f"New point {new_pt_vals}, {k+1}/{len(new_pts)}")
            predicted_val = self.gp.predict_mean_single(new_pts_u[k])
            log.debug(f"Objective function value = {new_vals[k].item():.4f}, GP predicted value = {predicted_val.item():.4f}")

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

    def _update_rotation(self, step: int) -> bool:
        """
        Re-estimate the covariance from current MC samples and rebuild the
        ParameterTransform + GP training set in the new rotated unit-cube space.

        Can be called even when no initial rotation matrix was provided:
        in that case the first call establishes the rotation from scratch using
        the current MC sample covariance, and the transform switches from a
        simple linear scaling to a rotated eigenspace transform.

        The new rotation center is taken from ``self.best_pt`` (physical space).
        A weighted sample covariance is computed from ``self.mc_samples`` in
        physical space.  When an existing rotation is in use, the update is
        skipped when the symmetric KL divergence between the old and new
        Gaussians is below ``self.rotation_kl_threshold``.  For the very first
        rotation (no prior rotation), the KL check is bypassed and the update
        always proceeds.

        After a successful update:
        - GP training points outside [0, 1] in the new coords are **dropped**
          (not clipped) to avoid corrupting the emulated function.
        - A 4-restart warm-start refit is performed (first restart from current
          hyperparameters, subsequent ones randomised).

        Parameters
        ----------
        step : int
            Current BO iteration number (used for logging).

        Returns
        -------
        bool
            True if the rotation was updated, False if skipped.
        """
        # Guard: rotation update is not applicable when using a FlowTransform
        if isinstance(self.transform, FlowTransform):
            log.debug("[Rotation update] Skipped — transform is FlowTransform; use flow_update_step instead.")
            return False

        # 1. Convert MC samples from old unit space to physical space
        mc_x_unit = np.array(self.mc_samples['x'])        # (N, r)
        mc_x_phys = self.transform.from_unit(mc_x_unit)   # (N, D)
        N = mc_x_phys.shape[0]

        if N < self.ndim + 2:
            log.warning(f"[Rotation update] Too few MC samples ({N}) to estimate covariance; skipping.")
            return False

        # 2. Weighted sample covariance in physical space
        weights = self.mc_samples.get('weights', None)
        if weights is not None:
            w = np.array(weights, dtype=np.float64)
            w = np.clip(w, 0.0, None)
            w_sum = w.sum()
            if w_sum <= 0.0:
                w = np.ones(N)
            w /= w.sum()
            # Covariance centered on sample mean (independent of rotation center)
            sample_mean = np.average(mc_x_phys, weights=w, axis=0)
            diff = mc_x_phys - sample_mean
            new_cov = (diff * w[:, None]).T @ diff
        else:
            new_cov = np.cov(mc_x_phys.T)
        new_cov = 0.5 * (new_cov + new_cov.T)
        new_cov += 1e-10 * np.eye(self.ndim)   # numerical guard

        # 3. New rotation center: use stored best-fit point (physical space)
        new_center = np.array(self.best_pt).flatten()

        # 4. KL divergence check between old and new physical-space Gaussians.
        #    Skipped when no rotation exists yet (first rotation is always applied).
        if self.transform.uses_rotation:
            old_cov    = self.transform._covariance_phys
            old_center = self.transform._theta_center
            try:
                kl_dict = kl_divergence_gaussian(old_center, old_cov, new_center, new_cov)
                kl_sym  = float(kl_dict['symmetric'])
            except (np.linalg.LinAlgError, Exception) as e:
                log.warning(f"[Rotation update] KL divergence computation failed: {e}; skipping.")
                return False

            log.info(f"[Rotation update {self.rotation_update_count + 1}] Symmetric KL = {kl_sym:.4f} "
                     f"(threshold = {self.rotation_kl_threshold:.4f})")
            if kl_sym < self.rotation_kl_threshold:
                log.info("[Rotation update] KL below threshold; skipping rotation update.")
                return False
        else:
            log.info(f"[Rotation update {self.rotation_update_count + 1}] No existing rotation; "
                     "establishing first rotation from sample covariance (KL check bypassed).")

        # 5. Build new ParameterTransform
        n_sigma = getattr(self.transform, '_n_sigma', 5.0)
        try:
            new_transform = ParameterTransform(
                param_bounds=self.loglikelihood.param_bounds,
                rotation_matrix=new_cov,
                rotation_center=new_center,
                rotation_is_fisher=False,
                n_sigma=n_sigma,
            )
        except (ValueError, np.linalg.LinAlgError) as e:
            log.warning(f"[Rotation update] Failed to build new transform: {e}; skipping.")
            return False

        # 5b+6. Merge current GP data with the full dropped pool, remap to the new unit cube,
        #       and split: in-bounds points go to the GP, out-of-bounds points become the new
        #       pool (auto-recovering any pool points that now fall in-bounds).
        if isinstance(self.gp, GPwithClassifier):
            src_x_phys = self.transform.from_unit(np.array(self.gp.train_x_clf))
            src_y_raw  = np.array(self.gp.train_y_clf).flatten()
        else:
            src_x_phys = self.transform.from_unit(np.array(self.gp.train_x))
            src_y_raw  = np.array(self.gp.train_y).flatten() * float(self.gp.y_std) + float(self.gp.y_mean)

        all_x_phys = np.concatenate([src_x_phys, self._dropped_pool_x_phys], axis=0)
        all_y_raw  = np.concatenate([src_y_raw,  self._dropped_pool_y_raw],  axis=0)

        all_u     = new_transform.to_unit(all_x_phys, clip=False)
        in_bounds = np.all((all_u >= 0.0) & (all_u <= 1.0), axis=1)
        n_src     = src_x_phys.shape[0]
        n_recovered = int(in_bounds[n_src:].sum())   # pool points now in-bounds
        n_dropped   = int((~in_bounds).sum())        # total points going to pool
        if n_recovered > 0:
            log.info(f"[Rotation update] {n_recovered} point(s) recovered from dropped pool.")
        if n_dropped > 0:
            log.info(f"[Rotation update] {n_dropped}/{len(all_x_phys)} points outside new unit cube → pool.")

        # Update the pool with OOB points
        self._dropped_pool_x_phys = all_x_phys[~in_bounds]
        self._dropped_pool_y_raw  = all_y_raw[~in_bounds]

        x_ib = all_u[in_bounds]
        y_ib = all_y_raw[in_bounds]

        if x_ib.shape[0] < self.ndim + 2:
            log.warning(f"[Rotation update] Only {x_ib.shape[0]} points remain after filtering — too few; skipping.")
            return False

        # Hand off to GP — threshold / standardisation handled inside
        if isinstance(self.gp, GPwithClassifier):
            self.gp.remap_from_full_dataset(x_ib, y_ib)
        else:
            self.gp.remap_from_raw(x_ib, y_ib)

        # 7. Commit new transform
        self.transform = new_transform

        # 8. Remap current MC samples; drop those outside new unit cube
        mc_x_new_unit  = new_transform.to_unit(mc_x_phys, clip=False)
        in_bounds_mc   = np.all((mc_x_new_unit >= 0.0) & (mc_x_new_unit <= 1.0), axis=1)
        n_dropped_mc   = int((~in_bounds_mc).sum())
        if n_dropped_mc > 0:
            log.info(f"[Rotation update] Dropping {n_dropped_mc}/{N} MC samples outside new unit cube.")
        mc_x_new_unit = mc_x_new_unit[in_bounds_mc]
        new_mc = {'x': mc_x_new_unit, 'method': self.mc_samples.get('method', 'unknown')}
        for key in ('weights', 'logl', 'logp'):
            if key in self.mc_samples:
                new_mc[key] = np.array(self.mc_samples[key])[in_bounds_mc]
        if 'best' in self.mc_samples:
            new_mc['best'] = self.mc_samples['best']
        self.mc_samples = new_mc

        # 9. Warm-start GP refit (pool.gp_fit always uses current hyperparams as first restart)
        log.info(f"[Rotation update {self.rotation_update_count + 1}] Warm-start refitting GP "
                 f"with {x_ib.shape[0]} points (4 restarts, 500 iters)")
        self.results_manager.start_timing('GP Training')
        self.pool.gp_fit(self.gp, n_restarts=4, maxiters=500, rng=self.np_rng, use_pool=True)
        self.results_manager.end_timing('GP Training')

        # 10. Persist updated transform, GP, and dropped pool
        self._save_transform()
        self._save_dropped_pool()
        self.gp.save(filename=f"{self.save_path}_gp")
        self.rotation_update_count += 1
        log.info(f"[Rotation update] Complete. Count={self.rotation_update_count}. "
                 f"New transform: {self.transform}")

        # Persist rotation state so it survives a resume
        self.results_manager.gp_info.update({
            'rotation_update_count': self.rotation_update_count,
            'last_rotation_ii': self.last_rotation_ii,     # set by caller after this returns
            'last_rotation_acq_val': self.last_rotation_acq_val,  # set by caller after this returns
        })

        # Diagnostic: train a stand-alone flow and compare to rotation Gaussian
        self._compute_flow_rotation_kl_diag(mc_x_phys, new_center, new_cov, weights=weights)

        return True

    # ------------------------------------------------------------------
    # Flow-vs-rotation diagnostic
    # ------------------------------------------------------------------

    def _compute_flow_rotation_kl_diag(self, mc_x_phys, rotation_center, rotation_cov, weights=None):
        """
        Diagnostic: train a normalising flow on ``mc_x_phys``, draw samples
        from it, compute the sample mean and covariance, then compute the
        symmetric KL divergence between that Gaussian and the rotation Gaussian
        N(rotation_center, rotation_cov) using ``kl_divergence_gaussian``.

        The diagnostic FlowTransform is stored in ``self._diag_flow`` and
        retrained on every call.

        Parameters
        ----------
        mc_x_phys : ndarray, shape (N, D)
            Physical-space MC posterior samples used to train the flow.
        rotation_center : ndarray, shape (D,)
            Mean of the rotation Gaussian.
        rotation_cov : ndarray, shape (D, D)
            Covariance of the rotation Gaussian.
        weights : ndarray or None
            Ignored (kept for call-signature compatibility).
        """
        N = mc_x_phys.shape[0]
        if N < max(self.ndim + 2, 32):
            log.warning(f"[Flow diag] Too few samples ({N}) for flow diagnostic; skipping.")
            return

        # ------ 1. Train (or retrain) diagnostic flow ------
        if self._diag_flow is None:
            self._diag_flow = FlowTransform(param_bounds=self.loglikelihood.param_bounds)
        _flow_kwargs = dict(
            flow_layers=6,
            nn_width=32,
            nn_depth=2,
            learning_rate=5e-4,
            max_epochs=300,
            batch_size=min(256, N),
            max_patience=20,
        )
        log.info(f"[Flow diag] Training diagnostic flow on {N} samples …")
        try:
            self._diag_flow.train_flow(mc_x_phys, **_flow_kwargs)
        except Exception as e:
            log.warning(f"[Flow diag] Flow training failed: {e}")
            return

        # ------ 2. Draw samples from the trained flow ------
        n_draw = max(N, 2000)
        try:
            jkey = jax.random.key(0)
            flow_samples = np.asarray(
                self._diag_flow._flow.sample(jkey, (n_draw,))
            )
        except Exception as e:
            log.warning(f"[Flow diag] Flow sampling failed: {e}")
            return

        # ------ 3. Gaussian moments of the flow samples (covariance only) ------
        # Centre the flow covariance on the MAP point (same as the rotation
        # Gaussian) so the KL compares shapes only, not mean shifts.
        diff_flow = flow_samples - rotation_center
        flow_cov  = (diff_flow.T @ diff_flow) / (n_draw - 1)
        flow_cov  = 0.5 * (flow_cov + flow_cov.T) + 1e-10 * np.eye(self.ndim)

        # ------ 4. Symmetric KL divergence (covariance-only, shared MAP center) ------
        try:
            kl_dict = kl_divergence_gaussian(
                rotation_center, flow_cov, rotation_center, rotation_cov
            )
        except Exception as e:
            log.warning(f"[Flow diag] KL computation failed: {e}")
            return

        kl_sym = float(kl_dict['symmetric'])
        log.info(
            f"[Flow diag | rotation {self.rotation_update_count}] "
            f"KL(flow || Gaussian) sym={kl_sym:.4f}  "
            f"fwd={kl_dict['forward']:.4f}  rev={kl_dict['reverse']:.4f}"
        )

        # Store result for later inspection
        self.results_manager.gp_info.setdefault('flow_rotation_diag', []).append({
            'rotation_count': self.rotation_update_count,
            'kl_symmetric': kl_sym,
            'kl_forward': float(kl_dict['forward']),
            'kl_reverse': float(kl_dict['reverse']),
            'n_samples': N,
        })

    # ------------------------------------------------------------------
    # Flow transform update
    # ------------------------------------------------------------------

    def _update_flow_transform(self, step: int, flow_kwargs: dict = None) -> bool:
        """
        Train (or retrain) the FlowTransform on current MC samples and remap
        the GP training set into the new unit-cube coordinates.

        This is the flow-transform analogue of ``_update_rotation()``.  It:

        1. Converts MC samples from the old unit cube back to physical space.
        2. Trains (or retrains) the flowjax coupling flow on those samples.
        3. Remaps all GP / classifier training data through the new transform.
        4. Rebuilds the dropped-point pool in the new coordinate system.
        5. Warm-start refits the GP and persists the updated state.

        Parameters
        ----------
        step : int
            Current BO iteration number (used for logging).
        flow_kwargs : dict, optional
            Extra keyword arguments forwarded to ``FlowTransform.train_flow()``.
            Supported keys: ``flow_layers``, ``nn_width``, ``nn_depth``,
            ``learning_rate``, ``max_epochs``, ``batch_size``, ``max_patience``.

        Returns
        -------
        bool
            ``True`` if the transform was updated, ``False`` if skipped.
        """
        if not isinstance(self.transform, FlowTransform):
            log.warning("[Flow update] _update_flow_transform called but transform is not FlowTransform; skipping.")
            return False

        # 1. Convert current MC samples to physical space
        mc_x_unit = np.array(self.mc_samples['x'])        # (N, D)
        mc_x_phys = self.transform.from_unit(mc_x_unit)   # (N, D)
        N = mc_x_phys.shape[0]

        if N < max(self.ndim + 2, 32):
            log.warning(
                f"[Flow update] Too few MC samples ({N}) to train flow; skipping."
            )
            return False

        # 2. Also pull in the full GP training data (or classifier data) in physical space
        if isinstance(self.gp, GPwithClassifier):
            src_x_phys = self.transform.from_unit(np.array(self.gp.train_x_clf))
            src_y_raw  = np.array(self.gp.train_y_clf).flatten()
        else:
            src_x_phys = self.transform.from_unit(np.array(self.gp.train_x))
            src_y_raw  = (
                np.array(self.gp.train_y).flatten() * float(self.gp.y_std) + float(self.gp.y_mean)
            )

        # Merge GP points with dropped pool
        all_x_phys = np.concatenate([src_x_phys, self._dropped_pool_x_phys], axis=0)
        all_y_raw  = np.concatenate([src_y_raw,  self._dropped_pool_y_raw],  axis=0)

        # 3. Train the flow on the MC posterior samples
        log.info(
            f"[Flow update {self.flow_update_count + 1}] Training flow on {N} MC samples (D={self.ndim})…"
        )
        kwargs = flow_kwargs or {}
        self.results_manager.start_timing('Flow Training')
        try:
            self.transform.train_flow(mc_x_phys, **kwargs)
        except Exception as e:
            log.warning(f"[Flow update] Flow training failed: {e}; skipping update.")
            self.results_manager.end_timing('Flow Training')
            return False
        self.results_manager.end_timing('Flow Training')

        # 4. Remap GP/classifier data through the new transform
        all_u     = self.transform.to_unit(all_x_phys, clip=False)
        in_bounds = np.all((all_u >= 0.0) & (all_u <= 1.0), axis=1)
        n_src     = src_x_phys.shape[0]
        n_recovered = int(in_bounds[n_src:].sum())
        n_dropped   = int((~in_bounds).sum())
        if n_recovered > 0:
            log.info(f"[Flow update] {n_recovered} point(s) recovered from dropped pool.")
        if n_dropped > 0:
            log.info(f"[Flow update] {n_dropped}/{len(all_x_phys)} points outside new unit cube → pool.")

        self._dropped_pool_x_phys = all_x_phys[~in_bounds]
        self._dropped_pool_y_raw  = all_y_raw[~in_bounds]

        x_ib = all_u[in_bounds]
        y_ib = all_y_raw[in_bounds]

        if x_ib.shape[0] < self.ndim + 2:
            log.warning(
                f"[Flow update] Only {x_ib.shape[0]} points remain after filtering — too few; skipping."
            )
            return False

        if isinstance(self.gp, GPwithClassifier):
            self.gp.remap_from_full_dataset(x_ib, y_ib)
        else:
            self.gp.remap_from_raw(x_ib, y_ib)

        # 5. Remap MC samples into new unit cube (flow already trained above)
        mc_x_new_unit = self.transform.to_unit(mc_x_phys, clip=False)
        in_bounds_mc  = np.all((mc_x_new_unit >= 0.0) & (mc_x_new_unit <= 1.0), axis=1)
        n_dropped_mc  = int((~in_bounds_mc).sum())
        if n_dropped_mc > 0:
            log.info(f"[Flow update] Dropping {n_dropped_mc}/{N} MC samples outside new unit cube.")
        mc_x_new_unit = mc_x_new_unit[in_bounds_mc]
        new_mc = {'x': mc_x_new_unit, 'method': self.mc_samples.get('method', 'unknown')}
        for key in ('weights', 'logl', 'logp'):
            if key in self.mc_samples:
                new_mc[key] = np.array(self.mc_samples[key])[in_bounds_mc]
        if 'best' in self.mc_samples:
            new_mc['best'] = self.mc_samples['best']
        self.mc_samples = new_mc

        # 6. Warm-start GP refit in the new coordinates
        log.info(
            f"[Flow update {self.flow_update_count + 1}] Warm-start refitting GP "
            f"with {x_ib.shape[0]} points (4 restarts, 500 iters)"
        )
        self.results_manager.start_timing('GP Training')
        self.pool.gp_fit(self.gp, n_restarts=4, maxiters=500, rng=self.np_rng, use_pool=True)
        self.results_manager.end_timing('GP Training')

        # 7. Persist
        self._save_transform()   # saves both .npz + _flow_model.pkl
        self._save_dropped_pool()
        self.gp.save(filename=f"{self.save_path}_gp")
        self.flow_update_count += 1
        log.info(
            f"[Flow update] Complete. Count={self.flow_update_count}. "
            f"Transform: {self.transform}"
        )
        self.results_manager.gp_info.update({
            'flow_update_count': self.flow_update_count,
            'last_flow_update_ii': step,
        })
        return True

    def finalise_results(self):
        # here finalize results
        if not self.is_main:
            return
        
        # Prepare return dictionary

        # Extract GP and classifier information
        gp_info = {
            'gp_training_set_size': self.gp.train_x.shape[0],
            'gp_final_best_loglike': float(self.best_f),  # Best value in true physical space
            'total_objective_evals': int(self.total_objective_evals),
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

        # if logz_dict is empty, warn user
        if not logz_dict:
            log.warning("No logz information found, nested sampling has not been run yet.")

        # Finalize results with comprehensive data
        self.results_manager.finalize(
            samples_dict=samples_dict,
            logz_dict=logz_dict,
            converged=self.converged,
            termination_reason=self.termination_reason,
            gp_info=gp_info,
            best_point=self.best_pt,
            best_loglike=self.best_f,
            best_iteration=self.best_pt_iteration
        )

        # ---- Final diagnostic: train a flow on the converged HMC samples ----
        flow_samples_phys = None
        try:
            hmc = getattr(self, 'last_hmc_mc_samples', None)
            if hmc and 'x' in hmc:
                mc_x_unit = np.array(hmc['x'])
                mc_x_phys = self.transform.from_unit(mc_x_unit)
                N_mc = mc_x_phys.shape[0]
                log.info(f"[Final flow] Training final diagnostic flow on {N_mc} HMC samples …")
                final_flow = FlowTransform(param_bounds=self.loglikelihood.param_bounds)
                final_flow.train_flow(
                    mc_x_phys,
                    flow_layers=8,
                    nn_width=64,
                    nn_depth=2,
                    learning_rate=5e-4,
                    max_epochs=500,
                    batch_size=min(512, N_mc),
                    max_patience=40,
                )
                n_draw = 10_000
                jkey = jax.random.key(1)
                flow_samples_std = np.asarray(final_flow._flow.sample(jkey, (n_draw,)))
                # Unstandardise: the flow was trained on (θ - μ) / σ
                flow_samples_phys = flow_samples_std * final_flow._train_std + final_flow._train_mean
                flow_samples_phys = np.clip(
                    flow_samples_phys,
                    self.loglikelihood.param_bounds[0],
                    self.loglikelihood.param_bounds[1],
                )
                log.info(f"[Final flow] Drew {n_draw} samples from trained flow.")
            else:
                log.warning("[Final flow] No HMC samples available; skipping final flow training.")
        except Exception as e:
            log.warning(f"[Final flow] Final flow training/sampling failed: {e}")

        # Create final results dictionary with only the specified keys
        self.results_dict = {
            'gp': self.gp,
            'likelihood': self.loglikelihood,
            'results_manager': self.results_manager,
            'best_val': self.best_f,
            'best_pt': self.best_pt,
            'logz': logz_dict,
            'termination_reason': self.termination_reason,
            'samples': samples_dict,
            'flow_samples': flow_samples_phys,  # (10000, ndim) array or None
        }

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

    def check_convergence_logz(self, step, logz_dict, equal_samples, equal_logl, verbose=True, save_checkpoint=True):
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
        delta = (logz_dict['upper'] - logz_dict['lower'])/2 
        
        # alternative cross-check using std, not used for convergence
        delta_crosscheck = logz_dict['std']

        converged = delta < self.logz_threshold
        
        # Compute KL divergences if we have nested sampling samples
        successive_kl = None
        
        equal_samples = np.asarray(self.transform.from_unit(equal_samples))
    

        if self.prev_samples is not None:

            prev_samples_x = self.prev_samples['x']
            mu1 = np.mean(prev_samples_x, axis=0)
            cov1 = np.cov(prev_samples_x, rowvar=False)
            mu2 = np.mean(equal_samples, axis=0)
            cov2 = np.cov(equal_samples, rowvar=False)
            successive_kl = kl_divergence_gaussian(mu1, np.atleast_2d(cov1), mu2, np.atleast_2d(cov2))

            log.info(f"Successive KL: symmetric={successive_kl.get('symmetric', 0):.4f}")
            # Store KL divergences if computed
            self.results_manager.update_kl_divergences(
                iteration=step,
                successive_kl=successive_kl
            )

        # Store current samples for next iteration
        self.prev_samples = {'x': equal_samples, 'logl': equal_logl}

        # Update results manager with convergence info and KL divergences.
        # Pass delta explicitly so the stored value matches what was used for the check.
        self.results_manager.update_convergence(
            iteration=step,
            logz_dict=logz_dict,
            converged=converged,
            threshold=self.logz_threshold,
            delta=delta
        )
        
        log.info(f"Convergence check: delta = {delta:.4f}, step = {step}, threshold = {self.logz_threshold}")
        
        if converged:
            self.convergence_counter += 1
            if self.convergence_counter >= self.convergence_n_iters:
                log.info(f"Convergence achieved after {self.convergence_n_iters} successive iterations")
                converged = True
            else:
                log.info(f"Convergence iteration {self.convergence_counter}/{self.convergence_n_iters}")
                converged = False
        else:
            self.convergence_counter = 0  # Reset counter if not converged
            converged = False

        # Check if this is the smallest delta seen so far and save checkpoint, also ensure delta is reasonably good
        if (delta < self.min_delta_seen) and (delta_crosscheck < 1.0) and save_checkpoint:
            self.min_delta_seen = delta

            # Create checkpoint filename with suffix
            checkpoint_filename = f"{self.output_file}_checkpoint"

            if not converged:

                # Save intermediate results checkpoint
                self.results_manager.save_intermediate(gp=self.gp, filename=f"{checkpoint_filename}")

                # Save getdist chains
                self.results_manager.save_chain_files(samples_dict=self.ns_samples, filename=f"{checkpoint_filename}")

                if verbose:
                    log.info(f"New minimum delta achieved: {delta:.4f}")
                    log.info("Saving checkpoint results for new minimum delta")
                    log.info(f"Saved GP checkpoint to {checkpoint_filename}_gp.npz")
                    log.info(f"Saved intermediate results checkpoint to {checkpoint_filename}.json")

        return converged
        
    # ============================================================================
    # MAIN RUN METHODS
    # ============================================================================

    def run(self, acq: Union[str, Tuple[str]] = 'wipstd',
            min_evals: Optional[int] = None,
            max_evals: Optional[int] = None,
            max_gp_size: Optional[int] = None,
            logz_threshold: Optional[float] = None,
            convergence_n_iters: int = 1,
            ei_goal: float = 1e-10,
            do_final_ns: bool = False,
            fit_n_points: Optional[int] = None,
            batch_size: Optional[int] = None,
            ns_n_points: Optional[int] = None,
            num_hmc_warmup: Optional[int] = None,
            num_hmc_samples: Optional[int] = None,
            mc_points_size: Optional[int] = None,
            thinning: int = 4,
            num_chains: Optional[int] = None,
            mc_points_method: str = 'NUTS',
            zeta_ei: float = 0.0,
            rotation_update_step: Optional[int] = None,
            rotation_update_min_evals: Optional[int] = None,
            max_rotation_updates: int = 10,
            rotation_logz_threshold: float = 4.0,
            rotation_kl_threshold: float = 1.0,
            flow_update_step: Optional[int] = None,
            flow_update_min_evals: Optional[int] = None,
            max_flow_updates: int = 5,
            flow_logz_threshold: Optional[float] = None,
            flow_kwargs: Optional[Dict[str, Any]] = None,
            ):
        """
        Run the Bayesian Optimization loop.
        
        Parameters
        ----------
        acq : str or tuple of str
            Acquisition function(s) to use: 'WIPV', 'EI', 'LogEI', 'WIPStd'.
        min_evals : int, optional
            Minimum number of likelihood evaluations before checking convergence.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        max_evals : int, optional
            Maximum number of likelihood evaluations.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        max_gp_size : int, optional
            Maximum number of points used to train the GP.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        logz_threshold : float, optional
            Convergence threshold for log evidence change (WIPV/WIPStd). 
            If None, uses dimension-based default from _get_dimension_based_defaults().
        convergence_n_iters : int, optional
            Number of successive iterations meeting threshold for convergence. Default is 1.
        ei_goal : float, optional
            Goal value for EI/LogEI acquisition convergence. Default is 1e-10.
        do_final_ns : bool, optional
            Whether to run final nested sampling at convergence (WIPV/WIPStd). Default is False.
        fit_n_points : int, optional
            Refit GP hyperparameters after adding this many new points to the GP.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        batch_size : int, optional
            Batch size for WIPV/WIPStd acquisition.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        ns_n_points : int, optional
            Run nested sampling after adding this many new points to the GP (for WIPV/WIPStd).
            If None, uses dimension-based default from _get_dimension_based_defaults().
        num_hmc_warmup : int, optional
            Number of HMC warmup steps.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        num_hmc_samples : int, optional
            Number of HMC samples to draw.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        mc_points_size : int, optional
            Number of MC points for WIPV acquisition.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        thinning : int, optional
            Thinning factor for MC samples. Default is 4.
        num_chains : int, optional
            Number of parallel HMC chains.
            If None, uses dimension-based default from _get_dimension_based_defaults().
        mc_points_method : str, optional
            Method for generating MC points: 'NUTS', 'NS', or 'uniform'. Default is 'NUTS'.
        zeta_ei : float, optional
            Exploration parameter for EI acquisition. Default is 0.0.
        rotation_update_step : int, optional
            If set, attempt to update the rotation matrix every this many BO iterations.
            Works both when an initial ``rotation_matrix`` was provided at construction
            time (incremental update) and when no initial rotation was given at all
            (the first call establishes the rotation from scratch using the current
            MC sample covariance, switching the transform from linear scaling to a
            rotated eigenspace).  If None (default), no rotation updates are performed.
        rotation_update_min_evals : int, optional
            Minimum number of likelihood evaluations before the first rotation update.
            Defaults to ``min_evals`` when not set.
        max_rotation_updates : int, optional
            Maximum number of rotation updates allowed during a run. Default is 10.
        rotation_kl_threshold : float, optional
            Minimum symmetric KL divergence between the old and new physical-space
            Gaussians required to trigger a rotation update.  Updates are skipped
            when the KL is below this value (i.e. the new estimate is not
            meaningfully different from the current one). Default is 1.0.
            
        Returns
        -------
        dict
            Results dictionary containing samples, GP, likelihood, and convergence information. Keys include:
        """
        # Workers don't run the optimization loop
        if not self.is_main:
            return None
        
        # Get dimension-based defaults
        dim_defaults = get_dimension_based_defaults(self.ndim)
        
        # Apply defaults for None values
        min_evals = min_evals if min_evals is not None else dim_defaults['min_evals']
        max_evals = max_evals if max_evals is not None else dim_defaults['max_evals']
        max_gp_size = max_gp_size if max_gp_size is not None else dim_defaults['max_gp_size']
        fit_n_points = fit_n_points if fit_n_points is not None else dim_defaults['fit_n_points']
        batch_size = batch_size if batch_size is not None else dim_defaults['batch_size']
        ns_n_points = ns_n_points if ns_n_points is not None else dim_defaults['ns_n_points']
        num_hmc_warmup = num_hmc_warmup if num_hmc_warmup is not None else dim_defaults['num_hmc_warmup']
        num_hmc_samples = num_hmc_samples if num_hmc_samples is not None else dim_defaults['num_hmc_samples']
        mc_points_size = mc_points_size if mc_points_size is not None else dim_defaults['mc_points_size']
        num_chains = num_chains if num_chains is not None else dim_defaults['num_chains']
        logz_threshold = logz_threshold if logz_threshold is not None else dim_defaults['logz_threshold']
        rotation_logz_threshold = rotation_logz_threshold if rotation_logz_threshold is not None else dim_defaults['rotation_logz_threshold']
        
        # Store convergence parameters
        self.min_evals = min_evals
        self.max_evals = max_evals
        self.max_gp_size = max_gp_size
        self.logz_threshold = logz_threshold

        # Check if already converged with new threshold when resuming.
        # Skip only if the new threshold is NOT stricter than the one used previously
        # (i.e. the user hasn't tightened the criterion).  Comparing new vs old threshold
        # is the correct guard; comparing delta vs new threshold would incorrectly skip
        # even when the user explicitly lowers logz_threshold to force more iterations.
        if self.prev_converged and self.prev_convergence_threshold is not None:
            if logz_threshold >= self.prev_convergence_threshold:
                log.info(f"Previous run converged with threshold={self.prev_convergence_threshold:.6f}; "
                         f"new threshold={logz_threshold:.6f} is not stricter — skipping BO loop.")
                self.converged = True
                self.termination_reason = "Already converged in previous run"
                
                # Restore samples and logz from previous run
                if self.results_manager.convergence_history:
                    last_conv = self.results_manager.convergence_history[-1]
                    self.results_dict['logz'] = last_conv.logz_dict.copy()
                
                # Restore samples from results_manager if available
                if self.results_manager.final_samples is not None and len(self.results_manager.final_samples) > 0:
                    self.samples_dict = {
                        'x': self.results_manager.final_samples,
                        'weights': self.results_manager.final_weights,
                        'logl': self.results_manager.final_loglikes
                    }
                    log.info(f"Restored {len(self.samples_dict['x'])} samples from previous run")
                else:
                    self.samples_dict = {}
                    log.warning("No samples found in previous run")
                
                self.finalise_results()
                self.pool.close()
                return self.results_dict
            else:
                log.info(f"Previous run converged with threshold={self.prev_convergence_threshold:.6f}; "
                         f"new threshold={logz_threshold:.6f} is stricter — continuing optimization.")
                self.converged = False
                self.convergence_counter = 0

        # Log run settings
        log.info("Using run settings:")
        log.info(f"min_evals = {min_evals}, max_evals = {max_evals}, max_gp_size = {max_gp_size}")
        if acq.lower() in ['wipv', 'wipstd']:
            acq_info = "logz_threshold = {:.4f}".format(logz_threshold)+f", mc_points_size = {mc_points_size}"
        else:
            acq_info = "ei_goal = {:.4e}".format(ei_goal)
        log.info(f"convergence_n_iters = {convergence_n_iters}, acq = {acq}, {acq_info}")
        log.info(f"fit_n_points = {fit_n_points}, batch_size = {batch_size}, ns_n_points = {ns_n_points}")
        
        # Initialize result containers
        self.samples_dict = {}
        self.results_dict = {}
        
        self.convergence_n_iters = convergence_n_iters
        self.ei_goal_log = np.log(ei_goal)
        self.do_final_ns = do_final_ns
        
        # Store run settings
        self.fit_n_points = fit_n_points
        self.ns_n_points = ns_n_points
        self.batch_size = batch_size
        
        # Initialize point counters for triggering GP refit and NS
        self.n_points_since_last_fit = 0
        self.n_points_since_last_ns = 0
        self.num_hmc_warmup = num_hmc_warmup
        self.num_hmc_samples = num_hmc_samples
        self.mc_points_size = mc_points_size
        self.hmc_thinning = thinning
        self.hmc_num_chains = num_chains
        self.mc_points_method = mc_points_method
        self.zeta_ei = zeta_ei

        # Rotation update settings (only active when transform uses rotation)
        self.rotation_update_step      = rotation_update_step
        self.rotation_update_min_evals = rotation_update_min_evals if rotation_update_min_evals is not None else min_evals
        self.max_rotation_updates      = max_rotation_updates
        self.rotation_kl_threshold     = rotation_kl_threshold
        self.rotation_logz_threshold = rotation_logz_threshold
        # Restore rotation state from a previous run if available, otherwise start fresh
        _saved_rotation = self.results_manager.gp_info
        self.rotation_update_count = int(_saved_rotation.get('rotation_update_count', 0))
        self.last_rotation_ii      = _saved_rotation.get('last_rotation_ii', None)
        self.last_rotation_acq_val = _saved_rotation.get('last_rotation_acq_val', None)
        if self.rotation_update_count > 0:
            log.info(f"Resuming with rotation state: count={self.rotation_update_count}, "
                     f"last_ii={self.last_rotation_ii}, last_acq_val={self.last_rotation_acq_val}")

        # Flow update settings (only active when transform is a FlowTransform)
        self.flow_update_step      = flow_update_step
        self.flow_update_min_evals = flow_update_min_evals if flow_update_min_evals is not None else min_evals
        self.max_flow_updates      = max_flow_updates
        # Use rotation_logz_threshold as default for flow if not specified
        self.flow_logz_threshold   = flow_logz_threshold if flow_logz_threshold is not None else rotation_logz_threshold
        self.flow_kwargs           = flow_kwargs or {}
        _saved_flow = self.results_manager.gp_info
        self.flow_update_count     = int(_saved_flow.get('flow_update_count', 0))
        self.last_flow_update_ii   = _saved_flow.get('last_flow_update_ii', None)
        if self.flow_update_count > 0:
            log.info(f"Resuming with flow state: count={self.flow_update_count}, "
                     f"last_ii={self.last_flow_update_ii}")

        # Load (or initialise) the persistent pool of dropped training points.
        # Points dropped from the GP/clf training set during a rotation update are stored
        # here and reconsidered at the next rotation update.
        # Only restore a saved pool when genuinely resuming — on a fresh start the pool
        # must be empty so stale files from previous runs with the same name cannot
        # inject phantom points that inflate pool size beyond total_objective_evals.
        if not self.fresh_start:
            self._load_dropped_pool()
        else:
            self._dropped_pool_x_phys = np.zeros((0, self.ndim))
            self._dropped_pool_y_raw  = np.zeros(0)
            log.debug("Fresh start: dropped pool initialised empty.")

        # Adjust batch_size for MPI load balancing
        if self.is_mpi:
            n_processes = self.pool.size
            original_batch = self.batch_size
            if self.batch_size % n_processes != 0:
                self.batch_size = (self.batch_size // n_processes) * n_processes
                if self.batch_size < n_processes:
                    self.batch_size = n_processes
                log.info(f"Adjusted batch_size from {original_batch} to {self.batch_size} "
                        f"(multiple of {n_processes} processes)")
        
        # Initialize convergence state
        self.converged = False
        self.prev_converged = False
        self.convergence_counter = 0
        self.min_delta_seen = np.inf
        self.termination_reason = "Max evaluation budget reached"
        
        # Update results manager settings with all run parameters
        self.results_manager.settings.update({
            'acq': acq,
            'min_evals': min_evals,
            'max_evals': max_evals,
            'max_gp_size': max_gp_size,
            'logz_threshold': logz_threshold,
            'convergence_n_iters': convergence_n_iters,
            'ei_goal': ei_goal,
            'do_final_ns': do_final_ns,
            'fit_n_points': fit_n_points,
            'batch_size': batch_size,
            'ns_n_points': ns_n_points,
            'num_hmc_warmup': num_hmc_warmup,
            'num_hmc_samples': num_hmc_samples,
            'mc_points_size': mc_points_size,
            'thinning': thinning,
            'num_chains': num_chains,
            'mc_points_method': mc_points_method,
            'zeta_ei': zeta_ei,
            'rotation_update_step': rotation_update_step,
            'rotation_update_min_evals': rotation_update_min_evals,
            'rotation_logz_threshold': rotation_logz_threshold,
            'max_rotation_updates': max_rotation_updates,
            'rotation_kl_threshold': rotation_kl_threshold,
            'flow_update_step': flow_update_step,
            'flow_update_min_evals': flow_update_min_evals,
            'max_flow_updates': max_flow_updates,
            'flow_logz_threshold': self.flow_logz_threshold,
        })
        
        acqs_funcs_available = list(_acq_funcs.keys())

        self.samples_dict = {}
        self.results_dict = {}

        if isinstance(acq, str):
            acqs = [acq]

        self.current_iteration = self.start_iteration

        for x in acqs:
            if x.lower() not in acqs_funcs_available:
                raise ValueError(f"Invalid acquisition function '{x}'. Valid options are: {acqs_funcs_available}")
            self.acquisition = _acq_funcs[x.lower()](optimizer=self.optimizer)  # Set acquisition function
            if x.lower() == 'wipv':
                self.run_WIPV(ii=self.current_iteration)
            elif x.lower() == 'wipstd':
                self.run_WIPStd(ii=self.current_iteration)
            else:
                self.run_EI(ii=self.current_iteration)

        log.info(f"Final best point {self.best} with value = {self.best_f:.6f}, found at iteration {self.best_pt_iteration}")


        #-------End of BO loop-------
        log.info(f"Sampling stopped: {self.termination_reason}")
        log.info(f"Final GP training set size: {self.gp.train_x.shape[0]}, max size: {self.max_gp_size}")

        self.finalise_results()
        
        # Close the pool and signal workers to exit
        self.pool.close()

        return self.results_dict

    def run_EI(self, ii = 0, ):
        """
        Run the optimization loop for EI/LogEI acquisition functions.
        """
        if not self.is_main:
            return
        
        current_evals = self.total_objective_evals
        log.info(f"Starting iteration {ii}")
        converged=False

        while not converged:
            ii += 1
            verbose = True

            if verbose:
                log.info(f"Iteration {ii} of {self.acquisition.name}, objective evals {current_evals}/{self.max_evals}")

            acq_kwargs = {'zeta': self.zeta_ei, 'best_y': max(self.gp.train_y.flatten()) if self.gp.train_y.size > 0 else 0.}
            n_batch = 1
            new_pts_u, acq_vals = self.get_next_batch(acq_kwargs, n_batch = n_batch, n_restarts = 50, maxiter = 1000, early_stop_patience = 50, step = ii, verbose=verbose)
            new_pts_u = jnp.atleast_2d(new_pts_u)

            new_vals = self.evaluate_likelihood(new_pts_u, ii, verbose=verbose)
            current_evals = self.total_objective_evals

            self.update_gp(new_pts_u, new_vals, step = ii, verbose=verbose)

            self.results_manager.update_best_loglike(ii, self.best_f)
            if verbose:
                log.info(f"Current best point {self.best} with value = {self.best_f:.6f}, found at iteration {self.best_pt_iteration}")

            # if current_evals >= self.min_evals:
            converged = self.check_convergence_ei(ii,acq_vals)

            # Update results manager with iteration info, also save results and gp if save_step
            if ii % self.save_step == 0:
                self.results_manager.save_intermediate(gp=self.gp)

            if converged:
                self.termination_reason = f"{self.acquisition.name.upper()} goal reached"
                self.results_dict['termination_reason'] = self.termination_reason
                break
            self.pool.clear_jax_caches()

            max_evals_or_gpsize_reached = self.check_max_evals_and_gpsize(current_evals)
            if max_evals_or_gpsize_reached:
                break

        # End EI
        self.current_iteration = ii

    def run_weighted_integrated_posterior(self, acq_func_class, ii=0):
        """
        Run the optimization loop for Weighted Integrated Posterior acquisition functions (WIPV or WIPStd).
        
        Parameters
        ----------
        acq_func_class : class
            The acquisition function class to use (WIPV or WIPStd).
        ii : int, optional
            Starting iteration number. Default is 0.
        """
        if not self.is_main:
            return
        
        # Set acquisition function
        self.acquisition = acq_func_class(optimizer=self.optimizer)
        acq_name = self.acquisition.name
        
        current_evals = self.total_objective_evals
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
        self.last_hmc_mc_samples = self.mc_samples   # keep HMC-only copy for final flow
        self.results_manager.end_timing('MCMC Sampling')
        self.ns_samples = None

        #logz keys to print
        logz_keys = ['mean', 'upper', 'lower', 'dlogz_sampler']


        while not self.converged:
            ii += 1
            # Check if we should run nested sampling based on points added
            self.n_points_since_last_ns += self.batch_size
            ns_flag = (self.n_points_since_last_ns >= self.ns_n_points) and current_evals >= self.min_evals
            verbose = True

            if verbose:
                log.info(f"Iteration {ii} of {acq_name}, objective evals {current_evals}/{self.max_evals}")

            acq_kwargs = {'mc_samples': self.mc_samples, 'mc_points_size': self.mc_points_size}
            new_pts_u, acq_vals = self.get_next_batch(acq_kwargs, n_batch = self.batch_size, n_restarts = 1, maxiter = 100, early_stop_patience = 10, step = ii, verbose=verbose)
            new_pts_u = jnp.atleast_2d(new_pts_u)
            new_vals = self.evaluate_likelihood(new_pts_u, ii, verbose=verbose)
            current_evals = self.total_objective_evals

            self.update_gp(new_pts_u, new_vals, step = ii)
            self.results_manager.update_best_loglike(ii, self.best_f)

            # Check convergence and update MCMC samples
            if ns_flag and (acq_vals[-1] <= self.logz_threshold):
                self.results_manager.start_timing('Nested Sampling')
                ns_samples, logz_dict, ns_success = nested_sampling_Dy(mode='convergence',
                    gp=self.gp, ndim=self.ndim, maxcall=int(5e6), dynamic=False, dlogz=0.01, equal_weights=False,
                    rng=self.np_rng
                )
                self.results_manager.end_timing('Nested Sampling')

                logz_str = ", ".join([f"{k}={logz_dict[k]:.4f}" for k in logz_keys if k in logz_dict])
                log.info(f"NS success = {ns_success}, LogZ info: {logz_str}")

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
                    self.results_dict['logz'] = logz_dict
                    self.converged = self.check_convergence_logz(ii, logz_dict, equal_samples, equal_logl)
                    if self.converged:
                        self.termination_reason = "LogZ converged"
                        self.results_dict['termination_reason'] = self.termination_reason
                
                # Reset counter after running NS
                self.n_points_since_last_ns = 0
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
                self.last_hmc_mc_samples = self.mc_samples   # keep HMC-only copy for final flow
                self.results_manager.end_timing('MCMC Sampling')
            
            if verbose:
                log.info(f"Current best point {self.best} with value = {self.best_f:.6f}, found at iteration {self.best_pt_iteration}")

            # Update results manager with iteration info, also save results and gp if save_step
            if ii % self.save_step == 0:
                self.results_manager.save_intermediate(gp=self.gp)

            # Periodic rotation update — fires with or without an initial rotation matrix.
            # First update: no step-count check; fires as soon as the acq threshold is met.
            # Subsequent updates: require acq_val to have improved since the last update
            #   AND at least rotation_update_step iterations to have elapsed.
            if (self.rotation_update_step is not None
                    and self.rotation_update_count < self.max_rotation_updates
                    and current_evals >= self.rotation_update_min_evals
                    and acq_vals[-1] <= self.rotation_logz_threshold):
                if self.rotation_update_count == 0:
                    # First rotation update — fire immediately upon reaching the threshold
                    did_update = self._update_rotation(step=ii)
                    if did_update:
                        self.last_rotation_ii      = ii
                        self.last_rotation_acq_val = float(acq_vals[-1])
                        self.results_manager.gp_info.update({
                            'last_rotation_ii': self.last_rotation_ii,
                            'last_rotation_acq_val': self.last_rotation_acq_val,
                        })
                elif (acq_vals[-1] < self.last_rotation_acq_val
                        and (ii - self.last_rotation_ii) >= self.rotation_update_step):
                    # Subsequent rotation: acq must have improved and enough steps elapsed
                    did_update = self._update_rotation(step=ii)
                    if did_update:
                        self.last_rotation_ii      = ii
                        self.last_rotation_acq_val = float(acq_vals[-1])
                        self.results_manager.gp_info.update({
                            'last_rotation_ii': self.last_rotation_ii,
                            'last_rotation_acq_val': self.last_rotation_acq_val,
                        })

            # Periodic flow transform update — only fires when transform is a FlowTransform.
            # The first update fires as soon as both the min-evals and the acq threshold are met.
            # Subsequent updates require at least flow_update_step iterations to have elapsed.
            if (self.flow_update_step is not None
                    and isinstance(self.transform, FlowTransform)
                    and self.flow_update_count < self.max_flow_updates
                    and current_evals >= self.flow_update_min_evals
                    and acq_vals[-1] <= self.flow_logz_threshold):
                last_ii = self.last_flow_update_ii
                if (last_ii is None) or ((ii - last_ii) >= self.flow_update_step):
                    did_update = self._update_flow_transform(step=ii, flow_kwargs=self.flow_kwargs)
                    if did_update:
                        self.last_flow_update_ii = ii

            if self.converged:
                break
            
            self.pool.clear_jax_caches()

            max_evals_or_gpsize_reached = self.check_max_evals_and_gpsize(current_evals)
            if max_evals_or_gpsize_reached:
                break

        # End of main BO loop
        self.current_iteration = ii

        # Final nested sampling if not yet converged and do_final_ns is True
        if self.do_final_ns and not self.converged:
            
            self.results_manager.start_timing('GP Training')
            self.pool.gp_fit(self.gp, n_restarts=4, maxiters=500, rng=self.np_rng, use_pool=True)
            self.results_manager.end_timing('GP Training')

            log.info("Final Nested Sampling")
            self.results_manager.start_timing('Nested Sampling')
            self.ns_samples, logz_dict, ns_success = nested_sampling_Dy(mode='convergence',
                gp=self.gp, ndim=self.ndim, maxcall=int(5e6), dynamic=True, dlogz=0.01, rng=self.np_rng
            )
            self.results_manager.end_timing('Nested Sampling')
            logz_str = ", ".join([f"{k}={logz_dict[k]:.4f}" for k in logz_keys if k in logz_dict])
            log.info(f"Final LogZ: {logz_str}")
            if ns_success:
                equal_samples, equal_logl = resample_equal(self.ns_samples['x'], self.ns_samples['logl'], weights=self.ns_samples['weights'])
                log.info(f"Using nested sampling results")
                self.check_convergence_logz(ii+1, logz_dict, equal_samples, equal_logl, save_checkpoint=False)
                self.results_dict['logz'] = logz_dict
                if self.converged:
                    self.termination_reason = "LogZ converged"
                    self.results_dict['termination_reason'] = self.termination_reason

        if (self.ns_samples is not None) and ns_success:
            samples = self.ns_samples['x']
            weights = self.ns_samples['weights']
            loglikes = self.ns_samples['logl']
        else:
            log.info("No nested sampling results found or nested sampling unsuccessful, MC samples from HMC/MCMC will be used instead.")
            self.results_manager.start_timing('MCMC Sampling')
            mc_samples = get_mc_samples(
                    self.gp, warmup_steps=self.num_hmc_warmup, num_samples=8*self.num_hmc_samples,
                    thinning=self.hmc_thinning, method="NUTS")
            self.results_manager.end_timing('MCMC Sampling')
            samples = mc_samples['x']
            weights = mc_samples['weights'] if 'weights' in mc_samples else np.ones(mc_samples['x'].shape[0])
            loglikes = mc_samples['logp']
                
        samples = np.asarray(self.transform.from_unit(samples))

        self.samples_dict = {
            'x': samples,
            'weights': weights,
            'logl': loglikes
        }

    def run_WIPStd(self, ii=0):
        """Run optimization loop for WIPStd acquisition function."""
        return self.run_weighted_integrated_posterior(WIPStd, ii)

    def run_WIPV(self, ii=0):
        """Run optimization loop for WIPV acquisition function."""
        return self.run_weighted_integrated_posterior(WIPV, ii)