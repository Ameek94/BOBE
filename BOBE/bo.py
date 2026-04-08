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
from .utils.core import scale_from_unit, scale_to_unit, kl_divergence_gaussian, get_threshold_for_nsigma
from .transforms import ParameterTransform, BaseTransform, IdentityTransform, RotationTransform, NormalisingFlowTransform, load_transform
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
        'transform_acq_threshold': min(4, 2 * (ndim/15))
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
                 save_step=1,
                 optimizer='scipy',
                 use_clf=False,
                 clf_type = "svm",
                 clf_nsigma_threshold=20,
                 clf_use_size = 10,
                 clf_update_step=1,
                 minus_inf=-1e10,
                 seed: Optional[int] = None,
                 verbosity: str = 'INFO',
                 transform: Optional[BaseTransform] = None,
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
        transform : BaseTransform instance, subclass, (subclass, kwargs), or None
            Parameter-space transform.  Three forms accepted:

            * ``None`` (default) — ``IdentityTransform`` (no transform).
            * A ``BaseTransform`` *subclass* — instantiated with ``param_bounds``
              from the likelihood; settings use the class defaults.
            * A ``(subclass, kwargs_dict)`` *tuple* — instantiated with
              ``param_bounds`` + ``**kwargs``.  This is the recommended form
              because it keeps all transform settings in one place without
              requiring ``param_bounds`` at call-site::

                  transform=(RotationTransform, {'kl_threshold': 0.5,
                                                 'max_updates': 5,
                                                 'update_step': 25})

                  transform=(NormalisingFlowTransform, {'kl_threshold': 1.0,
                                                        'max_updates': 3,
                                                        'update_step': 20})

            * A pre-built ``BaseTransform`` *instance* — used as-is (bounds
              must already be set).
            
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
        if self.is_mpi:
            self.pool.comm.Barrier()  # Ensure all processes start together
        
        # Convert to Likelihood instance and store for all processes
        self.loglikelihood = self._prepare_likelihood(
            loglikelihood, param_list, param_bounds, param_labels,
            likelihood_name, confidence_for_unbounded, minus_inf
        )
        self.ndim = len(self.loglikelihood.param_list)
        
        # Create the parameter transform (handles unit-cube scaling and optional rotation).
        # Accepted forms for `transform`:
        #   - BaseTransform instance  → used as-is
        #   - BaseTransform subclass  → instantiated with param_bounds only
        #   - (subclass, kwargs_dict) → instantiated with param_bounds + **kwargs
        #     e.g. (RotationTransform, {'kl_threshold': 0.5, 'update_step': 25})
        # This lets users configure the transform in one place without needing to
        # know param_bounds up front (they are resolved from the likelihood).
        if transform is None:
            self.transform = IdentityTransform(self.loglikelihood.param_bounds)
        elif isinstance(transform, tuple) and len(transform) == 2 and isinstance(transform[0], type):
            cls, kwargs = transform
            self.transform = cls(self.loglikelihood.param_bounds, **kwargs)
        elif isinstance(transform, type):
            self.transform = transform(self.loglikelihood.param_bounds)
        else:
            self.transform = transform
        
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
        
        # Save initial checkpoint
        self._save_checkpoint()
        
        # Initialize for KL divergence tracking
        self.prev_samples = None

        # Objective evaluation counter - restored from checkpoint on resume, or seeded
        # from GP size on fresh start.
        # On resume, _handle_resume already set this from the checkpoint's top-level key.
        # Only fall back to gp_info/gp.npoints for fresh starts to avoid overwriting the
        # correctly restored value with the stale one stored inside gp_info (which reflects
        # the last *completed* run, not the current run state).
        if not hasattr(self, 'total_objective_evals'):
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

        # Physical training points (always stored in BOBE, independent of GP transform)
        self.all_train_x_phys = np.empty((0, self.ndim))
        self.all_train_y_raw = np.empty(0)
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
        )
        
        self.fresh_start = not resume
    
    def _save_checkpoint(self):
        """Delegate checkpoint serialisation to the results manager."""
        if not self.is_main or not self.save:
            return
        self.results_manager.save_checkpoint({
            'gp_state':              self.gp.state_dict(),
            'use_clf':               isinstance(self.gp, GPwithClassifier),
            'transform_state':       self.transform.state_dict(),
            'all_train_x_phys':      self.all_train_x_phys,
            'all_train_y_raw':       self.all_train_y_raw,
            'total_objective_evals': getattr(self, 'total_objective_evals', 0),
            'start_iteration':       getattr(self, 'current_iteration', getattr(self, 'start_iteration', 0)),
            'best_f':                getattr(self, 'best_f', float('-inf')),
            'best_pt':               getattr(self, 'best_pt', None),
            'best_pt_iteration':     getattr(self, 'best_pt_iteration', 0),
            'convergence_counter':   getattr(self, 'convergence_counter', 0),
            'converged':             getattr(self, 'converged', False),
            'termination_reason':    getattr(self, 'termination_reason', ''),
            'prev_samples':          getattr(self, 'prev_samples', None),
        })
        # Persist flow weights separately (not serialisable in the pkl dict)
        if isinstance(self.transform, NormalisingFlowTransform) and self.transform._use_flow:
            self.transform.save(self.save_path)

    def _handle_resume(self, resume_file, use_clf):
        """Load a checkpoint via the results manager and restore all BO state."""
        data = self.results_manager.load_checkpoint(resume_file)
        if data is None:
            self.fresh_start = True
            return

        # Restore GP
        self.gp = load_gp_statedict(data['gp_state'], data.get('use_clf', use_clf))
        _ = self.gp.predict_mean_single(self.gp.train_x[0])
        log.info(f"Loaded GP with {self.gp.train_x.shape[0]} training points")

        # Restore transform
        ts = data.get('transform_state', {})
        type_key = ts.get('type', IdentityTransform._TYPE_KEY)
        if type_key == RotationTransform._TYPE_KEY:
            # override with new settings if given in BOBE init
            self.transform = RotationTransform.from_state_dict(ts)
        elif type_key == NormalisingFlowTransform._TYPE_KEY:
            self.transform = NormalisingFlowTransform.from_state_dict(ts)
            # Reload trained flow weights if they were saved alongside the checkpoint
            flow_pkl = resume_file + '_flow.pkl'
            if self.transform._use_flow and os.path.exists(flow_pkl):
                import pickle as _pickle
                from .flow import NormalisingFlow
                with open(flow_pkl, 'rb') as _f:
                    _flow_state = _pickle.load(_f)
                self.transform._flow = NormalisingFlow.from_state_dict(_flow_state)
                self.transform._build_to_u_fn()
                log.info(f"Loaded flow weights from {flow_pkl}")
        else:
            self.transform = IdentityTransform.from_state_dict(ts)
        log.info(f"Restored transform: {self.transform!r}")

        # Restore physical training data
        self.all_train_x_phys = np.array(data.get('all_train_x_phys', np.empty((0, self.ndim))))
        self.all_train_y_raw  = np.array(data.get('all_train_y_raw',  np.empty(0)))
        log.info(f"Restored {len(self.all_train_x_phys)} physical training points")

        # Restore run counters
        self.start_iteration       = int(data.get('start_iteration', 0))
        self.best_f                = float(data.get('best_f', float('-inf')))
        self.best_pt               = data.get('best_pt', None)
        self.best_pt_iteration     = int(data.get('best_pt_iteration', 0))
        self.convergence_counter   = int(data.get('convergence_counter', 0))
        self.converged             = bool(data.get('converged', False))
        self.termination_reason    = str(data.get('termination_reason', ''))
        self.prev_samples          = data.get('prev_samples', None)
        self.total_objective_evals = int(data.get('total_objective_evals', self.gp.npoints))

        # Restore convergence state
        cs = self.results_manager.get_last_convergence_state()
        self.prev_converged             = cs['converged']
        self.prev_convergence_delta     = cs['delta']
        self.prev_convergence_threshold = cs['threshold']

        log.info(f"Resuming from iteration {self.start_iteration}, best logL={self.best_f:.4f}")
        self.fresh_start = False
    
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

        # Store all evaluated physical points and loglikelihood values
        self.all_train_x_phys = init_points.copy()
        self.all_train_y_raw = init_vals.flatten().copy()

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
            log.info(f"Evaluating {n_cobaya_init} Cobaya initial points")
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
        log.info(f"Added {actual_points_added} new points to GP (size before: {gp_size_before}, size after: {gp_size_after})")
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

        # Accumulate all evaluated physical training points
        self.all_train_x_phys = np.vstack([self.all_train_x_phys, new_pts])
        self.all_train_y_raw = np.concatenate([self.all_train_y_raw, np.array(new_vals).flatten()])

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

        # Add rotation state from the transform object
        if hasattr(self.transform, 'update_count'):
            gp_info.update({
                'rotation_update_count': self.transform.update_count,
                'last_rotation_ii': self.transform.last_update_ii,
                'last_rotation_acq_val': self.transform.last_update_acq_val,
            })
        else:
            gp_info.update({
                'rotation_update_count': 0,
                'last_rotation_ii': None,
                'last_rotation_acq_val': None,
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

        # Create final results dictionary with only the specified keys
        self.results_dict = {
            'gp': self.gp,
            'likelihood': self.loglikelihood,
            'results_manager': self.results_manager,
            'best_val': self.best_f,
            'best_pt': self.best_pt,
            'logz': logz_dict,
            'termination_reason': self.termination_reason,
            'samples': samples_dict
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

    def check_convergence_logz(self, step, logz_dict, equal_samples, equal_logl):
        """
        Check convergence after nested sampling using logz_std + KL divergence.

        Convergence requires BOTH:
          1. logz_dict['std'] (GP uncertainty on logZ) < logz_threshold
          2. Symmetric KL between consecutive NS sample distributions < kl_convergence_threshold
        for convergence_n_iters successive calls.
        """
        if not self.is_main:
            return False

        delta = logz_dict['std']   # GP uncertainty = (upper - lower) / 2
        equal_samples = np.asarray(equal_samples)

        kl_sym = np.inf
        if self.prev_samples is not None:
            try:
                prev_x = self.prev_samples['x']
                mu1, cov1 = np.mean(prev_x, axis=0), np.atleast_2d(np.cov(prev_x, rowvar=False))
                mu2, cov2 = np.mean(equal_samples, axis=0), np.atleast_2d(np.cov(equal_samples, rowvar=False))
                kl = kl_divergence_gaussian(mu1, cov1, mu2, cov2)
                kl_sym = float(kl.get('symmetric', np.inf))
                self.results_manager.update_kl_divergences(iteration=step, successive_kl=kl)
            except Exception as e:
                log.warning(f"KL computation failed: {e}")

        self.prev_samples = {'x': equal_samples, 'logl': equal_logl}

        log.info(
            f"Convergence check: logz_std={delta:.4e} < thresh={self.logz_threshold:.4e} = {delta < self.logz_threshold}, "
            f"KL_sym={kl_sym:.4f} (thresh={self.kl_convergence_threshold:.4f})"
        )

        if (delta < self.logz_threshold): # and (kl_sym < self.kl_convergence_threshold):
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0

        self.results_manager.update_convergence(
            iteration=step, logz_dict=logz_dict, converged=False,
            threshold=self.logz_threshold, delta=delta
        )

        converged = self.convergence_counter >= self.convergence_n_iters
        if converged:
            log.info(f"Convergence achieved after {self.convergence_n_iters} successive iterations")
        else:
            log.info(f"Convergence iteration {self.convergence_counter}/{self.convergence_n_iters}")
        return converged

    def _check_convergence(self, acq_val: float) -> bool:
        """
        Check convergence for WIPV/WIPStd based on acquisition value and
        KL divergence between consecutive MC sample distributions.

        Convergence requires BOTH:
          1. acq_val <= self.logz_threshold
          2. Symmetric KL between consecutive physical-space MCMC sample Gaussians
             is below self.kl_convergence_threshold for convergence_n_iters iterations.

        Returns True when both conditions are satisfied for enough consecutive steps.
        """
        if not self.is_main:
            return False

        # Map current mc_samples to physical space once
        mc_x_phys = np.asarray(self.transform.from_unit(np.array(self.mc_samples['x'])))

        if acq_val > self.logz_threshold:
            self.convergence_counter = 0
            self.prev_samples = {'x': mc_x_phys}
            return False

        # acq below threshold — compute KL between consecutive sample distributions
        kl_sym = np.inf
        if self.prev_samples is not None:
            try:
                prev_x_phys = np.asarray(self.prev_samples['x'])
                mu1 = np.mean(prev_x_phys, axis=0)
                cov1 = np.atleast_2d(np.cov(prev_x_phys, rowvar=False))
                mu2 = np.mean(mc_x_phys, axis=0)
                cov2 = np.atleast_2d(np.cov(mc_x_phys, rowvar=False))
                kl = kl_divergence_gaussian(mu1, cov1, mu2, cov2)
                kl_sym = float(kl.get('symmetric', np.inf))
            except Exception as e:
                log.warning(f"KL computation failed: {e}")

        log.info(
            f"Convergence check: acq={acq_val:.4e} < thresh={self.logz_threshold:.4e}, "
            f"KL_sym={kl_sym:.4f} (thresh={self.kl_convergence_threshold:.4f})"
        )

        # Always update prev_samples so next iteration has a "previous"
        self.prev_samples = {'x': mc_x_phys}

        if kl_sym < self.kl_convergence_threshold:
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0

        converged = self.convergence_counter >= self.convergence_n_iters

        self.results_manager.update_convergence(
            iteration=self.current_iteration,
            logz_dict={'mean': acq_val},
            converged=converged,
            threshold=self.logz_threshold,
            delta=acq_val,
        )

        if converged:
            log.info(
                f"Convergence achieved after {self.convergence_n_iters} successive "
                f"iterations (KL={kl_sym:.4f})"
            )
        else:
            log.info(f"Convergence iteration {self.convergence_counter}/{self.convergence_n_iters}")

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
            transform_update_min_evals: Optional[int] = None,
            transform_acq_threshold: Optional[float] = None,
            kl_convergence_threshold: float = 0.1,
            adaptive_batch: bool = False,
            min_batch_size: Optional[int] = None):
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
        transform_update_min_evals : int, optional
            Minimum number of likelihood evaluations before the first transform update
            is attempted. Defaults to ``min_evals`` when not set.
        transform_acq_threshold : float, optional
            Acquisition value below which transform updates are triggered.  Reads from
            ``self.transform.acq_threshold`` when that attribute exists, otherwise falls
            back to the dimension-based default from
            ``get_dimension_based_defaults()['transform_acq_threshold']``.
            Pass explicitly to override both.
        adaptive_batch : bool, optional
            Enable adaptive batch sizing (default False). When True, the effective batch
            size each iteration is determined by how far the current acquisition value is
            from ``logz_threshold``:
              - acq_val >= 10x threshold  ->  min_batch_size
              - acq_val <= 2x  threshold  ->  batch_size
              - between                   ->  linear interpolation
            ``num_hmc_samples`` is also scaled proportionally to the effective batch size.
        min_batch_size : int, optional
            Minimum (starting) batch size when ``adaptive_batch=True``. Default is 1.

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
        min_batch_size = min_batch_size if min_batch_size is not None else 1
        ns_n_points = ns_n_points if ns_n_points is not None else dim_defaults['ns_n_points']
        num_hmc_warmup = num_hmc_warmup if num_hmc_warmup is not None else dim_defaults['num_hmc_warmup']
        num_hmc_samples = num_hmc_samples if num_hmc_samples is not None else dim_defaults['num_hmc_samples']
        mc_points_size = mc_points_size if mc_points_size is not None else dim_defaults['mc_points_size']
        num_chains = num_chains if num_chains is not None else dim_defaults['num_chains']
        logz_threshold = logz_threshold if logz_threshold is not None else dim_defaults['logz_threshold']
        # transform_acq_threshold: explicit arg > transform attribute > dimension default
        if transform_acq_threshold is None:
            transform_acq_threshold = getattr(self.transform, 'acq_threshold', None)
        if transform_acq_threshold is None:
            transform_acq_threshold = dim_defaults['transform_acq_threshold']
        transform_update_min_evals = transform_update_min_evals if transform_update_min_evals is not None else min_evals
        
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
        adaptive_str = f", min_batch_size = {min_batch_size} (adaptive)" if adaptive_batch else ""
        log.info(f"fit_n_points = {fit_n_points}, batch_size = {batch_size}{adaptive_str}, ns_n_points = {ns_n_points}")
        
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
        self.adaptive_batch = adaptive_batch
        self.min_batch_size = min_batch_size

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

        # Transform update scheduling (BO-loop concerns; transform internals live on the object)
        self.transform_update_min_evals = transform_update_min_evals
        self.transform_acq_threshold    = transform_acq_threshold
        self.kl_convergence_threshold   = kl_convergence_threshold

        if hasattr(self.transform, 'update_count') and self.transform.update_count > 0:
            log.info(
                f"Resuming with transform state: count={self.transform.update_count}, "
                f"last_ii={self.transform.last_update_ii}, "
                f"last_acq_val={self.transform.last_update_acq_val}"
            )

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
            'transform_update_min_evals': transform_update_min_evals,
            'transform_acq_threshold': transform_acq_threshold,
            'kl_convergence_threshold': kl_convergence_threshold,
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

            # Periodic checkpoint save
            if ii % self.save_step == 0:
                self._save_checkpoint()

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

    def _compute_effective_batch(self, acq_val: float) -> int:
        """Return effective batch size for this iteration.

        When adaptive_batch=True, interpolates linearly between min_batch_size
        and batch_size based on how close acq_val is to the convergence threshold:
          - ratio >= 10x threshold  ->  min_batch_size
          - ratio <= 2x  threshold  ->  batch_size
          - between                 ->  linear interpolation
        """
        if not self.adaptive_batch:
            return self.batch_size

        ratio_high = 10.0
        ratio_low  = 2.0
        ratio = abs(acq_val) / max(abs(self.logz_threshold), 1e-12)
        progress = max(0.0, min(1.0, (ratio_high - ratio) / (ratio_high - ratio_low)))
        effective = int(self.min_batch_size + (self.batch_size - self.min_batch_size) * progress)
        return max(self.min_batch_size, min(self.batch_size, effective))

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
            param_bounds=self.transform._original_bounds,
            transform=self.transform,
        )
        self.results_manager.end_timing('MCMC Sampling')
        self.ns_samples = None
        logz_keys = ['mean', 'upper', 'lower', 'dlogz_sampler']  # used in do_final_ns block
        prev_acq_val = abs(self.logz_threshold) * 10.0  # seed: far from convergence → min_batch_size
        _ns_evals_last = 0  # track evals at last periodic NS run

        while not self.converged:
            ii += 1
            verbose = True

            if verbose:
                log.info(f"Iteration {ii} of {acq_name}, objective evals {current_evals}/{self.max_evals}")

            acq_kwargs = {'mc_samples': self.mc_samples, 'mc_points_size': self.mc_points_size}
            effective_batch = self._compute_effective_batch(prev_acq_val)
            if self.adaptive_batch:
                log.debug(f"  Adaptive batch={effective_batch}/{self.batch_size} "
                          f"(acq_val={prev_acq_val:.4f}, threshold={self.logz_threshold:.4f})")
            new_pts_u, acq_vals = self.get_next_batch(acq_kwargs, n_batch = effective_batch, n_restarts = 1, maxiter = 100, early_stop_patience = 10, step = ii, verbose=verbose)
            mean_acq_val = float(np.mean(acq_vals))
            prev_acq_val = mean_acq_val
            new_pts_u = jnp.atleast_2d(new_pts_u)
            new_vals = self.evaluate_likelihood(new_pts_u, ii, verbose=verbose)
            current_evals = self.total_objective_evals

            self.update_gp(new_pts_u, new_vals, step = ii)
            self.results_manager.update_best_loglike(ii, self.best_f)

            # MCMC sampling to update mc_samples every iteration
            effective_hmc_samples = (
                max(512, int(self.num_hmc_samples * effective_batch / self.batch_size))
                if self.adaptive_batch else self.num_hmc_samples
            )
            self.results_manager.start_timing('MCMC Sampling')
            self.mc_samples = get_mc_samples(
                self.gp,
                warmup_steps=self.num_hmc_warmup,
                num_samples=effective_hmc_samples,
                thinning=self.hmc_thinning,
                num_chains=self.hmc_num_chains,
                method=self.mc_points_method,
                np_rng=self.np_rng,
                rng_key=get_jax_key(),
                param_bounds=self.transform._original_bounds,
                transform=self.transform,
            )
            self.results_manager.end_timing('MCMC Sampling')

            if verbose:
                log.info(f"Current best point {self.best} with value = {self.best_f:.6f}, found at iteration {self.best_pt_iteration}")

            # Periodic save
            if ii % self.save_step == 0:
                self._save_checkpoint()

            # Attempt transform update (no-op for IdentityTransform; RotationTransform
            # and NormalisingFlowTransform handle all trigger guard logic internally)
            if current_evals >= self.transform_update_min_evals:
                did_update = self.transform.update(
                    gp=self.gp,
                    mc_samples=self.mc_samples,
                    acq_val=mean_acq_val,
                    acq_threshold=self.transform_acq_threshold,
                    best_pt=self.best_pt,
                    all_x_phys=self.all_train_x_phys,
                    all_y_raw=self.all_train_y_raw,
                    step=ii,
                )
                if did_update:
                    self.mc_samples = self.transform.updated_mc_samples
                    # Warm-start refit GP in the new transformed space.
                    # Flow updates use more restarts: the non-linear unit-cube
                    # change means old length-scale values are further from optimal.
                    self.results_manager.start_timing('GP Training')
                    _n_restarts = 8 if isinstance(self.transform, NormalisingFlowTransform) else 4
                    self.pool.gp_fit(self.gp, n_restarts=_n_restarts, maxiters=500,
                                     rng=self.np_rng, use_pool=True)
                    self.results_manager.end_timing('GP Training')
                    # Persist everything in the single checkpoint
                    self._save_checkpoint()

            # Periodic nested sampling: trigger once acq is below threshold and ns_n_points new evals have elapsed.
            # When NS runs it owns the convergence_counter, so _check_convergence is skipped that iteration.
            # Neither NS nor convergence checks are run before min_evals have been reached.
            self.current_iteration = ii
            _ran_periodic_ns = False
            if current_evals >= self.min_evals and (mean_acq_val <= self.logz_threshold) and (current_evals - _ns_evals_last >= self.ns_n_points):
                _ns_evals_last = current_evals
                _ran_periodic_ns = True
                log.info(f"Periodic nested sampling triggered at eval {current_evals}")
                self.results_manager.start_timing('Nested Sampling')
                self.ns_samples, logz_dict, ns_success = nested_sampling_Dy(
                    mode='conv', gp=self.gp, ndim=self.ndim, dlogz=0.025,
                    maxcall = int(2e6), rng=self.np_rng,
                    param_bounds=self.transform._original_bounds,
                    transform=self.transform,
                )
                self.results_manager.end_timing('Nested Sampling')
                if ns_success:
                    logz_str = ", ".join([f"{k}={logz_dict[k]:.4f}" for k in logz_keys if k in logz_dict])
                    log.info(f"Periodic NS LogZ: {logz_str}")
                    # Replace mc_samples with equal-weighted NS samples for next acquisition step
                    self.mc_samples = {'x': self.ns_samples['eq_u'], 'logp': self.ns_samples['eq_logl'], 'method': 'nested'}
                    # Fake mode detection: NS best GP-predicted logL is much higher than best observed,
                    # scaled by ndim/2 (typical log-likelihood variation for a Gaussian posterior).
                    # If triggered and adaptive_batch is on, force min_batch_size next iteration
                    # so poor NS-guided points don't dominate the batch.
                    ns_best_logl = float(np.max(self.ns_samples['eq_logl']))
                    if self.adaptive_batch and (ns_best_logl > self.best_f + self.ndim / 2):
                        log.info(
                            f"NS may have found a fake mode (NS best logl={ns_best_logl:.4f} vs GP best={self.best_f:.4f}, "
                            f"gap={ns_best_logl - self.best_f:.4f} > ndim/2={self.ndim/2:.1f}), "
                            f"forcing min_batch_size next iteration"
                        )
                        prev_acq_val = 10.0 * abs(self.logz_threshold)
                    # Convergence check using logz_std + KL
                    self.converged = self.check_convergence_logz(ii, logz_dict, self.ns_samples['eq_x'], self.ns_samples['eq_logl'])
                    if self.converged:
                        self.termination_reason = "LogZ converged (periodic NS)"
                        self.results_dict['termination_reason'] = self.termination_reason
                        break

            # Check convergence via acquisition value + consecutive KL divergence.
            # Skipped when periodic NS ran this iteration (NS owns the convergence_counter).
            if not _ran_periodic_ns and _ns_evals_last == 0 and current_evals >= self.min_evals:
                self.converged = self._check_convergence(mean_acq_val)
                if self.converged:
                    self.termination_reason = "Converged (acq + KL)"
                    self.results_dict['termination_reason'] = self.termination_reason
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
                gp=self.gp, ndim=self.ndim, maxcall=int(5e6), dynamic=True, dlogz=0.01, rng=self.np_rng,
                param_bounds=self.transform._original_bounds, transform=self.transform,
            )
            self.results_manager.end_timing('Nested Sampling')
            logz_str = ", ".join([f"{k}={logz_dict[k]:.4f}" for k in logz_keys if k in logz_dict])
            log.info(f"Final LogZ: {logz_str}")
            if ns_success:
                log.info(f"Using nested sampling results")
                self.check_convergence_logz(ii+1, logz_dict, self.ns_samples['eq_x'], self.ns_samples['eq_logl'])
                self.results_dict['logz'] = logz_dict
                if self.converged:
                    self.termination_reason = "LogZ converged"
                    self.results_dict['termination_reason'] = self.termination_reason
        else:
            log.info("Skipping final nested sampling")

        if self.ns_samples is None:
            log.info("No nested sampling results found or nested sampling unsuccessful, MC samples from HMC/MCMC will be used instead.")
            self.results_manager.start_timing('MCMC Sampling')
            mc_samples = get_mc_samples(
                    self.gp, warmup_steps=self.num_hmc_warmup, num_samples=8*self.num_hmc_samples,
                    thinning=self.hmc_thinning, method="NUTS")
            self.results_manager.end_timing('MCMC Sampling')
            samples = mc_samples['x']
            weights = mc_samples['weights'] if 'weights' in mc_samples else np.ones(mc_samples['x'].shape[0])
            loglikes = mc_samples['logp']
            # Transform NUTS samples from unit cube to physical space
            samples = np.asarray(self.transform.from_unit(samples))

            # Check GP uncertainty at the MC samples: if high warn that convergence may not have been reached yet.
            # TODO
        else:
            log.info("Using nested sampling results for final samples and logZ estimation.")
            samples = self.ns_samples['x']
            weights = self.ns_samples['weights']
            loglikes = self.ns_samples['logl']

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