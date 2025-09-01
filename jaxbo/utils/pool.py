import numpy as np
import jax.numpy as jnp
import sys
import os
from typing import Callable, Dict, List, Any, Union, Optional, Tuple
import importlib
import pickle
import inspect
from .logging_utils import get_logger
log = get_logger(__name__)

def create_worker_config(obj, include_class_path=True):
    """
    Create a serializable configuration dictionary from an object
    
    Args:
        obj: The object to serialize
        include_class_path: Whether to include the class path
        
    Returns:
        Dict containing class path and parameters
    """
    config = {}
    
    if include_class_path:
        # Get the full class path
        cls = obj.__class__
        module_name = cls.__module__
        class_name = cls.__name__
        config['class_path'] = f"{module_name}.{class_name}"
    
    # Extract constructor parameters
    params = {}
    signature = inspect.signature(obj.__class__.__init__)
    
    for param_name in signature.parameters:
        if param_name != 'self' and hasattr(obj, param_name):
            value = getattr(obj, param_name)
            # Test if value is picklable
            try:
                pickle.dumps(value)
                params[param_name] = value
            except (pickle.PicklingError, TypeError, AttributeError) as e:
                log.warning(f"Parameter '{param_name}' is not picklable: {e}")
                # Store a placeholder or skip non-picklable parameters
                pass
    
    config['params'] = params
    return config

def get_likelihood_config(likelihood):
    """Create serializable configuration for likelihood object"""
    # First, try to test if the entire likelihood object is picklable
    try:
        pickle.dumps(likelihood)
        # If picklable, use the standard approach
        config = create_worker_config(likelihood)
        config['serialization_method'] = 'standard'
        log.debug("Likelihood object is picklable, using standard serialization")
        return config
    except (pickle.PicklingError, TypeError, AttributeError) as e:
        log.warning(f"Likelihood object is not picklable: {e}")
        
        # For non-picklable objects, we have a few options:
        
        # Option 1: Try to get module and function name if it's a simple function
        if hasattr(likelihood, '__module__') and hasattr(likelihood, '__name__'):
            config = {
                'class_path': f"{likelihood.__module__}.{likelihood.__name__}",
                'params': {},
                'serialization_method': 'function_reference'
            }
            log.debug("Using function reference serialization")
            return config
        
        # Option 2: Try to serialize just the class path and picklable parameters
        try:
            config = create_worker_config(likelihood)
            config['serialization_method'] = 'partial'
            log.debug("Using partial serialization (class path + picklable params only)")
            return config
        except Exception as e2:
            log.error(f"Failed to create partial config: {e2}")
            
            # Option 3: Fallback - store only the class information
            cls = likelihood.__class__
            config = {
                'class_path': f"{cls.__module__}.{cls.__name__}",
                'params': {},
                'serialization_method': 'class_only',
                'warning': 'Likelihood object could not be serialized - workers will need to recreate it'
            }
            log.warning("Using class-only serialization - likelihood must be recreatable from class constructor")
            return config

def get_gp_config(gp):
    """Create serializable configuration for GP object"""
    config = create_worker_config(gp, include_class_path=True)
    
    # Special handling for GP parameters that need custom serialization
    # Remove function references that can't be pickled
    if 'kernel' in config['params']:
        del config['params']['kernel']
    if 'mll_optimize' in config['params']:
        del config['params']['mll_optimize']
    
    # Store string identifiers instead of function references
    config['params']['kernel'] = gp.kernel_name
    config['params']['optimizer'] = gp.optimizer_method
    
    return config

class WorkerState:
    """Class to hold worker state data that persists across calls"""
    def __init__(self):
        self.loglikelihood = None
        self.gp = None
        self.acquisition = None
        self.initialized = False
        self.rank = 0  # Default for serial mode
    
    def __str__(self):
        return f"WorkerState(rank={self.rank}, initialized={self.initialized})"

class MPI_Pool:
    """Enhanced MPI Pool with support for managing worker state and multiple task types"""
    
    TASK_OBJECTIVE_EVAL = 0
    TASK_GP_FIT = 1
    TASK_ACQUISITION_OPT = 2
    TASK_INIT = 99
    TASK_EXIT = 100
    
    def __init__(self, comm=None, debug=False, likelihood=None):
        self.debug = debug
        
        # Try to import MPI, fall back to serial mode if not available
        try:
            from mpi4py import MPI
            self.mpi_available = True
            self.comm = comm if comm is not None else MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.is_master = self.rank == 0
            self.is_worker = self.rank != 0
            
            # Start the worker loop if we're not the master
            if self.is_worker:
                self._worker_loop(likelihood)
                sys.exit(0)  # Workers exit after loop
            
            if self.size == 1:
                self.worker_state = WorkerState()
                # Set likelihood directly if provided
                if likelihood is not None:
                    self.worker_state.loglikelihood = likelihood

            self.print(f"Initialized MPI_Pool with {self.size} processes")
            
        except ImportError:
            # Serial fallback mode
            self.mpi_available = False
            self.rank = 0
            self.size = 1
            self.is_master = True
            self.is_worker = False
            self.worker_state = WorkerState()  # Create local worker state for serial execution
            # Set likelihood directly if provided
            if likelihood is not None:
                self.worker_state.loglikelihood = likelihood
            self.print("MPI not available, running in serial mode")
    
    def print(self, msg):
        """Print with rank information if debug is enabled"""
        if self.debug:
            print(f"[Rank {self.rank}] {msg}", flush=True)
    
    def initialize_workers(self, gp_config=None):
        """Initialize worker states with GP configuration (likelihood should already be set)"""
        if self.is_worker:
            return
            
        if self.mpi_available and self.size > 1:
            self.print("Sending GP configuration to workers")
            # Send initialization task and config to all workers
            data = {
                'gp_config': gp_config
            }
            for i in range(1, self.size):
                self.comm.send((self.TASK_INIT, data), dest=i)
            
            # Wait for confirmation from all workers
            for i in range(1, self.size):
                response = self.comm.recv(source=i)
                self.print(f"Worker {i} initialized: {response}")
        else:
            # Serial mode: Initialize the local worker state
            self.print("Initializing local worker state")
            if gp_config is not None:
                self.worker_state.gp = self._create_object_from_config(gp_config)
                
            self.worker_state.initialized = True
            
    def run_map(self, items: List, **kwargs) -> Optional[np.ndarray]:
        """Run a map operation for objective evaluation in parallel or serial"""
        if self.is_worker:
            return None
        return self._map_objective_evaluation(items, **kwargs)
            
    def _map_objective_evaluation(self, points, batch_size=1):
        """Map objective evaluation across workers or run locally"""
        if self.is_worker:
            return None
            
        self.print(f"Mapping objective evaluation over {len(points)} points")
        results = np.zeros((len(points),))
        
        # Convert points to numpy array for consistent indexing
        points = np.array(points)
        
        if self.mpi_available and self.size > 1:
            # MPI mode: Distribute work to workers
            worker_assignments = self._get_worker_assignments(len(points))
            
            # Send data to workers
            for worker_id, indices in worker_assignments.items():
                worker_points = points[indices]
                self.comm.send((self.TASK_OBJECTIVE_EVAL, worker_points), dest=worker_id)
            
            # Collect results
            for worker_id, indices in worker_assignments.items():
                worker_results = self.comm.recv(source=worker_id)
                results[indices] = worker_results
        else:
            # Serial mode: Evaluate locally
            for i, point in enumerate(points):
                results[i] = self.worker_state.loglikelihood(point)
                
        return results
        
    def run_parallel_gp_fit(self, train_x, train_y, fit_params):
        """Run GP fitting in parallel or locally"""
        if self.is_worker:
            return None
            
        self.print("Running GP fitting")
        
        if self.mpi_available and self.size > 1:
            # MPI mode: Send to all workers and collect results
            for i in range(1, self.size):
                self.comm.send((self.TASK_GP_FIT, {
                    'train_x': train_x,
                    'train_y': train_y,
                    'fit_params': fit_params
                }), dest=i)
            
            # Collect results from all workers
            worker_results = []
            best_result = None
            best_loss = float('inf')
            
            for i in range(1, self.size):
                worker_result = self.comm.recv(source=i)
                worker_results.append({
                    'worker_id': i,
                    'loss': worker_result['loss'],
                    'hyperparams': worker_result.get('hyperparams', {}),
                    'params': worker_result['params']
                })
                
                if worker_result['loss'] < best_loss:
                    best_loss = worker_result['loss']
                    best_result = worker_result
            
            # Display results from all workers
            self.print("=== GP Fitting Results from All Workers ===")
            for result in worker_results:
                self.print(f"Worker {result['worker_id']}:")
                self.print(f"  MLL: {result['loss']:.6f}")
                if 'hyperparams' in result and result['hyperparams']:
                    for param_name, param_value in result['hyperparams'].items():
                        if hasattr(param_value, '__len__') and not isinstance(param_value, str):
                            # Handle arrays (both numpy and JAX)
                            param_str = np.array2string(np.asarray(param_value), precision=4, suppress_small=True)
                        else:
                            # Handle scalars
                            param_str = f"{float(param_value):.6f}"
                        self.print(f"  {param_name}: {param_str}")
                self.print("")
            
            self.print(f"Best result from Worker {[r['worker_id'] for r in worker_results if r['loss'] == best_loss][0]} with MLL = {best_loss:.6f}")
            
        else:
            # Serial mode: Run locally
            # Update GP training data
            self.worker_state.gp.train_x = train_x
            self.worker_state.gp.train_y = train_y
            
            # Perform GP fitting
            self.worker_state.gp.fit(**fit_params)
            
            # Compute current MLL after fitting
            current_hyperparams = jnp.concatenate([jnp.log10(self.worker_state.gp.lengthscales), jnp.log10(jnp.array([self.worker_state.gp.kernel_variance]))])
            current_mll = -self.worker_state.gp.neg_mll(current_hyperparams)
            
            best_result = {
                'loss': float(current_mll),
                'params': self.worker_state.gp.__getstate__(),
                'hyperparams': {
                    'lengthscales': self.worker_state.gp.lengthscales,
                    'kernel_variance': self.worker_state.gp.kernel_variance,
                    'noise': self.worker_state.gp.noise
                }
            }
            
            self.print("=== GP Fitting Results (Serial Mode) ===")
            self.print(f"MLL: {best_result['loss']:.6f}")
            for param_name, param_value in best_result['hyperparams'].items():
                if hasattr(param_value, '__len__') and not isinstance(param_value, str):
                    # Handle arrays (both numpy and JAX)
                    param_str = np.array2string(np.asarray(param_value), precision=4, suppress_small=True)
                else:
                    # Handle scalars
                    param_str = f"{float(param_value):.6f}"
                self.print(f"{param_name}: {param_str}")
            
        return best_result
        
    def run_parallel_acquisition(self, acq_type, acq_params, n_batch=1):
        """Run acquisition function optimization in parallel or locally"""
        if self.is_worker:
            return None
            
        self.print(f"Running acquisition optimization for {acq_type}")
        
        if self.mpi_available and self.size > 1:
            # MPI mode: Send to all workers
            for i in range(1, self.size):
                self.comm.send((self.TASK_ACQUISITION_OPT, {
                    'acq_type': acq_type,
                    'acq_params': acq_params,
                    'n_batch': n_batch
                }), dest=i)
            
            # Collect and combine results
            all_points = []
            all_values = []
            
            for i in range(1, self.size):
                worker_result = self.comm.recv(source=i)
                all_points.append(worker_result['points'])
                all_values.append(worker_result['values'])
                
            # Combine and select the best n_batch points
            all_points = np.vstack(all_points)
            all_values = np.concatenate(all_values)
        else:
            # Serial mode: Run locally
            # Create or update acquisition function
            if not hasattr(self.worker_state, 'acquisition') or self.worker_state.acquisition.name != acq_type:
                acq_module = importlib.import_module('jaxbo.acquisition')
                acq_class = getattr(acq_module, acq_type)
                self.worker_state.acquisition = acq_class()
            
            # Optimize acquisition function
            points, values = self.worker_state.acquisition.optimize(self.worker_state.gp, **acq_params['kwargs'])
            all_points = points
            all_values = values
            
        # Sort by acquisition value (assuming higher is better)
        indices = np.argsort(-all_values)[:n_batch]
        return all_points[indices], all_values[indices]
        
    def update_gp_state(self, gp_state_dict):
        """Update GP state on all workers or locally"""
        if self.is_worker:
            return
            
        if self.mpi_available and self.size > 1:
            self.print("Updating GP state on workers")
            # Send new state to all workers
            for i in range(1, self.size):
                self.comm.send(('update_gp', gp_state_dict), dest=i)
            
            # Wait for confirmation from all workers
            for i in range(1, self.size):
                response = self.comm.recv(source=i)
                self.print(f"Worker {i} GP updated: {response}")
        else:
            # Serial mode: Update local GP state
            if hasattr(self.worker_state, 'gp') and self.worker_state.gp is not None:
                self.worker_state.gp.__setstate__(gp_state_dict)
                
    def _get_worker_assignments(self, num_items):
        """Divide work among available workers"""
        assignments = {}
        num_workers = self.size - 1  # Exclude master
        
        if num_workers == 0:
            return {0: np.arange(num_items)}
        
        # If we have fewer items than workers, only assign to some workers
        if num_items <= num_workers:
            # Assign one item per worker up to num_items
            for i in range(min(num_items, num_workers)):
                assignments[i + 1] = np.array([i])
        else:
            # Standard case: more items than workers
            items_per_worker = num_items // num_workers
            remainder = num_items % num_workers
            
            start_idx = 0
            for i in range(num_workers):
                # Give extra items to first 'remainder' workers
                worker_items = items_per_worker + (1 if i < remainder else 0)
                end_idx = start_idx + worker_items
                
                if worker_items > 0:
                    assignments[i + 1] = np.arange(start_idx, end_idx)
                
                start_idx = end_idx
                
        return assignments
    
    def _create_object_from_config(self, config):
        """Dynamically create an object from its configuration dictionary"""
        class_path = config['class_path']
        params = config.get('params', {})
        serialization_method = config.get('serialization_method', 'standard')
        
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        
        if serialization_method == 'function_reference':
            # For functions, get the function directly
            return getattr(module, class_name)
        else:
            # For classes, instantiate with available parameters
            class_ = getattr(module, class_name)
            try:
                return class_(**params)
            except Exception as e:
                if serialization_method == 'class_only':
                    # Try creating with no parameters as fallback
                    log.warning(f"Failed to create {class_name} with params, trying no-args constructor: {e}")
                    try:
                        return class_()
                    except Exception as e2:
                        log.error(f"Failed to create {class_name} with no-args constructor: {e2}")
                        raise RuntimeError(f"Could not recreate {class_name} on worker. "
                                         f"The likelihood object was not picklable and could not be reconstructed. "
                                         f"Consider implementing a custom serialization method or ensuring "
                                         f"your likelihood class can be instantiated with no arguments.")
                else:
                    raise
        
    def _worker_loop(self, likelihood=None):
        """Main loop for worker processes - only used in MPI mode"""
        if not self.is_worker:
            return
            
        # Initialize worker state
        state = WorkerState()
        state.rank = self.rank
        # Set likelihood directly if provided (avoids pickling issues)
        if likelihood is not None:
            state.loglikelihood = likelihood
        self.print(f"Worker starting")
        
        while True:
            # Wait for task from master
            task_data = self.comm.recv(source=0)
            task_type, data = task_data
            
            try:
                if task_type == self.TASK_INIT:
                    # Initialize GP object (likelihood should already be set)
                    if data.get('gp_config'):
                        gp_config = data['gp_config']
                        state.gp = self._create_object_from_config(gp_config)
                        
                    state.initialized = True
                    self.print(f"Worker initialized with {state.loglikelihood}")
                    self.comm.send("initialized", dest=0)
                    
                elif task_type == self.TASK_OBJECTIVE_EVAL:
                    # Evaluate objective function
                    points = data
                    results = []
                    for point in points:
                        result = state.loglikelihood(point)
                        results.append(result)
                    self.comm.send(np.array(results), dest=0)
                    
                elif task_type == self.TASK_GP_FIT:
                    # Update GP training data and fit
                    train_x = data['train_x']
                    train_y = data['train_y']
                    fit_params = data['fit_params']
                    
                    # Update GP training data
                    state.gp.train_x = train_x
                    state.gp.train_y = train_y
                    
                    # Perform GP fitting with possibly different random restarts
                    result = state.gp.fit(**fit_params)
                    
                    # Compute current MLL after fitting
                    current_hyperparams = jnp.concatenate([jnp.log10(state.gp.lengthscales), jnp.log10(jnp.array([state.gp.kernel_variance]))])
                    current_mll = -state.gp.neg_mll(current_hyperparams)
                    
                    # Collect hyperparameters
                    hyperparams = {
                        'lengthscales': np.asarray(state.gp.lengthscales),
                        'kernel_variance': float(state.gp.kernel_variance),
                        'noise': float(state.gp.noise)
                    }
                    
                    self.comm.send({
                        'loss': float(current_mll),
                        'params': state.gp.__getstate__(),
                        'hyperparams': hyperparams
                    }, dest=0)
                    
                elif task_type == self.TASK_ACQUISITION_OPT:
                    # Run acquisition optimization
                    acq_type = data['acq_type']
                    acq_params = data['acq_params']
                    n_batch = data['n_batch']
                    
                    # Create or update acquisition function
                    if state.acquisition is None or state.acquisition.name != acq_type:
                        acq_module = importlib.import_module('jaxbo.acquisition')
                        acq_class = getattr(acq_module, acq_type)
                        state.acquisition = acq_class()
                    
                    # Optimize acquisition function
                    points, values = state.acquisition.optimize(state.gp, **acq_params['kwargs'])
                    
                    self.comm.send({
                        'points': points,
                        'values': values
                    }, dest=0)
                    
                elif task_type == 'update_gp':
                    # Update GP state
                    if state.gp is not None:
                        state.gp.__setstate__(data)
                    self.comm.send("gp_updated", dest=0)
                    
                elif task_type == self.TASK_EXIT:
                    # Exit worker loop
                    self.print("Worker exiting")
                    break
                    
            except Exception as e:
                import traceback
                self.print(f"Error in worker: {e}")
                self.print(traceback.format_exc())
                self.comm.send(("error", str(e)), dest=0)
                
        return
        
    def close(self):
        """Shut down the pool by telling all workers to exit"""
        if self.is_worker:
            return
            
        if self.mpi_available and self.size > 1:
            for i in range(1, self.size):
                self.comm.send((self.TASK_EXIT, None), dest=i)