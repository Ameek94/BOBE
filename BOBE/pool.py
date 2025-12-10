import numpy as np
import jax.numpy as jnp
from typing import Callable, Dict, List, Any, Union, Optional, Tuple
from BOBE.utils.seed import set_global_seed, get_numpy_rng, get_new_jax_key
from BOBE.utils.log import get_logger
from BOBE.gp import GP
from BOBE.clf_gp import GPwithClassifier
from BOBE.likelihood import Likelihood, CobayaLikelihood
log = get_logger('pool')

try:
    from mpi4py import MPI
    IS_MPI_AVAILABLE = True
except ImportError:
    MPI = None
    IS_MPI_AVAILABLE = False

class MPI_Pool:
    """
    Enhanced MPI Pool with support for managing worker state and multiple task types.
    
    This pool implements a master-worker pattern where workers enter a waiting loop
    and the master dispatches tasks dynamically. Workers automatically participate
    after initialization and don't need explicit management in user code.
    """
    
    TASK_OBJECTIVE_EVAL = 0
    TASK_GP_FIT = 1
    TASK_ACQUISITION_OPT = 3
    TASK_COBAYA_INIT = 4
    TASK_INIT = 99
    TASK_EXIT = 100
    
    def __init__(self):
        """Initializes the pool based on whether MPI is available and active."""
        if IS_MPI_AVAILABLE:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.is_mpi = self.size > 1
            self.is_main_process = self.rank == 0
            self.is_worker = self.rank > 0
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
            self.is_mpi = False
            self.is_main_process = True
            self.is_worker = False
        
        # Track if workers are in waiting loop
        self._workers_active = False

    def worker_wait(self, likelihood: Likelihood, gp: Union[GP, GPwithClassifier] = None, seed: Optional[int] = None):
        """
        Main loop for worker processes. Workers wait for tasks from master and execute them.
        
        This method should be called by worker processes after initialization is complete.
        It enters an infinite loop waiting for tasks until TASK_EXIT is received.
        
        Parameters
        ----------
        likelihood : Likelihood
            The likelihood object for evaluating objective function.
        gp : GP or GPwithClassifier, optional
            The GP object, can be updated via state_dict broadcasts.
        seed : int, optional
            Random seed for the worker. If provided, will be offset by rank.
            
        Notes
        -----
        This method only executes for worker processes (rank > 0).
        The master process (rank 0) immediately returns.
        """
        if not self.is_worker:
            return
        
        log.info(f"Worker {self.rank} entering wait loop")
        if seed is not None:
            seed = seed + self.rank
        set_global_seed(seed)
        rng = get_numpy_rng()
        
        # Store likelihood and GP for task execution
        self._likelihood = likelihood
        self._gp = gp

        while True:
            # Wait for task from master
            task_data = self.comm.recv(source=0, tag=MPI.ANY_TAG)
            task_type, data = task_data
            
            try:
                if task_type == self.TASK_OBJECTIVE_EVAL:
                    # Evaluate likelihood at a point
                    point, task_index = data
                    result = self._likelihood(point)
                    self.comm.send((result, task_index), dest=0)
                    
                elif task_type == self.TASK_GP_FIT:
                    # Fit GP with given starting points
                    payload = data
                    state_dict = payload['state_dict']
                    fit_params = payload['fit_params']
                    use_clf = payload.get('use_clf', False)

                    # Reconstruct GP from state dict
                    if use_clf:
                        worker_gp = GPwithClassifier.from_state_dict(state_dict)
                    else:
                        worker_gp = GP.from_state_dict(state_dict)

                    # Fit GP and return results
                    fit_results = worker_gp.fit(**fit_params)
                    self.comm.send(fit_results, dest=0)

                elif task_type == self.TASK_COBAYA_INIT:
                    # Get initial point from Cobaya reference prior
                    _, task_index = data
                    pt, logpost = self._likelihood._get_single_valid_point(rng)
                    self.comm.send(((pt, logpost), task_index), dest=0)

                elif task_type == self.TASK_EXIT:
                    log.info(f"Worker {self.rank} exiting")
                    break
                    
            except Exception as e:
                import traceback
                log.error(f"Worker {self.rank} error: {e}")
                log.error(traceback.format_exc())
                # Send error back to master with task index
                _, task_index = data
                self.comm.send(("error", str(e), task_index), dest=0)
        
        return

    def _dynamic_distribute(self, tasks: List[Any], task_type: int) -> List[Any]:
        """
        MASTER-ONLY METHOD: Distributes tasks to workers using dynamic scheduling.

        This is a generic utility that sends tasks one by one to available workers
        and collects the results in order. Workers must be in worker_wait() loop.
        
        Parameters
        ----------
        tasks : list
            List of task data to distribute to workers.
        task_type : int
            Type of task (TASK_OBJECTIVE_EVAL, TASK_GP_FIT, etc.).
            
        Returns
        -------
        list
            Results from all tasks in the same order as input tasks.
        """
        if not self.is_main_process or not self.is_mpi:
            raise RuntimeError("_dynamic_distribute is designed for the master process in MPI mode.")

        n_tasks = len(tasks)
        if n_tasks == 0:
            return []

        results = [None] * n_tasks
        task_index = 0
        tasks_in_progress = 0
        
        # Initial distribution to all available workers
        for worker_rank in range(1, self.size):
            if task_index < n_tasks:
                payload = (tasks[task_index], task_index)
                self.comm.send((task_type, payload), dest=worker_rank)
                task_index += 1
                tasks_in_progress += 1
        
        # Receive results and distribute remaining tasks
        while tasks_in_progress > 0:
            status = MPI.Status()
            
            # The worker returns (result, original_index) or (error, msg, original_index)
            response = self.comm.recv(source=MPI.ANY_SOURCE, status=status)
            worker_rank = status.Get_source()

            if len(response) == 3 and response[0] == "error":
                _, msg, original_index = response
                log.error(f"Worker {worker_rank} failed on task {original_index}: {msg}")
                # Propagate the error to be handled by the caller
                raise RuntimeError(f"Worker {worker_rank} failed: {msg}")

            result, original_index = response
            results[original_index] = result
            tasks_in_progress -= 1
            
            # If there are more tasks, send one to the newly freed worker
            if task_index < n_tasks:
                payload = (tasks[task_index], task_index)
                self.comm.send((task_type, payload), dest=worker_rank)
                task_index += 1
                tasks_in_progress += 1
        
        return results

    # REFACTORED: Now uses the central utility
    def run_map_objective(self, function: Callable, tasks: List[Any]) -> np.ndarray:
        """
        Maps a function over a list of tasks in parallel.
        
        In MPI mode, distributes tasks to workers dynamically. Workers must be
        in worker_wait() loop. In serial mode, evaluates locally.
        
        Parameters
        ----------
        function : callable
            The objective/likelihood function to evaluate. Only used in serial mode.
        tasks : list or array-like
            List of input points to evaluate, shape (n_tasks, ndim).
            
        Returns
        -------
        np.ndarray
            Array of results, shape (n_tasks,) or (n_tasks, 1).
        """
        if not self.is_main_process:
            return None

        if not self.is_mpi:
            # Serial execution if not in MPI mode
            results = [function(task) for task in tasks]
        else:
            results = self._dynamic_distribute(tasks, self.TASK_OBJECTIVE_EVAL)

        return np.array(results)

    def gp_fit(self, gp: GP, maxiters=1000, n_restarts=8, rng=None, use_pool=True):
        """
        Orchestrates a parallel GP hyperparameter fit across MPI processes.
        
        Distributes multiple random restarts across workers for hyperparameter
        optimization and selects the best result.
        
        Parameters
        ----------
        gp : GP or GPwithClassifier
            Gaussian Process model to fit.
        maxiters : int, optional
            Maximum iterations for each optimization. Default is 1000.
        n_restarts : int, optional
            Number of random restarts for optimization. Default is 8.
            In MPI mode, adjusted to at least one restart per process.
        rng : np.random.Generator, optional
            Random number generator for initial points. If None, creates new one.
        use_pool : bool, optional
            Whether to use MPI pool for parallelization. Default is True.
            
        Returns
        -------
        dict or None
            Best fit result for master process, None for workers.
        """
        if self.is_worker:
            return None

        # Adjust n_restarts to be at least equal to the number of processes
        if self.is_mpi and use_pool:
            n_restarts = max(self.size, n_restarts)
            n_restarts = min(n_restarts, 2 * self.size)

        rng = np.random.default_rng() if rng is None else rng
        n_params = gp.hyperparam_bounds.shape[1]  # hp bounds are (2, n_params) shaped
 
        # Prepare initial parameters for all restarts
        init_params = jnp.log(gp.get_hyperparams())
        if n_restarts > 1:
            x0_random = rng.uniform(
                gp.hyperparam_bounds[0], 
                gp.hyperparam_bounds[1], 
                size=(n_restarts - 1, n_params)
            )
            x0 = np.vstack([init_params, x0_random])
        else:
            x0 = np.atleast_2d(init_params)

        # If not running in MPI or use_pool=False, call the GP's local fit method
        if not self.is_mpi or not use_pool:
            log.info(f"Running serial GP fit with {n_restarts} restarts.")
            results = gp.fit(x0=x0, maxiter=maxiters)
            gp.update_hyperparams(results['params'])
            return results
        
        # MPI Parallel Block - distribute restarts across workers
        log.info(f"Running parallel GP fit with {n_restarts} restarts across {self.size} MPI processes.")
        
        # Split initial points across processes
        x0_chunks = np.array_split(x0, self.size)
        state_dict = gp.state_dict()

        # Send tasks to workers
        for i in range(1, self.size):
            worker_x0 = x0_chunks[i]
            fit_params = {'x0': worker_x0, 'maxiter': maxiters}
            payload = {
                'state_dict': state_dict, 
                'fit_params': fit_params, 
                'use_clf': isinstance(gp, GPwithClassifier)
            }
            self.comm.send((self.TASK_GP_FIT, payload), dest=i)

        # Master does its share of work
        master_x0 = x0_chunks[0]
        master_result = gp.fit(x0=master_x0, maxiter=maxiters)

        # Collect results from workers
        all_results = [master_result]
        for i in range(1, self.size):
            worker_result = self.comm.recv(source=i)
            all_results.append(worker_result)

        # Select best result and update GP
        best_result = max(all_results, key=lambda r: r['mll'])
        best_params = best_result['params']
        gp.update_hyperparams(best_params)
        
        return best_result

    def get_cobaya_initial_points(self, likelihood: CobayaLikelihood, n_points: int, rng=None) -> List[Tuple]:
        """
        Gets initial points from the Cobaya reference prior in parallel.
        
        Distributes the generation of Cobaya initial points across workers.
        Workers must be in worker_wait() loop.
        
        Parameters
        ----------
        likelihood : CobayaLikelihood
            Cobaya likelihood object with _get_single_valid_point method.
        n_points : int
            Number of initial points to generate.
        rng : np.random.Generator, optional
            Random number generator. Only used in serial mode.
            
        Returns
        -------
        list of tuple
            List of (point, logpost) tuples for master process, None for workers.
        """
        if not self.is_main_process:
            return None
        
        if not self.is_mpi:
            # Serial execution
            rng = np.random.default_rng() if rng is None else rng
            return [likelihood._get_single_valid_point(rng) for _ in range(n_points)]

        # The payload for this task is trivial; we just need to send n_points signals
        tasks = [None] * n_points
        results_tuples = self._dynamic_distribute(tasks, self.TASK_COBAYA_INIT)

        return results_tuples

    def close(self):
        """
        Shut down the pool by telling all workers to exit.
        
        Sends TASK_EXIT signal to all worker processes, allowing them to
        exit from the worker_wait() loop gracefully.
        """
        if self.is_worker:
            return
        if self.is_mpi and self.size > 1:
            log.info("Sending exit signal to all workers")
            for i in range(1, self.size):
                self.comm.send((self.TASK_EXIT, None), dest=i)