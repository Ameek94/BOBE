import time
import numpy as np
import jax.numpy as jnp
from typing import Callable, Dict, List, Any, Union, Optional, Tuple
from jaxbo.utils.seed import set_global_seed, get_numpy_rng, get_new_jax_key
from jaxbo.utils.log import get_logger
from jaxbo.gp import GP
from jaxbo.clf_gp import GPwithClassifier
log = get_logger('pool')

try:
    from mpi4py import MPI
    IS_MPI_AVAILABLE = True
except ImportError:
    MPI = None
    IS_MPI_AVAILABLE = False

class MPI_Pool:
    """Enhanced MPI Pool with support for managing worker state and multiple task types"""
    
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
            self.is_master = self.rank == 0
            self.is_worker = self.rank > 0
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
            self.is_mpi = False
            self.is_master = True
            self.is_worker = False

    def worker_wait(self, likelihood, seed=None):
        """Main loop for worker processes - only used in MPI mode"""
        if not self.is_worker:
            return
            
        print(f"Worker starting at rank {self.rank}")
        if seed is not None:
            seed = seed + self.rank
        set_global_seed(seed)
        rng = get_numpy_rng()
        rng_key = get_new_jax_key()

        while True:
            task_data = self.comm.recv(source=0)
            task_type, data = task_data
            try:    
                if task_type == self.TASK_OBJECTIVE_EVAL:
                    point, task_index = data
                    result = likelihood(point)
                    self.comm.send((result, task_index), dest=0)
                    
                elif task_type == self.TASK_GP_FIT:
                        # Receive payload and task_index
                        start = time.time()
                        payload = data
                        state_dict = payload['state_dict']
                        fit_params = payload['fit_params']
                        use_clf = payload.get('use_clf', False)

                        # if use_clf:
                            # worker_gp = GPwithClassifier.from_state_dict(state_dict)
                        # else:
                        worker_gp = GP.from_state_dict(state_dict)
                        end = time.time()
                        print(f"[{self.rank}]: Worker received GP fit task and initialised GP in {end - start:.2f} seconds.")

                        fit_results = worker_gp.fit(**fit_params)
                        self.comm.send(fit_results, dest=0)
                        end = time.time()
                        print(f"[{self.rank}]: Worker completed GP fit task in {end - start:.2f} seconds.")

                elif task_type == self.TASK_COBAYA_INIT:
                    # This task type doesn't need input data, just the index
                    _, task_index = data
                    pt, logpost = likelihood._get_single_valid_point(rng)
                    self.comm.send(((pt, logpost), task_index), dest=0)

                elif task_type == self.TASK_EXIT:
                    log.info("Worker exiting")
                    break
                    
            except Exception as e:
                import traceback
                log.info(f"Error in worker: {e}")
                log.info(traceback.format_exc())
                # Ensure the master receives an error message tuple with the index
                # This prevents the master from hanging while waiting for a result that will never come.
                _, task_index = data
                self.comm.send(("error", str(e), task_index), dest=0)
                
        return

    # NEW: Centralized utility for dynamic task distribution
    def _dynamic_distribute(self, tasks: List[Any], task_type: int) -> List[Any]:
        """
        MASTER-ONLY METHOD: Distributes tasks to workers using dynamic scheduling.

        This is a generic utility that sends tasks one by one to available workers
        and collects the results in order.
        """
        if not self.is_master or not self.is_mpi:
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
    def run_map_objective(self, function, tasks):
        """
        Maps a function over a list of tasks in parallel.
        """
        if not self.is_master:
            return None

        if not self.is_mpi:
            # Serial execution if not in MPI mode
            results =  [function(task) for task in tasks]

        else:
            results = self._dynamic_distribute(tasks, self.TASK_OBJECTIVE_EVAL)

        return np.array(results)

    def gp_fit(self, gp: GP, maxiters=1000, n_restarts=8, rng=None, use_pool=True):
        """
        Orchestrates a parallel GP hyperparameter fit, ensuring at least one
        restart per MPI process.
        """
        if self.is_worker:
            return None

        # Adjust n_restarts to be at least equal to the number of processes and cap to 2 restarts per process
        if self.is_mpi:
            n_restarts = max(self.size, n_restarts)
            n_restarts = min(n_restarts, 2 * self.size)

        rng = np.random.default_rng() if rng is None else rng
        n_params = gp.hyperparam_bounds.shape[1]  # hp boundsa are (2, n_params) shaped
 
        # Prepare initial parameters for all restarts.
        init_params = jnp.log(gp.get_hyperparams())
        if n_restarts > 1:
            x0_random = rng.uniform(gp.hyperparam_bounds[0], gp.hyperparam_bounds[1], size=(n_restarts - 1, n_params))
            x0 = np.vstack([init_params, x0_random])
        else:
            x0 = np.atleast_2d(init_params)

        # If not running in MPI, call the GP's local fit method and return
        if not self.is_mpi or not use_pool:
            log.info(f"Running serial GP fit with {n_restarts} restarts.")
            results = gp.fit(x0=x0, maxiter=maxiters)
            gp.update_hyperparams(results['params'])
        else:
            # MPI Specific Block
            x0_chunks = np.array_split(x0, self.size)
            state_dict = gp.state_dict()
            log.info(f"Running parallel GP fit with {n_restarts} restarts across {self.size} MPI processes.")

            for i in range(1, self.size):
                worker_x0 = x0_chunks[i]
                fit_params = {'x0': worker_x0, 'maxiter': maxiters}
                payload = {'state_dict': state_dict, 'fit_params': fit_params, 'use_clf': isinstance(gp, GPwithClassifier)}
                self.comm.send((self.TASK_GP_FIT, payload), dest=i)

            master_x0 = x0_chunks[0]
            master_result = gp.fit(x0=master_x0, maxiter=maxiters)

            all_results = [master_result]
            for i in range(1, self.size):
                worker_result = self.comm.recv(source=i)
                all_results.append(worker_result)

            best_result = max(all_results, key=lambda r: r['mll'])
            best_params = best_result['params']
            gp.update_hyperparams(best_params)

    def get_cobaya_initial_points(self, n_points: int) -> np.ndarray:
        """
        Gets initial points from the Cobaya reference prior in parallel.
        """
        if not self.is_master:
            return None

        # The payload for this task is trivial; we just need to send n_points signals.
        tasks = [None] * n_points
        results_tuples = self._dynamic_distribute(tasks, self.TASK_COBAYA_INIT)

        # The worker returns a list of tuples [(point, logpost)]
        return results_tuples

    def close(self):
        """Shut down the pool by telling all workers to exit"""
        if self.is_worker:
            return
        if self.is_mpi and self.size > 1:
            for i in range(1, self.size):
                self.comm.send((self.TASK_EXIT, None), dest=i)