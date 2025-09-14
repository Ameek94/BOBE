import numpy as np
import jax.numpy as jnp
import sys
import os
from typing import Callable, Dict, List, Any, Union, Optional, Tuple
import importlib
import pickle
import inspect
from .logging_utils import get_logger
from jaxbo.gp import GP
from jaxbo.clf_gp import GPwithClassifier
log = get_logger(__name__)


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
    
    def worker_wait(self, likelihood):
        """Main loop for worker processes - only used in MPI mode"""
        if not self.is_worker:
            return
            
        # Initialize worker state
        print(f"Worker starting at rank {self.rank}")
        
        while True:
            # Wait for task from master
            task_data = self.comm.recv(source=0)
            task_type, data = task_data
            try:    
                if task_type == self.TASK_OBJECTIVE_EVAL:
                    # Evaluate objective function
                    point, task_index = data # map sends (point, index)
                    result = likelihood(point)
                    self.comm.send((result,task_index), dest=0)
                    

                elif task_type == self.TASK_GP_FIT:
                    # Initialize GP object (likelihood should already be set)
                    state_dict = data['state_dict']
                    fit_params = data['fit_params']
                    worker_gp = GP.from_state_dict(state_dict)
                    fit_results = worker_gp.fit(**fit_params)
                    self.comm.send(fit_results, dest=0)

                elif task_type == self.TASK_EXIT:
                    # Exit worker loop
                    log.info("Worker exiting")
                    break
                    
            except Exception as e:
                import traceback
                log.info(f"Error in worker: {e}")
                log.info(traceback.format_exc())
                self.comm.send(("error", str(e)), dest=0)
                
        return
    
    def run_map_objective(self, function, tasks):
        """
        MASTER METHOD: Manages task distribution, does not execute tasks itself.
        """
        if not self.is_master:
            return None

        if not self.is_mpi:
            return function(tasks)

        # mpi specific block
        n_tasks = len(tasks)
        results = [None] * n_tasks
        task_index = 0
        
        for worker_rank in range(1, self.size):
            if task_index < n_tasks:
                self.comm.send((self.TASK_OBJECTIVE_EVAL, [tasks[task_index], task_index]), dest=worker_rank)
                task_index += 1
        
        for _ in range(n_tasks):
            status = MPI.Status()
            result, original_index = self.comm.recv(source=MPI.ANY_SOURCE, status=status)
            worker_rank = status.Get_source()
            
            results[original_index] = result
            
            if task_index < n_tasks:
                # since the worker is free send it another task
                self.comm.send((self.TASK_OBJECTIVE_EVAL, [tasks[task_index], task_index]), dest=worker_rank)
                task_index += 1
        
        return np.array(results)
    
    def gp_fit(self, gp: GP, maxiters=1000, n_restarts=8, rng=None):
        """
        Orchestrates a parallel GP hyperparameter fit, ensuring at least one
        restart per MPI process.
        """
        if self.is_worker:
            return None
        
        # Adjust n_restarts to match the number of processes, cap to 2 restarts per process
        # print(f"Original n_restarts: {n_restarts}, MPI size: {self.size}")
        if self.is_mpi:
            n_restarts = max(self.size, n_restarts)
            n_restarts = min(n_restarts, 2 * self.size)

        # print(f"Adjusted n_restarts: {n_restarts}, MPI size: {self.size}")


        rng = np.random.default_rng() if rng is None else rng
        n_params = gp.hyperparam_bounds.shape[1] # (2, n_params)

        # Prepare initial parameters for all restarts. This now uses the adjusted n_restarts.
        init_params = jnp.array(gp.lengthscales)
        if not gp.fixed_kernel_variance:
            init_params = jnp.concatenate([init_params, jnp.array([gp.kernel_variance])])
        if 'tausq' in gp.param_names:
            init_params = jnp.concatenate([init_params, jnp.array([gp.tausq])])

        if n_restarts > 1:
            x0_random = rng.uniform(gp.hyperparam_bounds[0], gp.hyperparam_bounds[1], size=(n_restarts - 1, n_params))
            x0 = np.vstack([init_params, x0_random])
        else:
            x0 = np.atleast_2d(init_params)

        # print(f'GP fit with x0 shape: {x0.shape}')

        # If not running in MPI, call the GP's local fit method and return
        if not self.is_mpi:
            # The n_restarts value might have been adjusted, which is fine.
            results = gp.fit(x0=x0, maxiter=maxiters)
            gp.update_hyperparams(results['params'])

        
        # MPI Specific Block 

        # Split the restart starting points
        x0_chunks = np.array_split(x0, self.size)
        
        
        state_dict = gp.state_dict()

        for i in range(1, self.size):
            worker_x0 = x0_chunks[i]
            # print(f"Master sending GP fit task to worker {i} with {worker_x0.shape} restarts")
            fit_params = {'x0': worker_x0, 'maxiter': maxiters}
            payload = {'state_dict': state_dict, 'fit_params': fit_params}
            self.comm.send((self.TASK_GP_FIT, payload), dest=i)

        # 4. Master processes its own chunk
        master_x0 = x0_chunks[0]
        # print(f"Master processing GP fit with {master_x0.shape} restarts")
        master_result = gp.fit(x0=master_x0, maxiter=maxiters)

        # 5. Gather results from all workers
        all_results = [master_result]
        for i in range(1, self.size):
            worker_result = self.comm.recv(source=i)
            # print(f"Master received GP fit result from worker {i}: {worker_result}")
            all_results.append(worker_result)
            
        best_result = max(all_results, key=lambda r: r['mll'])
        best_params = best_result['params']

        log.info(f"[Master] GP fit complete. Best MLL {-best_result['mll']:.4f} found.")

        gp.update_hyperparams(best_params)

    def close(self):
        """Shut down the pool by telling all workers to exit"""
        if self.is_worker:
            return

        if self.is_mpi and self.size > 1:
            for i in range(1, self.size):
                self.comm.send((self.TASK_EXIT, None), dest=i)
    
