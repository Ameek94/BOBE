from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import atexit
import numpy as np
import sys
import time
from .logging_utils import get_logger
log = get_logger("pool")


# A reusable module for MPI-based or serial task parallelism. 
# In BOBE, this is used to evaluate the likelihood in parallel for a batch of candidate points.
# Later will be modified to run more operations in parallel, most importantly gp fitting, maybe acquisition optimisations and possibly MCMC/Nested sampling.

try:
    from mpi4py import MPI
    IS_MPI_AVAILABLE = True
except ImportError:
    MPI = None
    IS_MPI_AVAILABLE = False

class MPI_Pool:
    """A hybrid pool that supports the 'pre-instantiate' pattern for MPI
    and provides a serial fallback."""
    
    def __init__(self):
        """Initializes the pool based on whether MPI is available and active."""
        if IS_MPI_AVAILABLE:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.is_mpi = self.size > 1
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
            self.is_mpi = False

    def is_master(self):
        return self.rank == 0
    
    def run_map(self, function, tasks):
        """
        MASTER METHOD: Manages task distribution.
        """
        if not self.is_master():
            return None

        if not self.is_mpi:
            return function(tasks)

        # This MPI-specific block is now safe because if we reach here,
        # we know 'is_mpi' is True, which means the MPI import succeeded.
        n_tasks = len(tasks)
        results = [None] * n_tasks
        task_index = 0
        
        for worker_rank in range(1, self.size):
            if task_index < n_tasks:
                self.comm.send((tasks[task_index], task_index), dest=worker_rank)
                task_index += 1
        
        for _ in range(n_tasks):
            status = MPI.Status()
            result, original_index = self.comm.recv(source=MPI.ANY_SOURCE, status=status)
            worker_rank = status.Get_source()
            
            results[original_index] = result
            
            if task_index < n_tasks:
                self.comm.send((tasks[task_index], task_index), dest=worker_rank)
                task_index += 1
        
        return np.array(results)

    def worker_wait(self, function):
        """
        WORKER METHOD: Waits for tasks. Only runs in MPI mode.
        """
        if self.is_master() or not self.is_mpi:
            return

        while True:
            payload = self.comm.recv(source=0)
            if payload is None:
                break

            point, original_index = payload
            result = function(point)
            self.comm.send((result, original_index), dest=0)
            
    def close(self):
        """
        MASTER METHOD: Sends a termination signal to all worker processes.
        """
        if self.is_master() and self.is_mpi:
            for worker_rank in range(1, self.size):
                self.comm.send(None, dest=worker_rank)

    # Method for evaluating loglikelihood on Master as well.
    # def run_map(self, function, tasks):
    #     """
    #     MASTER METHOD: Distributes tasks to all processes, including itself.
    #     """
    #     if not self.is_master():
    #         return None

    #     if not self.is_mpi:
    #         return function(tasks)

    #     n_tasks = len(tasks)
        
    #     # Create a task queue (using a list's pop method) and results array.
    #     task_queue = list(enumerate(tasks)) # Store as (original_index, task)
    #     results = [None] * n_tasks
    #     n_results_received = 0

    #     # Send the first wave of tasks to the actual workers (ranks > 0)
    #     worker_ranks = list(range(1, self.size))
    #     for worker_rank in worker_ranks:
    #         if task_queue:
    #             original_index, task = task_queue.pop(0)
    #             self.comm.send((task, original_index), dest=worker_rank)

    #     # Main loop: The master now does work and manages workers.
    #     while n_results_received < n_tasks:
    #         # If there are still tasks to do, the master takes one for itself.
    #         if task_queue:
    #             original_index, task = task_queue.pop(0)
    #             # Master computes the result locally
    #             result = function(task)
    #             results[original_index] = result
    #             n_results_received += 1
            
    #         # Whether the master worked or not, it must check for results from workers.
    #         # This call will block until any worker has finished its task.
    #         # It's safe to do this even if all tasks are done, as workers will
    #         # be sending back their final results.
    #         if self.size > 1 and n_results_received < n_tasks:
    #             status = MPI.Status()
    #             result, original_index = self.comm.recv(source=MPI.ANY_SOURCE, status=status)
    #             worker_rank = status.Get_source()
                
    #             results[original_index] = result
    #             n_results_received += 1
                
    #             # If there are still tasks left in the queue, send a new one
    #             # to the worker that just finished.
    #             if task_queue:
    #                 new_original_index, new_task = task_queue.pop(0)
    #                 self.comm.send((new_task, new_original_index), dest=worker_rank)

    #     return np.array(results)

    # def worker_wait(self, function):
    #     """
    #     WORKER METHOD: Waits for tasks. Only runs in MPI mode.
    #     """
    #     if self.is_master() or not self.is_mpi:
    #         return

    #     while True:
    #         payload = self.comm.recv(source=0)
    #         if payload is None:
    #             break

    #         point, original_index = payload
    #         result = function(point)
    #         self.comm.send((result, original_index), dest=0)