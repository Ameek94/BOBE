from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import atexit
import numpy as np
import sys
import time
from .logging_utils import get_logger
log = get_logger("[pool]")


# This block attempts to initialize MPI. If it fails or if only one process
# is used, it sets up variables for serial execution.
# A reusable module for MPI-based or serial task parallelism.

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    is_mpi_run = True if size > 1 else False
except ImportError:
    comm = None
    rank = 0
    size = 1
    is_mpi_run = False

class MPI_Pool:
    """
    A Pool class that abstracts MPI parallelism. It is agnostic of the
    function it will execute.
    """
    def __init__(self):
        self.comm = comm
        self.rank = rank
        self.size = size
        self.is_mpi = is_mpi_run

        # On worker processes, this call blocks and enters the listening loop.
        if self.is_mpi and not self.is_master():
            self._worker_loop()

    def is_master(self):
        """Returns True if the current process is the master (rank 0)."""
        return self.rank == 0

    def map(self, function, tasks):
        """
        Maps a function over a list of tasks, in parallel or serially.

        The 'function' object itself is sent to the workers.
        """
        if not self.is_master():
            return None

        if not self.is_mpi:
            # Serial execution is straightforward
            return [function(task) for task in tasks]
        else:
            # Parallel (MPI) Execution
            n_tasks = len(tasks)
            results = [None] * n_tasks
            task_index = 0
            workers_busy = 0
            
            # Send initial tasks to all workers
            for worker_rank in range(1, self.size):
                if task_index < n_tasks:
                    # ⭐ KEY CHANGE: Send the function along with the task
                    payload = (function, tasks[task_index], task_index)
                    self.comm.send(payload, dest=worker_rank)
                    task_index += 1
                    workers_busy += 1
            
            # Receive results and dispatch remaining tasks
            while workers_busy > 0:
                status = MPI.Status()
                result_payload = self.comm.recv(source=MPI.ANY_SOURCE, status=status)
                worker_rank = status.Get_source()
                
                result, original_index = result_payload
                results[original_index] = result
                
                if task_index < n_tasks:
                    payload = (function, tasks[task_index], task_index)
                    self.comm.send(payload, dest=worker_rank)
                    task_index += 1
                else:
                    workers_busy -= 1
            
            return results

    def _worker_loop(self):
        """
        Worker's main loop. It receives a function and a task, executes,
        and returns the result.
        """
        while True:
            payload = self.comm.recv(source=0)
            
            if payload is None: # Termination signal
                break
            
            # Unpack the function to be executed
            func_to_run, task, original_index = payload
            
            # Execute the received function with the task
            result = func_to_run(task)

            log.info(f"Worker {self.rank} completed a task.")
            
            result_payload = (result, original_index)
            self.comm.send(result_payload, dest=0)

    def close(self):
        """On the master, sends a termination signal to all workers."""
        if self.is_mpi and self.is_master():
            for worker_rank in range(1, self.size):
                self.comm.send(None, dest=worker_rank)