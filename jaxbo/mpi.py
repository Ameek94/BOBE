# Mpi utilities for JaxBo. Based on cobaya.mpi written by Jesus Torrado.

import functools
import os
import sys
import time
from collections.abc import Callable, Iterable
from enum import IntEnum
from typing import Any, TypeVar

import numpy as np

# Type variable for generic typing
T = TypeVar('T')

# Vars to keep track of MPI parameters
_mpi: Any = None if os.environ.get("JAXBO_NOMPI", False) else -1
_mpi_size = -1
_mpi_comm: Any = -1
_mpi_rank: int | None = -1


def set_mpi_disabled(disabled=True):
    """
    Disable MPI, e.g. for use on cluster head nodes where mpi4py may be installed
    but no MPI functions will work.
    """
    global _mpi, _mpi_size, _mpi_rank, _mpi_comm
    if disabled:
        _mpi = None
        _mpi_size = 0
        _mpi_comm = None
        _mpi_rank = None
    else:
        _mpi = -1
        _mpi_size = -1
        _mpi_comm = -1
        _mpi_rank = -1


def is_disabled():
    return _mpi is None


def get_mpi():
    """
    Import and returns the MPI object, or None if not running with MPI.

    Can be used as a boolean test if MPI is present.
    """
    global _mpi
    if _mpi == -1:
        try:
            from mpi4py import MPI

            _mpi = MPI
        except ImportError:
            _mpi = None
        else:
            if more_than_one_process():
                try:
                    import dill
                except ImportError:
                    pass
                else:
                    _mpi.pickle.__init__(dill.dumps, dill.loads)

    return _mpi


def get_mpi_size():
    """
    Returns the number of MPI processes that have been invoked,
    or 0 if not running with MPI.
    """
    global _mpi_size
    if _mpi_size == -1:
        _mpi_size = getattr(get_mpi_comm(), "Get_size", lambda: 0)()
    return _mpi_size


def get_mpi_comm():
    """
    Returns the MPI communicator, or `None` if not running with MPI.
    """
    global _mpi_comm
    if _mpi_comm == -1:
        _mpi_comm = getattr(get_mpi(), "COMM_WORLD", None)
    return _mpi_comm


def get_mpi_rank():
    """
    Returns the rank of the current MPI process:
        * None: not running with MPI
        * Z>=0: process rank, when running with MPI

    Can be used as a boolean that returns `False` for both the root process,
    if running with MPI, or always for a single process; thus, everything under
    `if not(get_mpi_rank()):` is run only *once*.
    """
    global _mpi_rank
    if _mpi_rank == -1:
        _mpi_rank = getattr(get_mpi_comm(), "Get_rank", lambda: None)()
    return _mpi_rank


# Aliases for simpler use
def is_main_process():
    """
    Returns true if primary process or MPI not available.
    """
    return not bool(get_mpi_rank())


def more_than_one_process():
    return get_mpi_size() > 1


def sync_processes():
    if get_mpi_size() > 1:
        get_mpi_comm().barrier()


def share_mpi(data: T = None, root=0) -> T:
    if get_mpi_size() > 1:
        return get_mpi_comm().bcast(data, root=root)
    else:
        return data


share = share_mpi


def scatter(data=None, root=0):
    if get_mpi_size() > 1:
        return get_mpi_comm().scatter(data, root=root)
    else:
        return data[0]


def size() -> int:
    return get_mpi_size() or 1


def rank() -> int:
    return get_mpi_rank() or 0


def gather(data, root=0) -> list:
    comm = get_mpi_comm()
    if comm and more_than_one_process():
        return comm.gather(data, root=root) or []
    else:
        return [data]


def allgather(data) -> list:
    if get_mpi_size() > 1:
        return get_mpi_comm().allgather(data)
    else:
        return [data]


def zip_gather(list_of_data, root=0) -> Iterable[tuple]:
    """
    Takes a list of items and returns an iterable of lists of items from each process
    e.g. for root node
    [(a_1, a_2),(b_1,b_2),...] = zip_gather([a,b,...])
    """
    if get_mpi_size() > 1:
        return zip(*(get_mpi_comm().gather(list_of_data, root=root) or [list_of_data]))
    else:
        return ((item,) for item in list_of_data)


def array_gather(list_of_data, root=0) -> list[np.ndarray]:
    return [np.array(i) for i in zip_gather(list_of_data, root=root)]


def map_parallel(func: Callable, tasks: list, root: int = 0) -> list:
    """
    Map a function over tasks in parallel using MPI scatter/gather.
    
    The main process (rank 0) distributes tasks across all processes,
    each process evaluates the function on its assigned tasks, and
    results are gathered back to the main process.
    
    Parameters
    ----------
    func : callable
        Function to apply to each task. Should accept a single task as input.
    tasks : list
        List of tasks to distribute. Only used by the main process.
    root : int, optional
        Root process rank. Default is 0.
        
    Returns
    -------
    list
        List of results in the same order as tasks. Only returned to main process;
        worker processes return None.
        
    Examples
    --------
    >>> # In serial mode (no MPI)
    >>> results = map_parallel(lambda x: x**2, [1, 2, 3, 4])
    >>> # returns [1, 4, 9, 16]
    
    >>> # In MPI mode with 4 processes
    >>> # Main process:
    >>> results = map_parallel(likelihood, points)
    >>> # Worker processes automatically participate and return None
    """
    if not more_than_one_process():
        # Serial execution
        return [func(task) for task in tasks]
    
    comm = get_mpi_comm()
    rank_val = rank()
    size_val = size()
    
    # Main process distributes tasks
    if rank_val == root:
        # Split tasks as evenly as possible across processes
        tasks_per_process = len(tasks) // size_val
        remainder = len(tasks) % size_val
        
        split_tasks = []
        start_idx = 0
        for i in range(size_val):
            # Give extra task to first 'remainder' processes
            end_idx = start_idx + tasks_per_process + (1 if i < remainder else 0)
            split_tasks.append(tasks[start_idx:end_idx])
            start_idx = end_idx
    else:
        split_tasks = None
    
    # Scatter tasks to all processes
    local_tasks = scatter(split_tasks, root=root)
    
    # Each process evaluates its assigned tasks
    local_results = [func(task) for task in local_tasks]
    
    # Gather results back to main process
    all_results = gather(local_results, root=root)
    
    if rank_val == root:
        # Flatten the nested list structure
        return [item for sublist in all_results for item in sublist]
    else:
        return None


def gp_fit_parallel(gp, maxiters=1000, n_restarts=8, rng=None, use_parallel=True):
    """
    Parallel GP hyperparameter optimization across MPI processes.
    
    Distributes multiple optimization restarts across available MPI processes,
    with each process running independent optimizations from different initial
    points. The best result across all processes is selected.
    
    Parameters
    ----------
    gp : GP or GPwithClassifier
        Gaussian Process model to fit.
    maxiters : int, optional
        Maximum iterations for each optimization restart. Default is 1000.
    n_restarts : int, optional
        Number of random restarts for optimization. Will be adjusted to ensure
        at least one restart per process. Default is 8.
    rng : np.random.Generator, optional
        Random number generator for initial points. Default is None.
    use_parallel : bool, optional
        Whether to use parallel execution. If False, runs serially. Default is True.
        
    Returns
    -------
    None
        Updates the GP hyperparameters in-place with the best result.
        
    Notes
    -----
    - In serial mode or when use_parallel=False, runs all restarts sequentially
    - In parallel mode, distributes restarts evenly across MPI processes
    - Number of restarts adjusted to be between size() and 2*size() in parallel mode
    - First restart uses current hyperparameters, rest use random initialization
    - Each process independently optimizes its assigned restarts
    - Results gathered and best marginal log-likelihood selected
    """
    from .utils.log import get_logger
    log = get_logger("mpi")
    
    # Import here to avoid circular dependencies
    try:
        from .gp import GP
        from .clf_gp import GPwithClassifier
    except ImportError:
        from gp import GP
        from clf_gp import GPwithClassifier
    
    # Adjust n_restarts for parallel execution
    if more_than_one_process() and use_parallel:
        n_restarts = max(size(), n_restarts)
        n_restarts = min(n_restarts, 2 * size())
    
    # Only main process prepares initial parameters
    if is_main_process():
        rng = np.random.default_rng() if rng is None else rng
        n_params = gp.hyperparam_bounds.shape[1]
        
        # Prepare initial parameters for all restarts
        init_params = np.log(gp.get_hyperparams())
        if n_restarts > 1:
            x0_random = rng.uniform(
                gp.hyperparam_bounds[0], 
                gp.hyperparam_bounds[1], 
                size=(n_restarts - 1, n_params)
            )
            x0 = np.vstack([init_params, x0_random])
        else:
            x0 = np.atleast_2d(init_params)
    else:
        x0 = None
    
    # Serial execution
    if not more_than_one_process() or not use_parallel:
        if is_main_process():
            log.info(f"Running serial GP fit with {n_restarts} restarts.")
            results = gp.fit(x0=x0, maxiter=maxiters)
            gp.update_hyperparams(results['params'])
        return
    
    # Parallel execution
    if is_main_process():
        log.info(f"Running parallel GP fit with {n_restarts} restarts across {size()} processes.")
    
    # Split initial parameters across processes (main process only)
    if is_main_process():
        x0_chunks = np.array_split(x0, size())
    else:
        x0_chunks = None
    
    # Broadcast GP state to all processes
    if is_main_process():
        state_dict = gp.state_dict()
    else:
        state_dict = None
    state_dict = share(state_dict, root=0)
    
    # Worker processes reconstruct GP from state
    if not is_main_process():
        # Determine which GP class to use
        if 'clf_use_size' in state_dict:
            gp = GPwithClassifier.from_state_dict(state_dict)
        else:
            gp = GP.from_state_dict(state_dict)
    
    # Each process gets its chunk of initial parameters
    local_x0 = scatter(x0_chunks, root=0)
    
    # Each process runs its optimization restarts
    local_result = gp.fit(x0=local_x0, maxiter=maxiters)
    
    # Gather all results to main process
    all_results = gather(local_result, root=0)
    
    if is_main_process():
        # Select best result based on marginal log-likelihood
        best_result = max(all_results, key=lambda r: r['mll'])
        best_params = best_result['params']
        gp.update_hyperparams(best_params)
        log.info(f"Best MLL: {best_result['mll']:.4f}")
    
    return None