import multiprocessing as mp
import numpy as np
from .logging_utils import get_logger

log = get_logger("[pool]")

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
    log.warning("mpi4py not available, MPI operations will be disabled.")


class BasePool:
    def map(self, func, iterable):
        raise NotImplementedError
    def close(self): pass
    def join(self): pass
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        self.join()


class SerialPool(BasePool):
    def map(self, func, iterable):
        return [func(x) for x in iterable]


class MultiprocessingPool(BasePool):
    def __init__(self, n_procs=None):
        self.pool = mp.Pool(processes=n_procs)

    def map(self, func, iterable):
        return self.pool.map(func, iterable)

    def close(self):
        self.pool.close()

    def join(self):
        self.pool.join()


class MPIPool(BasePool):
    def __init__(self):
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def map(self, func, iterable):
        iterable = list(iterable)
        chunks = np.array_split(iterable, self.size)
        local_chunk = chunks[self.rank]
        local_results = [func(x) for x in local_chunk]
        all_results = self.comm.gather(local_results, root=0)
        if self.rank == 0:
            return [item for sublist in all_results for item in sublist]
        else:
            return None
