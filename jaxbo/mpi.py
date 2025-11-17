# Simple MPI utilities, based on COBAYA MPI implementation written by Jesus Torrado


import numpy as np

_mpi, _mpi_size, _mpi_comm, _mpi_rank = -1, -1, -1, -1

def get_mpi():
    global _mpi
    if _mpi == -1:
        try:
            from mpi4py import MPI
            _mpi = MPI
        except ImportError:
            _mpi = None
    return _mpi

def get_mpi_comm():
    global _mpi_comm
    if _mpi_comm == -1:
        _mpi_comm = getattr(get_mpi(), "COMM_WORLD", None)
    return _mpi_comm

def get_mpi_size():
    global _mpi_size
    if _mpi_size == -1:
        _mpi_size = getattr(get_mpi_comm(), "Get_size", lambda: 1)()
    return _mpi_size

def get_mpi_rank():
    global _mpi_rank
    if _mpi_rank == -1:
        _mpi_rank = getattr(get_mpi_comm(), "Get_rank", lambda: 0)()
    return _mpi_rank

def is_main_process():
    return get_mpi_rank() == 0

def share(data=None, root=0):
    if get_mpi_size() > 1:
        return get_mpi_comm().bcast(data, root=root)
    return data

def scatter(data=None, root=0):
    if get_mpi_size() > 1:
        # Manually handle uneven splits for basic scatter
        if is_main_process():
            return get_mpi_comm().scatter(np.array_split(data, get_mpi_size()), root=root)
        return get_mpi_comm().scatter(None, root=root)
    return data

def gather(data, root=0):
    if get_mpi_size() > 1:
        return get_mpi_comm().gather(data, root=root)
    return [data]
