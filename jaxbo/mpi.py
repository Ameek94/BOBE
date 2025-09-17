# Simple MPI utilities, based on COBAYA MPI implementation written by Jesus Torrado


import numpy as np

_mpi, _mpi_size, _mpi_comm, _mpi_rank = -1, -1, -1, -1

def get_mpi():
    """
    Get the MPI module if available, else None.
    """
    global _mpi
    if _mpi == -1:
        try:
            from mpi4py import MPI
            _mpi = MPI
        except ImportError:
            _mpi = None
    return _mpi

def get_mpi_comm():
    """
    Get the MPI communicator if available, else None.
    """
    global _mpi_comm
    if _mpi_comm == -1:
        _mpi_comm = getattr(get_mpi(), "COMM_WORLD", None)
    return _mpi_comm

def get_mpi_size():
    """
    Get the size of the MPI communicator if available, else 1.
    """
    global _mpi_size
    if _mpi_size == -1:
        _mpi_size = getattr(get_mpi_comm(), "Get_size", lambda: 1)()
    return _mpi_size

def get_mpi_rank():
    """
    Get the rank of the current MPI process if available, else 0.
    """
    global _mpi_rank
    if _mpi_rank == -1:
        _mpi_rank = getattr(get_mpi_comm(), "Get_rank", lambda: 0)()
    return _mpi_rank

def is_main_process():
    """
    Check if the current process is the main MPI process (rank 0).
    """
    return get_mpi_rank() == 0

def share(data=None, root=0):
    """
    Share data among all MPI processes. If MPI is not available or size is 1, returns the data as is.
    """
    if get_mpi_size() > 1:
        return get_mpi_comm().bcast(data, root=root)
    return data

def scatter(data=None, root=0):
    """
    Scatter data among MPI processes. If MPI is not available or size is 1, returns the data as is.
    """
    if get_mpi_size() > 1:
        # Manually handle uneven splits for basic scatter
        if is_main_process():
            return get_mpi_comm().scatter(np.array_split(data, get_mpi_size()), root=root)
        return get_mpi_comm().scatter(None, root=root)
    return data

def gather(data, root=0):
    """
    Gather data from all MPI processes to the root process. If MPI is not available or size is 1, returns a list with the data.
    """
    if get_mpi_size() > 1:
        return get_mpi_comm().gather(data, root=root)
    return [data]

# def dynamic_distribute(tasks):
#     """
#     MASTER-ONLY METHOD: Distributes tasks to workers using dynamic scheduling.

#     This is a generic utility that sends tasks one by one to available workers
#     and collects the results in order.
#     """
#     if not self.is_master or not self.is_mpi:
#         raise RuntimeError("_dynamic_distribute is designed for the master process in MPI mode.")

#     n_tasks = len(tasks)
#     if n_tasks == 0:
#         return []

#     results = [None] * n_tasks
#     task_index = 0
#     tasks_in_progress = 0
    
#     # Initial distribution to all available workers
#     for worker_rank in range(1, self.size):
#         if task_index < n_tasks:
#             payload = (tasks[task_index], task_index)
#             self.comm.send((task_type, payload), dest=worker_rank)
#             task_index += 1
#             tasks_in_progress += 1
    
#     # Receive results and distribute remaining tasks
#     while tasks_in_progress > 0:
#         status = MPI.Status()
        
#         # The worker returns (result, original_index) or (error, msg, original_index)
#         response = self.comm.recv(source=MPI.ANY_SOURCE, status=status)
#         worker_rank = status.Get_source()

#         if len(response) == 3 and response[0] == "error":
#             _, msg, original_index = response
#             log.error(f"Worker {worker_rank} failed on task {original_index}: {msg}")
#             # Propagate the error to be handled by the caller
#             raise RuntimeError(f"Worker {worker_rank} failed: {msg}")

#         result, original_index = response
#         results[original_index] = result
#         tasks_in_progress -= 1
        
#         # If there are more tasks, send one to the newly freed worker
#         if task_index < n_tasks:
#             payload = (tasks[task_index], task_index)
#             self.comm.send((task_type, payload), dest=worker_rank)
#             task_index += 1
#             tasks_in_progress += 1
    
#     return results