from .bo import BOBE
from .loglike import CobayaLikelihood, ExternalLikelihood
from .utils.pool import MPI_Pool
from .utils.logging_utils import setup_logging, get_logger
from typing import Union, Callable, Dict, Any, Optional
    

def run_bobe(likelihood: Union[Callable,str], 
             likelihood_kwargs: Dict[str, Any] = {},
             verbosity: str ='INFO', 
             log_file: Optional[str] = None, 
             **sampler_kwargs):
    """
    High-level wrapper to run the BOBE sampler.
    This function handles all MPI setup and logic.

    Arguments
    ----------
    likelihood: Union[Callable, str]
        The log-likelihood function or a string identifier for a Cobaya model.
    likelihood_kwargs: Dict[str, Any]
        Additional keyword arguments to pass to the likelihood. See loglike.py for details.
    verbosity: str
        The logging verbosity level.
    log_file: Optional[str]
        The path to a log file.
    sampler_kwargs:
        Additional keyword arguments to pass to the sampler. See bo.py for details.
    """

    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        # Use a raw print that cannot be misconfigured
        print(f"[SANITY CHECK] Hello from rank {rank}/{size} inside run_bobe.", flush=True)
    except Exception as e:
        print(f"[SANITY CHECK] Rank unknown. Error: {e}", flush=True)


    setup_logging(verbosity=verbosity,log_file=log_file)

    pool = MPI_Pool()

    # setup likelihood
    if isinstance(likelihood, Callable):
        likelihood = ExternalLikelihood(loglikelihood=likelihood,pool=pool,**likelihood_kwargs)
    elif isinstance(likelihood, str):
        likelihood = CobayaLikelihood(input_file_dict=likelihood,pool=pool,**likelihood_kwargs)


    likelihood.pool = pool

    if pool.is_master():

        print(f"Rank {pool.rank} running BOBE with likelihood: {likelihood.name}")
        # here should setup default arguments for all necessary parameters
        n_log_ei_iters = sampler_kwargs.pop('n_log_ei_iters', 20)

        # Master creates the sampler and runs it
        sampler = BOBE(
            loglikelihood=likelihood,
            pool=pool,
            **sampler_kwargs
        )
        results = sampler.run(n_log_ei_iters=n_log_ei_iters)
        pool.close()
        return results
    else:
        print(f"Rank {pool.rank} running only likelihood evaluations")
        # Workers wait for tasks
        pool.worker_wait(likelihood)
        return None
