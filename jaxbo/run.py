from .bo import BOBE
from .loglike import BaseLikelihood
from .utils.pool import MPI_Pool
    

def run_bobe(likelihood: BaseLikelihood, **kwargs):
    """
    High-level wrapper to run the BOBE sampler.
    This function handles all MPI setup and logic.
    """
    pool = MPI_Pool()
    likelihood.pool = pool

    if pool.is_master():
        # Master creates the sampler and runs it
        sampler = BOBE(
            loglikelihood=likelihood,
            pool=pool,
            **kwargs
        )
        results = sampler.run(n_log_ei_iters=4)
        pool.close()
        return results
    else:
        # Workers enter the listening loop
        pool.worker_listen(likelihood)
        return None
