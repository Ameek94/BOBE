from .bo import BOBE
from .likelihood import BaseLikelihood, CobayaLikelihood, ExternalLikelihood
from .utils.pool import MPI_Pool
from .utils.logging_utils import setup_logging, get_logger
from typing import Union, Callable, Dict, Any, Optional
    

def run_bobe(likelihood: Union[Callable, str], 
             likelihood_kwargs: Dict[str, Any] = {},
             gp_kwargs: Dict[str, Any] = {},
             acq = 'wipv',
             use_clf: bool = False,
             clf_type: str = 'svm',
             clf_nsigma_threshold: float = 20,
             verbosity: str ='INFO', 
             log_file: Optional[str] = None, 
             min_evals: int = 100,
             max_evals: int = 1000,
             max_gp_size: int = 1200,
             optimizer='scipy',
             fit_step: int = 5,
             ns_step: int = 5,
             wipv_batch_size: int = 5,
             zeta_ei: float = 0.1,
             resume: bool = False,
             resume_file: Optional[str] = None,
             convergence_n_iters=1,
             **sampler_kwargs):
    """
    High-level wrapper to run the BOBE sampler.
    This function handles all MPI setup and logic.

    Arguments
    ----------
    likelihood: Union[BaseLikelihood, str]
        The log-likelihood object which inherits from the BaseLikelihood class or a string for a Cobaya model yaml file.
    likelihood_kwargs: Dict[str, Any]
        Additional keyword arguments to pass to the likelihood. See loglike.py for details.
    gp_kwargs: Dict[str, Any]
        Additional keyword arguments to pass to the GP constructor. These can include:
        - lengthscale_priors: str, choice of GP type ('DSLP', 'SAAS', 'uniform'), defaults to 'DSLP'
        - noise: Noise parameter for GP (float, default: 1e-8)
        - kernel: Kernel type ('rbf', 'matern', etc., default: 'rbf')
        - optimizer: Optimizer type ('optax', 'scipy', default: 'optax')
        - optimizer_kwargs: Dict for optimizer settings (e.g., {'lr': 1e-3, 'name': 'adam'})
        - kernel_variance_bounds: List of [lower, upper] bounds for kernel variance
        - lengthscale_bounds: List of [lower, upper] bounds for lengthscales  
        - lengthscales: Initial lengthscale values (array-like)
        - kernel_variance: Initial kernel variance value (float)
    verbosity: str
        The logging verbosity level.
    log_file: Optional[str]
        The path to a log file.
    sampler_kwargs:
        Additional keyword arguments to pass to the sampler. These can include:
        - optimizer: str, optimizer type ('optax', 'scipy', default: 'optax') - affects both GP and acquisition optimization
        - All other BOBE constructor parameters. See bo.py for details.
    """


    setup_logging(verbosity=verbosity,log_file=log_file)

    pool = MPI_Pool()

    # setup likelihood
    if isinstance(likelihood, Callable):
        My_Likelihood = ExternalLikelihood(loglikelihood=likelihood, pool=pool, **likelihood_kwargs)
    elif isinstance(likelihood, str):
        My_Likelihood = CobayaLikelihood(input_file_dict=likelihood,pool=pool,**likelihood_kwargs)

    My_Likelihood.pool = pool

    if pool.is_master:

        print(f"Rank {pool.rank} running BOBE with likelihood: {My_Likelihood.name}")
        # here should setup default arguments for all necessary parameters

        # Master creates the sampler and runs it
        sampler = BOBE(
            loglikelihood=My_Likelihood,
            gp_kwargs=gp_kwargs,
            pool=pool,
            min_evals=min_evals,
            max_evals=max_evals,
            max_gp_size=max_gp_size,
            optimizer=optimizer,
            fit_step=fit_step,
            ns_step=ns_step,
            wipv_batch_size=wipv_batch_size,
            zeta_ei=zeta_ei,
            resume=resume,
            resume_file=resume_file,
            use_clf=use_clf,
            clf_type=clf_type,
            clf_nsigma_threshold=clf_nsigma_threshold,
            convergence_n_iters=convergence_n_iters,
            **sampler_kwargs
        )
        results = sampler.run(acq)
        pool.close()
        return results
    else:
        print(f"Rank {pool.rank} running only likelihood evaluations")
        # Workers wait for tasks
        pool.worker_wait(My_Likelihood)
        return None