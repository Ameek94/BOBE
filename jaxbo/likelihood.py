# Class for implementing external loglikelihoods
import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
from functools import partial
from scipy.stats import qmc

from .utils.core import scale_to_unit, scale_from_unit
from .utils.log import get_logger
from .utils.pool import MPI_Pool

log = get_logger("likelihood")


class Likelihood:
    """
    Base class for log-likelihoods with common evaluation logic.
    
    Parameters
    ----------
    loglikelihood : callable
        Log-likelihood function that takes parameter array and returns float.
    param_list : list of str
        List of parameter names.
    logl_args : tuple, optional
        Additional positional arguments for the likelihood function. Default is None.
    logl_kwargs : dict, optional
        Additional keyword arguments for the likelihood function. Default is None.
    param_labels : list of str, optional
        LaTeX labels for parameters. Default is None.
    param_bounds : array-like, optional
        Parameter bounds, shape (2, ndim). Default is None (unit cube).
    name : str, optional
        Name for this likelihood. Default is "loglikelihood".
    minus_inf : float, optional
        Value to return for failed evaluations. Default is -1e5.
    pool : MPI_Pool, optional
        MPI pool for parallel evaluation. Default is None.
    """

    def __init__(self,
                 loglikelihood: Callable,
                 param_list: List[str],
                 logl_args: Optional[Tuple[Any, ...]] = None,
                 logl_kwargs: Optional[dict] = None,
                 param_labels: Optional[List[str]] = None,
                 param_bounds: Optional[Union[List, np.ndarray]] = None,
                 name: Optional[str] = None,
                 minus_inf: float = -1e5,
                 pool: MPI_Pool = None):
        
        if logl_args or logl_kwargs:
            self.logp = partial(loglikelihood, *(logl_args or ()), **(logl_kwargs or {}))
        else:
            self.logp = loglikelihood

        # check param list in correct format
        if not all(isinstance(p, str) for p in param_list):
            raise ValueError("All elements of param_list must be strings corresponding to parameter names.")

        self.ndim = len(param_list)
        self.param_list = param_list if param_list is not None else [f"x_{i+1}" for i in range(self.ndim)]
        self.param_labels = param_labels if param_labels is not None else [f"x_{{{i+1}}}" for i in range(self.ndim)]
        if param_bounds is None:
            self.param_bounds = np.array(self.ndim * [[0, 1]]).T
            log.warning("No param_bounds provided. Assuming unit cube [0,1] for all parameters.")
        else:
            self.param_bounds = np.array(param_bounds)
        self.name = name or "loglikelihood"
        self.minus_inf = minus_inf
        self.logprior_vol = np.log(np.prod(self.param_bounds[1] - self.param_bounds[0]))
        self.pool = pool if pool is not None else MPI_Pool()

        log.info(f"Initialized {self.name} with {self.ndim} params")

        if self.pool.is_master:
            log.info(f"Param list: {self.param_list}")
            log.info(f"Param lower bounds: {self.param_bounds[0]}")
            log.info(f"Param upper bounds: {self.param_bounds[1]}")

    def _safe_eval(self, x: np.ndarray) -> float:
        """
        Safely evaluate log-likelihood at a single point.
        
        Parameters
        ----------
        x : np.ndarray
            Parameter vector to evaluate.
            
        Returns
        -------
        float
            Log-likelihood value, or minus_inf if evaluation fails.
        """
        try:
            val = float(self.logp(x))
        except Exception:
            log.debug(f"Log-likelihood evaluation failed at point {x}", exc_info=True)
            return self.minus_inf

        if np.isnan(val) or np.isinf(val) or val < self.minus_inf:
            return self.minus_inf
        return val

    def __call__(self, X: Union[np.ndarray, List[float]]) -> float:
        """
        Evaluate the likelihood function at a single point.
        
        This method is designed to be called by workers in MPI mode or
        by pool.run_map_objective in serial mode, which handles iteration.
        
        Parameters
        ----------
        X : array-like, shape (ndim,) or (1, ndim)
            Single input parameter vector.
        
        Returns
        -------
        val : float
            Log-likelihood value at the input point.
        """
        X = np.atleast_1d(X)
        if X.ndim > 1:
            if X.shape[0] != 1:
                raise ValueError(
                    "__call__ expects a single point. "
                    "Use pool.run_map_objective for batch evaluations."
                )
            X = X.flatten()
        
        if X.shape[0] != self.ndim:
            raise ValueError(f"Input shape {X.shape} does not match ndim {self.ndim}")
        
        return self._safe_eval(X)

    def generate_initial_points(self, n_sobol_init=8, rng=None) -> np.ndarray:
        """
        Generate initial points using Sobol quasi-random sampling (without evaluation).
        
        Parameters
        ----------
        n_sobol_init : int, optional
            Number of Sobol points to generate. Minimum is 2. Default is 8.
        rng : np.random.Generator, optional
            Random number generator. Default is None.
            
        Returns
        -------
        points : np.ndarray
            Initial points in parameter space, shape (n_sobol_init, ndim).
        """
        if rng is None:
            rng = np.random.default_rng()
        
        n_sobol_init = max(2, n_sobol_init)  # Ensure at least 2 points for Sobol

        sobol = qmc.Sobol(d=self.ndim, scramble=True, rng=rng).random(n_sobol_init)
        sobol_points = scale_from_unit(sobol, self.param_bounds)
        
        return np.array(sobol_points)


class CobayaLikelihood(Likelihood):
    """Likelihood wrapper for Cobaya models."""

    def __init__(self,
                 input_file_dict: Union[str, Dict[str, Any]],
                 confidence_for_unbounded: float = 0.9999995,
                 minus_inf: float = -1e5,
                 name: str = "cobaya_model",
                 pool: MPI_Pool = None):
        
        try:
            from cobaya.yaml import yaml_load
            from cobaya.model import get_model
        except ImportError:
            log.error("Cobaya is not installed.",exc_info=True)
            raise

        if isinstance(input_file_dict, str):
            info = yaml_load(input_file_dict)
        else:
            info = input_file_dict

        cobaya_model = get_model(info)

        param_list = list(cobaya_model.parameterization.sampled_params())
        param_bounds = np.array(
            cobaya_model.prior.bounds(confidence_for_unbounded=confidence_for_unbounded)
        ).T
        param_labels = [cobaya_model.parameterization.labels()[k] for k in param_list]

        def cobaya_logp(x):
            return cobaya_model.logpost(x, make_finite=False)

        super().__init__(loglikelihood=cobaya_logp,
                         param_list=param_list,
                         param_labels=param_labels,
                         param_bounds=param_bounds,
                         name=name,
                         minus_inf=minus_inf,
                         pool=pool)

        self.cobaya_model = cobaya_model
        log.info(f"Logprior volume = {self.logprior_vol:.4f}")

    def __call__(self, X) -> float:
        """Evaluate Cobaya likelihood with logprior volume correction."""
        val = super().__call__(X)
        if val <= self.minus_inf:
            return self.minus_inf
        return val + self.logprior_vol

    def _get_single_valid_point(self, rng: np.random.Generator) -> tuple:
        """
        WORKER METHOD: Gets a single valid point from the Cobaya model.
        This is the task that will be executed in parallel by each worker.
        """
        pt, res = self.cobaya_model.get_valid_point(
            max_tries=100, 
            ignore_fixed_ref=False,
            logposterior_as_dict=True, 
            random_state=rng
        )
        
        lp = res["logpost"]
        logpost = self.minus_inf if lp < self.minus_inf else lp + self.logprior_vol
        
        return (pt, logpost)

    def generate_cobaya_points(self, n_points, rng=None) -> List[Tuple]:
        """
        Generate points from Cobaya reference prior.
        
        Parameters
        ----------
        n_points : int
            Number of points to generate.
        rng : np.random.Generator, optional
            Random number generator.
            
        Returns
        -------
        results : List[Tuple]
            List of (point, logpost) tuples.
        """
        if n_points <= 0:
            return []
            
        if self.pool.is_mpi:
            log.info(f"Generating {n_points} Cobaya points in parallel...")
            return self.pool.get_cobaya_initial_points(n_points=n_points)
        else:
            log.info(f"Generating {n_points} Cobaya points in serial...")
            if rng is None:
                rng = np.random.default_rng()
            return [self._get_single_valid_point(rng=rng) for _ in range(n_points)]

