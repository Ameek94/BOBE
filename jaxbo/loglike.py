# Class for implementing external loglikelihoods
from .utils.core_utils import scale_to_unit, scale_from_unit
from typing import Any, Callable, List,Optional, Tuple, Union, Dict
import numpy as np
from functools import partial
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from scipy.stats import qmc
from .utils.logging_utils import get_logger
from .utils.seed_utils import get_numpy_rng
from .utils.pool import MPI_Pool
import logging
import numpy as np
import tqdm
from typing import Any, Callable, List, Optional, Union, Dict
from .utils.core_utils import scale_from_unit
from .utils.logging_utils import get_logger

log = get_logger("[likelihood]")


def _loglike_worker(payload: tuple) -> float:
    """
    Worker function is now simpler: it only expects the function and a point.
    """

    logp_func, minus_inf_val, x = payload
    
    try:
        # The logp_func already has its args/kwargs baked in
        val = float(logp_func(x)) 
    except Exception:
        return minus_inf_val

    if np.isnan(val) or np.isinf(val) or val < minus_inf_val:
        return minus_inf_val
    return val

class BaseLikelihood:
    """Class for log-likelihoods with common evaluation logic."""

    def __init__(self,
                 loglikelihood: Callable,
                 logp_args: Optional[Tuple[Any, ...]] = None,
                 logp_kwargs: Optional[dict] = None,
                 ndim: int = 1,
                 param_list: Optional[List[str]] = None,
                 param_labels: Optional[List[str]] = None,
                 param_bounds: Optional[Union[List, np.ndarray]] = None,
                 noise_std: float = 0.,
                 name: Optional[str] = None,
                 minus_inf: float = -1e5,
                 pool: MPI_Pool = None):
        
        if logp_args or logp_kwargs:
            self.logp = partial(loglikelihood, *(logp_args or ()), **(logp_kwargs or {}))
        else:
            self.logp = loglikelihood

        self.ndim = int(ndim)
        self.param_list = param_list if param_list is not None else [f"x_{i+1}" for i in range(ndim)]
        self.param_labels = param_labels if param_labels is not None else [f"x_{{{i+1}}}" for i in range(ndim)]
        self.param_bounds = np.array(param_bounds) if param_bounds is not None else np.array(ndim * [[0, 1]]).T
        self.noise_std = noise_std
        self.name = name or "loglikelihood"
        self.minus_inf = minus_inf
        self.logprior_vol = np.log(np.prod(self.param_bounds[1] - self.param_bounds[0]))
        self.pool = pool if pool is not None else MPI_Pool()

        log.info(f"Initialized {self.name} with {self.ndim} params")

    def __call__(self, X: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Evaluate log-likelihood at one or more points X.
        """
        X = np.atleast_2d(X)
        if X.shape[1] != self.ndim:
            raise ValueError(f"Input shape {X.shape} does not match ndim {self.ndim}")

        tasks = [(self.logp, self.minus_inf, point) for point in X]
        
        vals = self.pool.map(_loglike_worker, tasks)

        vals = np.array(vals).reshape(X.shape[0], 1)
        return vals

    def get_initial_points(self, n_sobol_init=8, rng=None):
        """Sobol initialization by default, can be overridden in subclasses."""
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=self.ndim, scramble=True, rng=rng).random(n_sobol_init)
        sobol_points = scale_from_unit(sobol, self.param_bounds)
        vals = self.__call__(sobol_points)
        return sobol_points, vals

class ExternalLikelihood(BaseLikelihood):
    """Wrapper around a user-provided log-likelihood function."""

    def __init__(self, loglikelihood: Callable, ndim: int, pool: MPI_Pool, **kwargs):
        super().__init__(loglikelihood=loglikelihood, ndim=ndim, pool=pool, **kwargs)

class CobayaLikelihood(BaseLikelihood):
    """Likelihood wrapper for Cobaya models."""

    def __init__(self,
                 input_file_dict: Union[str, Dict[str, Any]],
                 confidence_for_unbounded: float = 0.9999995,
                 noise_std: float = 0,
                 minus_inf: float = -1e5,
                 name: str = "cobaya_model",
                 pool: MPI_Pool = None):

        if isinstance(input_file_dict, str):
            info = yaml_load(input_file_dict)
        else:
            info = input_file_dict

        cobaya_model = get_model(info)

        # Silence cobaya root logger
        rootlogger = logging.getLogger()
        if rootlogger.handlers:
            rootlogger.handlers.clear()

        param_list = list(cobaya_model.parameterization.sampled_params())
        param_bounds = np.array(
            cobaya_model.prior.bounds(confidence_for_unbounded=confidence_for_unbounded)
        ).T
        param_labels = [cobaya_model.parameterization.labels()[k] for k in param_list]
        ndim = len(param_list)

        def cobaya_logp(x):
            return cobaya_model.logpost(x, make_finite=False)

        super().__init__(loglikelihood=cobaya_logp,
                         ndim=ndim,
                         param_list=param_list,
                         param_labels=param_labels,
                         param_bounds=param_bounds,
                         noise_std=noise_std,
                         name=name,
                         minus_inf=minus_inf,
                         pool=pool)

        self.cobaya_model = cobaya_model

    def __call__(self, X, *args, **kwargs):
        vals = super().__call__(X, *args, **kwargs)
        vals = np.where(vals<=self.minus_inf, self.minus_inf, vals + self.logprior_vol)
        return vals

    def get_initial_points(self, n_cobaya_init=4, n_sobol_init=16, rng=None):
        points, logpost = [], []

        for _ in range(n_cobaya_init):
            pt, res = self.cobaya_model.get_valid_point(100, ignore_fixed_ref=False,
                                                        logposterior_as_dict=True, random_state=rng)
            points.append(pt)
            lp = res["logpost"]
            logpost.append(self.minus_inf if lp < self.minus_inf else lp + self.logprior_vol)

        sobol_points, sobol_vals = super().get_initial_points(n_sobol_init, rng=rng)
        points.extend(sobol_points)
        logpost.extend(sobol_vals.flatten())

        return np.array(points), np.array(logpost).reshape(len(points), 1)
