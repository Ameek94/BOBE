# Class for implementing external loglikelihoods
from .utils.core_utils import scale_to_unit, scale_from_unit
from typing import Any, Callable, List,Optional, Tuple, Union, Dict
import numpy as np
from functools import partial
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

log = get_logger("likelihood")


class BaseLikelihood:
    """Class for log-likelihoods with common evaluation logic."""

    def __init__(self,
                 loglikelihood: Callable,
                 param_list: List[str],
                 logp_args: Optional[Tuple[Any, ...]] = None,
                 logp_kwargs: Optional[dict] = None,
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
        self.noise_std = noise_std
        self.name = name or "loglikelihood"
        self.minus_inf = minus_inf
        self.logprior_vol = np.log(np.prod(self.param_bounds[1] - self.param_bounds[0]))
        self.pool = pool if pool is not None else MPI_Pool()

        log.info(f"Initialized {self.name} with {self.ndim} params")

        if self.pool.is_master():
            log.info(f"Param list: {self.param_list}")
            log.info(f"Param lower bounds: {self.param_bounds[0]}")
            log.info(f"Param upper bounds: {self.param_bounds[1]}")

    def _safe_eval(self, x: np.ndarray) -> float:
        """Helper method to safely evaluate a single point."""
        try:
            val = float(self.logp(x))
        except Exception:
            return self.minus_inf

        if np.isnan(val) or np.isinf(val) or val < self.minus_inf:
            return self.minus_inf
        return val

    def __call__(self, X: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        __call__ now performs direct evaluation. It handles both
        a single point (from a worker) and a batch of points (in a serial run).
        """
        points = np.atleast_2d(X)
        if points.shape[1] != self.ndim:
            raise ValueError(f"Input shape {points.shape} does not match ndim {self.ndim}")

        # In a serial run, it evaluates all points.
        # In an MPI run, a worker receives a single point, so it evaluates one.
        vals = [self._safe_eval(p) for p in points]

        return np.array(vals).reshape(len(vals), 1)

    def get_initial_points(self, n_sobol_init=8, rng=None):
        """
        Get initial points for the optimization process using Sobol quasi-random sampling.
        """
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=self.ndim, scramble=True, rng=rng).random(n_sobol_init)
        sobol_points = scale_from_unit(sobol, self.param_bounds)
        
        # The master process calls this method. It then calls pool.run_map(self, sobol_points)
        # to get the values, which works in both modes. Initial points are always generated on the master when going through run.py.
        
        if self.pool.is_master():
             log.info(f"Evaluating Likelihood at initial points")
             vals = self.pool.run_map(self.__call__, sobol_points)
        else:
            # Workers don't generate initial points
            vals = np.empty((n_sobol_init, 1))

        return sobol_points, vals


class ExternalLikelihood(BaseLikelihood):
    """Wrapper around a user-provided log-likelihood function."""

    def __init__(self, loglikelihood: Callable, pool: MPI_Pool, **kwargs):
        super().__init__(loglikelihood=loglikelihood, pool=pool, **kwargs)

class CobayaLikelihood(BaseLikelihood):
    """Likelihood wrapper for Cobaya models."""

    def __init__(self,
                 input_file_dict: Union[str, Dict[str, Any]],
                 confidence_for_unbounded: float = 0.9999995,
                 noise_std: float = 0,
                 minus_inf: float = -1e5,
                 name: str = "cobaya_model",
                 pool: MPI_Pool = None):
        
        try:
            from cobaya.yaml import yaml_load
            from cobaya.model import get_model
        except ImportError:
            log.error("Cobaya is not installed.")
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
                         noise_std=noise_std,
                         name=name,
                         minus_inf=minus_inf,
                         pool=pool)

        self.cobaya_model = cobaya_model
        log.info(f"Logprior volume = {self.logprior_vol:.4f}")

    def __call__(self, X) -> np.ndarray:
        # Calls the parent's now-functional
        # __call__ method and then applies its specific corrections.
        vals = super().__call__(X)
        vals = np.where(vals <= self.minus_inf, self.minus_inf, vals + self.logprior_vol) # add logprior volume  
        return vals

    def get_initial_points(self, n_cobaya_init=4, n_sobol_init=16, rng=None):
        # Can do further parallelization here by getting valid points from the pool.
        points, logpost = [], []
        if self.pool.is_master():
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
        else:
            # Workers return empty arrays of the correct shape to avoid errors
            return np.empty((n_cobaya_init + n_sobol_init, self.ndim)), np.empty((n_cobaya_init + n_sobol_init, 1))
