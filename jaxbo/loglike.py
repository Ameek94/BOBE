# Class for implementing external loglikelihoods
from .utils.core_utils import scale_to_unit, scale_from_unit
from typing import Any, Callable, List,Optional, Tuple, Union, Dict
import numpy as np
import tqdm
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from scipy.stats import qmc
from .utils.logging_utils import get_logger
import logging
log = get_logger("loglike")


class ExternalLikelihood:

    def __init__(self
                 ,loglikelihood: Callable
                 ,ndim: int
                 ,param_list: Optional[list[str]] = None # ndim length
                 ,param_labels: Optional[list[str]] = None# ndim length
                 ,param_bounds: Optional[Union[list,np.ndarray]] = None # 2 x ndim shaped
                 ,noise_std: float = 0.
                 ,name: Optional[str] = None,
                 vectorized: bool = False,
                 minus_inf: float = -1e5,
                 ) -> None:
        
        self.logp = loglikelihood
        self.ndim = int(ndim)
        self.param_list = param_list if param_list is not None else ['x_%i'%(i+1) for i in range(ndim)] # type: ignore
        assert len(self.param_list)==ndim
        self.param_labels = param_labels if param_labels is not None else ['x_{%i}'%(i+1) for i in range(ndim)] # type: ignore
        assert len(self.param_list)==ndim
        self.param_bounds = np.array(param_bounds) if param_bounds is not None else np.array(ndim*[[0,1]]).T # type: ignore
        assert len(self.param_bounds.T)==ndim
        self.noise_std = noise_std
        self.name = name if name is not None else 'loglikelihood'
        self.vectorized = vectorized
        self.minus_inf = minus_inf
        self.logprior_vol = np.log(np.prod(param_bounds[1] - param_bounds[0]))

        log.info(f"Loglikelihood {self.name} initialized with params : {self.param_list},  bounds: \n{self.param_bounds.T} \nand labels: \n{self.param_labels}")

    def __call__(self, x: Union[np.ndarray, List[float]]
                 ,logp_args: tuple = ()
                 ,logp_kwargs: dict[str,Any] = {}) -> Union[np.ndarray, float]:
        x = np.atleast_2d(x)
        if x.shape[1] != self.ndim:
                raise ValueError(f"Input shape {x.shape} does not match ndim {self.ndim}")
        if self.vectorized:
            vals = self.logp(x,*logp_args,**logp_kwargs)
        else:
            r = np.arange(x.shape[0])
            progress_bar = tqdm.tqdm(r, desc="Evaluating Loglikelihood", leave=False)
            vals = []
            for i in progress_bar:
                vals.append(self.logp(x[i],*logp_args,**logp_kwargs))
            vals = np.array(vals)
        # change nans and infs to minus_inf
        vals = np.where(np.isnan(vals),self.minus_inf,vals)
        vals = np.where(np.isinf(vals),self.minus_inf,vals)
        vals = np.where(vals<self.minus_inf,self.minus_inf,vals)
        vals = np.reshape(vals,(x.shape[0],1))

        # add noise if specified
        noise = self.noise_std * np.random.randn(x.shape[0],1)

        return vals + noise
        
    def get_initial_points(self,n_init_sobol=8,n_cobaya_init=0,rng=None):
        points, logpost = [], []        
        r = np.arange(n_init_sobol)
        progress_bar = tqdm.tqdm(r, desc="Evaluating Sobol points")
        sobol = qmc.Sobol(d=self.ndim, scramble=True, seed=rng).random(n_init_sobol)
        sobol_points = scale_from_unit(sobol,self.param_bounds)
        for i in progress_bar:
            pt = sobol_points[i]
            lp = self.logp(pt)
            logpost.append(self.minus_inf if lp < self.minus_inf else lp)
            points.append(pt)
        return np.array(points), np.array(logpost).reshape(n_init_sobol, 1)


class CobayaLikelihood(ExternalLikelihood):
    """
    Class for implementing external loglikelihoods using cobaya
    """

    def __init__(self, 
                 input_file_dict: str| Dict[str, Any],
                 confidence_for_unbounded: float = 0.9999995,
                 noise_std: float = 0,
                 minus_inf: float = -1e5,
                 name: str | None = 'cobaya_model') -> None:
        if isinstance(input_file_dict, str):
            info = yaml_load(input_file_dict)
        elif isinstance(input_file_dict, dict):
            info = input_file_dict
        self.cobaya_model = get_model(info)
        rootlogger = logging.getLogger() 
        rootlogger.handlers.pop()    
        # Parameter setup
        param_list = list(self.cobaya_model.parameterization.sampled_params())
        param_bounds = np.array(
            self.cobaya_model.prior.bounds(confidence_for_unbounded=confidence_for_unbounded)
        ).T
        self.logprior_vol = np.log(np.prod(param_bounds[1] - param_bounds[0]))
        param_labels = [
            self.cobaya_model.parameterization.labels()[k] for k in param_list
        ]
        ndim = len(param_list)

        loglikelihood = lambda x: self.cobaya_model.logpost(x, make_finite=True)

        super().__init__(loglikelihood=loglikelihood, 
                         ndim=ndim, 
                         param_list=param_list, param_labels=param_labels, param_bounds=param_bounds, 
                         noise_std=noise_std, name=name, vectorized=False, minus_inf=minus_inf)

    def __call__(self, x, *logp_args, **logp_kwargs) -> np.ndarray:
        x = np.atleast_2d(x)
        if x.shape[1] != self.ndim:
            raise ValueError(f"Input shape {x.shape} does not match ndim {self.ndim}")
        r = np.arange(x.shape[0])
        progress_bar = tqdm.tqdm(r, desc="Evaluating Loglikelihood", leave=False)
        vals = []
        for i in progress_bar:
            pt = x[i]
            lp = self.cobaya_model.logpost(pt,make_finite=False)
            vals.append(lp)
        vals = np.array(vals)
        vals = np.where(np.isnan(vals),self.minus_inf,vals)
        vals = np.where(np.isinf(vals),self.minus_inf,vals)
        vals = np.where(vals<self.minus_inf,self.minus_inf,vals)
        vals = np.reshape(vals,(x.shape[0],1))
        return vals + self.logprior_vol

    def get_initial_points(self, n_cobaya_init=4,n_init_sobol=16,rng=None):
        points, logpost = [], []
        progress_bar = tqdm.tqdm(range(n_cobaya_init), desc='Evaluating Cobaya reference points')
        for _ in progress_bar:
            pt, res = self.cobaya_model.get_valid_point(100, ignore_fixed_ref=False,
                                               logposterior_as_dict=True)
            points.append(pt)
            lp = res['logpost']
            logpost.append(self.minus_inf if lp < self.minus_inf else lp + self.logprior_vol)
        
        r = np.arange(n_init_sobol)
        progress_bar = tqdm.tqdm(r, desc="Evaluating Sobol points")
        sobol = qmc.Sobol(d=self.ndim, scramble=True,seed=rng).random(n_init_sobol)
        sobol_points = scale_from_unit(sobol,self.param_bounds)
        for i in progress_bar:
            pt = sobol_points[i]
            lp = self.cobaya_model.logpost(pt, make_finite=True)
            logpost.append(self.minus_inf if lp < self.minus_inf else lp + self.logprior_vol)
            points.append(pt)
        
        return np.array(points), np.array(logpost).reshape(n_cobaya_init+n_init_sobol, 1)