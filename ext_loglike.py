# external loglike baseclass


from typing import Any,Optional
import numpy as np

class external_loglike:

    def __init__(self
                 ,func # your logposterior/objective function f(x)
                 ,func_args
                 ,func_kwargs
                 ,ndim: int
                 ,param_list: Optional[list[str]]
                 ,param_labels: Optional[list[str]]
                 ,param_bounds: Optional[list]
                 ,noise_std: float = 0.
                 ,) -> None:
        
        self.ndim = int(ndim)
        self.param_list = param_list if param_list is not None else ['x_%i'%(i+1) for i in range(ndim)] # type: ignore
        assert len(self.param_list)==ndim
        self.param_labels = param_labels if param_labels is not None else ['x_%i'%(i+1) for i in range(ndim)] # type: ignore
        assert len(self.param_list)==ndim
        self.param_bounds = np.array(param_bounds) if param_bounds is not None else np.array(ndim*[[0,1]]).T # type: ignore
        assert len(self.param_bounds.T)==ndim

        self.obj = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs
        self.noise_std = noise_std

    def __call__(self, x) -> Any:
        noise = self.noise_std*np.random.randn()
        val = self.obj(x,*self.func_args,**self.func_kwargs)
        return val + noise
        

    
