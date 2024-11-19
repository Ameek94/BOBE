# external loglike baseclass


from re import X
from typing import Any, List,Optional, Tuple
import numpy as np

class external_loglike:

    def __init__(self
                 ,ndim: int
                 ,param_list: Optional[list[str]] # ndim length
                 ,param_labels: Optional[list[str]] # ndim length
                 ,param_bounds: Optional[list] # 2 x ndim shaped
                 ,noise_std: float = 0.
                 ,name: Optional[str] = None
                 ,) -> None:
        
        self.ndim = int(ndim)
        self.param_list = param_list if param_list is not None else ['x_%i'%(i+1) for i in range(ndim)] # type: ignore
        assert len(self.param_list)==ndim
        self.param_labels = param_labels if param_labels is not None else ['x_%i'%(i+1) for i in range(ndim)] # type: ignore
        assert len(self.param_list)==ndim
        self.param_bounds = np.array(param_bounds) if param_bounds is not None else np.array(ndim*[[0,1]]).T # type: ignore
        assert len(self.param_bounds.T)==ndim

        self.noise_std = noise_std

        self.name = name if name is not None else 'test_function'

    def __call__(self, x) -> Any:
        noise = self.noise_std*np.random.randn()
        val = self.logposterior(x)
        return val + noise
    
    def logposterior(self,x):
        raise NotImplementedError
        

    
# some examples

from scipy.stats import multivariate_normal

class ndgaussian(external_loglike):
    def __init__(
        self,
        ndim: int = 1,
        param_list: Optional[list[str]] = None,
        param_labels: Optional[list[str]] = None,
        param_bounds: Optional[list] = None,
        means: Optional[np.ndarray] = None,
        cov: Optional[np.ndarray] = None,
        noise_std: Optional[float] = 0,
    ) -> None:
        
        super().__init__(ndim=ndim
                         ,param_list=param_list
                         ,param_bounds=param_bounds
                         ,param_labels=param_labels
                         ,name="MultivariateGaussian"
                         )  
        
        if means is None:
            self.means = np.zeros(self.ndim)
        else:
            self.means = means
        if cov is None:
            self.cov = np.eye(self.ndim)
        else:
            self.cov = cov
        self.dist = multivariate_normal(mean=self.means,cov=self.cov) # type: ignore
    
    def logposterior(self,x):
        return self.dist.logpdf(x)
    

class banana(external_loglike):
    

    def __init__(
            self,
            ndim: int = 2,
            param_list: Optional[list[str]] = None,
            param_labels: Optional[list[str]] = None,
            param_bounds: Optional[list] = None,
            noise_std: float = 0,
        ) -> None:
            
        super().__init__(ndim=ndim
                         ,param_list=param_list
                         ,param_bounds=param_bounds
                         ,param_labels=param_labels
                         ,name="Banana"
                         ,noise_std=noise_std)      
            
    def evaluate_true(self, x):
        logpdf = -0.25*(5*(0.2-x[0]))**2 - (20*(x[1]/4 - x[0]**4))**2
        return logpdf
