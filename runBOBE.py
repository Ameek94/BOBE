import sys
import os
num_devices = 12
os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={num_devices}"
import matplotlib
import numpy as np
#sys.path.append('../')
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
from bo_be import sampler 
from ext_loglike import external_loglike
matplotlib.rc('font', size=16,family='serif')
matplotlib.rc('legend', fontsize=16)
ndim = int(sys.argv[1])

def gaussian(X):
    # X is a N x DIM shaped tensor, output is N tensor
    mean = np.array(0.5) #len(param_list)*
    sigma = np.array(0.1) #len(param_list)*
    return np.expand_dims((-0.5*np.sum((X-mean)**2/sigma**2, axis=-1, keepdims=False)), -1)

loglike = external_loglike(gaussian, ndim, name="Gaussian")

settings_file='testing_settings.yaml'
BOBE = sampler(ndim=ndim,
               cobaya_model=False,
               settings_file=settings_file,
               objfun=loglike)
BOBE.run()