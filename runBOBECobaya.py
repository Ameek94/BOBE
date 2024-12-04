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
#ndim = int(sys.argv[1])

settings_file='testing_settings.yaml'
cobaya_input_file='LCDM_6D.yaml'
BOBE = sampler(ndim=6,
               cobaya_model=True,
               cobaya_input_file=cobay_input_file,
               settings_file=settings_file)
BOBE.run()