import sys
import os
num_devices = int(sys.argv[1])
os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={num_devices}"
sys.path.append('../')
from bo_be import sampler
import jax
print(f"Num deivecs = {jax.device_count()}")
from nested_sampler import nested_sampling_jaxns



cobaya_init = 18
settings_file = './cosmo_full.yaml'
cobaya_input_file = './cosmo_input/LCDM_Planck_DESI.yaml'

cosmo = sampler(settings_file=settings_file,
                cobaya_model=True,
                cobaya_input_file=cobaya_input_file,
                cobaya_start=cobaya_init,
                seed=205,)
cosmo.run()

