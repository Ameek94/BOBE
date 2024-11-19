import sys
sys.path.append('../')
from bo_be import sampler



cobaya_init = 8
settings_file = './input_example.yaml'
cobaya_input_file = './cosmo_input/LCDM_6D.yaml'

cosmo = sampler(settings_file=settings_file,
                cobaya_model=True,
                cobaya_input_file=cobaya_input_file,
                cobaya_start=cobaya_init,
                seed=10,)
cosmo.run()