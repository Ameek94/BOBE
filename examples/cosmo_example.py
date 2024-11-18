import sys
sys.path.append('../')
from bo_be import sampler

max_steps = 1
nstart = 2 
cobaya_init = 0
ntot = max_steps + nstart + cobaya_init
save_file = 'test'
settings_file = './input_example.yaml'
cobaya_input_file = './cosmo_input/LCDM_2D.yaml'

cosmo = sampler(settings_file=settings_file,
                cobaya_model=True,
                cobaya_input_file=cobaya_input_file,
                cobaya_start=cobaya_init,
                seed=20,)
cosmo.run()