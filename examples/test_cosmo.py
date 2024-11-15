import sys
sys.path.append('../')
from bo_be import sampler

max_steps = 1
nstart = 4
cobaya_init = 4
ntot = max_steps + nstart + cobaya_init
save_file = 'cmb_bao'
input_file = './cosmo_input/LCDM_6D.yaml' #'./cosmo_input/LCDM_Planck_DESI.yaml'  #

cosmo = sampler(cobaya_model=True,input_file=input_file,cobaya_start=cobaya_init,seed=200000,
                max_steps=max_steps, nstart=nstart,mc_points_size=32,acq_goal=5e-7,noise=1e-6,
                save=True,save_file=save_file)

cosmo.run()