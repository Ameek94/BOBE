from jaxbo.bo import BOBE
from jaxbo.utils import plot_final_samples
from jaxbo.loglike import CobayaLikelihood
import time
import sys

cobaya_input_file = './cosmo_input/LCDM_Planck_DESI_omk.yaml'


clf_type = str(sys.argv[1]) if len(sys.argv) > 1 else 'svm'

likelihood = CobayaLikelihood(cobaya_input_file, confidence_for_unbounded=0.9999995,
        minus_inf=-1e5, noise_std=0.0,name=f'Planck_DESI_Omk_{clf_type}')

start = time.time()
sampler = BOBE(n_cobaya_init=32, n_sobol_init = 128, 
        miniters=750, maxiters=2000,max_gp_size=1800,
        loglikelihood=likelihood,
        resume=False,
        resume_file=f'{likelihood.name}.npz',
        save=True,
        fit_step = 50, update_mc_step = 5, ns_step = 50,
        num_hmc_warmup = 512,num_hmc_samples = 2048, mc_points_size = 64,
        lengthscale_priors='DSLP',logz_threshold=1.,
        use_clf=True,clf_use_size=200,clf_update_step=1,minus_inf=-1e5,
        clf_threshold=350,gp_threshold=5000.) #prev 5000

gp, ns_samples, logz_dict = sampler.run()
end = time.time()
print(f"Total time taken = {end-start:.4f} seconds")


param_list_LCDM = ['omk','omch2','logA','ns','H0','ombh2','tau']
plot_final_samples(gp, ns_samples,param_list=likelihood.param_list,param_bounds=likelihood.param_bounds,
                   plot_params=param_list_LCDM,
                   param_labels=likelihood.param_labels,output_file=f'{likelihood.name}',
                   reference_file='./cosmo_input/chains/Planck_DESI_LCDM_omk',reference_ignore_rows=0.3,
                   reference_label='MCMC',scatter_points=True,)

# 2025-04-21 18:27:42,039 INFO:[BO]:  Final LogZ: upper=-5527.4084, mean=-5529.4980, lower=-5530.0967, dlogz sampler=0.1720
# PolyChord result: # log-evidence
# logZ: -5529.65218118231
# logZstd: 0.447056743748251