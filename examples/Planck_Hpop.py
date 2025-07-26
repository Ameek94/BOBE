from jaxbo.bo import BOBE
from jaxbo.bo_utils import plot_final_samples
from jaxbo.loglike import cobaya_likelihood
import time

cobaya_input_file = './cosmo_input/LCDM_new_CMB.yaml'

likelihood = cobaya_likelihood(cobaya_input_file, confidence_for_unbounded=0.9999995,
        minus_inf=-1e6, noise_std=0.0,name='Planck_Hpop')

start = time.time()
sampler = BOBE(n_cobaya_init=8, n_sobol_init = 256, 
        miniters=500, maxiters=4000,max_gp_size=2000,
        loglikelihood=likelihood,
        resume=False,
        resume_file=f'{likelihood.name}.npz',
        save=True,
        fit_step = 40, update_mc_step = 15, ns_step = 100,
        num_hmc_warmup = 512,num_hmc_samples = 512, mc_points_size = 64,
        lengthscale_priors='DSLP',logz_threshold=5.,
        use_svm=True,svm_use_size=500,svm_update_step=4,minus_inf=-1e6,svm_threshold=300,svm_gp_threshold=10000)

gp, ns_samples, logz_dict = sampler.run()
end = time.time()
print(f"Total time taken = {end-start:.4f} seconds")


param_list_LCDM = ['omch2','logA','ns','H0','ombh2','tau']
plot_final_samples(gp, ns_samples,param_list=sampler.param_list,param_bounds=sampler.param_bounds,
                   plot_params=param_list_LCDM,
                   param_labels=sampler.param_labels,output_file=f'{likelihood.name}',
                   reference_file='./cosmo_input/chains/Planck_DESI_LCDM_pchord',reference_ignore_rows=0.0,
                   reference_label='PolyChord',scatter_points=False,)

# 2025-04-21 18:27:42,039 INFO:[BO]:  Final LogZ: upper=-5527.4084, mean=-5529.4980, lower=-5530.0967, dlogz sampler=0.1720
# PolyChord result: # log-evidence
# logZ: -5529.65218118231
# logZstd: 0.447056743748251