from jaxbo.bo import BOBE
from jaxbo.utils.summary_plots import plot_final_samples
from jaxbo.loglike import CobayaLikelihood
import time

cobaya_input_file = './cosmo_input/LCDM_Planck_DESI.yaml'

likelihood = CobayaLikelihood(cobaya_input_file, confidence_for_unbounded=0.9999995,
        minus_inf=-1e5, noise_std=0.0,name='Planck_Camspec_nn')

start = time.time()
sampler = BOBE(n_cobaya_init=4, n_sobol_init = 4, 
        miniters=600, maxiters=2500,max_gp_size=1500,
        loglikelihood=likelihood,
        resume=True,
        resume_file=f'{likelihood.name}.npz',
        save=True,
        fit_step = 50, update_mc_step = 5, ns_step = 50,
        num_hmc_warmup = 512,num_hmc_samples = 1024, mc_points_size = 64,
        lengthscale_priors='DSLP',
        use_clf=True,clf_type="nn",clf_use_size=10,clf_update_step=5,
        clf_threshold=500, gp_threshold=5000,
        minus_inf=-1e5,logz_threshold=5.)

results = sampler.run()
gp = results['gp']
ns_samples = results['samples']
end = time.time()
print(f"Total time taken = {end-start:.4f} seconds")


param_list_LCDM = ['omch2','logA','ns','H0','ombh2','tau']
plot_final_samples(gp, ns_samples,param_list=likelihood.param_list,param_bounds=likelihood.param_bounds,
                   plot_params=param_list_LCDM,
                   param_labels=likelihood.param_labels,output_file=f'{likelihood.name}',
                   reference_file='./cosmo_input/chains/Planck_DESI_LCDM_pchord',reference_ignore_rows=0.0,
                   reference_label='PolyChord',scatter_points=False,)

# 2025-04-21 18:27:42,039 INFO:[BO]:  Final LogZ: upper=-5527.4084, mean=-5529.4980, lower=-5530.0967, dlogz sampler=0.1720
# PolyChord result: # log-evidence
# logZ: -5529.65218118231
# logZstd: 0.447056743748251

# 2025-07-26 05:39:58,814 INFO:[BO]:  Final LogZ: mean=-5529.6915, dlogz sampler=0.1793, upper=-5509.8101, lower=-5530.1042
# Total time taken = 9495.8532 seconds
