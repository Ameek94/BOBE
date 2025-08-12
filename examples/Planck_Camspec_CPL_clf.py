from jaxbo.bo import BOBE
from jaxbo.utils.summary_plots import plot_final_samples
from jaxbo.loglike import CobayaLikelihood
import time
import sys


clf = sys.argv[1] if len(sys.argv) > 1 else 'ellipsoid'

cobaya_input_file = './cosmo_input/LCDM_Planck_DESI_CPL.yaml'

likelihood = CobayaLikelihood(cobaya_input_file, confidence_for_unbounded=0.9999995,
        minus_inf=-1e5, noise_std=0.0,name=f'Planck_Camspec_CPL_{clf}')

if clf == 'svm':
    clf_update_step = 1
else:
    clf_update_step = 5

start = time.time()
sampler = BOBE(n_cobaya_init=32, n_sobol_init = 128, 
        miniters=0, maxiters=5000,max_gp_size=1900,
        loglikelihood=likelihood,
        resume=True,
        resume_file=f'{likelihood.name}.npz',
        save=True,
        fit_step = 50, update_mc_step = 5, ns_step = 50,
        num_hmc_warmup = 512,num_hmc_samples = 2048, mc_points_size = 96,
        lengthscale_priors='DSLP',logz_threshold=5.,clf_threshold=350,gp_threshold=5000,
        use_clf=True,clf_type=clf,clf_use_size=100,clf_update_step=clf_update_step,minus_inf=-1e5,)

# gp, ns_samples, logz_dict = sampler.run()
results = sampler.run()
gp = results['gp']
samples = results['samples']
logz_dict = results.get('logz', {})

end = time.time()
print(f"Total time taken = {end-start:.4f} seconds")

print("Creating parameter samples plot...")
if hasattr(samples, 'samples'):  # GetDist samples
    sample_array = samples.samples
    weights_array = samples.weights
else:  # Dictionary format
    sample_array = samples['x']
    weights_array = samples['weights']
plot_params = ['w','wa','omch2','logA','ns','H0','ombh2','tau'] #'omk'
plot_final_samples(gp, samples,param_list=likelihood.param_list,param_bounds=likelihood.param_bounds,
                   plot_params=plot_params,
                   param_labels=likelihood.param_labels,output_file=likelihood.name,
                   reference_file='./cosmo_input/chains/Planck_DESI_LCDM_CPL_pchord_flat',reference_ignore_rows=0.3,
                   reference_label='MCMC',scatter_points=False,)

# 2025-04-21 18:27:42,039 INFO:[BO]:  Final LogZ: upper=-5527.4084, mean=-5529.4980, lower=-5530.0967, dlogz sampler=0.1720
# PolyChord result: # log-evidence
# logZ: -5529.65218118231
# logZstd: 0.447056743748251