from jaxbo.bo import BOBE
from jaxbo.utils.summary_plots import plot_final_samples
from jaxbo.loglike import CobayaLikelihood
import time

cobaya_input_file = './cosmo_input/LCDM_6D.yaml'

likelihood = CobayaLikelihood(cobaya_input_file, confidence_for_unbounded=0.9999995,
        minus_inf=-1e5, noise_std=0.0,name='LCDM_6D_lite')

start = time.time()
sampler = BOBE(n_cobaya_init=4, n_sobol_init = 16, 
        miniters=75, maxiters=150,max_gp_size=200,
        loglikelihood=likelihood,
        fit_step = 5, update_mc_step = 5, ns_step = 10,
        num_hmc_warmup = 512,num_hmc_samples = 512, mc_points_size = 64,
        lengthscale_priors='DSLP', use_clf=False,minus_inf=-1e5,logz_threshold=0.5)

gp, ns_samples, logz_dict = sampler.run()
end = time.time()
print(f"Total time taken = {end-start:.4f} seconds")


plot_final_samples(gp, ns_samples,param_list=likelihood.param_list,param_bounds=likelihood.param_bounds,
                   param_labels=likelihood.param_labels,output_file=likelihood.name,
                   reference_file='./cosmo_input/chains/Planck_lite_LCDM',reference_ignore_rows=0.3,
                   reference_label='PolyChord',scatter_points=True,)

# Total time taken = 292.6820 seconds with fast updates
# Total time taken = 387.6565 seconds with older code

