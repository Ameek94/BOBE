from jaxbo.bo import BOBE
from jaxbo.bo_utils import plot_final_samples
from jaxbo.loglike import cobaya_loglike
import time

cobaya_input_file = './cosmo_input/Planck_lite_BAO_SN_CPL.yaml'

likelihood = cobaya_loglike(cobaya_input_file, confidence_for_unbounded=0.9999995,
        minus_inf=-1e5, noise_std=0.0,name='CPL_lite')

start = time.time()
sampler = BOBE(n_cobaya_init=8, n_sobol_init = 32, 
        miniters=200, maxiters=500,max_gp_size=500,
        loglikelihood=likelihood,
        fit_step = 10, update_mc_step = 5, ns_step = 25,
        num_hmc_warmup = 512,num_hmc_samples = 512, mc_points_size = 64,
        use_svm=True,svm_use_size=150,
        lengthscale_priors='DSLP', minus_inf=-1e5,)

gp, ns_samples, logz_dict = sampler.run()
end = time.time()
print(f"Total time taken = {end-start:.4f} seconds")


plot_params = ['w','wa','omk','omch2','logA','ns','H0','ombh2','tau']
plot_final_samples(gp, ns_samples,param_list=sampler.param_list,param_bounds=sampler.param_bounds,
                   param_labels=sampler.param_labels,output_file=likelihood.name,
                   plot_params=plot_params,
                   reference_file=None,reference_ignore_rows=0.3,
                   reference_label='PolyChord',scatter_points=True,)