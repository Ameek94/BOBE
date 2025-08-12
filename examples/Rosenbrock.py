from jaxbo.bo import BOBE
from jaxbo.utils.summary_plots import plot_final_samples
from jaxbo.loglike import ExternalLikelihood,CobayaLikelihood
from jaxbo.nested_sampler import renormalise_log_weights
from getdist import MCSamples
from dynesty import DynamicNestedSampler
import numpy as np
import time


ndim = 2
param_bounds = np.array([[-1,4],[-1,7]]).T
param_list = ['x1','x2']
param_labels = ['x_1','x_2']

def loglike(x):
    res = (1-x[0])**2 + 100*(x[1] - x[0]**2)**2
    return -res

def prior_transform(x):
    x[0] = x[0]*5 - 1
    x[1] = x[1]*8 - 1
    return x

dns_sampler =  DynamicNestedSampler(loglike,prior_transform,ndim=ndim,
                                       sample='rwalk')

dns_sampler.run_nested(print_progress=True,dlogz_init=0.01) 
res = dns_sampler.results  
mean = res['logz'][-1]
logz_err = res['logzerr'][-1]
print(f"Mean logz from dynesty = {mean:.4f} +/- {logz_err:.4f}")

samples = res['samples']
weights = renormalise_log_weights(res['logwt'])

reference_samples = MCSamples(samples=samples, names=param_list, labels=param_labels,
                            weights=weights, 
                            ranges= dict(zip(param_list,param_bounds.T)))


likelihood = ExternalLikelihood(loglikelihood=loglike,ndim=ndim,param_list=param_list,
        param_bounds=param_bounds,param_labels=param_labels,
        name='Rosenbrock',noise_std=0.0,minus_inf=-1e5)
start = time.time()
sampler = BOBE(n_cobaya_init=4, n_sobol_init = 16, 
        miniters=50, maxiters=200,max_gp_size=200,
        loglikelihood=likelihood,mc_points_method='NS',
        fit_step = 2, update_mc_step = 5, ns_step = 10,
        num_hmc_warmup = 512,num_hmc_samples = 512, mc_points_size = 32,
        logz_threshold=0.5,
        lengthscale_priors='DSLP', use_clf=False,minus_inf=-1e10,)

gp, ns_samples, logz_dict = sampler.run()
end = time.time()
print(f"Total time taken = {end-start:.4f} seconds")
print(f"Mean logz from dynesty = {mean:.4f} +/- {logz_err:.4f}")

plot_final_samples(gp, ns_samples,param_list=likelihood.param_list,param_bounds=likelihood.param_bounds,
                   param_labels=likelihood.param_labels,output_file=likelihood.name,reference_samples=reference_samples,
                   reference_file=None,scatter_points=True,reference_label='Dynesty')