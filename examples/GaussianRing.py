from jaxbo.bo import BOBE
from jaxbo.bo_utils import plot_final_samples
from jaxbo.loglike import external_loglike,cobaya_loglike
from jaxbo.nested_sampler import renormalise_log_weights
from getdist import MCSamples
from dynesty import DynamicNestedSampler
import numpy as np
import time

ndim = 2
param_list = ['x1','x2']
param_labels = ['x_1','x_2']
param_bounds = np.array([[0,1],[0,1]]).T

mean_r = 0.2
scale = 0.02


def loglike(X):
    r2 = (X[0]-0.5)**2 + (X[1]-0.5)**2
    r = np.sqrt(r2)
    return -0.5*((r-mean_r)/scale)**2

def prior_transform(x):
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


likelihood = external_loglike(loglikelihood=loglike,ndim=ndim,param_list=param_list,
        param_bounds=param_bounds,param_labels=param_labels,
        name='GaussianRing',noise_std=0.0,minus_inf=-1e5)
start = time.time()
sampler = BOBE(n_cobaya_init=4, n_sobol_init = 16, 
        miniters=75, maxiters=150,max_gp_size=200,
        loglikelihood=likelihood,
        fit_step = 4, update_mc_step = 4, ns_step = 20,
        num_hmc_warmup = 512,num_hmc_samples = 512, mc_points_size = 64,
        logz_threshold=0.1,mc_points_method='NS',
        lengthscale_priors='DSLP', use_svm=False,minus_inf=-1e5,)

gp, ns_samples, logz_dict = sampler.run()
print(f"Mean logz from dynesty = {mean:.4f} +/- {logz_err:.4f}")
end = time.time()
print(f"Total time taken = {end-start:.4f} seconds")

plot_final_samples(gp, ns_samples,param_list=sampler.param_list,param_bounds=sampler.param_bounds,
                   param_labels=sampler.param_labels,output_file=likelihood.name,reference_samples=reference_samples,
                   reference_file=None,scatter_points=True,reference_label='Dynesty')