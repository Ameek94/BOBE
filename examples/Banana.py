from jaxbo.bo import BOBE
from jaxbo.utils import plot_final_samples
from jaxbo.loglike import ExternalLikelihood, CobayaLikelihood
from jaxbo.nested_sampler import renormalise_log_weights
from getdist import MCSamples
from dynesty import DynamicNestedSampler
import numpy as np
import time

ndim = 2
param_list = ['x1','x2']
param_labels = ['x_1','x_2']
param_bounds = np.array([[-1,1],[-1,2]]).T

def loglike(X):
    logpdf = -0.25*(5*(0.2-X[0]))**2 - (20*(X[1]/4 - X[0]**4))**2
    return logpdf

def prior_transform(x):
    x[0] = x[0]*2 - 1 #x[0] * (param_bounds[0,1] - param_bounds[0,0]) + param_bounds[0,0]
    x[1] = x[1]*3 - 1 #x[1] * (param_bounds[1,1] - param_bounds[1,0]) + param_bounds[1,0]
    return x


likelihood = ExternalLikelihood(loglikelihood=loglike,ndim=ndim,param_list=param_list,
        param_bounds=param_bounds,param_labels=param_labels,
        name='banana',noise_std=0.0,minus_inf=-1e5)
start = time.time()
sampler = BOBE(n_cobaya_init=4, n_sobol_init = 8, 
        miniters=50, maxiters=120,max_gp_size=200,
        loglikelihood=likelihood,
        fit_step = 2, update_mc_step = 2, ns_step = 10,
        num_hmc_warmup = 512,num_hmc_samples = 512, mc_points_size = 32,
        logz_threshold=0.1,
        lengthscale_priors='DSLP', use_clf=False,minus_inf=-1e5,)

results = sampler.run()
gp = results['gp']
ns_samples = results['samples']

end = time.time()
print(f"Total time taken = {end-start:.4f} seconds")
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


plot_final_samples(gp, ns_samples,param_list=likelihood.param_list,param_bounds=likelihood.param_bounds,
                   param_labels=likelihood.param_labels,output_file=likelihood.name,reference_samples=reference_samples,
                   reference_file=None,scatter_points=True,reference_label='Dynesty')