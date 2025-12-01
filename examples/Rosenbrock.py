from jaxbo.bo import BOBE
from jaxbo.utils.plot import plot_final_samples
from jaxbo.likelihood import Likelihood, CobayaLikelihood
from jaxbo.samplers import renormalise_log_weights
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


likelihood_name = 'Rosenbrock'
start = time.time()

gp_kwargs = {'lengthscale_prior': 'DSLP'}

sampler = BOBE(
        loglikelihood=loglike,
        param_list=param_list,
        param_bounds=param_bounds,
        param_labels=param_labels,
        likelihood_name=likelihood_name,
        gp_kwargs=gp_kwargs,
        noise_std=0.0,
        minus_inf=-1e10,
        n_cobaya_init=4,
        n_sobol_init=16,
        use_clf=False,
        verbosity='INFO',
)

results = sampler.run(
        acqs='wipv',
        min_evals=50,
        max_evals=200,
        max_gp_size=200,
        fit_step=2,
        ns_step=10,
        num_hmc_warmup=512,
        num_hmc_samples=512,
        mc_points_size=32,
        mc_points_method='NS',
        logz_threshold=0.5,
)
print(f"Total time taken = {end-start:.4f} seconds")
print(f"Mean logz from dynesty = {mean:.4f} +/- {logz_err:.4f}")

if results is not None:
    gp = results['gp']
    samples = results['samples']
    logz_dict = results.get('logz', {})
    likelihood = results['likelihood']

    plot_final_samples(gp, samples, param_list=likelihood.param_list, param_bounds=likelihood.param_bounds,
                       param_labels=likelihood.param_labels, output_file=likelihood_name, reference_samples=reference_samples,
                       reference_file=None, scatter_points=True, reference_label='Dynesty')