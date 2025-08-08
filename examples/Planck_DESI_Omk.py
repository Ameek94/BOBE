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
        resume=True,
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
                   reference_label='MCMC',scatter_points=False,)

# 2025-04-21 18:27:42,039 INFO:[BO]:  Final LogZ: upper=-5527.4084, mean=-5529.4980, lower=-5530.0967, dlogz sampler=0.1720
# PolyChord result: # log-evidence
# logZ: -5529.65218118231
# logZstd: 0.447056743748251

# INFO:[NS]: Nested Sampling took 289.45s
# INFO:[NS]: Log Z evaluated using (30859,) points
# INFO:[NS]: Dynesty made 1084416 function calls, max value of logl = -5488.9169
# 2025-08-06 17:34:02,615 INFO:[BO]:  LogZ info: mean=-5529.2378, dlogz sampler=0.2430, upper=-5513.5731, lower=-5529.9985
# 2025-08-06 17:34:02,616 INFO:[BO]:  Convergence check: delta = 0.7608, step = 1349
# 2025-08-06 17:34:02,616 INFO:[BO]:  Converged
# 2025-08-06 17:34:02,621 INFO:[BO]:  Sampling stopped: LogZ converged
# 2025-08-06 17:34:02,621 INFO:[BO]:  Final GP training set size: 1360, max size: 1800
# 2025-08-06 17:34:02,621 INFO:[BO]:  Number of iterations: 1350, max iterations: 2000
# 2025-08-06 17:34:02,621 INFO:[BO]: Using nested sampling results
# Total time taken = 10917.0748 seconds
# INFO:jaxbo.utils:Parameter limits from GP
# INFO:jaxbo.utils:\Omega_K = 0.0023\pm 0.0014
# INFO:jaxbo.utils:\Omega_\mathrm{c} h^2 = 0.1199^{+0.0011}_{-0.00099}
# INFO:jaxbo.utils:\log(10^{10} A_\mathrm{s}) = 3.039^{+0.015}_{-0.017}
# INFO:jaxbo.utils:n_\mathrm{s} = 0.9626\pm 0.0041
# INFO:jaxbo.utils:H_0 = 68.32^{+0.48}_{-0.55}
# INFO:jaxbo.utils:\Omega_\mathrm{b} h^2 = 0.02217\pm 0.00014
# INFO:jaxbo.utils:\tau_\mathrm{reio} = 0.0533^{+0.0073}_{-0.0083}
# INFO:jaxbo.utils:Parameter limits from MCMC
# INFO:jaxbo.utils:\Omega_K = 0.0022^{+0.0015}_{-0.0013}
# INFO:jaxbo.utils:\Omega_\mathrm{c} h^2 = 0.1194^{+0.0011}_{-0.0010}
# INFO:jaxbo.utils:\log(10^{10} A_\mathrm{s}) = 3.044\pm 0.014
# INFO:jaxbo.utils:n_\mathrm{s} = 0.9639\pm 0.0040
# INFO:jaxbo.utils:H_0 = 68.45\pm 0.49
# INFO:jaxbo.utils:\Omega_\mathrm{b} h^2 = 0.02220\pm 0.00013
# INFO:jaxbo.utils:\tau_\mathrm{reio} = 0.0558\pm 0.0070