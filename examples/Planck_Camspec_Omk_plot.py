from jaxbo.bo import BOBE
from jaxbo.utils import plot_final_samples
from jaxbo.loglike import CobayaLikelihood
import time
from getdist import MCSamples, loadMCSamples, plots
import numpy as np      

cobaya_input_file = './cosmo_input/LCDM_Planck_DESI_omk.yaml'

likelihood = CobayaLikelihood(cobaya_input_file, confidence_for_unbounded=0.9999995,
        minus_inf=-1e5, noise_std=0.0,name='Planck_Camspec_fast_Omk')


samples_GP = np.load(f'Planck_Camspec_fast_Omk_samples.npz')
print(f"Samples keys: {samples_GP.keys()}")
bounds = samples_GP['param_bounds']
samples = samples_GP['samples']
weights = samples_GP['weights']

param_list = likelihood.param_list

ranges_dict = dict(zip(param_list,bounds.T))

gp_samples = MCSamples(samples=samples, names=param_list, labels=likelihood.param_labels, ranges=ranges_dict, weights=weights,sampler='nested')


reference_file = './cosmo_input/chains/PPlus_curved_LCDM'
ref_samples = loadMCSamples(reference_file,settings={'ignore_rows': 0.3})

g = plots.get_subplot_plotter(subplot_size=2.5,subplot_size_ratio=1)
g.settings.legend_fontsize = 22
g.settings.axes_fontsize=18
g.settings.axes_labelsize=18
g.settings.title_limit_fontsize = 14   
g.triangle_plot([gp_samples,ref_samples], params = ['omk','omch2','logA','ns','H0','ombh2','tau'],filled=[True,False],
                    contour_colors=['#006FED', 'black'],contour_lws=[1,1.5],
                    legend_labels=['GP','MCMC'],marker_args={'lw': 1, 'ls': ':'})
# if scatter_points:
#         points = scale_from_unit(gp.train_x,param_bounds)
#         for i in range(ndim):
#         # ax = g.subplots[i,i]
#         for j in range(i+1,ndim):
#         ax = g.subplots[j,i]
#         ax.scatter(points[:,i],points[:,j],alpha=0.5,color='forestgreen',s=5)

g.export(likelihood.name+'_samples.pdf')

print(f"Ranges dict: {ranges_dict}")

