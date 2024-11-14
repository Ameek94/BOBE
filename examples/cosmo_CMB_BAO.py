import sys

import cobaya
sys.path.append('../')
from fb_gp import saas_fbgp
import numpy as np
import time
from jaxns.framework.model import Model
from jaxns.framework.prior import Prior
import jax
import tensorflow_probability.substrates.jax as tfp
import jax.numpy as jnp
from jax import random,vmap, grad
tfpd = tfp.distributions
from bo_be import sampler
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from acquisition import EI, IPV, optim_scipy_bh
import scipy.optimize
from scipy.stats import qmc
from jaxns import NestedSampler
import corner
from bo_utils import input_unstandardize
from nested_sampler import nested_sampling_jaxns, nested_sampling_Dy
from getdist import plots,MCSamples,loadMCSamples
matplotlib.rc('font', size=16,family='serif')
matplotlib.rc('legend', fontsize=16)


input_file = './cosmo_input/LCDM_Planck_DESI.yaml'

# input_file = './cosmo_input/LCDM_6D.yaml'

max_steps = 160
nstart = 32
cobaya_init = 8
ntot = max_steps + nstart + cobaya_init
save_file = 'cmb_bao'
cosmo = sampler(cobaya_model=True,input_file=input_file,cobaya_start=cobaya_init,seed=200000,
                max_steps=max_steps, nstart=nstart,mc_points_size=32,acq_goal=5e-7,noise=1e-6,
                save=True,save_file=save_file)

cosmo.run()

from fb_gp import sample_GP_NUTS

train_x = input_unstandardize(cosmo.train_x,cosmo.param_bounds)
seed = 0
rng_key, _ = random.split(random.PRNGKey(seed), 2)
samples = sample_GP_NUTS(gp = cosmo.gp,rng_key=rng_key,num_warmup=1024,num_samples=8192,thinning=8)

samples = input_unstandardize(samples,cosmo.param_bounds)

np.savetxt('samples_LCDM_Full_'+str(ntot)+'.txt',samples)

np.savetxt('train_x_LCDM_Full_'+str(ntot)+'.txt',train_x)

keys = ['omch2', 'logA', 'ns', 'H0', 'ombh2', 'tau']

bounds_dict = {key: cosmo.bounds_dict[key] for key in keys}

bf  = [1.186384753e-01 ,3.040776555e+00,9.652909209e-01,6.771234910e+01 ,2.224954976e-02, 5.546101493e-02]

markers_bf = dict(zip(keys,bf))

mcmc_samples = loadMCSamples('./cosmo_input/chains/Planck_DESI_LCDM',settings={'ignore_rows': 0.3})

gp_samples_nuts = MCSamples(samples=samples[:,:6],names=keys, labels = cosmo.param_labels[:6]
                            ,ranges=bounds_dict)


g = plots.get_subplot_plotter(subplot_size=3,subplot_size_ratio=1)
for s in [gp_samples_nuts,mcmc_samples]:
        for p in keys:
            print(s.getInlineLatex(p,limit=1))
# g.settings.num_plot_contours = 2
g.settings.axes_fontsize=18
g.settings.axes_labelsize = 18
g.settings.legend_fontsize = 18
g.settings.title_limit_fontsize = 14
g.triangle_plot([gp_samples_nuts,mcmc_samples], keys,filled=[True,False],contour_lws=[1.25,1.5,1.],
                  contour_colors=['red','blue','black'],markers = markers_bf,title_limit=1,
                  marker_args={'lw': 1.25, 'ls': '-.', 'color': 'C2'}
                  ,legend_labels=[f'BayOp, {ntot} samples','MCMC']) #,param_limits=cosmo.bounds_dict) # type: ignore
# #                                 legend_labels=[f'GP fit, N = {gp.train_y.shape[0]} samples','True Distribution','HMC on GP fit'],
# #                                 contour_lws=[1,1.5,1.],markers = dict(zip(names,f_mean)),
# #                                 marker_args={'lw': 1.25, 'ls': '-', 'color': 'C2'},title_limit=1 ) # type: ignore


g.export('LCDM_full_'+str(ntot)+'.pdf')
for i in range(6):
    ax = g.subplots[i,i]
    ax.axvline(bf[i], color='C2', ls='-.',lw=1.25)
    for j in range(i+1,6):
        ax = g.subplots[j,i]
        ax.scatter(train_x[:,i],train_x[:,j],alpha=0.4,color='c',s=8)
g.export('LCDM_full_'+str(ntot)+'_points.pdf')