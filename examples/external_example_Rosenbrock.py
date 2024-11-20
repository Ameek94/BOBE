import sys
import os
num_devices = int(sys.argv[1])
os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={num_devices}"
sys.path.append('../')
from bo_utils import input_unstandardize
from fb_gp import sample_GP_NUTS
from bo_be import sampler
from ext_loglike import external_loglike
import numpy as np
from jax import random
from getdist import plots,MCSamples,loadMCSamples
import matplotlib
matplotlib.rc('font', size=16,family='serif')
matplotlib.rc('legend', fontsize=16)
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{lmodern}')

# create external likelihood
ndim = 2
def logp(x,): # input should be (n x d) and output should be (n x 1) or n
    res = (1-x[:,0])**2 + 100*(x[:,1] - x[:,0]**2)**2
    return np.atleast_2d(-res)

param_bounds= np.array(ndim*[[-2.,2.]]).T
rosenbrock = external_loglike(logposterior=logp,ndim=ndim, param_bounds=param_bounds
                            ,name="Rosenbrock")

settings_file = './input_example.yaml'

bobe = sampler(settings_file=settings_file,
                cobaya_model=False,
                objfun=rosenbrock,
                seed=10,)
bobe.run()

train_x = input_unstandardize(bobe.train_x,bobe.param_bounds)

seed = 0
rng_key, _ = random.split(random.PRNGKey(seed), 2)
samples = sample_GP_NUTS(gp = bobe.gp,rng_key=rng_key,warmup_steps=1024,num_samples=8192,thinning=8)
print(samples.shape)
samples = input_unstandardize(samples,bobe.param_bounds)


keys = bobe.param_list

bounds_dict = {key: bobe.bounds_dict[key] for key in keys}

bf  = [1,1]

markers_bf = dict(zip(keys,bf))


gp_samples_nuts = MCSamples(samples=samples,names=keys, labels = bobe.param_labels
                            ,ranges=bounds_dict)

# test_samples = np.random.multivariate_normal(mean=mean,cov=cov,size=int(1e5))

# true_samples = MCSamples(samples=test_samples,names=keys, labels = bobe.param_labels
#                             ,ranges=bounds_dict)

g = plots.get_subplot_plotter(width_inch=8)
g.settings.axes_fontsize=18
g.settings.axes_labelsize = 18
g.settings.legend_fontsize = 18
g.settings.title_limit_fontsize = 14
g.triangle_plot([gp_samples_nuts,], keys,filled=[True,False],contour_lws=[1.25,1.5,1.],
                  contour_colors=['red','blue','black'],title_limit=1
                  ,markers = markers_bf
                  ,marker_args={'lw': 1.25, 'ls': '-.', 'color': 'C2'}
                  ,legend_labels=[f'BayOp {bobe.train_y.shape[0]}, samples','True']) 
g.export('Rosenbrock.pdf')

for i in range(bobe.ndim):
    ax = g.subplots[i,i]
    ax.axvline(bf[i], color='C2', ls='-.',lw=1.25)
    for j in range(i+1,bobe.ndim):
        ax = g.subplots[j,i]
        ax.scatter(train_x[:,i],train_x[:,j],alpha=0.33,color='c',s=8)
g.export('Rosenbrock_BO_points.pdf')

# expected logz is -4.258337519207816