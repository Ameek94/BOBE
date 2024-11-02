from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
from typing import Dict
import pandas as pd
import numpy as np
from getdist import plots,MCSamples,loadMCSamples
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap

# use this to suppress unecessary output, https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

# this will mainly be used for the GP prediction so func will return mean and var, each with shape num_samples x num_test_points
# minor modifications of https://github.com/martinjankowiak/saasbo/blob/main/util.py
def split_vmap(func,input_arrays,batch_size=10):
    num_inputs = input_arrays[0].shape[0]
    num_batches = (num_inputs + batch_size - 1 ) // batch_size
    batch_idxs = [jnp.arange( i*batch_size, min( (i+1)*batch_size,num_inputs  )) for i in range(num_batches)]
    res = [vmap(func)(*tuple([arr[idx] for arr in input_arrays])) for idx in batch_idxs]
    nres = len(res[0])
    # now combine results across batches and function outputs to return a tuple (num_outputs, num_inputs, ...)
    results = tuple( jnp.concatenate([x[i] for x in res]) for i in range(nres))
    return results

def input_standardize(x,param_bounds):
    """
    Project from original domain to unit hypercube, X is N x d shaped, param_bounds are 2 x d
    """
    x =  (x - param_bounds[0])/(param_bounds[1] - param_bounds[0])
    return x

def input_unstandardize(x,param_bounds):
    """
    Project from unit hypercube to original domain, X is N x d shaped, param_bounds are 2 x d
    """
    x = x * (param_bounds[1] - param_bounds[0]) + param_bounds[0]
    return x

def output_standardize(y): # y is N x 1 shaped
    """
    Convert training data to zero mean and unit variance
    """
    ystd = y.std(axis=0)
    ymean = y.mean(axis=0)
    return (y-ymean)/ystd

def output_unstandardize(y,std,mean):
    """
    Convert training data from zero mean and unit variance to original domain
    """
    return y*std + mean

def plot_gp(samples,gp,names,labels,param_bounds,thin=1):
    ranges =  param_bounds.T
    gp_samples = MCSamples(samples=samples[::thin],names=names, labels = labels,ranges=ranges) # a comparison run
    g = plots.get_subplot_plotter(subplot_size=2.5,subplot_size_ratio=1)
    # g.settings.num_plot_contours = 2
    g.settings.axes_labelsize = 18
    g.settings.axes_fontsize = 16
    g.settings.legend_fontsize = 14
    g.settings.title_limit_fontsize = 14
    if samples.shape[1]==1:
        g.plot_1d(gp_samples,'x_1',filled=[True,False],colors=['red','blue'],
                        legend_labels=[f'GP fit, N = {gp.train_y.shape[0]} samples','True Distribution'],title_limit=1)

    else:
        g.triangle_plot([gp_samples], names,filled=[True,False],contour_colors=['red','blue'],
                                legend_labels=[f'GP fit, N = {gp.train_y.shape[0]} ','True Distribution'],
                                contour_lws=[1,1.5],title_limit=1,) # type: ignore
    plt.show()