from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
from typing import Dict
import pandas as pd
import numpy as np
from getdist import plots,MCSamples,loadMCSamples
import matplotlib.pyplot as plt

# use this to suppress most of the unecessary polychord output, https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


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