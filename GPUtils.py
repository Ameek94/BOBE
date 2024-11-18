from torch import Tensor
from botorch.utils.transforms import normalize, unnormalize
import math
import numpy as np

import torch
from pyro.ops import stats
import functools
from collections import OrderedDict
import pyro.poutine as poutine

from getdist import plots,MCSamples,loadMCSamples
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import glob
import contextlib
from PIL import Image

import logging
log = logging.getLogger("[GP UTILS]")

from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
from typing import Dict
import pandas as pd

import jax.numpy as jnp
from jax import vmap

# use this to suppress unecessary output, https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
#@contextmanager
#def suppress_stdout_stderr():
#    """A context manager that redirects stdout and stderr to devnull"""
#    with open(devnull, 'w') as fnull:
#        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
#            yield (err, out)

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



def prior_transform(u, bounds):
    for i in range(len(u)):
        u[i] = bounds[i][0] + u[i]*(bounds[i][1] - bounds[i][0])
    return u
    
def ext_logp_torch(X: Tensor, loglike, interp_logp, torch_bounds) -> Tensor: # logposterior for external likelihoods, takes input in [0,1] then calls the user defined function 
        # Internally X should be a N x DIM tensor with parameters in the same order as the param_list in range [0,1]^DIM
        # So we need to unnormalize x if physical range is not in [0,1]
        x =  unnormalize(X, torch_bounds.t())
        logpdf = loglike(x).unsqueeze(-1) #output should be N x 1    
        if interp_logp: # if GPR on loglikelihood
            return logpdf 
        else: # if GPR on exp(loglikelihood)
            return torch.exp(logpdf)


def ext_logp_np(X, loglike, interp_logp, np_bounds): # logposterior for external likelihoods, takes input in [0,1] then calls the user defined function 
        # Internally X should be a N x DIM tensor with parameters in the same order as the param_list in range [0,1]^DIM
        # So we need to unnormalize x if physical range is not in [0,1]
        x =  input_unstandardize(X, np_bounds.T)
        logpdf = np.expand_dims(loglike(x), -1) #output should be N x 1    
        if interp_logp: # if GPR on loglikelihood
            return np.array(logpdf) 
        else: # if GPR on exp(loglikelihood)
            return np.exp(logpdf)


def compute_log_density(model, hyperparameters):
    data={'kernel_tausq': hyperparameters['kernel_tausq'].unsqueeze(-1) , 
          '_kernel_inv_length_sq': hyperparameters['_kernel_inv_length_sq'], 
          'outputscale': hyperparameters['outputscale']}
    print(data)
    conditioned_model = poutine.condition(model, 
                                          data=data)
    trace = poutine.trace(conditioned_model).get_trace()
    return -trace.log_prob_sum()
    
def _safe(fn):
    """
    Safe version of utilities in the :mod:`pyro.ops.stats` module. Wrapped
    functions return `NaN` tensors instead of throwing exceptions.

    :param fn: stats function from :mod:`pyro.ops.stats` module.
    """

    @functools.wraps(fn)
    def wrapped(sample, *args, **kwargs):
        try:
            val = fn(sample, *args, **kwargs)
        except Exception:
            warnings.warn(tb.format_exc())
            val = torch.full(
                sample.shape[2:], float("nan"), dtype=sample.dtype, device=sample.device
            )
        return val

    return wrapped

def summary_NUTS(samples, prob=0.9, group_by_chain=True):
    """
    Returns a summary table displaying diagnostics of ``samples`` from the
    posterior. The diagnostics displayed are mean, standard deviation, median,
    the 90% Credibility Interval, :func:`~pyro.ops.stats.effective_sample_size`,
    :func:`~pyro.ops.stats.split_gelman_rubin`.

    :param dict samples: dictionary of samples keyed by site name.
    :param float prob: the probability mass of samples within the credibility interval.
    :param bool group_by_chain: If True, each variable in `samples`
        will be treated as having shape `num_chains x num_samples x sample_shape`.
        Otherwise, the corresponding shape will be `num_samples x sample_shape`
        (i.e. without chain dimension).
    """
    if not group_by_chain:
        samples = {k: v.unsqueeze(0) for k, v in samples.items()}

    summary_dict = {}
    for name, value in samples.items():
        value_flat = torch.reshape(value, (-1,) + value.shape[2:])
        mean = value_flat.mean(dim=0)
        std = value_flat.std(dim=0)
        median = value_flat.median(dim=0)[0]
        hpdi = stats.hpdi(value_flat, prob=prob)
        n_eff = _safe(stats.effective_sample_size)(value)
        r_hat = stats.split_gelman_rubin(value)
        hpd_lower = "{:.1f}%".format(50 * (1 - prob))
        hpd_upper = "{:.1f}%".format(50 * (1 + prob))
        summary_dict[name] = OrderedDict(
            [
                ("mean", mean),
                ("std", std),
                ("median", median),
                (hpd_lower, hpdi[0]),
                (hpd_upper, hpdi[1]),
                ("n_eff", n_eff),
                ("r_hat", r_hat),
            ]
        )
    return summary_dict
    

def png2gif(curr_step, loglike_name, ndim):
    log.info(f"Generating GIF from plot frames: GIFs/{ndim}D/{loglike_name}_{curr_step}steps_{ndim}D.gif")
    fp_in = []
    for i in range(1, curr_step):
        fp_in.append(f"GIFs/{ndim}D/{loglike_name}_step_{i}.png")
    fp_out = f"GIFs/{loglike_name}_{curr_step}steps_{ndim}D.gif"
    
    # use exit stack to automatically close opened images
    with contextlib.ExitStack() as stack:
    
        # lazily load images
        img, *imgs = [Image.open(f) for f in fp_in]
        
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=300, loop=1)


def slice_plot_2d(ax, FBGP, train_X, train_Y, logp):
    X_plot = np.linspace(start=0, stop=1, num=100)
    Y_plot = np.full_like(X_plot, 0.5)
    
    ### X1 ###
    x1_samples = np.vstack([X_plot.ravel(), Y_plot.ravel()]).T
    ax[0, 0].plot(X_plot, logp(x1_samples), label=r"Log Likelihood", linestyle="dotted")
    mean_FB = np.mean(FBGP.posterior(x1_samples)[0], 0)
    std_FB = np.sqrt(FBGP.posterior(x1_samples)[1]).mean()
    ax[0, 0].scatter(train_X[:, 0], train_Y, label="Observations")
    ax[0, 0].plot(X_plot, mean_FB, label="Mean prediction")
    ax[0, 0].fill_between(
        X_plot.ravel(),
        mean_FB - 1.96 * std_FB,
        mean_FB + 1.96 * std_FB,
        alpha=0.5,
        label=r"95% confidence interval",
        )
    ax[0, 0].legend()
    ax[0, 0].set_xlabel("$x1$")
    ax[0, 0].set_ylabel("$f(x1)$")
    ax[0, 0].set_title("FBGP x1")

    ### X2 ###
    x2_samples = np.vstack([Y_plot.ravel(), X_plot.ravel()]).T
    ax[0, 1].plot(X_plot, logp(x2_samples), label=r"Log Likelihood", linestyle="dotted")
    mean_FB = np.mean(FBGP.posterior(x2_samples)[0], 0)
    std_FB = np.sqrt(FBGP.posterior(x2_samples)[1]).mean()
    ax[0, 1].scatter(train_X[:, 1], train_Y, label="Observations")

    ax[0, 1].plot(X_plot, mean_FB, label="Mean prediction")
    ax[0, 1].fill_between(
        X_plot.ravel(),
        mean_FB - 1.96 * std_FB,
        mean_FB + 1.96 * std_FB,
        alpha=0.5,
        label=r"95% confidence interval",
        )
    ax[0, 1].legend()
    ax[0, 1].set_xlabel("$x2$")
    ax[0, 1].set_ylabel("$f(x2)$")
    ax[0, 1].set_title("FBGP x2")

    return ax

def acq_check_metric_plot(ax, curr_step, acq_goal, post_var_plot, pre_var_plot, acq_check_plot, integral_true_accuracy):
    x = np.linspace(0, curr_step, curr_step)
    ax[0, 2].loglog(x, post_var_plot, label="Post Var", marker="x")
    ax[0, 2].loglog(x, pre_var_plot, label="Pre Var", marker="x")
    ax[0, 2].loglog(x, acq_check_plot, label="Acq Check", marker="x")
    #ax[0, 2].loglog(x, np.array(integral_true_accuracy)[:, 3], label="Uncertainty on BOBE estimate of Integral", marker='x')
    ax[0, 2].axhline(y = acq_goal, xmin=0, xmax = curr_step, linestyle="dotted", label="Acq Goal")
    ax[0, 2].set_title("Convergence Metrics")
    return ax
    
def FBGP_hyperparameter_plot(ax, curr_step, FBGP_outputscale, FBGP_lengthscales):
    ndim = np.array(FBGP_lengthscales).shape[-1]
    
    ### Outputscale ###
    ax[1, 0].plot(np.linspace(1, curr_step, curr_step), FBGP_outputscale, marker="o")
    ax[1, 0].text(0.1, 0.99, f"Max: {np.max(FBGP_outputscale[-1])} Min: {np.min(FBGP_outputscale[-1])}", ha='left', va='top', transform=ax[1, 0].transAxes)
    ax[1, 0].set_xlabel("Step Number")
    ax[1, 0].set_ylabel("Outputscale")
    ax[1, 0].set_title("FBGP Outputscale")
    
    if ndim == 2:
        ### Lengthscales ###
        for l in range(0, ndim):
            ax[1, l+1].plot(np.linspace(1, curr_step, curr_step), np.array(FBGP_lengthscales)[..., l], marker="o")
            ax[1, l+1].text(0.1, 0.90, f"Max: {np.array(FBGP_lengthscales)[..., l][-1].max()} Min: {np.array(FBGP_lengthscales)[:, l][-1].min()}", ha='left', va='top', transform=ax[1, l+1].transAxes)
            ax[1, l+1].set_xlabel("Step Number")
            ax[1, l+1].set_ylabel(f"Lengthscale x{l+1}")
            ax[1, l+1].set_title(f"FBGP Lengthscale x{l+1}")

    if ndim > 2:
        for l in range(0,  ndim):
            #We want to alternate between each plot until we run out of lengthscales
            if l%2 == 0: #Even for [1, 1]
                lengthscale = np.mean(np.array(FBGP_lengthscales)[..., l], axis=1)
                ax[1, 1].plot(np.linspace(1, curr_step, curr_step),lengthscale, marker="o", label=f"Lengthscale x{l+1}")
                #We want to move the text label down each time we plot again on the same plot we know there have been n/2 + 1 even numbers
                ax[1, 1].text(0.1, 0.99-(l/2)*(.05), f"Max: {np.array(FBGP_lengthscales)[..., l][-1].max()} Min: {np.array(FBGP_lengthscales)[:, l][-1].min()}", ha='left', va='top', transform=ax[1, 1].transAxes)
            if l%2 == 1: #Odd for [1, 2]
                lengthscale = np.mean(np.array(FBGP_lengthscales)[..., l], axis=1)
                ax[1, 2].plot(np.linspace(1, curr_step, curr_step),lengthscale, marker="o", label=f"Lengthscale x{l+1}")
                #We want to move the text label down each time we plot again on the same plot we know there have been floor(n/2 + 1) odd numbers
                ax[1, 2].text(0.1, 0.99-(np.floor(l/2))*(.05), f"Max: {np.array(FBGP_lengthscales)[..., l][-1].max()} Min: {np.array(FBGP_lengthscales)[:, l][-1].min()}", ha='left', va='top', transform=ax[1, 2].transAxes)
        ax[1, 1].set_title('FBGP Lengthscales')
        ax[1, 1].sharey(ax[1, 2])
        ax[1, 1].legend(loc='lower left')
        ax[1, 2].set_title('FBGP Lengthscales')
        ax[1, 2].legend(loc='lower left')
                        
    return ax
    
def integral_metrics_plot(ax, curr_step, integral_true_accuracy, acq_goal):
    ax[2, 0].plot(np.linspace(0, curr_step, curr_step), np.array(integral_true_accuracy)[:, 0], marker="o", label="Nested Sampler Integral - Analytic Integral")
    #ax[2, 0].plot(np.linspace(1, curr_step, curr_step), np.array(integral_true_accuracy)[:, 2], marker="o", label="Dbl Quad Integral - Analytic Integral")
    ax[2, 0].plot(np.linspace(0, curr_step, curr_step), np.array(integral_true_accuracy)[:, 2], marker="o", label="(BOBE Upper - BOBE Lower)/2")
    ax[2, 0].plot(np.linspace(0, curr_step, curr_step), np.array(integral_true_accuracy)[:, 3], marker="o", label="Intrinsic Error in Nested Sampler")
    ax[2, 0].axhline(y = acq_goal, xmin=0, xmax = curr_step, linestyle="dotted", label="Precision Goal")
    ax[2, 0].set_yscale('log')
    ax[2, 0].legend()
    ax[2, 0].sharex(ax[1, 1])
    ax[2, 0].set_title("Comparison to Analytic Integral")

    return ax


def FBGP_mll_plot(ax, curr_step, computed_mlls):
    ax[2, 1].plot(np.linspace(1, curr_step, curr_step), np.array(computed_mlls).reshape(curr_step, -1), marker="o")
    ax[2, 1].set_xlabel("Step Number")
    ax[2, 1].set_ylabel("Exact Marginal Log Likelihood")
    ax[2, 1].set_title("Exact Marginal Log Likelihoood")

    return ax

def integral_accuracy_plot(ax, curr_step, acq_goal, log_z_mean, log_z_upper, log_z_lower, logz_truth, prior_fac):
    ax[2, 2].plot(np.linspace(0, curr_step, curr_step), log_z_mean, label="Mean", marker="x")
    ax[2, 2].plot(np.linspace(0, curr_step, curr_step), log_z_upper, linestyle="dotted", label="Upper", marker="x")
    ax[2, 2].plot(np.linspace(0, curr_step, curr_step), log_z_lower, linestyle="dotted", label="Lower", marker="x")
    ax[2, 2].axhline(y = logz_truth - prior_fac, xmin=0, xmax = curr_step, linestyle="dotted", label="Real Value")
    ax[2, 2].axhline(y = (logz_truth - prior_fac) + acq_goal, xmin=0, xmax = curr_step, linestyle="dotted", label="Precision Goal Upper")
    ax[2, 2].axhline(y = (logz_truth - prior_fac) - acq_goal, xmin=0, xmax = curr_step, linestyle="dotted", label="Precision Goal Lower")
    ax[2, 2].legend()
    ax[2, 2].sharex(ax[1, 1])
    ax[2, 2].set_title("Integral Precision")

    return ax

def FBGP_prediction_plots(ax, FBGP, logp, train_X, train_Y, samples_unit, param_list, bounds_dict, nstart):
    ### MESHGRID CALCULATION ###
    x = np.linspace(0,1, 100)
    y = x
    xx, yy = np.meshgrid(x, y)
    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    ##############################
    
    mean, var = FBGP.posterior(grid)
    mean = np.mean(mean, 0)
    var = np.mean(var, 0)
    upper = mean + var
    lower = mean - var
    gp_fit = gp_fit_plot = mean
    actual = logp(grid).squeeze(-1)
    var_norm = train_Y.var()

    dy_samples = MCSamples(samples=samples_unit[::8], names=param_list, ranges=bounds_dict)

    diff = abs((gp_fit-actual))#/actual)
    diff = diff.reshape(x.shape[0],y.shape[0])
    unc = abs(upper-lower) #/abs(gp_fit)
    unc = unc.reshape(x.shape[0],y.shape[0])
    vmin,vmax = np.min(unc),np.max(unc)

    try:
        g = plots.get_subplot_plotter(subplot_size=6,subplot_size_ratio=3/4)
        g.settings.num_plot_contours = 2
        g.settings.axes_labelsize = 16
        g.settings.axes_fontsize = 14
        g.plots_2d(dy_samples, param_pairs=[['x1','x2'],['x1','x2']],lws=[1.5],colors=['k'],nx=2, ax=ax[3, 0]) #dy_samples
        g.plots_2d(dy_samples, param_pairs=[['x1','x2'],['x1','x2']],lws=[1.5],colors=['k'],nx=2, ax=ax[3, 1])
        ax1 = g.get_axes(ax=ax[3,0])
        cont = ax1.contourf(x,y,diff,cmap='jet', locator=ticker.LogLocator())
        plt.colorbar(cont,ax=ax1)
        ax1.set_title(r"GP Mean vs True")
        ax2 = g.get_axes(ax=ax[3,1])
        cont2 = ax2.contourf(x, y,unc/var_norm,cmap='jet', locator=ticker.LogLocator())
        plt.colorbar(cont2,ax=ax2)
        ax2.set_title(r'GP uncertainity')
        for axes in [ax1,ax2]:
            axes.set_xlim(0,1)
            axes.set_ylim(0,1)
            with torch.no_grad():
                axes.scatter(train_X[:nstart,0],train_X[:nstart,1],s=25,label='Sobol',color='C2',alpha=0.9)
                axes.scatter(train_X[nstart:,0],train_X[nstart:,1],s=20,color='r',alpha=0.75)
    except Exception as e:
        print(f"Get dist plots failed: {e}")

    cntr1 = ax[3, 2].contourf(xx, yy, np.exp(gp_fit_plot.reshape(100, 100)), cmap="jet")
    plt.colorbar(cntr1, ax=ax[3, 2])
    ax[3, 2].set_xlim(0, 1)
    ax[3, 2].set_ylim(0, 1)
    ax[3, 2].set_xlabel('x1')
    ax[3, 2].set_ylabel('x2')
    ax[3, 2].set_title('FB GP Prediction')

    return ax

def timing_plot(ax, curr_step, timing, ndim):
    '''if ndim <= 2: 
        row = 4
    else:'''
    row = 3
    ax[row, 1].plot(np.linspace(1, curr_step, curr_step), timing['Nested_Sampling'], linestyle="dotted", marker="x", label="Nested Sampling")
    ax[row, 1].plot(np.linspace(1, curr_step, curr_step), timing['Acq_Fnct'], linestyle="dotted", marker="x", label="Acqusition Function")
    ax[row, 1].plot(np.linspace(1, curr_step, curr_step), timing['Likelihood'], linestyle="dotted", marker="x", label="Likelihood Evaluation")
    ax[row, 1].plot(np.linspace(1, curr_step, curr_step), timing['GP_Train'], linestyle="dotted", marker="x", label="GP Training")
    ax[row, 1].plot(np.linspace(1, curr_step, curr_step-1), timing['Plot'], linestyle="dotted", marker="x", label="Plotting")
    ax[row, 1].plot(np.linspace(1, curr_step, curr_step-1), timing['Converge_Check'], linestyle="dotted", marker="x", label="Convergence Check")
    ax[row, 1].set_title("Timing")
    ax[row, 1].legend()

    return ax


