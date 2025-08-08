from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
import numpy as np
from getdist import plots,MCSamples,loadMCSamples
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap
from scipy.special import logsumexp
from .logging_utils import get_logger
from .seed_utils import get_numpy_rng
log = get_logger("[utils]")

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

def scale_to_unit(x,param_bounds):
    """
    Project from original domain to unit hypercube, X is N x d shaped, param_bounds are 2 x d
    """
    x =  (x - param_bounds[0])/(param_bounds[1] - param_bounds[0])
    return x

def scale_from_unit(x,param_bounds):
    """
    Project from unit hypercube to original domain, X is N x d shaped, param_bounds are 2 x d
    """
    x = x * (param_bounds[1] - param_bounds[0]) + param_bounds[0]
    return x

def renormalise_log_weights(log_weights):
    log_total = logsumexp(log_weights)
    normalized_weights = np.exp(log_weights - log_total)
    return normalized_weights

def resample_equal(samples, aux, logwts):
    rstate = get_numpy_rng()
    # Resample samples to obtain equal weights. Taken from jaxns
    wts = renormalise_log_weights(logwts)
    weights = wts / wts.sum()
    cumulative_sum = np.cumsum(weights)
    cumulative_sum /= cumulative_sum[-1]
    nsamples = len(weights)
    positions = (rstate.random() + np.arange(nsamples)) / nsamples
    idx = np.zeros(nsamples, dtype=int)
    i, j = 0, 0
    while i < nsamples:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    perm = rstate.permutation(nsamples)
    resampled_samples = samples[idx][perm]
    resampled_aux = aux[idx][perm]
    return resampled_samples, resampled_aux

def plot_final_samples(gp,samples_dict,param_list,param_labels,plot_params=None,param_bounds=None,
                       reference_samples = None,
                       reference_file = None, reference_ignore_rows=0.,reference_label='MCMC',
                       scatter_points=False,markers=None,output_file='output'):
    """
    Plot the final samples from the Bayesian optimization process.

    Arguments
    ----------
    gp : GP object
        The Gaussian process object used for the optimization.
    samples_dict : dict
        The samples from the nested sampling or MCMC process.
    param_list : list
        The list of parameter names.
    param_labels : list
        The list of parameter labels for plotting.
    plot_params : list, optional
        The list of parameters to plot. If None, all parameters will be plotted.
    param_bounds : np.ndarray, optional
        The bounds of the parameters. If None, assumed to be [0,1] for all parameters.
    reference_samples : MCSamples, optional
        The reference getdist MCsamples from the MCMC/Nested Sampling to comparea gainst. 
        If None, will be loaded from the reference_file.
    reference_file : str, optional
        The getdist file root containing the reference samples. If None, will be loaded from the reference_samples.
        If both are None, no reference samples will be plotted.
    reference_ignore_rows : float, optional
        The fraction of rows to ignore in the reference file. Default is 0.0.
    reference_label : str, optional
        The label for the reference samples. Default is 'MCMC'.
    scatter_points : bool, optional
        If True, scatter the training points on the plot. Default is False.
    output_file : str, optional
        The output file name for the plot. Default is 'output'.
    """

    if plot_params is None:
        plot_params = param_list
    ranges = dict(zip(param_list,param_bounds.T))

    samples = samples_dict['x']

    if param_bounds is None:
        param_bounds = np.array([[0,1]]*len(param_list)).T
    # samples = scale_from_unit(samples,param_bounds)
    weights = samples_dict['weights']
    gd_samples = MCSamples(samples=samples, names=param_list, labels=param_labels, 
                           ranges=ranges, weights=weights)
    

    plot_samples = [gd_samples]

    if reference_file is not None:
        ref_samples = loadMCSamples(reference_file,settings={'ignore_rows': reference_ignore_rows})
        plot_samples.append(ref_samples)
    elif reference_samples is not None:
        plot_samples.append(reference_samples)

    labels = ['GP',reference_label]

    for label,s in zip(labels,plot_samples):
        log.info(f"Parameter limits from {label}")
        for key in plot_params:
            log.info(s.getInlineLatex(key,limit=1))
    
    ndim = len(plot_params)

    g = plots.get_subplot_plotter(subplot_size=2.5,subplot_size_ratio=1)
    g.settings.legend_fontsize = 22
    g.settings.axes_fontsize=18
    g.settings.axes_labelsize=18
    g.settings.title_limit_fontsize = 14   
    g.triangle_plot(plot_samples, params = plot_params,filled=[True,False],
                    contour_colors=['#006FED', 'black'],contour_lws=[1,1.5],
                    legend_labels=['GP',f'{reference_label}']
                    ,markers=markers,marker_args={'lw': 1, 'ls': ':'}) 
    if scatter_points:
        points = scale_from_unit(gp.train_x,param_bounds)
        for i in range(ndim):
            # ax = g.subplots[i,i]
            for j in range(i+1,ndim):
                ax = g.subplots[j,i]
                ax.scatter(points[:,i],points[:,j],alpha=0.5,color='forestgreen',s=5)

    g.export(output_file+'_samples.pdf')