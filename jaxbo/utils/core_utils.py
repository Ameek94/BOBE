from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
import numpy as np
import jax.numpy as jnp
from jax import vmap
from scipy.special import logsumexp, erfc
from scipy import stats
from scipy.stats import chi2
from .logging_utils import get_logger
from .seed_utils import get_numpy_rng
import math
log = get_logger("utils")

def renormalise_log_weights(log_weights):
    log_total = logsumexp(log_weights)
    normalized_weights = np.exp(log_weights - log_total)
    return normalized_weights

def resample_equal(samples, aux, weights=None, logwts=None):
    rstate = get_numpy_rng()
    # Resample samples to obtain equal weights. Taken from jaxns
    if logwts is not None:
        wts = renormalise_log_weights(logwts)
    else:
        wts = weights
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


#----Convergence KL----
    
def kl_divergence_samples(prev_loglike, curr_loglike):
    """Compute KL divergence between successive iterations."""
    
        
    prev_norm = prev_loglike - np.max(prev_loglike)
    curr_norm = curr_loglike - np.max(curr_loglike)

    p_prev = np.exp(prev_norm)
    p_curr = np.exp(curr_norm)
        
    p_prev = p_prev / np.sum(p_prev)
    p_curr = p_curr / np.sum(p_curr)
        
    # Forward and reverse KL
    kl_forward = stats.entropy(p_prev, p_curr)
    kl_reverse = stats.entropy(p_curr, p_prev)
    kl_sym = 0.5 * (kl_forward + kl_reverse)
        
    return {
            'forward': kl_forward,
            'reverse': kl_reverse,
            'symmetric': kl_sym
    }


def _kl_gaussian_single(mu1, Cov1, mu2, Cov2):
    """
    Computes the KL divergence KL(N1 || N2) between two multivariate Gaussians
    N1 = N(mu1, Cov1) and N2 = N(mu2, Cov2).
    """
    d = mu1.shape[0]

    # Log determinant term
    _, logdet_Cov1 = np.linalg.slogdet(Cov1)
    _, logdet_Cov2 = np.linalg.slogdet(Cov2)
    log_det_term = logdet_Cov2 - logdet_Cov1

    # Trace term 
    trace_term = np.trace(np.linalg.solve(Cov2, Cov1))

    # Quadratic term
    diff = mu2 - mu1
    quad_term = np.dot(diff, np.linalg.solve(Cov2, diff))

    # Combine terms
    kl_div = 0.5 * (log_det_term - d + trace_term + quad_term)
    
    return kl_div


def kl_divergence_gaussian(mu1, Cov1, mu2, Cov2):
    """
    Computes the forward, reverse, and symmetric KL divergence between two 
    multivariate Gaussian distributions N1=N(mu1, Cov1) and N2=N(mu2, Cov2).
    """
    kl_forward = _kl_gaussian_single(mu1, Cov1, mu2, Cov2)  # KL(N1 || N2)
    kl_reverse = _kl_gaussian_single(mu2, Cov2, mu1, Cov1)  # KL(N2 || N1)
    kl_symmetric = 0.5 * (kl_forward + kl_reverse)

    return {
        'forward': kl_forward,
        'reverse': kl_reverse,
        'symmetric': kl_symmetric
    }



#----Misc----

# Classifier threshold util
def get_threshold_for_nsigma(nsigma,d):
    """
    Difference between peak of Gaussian and logprob level for nsigma (taken from GPry).

    Arguments
    ---------
    nsigma : float
        The number of standard deviations to consider.
    d : int
        The dimensionality of the space.

    Returns
    -------
    float
        The threshold value.
    """
    nstd = np.sqrt(chi2.isf(erfc(nsigma / np.sqrt(2)), d))
    return 0.5 * nstd ** 2

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

# use this to suppress unecessary output, https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

