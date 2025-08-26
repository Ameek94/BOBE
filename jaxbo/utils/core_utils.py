from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
import numpy as np
import jax.numpy as jnp
from jax import vmap
from scipy.special import logsumexp
from scipy import stats
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

def compute_kl_divergences(mean_ll, upper_ll, lower_ll, log_weights):
    """
    Compute KL divergences between all combinations.
        
    Returns:
    --------
    dict with KL divergence results
    """
    # Convert to probability distributions
    # Normalize using log weights
    weights = np.exp(log_weights - np.max(log_weights))
    weights = weights / np.sum(weights)
        
    # Normalize likelihoods relative to their max for numerical stability
    mean_norm = mean_ll - np.max(mean_ll)
    upper_norm = upper_ll - np.max(upper_ll)
    lower_norm = lower_ll - np.max(lower_ll)
        
    # Convert to probabilities
    p_mean = np.exp(mean_norm) * weights
    p_upper = np.exp(upper_norm) * weights
    p_lower = np.exp(lower_norm) * weights
        
    # Normalize probability distributions
    p_mean = p_mean / np.sum(p_mean)
    p_upper = p_upper / np.sum(p_upper)
    p_lower = p_lower / np.sum(p_lower)
        
    # Compute KL divergences
    kl_results = {}
        
    # KL between mean and bounds
    kl_results['mean_upper'] = stats.entropy(p_mean, p_upper)
    kl_results['mean_lower'] = stats.entropy(p_mean, p_lower)
    kl_results['upper_mean'] = stats.entropy(p_upper, p_mean)
    kl_results['lower_mean'] = stats.entropy(p_lower, p_mean)
        
    # KL between bounds
    kl_results['upper_lower'] = stats.entropy(p_upper, p_lower)
    kl_results['lower_upper'] = stats.entropy(p_lower, p_upper)
        
    # Symmetrized versions
    kl_results['sym_mean_upper'] = 0.5 * (kl_results['mean_upper'] + kl_results['upper_mean'])
    kl_results['sym_mean_lower'] = 0.5 * (kl_results['mean_lower'] + kl_results['lower_mean'])
    kl_results['sym_upper_lower'] = 0.5 * (kl_results['upper_lower'] + kl_results['lower_upper'])
        
    return kl_results
    
def compute_successive_kl(prev_loglike, curr_loglike, log_weights):
    """Compute KL divergence between successive iterations."""
    # Convert to probability distributions
    weights = np.exp(log_weights - np.max(log_weights))
    weights = weights / np.sum(weights)
        
    prev_norm = prev_loglike - np.max(prev_loglike)
    curr_norm = curr_loglike - np.max(curr_loglike)
    
    p_prev = np.exp(prev_norm) * weights
    p_curr = np.exp(curr_norm) * weights
        
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

#----Misc----

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

