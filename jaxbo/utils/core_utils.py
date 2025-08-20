from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
import numpy as np
import jax.numpy as jnp
from jax import vmap
from scipy.special import erfc
from scipy.stats import chi2, entropy
from .logging_utils import get_logger
from .seed_utils import get_numpy_rng
import math
log = get_logger("[utils]")

#----Convergence KL----
    
def compute_successive_kl(prev_loglike, curr_loglike, log_weights):
    """Compute KL divergence between successive iterations."""
    
    # Convert to normalised probability distributions
    weights = np.exp(log_weights - np.max(log_weights))
    weights = weights / np.sum(weights)
        
    prev_norm = prev_loglike - np.max(prev_loglike)
    curr_norm = curr_loglike - np.max(curr_loglike)
    
    p_prev = np.exp(prev_norm) * weights
    p_curr = np.exp(curr_norm) * weights
        
    p_prev = p_prev / np.sum(p_prev)
    p_curr = p_curr / np.sum(p_curr)
        
    # Forward and reverse KL
    kl_forward = entropy(p_prev, p_curr)
    kl_reverse = entropy(p_curr, p_prev)
    kl_sym = 0.5 * (kl_forward + kl_reverse)
        
    return {
            'forward': kl_forward,
            'reverse': kl_reverse,
            'symmetric': kl_sym
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

# # use this to suppress unecessary output, https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
# @contextmanager
# def suppress_stdout_stderr():
#     """A context manager that redirects stdout and stderr to devnull"""
#     with open(devnull, 'w') as fnull:
#         with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
#             yield (err, out)

