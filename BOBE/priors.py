from math import sqrt
import jax.numpy as jnp
import jax
import numpyro.distributions as dist
from .utils.log import get_logger
log = get_logger("prior")

jax.config.update("jax_enable_x64", True)

sqrt2 = sqrt(2.)
sqrt3 = sqrt(3.)

class DummyDistribution:
    """A dummy distribution that always returns log_prob = 0.0"""
    def log_prob(self, x):
        return 0.0

def make_distribution(spec: dict) -> dist.Distribution:
    """
    Turn a dictionary specification into a NumPyro distribution.
    
    Parameters
    ----------
    spec : dict
        Dictionary with 'name' key for distribution type and additional
        keyword arguments for the distribution parameters.
        
    Returns
    -------
    dist.Distribution
        NumPyro distribution object.
        
    Examples
    --------
    >>> spec = {"name": "Normal", "loc": 0.0, "scale": 1.0}
    >>> dist = make_distribution(spec)
    """
    # Ensure distribution exists
    dist_class = getattr(dist, spec["name"], None)
    if dist_class is None:
        raise ValueError(f"Distribution {spec['name']} not found in numpyro.distributions.")
    
    # Remove "name"
    kwargs = {k: v for k, v in spec.items() if k != "name"}
    return dist_class(**kwargs)

def saas_prior_logprob(lengthscales, kernel_variance, tausq):
    """
    Compute SAAS prior log probability.
    
    Parameters
    ----------
    lengthscales : jnp.ndarray
        Lengthscale parameters.
    kernel_variance : float
        Kernel variance parameter.
    tausq : float
        SAAS tausq parameter.
        
    Returns
    -------
    float
        Log probability under SAAS priors.
    """
    logprior = dist.LogNormal(0., 1.).log_prob(kernel_variance)
    logprior += dist.HalfCauchy(0.1).log_prob(tausq)
    inv_lengthscales_sq = 1 / (tausq * lengthscales**2)
    logprior += jnp.sum(dist.HalfCauchy(1.).log_prob(inv_lengthscales_sq))
    return logprior

def dslp_lengthscale_dist(ndim: int) -> dist.Distribution:
    return dist.LogNormal(loc=sqrt2 + 0.5 * jnp.log(ndim), scale=sqrt3)

def dsp_unscaled_lengthscale_dist() -> dist.Distribution:
    return dist.LogNormal(loc=sqrt2 + jnp.log(1) * 0.5, scale=sqrt3)

def build_prior_state(ndim: int, kernel_variance_prior_spec, kernel_variance_bounds, lengthscale_prior_spec,  lengthscale_bounds):
    # Kernel variance prior
    if kernel_variance_prior_spec is None:
        kernel_variance_prior_spec = {"name": "Uniform", 'low': kernel_variance_bounds[0], 'high': kernel_variance_bounds[1]}
    fixed_kernel_variance = (kernel_variance_prior_spec == 'fixed')
    kernel_variance_dist = DummyDistribution() if fixed_kernel_variance else make_distribution(kernel_variance_prior_spec)

    # Lengthscale prior
    
    if lengthscale_prior_spec is None:
        lengthscale_prior_spec= {"name": "Uniform", 'low': lengthscale_bounds[0], 'high': lengthscale_bounds[1]}

    tausq_enabled = False
    lengthscale_dist = None
    if lengthscale_prior_spec == "DSLP":
        log.info("DSLP prior chosen")
        lengthscale_prior = "DSLP"
        lengthscale_dist = dslp_lengthscale_dist(ndim)

        def logprior_fn(lengthscales, kernel_variance, tausq):
            lp = kernel_variance_dist.log_prob(kernel_variance)
            lp += jnp.sum(lengthscale_dist.log_prob(lengthscales))
            return lp
        
    elif lengthscale_prior_spec == "SAAS":
        log.info("SAAS prior chosen")
        lengthscale_prior = "SAAS"
        lengthscale_dist = None
        tausq_enabled = True

        def logprior_fn(lengthscales, kernel_variance, tausq):
            return saas_prior_logprob(lengthscales, kernel_variance, tausq)
    
    elif lengthscale_prior_spec == "dsp_unscaled":
        log.info("DSP Unscaled prior chosen")
        lengthscale_prior = "DSP_UNSCALED"
        lengthscale_dist = dsp_unscaled_lengthscale_dist()

        def logprior_fn(lengthscales, raw_coeffs, raw_global_lengthscale):
            lp = jnp.sum(lengthscale_dist.log_prob(lengthscales))

            return lp

    else:
        log.info("Custom prior chosen")
        lengthscale_prior = "CUSTOM"
        lengthscale_dist = make_distribution(lengthscale_prior_spec)
        tausq_enabled = False

        def logprior_fn(lengthscales, kernel_variance, tausq):
            lp = kernel_variance_dist.log_prob(kernel_variance)
            lp += jnp.sum(lengthscale_dist.log_prob(lengthscales))
            return lp

    return {
        "fixed_kernel_variance": fixed_kernel_variance,
        "kernel_variance_dist": kernel_variance_dist,
        "lengthscale_prior": lengthscale_prior,
        "lengthscale_dist": lengthscale_dist,
        "tausq_enabled": tausq_enabled,
        "kernel_variance_prior_spec": kernel_variance_prior_spec,
        "lengthscale_prior_spec": lengthscale_prior_spec,
        "logprior_fn": logprior_fn,
    }


class Prior:
    """
    A prior that knows how to configure itself from kernel.prior_spec and kernel.bounds_spec, and how to score hyperparameters in natural space (NOT optimiser/log space)
    """

    def configure(self, kernel):
        """
        Reads kernel.prior_spec and kernel.bounds_spec and stores whatever it needss.
        May also set kernel flags (e.g fixed_kernel_variance, tausq_enabled).
        """
        raise NotImplementedError
    def extra_hyperparams(self, kernel):
        #default: none
        return ()
    
    def logprior(self,  **kwargs):
        """
        Returns log p(hyperparams). Inputs are natural-space values
        """
        raise NotImplementedError

class DSLPPrior(Prior):
    def __init__(self):
        self.ndim = None
        self.lengthscale_bounds = None
        self.kernel_variance_bounds = None

        self.kernel_variance_prior_spec = None
        self.fixed_kernel_variance = False

        self._lengthscale_dist = None
        self._kernel_variance_dist = None

        self._kernel_variance_logprior = None

    def configure(self, kernel):
        self.ndim = kernel.ndim
        self.lengthscale_bounds = kernel.bounds_spec.get("lengthscales")
        self.kernel_variance_bounds = kernel.bounds_spec.get("kernel_variance")

        ps = kernel.prior_spec or {}
        self.kernel_variance_prior_spec = ps.get("kernel_variance", None)
        self.fixed_kernel_variance = bool(ps.get("fixed_kernel_variance", False))

        # Push flags to kernel for layout decisions
        kernel.fixed_kernel_variance = self.fixed_kernel_variance

        # Build cached dists
        self.lengthscale_dist = dist.LogNormal(loc=sqrt2 + 0.5 * jnp.log(self.ndim), scale=sqrt3)

        if self.fixed_kernel_variance:
            self.kernel_variance_dist = DummyDistribution()
            self._kernel_variance_logprior = lambda kernel_variance: 0.0
        else:
            if self.kernel_variance_prior_spec is None:
                if self.kernel_variance_bounds is None:
                    raise ValueError("DSLP: Must specify either kernel_variance bounds or kernel_variance prior")
                lo, hi = self.kernel_variance_bounds[0], self.kernel_variance_bounds[1]
                self.kernel_variance_prior_spec = {"name": "Uniform", 'low': lo, 'high': hi}
            self.kernel_variance_dist = make_distribution(self.kernel_variance_prior_spec)
            self._kernel_variance_logprior = lambda kernel_variance: self.kernel_variance_dist.log_prob(kernel_variance)

        return self

    def logprior(self, *, lengthscales, kernel_variance = None, **kwargs):
        lp = jnp.sum(self.lengthscale_dist.log_prob(lengthscales))
        lp += self._kernel_variance_logprior(kernel_variance)
        return lp
