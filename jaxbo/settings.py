"""
Default settings for JaxBo modules.

This module provides centralized configuration for all JaxBo components,
including the main BOBE sampler, Gaussian Processes, classifiers,
nested sampling, and optimization routines.

Usage:
    from jaxbo.settings import BOBESettings, GPSettings, ClassifierSettings
    
    # Use default settings
    bobe = BOBE(loglikelihood=likelihood, **BOBESettings().to_dict())
    
    # Customize settings
    custom_settings = BOBESettings(maxiters=2000, use_clf=False)
    bobe = BOBE(loglikelihood=likelihood, **custom_settings.to_dict())
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Union


@dataclass
class BOBESettings:
    """
    Default settings for the main BOBE (Bayesian Optimization Bayesian Evidence) sampler.
    
    These settings control the overall behavior of the iterative optimization loop,
    including convergence criteria, iteration limits, and integration with other modules.
    """
    
    # Initial sampling settings
    n_cobaya_init: int = 4
    """Number of initial points from the cobaya reference distribution when starting a run.
    Only used when the likelihood is an instance of cobaya_loglike, otherwise ignored."""
    
    n_sobol_init: int = 16
    """Number of initial Sobol points for quasi-random initialization when starting a run."""
    
    # Iteration control
    miniters: int = 100
    """Minimum number of iterations before checking convergence."""
    
    maxiters: int = 1000
    """Maximum number of iterations."""
    
    max_gp_size: int = 1200
    """Maximum number of points used to train the GP. 
    If using SVM, this is not the same as the number of points used to train the SVM."""
    
    # Execution control
    resume: bool = False
    """If True, resume from a previous run. The resume_file argument must be provided."""
    
    resume_file: Optional[str] = None
    """The file to resume from. Must be a .npz file containing the training data."""
    
    save: bool = True
    """If True, save the GP training data to a file so that it can be resumed from later."""
    
    # Update frequencies
    fit_step: int = 5
    """Number of iterations between GP refits."""
    
    update_mc_step: int = 5
    """Number of iterations between MC point updates."""
    
    ns_step: int = 10
    """Number of iterations between nested sampling runs."""
    
    # Monte Carlo settings
    num_hmc_warmup: int = 256
    """Number of warmup steps for HMC sampling."""
    
    num_hmc_samples: int = 512
    """Number of samples to draw from the GP."""
    
    mc_points_size: int = 64
    """Number of points to use for the weighted integrated posterior variance acquisition function."""
    
    mc_points_method: str = 'NUTS'
    """Method to use for generating the MC points. Options are 'NUTS', 'NS', or 'uniform'. 
    Recommend to use 'NUTS' for most cases, 'NS' can be a good choice if you expect that the underlying 
    likelihood has a highly complex structure."""
    
    # GP and classifier settings
    lengthscale_priors: str = 'DSLP'
    """Lengthscale priors to use. Options are 'DSLP' or 'SAAS'."""
    
    acq: str = 'WIPV'
    """Acquisition function to use. Currently only 'WIPV' (Weighted Integrated Posterior Variance) is supported."""
    
    use_clf: bool = True
    """If True, use classifier to filter the GP predictions. 
    This is only required for high dimensional problems and when the scale of variation 
    of the likelihood is extremely large. For cosmological likelihoods with nuisance 
    parameters, this is highly recommended."""
    
    clf_type: str = "svm"
    """Type of classifier to use. Options include 'svm', 'nn', 'ellipsoid'."""
    
    clf_use_size: int = 100
    """Minimum size of the classifier training set before the classifier filter is used in the GP."""
    
    clf_update_step: int = 1
    """Number of iterations between classifier updates."""
    
    clf_threshold: int = 250
    """Threshold for initial classifier training labels."""
    
    gp_threshold: int = 5000
    """Threshold for adding points to the GP training set."""
    
    # Convergence settings
    logz_threshold: float = 1.0
    """Threshold for convergence of the nested sampling logz. 
    If the difference between the upper and lower bounds of logz is less than this value, 
    the sampling will end."""
    
    minus_inf: float = -1e5
    """Value to use for minus infinity. This is used to set the lower bound of the loglikelihood."""
    
    # Output settings
    do_final_ns: bool = True
    """Whether to perform a final nested sampling run at the end."""
    
    return_getdist_samples: bool = False
    """Whether to return GetDist MCSamples object or raw samples dictionary."""
    
    seed: Optional[int] = None
    """Random seed for reproducibility."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for passing to BOBE constructor."""
        return asdict(self)


@dataclass
class GPSettings:
    """
    Default settings for Gaussian Process models (DSLP_GP and SAAS_GP).
    
    These settings control the GP kernel, optimization, and hyperparameter bounds.
    """
    
    # Basic GP settings
    noise: float = 1e-8
    """Scalar noise level for the GP."""
    
    kernel: str = "rbf"
    """Kernel type for the GP. Options are 'rbf' or 'matern'."""
    
    optimizer: str = "adam"
    """Optimizer type for the GP hyperparameter optimization."""
    
    # Hyperparameter bounds (in log10 space)
    outputscale_bounds: list = None
    """Bounds for the output scale of the GP (in log10 space)."""
    
    lengthscale_bounds: list = None
    """Bounds for the length scale of the GP (in log10 space)."""
    
    # Optimization settings
    fit_lr: float = 1e-2
    """Learning rate for GP hyperparameter optimization."""
    
    fit_maxiter: int = 150
    """Maximum number of iterations for GP fitting."""
    
    fit_n_restarts: int = 4
    """Number of restarts for GP optimization."""
    
    resume_fit_maxiter: int = 100
    """Maximum iterations for GP fitting when resuming."""
    
    resume_fit_n_restarts: int = 2
    """Number of restarts for GP optimization when resuming."""
    
    def __post_init__(self):
        """Set default bounds if not provided."""
        if self.outputscale_bounds is None:
            self.outputscale_bounds = [-4, 4]
        if self.lengthscale_bounds is None:
            self.lengthscale_bounds = [np.log10(0.05), 2]
        # Only set tausq_bounds if it exists (for SAAS GP subclass)
        if hasattr(self, 'tausq_bounds') and self.tausq_bounds is None:
            self.tausq_bounds = [-4, 4]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)


@dataclass
class SAASGPSettings(GPSettings):
    """
    Settings specific to SAAS (Sparse Additive GP) models.
    Inherits from GPSettings and adds SAAS-specific parameters.
    """
    tausq_bounds: list = None
    """Bounds for the tausq parameter of the SAAS GP (in log10 space)."""
    
    def __post_init__(self):
        """Set default bounds if not provided."""
        if self.tausq_bounds is None:
            self.tausq_bounds = [-4, 4]

@dataclass
class ClassifierSettings:
    """
    Default settings for classifier models used in ClassifierGP.
    
    These settings control the behavior of SVM, neural network, and ellipsoid classifiers.
    """
    
    # General classifier settings
    clf_use_size: int = 100
    """Minimum number of points to start using the classifier."""
    
    clf_update_step: int = 5
    """Update classifier every `clf_update_step` points after `clf_use_size` is reached."""
    
    probability_threshold: float = 0.5
    """Threshold for classifier probability/score to consider a point feasible."""
    
    minus_inf: float = -1e5
    """Value used for infeasible predictions."""
    
    clf_threshold: int = 250
    """Threshold for initial classifier training labels."""
    
    gp_threshold: int = 1000
    """Threshold for adding points to the GP training set."""
    
    # SVM-specific settings
    svm_gamma: Union[str, float] = "scale"
    """Gamma parameter for SVM. Can be 'scale', 'auto', or a float value."""
    
    svm_C: float = 1e7
    """Regularization parameter for SVM."""
    
    svm_kernel: str = 'rbf'
    """Kernel type for SVM ('linear', 'poly', 'rbf', 'sigmoid')."""
    
    # Neural network settings
    nn_hidden_dims: list = None
    """Hidden layer dimensions for neural network classifier."""
    
    nn_learning_rate: float = 1e-3
    """Learning rate for neural network training."""
    
    nn_epochs: int = 1000
    """Number of training epochs for neural network."""
    
    nn_batch_size: int = 64
    """Batch size for neural network training."""
    
    nn_val_frac: float = 0.2
    """Fraction of data to use for validation."""
    
    nn_patience: int = 100
    """Early stopping patience for neural network."""
    
    # Ellipsoid classifier settings
    ellipsoid_reg_lambda: float = 1e-3
    """Regularization parameter for ellipsoid classifier."""
    
    def __post_init__(self):
        """Set default values for list parameters."""
        if self.nn_hidden_dims is None:
            self.nn_hidden_dims = [64, 32]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)


@dataclass
class NestedSamplingSettings:
    """
    Default settings for nested sampling with Dynesty and JaxNS.
    
    These settings control the nested sampling runs for evidence computation and posterior sampling.
    """
    
    # Dynesty settings
    dlogz: float = 0.1
    """Log evidence goal for Dynesty."""
    
    dynamic: bool = False
    """Use dynamic nested sampling."""
    
    logz_std: bool = True
    """Compute the upper and lower bounds on logZ using the GP uncertainty."""
    
    maxcall: Optional[int] = int(2e6)
    """Maximum number of function calls for nested sampling."""
    
    boost_maxcall: Optional[int] = 1
    """Boost factor for maximum calls."""
    
    print_progress: bool = True
    """Whether to print progress during nested sampling."""
    
    equal_weights: bool = True
    """Whether to use equal weights for samples."""
    
    sample_method: str = 'rwalk'
    """Sampling method for Dynesty ('rwalk', 'slice', 'rslice', etc.)."""
    
    # Final nested sampling settings
    final_maxcall: int = int(1e7)
    """Maximum calls for final nested sampling run."""
    
    final_dlogz: float = 0.01
    """Tighter convergence criterion for final run."""
    
    final_dynamic: bool = True
    """Use dynamic sampling for final run."""
    
    # JaxNS settings (alternative to Dynesty)
    jaxns_num_live_points: int = 500
    """Number of live points for JaxNS."""
    
    jaxns_max_samples: int = 5e6
    """Maximum samples for JaxNS."""
    
    jaxns_collect_samples: bool = True
    """Whether to collect samples in JaxNS."""
    
    jaxns_termination_frac: float = 0.01
    """Termination fraction for JaxNS."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)


@dataclass
class OptimizationSettings:
    """
    Default settings for optimization routines used in acquisition function optimization.
    
    These settings control the behavior of the Adam optimizer and other optimization parameters.
    """
    
    # General optimization settings
    learning_rate: float = 5e-3
    """Learning rate for optimization."""
    
    maxiter: int = 200
    """Maximum number of optimization iterations."""
    
    n_restarts: int = 4
    """Number of random restarts for optimization."""
    
    verbose: bool = True
    """Whether to show progress during optimization."""
    
    # Adam optimizer specific
    adam_beta1: float = 0.9
    """Beta1 parameter for Adam optimizer."""
    
    adam_beta2: float = 0.999
    """Beta2 parameter for Adam optimizer."""
    
    adam_eps: float = 1e-8
    """Epsilon parameter for Adam optimizer."""
    
    # Convergence settings
    tolerance: float = 1e-6
    """Convergence tolerance for optimization."""
    
    max_line_search_steps: int = 10
    """Maximum steps for line search if used."""
    
    # Bounds handling
    clip_bounds: bool = True
    """Whether to clip parameters to stay within bounds."""
    
    bound_epsilon: float = 1e-8
    """Small epsilon to stay away from exact bounds."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)


@dataclass
class MCMCSettings:
    """
    Default settings for MCMC sampling (NUTS) used in GP posterior sampling.
    
    These settings control the NUTS sampler for drawing samples from the GP posterior.
    """
    
    # NUTS settings
    warmup_steps: int = 512
    """Number of warmup steps for NUTS."""
    
    num_samples: int = 512
    """Number of samples to draw."""
    
    thinning: int = 4
    """Thinning factor for samples."""
    
    num_chains: int = 1
    """Number of parallel chains."""
    
    # NUTS-specific parameters
    target_accept_prob: float = 0.8
    """Target acceptance probability for NUTS."""
    
    max_tree_depth: int = 10
    """Maximum tree depth for NUTS."""
    
    dense_mass: bool = False
    """Whether to use dense mass matrix."""
    
    # Initialization
    init_strategy: str = "median"
    """Initialization strategy ('median', 'uniform', 'prior')."""
    
    progress_bar: bool = True
    """Whether to show progress bar during sampling."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)


@dataclass
class LoggingSettings:
    """
    Default settings for the JaxBo logging system.
    
    These settings control the verbosity and output format of logging messages.
    """
    
    verbosity: str = 'INFO'
    """Logging verbosity level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'QUIET')."""
    
    log_file: Optional[str] = None
    """Optional file to log to."""
    
    log_file_max_bytes: int = 10 * 1024 * 1024  # 10MB
    """Maximum size of log file before rotation."""
    
    log_file_backup_count: int = 5
    """Number of backup log files to keep."""
    
    format_stdout: str = '%(asctime)s [%(name)s] %(message)s'
    """Format string for stdout messages."""
    
    format_stderr: str = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    """Format string for stderr messages."""
    
    format_file: str = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    """Format string for file logging."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)


@dataclass
class JaxBoSettings:
    """
    Master settings class containing all JaxBo module settings.
    
    This provides a convenient way to configure all aspects of JaxBo from a single object.
    """
    
    bobe: BOBESettings = None
    gp: GPSettings = None
    classifier: ClassifierSettings = None
    nested_sampling: NestedSamplingSettings = None
    optimization: OptimizationSettings = None
    mcmc: MCMCSettings = None
    logging: LoggingSettings = None
    
    def __post_init__(self):
        """Initialize sub-settings with defaults if not provided."""
        if self.bobe is None:
            self.bobe = BOBESettings()
        if self.gp is None:
            self.gp = GPSettings()
        if self.classifier is None:
            self.classifier = ClassifierSettings()
        if self.nested_sampling is None:
            self.nested_sampling = NestedSamplingSettings()
        if self.optimization is None:
            self.optimization = OptimizationSettings()
        if self.mcmc is None:
            self.mcmc = MCMCSettings()
        if self.logging is None:
            self.logging = LoggingSettings()
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert all settings to nested dictionary."""
        return {
            'bobe': self.bobe.to_dict(),
            'gp': self.gp.to_dict(),
            'classifier': self.classifier.to_dict(),
            'nested_sampling': self.nested_sampling.to_dict(),
            'optimization': self.optimization.to_dict(),
            'mcmc': self.mcmc.to_dict(),
            'logging': self.logging.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Dict[str, Any]]) -> 'JaxBoSettings':
        """Create JaxBoSettings from nested dictionary."""
        return cls(
            bobe=BOBESettings(**config_dict.get('bobe', {})),
            gp=GPSettings(**config_dict.get('gp', {})),
            classifier=ClassifierSettings(**config_dict.get('classifier', {})),
            nested_sampling=NestedSamplingSettings(**config_dict.get('nested_sampling', {})),
            optimization=OptimizationSettings(**config_dict.get('optimization', {})),
            mcmc=MCMCSettings(**config_dict.get('mcmc', {})),
            logging=LoggingSettings(**config_dict.get('logging', {})),
        )


# Convenience functions for quick access to common configurations

def get_default_settings() -> JaxBoSettings:
    """Get default settings for all JaxBo modules."""
    return JaxBoSettings()


def get_high_dimensional_settings() -> JaxBoSettings:
    """Get settings optimized for high-dimensional problems (>10 dimensions)."""
    settings = JaxBoSettings()
    
    # Adjust BOBE settings for high-dimensional problems
    settings.bobe.use_clf = True
    settings.bobe.clf_threshold = 500
    settings.bobe.gp_threshold = 10000
    settings.bobe.ns_step = 15
    settings.bobe.mc_points_size = 128
    
    # More conservative GP settings
    settings.gp.fit_maxiter = 200
    settings.gp.fit_n_restarts = 6
    
    # More thorough nested sampling
    settings.nested_sampling.maxcall = int(1e7)
    
    return settings


def get_fast_settings() -> JaxBoSettings:
    """Get settings optimized for faster execution (less accurate)."""
    settings = JaxBoSettings()
    
    # Reduce iterations and sampling
    settings.bobe.maxiters = 500
    settings.bobe.ns_step = 20
    settings.bobe.num_hmc_samples = 256
    settings.bobe.mc_points_size = 32
    
    # Faster GP fitting
    settings.gp.fit_maxiter = 100
    settings.gp.fit_n_restarts = 2
    
    # Less thorough nested sampling
    settings.nested_sampling.maxcall = int(1e6)
    settings.nested_sampling.dlogz = 0.5
    
    return settings


def get_accurate_settings() -> JaxBoSettings:
    """Get settings optimized for maximum accuracy (slower execution)."""
    settings = JaxBoSettings()
    
    # More iterations and sampling
    settings.bobe.maxiters = 3000
    settings.bobe.ns_step = 5
    settings.bobe.num_hmc_samples = 1024
    settings.bobe.mc_points_size = 128
    
    # More thorough GP fitting
    settings.gp.fit_maxiter = 300
    settings.gp.fit_n_restarts = 8
    
    # More thorough nested sampling
    settings.nested_sampling.maxcall = int(2e7)
    settings.nested_sampling.dlogz = 0.01
    settings.nested_sampling.final_dlogz = 0.001
    
    return settings
