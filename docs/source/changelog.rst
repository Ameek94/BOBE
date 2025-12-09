Changelog
=========

All notable changes to BOBE will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[0.1.0] - 2025-12-XX
--------------------

Initial release of BOBE (Bayesian Optimization for Bayesian Evidence).

Added
~~~~~
- Bayesian optimization for efficient Bayesian evidence estimation
- JAX-powered Gaussian Process surrogates with RBF and Matérn kernels
- GP priors: DSLP (Deep Sigmoidal Location Process) and SAAS (Sparse Axis-Aligned Subspace)
- Classifier-augmented GP (SVM, neural network, ellipsoid) for filtering low-likelihood regions
- Acquisition functions: WIPV, WIPStd, EI, LogEI
- Nested sampling via Dynesty for evidence computation on GP surrogate
- Cobaya integration for cosmological likelihoods
- MPI parallelization for likelihood evaluations and GP fitting
- Result saving and visualization utilities
- Convergence monitoring via KL divergence and log-evidence thresholds
