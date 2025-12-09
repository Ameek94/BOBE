BOBE
====

BOBE (Bayesian Optimisation for Bayesian Evidence) is a JAX-powered Bayesian Optimisation package for efficiently estimating the Bayesian evidence 
for expensive likelihood functions. It is designed primarily for cosmological applications but works 
with any likelihood that can be called from Python. BOBE trains a Gaussian Process surrogate for the expensive likelihood and runs nested sampling on the 
surrogate to compute Bayesian Evidence (and parameter posteriors) efficiently. The code is implemented in JAX for high performance and automatic differentiation support. 
BOBE is available for download from `GitHub <https://github.com/Ameek94/BOBE>`_ and its core algorithm is explained in detail in our accompanying `paper <https://arxiv.org/abs/2512.xxxxx>`_.
We welcome bug reports, patches, feature requests, and other comments via the 
`GitHub issue tracker <https://github.com/Ameek94/BOBE/issues>`_.


.. This documentation won't teach you Bayesian Optimization or nested sampling in detail, but there 
.. are many excellent resources available. We recommend checking out the references in our bibliography.

When to use BOBE
----------------

BOBE is designed for scenarios where the likelihood evaluation is costly (e.g., takes more than a second per evaluation). 
In such cases, standard nested sampling or MCMC methods become computationally prohibitive since they typically require thousands to millions of likelihood evaluations. 

Use BOBE if:

- **Your likelihood function is expensive to evaluate**: Each evaluation takes more than ~1 second (e.g., due to slow forward models or complex likelihood calculations)
- **You need Bayesian evidence estimates**: For model comparison via Bayes factors
- **You want posterior samples efficiently**: BOBE naturally provides posterior samples alongside evidence estimates 

BOBE works best for problems with up to ~20 parameters, although this can vary based on the specific problem and likelihood structure. 
It has been tested to work well upto 30 dimensions for simple multivariate Gaussian likelihoods and upto 16 dimensions for cosmological likelihoods (LCDM+ :math:`\Omega_k` with the Planck Camspec likelihood).
BOBE may not be necessary if your likelihood already evaluates in milliseconds, as the overhead of training the GP surrogate and running Bayesian optimisation
may not be worth the speedup. For such cases, there is generally no need to go beyond standard MCMC/nested sampling tools.


Getting Started
---------------------

To begin, follow the :doc:`installation` guide to get BOBE installed on your system. 
After that, you can learn most of what you need from the examples section (start with 
:doc:`examples/quickstart` and continue from there). If you need more details about any specific functionality, 
the User Guide should have you covered.

.. toctree::
   :maxdepth: 1
   :caption: Install

   installation

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/quickstart
   examples/detailed_usage
   examples/cosmology

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user guide/gp
   user guide/acquisition
   user guide/convergence
   user guide/advanced
   user guide/faq

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   bibliography
   changelog
   contributing

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
