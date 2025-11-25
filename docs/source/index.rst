JaxBo Documentation
===================

**JaxBo** is a Bayesian Optimization package for efficiently estimating Bayesian evidence and parameter posteriors, 
designed for expensive likelihoods such as those in cosmology. 

Key Features
------------

- **JAX-powered**: Automatic differentiation, jit compilation, and in the future GPU/TPU acceleration
- **Bayesian inference**: Specialized for Bayesian model comparison and parameter inference
- **Cobaya integration**: Works seamlessly with cosmological likelihoods through the cobaya interface
- **MPI support**: Parallel likelihood evaluations and gp fitting
- **Flexible**: Works with any likelihood function that can be called from Python

Getting Started
---------------

.. toctree::
   :maxdepth: 1

   installation
   quickstart
   examples/index

Indices
-------

* :ref:`genindex`
* :ref:`search`
