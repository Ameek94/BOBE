.. JaxBo documentation master file

JaxBo: Bayesian Optimization for Bayesian Evidence
===================================================

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/Ameek94/JaxBo/blob/main/LICENSE
   :alt: License

**JaxBo** is a high-performance Bayesian Optimization package designed specifically for cosmology applications, 
built on top of JAX for automatic differentiation and GPU acceleration. It provides tools for efficiently 
estimating Bayesian evidence using advanced Gaussian Process models and acquisition functions.

Key Features
------------

- 🚀 **JAX-powered**: Built on JAX for automatic differentiation and GPU/TPU acceleration
- 🎯 **Cosmology-focused**: Designed specifically for cosmological parameter estimation
- 🔬 **Advanced GP Models**: Support for Deep Sigmoidal Location Process (DSLP) and Sparse Axis-Aligned Subspace (SAAS) Gaussian Processes
- 📊 **Multiple Acquisition Functions**: Including Weighted Integrated Posterior Variance (WIPV) and Expected Improvement (EI)
- 🔗 **Cobaya Integration**: Seamless integration with the Cobaya cosmological parameter estimation package
- 📈 **Comprehensive Analysis**: Built-in tools for convergence analysis, plotting, and result visualization
- ⚡ **Parallel Computing**: MPI support for distributed computing

Quick Start
-----------

Installation
~~~~~~~~~~~~

Install JaxBo from the package directory:

.. code-block:: bash

   python -m pip install .

For development installation:

.. code-block:: bash

   python -m pip install -e .

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from jaxbo import BOBE
   from jaxbo.likelihood import CobayaLikelihood
   
   # Define your likelihood
   likelihood = CobayaLikelihood(info_dict)
   
   # Create BOBE instance
   bobe = BOBE(
       loglikelihood=likelihood,
       max_eval_budget=1000,
       acquisition_func="wipv"
   )
   
   # Run optimization
   results = bobe.run()

Package Structure
-----------------

The JaxBo package is organized into several main modules:

- **bo.py**: Core Bayesian Optimization implementation (BOBE class)
- **gp.py**: Gaussian Process models including DSLP and SAAS variants
- **clf_gp.py**: Gaussian Processes with classification for constraint handling
- **acquisition.py**: Acquisition functions (WIPV, EI, LogEI)
- **likelihood.py**: Likelihood interfaces including Cobaya integration
- **nested_sampler.py**: Nested sampling utilities
- **utils/**: Utility modules for results, plotting, convergence analysis, and more

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/gp_models
   api/acquisition
   api/likelihood
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   contributing
   development
   changelog

.. toctree::
   :maxdepth: 1
   :caption: Reference

   bibliography
   glossary

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
========

If you use JaxBo in your research, please cite:

.. code-block:: bibtex

   @software{jaxbo2025,
     title={JaxBo: Bayesian Optimization for Bayesian Evidence using JAX},
     author={Malhotra, Ameek and Cohen, Nathan and Hamann, Jan},
     year={2025},
     url={https://github.com/Ameek94/JaxBo}
   }

License
=======

JaxBo is released under the MIT License. See the LICENSE file for details.

Contact
=======

- **Ameek Malhotra**: ameekmalhotra@gmail.com
- **Nathan Cohen**: nathan.cohen@unsw.edu.au
- **Jan Hamann**

For bug reports and feature requests, please use the `GitHub Issues <https://github.com/Ameek94/JaxBo/issues>`_ page.
