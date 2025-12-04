BOBE
====

BOBE is a JAX-powered Bayesian Optimization package for efficiently estimating Bayesian evidence 
from expensive likelihood functions. It's designed primarily for cosmological applications but works 
with any likelihood that can be called from Python.

BOBE trains Gaussian Process surrogates for expensive likelihoods and runs nested sampling on the 
surrogate to compute Bayesian Evidence efficiently. This documentation will show you how to use it.

This documentation won't teach you Bayesian Optimization or nested sampling in detail, but there 
are many excellent resources available. We recommend checking out the references in our bibliography.

BOBE is being actively developed on `GitHub <https://github.com/Ameek94/BOBE>`_.

Basic Usage
-----------

If you wanted to compute the Bayesian evidence for a simple likelihood, you would do something like:

.. code-block:: python

   import numpy as np
   from BOBE.bo import BOBE
   from BOBE.likelihood import Likelihood
   
   class MyLikelihood(Likelihood):
       def __call__(self, theta):
           # Your likelihood function here
           return -0.5 * np.sum((theta - 0.5) ** 2 / 0.1 ** 2)
   
   # Setup
   likelihood = MyLikelihood(
       param_names=['x', 'y', 'z'],
       bounds=[(0, 1), (0, 1), (0, 1)]
   )
   
   bobe = BOBE(
       likelihood=likelihood,
       use_clf=True,
       clf_type='svm'
   )
   
   # Run Bayesian Optimization with nested sampling
   results = bobe.run(acqs='wipv', max_evals=500)
   
   print(f"Log Evidence: {results['logz']:.2f} ± {results['logz_err']:.2f}")

A more complete example is available in the :doc:`quickstart` tutorial.

How to Use This Guide
---------------------

To start, you'll need to follow the :doc:`installation` guide to get BOBE installed on your computer. 
After that, you can learn most of what you need from the tutorials listed below (start with 
:doc:`quickstart` and go from there). If you need more details about specific functionality, 
the User Guide should have what you need.

We welcome bug reports, patches, feature requests, and other comments via the 
`GitHub issue tracker <https://github.com/Ameek94/BOBE/issues>`_.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   user guide/basic_usage
   user guide/gp
   user guide/acquisition
   user guide/convergence
   user guide/advanced
   user guide/faq

.. toctree::
   :maxdepth: 2
   :caption: Examples

   quickstart
   examples/banana
   examples/cosmology

.. toctree::
   :maxdepth: 2
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
