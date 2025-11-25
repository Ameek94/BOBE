Gaussian Process Models
=======================

JaxBo provides several Gaussian Process implementations optimized for different scenarios.

Base GP Class
-------------

.. autoclass:: jaxbo.gp.GP
   :members:
   :undoc-members:
   :show-inheritance:

Gaussian Process with Classifier
---------------------------------

For handling constraints and invalid regions.

.. autoclass:: jaxbo.clf_gp.GPwithClassifier
   :members:
   :undoc-members:
   :show-inheritance:

Classifier Module
-----------------

.. automodule:: jaxbo.clf
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

GP-related utility functions.

.. autofunction:: jaxbo.gp.load_gp

.. autofunction:: jaxbo.clf_gp.load_clf_gp
