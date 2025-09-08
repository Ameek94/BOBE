Acquisition Functions
=====================

Acquisition functions determine where to sample next in the parameter space.

Base Acquisition Function
--------------------------

.. autoclass:: jaxbo.acquisition.AcquisitionFunction
   :members:
   :undoc-members:
   :show-inheritance:

Weighted Integrated Posterior Variance (WIPV)
----------------------------------------------

.. autoclass:: jaxbo.acquisition.WIPV
   :members:
   :undoc-members:
   :show-inheritance:

Expected Improvement (EI)
-------------------------

.. autoclass:: jaxbo.acquisition.EI
   :members:
   :undoc-members:
   :show-inheritance:

Log Expected Improvement (LogEI)
--------------------------------

.. autoclass:: jaxbo.acquisition.LogEI
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

Acquisition-related utility functions.

.. automodule:: jaxbo.acquisition
   :members:
   :exclude-members: AcquisitionFunction, WIPV, EI, LogEI
   :undoc-members:
   :show-inheritance:
