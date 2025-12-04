Acquisition Functions
=====================

Acquisition functions determine where to sample next in the parameter space.

Base Acquisition Function
--------------------------

.. autoclass:: BOBE.acquisition.AcquisitionFunction
   :members:
   :undoc-members:
   :show-inheritance:

Weighted Integrated Posterior Variance (WIPV)
----------------------------------------------

.. autoclass:: BOBE.acquisition.WIPV
   :members:
   :undoc-members:
   :show-inheritance:

Expected Improvement (EI)
-------------------------

.. autoclass:: BOBE.acquisition.EI
   :members:
   :undoc-members:
   :show-inheritance:

Log Expected Improvement (LogEI)
--------------------------------

.. autoclass:: BOBE.acquisition.LogEI
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

Acquisition-related utility functions.

.. automodule:: BOBE.acquisition
   :members:
   :exclude-members: AcquisitionFunction, WIPV, EI, LogEI
   :undoc-members:
   :show-inheritance:
