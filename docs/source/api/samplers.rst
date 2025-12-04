Sampling the likelihood surrogate
==================================

JaxBo provides interfaces for nested sampling via Dynesty and HMC using NUTS from NumPyro. 
These are used for sampling the Gaussian Process surrogate of the likelihood function to calculate the evidence or generate posterior samples.

Nested Sampling
---------------

Nested sampling using Dynesty for evidence computation and posterior sampling.

.. autofunction:: BOBE.samplers.nested_sampling_Dy

HMC/NUTS Sampling
-----------------

Hamiltonian Monte Carlo sampling using NumPyro's NUTS sampler.

.. autofunction:: BOBE.samplers.sample_GP_NUTS

.. autofunction:: BOBE.samplers.get_hmc_settings

Utility Functions
-----------------

.. autofunction:: BOBE.samplers.compute_integrals

.. autofunction:: BOBE.samplers.prior_transform
