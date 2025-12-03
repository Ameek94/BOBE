Glossary
========

.. glossary::

   Acquisition Function
      A function that determines where to sample next in the parameter space during Bayesian optimization. It balances exploration (sampling in uncertain regions) and exploitation (sampling where the objective is expected to be optimal).

   Bayesian Evidence
      The marginal likelihood of a model, computed by integrating the likelihood over the prior distribution of parameters. Also known as the model evidence or partition function.

   Bayesian Optimization
      A global optimization technique for expensive-to-evaluate functions. It uses a probabilistic model (typically a Gaussian Process) to make decisions about where to sample next.

   BOBE
      Bayesian Optimization for Bayesian Evidence - the core algorithm implemented in BOBE for efficiently estimating the Bayesian evidence of cosmological models.

   Cobaya
      A framework for cosmological parameter estimation that handles theory codes, likelihoods, and sampling algorithms in a modular way.

   Deep Sigmoidal Location Process (DSLP)
      A type of Gaussian Process that uses deep neural networks with sigmoidal activations to model the mean function, allowing for more flexible function approximation.

   Expected Improvement (EI)
      A popular acquisition function that computes the expected improvement over the current best observation.

   Gaussian Process (GP)
      A collection of random variables, any finite number of which have a joint Gaussian distribution. Used as a probabilistic model for functions in Bayesian optimization.

   JAX
      A Python library for high-performance machine learning research that provides automatic differentiation and compilation to GPU/TPU.

   Log Evidence
      The natural logarithm of the Bayesian evidence, often easier to work with numerically and the primary quantity estimated by BOBE.

   Marginal Likelihood
      Another term for Bayesian evidence - the probability of the observed data given a model, marginalized over all possible parameter values.

   Nested Sampling
      A Monte Carlo method for computing Bayesian evidence and posterior samples simultaneously, commonly used in cosmology.

   NumPyro
      A probabilistic programming library built on JAX, providing tools for Bayesian modeling and inference.

   Posterior
      The probability distribution of model parameters given the observed data, computed using Bayes' theorem.

   Prior
      The probability distribution expressing beliefs about model parameters before observing data.

   Sparse Axis-Aligned Subspace (SAAS)
      A Gaussian Process variant that assumes the function varies primarily along axis-aligned subspaces, making it more efficient for high-dimensional problems.

   Surrogate Model
      A computationally cheap approximation of an expensive function, used in optimization to guide the search. In BOBE, Gaussian Processes serve as surrogate models.

   Weighted Integrated Posterior Variance (WIPV)
      An acquisition function that weights the integrated posterior variance by the posterior probability, particularly useful for evidence estimation.
