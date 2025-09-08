# JaxBO

JaxBO is a package for performing Bayesian model comparison for expensive likelihood functions, developed for applications to cosmology. It computes the Bayesian Evidence using Bayesian Optimization to train Gaussian process surrogate for the expensive likelihood function and runs nested sampling/MCMC on the surrogate instead of the underlying expensive likelihood. Acquisition functions such as EI, LogEI and integrated posterior variance as currently supported. 
To install run

```bash
python -m pip install .
```

from the package directory. For an editable (dev) install do

```bash
python -m pip install -e .
```

If you face installations issues related to incompatible versions of some dependencies, you can also set up an environment for JaxBo with the exact package versions it was developed and tested with.

```bash
# Create environment from the minimal essential packages
conda env create -f environment.yml

# Activate the environment
conda activate jaxbo

# Install JaxBo in development mode
pip install -e .
```

Documentation is currently a work in progress, however the examples folder contains several example on how to run the code with different likelihoods, including cosmological likelihoods interfaced through the Cobaya package or your own custom likelihoods. The examples can simply be run as 

``
python your_chosen_example.py
``

The code can also be run in MPI mode using

``
mpirun -n X python your_bo_script.py
``

where X is the number of MPI process. In mpi mode, the code will distribute the computation of the likelihood functions at several candidiate points across the different MPI processes. This can be very useful when dealing with likelihoods with typical evaluation times of a few seconds or more. 

## References

[1] Eriksson, D. and Jankowiak, M., “High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces” (2021), [arXiv:2103.00349](https://arxiv.org/abs/1710.03206),  doi: 10.1080/00401706.2018.1469433 
(see also [SAASBO](https://github.com/martinjankowiak/saasbo/) on GitHub)

[2] Binois, M., Huang, J., Gramacy, R. B., and Ludkovski, M., "Replication or exploration? Sequential design for stochastic simulation experiments
" (2017), [arXiv:1710.03206](https://arxiv.org/abs/1710.03206), doi: 10.1080/00401706.2018.1469433

[3] Hvarfner, C. and Hellsten, E. and Nardi, L., "Vanilla Bayesian Optimization Performs Great in High Dimensions" (2024), [arxiv:2402.02229](https://arxiv.org/abs/2402.02229), 
