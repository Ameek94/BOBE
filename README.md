# JaxBO

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/jaxbo/badge/?version=latest)](https://jaxbo.readthedocs.io/en/latest/?badge=latest)

JaxBO is a high-performance package for Bayesian model comparison using expensive likelihood functions, developed for applications to cosmology. It computes the Bayesian Evidence using Bayesian Optimization to train Gaussian process surrogates for expensive likelihood functions and runs nested sampling/MCMC on the surrogate instead of the underlying likelihood.

## Key Features

- **Efficient Bayesian Evidence Estimation**: Train GP surrogates for expensive likelihoods and compute evidence via nested sampling
- **Multiple Acquisition Functions**: Support for Expected Improvement (EI), LogEI, Weighted Integrated Posterior Variance (WIPV), and more
- **JAX-Powered Performance**: Leverages JAX for automatic differentiation and GPU/TPU acceleration
- **MPI Parallelization**: Distribute likelihood evaluations across multiple processes for expensive computations
- **Classifier-Enhanced GPs**: Optional classifier to filter low-likelihood regions for improved efficiency
- **Flexible Likelihood Interface**: Built-in support for Cobaya cosmological likelihoods or custom functions
- **SAAS Priors**: Sparse Axis-Aligned Subspace priors for high-dimensional problems
- **Comprehensive Results Management**: Built-in tracking and visualization of optimization progress

## Quick Start

```python
from jaxbo import BOBE, ExternalLikelihood

# Define your likelihood function
def my_likelihood(params):
    # Your expensive computation here
    return log_likelihood

# Set up the likelihood with parameter bounds
likelihood = ExternalLikelihood(
    loglikelihood=my_likelihood,
    bounds=[(0, 1), (-5, 5), (0, 10)],  # bounds for 3 parameters
    param_names=['x', 'y', 'z']
)

# Run Bayesian Optimization for Bayesian Evidence
bobe = BOBE(
    loglikelihood=likelihood,
    min_evals=100,
    max_evals=500,
    save_dir='./results'
)
results = bobe.run()

# Access the evidence and posterior samples
print(f"Log Evidence: {results.logZ}")
```

## Installation

To install run

```bash
python -m pip install .
```

## Installation

### From Source

```bash
python -m pip install .
```

from the package directory. For an editable (dev) install do

```bash
python -m pip install -e .
```

### Using Conda (Recommended for Development)

If you face installation issues related to incompatible versions of some dependencies, you can set up an environment with the exact package versions JaxBO was developed and tested with:

```bash
# Create environment from the minimal essential packages
conda env create -f environment.yml

# Activate the environment
conda activate jaxbo

# Install JaxBO in development mode
pip install -e .
```

### Optional Dependencies

JaxBO has several optional dependencies for extended functionality:

- **Neural Network Classifiers** (Flax + Optax): Install with `pip install 'jaxbo[nn]'`
  - Enables MLPClassifier and EllipsoidClassifier for GP filtering
  - Required for using `clf_type='nn'` or `clf_type='ellipsoid'`
  
- **Cobaya Likelihoods**: Install with `pip install 'jaxbo[cobaya]'`
  - Enables `CobayaLikelihood` class for cosmological likelihoods
  - Required for interfacing with Cobaya cosmological models
  
- **MPI Parallelization**: Install with `pip install 'jaxbo[mpi]'`
  - Enables parallel likelihood evaluation across multiple processes
  - Recommended for expensive likelihoods (evaluation time > 1 second)

- **All Optional Dependencies**: Install with `pip install 'jaxbo[all]'`

**Note:** By default, only SVM classifiers are available. The core BO functionality works with scipy optimization and does not require optax.

## Requirements

- Python 3.11 or 3.12
- JAX (with GPU support optional)
- NumPyro for probabilistic programming
- scikit-learn for classifiers
- scipy for optimization

### Optional:
- optax (for neural network classifiers)
- flax (for neural network classifiers)
- cobaya (for cosmological likelihoods)
- mpi4py (for MPI parallelization)

See `pyproject.toml` for full dependencies.

## Usage

### Basic Example

Documentation is currently a work in progress, however the `examples/` folder contains several examples on how to run the code with different likelihoods, including cosmological likelihoods interfaced through the Cobaya package or your own custom likelihoods. The examples can simply be run as 

```bash
python your_chosen_example.py
```

### MPI Parallelization

For expensive likelihoods (evaluation time > 1 second), you can use MPI to parallelize likelihood evaluations across multiple processes:

```bash
mpirun -n 4 python your_bo_script.py
```

where `-n 4` specifies the number of MPI processes. In MPI mode, the code distributes the computation of the likelihood function at several candidate points across different MPI processes, significantly reducing wall-clock time for expensive likelihoods.

**Example with MPI:**

```python
from jaxbo import run_bobe, CobayaLikelihood

# Use the high-level run_bobe function for automatic MPI handling
results = run_bobe(
    likelihood="planck_2018_lowl.TT",  # Cobaya likelihood
    min_evals=200,
    max_evals=1000,
    save_dir='./results_mpi'
)
```

The `run_bobe` function automatically detects and uses MPI when available.

## Citation

If you use JaxBO in your research, please cite:

```bibtex
@article{jaxbo2024,
  author = {Cohen, Nathan and Malhotra, Ameek and Hamann, Jan},
  title = {Bayesian Optimisation for Bayesian Evidence},
  year = {2025},
  url = {https://github.com/CosmologicalEmulators/JaxBo}
}
```

## Support

- **Documentation**: [https://jaxbo.readthedocs.io](https://jaxbo.readthedocs.io)
- **Issues**: [https://github.com/CosmologicalEmulators/JaxBo/issues](https://github.com/CosmologicalEmulators/JaxBo/issues)
- **Repository**: [https://github.com/CosmologicalEmulators/JaxBo](https://github.com/CosmologicalEmulators/JaxBo)

## References

[1] Eriksson, D. and Jankowiak, M., "High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces" (2021), [arXiv:2103.00349](https://arxiv.org/abs/2103.00349)  
(see also [SAASBO on GitHub](https://github.com/martinjankowiak/saasbo/))

[2] Binois, M., Huang, J., Gramacy, R. B., and Ludkovski, M., "Replication or exploration? Sequential design for stochastic simulation experiments" (2017), [arXiv:1710.03206](https://arxiv.org/abs/1710.03206), doi: [10.1080/00401706.2018.1469433](https://doi.org/10.1080/00401706.2018.1469433)

[3] Hvarfner, C., Hellsten, E., and Nardi, L., "Vanilla Bayesian Optimization Performs Great in High Dimensions" (2024), [arXiv:2402.02229](https://arxiv.org/abs/2402.02229)

## License

JaxBO is released under the [MIT License](LICENSE).
