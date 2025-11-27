# JaxBO

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/jaxbo/badge/?version=latest)](https://jaxbo.readthedocs.io/en/latest/?badge=latest)

JaxBO is a high-performance package for Bayesian model comparison using expensive likelihood functions, developed for applications to cosmology. It computes the Bayesian Evidence using Bayesian Optimization by training a Gaussian process surrogate for the expensive likelihood function and runs nested sampling/MCMC on the surrogate instead of the underlying likelihood.

## Key Features

- **Efficient Bayesian Evidence Estimation**: Train GP surrogates for expensive likelihoods and compute evidence and parameter posteriors
- **Multiple Acquisition Functions**: Support for Expected Improvement (EI), LogEI, Weighted Integrated Posterior Variance (WIPV), and more
- **JAX-Powered Performance**: Leverages JAX for automatic differentiation, jit compilation (and in the future GPU/TPU acceleration...)
- **MPI Parallelization**: Distribute likelihood evaluations and GP fitting across multiple processes
- **Classifier-Enhanced GPs**: Optional classifier to filter low-likelihood regions for improved efficiency
- **Flexible Likelihood Interface**: Built-in support for cosmological likelihoods through [Cobaya](https://cobaya.readthedocs.io/en/latest/index.html) or your own custom functions
- **Comprehensive Results Management**: Built-in tracking and visualization of optimization progress

## Quick Start

```python
from jaxbo import BOBE, Likelihood

# Define your likelihood function
def my_likelihood(params):
    # Your expensive computation here
    return log_likelihood

# Set up the likelihood with parameter bounds
likelihood = Likelihood(
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

### From Source

```bash
git clone https://github.com/Ameek94/JaxBo.git
python -m pip install .
```

from the package directory. For an editable (dev) install do

```bash
python -m pip install -e .
```

### Using Conda (Recommended for Development)

You can also set up an environment with the exact package versions JaxB was developed and tested with:

```bash
# Create environment from the minimal essential packages
conda env create -f environment.yml

# Activate the environment
conda activate jaxbo

# Install JaxBo in development mode
pip install -e .
```

### Optional Dependencies

JaxBo has several optional dependencies for extended functionality. These can be installed separately or with JaxBo as follows:

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

- Python 3.11 or higher
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

Documentation is available at [https://jaxbo.readthedocs.io](https://jaxbo.readthedocs.io). The `examples/` folder also contains several examples on how to run the code with different likelihoods, including cosmological likelihoods interfaced through the Cobaya package or your own custom likelihoods.

```bash
python your_chosen_example.py
```

### MPI Parallelization

For expensive likelihoods (evaluation time > 1 second), you can use MPI to parallelize likelihood evaluations across multiple processes:

```bash
mpirun -n 4 python your_bo_script.py
```

where `-n 4` specifies the number of MPI processes. In MPI mode, the code distributes the computation of the likelihood function at several candidate points across different MPI processes, significantly reducing wall-clock time for expensive likelihoods. It also distributes GP fitting by running multiple restarts across the different MPI processes.

<!-- **Example with MPI:**

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

The `run_bobe` function automatically detects and uses MPI when available. -->

## Citation

If you use JaxBO in your research, please cite:

```bibtex
@article{jaxbo2025,
  author = {Cohen, Nathan and Malhotra, Ameek and Hamann, Jan},
  title = {Bayesian Optimisation for Bayesian Evidence},
  year = {2025},
  url = {https://github.com/Ameek94/JaxBo}
}
```

## Support

- **Documentation**: [https://jaxbo.readthedocs.io](https://jaxbo.readthedocs.io)
- **Issues**: [https://github.com/Ameek94/JaxBoissues](https://github.com/Ameek94/JaxBo/issues)
- **Repository**: [https://github.com/Ameek94/JaxBo](https://github.com/Ameek94/JaxBo)



## License

JaxBO is released under the [MIT License](LICENSE).
