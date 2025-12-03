# BOBE

(Working on mpi atm so things might break for a while.)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/BOBE/badge/?version=latest)](https://BOBE.readthedocs.io/en/latest/?badge=latest)

BOBE is a high-performance package for Bayesian model comparison using expensive likelihood functions, developed for applications to cosmology. It computes the Bayesian Evidence using Bayesian Optimization by training a Gaussian process surrogate for the expensive likelihood function and runs nested sampling/MCMC on the surrogate instead of the underlying likelihood.

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
import numpy as np
from BOBE import BOBE

# Define your likelihood function
def my_likelihood(params):
    # Your expensive computation here
    return -np.sum(params**2)  # Example: simple quadratic

# Initialize BOBE with setup parameters
bobe = BOBE(
    loglikelihood=my_likelihood,
    param_list=['x', 'y', 'z'],
    param_bounds=np.array([[0, 1], [-5, 5], [0, 10]]).T,
    save_dir='./results',
)

# Run optimization with convergence and run settings
results = bobe.run(
    acqs='wipv',
    min_evals=100,
    max_evals=500,
)

# Access the evidence and posterior samples
print(f"Log Evidence: {results['logz']['mean']}")
print(f"Samples shape: {results['samples']['x'].shape}")
```

## Installation

### From Source

```bash
git clone https://github.com/Ameek94/BOBE.git
python -m pip install .
```

from the package directory. For an editable (dev) install do

```bash
python -m pip install -e .
```

### Using Conda (Recommended for Development)

You can also set up an environment with the exact package versions BOBE was developed and tested with:

```bash
# Create environment from the minimal essential packages
conda env create -f environment.yml

# Activate the environment
conda activate BOBE

# Install BOBE in development mode
pip install -e .
```

### Optional Dependencies

BOBE has several optional dependencies for extended functionality. When installing from source, you can install them as follows:

- **Neural Network Classifiers** (Flax + Optax): Install with `pip install -e '.[nn]'`
  - Enables MLPClassifier and EllipsoidClassifier for GP filtering
  - Required for using `clf_type='nn'` or `clf_type='ellipsoid'`
  
- **Cobaya Likelihoods**: Install with `pip install -e '.[cobaya]'`
  - Enables `CobayaLikelihood` class for cosmological likelihoods
  - Required for interfacing with Cobaya cosmological models
  
- **MPI Parallelization**: Install with `pip install -e '.[mpi]'`
  - Enables parallel likelihood evaluation across multiple processes
  - Recommended for expensive likelihoods (evaluation time > 1 second)

- **All Optional Dependencies**: Install with `pip install -e '.[all]'`

**Note:** The `-e` flag installs in editable mode. For a regular install, use `pip install '.[extra]'` instead. By default, only SVM classifiers are available. The core BO functionality works with scipy optimization and does not require optax.

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

Documentation is available at [https://BOBE.readthedocs.io](https://BOBE.readthedocs.io). The `examples/` folder also contains several examples on how to run the code with different likelihoods, including cosmological likelihoods interfaced through the Cobaya package or your own custom likelihoods.

```bash
python your_chosen_example.py
```

### MPI Parallelization

For expensive likelihoods (evaluation time > 1 second), you can use MPI to parallelize likelihood evaluations across multiple processes:

```bash
mpirun -n 4 python your_bo_script.py
```

where `-n 4` specifies the number of MPI processes. In MPI mode, the code distributes the computation of the likelihood function at several candidate points across different MPI processes, significantly reducing wall-clock time for expensive likelihoods. It also distributes GP fitting by running multiple restarts across the different MPI processes.

### Cosmology Example with Cobaya

For cosmological likelihoods, simply pass the Cobaya YAML file path:

```python
from BOBE import BOBE

# Initialize BOBE with Cobaya YAML file - CobayaLikelihood created internally
bobe = BOBE(
    loglikelihood='path/to/cobaya_input.yaml',
    likelihood_name='planck_lcdm',
    confidence_for_unbounded=0.9999995,
    save_dir='./results'
)

# Run with optimization settings
results = bobe.run(
    acqs='wipv',
    min_evals=200,
    max_evals=1000,
)
```

<!-- **Example with MPI:**

```python
from BOBE import run_bobe, CobayaLikelihood

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

If you use BOBE in your research, please cite:

```bibtex
@article{BOBE2025,
  author = {Cohen, Nathan and Malhotra, Ameek and Hamann, Jan},
  title = {Bayesian Optimisation for Bayesian Evidence},
  year = {2025},
  url = {https://github.com/Ameek94/BOBE}
}
```

## Support

- **Documentation**: [https://BOBE.readthedocs.io](https://BOBE.readthedocs.io)
- **Issues**: [https://github.com/Ameek94/BOBE/issues](https://github.com/Ameek94/BOBE/issues)
- **Repository**: [https://github.com/Ameek94/BOBE](https://github.com/Ameek94/BOBE)



## License

BOBE is released under the [MIT License](LICENSE).
