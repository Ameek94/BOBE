# BOBE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/BOBE/badge/?version=latest)](https://BOBE.readthedocs.io/en/latest/?badge=latest)

BOBE is a high-performance package for doing Bayesian model comparison with expensive likelihood functions, developed for applications to cosmology. It computes the Bayesian Evidence using Bayesian Optimization by training a Gaussian process surrogate for the expensive likelihood function and runs Nested sampling/MCMC on the surrogate instead of the underlying likelihood. Training the surrogate requires around ~100x fewer true likelihood evaluations compared to running Nested sampling/MCMC on the true likelihood, leading to significant speed-ups for slow likelihoods (t>1s). BOBE uses acquisition functions that minimise the integrated uncertainty of the surrogate, prioritising regions that matter the most for the evidence. The algorithm is explained in more detail in arxiv:2512.xxxxx. Code [documentation](https://BOBE.readthedocs.io/en/) is also available.

## Key Features

- **Efficient Bayesian Evidence Estimation**: Train GP surrogates for expensive likelihoods and compute evidence and parameter posteriors
- **JAX-Powered Performance**: Leverages JAX for automatic differentiation, jit compilation (and in the future GPU/TPU acceleration...)
- **MPI Parallelization**: Distribute likelihood evaluations and GP fitting across multiple processes
- **Flexible Likelihood Interface**: Built-in support for cosmological likelihoods through [Cobaya](https://cobaya.readthedocs.io/en/latest/index.html) or your own custom functions

## Requirements

- Python >=3.10 and <3.14
- JAX
- NumPyro
- scipy 
- scikit-learn 

### Optional:
- cobaya (for cosmological likelihoods)
- mpi4py (for MPI parallelization)

See `pyproject.toml` and the [documentation](https://BOBE.readthedocs.io/en/) for full list of dependencies.


## Installation

### From source using pip

```bash
git clone https://github.com/Ameek94/BOBE.git
cd BOBE
python -m pip install .
```

For an editable (dev) install do 

```bash
python -m pip install -e .
```

from the package directory. 

### Optional Dependencies

BOBE has several optional dependencies for extended functionality. When installing from source, you can install them as follows:
  
- **Cobaya Likelihoods**: Install with `pip install -e '.[cobaya]'`
  - Enables `CobayaLikelihood` class for cosmological likelihoods interfaced through Cobaya
  - Required for interfacing with Cobaya cosmological models
  
- **MPI Parallelization**: Install with `pip install -e '.[mpi]'`
  - Enables parallel likelihood evaluation across multiple processes using mpi4py
  - Recommended for expensive likelihoods (evaluation time > 1 second)

- **All Optional Dependencies**: Install with `pip install -e '.[all]'`

**Note:** The `-e` flag installs in editable mode. For a regular install, use `pip install '.[extra]'` instead (replace extra with name of dependency).

## Quick Start

### Simple python function

```python
from BOBE import BOBE

# Define your likelihood function
def my_likelihood(X):
    x, y = X[0], X[1]
    logpdf = -0.25 * (5 * (0.2 - x))**2 - (20 * (y/4 - x**4))**2 # Example: a likelihood with a curved degeneracy
    return logpdf 


# Initialize BOBE with setup parameters
sampler = BOBE(
    loglikelihood=my_likelihood,
    param_list=['x1', 'x2'], # list of parameter names
    param_bounds=np.array([[-1, 1], [-1, 2]]).T, # lower and upper bounds for parameters (2, ndim) shaped
    n_sobol_init=2, # number of initial Sobol samples to start the run from
    save_dir='./results',
)

# Run optimization with convergence and run settings
results = sampler.run(
    min_evals=10, # do at least 20 evaluations
    max_evals=100, # max evaluation budget
    batch_size=2, # acquisition function batch size
    fit_n_points=4, # fit gp after every 4 likelihood evaluations
    ns_n_points=4, # run nested sampling after every 4 likelihood evaluations
    logz_threshold=0.1, # target logz uncertainty from GP
)

# Access the evidence and posterior samples
print(f"Log Evidence: {results['logz']['mean']}")
samples = results['samples'] # dictionary containing keys 'x', 'logl', 'weights'
```

### Cosmology Example with Cobaya

For cosmological likelihoods you will need to have [Cobaya](https://cobaya.readthedocs.io/en/latest/index.html) installed. Then, simply pass the Cobaya YAML file path in your script to the BOBE sampler.

```python
from BOBE import BOBE

# Initialize BOBE with Cobaya YAML file - CobayaLikelihood created internally
   # Initialize BOBE with Cobaya YAML file - CobayaLikelihood created internally
   sampler = BOBE(
       loglikelihood='path/to/cobaya_input.yaml',
       likelihood_name='CobayaLikelihood',
       n_sobol_init=4,
       n_cobaya_init=4,  # We can also specify reference dists in the Cobaya yaml file to generate additional initial points
       likelihood_name='quickstart_cobaya_example', # name for output files
       save_dir='./results',
       use_clf=True # recommended to enable classifiers for cosmological examples where likelihood can sometimes return -inf values
   )
   
   # Run with optimization settings
   results = sampler.run(
       min_evals=10,
       max_evals=1000, # adjust according to your evaluation budget
       batch_size=5,
       fit_n_points=10,
       ns_n_points=10,
       logz_threshold=0.5,
   )

# rest of the run remains the same as above
```

Full documentation is available at [https://BOBE.readthedocs.io](https://BOBE.readthedocs.io). The `examples/` folder also contains several examples on how to run the code with different likelihoods, including cosmological likelihoods interfaced through the Cobaya package or your own custom likelihoods.

```bash
python your_chosen_example.py
```

### MPI Parallelization

For expensive likelihoods (evaluation time > 1 second), you can use MPI to parallelize likelihood evaluations at points proposed by the batch acquisition function. Make sure you have mpi4py installed.

```bash
mpirun -n 4 python your_bo_script.py
```

where `-n 4` specifies the number of MPI processes. In MPI mode, the code distributes the computation of the likelihood function at several candidate points across different MPI processes, significantly reducing wall-clock time for expensive likelihoods. It also distributes GP fitting by running multiple restarts across the different MPI processes. Note than on SLURM clusters you might need to substitute mpirun with srun.

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

## Troubleshooting

### Installation issues

In case you run into installation issues related to incompatible versions of packages, you can also set up an environment with the exact package versions BOBE was developed and tested with

```bash
# Create environment from the minimal essential packages
conda env create -f environment.yml

# Activate the environment
conda activate BOBE

# Install BOBE
pip install .
```


## Support

- **Documentation**: [https://BOBE.readthedocs.io](https://BOBE.readthedocs.io)
- **Issues**: [https://github.com/Ameek94/BOBE/issues](https://github.com/Ameek94/BOBE/issues)
- **Repository**: [https://github.com/Ameek94/BOBE](https://github.com/Ameek94/BOBE)



## License

BOBE is released under the [MIT License](LICENSE).
