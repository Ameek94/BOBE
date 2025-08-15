# JaxBo Settings Module

The JaxBo settings module provides a centralized, type-safe configuration system for all JaxBo components. This module allows you to easily configure and customize the behavior of the BOBE sampler, Gaussian Processes, classifiers, nested sampling, and other components.

## Features

- **Type-safe configuration** with dataclasses and automatic validation
- **Centralized settings** for all JaxBo modules in one place
- **Preset configurations** for common use cases (fast, accurate, high-dimensional)
- **Easy serialization** to/from JSON and YAML files
- **Modular customization** - override only the settings you need
- **Comprehensive documentation** for every parameter

## Quick Start

```python
from jaxbo.settings import BOBESettings, get_default_settings
from jaxbo.bo import BOBE

# Use default settings
settings = BOBESettings()
bobe = BOBE(loglikelihood=my_likelihood, **settings.to_dict())

# Or customize specific parameters
custom_settings = BOBESettings(
    max_eval_budget=2000,
    use_clf=True,
    lengthscale_priors='SAAS'
)
bobe = BOBE(loglikelihood=my_likelihood, **custom_settings.to_dict())
```

## Available Settings Classes

### Core Classes

- **`BOBESettings`**: Main sampler configuration (iterations, convergence, etc.)
- **`GPSettings`**: Gaussian Process configuration (kernels, optimization, bounds)
- **`ClassifierSettings`**: Classifier configuration (SVM, NN, ellipsoid parameters)
- **`NestedSamplingSettings`**: Nested sampling configuration (Dynesty, JaxNS)
- **`OptimizationSettings`**: Acquisition optimization configuration
- **`MCMCSettings`**: MCMC/NUTS sampling configuration
- **`LoggingSettings`**: Logging verbosity and output configuration

### Master Class

- **`JaxBoSettings`**: Contains all module settings in one object

## Preset Configurations

```python
from jaxbo.settings import (
    get_default_settings,
    get_fast_settings,
    get_accurate_settings,
    get_high_dimensional_settings
)

# For quick testing
fast_config = get_fast_settings()

# For high-dimensional problems (>10 dimensions)
high_dim_config = get_high_dimensional_settings()

# For maximum accuracy (slower)
accurate_config = get_accurate_settings()
```

## Configuration Files

Save and load settings to configuration files:

```python
# Save settings
settings = get_high_dimensional_settings()
config_dict = settings.to_dict()

import json
with open('my_config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)

# Load settings
with open('my_config.json', 'r') as f:
    config_dict = json.load(f)

settings = JaxBoSettings.from_dict(config_dict)
```

## Module-Specific Settings

### BOBE Sampler Settings

```python
bobe_settings = BOBESettings(
    max_eval_budget=1500,    # Maximum evaluation budget
    miniters=200,            # Minimum iterations before convergence check
    use_clf=True,            # Use classifier for filtering
    clf_type='svm',          # Classifier type
    lengthscale_priors='DSLP', # GP lengthscale priors
    mc_points_method='NUTS', # Monte Carlo sampling method
    logz_threshold=1.0,      # Convergence threshold
)
```

### Gaussian Process Settings

```python
gp_settings = GPSettings(
    kernel='rbf',            # Kernel type ('rbf' or 'matern')
    noise=1e-8,              # Noise level
    fit_maxiter=150,         # Optimization iterations
    fit_n_restarts=4,        # Number of optimization restarts
    outputscale_bounds=[-4, 4], # Output scale bounds (log10)
    lengthscale_bounds=[np.log10(0.05), 2], # Lengthscale bounds (log10)
)
```

### Classifier Settings

```python
clf_settings = ClassifierSettings(
    clf_use_size=400,        # Minimum points before using classifier
    clf_update_step=5,       # Update frequency
    svm_C=1e7,              # SVM regularization
    svm_gamma='scale',       # SVM gamma parameter
    nn_hidden_dims=[64, 32], # NN architecture
    nn_learning_rate=1e-3,   # NN learning rate
)
```

### Nested Sampling Settings

```python
ns_settings = NestedSamplingSettings(
    dlogz=0.1,              # Evidence convergence criterion
    dynamic=True,           # Use dynamic nested sampling
    maxcall=int(5e6),       # Maximum function evaluations
    sample_method='rwalk',  # Sampling method
)
```

## Examples

See the example files for detailed usage:

- `examples/settings_example.py` - Basic usage and customization
- `examples/config_file_example.py` - Configuration file handling

## Integration with Existing Code

The settings system is designed to be backward compatible. You can gradually adopt it:

```python
# Old way (still works)
bobe = BOBE(
    loglikelihood=likelihood,
    max_eval_budget=1500,
    use_clf=True,
    lengthscale_priors='SAAS'
)

# New way (recommended)
settings = BOBESettings(
    max_eval_budget=1500,
    use_clf=True,
    lengthscale_priors='SAAS'
)
bobe = BOBE(loglikelihood=likelihood, **settings.to_dict())
```

## Environment-Based Configuration

Use environment variables to select configurations:

```bash
export JAXBO_CONFIG=fast
python my_analysis.py

export JAXBO_CONFIG=high_dim  
python my_analysis.py
```

```python
import os
from jaxbo.settings import get_fast_settings, get_high_dimensional_settings

config_map = {
    'fast': get_fast_settings,
    'high_dim': get_high_dimensional_settings,
}

config_type = os.environ.get('JAXBO_CONFIG', 'default')
settings = config_map.get(config_type, get_default_settings)()
```

## Parameter Documentation

Every parameter in the settings classes includes comprehensive documentation. Access it via:

```python
help(BOBESettings)
help(GPSettings.kernel)
```

Or view the docstrings in the source code for detailed explanations of each parameter's purpose and recommended values.

## Best Practices

1. **Start with presets**: Use `get_fast_settings()` for testing, `get_high_dimensional_settings()` for complex problems
2. **Use configuration files**: Save successful configurations for reproducibility
3. **Gradual customization**: Start with defaults and modify only what you need
4. **Document your configs**: Include metadata in configuration files
5. **Version control**: Track configuration files with your analysis code

## Common Use Cases

### Quick Testing
```python
settings = get_fast_settings()
# Reduced iterations and sampling for fast feedback
```

### Production Cosmology Analysis
```python
settings = BOBESettings(
    max_eval_budget=2000,
    use_clf=True,
    clf_type='svm',
    lengthscale_priors='SAAS',
    logz_threshold=0.5
)
```

### High-Dimensional Parameter Estimation
```python
settings = get_high_dimensional_settings()
# Optimized for >10 parameters with classifier filtering
```

### Maximum Accuracy Analysis
```python
settings = get_accurate_settings()
# More iterations, restarts, and tighter convergence
```
