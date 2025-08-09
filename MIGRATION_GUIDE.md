# Migration Guide: From Hardcoded Parameters to Settings System

This guide helps you transition from using hardcoded parameters in JaxBo to the new centralized settings system.

## Overview

The settings system replaces scattered parameter definitions with:
- Centralized configuration in `jaxbo.settings`
- Type safety and validation
- Easy serialization to configuration files
- Preset configurations for common use cases
- Better documentation and discoverability

## Migration Examples

### BOBE Constructor

**Before:**
```python
from jaxbo.bo import BOBE

bobe = BOBE(
    loglikelihood=likelihood,
    maxiters=1500,
    miniters=200,
    use_clf=True,
    clf_type='svm',
    lengthscale_priors='DSLP',
    mc_points_method='NUTS',
    num_hmc_samples=512,
    logz_threshold=1.0
)
```

**After:**
```python
from jaxbo.bo import BOBE
from jaxbo.settings import BOBESettings

# Option 1: Direct parameter passing (easiest migration)
settings = BOBESettings(
    maxiters=1500,
    miniters=200,
    use_clf=True,
    clf_type='svm',
    lengthscale_priors='DSLP',
    mc_points_method='NUTS',
    num_hmc_samples=512,
    logz_threshold=1.0
)
bobe = BOBE(loglikelihood=likelihood, **settings.to_dict())

# Option 2: Use presets and customize
from jaxbo.settings import get_default_settings

settings = get_default_settings()
settings.bobe.maxiters = 1500
settings.bobe.logz_threshold = 1.0
bobe = BOBE(loglikelihood=likelihood, **settings.bobe.to_dict())
```

### GP Configuration

**Before:**
```python
from jaxbo.gp import DSLP_GP

gp = DSLP_GP(
    train_x=x,
    train_y=y,
    noise=1e-8,
    kernel='rbf',
    outputscale_bounds=[-4, 4],
    lengthscale_bounds=[np.log10(0.05), 2]
)
```

**After:**
```python
from jaxbo.gp import DSLP_GP
from jaxbo.settings import GPSettings

gp_settings = GPSettings(
    noise=1e-8,
    kernel='rbf',
    outputscale_bounds=[-4, 4],
    lengthscale_bounds=[np.log10(0.05), 2]
)

# Extract only the GP-relevant parameters
gp_params = {k: v for k, v in gp_settings.to_dict().items() 
             if k in ['noise', 'kernel', 'outputscale_bounds', 'lengthscale_bounds']}
gp = DSLP_GP(train_x=x, train_y=y, **gp_params)
```

### Classifier Configuration

**Before:**
```python
from jaxbo.clf_gp import ClassifierGP

clf_gp = ClassifierGP(
    train_x=x,
    train_y=y,
    clf_type='svm',
    clf_use_size=400,
    clf_update_step=5,
    probability_threshold=0.5,
    clf_threshold=250
)
```

**After:**
```python
from jaxbo.clf_gp import ClassifierGP
from jaxbo.settings import ClassifierSettings

clf_settings = ClassifierSettings(
    clf_use_size=400,
    clf_update_step=5,
    probability_threshold=0.5,
    clf_threshold=250
)

clf_gp = ClassifierGP(
    train_x=x,
    train_y=y,
    clf_type='svm',
    **clf_settings.to_dict()
)
```

### Nested Sampling

**Before:**
```python
from jaxbo.nested_sampler import nested_sampling_Dy

samples, logz, success = nested_sampling_Dy(
    gp=gp,
    ndim=ndim,
    dlogz=0.1,
    dynamic=True,
    maxcall=int(5e6)
)
```

**After:**
```python
from jaxbo.nested_sampler import nested_sampling_Dy
from jaxbo.settings import NestedSamplingSettings

ns_settings = NestedSamplingSettings(
    dlogz=0.1,
    dynamic=True,
    maxcall=int(5e6)
)

# Extract relevant parameters for the function
ns_params = {k: v for k, v in ns_settings.to_dict().items() 
             if k in ['dlogz', 'dynamic', 'maxcall']}

samples, logz, success = nested_sampling_Dy(
    gp=gp,
    ndim=ndim,
    **ns_params
)
```

## Step-by-Step Migration Process

### Step 1: Identify Current Parameters

First, identify all the parameters you're currently using:

```python
# Collect all your current parameter values
current_params = {
    'maxiters': 1500,
    'use_clf': True,
    'clf_type': 'svm',
    'lengthscale_priors': 'DSLP',
    'kernel': 'rbf',
    'noise': 1e-8,
    # ... etc
}
```

### Step 2: Create Settings Objects

Create the appropriate settings objects:

```python
from jaxbo.settings import BOBESettings, GPSettings, ClassifierSettings

# Map your parameters to the right settings classes
bobe_settings = BOBESettings(
    maxiters=current_params['maxiters'],
    use_clf=current_params['use_clf'],
    clf_type=current_params['clf_type'],
    lengthscale_priors=current_params['lengthscale_priors']
)

gp_settings = GPSettings(
    kernel=current_params['kernel'],
    noise=current_params['noise']
)
```

### Step 3: Update Your Code

Replace your direct parameter passing:

```python
# Before
bobe = BOBE(loglikelihood=likelihood, maxiters=1500, use_clf=True, ...)

# After
bobe = BOBE(loglikelihood=likelihood, **bobe_settings.to_dict())
```

### Step 4: Save Configuration

Save your configuration for reproducibility:

```python
import json

# Save your settings
config = {
    'bobe': bobe_settings.to_dict(),
    'gp': gp_settings.to_dict(),
    'metadata': {
        'description': 'Migration from hardcoded parameters',
        'date': '2025-01-08'
    }
}

with open('my_config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

## Common Parameter Mappings

| Old Parameter Location | New Settings Class | Notes |
|----------------------|-------------------|-------|
| `BOBE.__init__()` | `BOBESettings` | All BOBE constructor parameters |
| `DSLP_GP.__init__()` | `GPSettings` | GP configuration |
| `ClassifierGP.__init__()` | `ClassifierSettings` | Classifier configuration |
| `nested_sampling_Dy()` | `NestedSamplingSettings` | Function parameters |
| `optimize()` | `OptimizationSettings` | Optimization parameters |
| HMC/NUTS parameters | `MCMCSettings` | MCMC configuration |

## Benefits After Migration

1. **Centralized Configuration**: All settings in one place
2. **Type Safety**: Automatic validation of parameter types
3. **Documentation**: Built-in documentation for every parameter
4. **Presets**: Use `get_fast_settings()`, `get_accurate_settings()`, etc.
5. **Reproducibility**: Easy to save/load configurations
6. **Environment Control**: Switch configs based on environment variables

## Backward Compatibility

The old parameter-passing style still works! You can migrate gradually:

```python
# This still works
bobe = BOBE(loglikelihood=likelihood, maxiters=1500, use_clf=True)

# But this is recommended
settings = BOBESettings(maxiters=1500, use_clf=True)
bobe = BOBE(loglikelihood=likelihood, **settings.to_dict())
```

## Example: Complete Migration

**Before (hardcoded):**
```python
#!/usr/bin/env python3
import numpy as np
from jaxbo.bo import BOBE
from jaxbo.loglike import CobayaLikelihood

# Hardcoded parameters scattered throughout
likelihood = CobayaLikelihood('my_config.yaml')

bobe = BOBE(
    loglikelihood=likelihood,
    maxiters=2000,
    miniters=300,
    use_clf=True,
    clf_type='svm',
    clf_threshold=400,
    lengthscale_priors='SAAS',
    mc_points_method='NUTS',
    num_hmc_samples=1024,
    logz_threshold=0.5
)

results = bobe.run()
```

**After (using settings):**
```python
#!/usr/bin/env python3
import numpy as np
from jaxbo.bo import BOBE
from jaxbo.loglike import CobayaLikelihood
from jaxbo.settings import BOBESettings

# Centralized configuration
likelihood = CobayaLikelihood('my_config.yaml')

settings = BOBESettings(
    maxiters=2000,
    miniters=300,
    use_clf=True,
    clf_type='svm',
    clf_threshold=400,
    lengthscale_priors='SAAS',
    mc_points_method='NUTS',
    num_hmc_samples=1024,
    logz_threshold=0.5
)

# Save configuration for reproducibility
import json
with open('run_config.json', 'w') as f:
    json.dump(settings.to_dict(), f, indent=2)

bobe = BOBE(loglikelihood=likelihood, **settings.to_dict())
results = bobe.run()
```

## Tips for Migration

1. **Start Small**: Migrate one module at a time
2. **Use Presets**: Start with `get_default_settings()` and customize
3. **Save Configs**: Always save your working configurations
4. **Check Documentation**: Use `help(BOBESettings)` to see all options
5. **Test Equivalence**: Ensure your migrated code produces the same results

## Getting Help

- Check the docstrings: `help(BOBESettings)`
- View examples: `examples/settings_example.py`
- Read the full documentation: `jaxbo/SETTINGS_README.md`
- Use presets as starting points: `get_default_settings()`, `get_fast_settings()`
