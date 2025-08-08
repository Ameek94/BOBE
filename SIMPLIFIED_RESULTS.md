# Simplified BOBE Results System

## Overview

The BOBE results system has been simplified to store only essential data while maintaining compatibility with standard analysis tools. This significantly reduces storage requirements and complexity.

## What is Stored

### ✅ Essential Data (Kept)
- **Final samples and weights**: Complete posterior chains for analysis
- **Evidence evolution**: logZ values during convergence checks
- **Convergence information**: Convergence status, thresholds, and diagnostics
- **Basic metadata**: Runtime, parameter names/labels/bounds, settings

### ❌ Detailed Diagnostics (Removed)
- Iteration-by-iteration GP diagnostics
- Acquisition function value history
- GP hyperparameter evolution  
- Best point tracking over iterations
- GP training set size evolution
- Detailed per-iteration metadata

## File Outputs

The simplified system creates these files:

1. **`{output_file}_results.npz`** - Main compressed results
2. **`{output_file}_results.pkl`** - Full Python object
3. **`{output_file}.txt`** - GetDist format chain
4. **`{output_file}_1.txt`** - CosmoMC format chain  
5. **`{output_file}_stats.json`** - Summary statistics
6. **`{output_file}_convergence.npz`** - Convergence diagnostics
7. **`{output_file}_intermediate.json`** - Intermediate results

**No longer created:**
- `{output_file}_gp_evolution.npz` (GP diagnostics)

## Code Changes

### Modified Methods

1. **`update_iteration()`** - Simplified to only save intermediate results periodically
2. **`get_results_dict()`** - Removed detailed iteration tracking and BO diagnostics
3. **`save_diagnostics()`** - Only saves convergence information
4. **Various other methods** - Streamlined to focus on essential data

### Removed Classes
- `IterationInfo` dataclass - No longer needed for simplified tracking

### Updated Integration
- **`bo.py`** - Modified to use simplified `update_iteration()` call

## Usage

The simplified system works exactly the same way from a user perspective:

```python
from jaxbo.results import BOBEResults, load_bobe_results

# Usage remains the same
results_manager = BOBEResults(...)
results_manager.finalize(samples, weights, loglikes, ...)

# Loading and analysis unchanged
loaded_results = load_bobe_results("my_run")
results_dict = loaded_results.get_results_dict()
```

## Benefits

1. **Reduced Storage**: Significantly smaller file sizes
2. **Simplified Code**: Easier to maintain and understand
3. **Faster I/O**: Less data to read/write
4. **Essential Focus**: Only stores data needed for posterior analysis
5. **Maintained Compatibility**: Still works with GetDist, ChainConsumer, etc.

## What's Preserved

- Full posterior samples and weights for complete analysis
- Evidence evolution for understanding convergence behavior
- All parameter information and metadata
- Compatibility with standard cosmology analysis tools
- Crash recovery through intermediate saves
- Multiple output formats for different use cases

## Migration

Existing code using the results system will continue to work. The only change is that detailed iteration-by-iteration diagnostics are no longer available. If you need this information, you can add custom logging in your BOBE run loop.

## Example

See `examples/results_example.py` for a complete demonstration of the simplified results system.
