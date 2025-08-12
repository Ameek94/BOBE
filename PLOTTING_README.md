# BOBE Summary Plotting Module

The `jaxbo.summary_plots` module provides comprehensive visualization capabilities for analyzing BOBE runs, including evidence evolution, GP hyperparameters, timing information, and convergence diagnostics.

## Quick Start

```python
from jaxbo.summary_plots import BOBESummaryPlotter, create_summary_plots

# Load results and create plotter
plotter = BOBESummaryPlotter('your_results_file')

# Create comprehensive dashboard
plotter.create_summary_dashboard(
    gp_data=gp_data,
    best_loglike_data=best_loglike_data,
    timing_data=timing_data,
    param_evolution_data=param_evolution_data,
    save_path='summary_dashboard.png'
)

# Or use convenience function to create all plots
create_summary_plots(
    results_file='your_results_file',
    gp_data=gp_data,
    best_loglike_data=best_loglike_data,
    timing_data=timing_data,
    param_evolution_data=param_evolution_data,
    output_dir='plots'
)
```

## Available Plots

### 1. Evidence Evolution (**Updated**)
- **Mean logZ** evolution line (blue solid)
- **Upper bound** line (red dashed) 
- **Lower bound** line (green dashed)
- **Shaded uncertainty region** between bounds
- Marks convergence points
- Uses real data from BOBE convergence history

### 2. GP Hyperparameters (**Updated - Now Separate Plots**)
#### GP Lengthscales Plot
- Evolution of lengthscales for each parameter dimension
- Color-coded by parameter with legend
- Logarithmic y-axis for better visualization

#### GP Output Scale Plot
- Output scale evolution over time as separate plot
- Clean, focused visualization
- Logarithmic y-axis

#### Backward Compatibility
- `plot_gp_hyperparameters()` method still available (calls lengthscales)
- Requires GP hyperparameter tracking data

### 3. Best Log-likelihood Evolution
- Best log-likelihood found so far at each iteration
- Marks improvements
- Useful for monitoring optimization progress

### 4. Convergence Diagnostics
- Delta log Z vs threshold over time
- Convergence flags
- Uses real data from BOBE convergence checks

### 5. Timing Breakdown
- Bar plot of time spent in different phases
- Helps identify bottlenecks
- Can show total runtime or detailed breakdown

### 6. Summary Dashboard (**Updated Layout**)
- **New 2×3 layout** for cleaner presentation:
  - **Top row**: Evidence evolution | GP lengthscales | GP output scale  
  - **Bottom row**: Best log-likelihood | Convergence diagnostics | Timing breakdown
- **Parameter evolution removed** from dashboard (still available as individual plot)
- Larger, more readable plots in streamlined layout
- Summary statistics and comprehensive overview
- Final parameter estimates with uncertainties

## Data Formats

### GP Data
```python
gp_data = {
    'iterations': [10, 20, 30, 40, 50],
    'lengthscales': [[1.0, 0.5], [0.8, 0.6], [0.7, 0.7], [0.6, 0.8], [0.5, 0.9]],
    'outputscales': [2.0, 1.8, 1.6, 1.4, 1.2]
}
```

### Best Log-likelihood Data
```python
best_loglike_data = {
    'iterations': [1, 5, 10, 15, 20],
    'best_loglike': [-10.0, -8.5, -7.2, -6.8, -6.5]
}
```

### Timing Data
```python
timing_data = {
    'GP Training': 45.2,
    'Nested Sampling': 120.8,
    'Optimization': 30.1,
    'I/O Operations': 5.3
}
```

### Parameter Evolution Data
```python
param_evolution_data = {
    'x1': {
        'iterations': [1, 5, 10, 15, 20],
        'values': [0.1, 0.3, 0.5, 0.4, 0.45]
    },
    'x2': {
        'iterations': [1, 5, 10, 15, 20], 
        'values': [-0.2, 0.0, 0.2, 0.15, 0.18]
    }
}
```

## Individual Plot Functions

- `plot_evidence_evolution(ax=None, show_convergence=True)`
- `plot_gp_hyperparameters(gp_data=None, ax=None)`
- `plot_best_loglike_evolution(best_loglike_data=None, ax=None)`
- `plot_timing_breakdown(timing_data=None, ax=None)`
- `plot_convergence_diagnostics(ax=None)`
- `plot_parameter_evolution(param_evolution_data=None, max_params=4)`

## Dependencies

### Required
- `numpy`
- `matplotlib`

### Optional
- `seaborn` (for enhanced styling)
- `getdist` (for triangle plots, not currently used)

## Integration with BOBE

To collect runtime data during a BOBE run, you would typically:

1. **GP Hyperparameters**: Track GP lengthscales and output scales after each GP training step
2. **Best Log-likelihood**: Track the best log-likelihood value found so far after each BO iteration
3. **Timing**: Measure time spent in different phases (GP training, nested sampling, etc.)
4. **Parameter Evolution**: Track the best parameter values found during optimization

This data can then be passed to the plotting functions for visualization and analysis.

## Example Usage

See `test_summary_plots.py` for a complete example using the Banana function test case.
