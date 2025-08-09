# BOBE Summary Plotting Integration Guide

This guide shows how to integrate the summary plotting system into BOBE for comprehensive runtime visualization.

## Quick Start

1. **Import the modules:**
```python
from jaxbo.summary_plots import BOBESummaryPlotter
from examples.runtime_data_collection_example import BOBEDataCollector
```

2. **Initialize data collection in your BOBE run:**
```python
# In BOBE.__init__()
self.data_collector = BOBEDataCollector(self.param_names)
```

3. **Collect data during the run:**
```python
# Record GP hyperparameters after training
self.data_collector.record_gp_hyperparams(iteration, lengthscales, outputscale)

# Record best log-likelihood evolution
self.data_collector.record_best_loglike(iteration, best_loglike)

# Record timing for different phases
self.data_collector.start_phase("GP Training")
# ... GP training code ...
self.data_collector.end_phase("GP Training")
```

4. **Create plots at the end:**
```python
# Save runtime data
self.data_collector.save_runtime_data(self.output_file)

# Create comprehensive plots
plotter = BOBESummaryPlotter(self.results_manager)
runtime_data = {
    'gp_data': self.data_collector.get_gp_data(),
    'best_loglike_data': self.data_collector.get_best_loglike_data(),
    'timing_data': self.data_collector.get_timing_data(),
    'param_evolution_data': self.data_collector.get_param_evolution_data()
}
plotter.create_summary_dashboard(**runtime_data)
```

## What You Get

The summary plotting system provides:

### 📊 **Evidence Evolution Plot** (**Updated**)
- **Mean logZ** line with upper and lower bounds
- **Shaded uncertainty region** between bounds  
- Convergence assessment and trend analysis
- Customizable styling and confidence intervals

### 🔧 **GP Hyperparameter Evolution** (**Updated - Separate Plots**)
- **GP Lengthscales**: Individual plots for each parameter dimension
- **GP Output Scale**: Dedicated plot for output scale evolution
- Optional logarithmic scaling for better visualization
- Backward compatibility maintained

### ⏱️ **Timing Breakdown**
- Pie chart showing time spent in different phases
- Helps identify computational bottlenecks
- Useful for performance optimization

### 📈 **Best Log-likelihood Evolution**
- Best log-likelihood found so far vs iteration
- Shows optimization progress and convergence
- Can reveal plateaus or continued improvement

### 🔍 **Convergence Diagnostics**
- Effective sample size evolution
- R-hat convergence statistic over time
- Helps assess when to stop the run

### 📋 **Summary Dashboard** (**Updated Layout**)
- **New 2×3 layout** instead of 3×3
- **Top row**: Evidence evolution | GP lengthscales | GP output scale
- **Bottom row**: Best log-likelihood | Convergence diagnostics | Timing breakdown
- **Parameter evolution removed** from dashboard for cleaner presentation
- Perfect for reports and presentations
- Customizable layout and styling

## File Outputs

After running with plotting enabled, you'll get:

```
your_output_prefix_chain.txt          # GetDist-compatible samples
your_output_prefix.ranges             # Parameter ranges
your_output_prefix_runtime_data.json  # Runtime data for plotting

# Individual plots (if created separately)
your_output_prefix_evidence_evolution.png
your_output_prefix_gp_hyperparams.png
your_output_prefix_timing_breakdown.png
your_output_prefix_best_loglike.png
your_output_prefix_param_evolution.png
your_output_prefix_convergence.png

# Summary dashboard
your_output_prefix_summary_dashboard.png
```

## Advanced Usage

### Custom Plot Styling
```python
# Create plotter with custom styling
plotter = BOBESummaryPlotter(results_manager)

# Individual plots with custom parameters
plotter.plot_evidence_evolution(
    figsize=(12, 8),
    confidence_bands=True,
    log_scale=False,
    grid=True
)

# Summary dashboard with custom layout
plotter.create_summary_dashboard(
    figsize=(20, 16),
    layout=(3, 3),
    title="My BOBE Run Summary",
    **runtime_data
)
```

### Real-time Monitoring
```python
# Create intermediate plots during long runs
if iteration % 100 == 0:  # Every 100 iterations
    plotter.plot_evidence_evolution(output_file=f"intermediate_{iteration}")
    plotter.plot_best_loglike_evolution(
        **self.data_collector.get_best_loglike_data(),
        output_file=f"loglike_{iteration}"
    )
```

### Data Analysis
```python
# Load and analyze saved runtime data
import json
with open(f"{output_file}_runtime_data.json", 'r') as f:
    runtime_data = json.load(f)

# Analyze timing breakdown
timing = runtime_data['timing_data']
total_time = timing['Total Runtime']
gp_fraction = timing['GP Training'] / total_time
print(f"GP training took {gp_fraction:.1%} of total time")

# Analyze convergence
gp_data = runtime_data['gp_data']
final_lengthscales = gp_data['lengthscales'][-1]
print(f"Final GP lengthscales: {final_lengthscales}")
```

## Integration Checklist

- [ ] Import `BOBEDataCollector` in your BOBE class
- [ ] Initialize data collector in `__init__()`
- [ ] Add timing calls around major phases
- [ ] Record GP hyperparameters after each training
- [ ] Record best log-likelihood after each update
- [ ] Record best parameters when they improve
- [ ] Save runtime data at the end of the run
- [ ] Create summary plots using `BOBESummaryPlotter`
- [ ] Test with a simple example problem

## Dependencies

**Required:**
- `matplotlib` (plotting backend)
- `numpy` (data handling)

**Optional:**
- `seaborn` (enhanced styling)
- `getdist` (advanced posterior analysis)

## Examples

See `examples/runtime_data_collection_example.py` for a complete integration example and `test_summary_plots.py` for plotting examples.

## Troubleshooting

**Issue:** Plots look cluttered with too many points
**Solution:** Use `thin_factor` parameter to subsample data

**Issue:** GP hyperparameter plots are hard to read
**Solution:** Use `log_scale=True` for lengthscales

**Issue:** Memory usage is high during long runs
**Solution:** Save/clear runtime data periodically and create intermediate plots

**Issue:** Plots are missing some data
**Solution:** Ensure data collection calls are in the right places in your BOBE loop

---

The summary plotting system is designed to be flexible and easy to integrate. Start with the basic integration and then customize as needed for your specific use case!
