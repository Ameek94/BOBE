# BOBE Timing System

The timing system in BOBE has been simplified and integrated directly into the results manager. Timing data is automatically collected for key phases during BOBE runs and is available in the results.

## Automatic Timing Collection

BOBE now automatically measures timing for these key phases:

- **GP Training**: Time spent fitting/refitting GP hyperparameters
- **Acquisition Optimization**: Time spent optimizing acquisition function  
- **True Objective Evaluations**: Time spent evaluating the true likelihood
- **Nested Sampling**: Time spent in nested sampling runs
- **MCMC Sampling**: Time spent in MCMC sampling for GP posterior

## Accessing Timing Data

### From BOBE Results

```python
from jaxbo.bo import BOBE

# Run BOBE normally - timing is collected automatically
results = sampler.run()

# Access timing data from comprehensive results
timing_data = results['comprehensive']['timing']

print(f"Total Runtime: {timing_data['total_runtime']:.1f} seconds")

for phase, time_spent in timing_data['phase_times'].items():
    if time_spent > 0:
        percentage = timing_data['percentages'].get(phase, 0)
        print(f"{phase}: {time_spent:.1f}s ({percentage:.1f}%)")
```

### Timing Data Structure

```python
timing_data = {
    'phase_times': {
        'GP Training': 45.2,
        'Acquisition Optimization': 23.1, 
        'True Objective Evaluations': 12.5,
        'Nested Sampling': 120.8,
        'MCMC Sampling': 67.3
    },
    'percentages': {
        'GP Training': 16.8,
        'Acquisition Optimization': 8.6,
        'True Objective Evaluations': 4.6,
        'Nested Sampling': 44.9,
        'MCMC Sampling': 25.0
    },
    'total_runtime': 268.9
}
```

## Saved Files

Timing data is automatically saved to:

- `{output_file}_timing.json`: Standalone timing data in JSON format
- `{output_file}_results.pkl`: Comprehensive results including timing
- `{output_file}_results.pkl`: Full Python objects including timing

## Terminal Output

BOBE automatically prints a timing summary at the end of each run:

```
==================================================
TIMING SUMMARY
==================================================
Total Runtime: 268.93 seconds (4.48 minutes)
GP Training: 45.2s (16.8%)
Acquisition Optimization: 23.1s (8.6%)
True Objective Evaluations: 12.5s (4.6%)
Nested Sampling: 120.8s (44.9%)
MCMC Sampling: 67.3s (25.0%)
==================================================
```

## No Manual Integration Required

Unlike the previous complex timing system, this simplified version requires no manual integration:

- ✅ **Automatic**: Timing starts automatically when BOBE runs
- ✅ **Comprehensive**: All key phases are timed
- ✅ **Integrated**: Results include timing in standard format
- ✅ **Saved**: Timing data saved in multiple formats
- ✅ **Simple**: No complex setup or teardown needed

## Legacy Compatibility

The timing data is integrated into the existing results structure and plotting system, so all existing plotting and analysis code will work seamlessly with the new timing information.
