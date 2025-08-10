#!/usr/bin/env python3
"""
Example showing how to access timing information from BOBE results.
"""

import numpy as np
from jaxbo.loglike import ExternalLikelihood
from jaxbo.bo import BOBE

def simple_loglike(x):
    """Simple test function."""
    return -0.5 * np.sum(x**2)

def example_timing_usage():
    """Example showing how timing data is automatically collected and accessed."""
    
    # Setup a simple problem
    ndim = 2
    param_bounds = np.array([[-2, 2], [-2, 2]]).T
    param_list = ['x1', 'x2']
    param_labels = ['x_1', 'x_2']
    
    likelihood = ExternalLikelihood(
        loglikelihood=simple_loglike,
        ndim=ndim,
        param_list=param_list,
        param_bounds=param_bounds,
        param_labels=param_labels,
        name='simple_test'
    )
    
    # Create BOBE sampler (timing is automatically integrated)
    sampler = BOBE(
        n_cobaya_init=4, 
        n_sobol_init=8,
        miniters=5, 
        maxiters=15,  # Keep small for demo
        loglikelihood=likelihood,
        mc_points_method='NUTS',
        lengthscale_priors='DSLP'
    )
    
    print("Running BOBE with automatic timing measurement...")
    results = sampler.run()
    
    # Access timing information from results
    timing_data = results['comprehensive']['timing']
    
    print("\n" + "="*50)
    print("AUTOMATIC TIMING RESULTS")
    print("="*50)
    
    print(f"Total Runtime: {timing_data['total_runtime']:.1f} seconds")
    print(f"Total Runtime: {timing_data['total_runtime']/60:.1f} minutes")
    
    print("\nPhase Breakdown:")
    print("-" * 40)
    for phase, time_spent in timing_data['phase_times'].items():
        if time_spent > 0:
            percentage = timing_data['percentages'].get(phase, 0)
            print(f"{phase:25s}: {time_spent:6.1f}s ({percentage:5.1f}%)")
    
    print("="*50)
    
    # The timing data is also saved automatically to files
    print(f"\n✓ Timing data saved to: {likelihood.name}_timing.json")
    print(f"✓ Full results saved to: {likelihood.name}_results.npz")
    print(f"✓ Comprehensive results available in results['comprehensive']")
    
    return results

if __name__ == "__main__":
    results = example_timing_usage()
