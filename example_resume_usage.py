#!/usr/bin/env python3
"""
Example of how to use the resume functionality in a real BOBE optimization.

This shows the minimal changes needed to add resume capability to an existing optimization script.
"""

import numpy as np
import sys
import os

# Add the jaxbo package to the path
sys.path.insert(0, os.path.abspath('.'))

from jaxbo.utils.results import create_resumable_results


def example_optimization_with_resume():
    """
    Example showing how to modify an existing BOBE optimization to support resuming.
    
    Key changes from a regular BOBE run:
    1. Use create_resumable_results() instead of BOBEResults()
    2. Check if resuming to determine starting iteration
    3. All other code remains the same!
    """
    
    print("=== Example BOBE Optimization with Resume Support ===")
    
    # Problem setup (same as always)
    param_names = ['amplitude', 'frequency', 'phase']
    param_labels = [r'A', r'\omega', r'\phi']
    param_bounds = np.array([[0.1, 2.0], [1.0, 10.0], [0.0, 2*np.pi]])
    
    # CHANGE 1: Use create_resumable_results instead of BOBEResults
    # This automatically detects and loads existing results if available
    results_manager = create_resumable_results(
        output_file="example_optimization",
        param_names=param_names,
        param_labels=param_labels,
        param_bounds=param_bounds,
        settings={'max_iterations': 50, 'acquisition': 'LogEI'},
        likelihood_name="sinusoidal_fit"
    )
    
    # CHANGE 2: Check if we're resuming and determine starting iteration
    if results_manager.is_resuming():
        start_iteration = results_manager.get_last_iteration() + 1
        print(f"Resuming from iteration {start_iteration}")
        print(f"Previous data: {len(results_manager.acquisition_values)} acquisition evaluations")
    else:
        start_iteration = 1
        print("Starting fresh optimization")
    
    # CHANGE 3: Start loop from the appropriate iteration
    # (Rest of the optimization code remains exactly the same!)
    max_iterations = 50
    
    for iteration in range(start_iteration, max_iterations + 1):
        print(f"Iteration {iteration}")
        
        # Simulate your normal BOBE optimization steps here
        # 1. Train GP
        # 2. Optimize acquisition function
        # 3. Evaluate true objective
        # 4. Check convergence
        
        # Example simulated values
        acquisition_value = np.random.exponential(1.5)
        lengthscales = [np.random.gamma(1.5, 0.3) for _ in range(3)]
        outputscale = np.random.gamma(1, 0.8)
        best_loglike = -100 + iteration * 1.5 + np.random.normal(0, 0.5)
        
        # Update results (same as always)
        results_manager.update_acquisition(iteration, acquisition_value, 'LogEI')
        results_manager.update_gp_hyperparams(iteration, lengthscales, outputscale)
        results_manager.update_best_loglike(iteration, best_loglike)
        
        # Convergence check every 10 iterations
        if iteration % 10 == 0:
            logz_dict = {
                'mean': -105 + iteration * 1.2,
                'lower': -106 + iteration * 1.2,
                'upper': -104 + iteration * 1.2
            }
            converged = iteration >= 40  # Simulate convergence at iteration 40
            results_manager.update_convergence(iteration, logz_dict, converged, threshold=0.5)
            
            if converged:
                print(f"Converged at iteration {iteration}!")
                break
        
        # Optional: Save intermediate results every 10 iterations for crash recovery
        if iteration % 10 == 0:
            results_manager.save_intermediate()
            print(f"Saved intermediate results at iteration {iteration}")
    
    # Final results (same as always)
    n_samples = 500
    samples = np.random.multivariate_normal([1.0, 5.0, np.pi], 
                                          np.diag([0.1, 0.5, 0.2]), 
                                          n_samples)
    weights = np.random.exponential(1, n_samples)
    weights /= np.sum(weights)
    loglikes = np.random.normal(-50, 5, n_samples)
    
    final_logz = {'mean': -45, 'lower': -46, 'upper': -44}
    
    results_manager.finalize(
        samples=samples,
        weights=weights,
        loglikes=loglikes,
        logz_dict=final_logz,
        converged=True,
        termination_reason="Convergence achieved",
        gp_info={'final_performance': 'excellent'}
    )
    
    print(f"Optimization completed!")
    print(f"Total iterations: {results_manager.get_last_iteration()}")
    print(f"Final logZ: {final_logz['mean']:.2f} ± {final_logz['upper'] - final_logz['lower']:.2f}")
    
    return results_manager


def demonstrate_interruption_and_resume():
    """Demonstrate what happens when an optimization is interrupted and resumed."""
    
    print("\n" + "="*60)
    print("DEMONSTRATION: Interruption and Resume")
    print("="*60)
    
    # Clean up existing files
    import glob
    for f in glob.glob("example_optimization*"):
        os.remove(f)
        print(f"Cleaned up: {f}")
    
    print("\n--- Running optimization for first time ---")
    results1 = example_optimization_with_resume()
    
    print("\n--- 'Interrupting' and resuming the same optimization ---")
    print("(In practice, this would be running the same script again after a crash)")
    results2 = example_optimization_with_resume()
    
    print(f"\n--- Comparison ---")
    print(f"First run total iterations: {results1.get_last_iteration()}")
    print(f"Resume run total iterations: {results2.get_last_iteration()}")
    print(f"Same number of data points: {len(results1.acquisition_values) == len(results2.acquisition_values)}")


if __name__ == "__main__":
    demonstrate_interruption_and_resume()
    
    print("\n" + "="*60)
    print("USAGE SUMMARY:")
    print("="*60)
    print("To add resume capability to any BOBE optimization:")
    print("1. Replace BOBEResults() with create_resumable_results()")
    print("2. Check results_manager.is_resuming() to see if resuming")
    print("3. Use results_manager.get_last_iteration() + 1 for start iteration")
    print("4. All other code remains exactly the same!")
    print("5. Optionally call save_intermediate() periodically for crash recovery")
