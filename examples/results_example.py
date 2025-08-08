#!/usr/bin/env python3
"""
Example demonstrating the simplified BOBE results management system.

This example shows how the streamlined results storage works with only
essential data: final samples, weights, evidence evolution, and convergence info.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the results system
from jaxbo.results import BOBEResults, load_bobe_results
from jaxbo.bo import BOBE
from jaxbo.loglike import ExternalLikelihood

# Mock likelihood for demonstration
class MockLikelihood(ExternalLikelihood):
    """Simple 2D likelihood for testing."""
    
    def __init__(self):
        self.name = "mock_2d_gaussian"
        self.param_list = ['x', 'y']
        self.param_labels = [r'$x$', r'$y$']
        self.param_bounds = np.array([[-3, 3], [-3, 3]])
        
    def __call__(self, params, logp_args=(), logp_kwargs={}):
        """2D Gaussian likelihood."""
        x, y = params.flatten()[:2]
        return -(x**2 + y**2) / 2.0  # Log of 2D Gaussian


def demonstrate_simplified_results():
    """Show how simplified results are created during a BOBE run."""
    print("=== Creating Simplified BOBE Results ===")
    
    # Create a results manager directly for demonstration
    results_manager = BOBEResults(
        output_file="demo_results",
        param_names=['x', 'y'],
        param_labels=[r'$x$', r'$y$'],
        param_bounds=np.array([[-3, 3], [-3, 3]]),
        settings={'maxiters': 100, 'use_clf': True},
        likelihood_name="mock_2d_gaussian"
    )
    
    print(f"Initialized results manager for {results_manager.ndim}D problem")
    
    # Simulate some convergence checks (this is what we track)
    np.random.seed(42)
    for i in [5, 10, 15, 20]:
        logz_dict = {
            'mean': -5.0 + np.random.normal(0, 0.1),
            'lower': -5.2 + np.random.normal(0, 0.05),
            'upper': -4.8 + np.random.normal(0, 0.05)
        }
        results_manager.update_convergence(
            iteration=i,
            logz_dict=logz_dict,
            converged=(i >= 15),
            threshold=0.1
        )
    
    # Generate mock final samples
    n_samples = 1000
    samples = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], n_samples)
    weights = np.random.exponential(1.0, n_samples)
    weights /= np.sum(weights)  # Normalize
    loglikes = -(samples[:, 0]**2 + samples[:, 1]**2) / 2.0
    
    final_logz_dict = {
        'mean': -4.95,
        'lower': -5.05,
        'upper': -4.85
    }
    
    # Finalize results
    results_manager.finalize(
        samples=samples,
        weights=weights,
        loglikes=loglikes,
        logz_dict=final_logz_dict,
        converged=True,
        termination_reason="Convergence achieved"
    )
    
    print(f"Finalized results with {len(samples)} samples")
    return results_manager


def demonstrate_simplified_analysis(results_manager):
    """Show how to analyze the simplified results."""
    print("\n=== Analyzing Simplified Results ===")
    
    # Get the simplified results dictionary
    results = results_manager.get_results_dict()
    
    print(f"Evidence: logZ = {results['logz']:.3f} ± {results['logzerr']:.3f}")
    print(f"Effective samples: {results['n_effective']}")
    print(f"Runtime: {results['run_info']['runtime_hours']:.2f} hours")
    print(f"Converged: {results['converged']}")
    print(f"Termination: {results['termination_reason']}")
    
    # Convergence information
    print(f"\nConvergence checks: {len(results['convergence_history'])}")
    print(f"Evidence evolution: {len(results['logz_history'])} points")
    
    # Parameter bounds
    print(f"\nParameter bounds:")
    for i, name in enumerate(results['param_names']):
        bounds = results['param_bounds'][i]
        print(f"  {name}: [{bounds[0]:.2f}, {bounds[1]:.2f}]")


def demonstrate_simplified_file_formats():
    """Show the simplified file formats that are saved."""
    print("\n=== Simplified File Formats ===")
    
    # List files that would be created
    output_file = "demo_results"
    expected_files = [
        f"{output_file}_results.npz",      # Main compressed results
        f"{output_file}_results.pkl",      # Full Python object
        f"{output_file}.txt",              # GetDist format chain
        f"{output_file}.paramnames",       # GetDist parameter names
        f"{output_file}.ranges",           # GetDist parameter ranges
        f"{output_file}_1.txt",            # CosmoMC format chain
        f"{output_file}_stats.json",       # Summary statistics
        f"{output_file}_convergence.npz",  # Convergence diagnostics only
        f"{output_file}_intermediate.json" # Intermediate results
    ]
    
    print("Simplified files created:")
    for filename in expected_files:
        print(f"  {filename}")
        if Path(filename).exists():
            size = Path(filename).stat().st_size
            print(f"    → {size} bytes")
    
    print("\nFiles NOT created in simplified version:")
    print("  • GP evolution diagnostics")
    print("  • Iteration-by-iteration tracking") 
    print("  • Acquisition function history")
    print("  • Hyperparameter evolution")


def demonstrate_evidence_evolution_plotting():
    """Show how to plot evidence evolution."""
    print("\n=== Evidence Evolution Plotting ===")
    
    print("Example code for plotting evidence evolution:")
    print("""
    results = results_manager.get_results_dict()
    logz_history = results['logz_history']
    
    if logz_history:
        iterations = [x['iteration'] for x in logz_history]
        logz_values = [x['logz'] for x in logz_history]
        logz_errors = [x['logz_err'] for x in logz_history]
        
        plt.figure(figsize=(8, 6))
        plt.errorbar(iterations, logz_values, yerr=logz_errors, 
                    marker='o', capsize=5)
        plt.xlabel('Iteration')
        plt.ylabel('Log Evidence')
        plt.title('Evidence Convergence')
        plt.grid(True, alpha=0.3)
        plt.savefig('evidence_evolution.png')
    """)


def demonstrate_convergence_analysis():
    """Show how to analyze convergence information."""
    print("\n=== Convergence Analysis ===")
    
    print("Example code for convergence analysis:")
    print("""
    results = results_manager.get_results_dict()
    conv_history = results['convergence_history']
    
    if conv_history:
        print("Convergence timeline:")
        for conv in conv_history:
            print(f"Iteration {conv['iteration']}: "
                  f"delta={conv['delta']:.3f}, "
                  f"threshold={conv['threshold']:.3f}, "
                  f"converged={conv['converged']}")
        
        # Plot convergence delta over time
        iterations = [conv['iteration'] for conv in conv_history]
        deltas = [conv['delta'] for conv in conv_history]
        thresholds = [conv['threshold'] for conv in conv_history]
        
        plt.figure(figsize=(8, 6))
        plt.plot(iterations, deltas, 'o-', label='Evidence uncertainty')
        plt.plot(iterations, thresholds, '--', label='Convergence threshold')
        plt.xlabel('Iteration')
        plt.ylabel('Delta logZ')
        plt.title('Convergence Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('convergence_progress.png')
    """)


if __name__ == "__main__":
    # Run simplified demonstrations
    results_manager = demonstrate_simplified_results()
    demonstrate_simplified_analysis(results_manager)
    demonstrate_simplified_file_formats()
    demonstrate_evidence_evolution_plotting()
    demonstrate_convergence_analysis()
    
    print("\n=== Simplified Results Summary ===")
    print("The simplified BOBE results system stores only:")
    print("✓ Final samples and weights")
    print("✓ Evidence evolution during convergence checks")
    print("✓ Convergence information and diagnostics")
    print("✓ Basic run metadata")
    print("\nRemoved from storage:")
    print("✗ Iteration-by-iteration GP diagnostics")
    print("✗ Acquisition function values")
    print("✗ GP hyperparameter evolution")
    print("✗ Detailed per-iteration tracking")
    
    print("\nThis significantly reduces storage requirements while")
    print("preserving all essential information for posterior analysis.")
    
    # Clean up demo files
    import os
    demo_files = [
        "demo_results_results.npz",
        "demo_results_results.pkl",
        "demo_results.txt",
        "demo_results.paramnames",
        "demo_results.ranges",
        "demo_results_1.txt",
        "demo_results_stats.json",
        "demo_results_convergence.npz",
        "demo_results_intermediate.json"
    ]
    
    print(f"\nCleaning up {len(demo_files)} demo files...")
    for file in demo_files:
        if os.path.exists(file):
            os.remove(file)
