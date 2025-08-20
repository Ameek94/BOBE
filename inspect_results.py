#!/usr/bin/env python3
"""
Quick script to inspect the BOBEResults structure and KL divergence data.
"""

import sys
sys.path.insert(0, '/Users/amkpd/cosmocodes/JaxBo')

from jaxbo.utils.results import load_bobe_results

def inspect_results():
    """Inspect the loaded results to understand the data structure."""
    
    results = load_bobe_results('banana_comprehensive_test')
    
    print("=== BOBE RESULTS INSPECTION ===")
    print(f"Results type: {type(results)}")
    print(f"Available attributes: {[attr for attr in dir(results) if not attr.startswith('_')]}")
    
    print(f"\nBasic info:")
    print(f"  Output file: {results.output_file}")
    print(f"  Number of dimensions: {results.ndim}")
    print(f"  Parameter names: {results.param_names}")
    print(f"  Parameter labels: {results.param_labels}")
    
    if hasattr(results, 'final_samples') and results.final_samples is not None:
        print(f"  Final samples shape: {results.final_samples.shape}")
    
    print(f"\nKL divergence data:")
    if hasattr(results, 'kl_iterations'):
        print(f"  kl_iterations: {results.kl_iterations}")
        print(f"  kl_iterations type: {type(results.kl_iterations)}")
    else:
        print(f"  No kl_iterations attribute")
    
    if hasattr(results, 'kl_divergences'):
        print(f"  kl_divergences: {len(results.kl_divergences)} entries")
        print(f"  kl_divergences type: {type(results.kl_divergences)}")
        if results.kl_divergences:
            print(f"  First KL entry: {results.kl_divergences[0]}")
    else:
        print(f"  No kl_divergences attribute")
    
    print(f"\nGP data:")
    if hasattr(results, 'gp_hyperparameters'):
        print(f"  gp_hyperparameters: {len(results.gp_hyperparameters)} entries")
        if results.gp_hyperparameters:
            first_key = list(results.gp_hyperparameters.keys())[0]
            print(f"  First GP entry (iter {first_key}): {results.gp_hyperparameters[first_key]}")
    else:
        print(f"  No gp_hyperparameters attribute")
    
    print(f"\nConvergence data:")
    if hasattr(results, 'convergence_history'):
        print(f"  convergence_history: {len(results.convergence_history)} entries")
        if results.convergence_history:
            print(f"  First convergence entry: {results.convergence_history[0]}")
    else:
        print(f"  No convergence_history attribute")

if __name__ == "__main__":
    inspect_results()
