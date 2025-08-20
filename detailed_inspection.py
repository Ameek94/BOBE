#!/usr/bin/env python3
"""
Detailed investigation of available data in BOBEResults for plotting.
"""

import sys
sys.path.insert(0, '/Users/amkpd/cosmocodes/JaxBo')

from jaxbo.utils.results import load_bobe_results

def detailed_inspection():
    """Detailed inspection of what data is available for plotting."""
    
    results = load_bobe_results('banana_comprehensive_test')
    
    print("=== DETAILED BOBE RESULTS DATA INSPECTION ===")
    
    # Check GP data
    print(f"\n🧠 GP HYPERPARAMETER DATA:")
    if hasattr(results, 'gp_iterations') and results.gp_iterations:
        print(f"  ✓ gp_iterations: {len(results.gp_iterations)} entries")
        print(f"    Iterations: {results.gp_iterations[:5]}..." if len(results.gp_iterations) > 5 else f"    Iterations: {results.gp_iterations}")
    else:
        print(f"  ❌ No gp_iterations")
    
    if hasattr(results, 'gp_lengthscales') and results.gp_lengthscales:
        print(f"  ✓ gp_lengthscales: {len(results.gp_lengthscales)} entries")
        print(f"    First few: {results.gp_lengthscales[:3]}")
    else:
        print(f"  ❌ No gp_lengthscales")
    
    if hasattr(results, 'gp_outputscales') and results.gp_outputscales:
        print(f"  ✓ gp_outputscales: {len(results.gp_outputscales)} entries")
        print(f"    First few: {results.gp_outputscales[:3]}")
    else:
        print(f"  ❌ No gp_outputscales")
    
    # Check best log-likelihood data
    print(f"\n📈 BEST LOG-LIKELIHOOD DATA:")
    if hasattr(results, 'best_loglike_iterations') and results.best_loglike_iterations:
        print(f"  ✓ best_loglike_iterations: {len(results.best_loglike_iterations)} entries")
        print(f"    Iterations: {results.best_loglike_iterations[:5]}..." if len(results.best_loglike_iterations) > 5 else f"    Iterations: {results.best_loglike_iterations}")
    else:
        print(f"  ❌ No best_loglike_iterations")
    
    if hasattr(results, 'best_loglike_values') and results.best_loglike_values:
        print(f"  ✓ best_loglike_values: {len(results.best_loglike_values)} entries")
        print(f"    First few: {results.best_loglike_values[:3]}")
    else:
        print(f"  ❌ No best_loglike_values")
    
    # Check acquisition data
    print(f"\n🎯 ACQUISITION DATA:")
    if hasattr(results, 'acquisition_iterations') and results.acquisition_iterations:
        print(f"  ✓ acquisition_iterations: {len(results.acquisition_iterations)} entries")
        print(f"    Iterations: {results.acquisition_iterations[:5]}..." if len(results.acquisition_iterations) > 5 else f"    Iterations: {results.acquisition_iterations}")
    else:
        print(f"  ❌ No acquisition_iterations")
    
    if hasattr(results, 'acquisition_values') and results.acquisition_values:
        print(f"  ✓ acquisition_values: {len(results.acquisition_values)} entries")
        print(f"    First few: {results.acquisition_values[:3]}")
    else:
        print(f"  ❌ No acquisition_values")
    
    # Check timing data
    print(f"\n⏱️  TIMING DATA:")
    if hasattr(results, 'phase_times') and results.phase_times:
        print(f"  ✓ phase_times: {results.phase_times}")
    else:
        print(f"  ❌ No phase_times")
    
    # Check evidence evolution
    print(f"\n📊 EVIDENCE EVOLUTION DATA:")
    if hasattr(results, 'logz_evolution') and results.logz_evolution:
        print(f"  ✓ logz_evolution: {len(results.logz_evolution)} entries")
        print(f"    Type: {type(results.logz_evolution)}")
        if isinstance(results.logz_evolution, dict):
            print(f"    First few: {list(results.logz_evolution.items())[:3]}")
        else:
            print(f"    First few: {results.logz_evolution[:3]}")
    else:
        print(f"  ❌ No logz_evolution")
    
    # Check convergence data
    print(f"\n🎯 CONVERGENCE DATA:")
    if hasattr(results, 'convergence_history') and results.convergence_history:
        print(f"  ✓ convergence_history: {len(results.convergence_history)} entries")
        for i, conv in enumerate(results.convergence_history):
            print(f"    Entry {i}: iter={conv.iteration}, delta={conv.delta:.3f}, converged={conv.converged}")
    else:
        print(f"  ❌ No convergence_history")
    
    # Check KL data (again)
    print(f"\n🔬 KL DIVERGENCE DATA:")
    if hasattr(results, 'kl_iterations') and results.kl_iterations:
        print(f"  ✓ kl_iterations: {results.kl_iterations}")
    else:
        print(f"  ❌ No kl_iterations data")
    
    if hasattr(results, 'kl_divergences') and results.kl_divergences:
        print(f"  ✓ kl_divergences: {len(results.kl_divergences)} entries")
        if results.kl_divergences:
            print(f"    First entry: {results.kl_divergences[0]}")
    else:
        print(f"  ❌ No kl_divergences data")
    
    # Test data extraction methods
    print(f"\n🔧 TESTING DATA EXTRACTION METHODS:")
    
    try:
        gp_data = results.get_gp_data()
        print(f"  ✓ get_gp_data() works: {type(gp_data)}")
        if gp_data:
            print(f"    Keys: {list(gp_data.keys())}")
    except Exception as e:
        print(f"  ❌ get_gp_data() failed: {e}")
    
    try:
        loglike_data = results.get_best_loglike_data()
        print(f"  ✓ get_best_loglike_data() works: {type(loglike_data)}")
        if loglike_data:
            print(f"    Keys: {list(loglike_data.keys())}")
    except Exception as e:
        print(f"  ❌ get_best_loglike_data() failed: {e}")
    
    try:
        acq_data = results.get_acquisition_data()
        print(f"  ✓ get_acquisition_data() works: {type(acq_data)}")
        if acq_data:
            print(f"    Keys: {list(acq_data.keys())}")
    except Exception as e:
        print(f"  ❌ get_acquisition_data() failed: {e}")
    
    try:
        timing_data = results.get_timing_summary()
        print(f"  ✓ get_timing_summary() works: {type(timing_data)}")
        if timing_data:
            print(f"    Keys: {list(timing_data.keys())}")
    except Exception as e:
        print(f"  ❌ get_timing_summary() failed: {e}")

if __name__ == "__main__":
    detailed_inspection()
