#!/usr/bin/env python3
"""
Generate comprehensive plotting dashboard for the Banana function BOBE test results.

This script creates all diagnostic plots including the new KL divergence evolution plots
for the comprehensive test results from test_comprehensive_functionality.py.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the JaxBo path
sys.path.insert(0, '/Users/amkpd/cosmocodes/JaxBo')

from jaxbo.utils.summary_plots import BOBESummaryPlotter, create_summary_plots
from jaxbo.utils.results import load_bobe_results
from jaxbo.utils.logging_utils import get_logger

log = get_logger("[dashboard]")

def main():
    """Generate comprehensive plotting dashboard for Banana test results."""
    
    # File paths for the comprehensive test results
    results_file = 'banana_comprehensive_test_results.pkl'
    output_file = 'banana_comprehensive_test'
    
    print("🎨 GENERATING COMPREHENSIVE PLOTTING DASHBOARD")
    print("=" * 60)
    print(f"Results file: {results_file}")
    print(f"Output file base: {output_file}")
    
    # Check if results file exists
    if not Path(results_file).exists():
        print(f"❌ Error: Results file {results_file} not found!")
        print("Please run test_comprehensive_functionality.py first.")
        return 1
    
    try:
        # Load results
        print("📊 Loading BOBE results...")
        results = load_bobe_results('banana_comprehensive_test')
        if hasattr(results, 'final_samples') and results.final_samples is not None:
            print(f"✓ Loaded results with {len(results.final_samples)} samples")
        else:
            print("✓ Loaded results (no final samples available)")
        
        # Create plotter
        print("🎯 Creating summary plotter...")
        plotter = BOBESummaryPlotter(results)
        
        # Prepare data for plotting
        print("📊 Extracting data for plotting...")
        
        # Extract GP data using the proper method
        try:
            gp_data = results.get_gp_data()
            if gp_data and gp_data['iterations']:
                print(f"  ✓ GP data: {len(gp_data['iterations'])} iterations")
            else:
                gp_data = None
                print("  ⚠️  No GP data available")
        except Exception as e:
            print(f"  ❌ Failed to get GP data: {e}")
            gp_data = None
        
        # Extract best log-likelihood data using the proper method
        try:
            best_loglike_data = results.get_best_loglike_data()
            if best_loglike_data and best_loglike_data['iterations']:
                print(f"  ✓ Best log-likelihood data: {len(best_loglike_data['iterations'])} iterations")
            else:
                best_loglike_data = None
                print("  ⚠️  No best log-likelihood data available")
        except Exception as e:
            print(f"  ❌ Failed to get best log-likelihood data: {e}")
            best_loglike_data = None
        
        # Extract acquisition data using the proper method
        try:
            acquisition_data = results.get_acquisition_data()
            if acquisition_data and acquisition_data['iterations']:
                print(f"  ✓ Acquisition data: {len(acquisition_data['iterations'])} iterations")
            else:
                acquisition_data = None
                print("  ⚠️  No acquisition data available")
        except Exception as e:
            print(f"  ❌ Failed to get acquisition data: {e}")
            acquisition_data = None
        
        # Extract timing data using the proper method
        try:
            timing_data = results.get_timing_summary()
            if timing_data and timing_data['phase_times']:
                print(f"  ✓ Timing data available")
            else:
                timing_data = None
                print("  ⚠️  No timing data available")
        except Exception as e:
            print(f"  ❌ Failed to get timing data: {e}")
            timing_data = None
        
        # Check for KL divergence data
        if hasattr(results, 'kl_iterations') and results.kl_iterations:
            print("🔬 KL divergence data found:")
            print(f"  ✓ {len(results.kl_iterations)} KL measurements at iterations: {results.kl_iterations}")
            if hasattr(results, 'kl_divergences') and results.kl_divergences:
                sample_kl = results.kl_divergences[0]
                kl_types = list(sample_kl.keys())
                print(f"  ✓ KL types available: {kl_types}")
        else:
            print("⚠️  No KL divergence data found")
        
        # Generate individual plots
        print("\n📋 Generating individual diagnostic plots...")
        
        # Evidence evolution
        print("  📊 Evidence evolution...")
        fig, ax = plt.subplots(figsize=(10, 6))
        plotter.plot_evidence_evolution(ax=ax)
        plt.tight_layout()
        plt.savefig('banana_comprehensive_test_evidence_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: banana_comprehensive_test_evidence_evolution.png")
        
        # GP hyperparameters
        if gp_data:
            print("  🧠 GP lengthscales...")
            fig, ax = plt.subplots(figsize=(10, 6))
            plotter.plot_gp_lengthscales(gp_data=gp_data, ax=ax)
            plt.tight_layout()
            plt.savefig('banana_comprehensive_test_gp_lengthscales.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("    ✓ Saved: banana_comprehensive_test_gp_lengthscales.png")
            
            print("  🧠 GP output scale...")
            fig, ax = plt.subplots(figsize=(10, 6))
            plotter.plot_gp_outputscale(gp_data=gp_data, ax=ax)
            plt.tight_layout()
            plt.savefig('banana_comprehensive_test_gp_outputscale.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("    ✓ Saved: banana_comprehensive_test_gp_outputscale.png")
        
        # Best log-likelihood
        if best_loglike_data:
            print("  📈 Best log-likelihood evolution...")
            fig, ax = plt.subplots(figsize=(10, 6))
            plotter.plot_best_loglike_evolution(best_loglike_data=best_loglike_data, ax=ax)
            plt.tight_layout()
            plt.savefig('banana_comprehensive_test_best_loglike.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("    ✓ Saved: banana_comprehensive_test_best_loglike.png")
        
        # Convergence diagnostics
        print("  🎯 Convergence diagnostics...")
        fig, ax = plt.subplots(figsize=(10, 6))
        plotter.plot_convergence_diagnostics(ax=ax)
        plt.tight_layout()
        plt.savefig('banana_comprehensive_test_convergence.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: banana_comprehensive_test_convergence.png")
        
        # KL divergences
        print("  🔬 KL divergences evolution...")
        fig, ax = plt.subplots(figsize=(10, 6))
        plotter.plot_kl_divergences(ax=ax)
        plt.tight_layout()
        plt.savefig('banana_comprehensive_test_kl_divergences.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✓ Saved: banana_comprehensive_test_kl_divergences.png")
        
        # Timing breakdown
        if timing_data:
            print("  ⏱️  Timing breakdown...")
            fig, ax = plt.subplots(figsize=(10, 6))
            plotter.plot_timing_breakdown(timing_data=timing_data, ax=ax)
            plt.tight_layout()
            plt.savefig('banana_comprehensive_test_timing.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("    ✓ Saved: banana_comprehensive_test_timing.png")
        
        # Generate comprehensive dashboard
        print("\n🎨 Creating comprehensive summary dashboard...")
        dashboard_fig = plotter.create_summary_dashboard(
            gp_data=gp_data,
            best_loglike_data=best_loglike_data,
            acquisition_data=acquisition_data,
            timing_data=timing_data,
            save_path='banana_comprehensive_test_summary_dashboard.png'
        )
        plt.close(dashboard_fig)
        print("    ✓ Saved: banana_comprehensive_test_summary_dashboard.png")
        
        # Generate parameter triangle plot if possible
        try:
            from jaxbo.utils.summary_plots import plot_final_samples
            if hasattr(results, 'final_samples') and results.final_samples is not None:
                print("\n🔺 Creating parameter triangle plot...")
                
                # Prepare samples dictionary
                samples_dict = {
                    'x': results.final_samples,
                    'weights': getattr(results, 'final_weights', np.ones(len(results.final_samples)))
                }
                
                # Mock GP for the triangle plot
                class MockGP:
                    def __init__(self, samples):
                        self.train_x = samples[:50]  # Use first 50 samples as training points
                
                mock_gp = MockGP(results.final_samples)
                param_list = results.param_names if hasattr(results, 'param_names') else [f'x{i+1}' for i in range(results.final_samples.shape[1])]
                param_labels = results.param_labels if hasattr(results, 'param_labels') else [f'$x_{{{i+1}}}$' for i in range(results.final_samples.shape[1])]
                param_bounds = results.param_bounds.T if hasattr(results, 'param_bounds') else None
                
                plot_final_samples(
                    gp=mock_gp,
                    samples_dict=samples_dict,
                    param_list=param_list,
                    param_labels=param_labels,
                    param_bounds=param_bounds,
                    output_file='banana_comprehensive_test'
                )
                print("    ✓ Saved: banana_comprehensive_test_samples.pdf")
        except Exception as e:
            print(f"    ⚠️  Could not create triangle plot: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 DASHBOARD GENERATION COMPLETED!")
        print("\nGenerated files:")
        print("  📊 Individual diagnostic plots:")
        print("    - banana_comprehensive_test_evidence_evolution.png")
        if gp_data:
            print("    - banana_comprehensive_test_gp_lengthscales.png")
            print("    - banana_comprehensive_test_gp_outputscale.png")
        if best_loglike_data:
            print("    - banana_comprehensive_test_best_loglike.png")
        print("    - banana_comprehensive_test_convergence.png")
        print("    - banana_comprehensive_test_kl_divergences.png")
        if timing_data:
            print("    - banana_comprehensive_test_timing.png")
        print("  🎨 Summary dashboard:")
        print("    - banana_comprehensive_test_summary_dashboard.png")
        print("  🔺 Parameter analysis:")
        print("    - banana_comprehensive_test_samples.pdf (if available)")
        
        print(f"\n📁 All plots saved in: {Path.cwd()}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating dashboard: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
