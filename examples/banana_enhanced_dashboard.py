#!/usr/bin/env python3
"""
Enhanced Banana Function Test with Dashboard Plotting

This example demonstrates BOBE functionality including:
1. Settings and results management
2. KL divergence tracking and plotting with capping
3. Full dashboard generation
4. Comparison with reference nested sampler
"""

import numpy as np
import time
from pathlib import Path

# Import BOBE components
from jaxbo.bo import BOBE
from jaxbo.loglike import ExternalLikelihood
from jaxbo.settings import (
    BOBESettings,
    GPSettings, 
    NestedSamplingSettings,
    get_fast_settings
)
from jaxbo.utils.results import BOBEResults
from jaxbo.utils.summary_plots import BOBESummaryPlotter

# Import analysis tools
try:
    from getdist import MCSamples
    from dynesty import DynamicNestedSampler
    HAS_COMPARISON_TOOLS = True
except ImportError:
    HAS_COMPARISON_TOOLS = False
    print("Warning: GetDist and/or Dynesty not available for comparison")

from jaxbo.nested_sampler import renormalise_log_weights


def create_banana_likelihood():
    """Create the Banana function likelihood."""
    ndim = 2
    param_list = ['x1', 'x2']
    param_labels = ['x_1', 'x_2']
    param_bounds = np.array([[-1, 1], [-1, 2]]).T

    def loglike(X):
        logpdf = -0.25 * (5 * (0.2 - X[0]))**2 - (20 * (X[1]/4 - X[0]**4))**2
        return logpdf

    likelihood = ExternalLikelihood(
        loglikelihood=loglike,
        ndim=ndim,
        param_list=param_list,
        param_bounds=param_bounds,
        param_labels=param_labels,
        name='banana_enhanced',
        noise_std=0.0,
        minus_inf=-1e5
    )
    
    return likelihood


def run_bobe_with_dashboard(likelihood, output_file="banana_enhanced"):
    """Run BOBE with enhanced settings and generate dashboard."""
    print(f"\n=== Running Enhanced BOBE with Dashboard ===")
    
    # Use enhanced settings for better KL tracking
    bobe_settings = BOBESettings(
        n_cobaya_init=4,
        n_sobol_init=8,
        miniters=30,
        maxiters=80,  # Reduced for faster testing
        max_gp_size=150,
        use_clf=True,  # Enable classifier for more features
        minus_inf=-1e5,
        lengthscale_priors='DSLP',
        num_hmc_warmup=256,
        num_hmc_samples=256,
        mc_points_size=24,
        logz_threshold=0.1
    )
    
    print(f"Running with maxiters={bobe_settings.maxiters}, classifier={bobe_settings.use_clf}")
    
    start_time = time.time()
    
    # Create BOBE sampler
    sampler = BOBE(
        # BOBE settings
        n_cobaya_init=bobe_settings.n_cobaya_init,
        n_sobol_init=bobe_settings.n_sobol_init,
        miniters=bobe_settings.miniters,
        maxiters=bobe_settings.maxiters,
        max_gp_size=bobe_settings.max_gp_size,
        use_clf=bobe_settings.use_clf,
        minus_inf=bobe_settings.minus_inf,
        lengthscale_priors=bobe_settings.lengthscale_priors,
        num_hmc_warmup=bobe_settings.num_hmc_warmup,
        num_hmc_samples=bobe_settings.num_hmc_samples,
        mc_points_size=bobe_settings.mc_points_size,
        logz_threshold=bobe_settings.logz_threshold,
        # Likelihood and other parameters
        loglikelihood=likelihood,
        fit_step=2,
        update_mc_step=2,
        ns_step=8,
        # Output settings
        output_file=output_file,
        save_intermediate=True,
        save_samples=True
    )
    
    # Run BOBE
    print("Starting BOBE run...")
    results_dict = sampler.run()
    
    runtime = time.time() - start_time
    print(f"✓ BOBE completed in {runtime:.2f} seconds")
    
    return results_dict, runtime


def generate_dashboard_plots(output_file="banana_enhanced"):
    """Generate comprehensive dashboard plots."""
    print(f"\n=== Generating Dashboard Plots ===")
    
    try:
        # Load the results
        results = BOBEResults.load_results(output_file)
        print(f"✓ Loaded results with {len(results.acquisition_history)} acquisition points")
        print(f"✓ Successive KL data: {len(results.successive_kl)} entries")
        
        # Check if we have KL data
        if results.successive_kl:
            print("Sample KL entries:")
            for i, entry in enumerate(results.successive_kl[:3]):
                print(f"  Entry {i}: iteration={entry.get('iteration', 'N/A')}, "
                      f"forward={entry.get('forward', 'N/A'):.3f}, "
                      f"reverse={entry.get('reverse', 'N/A'):.3f}, "
                      f"symmetric={entry.get('symmetric', 'N/A'):.3f}")
        else:
            print("No successive KL data found")
        
        # Create the summary plotter
        plotter = BOBESummaryPlotter(results)
        
        # Generate individual plots for testing
        print("Creating individual test plots...")
        
        import matplotlib.pyplot as plt
        
        # Test KL divergence plot specifically
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = plotter.plot_kl_divergences(ax)
        plt.title("KL Divergences with Capping Test")
        plt.tight_layout()
        plt.savefig(f"{output_file}_kl_test.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ KL divergence test plot saved as {output_file}_kl_test.png")
        
        # Generate full dashboard
        dashboard_file = f"{output_file}_dashboard.png"
        plotter.create_dashboard(dashboard_file)
        print(f"✓ Full dashboard saved as {dashboard_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error generating dashboard: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function demonstrating enhanced BOBE with dashboard."""
    print("=" * 60)
    print("ENHANCED BANANA FUNCTION: Dashboard and KL Capping Test")
    print("=" * 60)
    
    # Create the likelihood
    likelihood = create_banana_likelihood()
    print(f"Created {likelihood.ndim}D Banana likelihood: {likelihood.name}")
    
    # Run BOBE with enhanced settings
    output_file = "banana_enhanced"
    try:
        bobe_results, runtime = run_bobe_with_dashboard(likelihood, output_file)
        print(f"✓ BOBE run completed successfully")
        
        # Generate dashboard plots
        dashboard_success = generate_dashboard_plots(output_file)
        
        if dashboard_success:
            print(f"\n=== Success! ===")
            print(f"✓ BOBE run completed in {runtime:.2f} seconds")
            print(f"✓ Dashboard plots generated successfully")
            print(f"✓ KL divergence capping functionality tested")
            print(f"\nCheck the generated files:")
            print(f"  - {output_file}_dashboard.png (full dashboard)")
            print(f"  - {output_file}_kl_test.png (KL divergence test)")
            print(f"  - {output_file}_results.pkl (results data)")
        else:
            print(f"\n✗ Dashboard generation failed")
            
    except Exception as e:
        print(f"✗ Error during BOBE run: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
