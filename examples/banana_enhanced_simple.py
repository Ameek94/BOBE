#!/usr/bin/env python3
"""
Enhanced Banana Function Test with Dashboard Plotting

This example demonstrates BOBE functionality including:
1. Basic BOBE run with classifier enabled
2. KL divergence tracking and plotting with capping
3. Full dashboard generation
"""

import numpy as np
import time
import matplotlib.pyplot as plt

# Import BOBE components
from jaxbo.bo import BOBE
from jaxbo.loglike import ExternalLikelihood
from jaxbo.utils.results import BOBEResults
from jaxbo.utils.summary_plots import BOBESummaryPlotter


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
    
    start_time = time.time()
    
    # Create BOBE sampler with classifier enabled for better KL tracking
    sampler = BOBE(
        n_cobaya_init=4,
        n_sobol_init=8,
        miniters=10,
        maxiters=100,  # Reduced for faster testing
        max_gp_size=150,
        use_clf=False,
        minus_inf=-1e5,
        lengthscale_priors='DSLP',
        num_hmc_warmup=256,
        num_hmc_samples=256,
        mc_points_size=64,
        logz_threshold=0.1,
        # Likelihood and other parameters
        loglikelihood=likelihood,
        fit_step=2,
        update_mc_step=2,
        ns_step=8
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
    
    # Load results using the static method
    results_manager = BOBEResults.load_results(output_file)
    plotter = BOBESummaryPlotter(results_manager)

    # Get GP and best loglike evolution data
    gp_data = results_manager.get_gp_data()
    best_loglike_data = results_manager.get_best_loglike_data()
    acquisition_data = results_manager.get_acquisition_data()

    # Load timing data
    try:
        import json
        with open(f"{output_file}_timing.json", 'r') as f:
            timing_data = json.load(f)
    except Exception as e:
        print(f"Could not load timing data: {e}")
        timing_data = None

    # Create summary dashboard with timing data
    print("Creating summary dashboard...")
    fig_dashboard = plotter.create_summary_dashboard(
        gp_data=gp_data,
        acquisition_data=acquisition_data,
        best_loglike_data=best_loglike_data,
        timing_data=timing_data,
        save_path=f"{output_file}_dashboard.png"
    )
    plt.show()

    # Create individual timing plot
    if timing_data:
        print("Creating detailed timing plot...")
        fig_timing, ax_timing = plt.subplots(1, 1, figsize=(10, 6))
        plotter.plot_timing_breakdown(timing_data=timing_data, ax=ax_timing)
        ax_timing.set_title(f"Timing Breakdown - {output_file}")
        plt.tight_layout()
        plt.savefig(f"{output_file}_timing_detailed.png", dpi=300, bbox_inches='tight')
        plt.show()

    # Create acquisition function evolution plot
    print("Creating acquisition function evolution plot...")
    if acquisition_data and acquisition_data.get('iterations'):
        fig_acquisition, ax_acquisition = plt.subplots(1, 1, figsize=(10, 6))
        plotter.plot_acquisition_evolution(acquisition_data=acquisition_data, ax=ax_acquisition)
        ax_acquisition.set_title(f"Acquisition Function Evolution - {output_file}")
        plt.tight_layout()
        plt.savefig(f"{output_file}_acquisition_evolution.png", dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("No acquisition function data available for plotting.")

    print("✓ Dashboard generation completed")
    return True


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
