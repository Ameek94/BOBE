import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    os.cpu_count()
)
from jaxbo.bo import BOBE
from jaxbo.utils import plot_final_samples
from jaxbo.loglike import CobayaLikelihood
from jaxbo.summary_plots import BOBESummaryPlotter
import time
import sys
import matplotlib.pyplot as plt

# Configure matplotlib for better LaTeX rendering
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# Get classifier type from command line argument
clf = sys.argv[1] if len(sys.argv) > 1 else 'svm'

# Set up the cosmological likelihood
cobaya_input_file = './cosmo_input/LCDM_Planck_DESI_CPL.yaml'

likelihood = CobayaLikelihood(cobaya_input_file, confidence_for_unbounded=0.9999995,
        minus_inf=-1e5, noise_std=0.0, name=f'Planck_Camspec_CPL_pNUTS_{clf}')

# Print detailed setup information
print("="*60)
print(f"PLANCK CAMSPEC CPL ANALYSIS - {clf.upper()} CLASSIFIER")
print("="*60)
print(f"Likelihood: {likelihood.name}")
print(f"Parameters: {likelihood.param_list}")
print(f"Dimensions: {len(likelihood.param_list)}")
print(f"Classifier type: {clf}")
print("="*60)

# Set classifier-specific parameters
if clf == 'svm':
    clf_update_step = 1
else:
    clf_update_step = 2

# Record start time
start = time.time()

# Initialize BOBE sampler with comprehensive configuration
sampler = BOBE(
    n_cobaya_init=32, 
    n_sobol_init=32, 
    miniters=750, 
    maxiters=4000,
    max_gp_size=1800,
    loglikelihood=likelihood,
    resume=False,
    resume_file=f'{likelihood.name}.npz',
    save=True,
    fit_step=25, 
    update_mc_step=5, 
    ns_step=50,
    num_hmc_warmup=512,
    num_hmc_samples=4096, 
    mc_points_size=128,
    lengthscale_priors='DSLP',
    logz_threshold=5.,
    clf_threshold=400,
    gp_threshold=500,
    use_clf=True,
    clf_type=clf,
    clf_use_size=50,
    clf_update_step=clf_update_step,
    minus_inf=-1e5,
    seed=42,  # For reproducibility
)

# Run BOBE with automatic timing collection
print("Starting BOBE run with automatic timing measurement...")
results = sampler.run()

# Record end time
end = time.time()
manual_timing = end - start

print("\n" + "="*60)
print("RUN COMPLETED")
print("="*60)
print(f"Manual timing: {manual_timing:.2f} seconds ({manual_timing/60:.2f} minutes)")

# Extract components for backward compatibility and analysis
gp = results['gp']
samples = results['samples']
logz_dict = results.get('logz', {})
comprehensive_results = results['comprehensive']
timing_data = comprehensive_results['timing']

# Print detailed timing analysis
print("\n" + "="*60)
print("DETAILED TIMING ANALYSIS")
print("="*60)

print(f"Automatic timing: {timing_data['total_runtime']:.2f} seconds ({timing_data['total_runtime']/60:.2f} minutes)")
print(f"Timing difference: {abs(manual_timing - timing_data['total_runtime']):.2f} seconds")

print("\nPhase Breakdown:")
print("-" * 40)
for phase, time_spent in timing_data['phase_times'].items():
    if time_spent > 0:
        percentage = timing_data['percentages'].get(phase, 0)
        print(f"{phase:25s}: {time_spent:8.2f}s ({percentage:5.1f}%)")

# Analyze timing efficiency
print("\nTiming Efficiency Analysis:")
print("-" * 40)
total_measured = sum(t for t in timing_data['phase_times'].values() if t > 0)
overhead = timing_data['total_runtime'] - total_measured
overhead_pct = (overhead / timing_data['total_runtime']) * 100

print(f"Total measured phases: {total_measured:.2f}s ({(total_measured/timing_data['total_runtime']*100):.1f}%)")
print(f"Overhead/unmeasured: {overhead:.2f}s ({overhead_pct:.1f}%)")

# Find dominant phase
max_phase = max(timing_data['phase_times'].items(), key=lambda x: x[1])
print(f"Dominant phase: {max_phase[0]} ({timing_data['percentages'][max_phase[0]]:.1f}%)")

# Print convergence info
print("\n" + "="*60)
print("CONVERGENCE ANALYSIS")
print("="*60)
print(f"Converged: {comprehensive_results['converged']}")
print(f"Termination reason: {comprehensive_results['termination_reason']}")
print(f"Final GP size: {gp.train_x.shape[0]}")

if logz_dict:
    print(f"Final LogZ: {logz_dict.get('mean', 'N/A'):.4f}")
    if 'upper' in logz_dict and 'lower' in logz_dict:
        print(f"LogZ uncertainty: ±{(logz_dict['upper'] - logz_dict['lower'])/2:.4f}")

# Create comprehensive plots
print("\n" + "="*60)
print("GENERATING PLOTS")
print("="*60)

# Initialize plotter
plotter = BOBESummaryPlotter(results['results_manager'])

# Get GP and best loglike evolution data
gp_data = results['results_manager'].get_gp_data()
best_loglike_data = results['results_manager'].get_best_loglike_data()

# Create summary dashboard with timing data
print("Creating summary dashboard...")
fig_dashboard = plotter.create_summary_dashboard(
    gp_data=gp_data,
    best_loglike_data=best_loglike_data,
    timing_data=timing_data,
    save_path=f"{likelihood.name}_dashboard.png"
)
plt.close(fig_dashboard)

# Create individual timing plot
print("Creating detailed timing plot...")
fig_timing, ax_timing = plt.subplots(1, 1, figsize=(10, 6))
plotter.plot_timing_breakdown(timing_data=timing_data, ax=ax_timing)
ax_timing.set_title(f"Timing Breakdown - {likelihood.name}")
plt.tight_layout()
plt.savefig(f"{likelihood.name}_timing_detailed.png", dpi=300, bbox_inches='tight')
plt.close(fig_timing)

# Create evidence evolution plot if available
if comprehensive_results.get('logz_history'):
    print("Creating evidence evolution plot...")
    fig_evidence, ax_evidence = plt.subplots(1, 1, figsize=(10, 6))
    plotter.plot_evidence_evolution(ax=ax_evidence)
    ax_evidence.set_title(f"Evidence Evolution - {likelihood.name}")
    plt.tight_layout()
    plt.savefig(f"{likelihood.name}_evidence.png", dpi=300, bbox_inches='tight')
    plt.close(fig_evidence)

# Create parameter samples plot
print("Creating parameter samples plot...")
if hasattr(samples, 'samples'):  # GetDist samples
    sample_array = samples.samples
    weights_array = samples.weights
else:  # Dictionary format
    sample_array = samples['x']
    weights_array = samples['weights']

# Define specific parameters for CPL cosmology
plot_params = ['w','wa','omch2','ombh2','logA','ns','H0','tau'] #'omk'

plot_final_samples(
    gp, 
    samples,
    param_list=likelihood.param_list,
    param_bounds=likelihood.param_bounds,
    plot_params=plot_params,
    param_labels=likelihood.param_labels,
    output_file=likelihood.name,
    reference_file='./cosmo_input/chains/Planck_DESI_LCDM_CPL_pchord_flat',
    reference_ignore_rows=0.,
    reference_label='Polychord',
    scatter_points=False
)

# Save comprehensive results
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Results are automatically saved by BOBE, but let's summarize what was saved
print(f"✓ Main results: {likelihood.name}_results.npz")
print(f"✓ Timing data: {likelihood.name}_timing.json")
print(f"✓ Legacy samples: {likelihood.name}_samples.npz")
print(f"✓ Summary dashboard: {likelihood.name}_dashboard.png")
print(f"✓ Detailed timing: {likelihood.name}_timing_detailed.png")
if comprehensive_results.get('logz_history'):
    print(f"✓ Evidence evolution: {likelihood.name}_evidence.png")
print(f"✓ Parameter samples: {likelihood.name}_samples.pdf")

# Performance comparison with previous runs (if applicable)
print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)

current_time = timing_data['total_runtime']
print(f"Current run ({clf}): {current_time:.2f} seconds ({current_time/60:.2f} minutes)")

# Compare with different classifier types if multiple runs exist
classifier_times = {
    "ellipsoid": None,  # These would be filled from previous runs
    "svm": None,
    "nn": None
}

# If we have timing data from previous runs, we could compare
print(f"Classifier efficiency for {clf} configuration")
print(f"Total runtime: {current_time:.2f}s")
if 'clf_phase' in timing_data['phase_times']:
    clf_time = timing_data['phase_times']['clf_phase']
    clf_efficiency = (clf_time / current_time) * 100
    print(f"Classification overhead: {clf_time:.2f}s ({clf_efficiency:.1f}%)")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("Check the generated plots and saved files for detailed analysis.")
print(f"Final LogZ comparison with reference:")
print(f"BOBE result: LogZ = {logz_dict.get('mean', 'N/A'):.4f}")
print("PolyChord reference: LogZ = -6231")

# Calculate deviation from reference if we have a result
if 'mean' in logz_dict:
    reference_logz = -6231
    deviation = abs(logz_dict['mean'] - reference_logz)
    print(f"Deviation from reference: {deviation:.4f}")
    
    if 'upper' in logz_dict and 'lower' in logz_dict:
        uncertainty = (logz_dict['upper'] - logz_dict['lower']) / 2
        sigma_deviation = deviation / uncertainty if uncertainty > 0 else float('inf')
        print(f"Deviation in σ: {sigma_deviation:.2f}")
