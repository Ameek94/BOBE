import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    os.cpu_count()
)
from jaxbo.bo import BOBE
from jaxbo.utils.summary_plots import plot_final_samples, BOBESummaryPlotter
from jaxbo.loglike import CobayaLikelihood
import time
import matplotlib.pyplot as plt

# Set up the cosmological likelihood
cobaya_input_file = './cosmo_input/LCDM_6D.yaml'

likelihood = CobayaLikelihood(cobaya_input_file, confidence_for_unbounded=0.9999995,
        minus_inf=-1e5, noise_std=0.0, name='Planck_lite_clf_timing_test_logEI_WIPV_logstd')

print("="*60)
print("PLANCK LITE CLF TIMING TEST")
print("="*60)
print(f"Likelihood: {likelihood.name}")
print(f"Parameters: {likelihood.param_list}")
print(f"Dimensions: {len(likelihood.param_list)}")
print("="*60)

start = time.time()
sampler = BOBE(
    n_cobaya_init=2, 
    n_sobol_init=4, 
    min_iters=25, 
    max_eval_budget=200,
    max_gp_size=200,
    loglikelihood=likelihood,
    fit_step=5, 
    update_mc_step=5, 
    ns_step=10,
    num_hmc_warmup=512,
    num_hmc_samples=512, 
    mc_points_size=64,
    lengthscale_priors='DSLP', 
    use_clf=True,
    clf_use_size=10,
    clf_threshold=350,
    clf_update_step=1,  # SVM update step
    clf_type='svm',  # Using SVM for classification
    minus_inf=-1e5,
    logz_threshold=0.1,
    seed=10,  # For reproducibility
    do_final_ns=False,
)

# Run BOBE with automatic timing collection
print("Starting BOBE run with automatic timing measurement...")
results = sampler.run(n_log_ei_iters=30)

end = time.time()
manual_timing = end - start

print("\n" + "="*60)
print("RUN COMPLETED")
print("="*60)
print(f"Manual timing: {manual_timing:.2f} seconds ({manual_timing/60:.2f} minutes)")

# Extract components for backward compatibility
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
acquisition_data = results['results_manager'].get_acquisition_data()

# Create summary dashboard with timing data
print("Creating summary dashboard...")
fig_dashboard = plotter.create_summary_dashboard(
    gp_data=gp_data,
    acquisition_data=acquisition_data,
    best_loglike_data=best_loglike_data,
    timing_data=timing_data,
    save_path=f"{likelihood.name}_dashboard.png"
)
plt.show()

# Create individual timing plot
print("Creating detailed timing plot...")
fig_timing, ax_timing = plt.subplots(1, 1, figsize=(10, 6))
plotter.plot_timing_breakdown(timing_data=timing_data, ax=ax_timing)
ax_timing.set_title(f"Timing Breakdown - {likelihood.name}")
plt.tight_layout()
plt.savefig(f"{likelihood.name}_timing_detailed.png", dpi=300, bbox_inches='tight')
plt.show()

# Create evidence evolution plot if available
if comprehensive_results.get('logz_history'):
    print("Creating evidence evolution plot...")
    fig_evidence, ax_evidence = plt.subplots(1, 1, figsize=(10, 6))
    plotter.plot_evidence_evolution(ax=ax_evidence)
    ax_evidence.set_title(f"Evidence Evolution - {likelihood.name}")
    plt.tight_layout()
    plt.savefig(f"{likelihood.name}_evidence.png", dpi=300, bbox_inches='tight')
    plt.show()

# Create acquisition function evolution plot
print("Creating acquisition function evolution plot...")
acquisition_data = results['results_manager'].get_acquisition_data()
if acquisition_data and acquisition_data.get('iterations'):
    fig_acquisition, ax_acquisition = plt.subplots(1, 1, figsize=(10, 6))
    plotter.plot_acquisition_evolution(acquisition_data=acquisition_data, ax=ax_acquisition)
    ax_acquisition.set_title(f"Acquisition Function Evolution - {likelihood.name}")
    plt.tight_layout()
    plt.savefig(f"{likelihood.name}_acquisition_evolution.png", dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No acquisition function data available for plotting.")

# Create parameter samples plot
print("Creating parameter samples plot...")
if hasattr(samples, 'samples'):  # GetDist samples
    sample_array = samples.samples
    weights_array = samples.weights
else:  # Dictionary format
    sample_array = samples['x']
    weights_array = samples['weights']

plot_final_samples(
    gp, 
    {'x': sample_array, 'weights': weights_array, 'logl': samples.get('logl', [])},
    param_list=likelihood.param_list,
    param_bounds=likelihood.param_bounds,
    param_labels=likelihood.param_labels,
    output_file=likelihood.name,
    reference_file='./cosmo_input/chains/Planck_lite_LCDM',
    reference_ignore_rows=0.3,
    reference_label='MCMC',
    scatter_points=False
)

# Save comprehensive results
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Results are automatically saved by BOBE, but let's summarize what was saved
print(f"✓ Main results: {likelihood.name}_results.pkl")
print(f"✓ Timing data: {likelihood.name}_timing.json")
print(f"✓ Legacy samples: {likelihood.name}_samples.npz")
print(f"✓ Summary dashboard: {likelihood.name}_dashboard.png")
print(f"✓ Detailed timing: {likelihood.name}_timing_detailed.png")
print(f"✓ Evidence evolution: {likelihood.name}_evidence.png")
print(f"✓ Acquisition evolution: {likelihood.name}_acquisition_evolution.png")
print(f"✓ Parameter samples: {likelihood.name}_samples.pdf")

# Create timing comparison with previous runs (if comments in original file are accurate)
print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)

# Previous timing from comments in original file
previous_times = {
    "Fast updates": 292.68,
    "Older code": 387.66
}

current_time = timing_data['total_runtime']
print(f"Current run: {current_time:.2f} seconds")

for version, prev_time in previous_times.items():
    speedup = prev_time / current_time
    improvement = ((prev_time - current_time) / prev_time) * 100
    print(f"vs {version}: {speedup:.2f}x speedup ({improvement:+.1f}%)")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("Check the generated plots and saved files for detailed analysis.")
