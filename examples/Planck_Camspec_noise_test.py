import sys
clf = str(sys.argv[1]) if len(sys.argv) > 1 else 'svm'
clf_update_step = 1 if clf == 'svm' else 5
num_devices = int(sys.argv[2]) if len(sys.argv) > 2 else 8
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(num_devices)
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
#     os.cpu_count()
# )
from jaxbo.bo import BOBE
from jaxbo.utils.summary_plots import plot_final_samples,BOBESummaryPlotter 
from jaxbo.loglike import CobayaLikelihood
import matplotlib.pyplot as plt
import time

cobaya_input_file = './cosmo_input/LCDM_Planck_DESI.yaml'

log10_noise_level = float(sys.argv[3]) if len(sys.argv) > 3 else -8.0


likelihood = CobayaLikelihood(cobaya_input_file, confidence_for_unbounded=0.9999995,
        minus_inf=-1e5, noise_std=0.0,name=f'Camspec_{clf}_mcacq_noise_{-log10_noise_level}')


print("="*60)
print("PLANCK CAMSPEC CLF TEST")
print("="*60)
print(f"Likelihood: {likelihood.name}")
print(f"Parameters: {likelihood.param_list}")
print(f"Dimensions: {len(likelihood.param_list)}")
print("="*60)



start = time.time()
sampler = BOBE(n_cobaya_init=16, n_sobol_init=16,
        min_iters=750, max_eval_budget=2500, max_gp_size=1500,
        loglikelihood=likelihood,
        resume=False,
        resume_file=f'{likelihood.name}',
        save=True,
        fit_step=50, update_mc_step=5, ns_step=25,
        noise = 10**log10_noise_level,
        num_hmc_warmup=512, num_hmc_samples=4096, mc_points_size=300,
        lengthscale_priors='DSLP',
        use_clf=True, clf_type=clf, clf_use_size=30, clf_update_step=clf_update_step,
        clf_threshold=300, gp_threshold=400,
        minus_inf=-1e5, logz_threshold=1., seed=100)

# Run BOBE with automatic timing collection
print("Starting BOBE run with automatic timing measurement...")
results = sampler.run(n_log_ei_iters=250)

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

# Create summary dashboard with timing data
print("Creating summary dashboard...")
fig_dashboard = plotter.create_summary_dashboard(
    gp_data=gp_data,
    best_loglike_data=best_loglike_data,
    timing_data=timing_data,
    save_path=f"{likelihood.name}_dashboard.pdf"
)
# plt.show()

# Create individual timing plot
print("Creating detailed timing plot...")
fig_timing, ax_timing = plt.subplots(1, 1, figsize=(10, 6))
plotter.plot_timing_breakdown(timing_data=timing_data, ax=ax_timing)
ax_timing.set_title(f"Timing Breakdown - {likelihood.name}")
plt.tight_layout()
plt.savefig(f"{likelihood.name}_timing_detailed.pdf", bbox_inches='tight')
# plt.show()

# Create evidence evolution plot if available
if comprehensive_results.get('logz_history'):
    print("Creating evidence evolution plot...")
    fig_evidence, ax_evidence = plt.subplots(1, 1, figsize=(10, 6))
    plotter.plot_evidence_evolution(ax=ax_evidence)
    ax_evidence.set_title(f"Evidence Evolution - {likelihood.name}")
    plt.tight_layout()
    plt.savefig(f"{likelihood.name}_evidence.pdf", bbox_inches='tight')
    # plt.show()

# Create acquisition function evolution plot
print("Creating acquisition function evolution plot...")
acquisition_data = results['results_manager'].get_acquisition_data()
if acquisition_data and acquisition_data.get('iterations'):
    fig_acquisition, ax_acquisition = plt.subplots(1, 1, figsize=(10, 6))
    plotter.plot_acquisition_evolution(acquisition_data=acquisition_data, ax=ax_acquisition)
    ax_acquisition.set_title(f"Acquisition Function Evolution - {likelihood.name}")
    plt.tight_layout()
    plt.savefig(f"{likelihood.name}_acquisition_evolution.pdf", bbox_inches='tight')
    # plt.show()
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



param_list_LCDM = ['omch2','ombh2','logA','ns','H0','tau']
plot_final_samples(
    gp, 
    {'x': sample_array, 'weights': weights_array, 'logl': samples.get('logl', [])},
    plot_params=param_list_LCDM,
    param_list=likelihood.param_list,
    param_bounds=likelihood.param_bounds,
    param_labels=likelihood.param_labels,
    output_file=f"{likelihood.name}_cosmo",
    reference_file='./cosmo_input/chains/Planck_DESI_LCDM_pchord',
    reference_ignore_rows=0.0,
    reference_label='PolyChord',
    scatter_points=False,
)

plot_final_samples(
    gp, 
    {'x': sample_array, 'weights': weights_array, 'logl': samples.get('logl', [])},
    param_list=likelihood.param_list,
    param_bounds=likelihood.param_bounds,
    param_labels=likelihood.param_labels,
    output_file=f"{likelihood.name}_full",
    reference_file='./cosmo_input/chains/Planck_DESI_LCDM_pchord',
    reference_ignore_rows=0.0,
    reference_label='PolyChord',
    scatter_points=False,
)

# Save comprehensive results
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Results are automatically saved by BOBE, but let's summarize what was saved
print(f"✓ Main results: {likelihood.name}_results.pkl")
print(f"✓ Timing data: {likelihood.name}_timing.json")
print(f"✓ Legacy samples: {likelihood.name}_samples.npz")
print(f"✓ Summary dashboard: {likelihood.name}_dashboard.pdf")
print(f"✓ Detailed timing: {likelihood.name}_timing_detailed.pdf")
print(f"✓ Evidence evolution: {likelihood.name}_evidence.pdf")
print(f"✓ Parameter samples: {likelihood.name}_samples.pdf")


# 2025-04-21 18:27:42,039 INFO:[BO]:  Final LogZ: upper=-5527.4084, mean=-5529.4980, lower=-5530.0967, dlogz sampler=0.1720
# PolyChord result: # log-evidence
# logZ: -5529.65218118231
# logZstd: 0.447056743748251

# 2025-07-26 05:39:58,814 INFO:[BO]:  Final LogZ: mean=-5529.6915, dlogz sampler=0.1793, upper=-5509.8101, lower=-5530.1042
# Total time taken = 9495.8532 seconds

# Ellipsoid classifier used for Planck Camspec data
# INFO:[bo]: LogZ info: mean=-5529.5996, dlogz sampler=0.2407, upper=-5529.2020, lower=-5529.9512
# INFO:[bo]: Convergence check: delta = 0.7493, step = 350
# INFO:[bo]: Converged
# INFO:[bo]: Sampling stopped: LogZ converged
# INFO:[bo]: Final GP training set size: 1251, max size: 1500
# INFO:[bo]: Number of iterations: 350, max iterations: 2500
