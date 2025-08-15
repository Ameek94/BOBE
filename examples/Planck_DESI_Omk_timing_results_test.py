import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    os.cpu_count()
)
from jaxbo.bo import BOBE
from jaxbo.utils.summary_plots import plot_final_samples, BOBESummaryPlotter
from jaxbo.loglike import CobayaLikelihood
import matplotlib.pyplot as plt
import time
import sys

cobaya_input_file = './cosmo_input/LCDM_Planck_DESI_Omk_PPlus.yaml'

clf_type = str(sys.argv[1]) if len(sys.argv) > 1 else 'svm'
clf_update_step = 1 if clf_type == 'svm' else 5

likelihood = CobayaLikelihood(cobaya_input_file, confidence_for_unbounded=0.9999995,
        minus_inf=-1e5, noise_std=0.0,name=f'Planck_DESI_Omk_{clf_type}_mix_acq')


print("="*60)
print("PLANCK DESI OMK CLF TEST")
print("="*60)
print(f"Likelihood: {likelihood.name}")
print(f"Parameters: {likelihood.param_list}")
print(f"Dimensions: {len(likelihood.param_list)}")
print(f"Classifier type: {clf_type}")
print("="*60)



start = time.time()
sampler = BOBE(n_cobaya_init=16, n_sobol_init=32,
        miniters=500, maxiters=2500, max_gp_size=1800,
        loglikelihood=likelihood,
        resume=False,
        resume_file=f'{likelihood.name}.npz',
        save=True,
        fit_step=50, update_mc_step=5, ns_step=50,
        num_hmc_warmup=512, num_hmc_samples=4000, mc_points_size=200,
        lengthscale_priors='DSLP',
        use_clf=True, clf_type=clf_type, clf_use_size=50, clf_update_step=clf_update_step,
        clf_threshold=300, gp_threshold=500,
        minus_inf=-1e5, logz_threshold=1.)

# Run BOBE with automatic timing collection
print("Starting BOBE run with automatic timing measurement...")
results = sampler.run()

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
    save_path=f"{likelihood.name}_dashboard.png"
)
# plt.show()

# Create individual timing plot
print("Creating detailed timing plot...")
fig_timing, ax_timing = plt.subplots(1, 1, figsize=(10, 6))
plotter.plot_timing_breakdown(timing_data=timing_data, ax=ax_timing)
ax_timing.set_title(f"Timing Breakdown - {likelihood.name}")
plt.tight_layout()
plt.savefig(f"{likelihood.name}_timing_detailed.png", dpi=300, bbox_inches='tight')
# plt.show()

# Create evidence evolution plot if available
if comprehensive_results.get('logz_history'):
    print("Creating evidence evolution plot...")
    fig_evidence, ax_evidence = plt.subplots(1, 1, figsize=(10, 6))
    plotter.plot_evidence_evolution(ax=ax_evidence)
    ax_evidence.set_title(f"Evidence Evolution - {likelihood.name}")
    plt.tight_layout()
    plt.savefig(f"{likelihood.name}_evidence.png", dpi=300, bbox_inches='tight')
    # plt.show()

# Create parameter samples plot
print("Creating parameter samples plot...")
if hasattr(samples, 'samples'):  # GetDist samples
    sample_array = samples.samples
    weights_array = samples.weights
else:  # Dictionary format
    sample_array = samples['x']
    weights_array = samples['weights']



param_list_LCDM = ['omk','omch2','ombh2','logA','ns','H0','tau']
plot_final_samples(
    gp, 
    {'x': sample_array, 'weights': weights_array, 'logl': samples.get('logl', [])},
    param_list=likelihood.param_list,
    param_bounds=likelihood.param_bounds,
    plot_params=param_list_LCDM,
    param_labels=likelihood.param_labels,
    output_file=likelihood.name,
    reference_file='./cosmo_input/chains/PPlus_curved_LCDM',
    reference_ignore_rows=0.3,
    reference_label='MCMC',
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
print(f"✓ Summary dashboard: {likelihood.name}_dashboard.png")
print(f"✓ Detailed timing: {likelihood.name}_timing_detailed.png")
if comprehensive_results.get('logz_history'):
    print(f"✓ Evidence evolution: {likelihood.name}_evidence.png")
print(f"✓ Parameter samples: {likelihood.name}_samples.pdf")

print("\n" + "="*60)
print("HISTORICAL RESULTS COMPARISON")
print("="*60)
print("Previous results from comments in original file:")
print("- LogZ: -5529.2378 ± 0.2430 (Dynesty)")
print("- PolyChord: -5529.65218118231 ± 0.447056743748251")
print("- Total time: ~10917 seconds (~3.0 hours)")
print("- Final GP size: 1360")
print("- Iterations: 1350")
print("\nParameter constraints (GP vs MCMC):")
print("- Ωₖ: 0.0023±0.0014 vs 0.0022⁺⁰·⁰⁰¹⁵₋₀.₀₀₁₃")
print("- Ωc h²: 0.1199⁺⁰·⁰⁰¹¹₋₀.₀₀₀₉₉ vs 0.1194⁺⁰·⁰⁰¹¹₋₀.₀₀₁₀")
print("- log(10¹⁰ As): 3.039⁺⁰·⁰¹⁵₋₀.₀₁₇ vs 3.044±0.014")
print("- ns: 0.9626±0.0041 vs 0.9639±0.0040")
print("- H₀: 68.32⁺⁰·⁴⁸₋₀.₅₅ vs 68.45±0.49")
print("- Ωb h²: 0.02217±0.00014 vs 0.02220±0.00013")
print("- τ: 0.0533⁺⁰·⁰⁰⁷³₋₀.₀₀₈₃ vs 0.0558±0.0070")
print("="*60)


# 2025-04-21 18:27:42,039 INFO:[BO]:  Final LogZ: upper=-5527.4084, mean=-5529.4980, lower=-5530.0967, dlogz sampler=0.1720
# PolyChord result: # log-evidence
# logZ: -5529.65218118231
# logZstd: 0.447056743748251

# INFO:[NS]: Nested Sampling took 289.45s
# INFO:[NS]: Log Z evaluated using (30859,) points
# INFO:[NS]: Dynesty made 1084416 function calls, max value of logl = -5488.9169
# 2025-08-06 17:34:02,615 INFO:[BO]:  LogZ info: mean=-5529.2378, dlogz sampler=0.2430, upper=-5513.5731, lower=-5529.9985
# 2025-08-06 17:34:02,616 INFO:[BO]:  Convergence check: delta = 0.7608, step = 1349
# 2025-08-06 17:34:02,616 INFO:[BO]:  Converged
# 2025-08-06 17:34:02,621 INFO:[BO]:  Sampling stopped: LogZ converged
# 2025-08-06 17:34:02,621 INFO:[BO]:  Final GP training set size: 1360, max size: 1800
# 2025-08-06 17:34:02,621 INFO:[BO]:  Number of iterations: 1350, max iterations: 2000
# 2025-08-06 17:34:02,621 INFO:[BO]: Using nested sampling results
# Total time taken = 10917.0748 seconds
# INFO:jaxbo.utils:Parameter limits from GP
# INFO:jaxbo.utils:\Omega_K = 0.0023\pm 0.0014
# INFO:jaxbo.utils:\Omega_\mathrm{c} h^2 = 0.1199^{+0.0011}_{-0.00099}
# INFO:jaxbo.utils:\log(10^{10} A_\mathrm{s}) = 3.039^{+0.015}_{-0.017}
# INFO:jaxbo.utils:n_\mathrm{s} = 0.9626\pm 0.0041
# INFO:jaxbo.utils:H_0 = 68.32^{+0.48}_{-0.55}
# INFO:jaxbo.utils:\Omega_\mathrm{b} h^2 = 0.02217\pm 0.00014
# INFO:jaxbo.utils:\tau_\mathrm{reio} = 0.0533^{+0.0073}_{-0.0083}
# INFO:jaxbo.utils:Parameter limits from MCMC
# INFO:jaxbo.utils:\Omega_K = 0.0022^{+0.0015}_{-0.0013}
# INFO:jaxbo.utils:\Omega_\mathrm{c} h^2 = 0.1194^{+0.0011}_{-0.0010}
# INFO:jaxbo.utils:\log(10^{10} A_\mathrm{s}) = 3.044\pm 0.014
# INFO:jaxbo.utils:n_\mathrm{s} = 0.9639\pm 0.0040
# INFO:jaxbo.utils:H_0 = 68.45\pm 0.49
# INFO:jaxbo.utils:\Omega_\mathrm{b} h^2 = 0.02220\pm 0.00013
# INFO:jaxbo.utils:\tau_\mathrm{reio} = 0.0558\pm 0.0070
