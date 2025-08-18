from jaxbo.bo import BOBE
from jaxbo.loglike import ExternalLikelihood
from jaxbo.utils.summary_plots import plot_final_samples, BOBESummaryPlotter 
import matplotlib.pyplot as plt
import time
import sys
from jaxbo.nested_sampler import renormalise_log_weights
from getdist import MCSamples
from dynesty import DynamicNestedSampler
import numpy as np
import time

ndim = 2
param_list = ['x1','x2']
param_labels = ['x_1','x_2']
param_bounds = np.array([[-1,1],[-1,2]]).T

ndim = 2
param_bounds = np.array([[-4,4],[-4,4]]).T
param_list = ['x1','x2']
param_labels = ['x_1','x_2']

afac= 0.1

def prior_transform(x):
    return 8*x - 4

def loglike(X):
    r1 = (X[0] + X[1]**2 -7)**2
    r2 = (X[0]**2 + X[1]-11)**2
    return -0.5*(afac*r1 + r2)


likelihood = ExternalLikelihood(loglikelihood=loglike,ndim=ndim,param_list=param_list,
        param_bounds=param_bounds,param_labels=param_labels,
        name='Himmelblau',noise_std=0.0,minus_inf=-1e5)
start = time.time()
sampler = BOBE(n_cobaya_init=4, n_sobol_init = 8, 
        min_iters=10, max_eval_budget=250,max_gp_size=250,
        loglikelihood=likelihood,
        fit_step = 2, update_mc_step = 2, ns_step = 10,
        num_hmc_warmup = 256,num_hmc_samples = 1024, mc_points_size = 64,
        logz_threshold=0.01,resume=False,
        lengthscale_priors='DSLP', use_clf=False,minus_inf=-1e5,)

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
    plt.savefig(f"{likelihood.name}_acquisition_evolution.pdf",  bbox_inches='tight')
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



dns_sampler =  DynamicNestedSampler(loglike,prior_transform,ndim=ndim,
                                       sample='rwalk')

dns_sampler.run_nested(print_progress=True,dlogz_init=0.01) 
res = dns_sampler.results  
mean = res['logz'][-1]
logz_err = res['logzerr'][-1]
print(f"Mean logz from dynesty = {mean:.4f} +/- {logz_err:.4f}")

samples = res['samples']
weights = renormalise_log_weights(res['logwt'])

reference_samples = MCSamples(samples=samples, names=param_list, labels=param_labels,
                            weights=weights, 
                            ranges= dict(zip(param_list,param_bounds.T)))


plot_final_samples(gp,{'x': sample_array, 'weights': weights_array},
                   param_list=likelihood.param_list,param_bounds=likelihood.param_bounds,
                   param_labels=likelihood.param_labels,output_file=likelihood.name,reference_samples=reference_samples,
                   reference_file=None,scatter_points=True,reference_label='Dynesty')


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
