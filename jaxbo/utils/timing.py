"""
Timing integration for BOBE class to measure key phases.

This module provides proper timing measurement for:
- GP Training
- Acquisition optimization  
- True objective function evaluations
- Nested sampling
- MCMC sampling
"""

import time
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path


class BOBETimingCollector:
    """
    Comprehensive timing collector for BOBE phases.
    
    Measures timing for:
    - GP Training: Time spent fitting/refitting GP hyperparameters
    - Acquisition Optimization: Time spent optimizing acquisition function
    - True Objective Evaluations: Time spent evaluating the true likelihood
    - Nested Sampling: Time spent in nested sampling runs
    - MCMC Sampling: Time spent in MCMC sampling for GP posterior
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all timing data."""
        # Phase timings - cumulative times
        self.phase_times = {
            'GP Training': 0.0,
            'Acquisition Optimization': 0.0, 
            'True Objective Evaluations': 0.0,
            'Nested Sampling': 0.0,
            'MCMC Sampling': 0.0,
            'Total Runtime': 0.0
        }
        
        # Active timing tracking
        self.phase_start_times = {}
        
        # Detailed per-iteration timing
        self.iteration_times = {
            'iteration': [],
            'gp_training': [],
            'acquisition_opt': [],
            'objective_eval': [],
            'mcmc_sampling': [],
            'nested_sampling': []
        }
        
        # Counters for operations
        self.operation_counts = {
            'gp_fits': 0,
            'acquisition_optimizations': 0,
            'objective_evaluations': 0,
            'mcmc_runs': 0,
            'nested_sampling_runs': 0
        }
    
    def start_phase(self, phase_name: str):
        """Start timing a phase."""
        if phase_name not in self.phase_times:
            self.phase_times[phase_name] = 0.0
        self.phase_start_times[phase_name] = time.time()
    
    def end_phase(self, phase_name: str, iteration: Optional[int] = None):
        """End timing a phase and record the elapsed time."""
        if phase_name in self.phase_start_times:
            elapsed = time.time() - self.phase_start_times[phase_name]
            self.phase_times[phase_name] += elapsed
            
            # Record per-iteration timing if iteration provided
            if iteration is not None:
                if phase_name == 'GP Training':
                    self._record_iteration_time(iteration, 'gp_training', elapsed)
                    self.operation_counts['gp_fits'] += 1
                elif phase_name == 'Acquisition Optimization':
                    self._record_iteration_time(iteration, 'acquisition_opt', elapsed)
                    self.operation_counts['acquisition_optimizations'] += 1
                elif phase_name == 'True Objective Evaluations':
                    self._record_iteration_time(iteration, 'objective_eval', elapsed)
                    self.operation_counts['objective_evaluations'] += 1
                elif phase_name == 'MCMC Sampling':
                    self._record_iteration_time(iteration, 'mcmc_sampling', elapsed)
                    self.operation_counts['mcmc_runs'] += 1
                elif phase_name == 'Nested Sampling':
                    self._record_iteration_time(iteration, 'nested_sampling', elapsed)
                    self.operation_counts['nested_sampling_runs'] += 1
            
            del self.phase_start_times[phase_name]
    
    def _record_iteration_time(self, iteration: int, phase_key: str, elapsed_time: float):
        """Record timing for a specific iteration and phase."""
        # Ensure we have entries for this iteration
        while len(self.iteration_times['iteration']) < iteration:
            self.iteration_times['iteration'].append(len(self.iteration_times['iteration']) + 1)
            for key in ['gp_training', 'acquisition_opt', 'objective_eval', 'mcmc_sampling', 'nested_sampling']:
                self.iteration_times[key].append(0.0)
        
        # Add this iteration if not present
        if iteration not in self.iteration_times['iteration']:
            self.iteration_times['iteration'].append(iteration)
            for key in ['gp_training', 'acquisition_opt', 'objective_eval', 'mcmc_sampling', 'nested_sampling']:
                self.iteration_times[key].append(0.0)
        
        # Record the time
        idx = self.iteration_times['iteration'].index(iteration)
        self.iteration_times[phase_key][idx] += elapsed_time
    
    def get_phase_times(self) -> Dict[str, float]:
        """Get cumulative phase times."""
        return self.phase_times.copy()
    
    def get_iteration_times(self) -> Dict[str, List]:
        """Get per-iteration timing data."""
        return {k: v.copy() for k, v in self.iteration_times.items()}
    
    def get_operation_counts(self) -> Dict[str, int]:
        """Get operation counts."""
        return self.operation_counts.copy()
    
    def get_timing_summary(self) -> Dict[str, any]:
        """Get comprehensive timing summary."""
        total_time = self.phase_times.get('Total Runtime', 0.0)
        
        # Calculate percentages
        percentages = {}
        if total_time > 0:
            for phase, time_spent in self.phase_times.items():
                if phase != 'Total Runtime':
                    percentages[phase] = (time_spent / total_time) * 100
        
        # Average times per operation
        averages = {}
        for operation, count in self.operation_counts.items():
            if count > 0:
                phase_mapping = {
                    'gp_fits': 'GP Training',
                    'acquisition_optimizations': 'Acquisition Optimization',
                    'objective_evaluations': 'True Objective Evaluations',
                    'mcmc_runs': 'MCMC Sampling',
                    'nested_sampling_runs': 'Nested Sampling'
                }
                phase_name = phase_mapping.get(operation)
                if phase_name and phase_name in self.phase_times:
                    averages[f'avg_{operation}'] = self.phase_times[phase_name] / count
        
        return {
            'phase_times': self.phase_times,
            'percentages': percentages,
            'operation_counts': self.operation_counts,
            'average_times': averages,
            'total_runtime': total_time
        }
    
    def save_timing_data(self, output_file: str):
        """Save timing data to JSON file."""
        import json
        
        timing_data = {
            'phase_times': self.get_phase_times(),
            'iteration_times': self.get_iteration_times(),
            'operation_counts': self.get_operation_counts(),
            'summary': self.get_timing_summary()
        }
        
        output_path = f"{output_file}_timing_data.json"
        with open(output_path, 'w') as f:
            json.dump(timing_data, f, indent=2)
        
        print(f"Saved timing data to {output_path}")
    
    def print_timing_summary(self):
        """Print a formatted timing summary."""
        summary = self.get_timing_summary()
        
        print("\n" + "="*60)
        print("BOBE TIMING SUMMARY")
        print("="*60)
        
        print(f"\nTotal Runtime: {summary['total_runtime']:.2f} seconds")
        print(f"Total Runtime: {summary['total_runtime']/60:.2f} minutes")
        
        print(f"\nPhase Breakdown:")
        print("-" * 40)
        for phase, time_spent in summary['phase_times'].items():
            if phase != 'Total Runtime' and time_spent > 0:
                percentage = summary['percentages'].get(phase, 0)
                print(f"{phase:25s}: {time_spent:8.2f}s ({percentage:5.1f}%)")
        
        print(f"\nOperation Counts:")
        print("-" * 40)
        for operation, count in summary['operation_counts'].items():
            if count > 0:
                avg_key = f'avg_{operation}'
                avg_time = summary['average_times'].get(avg_key, 0)
                print(f"{operation:25s}: {count:5d} ops, {avg_time:6.3f}s avg")
        
        print("="*60)


def integrate_timing_with_bobe():
    """
    Documentation showing where to integrate timing in BOBE code.
    """
    
    integration_guide = """
    BOBE Timing Integration Points:
    
    1. BOBE.__init__():
       - self.timing_collector = BOBETimingCollector()
    
    2. BOBE.run() method - start of method:
       - self.timing_collector.start_phase('Total Runtime')
    
    3. GP Training (in BOBE.run() when refit=True):
       - self.timing_collector.start_phase('GP Training')
       - self.gp.update(new_pt_u, new_val, refit=True, ...)  # or self.gp.fit()
       - self.timing_collector.end_phase('GP Training', iteration=ii)
    
    4. Acquisition Optimization (in BOBE.run()):
       - self.timing_collector.start_phase('Acquisition Optimization')
       - new_pt_u, acq_val = optimize(WIPV, ...)
       - self.timing_collector.end_phase('Acquisition Optimization', iteration=ii)
    
    5. True Objective Function Evaluation:
       - self.timing_collector.start_phase('True Objective Evaluations')
       - new_val = self.loglikelihood(new_pt, ...)
       - self.timing_collector.end_phase('True Objective Evaluations', iteration=ii)
    
    6. MCMC Sampling (when update_mc=True):
       - self.timing_collector.start_phase('MCMC Sampling') 
       - self.mc_samples = get_mc_samples(self.gp, ...)
       - self.timing_collector.end_phase('MCMC Sampling', iteration=ii)
    
    7. Nested Sampling (when ns_flag=True):
       - self.timing_collector.start_phase('Nested Sampling')
       - ns_samples, logz_dict, ns_success = nested_sampling_Dy(...)
       - self.timing_collector.end_phase('Nested Sampling', iteration=ii)
    
    8. End of BOBE.run():
       - self.timing_collector.end_phase('Total Runtime')
       - self.timing_collector.save_timing_data(self.output_file)
       - self.timing_collector.print_timing_summary()
    
    9. For plotting integration:
       - timing_data = self.timing_collector.get_phase_times()
       - plotter = BOBESummaryPlotter(self.results_manager)
       - plotter.create_summary_dashboard(timing_data=timing_data, ...)
    """
    
    print(integration_guide)


if __name__ == "__main__":
    # Demonstrate the timing collector
    collector = BOBETimingCollector()
    
    # Simulate some timing
    collector.start_phase('Total Runtime')
    time.sleep(0.1)
    
    for i in range(3):
        collector.start_phase('GP Training')
        time.sleep(0.02)
        collector.end_phase('GP Training', iteration=i+1)
        
        collector.start_phase('Acquisition Optimization')
        time.sleep(0.01)
        collector.end_phase('Acquisition Optimization', iteration=i+1)
        
        collector.start_phase('True Objective Evaluations')
        time.sleep(0.005)
        collector.end_phase('True Objective Evaluations', iteration=i+1)
    
    collector.end_phase('Total Runtime')
    
    # Show summary
    collector.print_timing_summary()
    
    # Show integration guide
    integrate_timing_with_bobe()
