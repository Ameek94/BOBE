"""
Example of integrating runtime data collection with BOBE for summary plotting.

This example shows how to modify a BOBE run to collect runtime information
that can be used with the summary plotting module.
"""

import numpy as np
import time
from typing import Dict, List, Any
from pathlib import Path

# This would be your normal BOBE imports
# from jaxbo.bo import BOBE
# from jaxbo.loglike import ExternalLikelihood
# from jaxbo.summary_plots import BOBESummaryPlotter


class BOBEDataCollector:
    """
    Data collector for runtime information during BOBE runs.
    
    This class can be integrated into BOBE to collect data for summary plotting.
    """
    
    def __init__(self, param_names: List[str]):
        self.param_names = param_names
        self.reset()
    
    def reset(self):
        """Reset all collected data."""
        # GP hyperparameter evolution
        self.gp_iterations = []
        self.gp_lengthscales = []
        self.gp_outputscales = []
        
        # Best log-likelihood evolution
        self.best_loglike_iterations = []
        self.best_loglike_values = []
        
        # Parameter evolution (best point found so far)
        self.param_iterations = []
        self.param_values = {name: [] for name in self.param_names}
        
        # Timing information
        self.phase_times = {}
        self.phase_start_times = {}
        
    def start_phase(self, phase_name: str):
        """Start timing a phase."""
        self.phase_start_times[phase_name] = time.time()
    
    def end_phase(self, phase_name: str):
        """End timing a phase."""
        if phase_name in self.phase_start_times:
            elapsed = time.time() - self.phase_start_times[phase_name]
            if phase_name in self.phase_times:
                self.phase_times[phase_name] += elapsed
            else:
                self.phase_times[phase_name] = elapsed
            del self.phase_start_times[phase_name]
    
    def record_gp_hyperparams(self, iteration: int, lengthscales: np.ndarray, outputscale: float):
        """Record GP hyperparameters."""
        self.gp_iterations.append(iteration)
        self.gp_lengthscales.append(lengthscales.tolist())
        self.gp_outputscales.append(outputscale)
    
    def record_best_loglike(self, iteration: int, best_loglike: float):
        """Record best log-likelihood found so far."""
        self.best_loglike_iterations.append(iteration)
        self.best_loglike_values.append(best_loglike)
    
    def record_best_params(self, iteration: int, best_params: Dict[str, float]):
        """Record best parameter values found so far."""
        self.param_iterations.append(iteration)
        for param_name, value in best_params.items():
            if param_name in self.param_values:
                self.param_values[param_name].append(value)
    
    def get_gp_data(self) -> Dict[str, List]:
        """Get GP hyperparameter data for plotting."""
        return {
            'iterations': self.gp_iterations,
            'lengthscales': self.gp_lengthscales,
            'outputscales': self.gp_outputscales
        }
    
    def get_best_loglike_data(self) -> Dict[str, List]:
        """Get best log-likelihood data for plotting."""
        return {
            'iterations': self.best_loglike_iterations,
            'best_loglike': self.best_loglike_values
        }
    
    def get_timing_data(self) -> Dict[str, float]:
        """Get timing data for plotting."""
        return self.phase_times.copy()
    
    def get_param_evolution_data(self) -> Dict[str, Dict[str, List]]:
        """Get parameter evolution data for plotting."""
        return {
            param_name: {
                'iterations': self.param_iterations,
                'values': values
            }
            for param_name, values in self.param_values.items()
        }
    
    def save_runtime_data(self, output_file: str):
        """Save all collected runtime data."""
        import json
        
        runtime_data = {
            'gp_data': self.get_gp_data(),
            'best_loglike_data': self.get_best_loglike_data(),
            'timing_data': self.get_timing_data(),
            'param_evolution_data': self.get_param_evolution_data()
        }
        
        with open(f"{output_file}_runtime_data.json", 'w') as f:
            json.dump(runtime_data, f, indent=2)
        
        print(f"Saved runtime data to {output_file}_runtime_data.json")
    
    def load_runtime_data(self, output_file: str) -> Dict[str, Any]:
        """Load runtime data from file."""
        import json
        
        with open(f"{output_file}_runtime_data.json", 'r') as f:
            return json.load(f)


def example_bobe_run_with_data_collection():
    """
    Example of how to integrate data collection into a BOBE run.
    
    This is a template showing where to add data collection calls
    in your BOBE implementation.
    """
    
    # Setup (this would be your normal BOBE setup)
    param_names = ['x1', 'x2']
    output_file = "example_run"
    
    # Initialize data collector
    collector = BOBEDataCollector(param_names)
    
    # === Simulated BOBE run with data collection ===
    
    collector.start_phase("Total Runtime")
    collector.start_phase("Initialization")
    
    # Your BOBE initialization code here...
    time.sleep(0.1)  # Simulate initialization time
    
    collector.end_phase("Initialization")
    
    # Main BOBE loop
    n_iterations = 50
    best_loglike = -np.inf
    best_params = {'x1': 0.0, 'x2': 0.0}
    
    for iteration in range(n_iterations):
        
        # === GP Training Phase ===
        collector.start_phase("GP Training")
        
        # Your GP training code here...
        # Simulate GP hyperparameter evolution
        lengthscales = np.array([1.0, 1.0]) * np.exp(-iteration / 20.0) + 0.1
        outputscale = 2.0 + iteration * 0.1
        
        time.sleep(0.02)  # Simulate GP training time
        collector.end_phase("GP Training")
        
        # Record GP hyperparameters
        collector.record_gp_hyperparams(iteration, lengthscales, outputscale)
        
        # === Bayesian Optimization Phase ===
        collector.start_phase("BO Iteration")
        
        # Your BO iteration code here...
        # Simulate finding better points
        new_loglike = -10.0 + 9.5 * (1 - np.exp(-iteration / 15.0)) + np.random.normal(0, 0.1)
        if new_loglike > best_loglike:
            best_loglike = new_loglike
            # Simulate parameter convergence
            best_params['x1'] = 0.2 + 0.3 * np.exp(-iteration / 10.0) * np.sin(iteration / 5.0)
            best_params['x2'] = 0.1 + 0.2 * np.exp(-iteration / 8.0) * np.cos(iteration / 4.0)
        
        time.sleep(0.01)  # Simulate BO time
        collector.end_phase("BO Iteration")
        
        # Record best values
        collector.record_best_loglike(iteration, best_loglike)
        collector.record_best_params(iteration, best_params)
        
        # === Nested Sampling Phase (periodic) ===
        if iteration % 10 == 0:
            collector.start_phase("Nested Sampling")
            
            # Your nested sampling code here...
            time.sleep(0.05)  # Simulate NS time
            
            collector.end_phase("Nested Sampling")
    
    collector.end_phase("Total Runtime")
    
    # Save collected data
    collector.save_runtime_data(output_file)
    
    return collector


def example_plotting_with_collected_data():
    """
    Example of creating plots using collected runtime data.
    """
    output_file = "example_run"
    
    # Run the example (or load existing data)
    try:
        collector = BOBEDataCollector(['x1', 'x2'])
        runtime_data = collector.load_runtime_data(output_file)
        print("Loaded existing runtime data")
    except FileNotFoundError:
        print("Running example BOBE simulation...")
        collector = example_bobe_run_with_data_collection()
        runtime_data = {
            'gp_data': collector.get_gp_data(),
            'best_loglike_data': collector.get_best_loglike_data(),
            'timing_data': collector.get_timing_data(),
            'param_evolution_data': collector.get_param_evolution_data()
        }
    
    print("Collected runtime data:")
    print(f"  - GP iterations: {len(runtime_data['gp_data']['iterations'])}")
    print(f"  - Best loglike points: {len(runtime_data['best_loglike_data']['iterations'])}")
    print(f"  - Timing phases: {list(runtime_data['timing_data'].keys())}")
    print(f"  - Parameter evolution: {list(runtime_data['param_evolution_data'].keys())}")
    
    # Create plots (requires actual BOBE results file)
    try:
        from jaxbo.summary_plots import BOBESummaryPlotter
        
        # This would use your actual BOBE results
        # plotter = BOBESummaryPlotter("your_actual_results_file")
        # plotter.create_summary_dashboard(**runtime_data)
        
        print("\nTo create plots with real BOBE results:")
        print("plotter = BOBESummaryPlotter('your_results_file')")
        print("plotter.create_summary_dashboard(**runtime_data)")
        
    except ImportError:
        print("Summary plotting module not available")
    
    return runtime_data


def integration_points_in_bobe():
    """
    Documentation of where to add data collection in the actual BOBE code.
    """
    integration_points = """
    Integration Points in BOBE Code:
    
    1. In BOBE.__init__():
       - Initialize BOBEDataCollector
       - self.data_collector = BOBEDataCollector(self.param_names)
    
    2. In GP.fit() method:
       - After training, record hyperparameters:
       - collector.record_gp_hyperparams(iteration, self.lengthscales, self.outputscale)
    
    3. In BOBE.run() main loop:
       - Before each phase: collector.start_phase("phase_name")
       - After each phase: collector.end_phase("phase_name")
       - After finding new best point: collector.record_best_loglike(iter, best_logl)
       - collector.record_best_params(iter, best_params_dict)
    
    4. In BOBE.run() at the end:
       - collector.save_runtime_data(self.output_file)
       - Create plots: plotter = BOBESummaryPlotter(self.results_manager)
       - plotter.create_summary_dashboard(**collector.get_all_data())
    
    5. Optional: Real-time plotting
       - Create plots every N iterations for monitoring
       - Save intermediate plots during long runs
    """
    
    print(integration_points)


if __name__ == "__main__":
    print("BOBE Runtime Data Collection Example")
    print("=" * 40)
    
    # Show integration points
    integration_points_in_bobe()
    
    print("\nRunning example data collection...")
    runtime_data = example_plotting_with_collected_data()
    
    print("\n✅ Example completed!")
    print("Check 'example_run_runtime_data.json' for collected data format.")
