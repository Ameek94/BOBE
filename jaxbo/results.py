"""
Results management system for JaxBo BOBE sampler.

This module provides comprehensive result storage and formatting similar to 
typical nested samplers like Dynesty, PolyChord, MultiNest, etc.
"""

import numpy as np
import jax.numpy as jnp
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import warnings

try:
    from getdist import MCSamples
    HAS_GETDIST = True
except ImportError:
    HAS_GETDIST = False
    warnings.warn("GetDist not available. Some functionality will be limited.")

from .logging_utils import get_logger

log = get_logger("[results]")


# Removed IterationInfo dataclass - not needed for simplified tracking


@dataclass
class ConvergenceInfo:
    """Information about convergence checks and nested sampling runs."""
    iteration: int
    logz_dict: Dict[str, float]
    converged: bool
    delta: float
    threshold: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'iteration': self.iteration,
            'logz_dict': self.logz_dict,
            'converged': bool(self.converged),
            'delta': float(self.delta),
            'threshold': float(self.threshold)
        }


class BOBEResults:
    """
    Comprehensive results management for BOBE runs.
    
    This class handles storing, organizing, and outputting results in formats
    compatible with standard nested sampling analysis tools.
    """
    
    def __init__(self, 
                 output_file: str,
                 param_names: List[str],
                 param_labels: List[str],
                 param_bounds: np.ndarray,
                 settings: Optional[Dict[str, Any]] = None,
                 likelihood_name: str = "unknown"):
        """
        Initialize the results manager.
        
        Args:
            output_file: Base name for output files
            param_names: List of parameter names
            param_labels: List of parameter LaTeX labels
            param_bounds: Parameter bounds array [n_params, 2]
            settings: Dictionary of BOBE settings
            likelihood_name: Name of the likelihood function
        """
        self.output_file = output_file
        self.param_names = param_names
        self.param_labels = param_labels
        self.param_bounds = np.array(param_bounds)
        self.ndim = len(param_names)
        self.likelihood_name = likelihood_name
        
        # Store settings
        self.settings = settings or {}
        
        # Initialize tracking variables
        self.start_time = time.time()
        self.end_time = None
        
        # Storage for convergence data
        self.convergence_history: List[ConvergenceInfo] = []
        
        # Evidence tracking
        self.logz_evolution = []
        
        # Simple timing system - cumulative times for each phase
        self.phase_times = {
            'GP Training': 0.0,
            'Acquisition Optimization': 0.0,
            'True Objective Evaluations': 0.0,
            'Nested Sampling': 0.0,
            'MCMC Sampling': 0.0
        }
        self._active_timers = {}  # Track start times for active phases
        
        # Final results
        self.final_samples = None
        self.final_weights = None
        self.final_loglikes = None
        self.final_logz_dict = None
        self.converged = False
        self.termination_reason = "Unknown"
        
        log.info(f"Initialized BOBE results manager for {self.ndim}D problem")
    
    def update_iteration(self, iteration: int, **kwargs):
        """
        Simplified iteration update - only saves intermediate results periodically.
        
        Args:
            iteration: Current iteration number
            **kwargs: Additional arguments (ignored in simplified version)
        """
        # Save intermediate results periodically
        if iteration % 50 == 0:
            self.save_intermediate()
    
    def update_convergence(self,
                          iteration: int,
                          logz_dict: Dict[str, float],
                          converged: bool,
                          threshold: float):
        """
        Update convergence information from a nested sampling check.
        
        Args:
            iteration: Current iteration number
            logz_dict: Dictionary with logz information
            converged: Whether convergence was achieved
            threshold: Convergence threshold used
        """
        delta = logz_dict.get('upper', 0) - logz_dict.get('lower', 0)
        
        conv_info = ConvergenceInfo(
            iteration=iteration,
            logz_dict=logz_dict.copy(),
            converged=converged,
            delta=delta,
            threshold=threshold
        )
        
        self.convergence_history.append(conv_info)
        
        # Track logz evolution
        self.logz_evolution.append({
            'iteration': iteration,
            'logz': logz_dict.get('mean', np.nan),
            'logz_err': delta
        })
    
    def start_timing(self, phase_name: str):
        """Start timing a specific phase."""
        if phase_name in self.phase_times:
            self._active_timers[phase_name] = time.time()
    
    def end_timing(self, phase_name: str):
        """End timing a specific phase and accumulate the time."""
        if phase_name in self._active_timers:
            elapsed = time.time() - self._active_timers[phase_name]
            self.phase_times[phase_name] += elapsed
            del self._active_timers[phase_name]
    
    def get_timing_summary(self) -> Dict[str, Any]:
        """Get a summary of timing information."""
        total_runtime = (self.end_time or time.time()) - self.start_time
        
        # Calculate percentages
        percentages = {}
        if total_runtime > 0:
            for phase, time_spent in self.phase_times.items():
                percentages[phase] = (time_spent / total_runtime) * 100
        
        return {
            'phase_times': self.phase_times.copy(),
            'percentages': percentages,
            'total_runtime': total_runtime
        }
    
    def save_timing_data(self):
        """Save timing data to JSON file."""
        timing_data = self.get_timing_summary()
        timing_file = f"{self.output_file}_timing.json"
        
        with open(timing_file, 'w') as f:
            json.dump(timing_data, f, indent=2)
        
        log.info(f"Saved timing data to {timing_file}")
    
    def finalize(self,
                 samples: np.ndarray,
                 weights: np.ndarray,
                 loglikes: np.ndarray,
                 logz_dict: Optional[Dict[str, float]] = None,
                 converged: bool = False,
                 termination_reason: str = "Max iterations reached"):
        """
        Finalize the results with final samples and metadata.
        
        Args:
            samples: Final parameter samples [n_samples, n_params]
            weights: Sample weights [n_samples]
            loglikes: Log-likelihood values [n_samples]
            logz_dict: Final evidence information
            converged: Whether the run converged
            termination_reason: Reason for termination
        """
        self.end_time = time.time()
        
        self.final_samples = np.array(samples)
        self.final_weights = np.array(weights)
        self.final_loglikes = np.array(loglikes)
        self.final_logz_dict = logz_dict or {}
        self.converged = converged
        self.termination_reason = termination_reason
        
        log.info(f"Finalized BOBE results: {len(samples)} samples, "
                f"converged={converged}, reason={termination_reason}")
        
        # Save all results
        self.save_all_formats()
    
    def get_results_dict(self) -> Dict[str, Any]:
        """
        Get simplified results dictionary with only essential data.
        
        Returns:
            Dictionary containing samples, weights, evidence evolution, and convergence info
        """
        if self.final_samples is None:
            raise ValueError("Results not finalized. Call finalize() first.")
        
        # Calculate effective sample size
        if len(self.final_weights) > 0:
            n_effective = int(np.sum(self.final_weights)**2 / np.sum(self.final_weights**2))
        else:
            n_effective = 0
        
        # Runtime
        runtime = self.end_time - self.start_time if self.end_time else 0
        
        results = {
            # === SAMPLES AND WEIGHTS ===
            'samples': self.final_samples,
            'weights': self.final_weights,
            'logl': self.final_loglikes,
            'logwt': np.log(self.final_weights) if len(self.final_weights) > 0 else np.array([]),
            
            # === EVIDENCE INFORMATION ===
            'logz': self.final_logz_dict.get('mean', np.nan),
            'logzerr': self.final_logz_dict.get('upper', 0) - self.final_logz_dict.get('lower', 0),
            'logz_bounds': {
                'lower': self.final_logz_dict.get('lower', np.nan),
                'upper': self.final_logz_dict.get('upper', np.nan),
                'mean': self.final_logz_dict.get('mean', np.nan)
            },
            'logz_history': self.logz_evolution,
            
            # === PARAMETER INFORMATION ===
            'param_names': self.param_names,
            'param_labels': self.param_labels,
            'param_bounds': self.param_bounds,
            'ndim': self.ndim,
            
            # === BASIC SAMPLING INFORMATION ===
            'n_samples': len(self.final_samples),
            'n_effective': n_effective,
            
            # === CONVERGENCE INFORMATION ===
            'converged': self.converged,
            'termination_reason': self.termination_reason,
            'convergence_history': [conv.to_dict() for conv in self.convergence_history],
            
            # === TIMING INFORMATION ===
            'timing': self.get_timing_summary(),
            
            # === MINIMAL METADATA ===
            'run_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
                'runtime_hours': runtime / 3600,
                'likelihood_name': self.likelihood_name,
                'output_file': self.output_file,
                'settings': self.settings
            }
        }
        
        return results
    
    def save_all_formats(self):
        """Save results in multiple formats for compatibility."""
        if self.final_samples is None:
            log.warning("No final samples to save")
            return
        
        # Main results file (comprehensive)
        self.save_main_results()
        
        # Chain files for compatibility
        self.save_chain_files()
        
        # Summary statistics
        self.save_summary_stats()
        
        # Diagnostic files
        self.save_diagnostics()
        
        # Timing data
        self.save_timing_data()
    
    def save_main_results(self):
        """Save main comprehensive results file."""
        results = self.get_results_dict()
        
        # Save as compressed numpy file
        main_file = f"{self.output_file}_results.npz"
        
        # Prepare arrays for npz (can't save nested dicts directly)
        save_dict = {}
        for key, value in results.items():
            if isinstance(value, (dict, list)):
                # Save complex objects as pickled bytes
                save_dict[f"{key}_pickle"] = pickle.dumps(value)
            else:
                save_dict[key] = value
        
        np.savez_compressed(main_file, **save_dict)
        log.info(f"Saved main results to {main_file}")
        
        # Also save as pickle for full Python object preservation
        pickle_file = f"{self.output_file}_results.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"Saved pickle results to {pickle_file}")
    
    def save_chain_files(self):
        """Save chain files in standard formats."""
        if len(self.final_samples) == 0:
            return
        
        # GetDist format: weight like param1 param2 ...
        getdist_file = f"{self.output_file}.txt"
        header = "weight like " + " ".join(self.param_names)
        
        # Prepare data in correct GetDist format
        data = np.column_stack([
            self.final_weights,
            -self.final_loglikes,  # GetDist uses -logL (like = -log(posterior))
            self.final_samples
        ])
        
        np.savetxt(getdist_file, data, header=header, fmt='%.8e')
        log.info(f"Saved GetDist format chain to {getdist_file}")
        
        # CosmoMC format: weight -logL param1 param2 ... (for compatibility)
        cosmomc_file = f"{self.output_file}_1.txt"
        header_cosmomc = "weight -logL " + " ".join(self.param_names)
        
        data_cosmomc = np.column_stack([
            self.final_weights,
            -self.final_loglikes,
            self.final_samples
        ])
        
        np.savetxt(cosmomc_file, data_cosmomc, header=header_cosmomc, fmt='%.8e')
        log.info(f"Saved CosmoMC format chain to {cosmomc_file}")
        
        # Create .paramnames file for GetDist
        paramnames_file = f"{self.output_file}.paramnames"
        with open(paramnames_file, 'w') as f:
            for name, label in zip(self.param_names, self.param_labels):
                f.write(f"{name}    {label}\n")
        log.info(f"Saved parameter names to {paramnames_file}")
        
        # Create .ranges file for GetDist  
        ranges_file = f"{self.output_file}.ranges"
        with open(ranges_file, 'w') as f:
            for i, name in enumerate(self.param_names):
                # param_bounds is shape (2, d): [lower_bounds, upper_bounds]
                lower = self.param_bounds[0, i]
                upper = self.param_bounds[1, i]
                f.write(f"{name}    {lower}    {upper}\n")
        log.info(f"Saved parameter ranges to {ranges_file}")
        
    
    def save_summary_stats(self):
        """Save summary statistics in JSON format."""
        if len(self.final_samples) == 0:
            return
        
        # Calculate parameter statistics
        param_stats = {}
        for i, name in enumerate(self.param_names):
            values = self.final_samples[:, i]
            weights = self.final_weights
            
            # Weighted statistics
            mean = np.average(values, weights=weights)
            var = np.average((values - mean)**2, weights=weights)
            std = np.sqrt(var)
            
            # Percentiles (approximate for weighted samples)
            sorted_idx = np.argsort(values)
            sorted_weights = weights[sorted_idx]
            cumsum = np.cumsum(sorted_weights) / np.sum(sorted_weights)
            
            def weighted_percentile(p):
                idx = np.searchsorted(cumsum, p/100.0)
                if idx >= len(values):
                    idx = len(values) - 1
                return values[sorted_idx[idx]]
            
            param_stats[name] = {
                "mean": float(mean),
                "std": float(std),
                "2.5_percentile": float(weighted_percentile(2.5)),
                "97.5_percentile": float(weighted_percentile(97.5)),
                "16_percentile": float(weighted_percentile(16)),
                "84_percentile": float(weighted_percentile(84)),
                "median": float(weighted_percentile(50))
            }
        
        # Overall statistics
        stats = {
            "evidence": {
                "logz": float(self.final_logz_dict.get('mean', np.nan)),
                "logz_err": float(self.final_logz_dict.get('upper', 0) - self.final_logz_dict.get('lower', 0)),
                "logz_lower": float(self.final_logz_dict.get('lower', np.nan)),
                "logz_upper": float(self.final_logz_dict.get('upper', np.nan))
            },
            "parameters": param_stats,
            "diagnostics": {
                "n_samples": int(len(self.final_samples)),
                "n_effective": int(np.sum(self.final_weights)**2 / np.sum(self.final_weights**2)) if len(self.final_weights) > 0 else 0,
                "runtime_hours": float((self.end_time - self.start_time) / 3600) if self.end_time else 0,
                "converged": bool(self.converged),
                "termination_reason": str(self.termination_reason)
            }
        }
        
        stats_file = f"{self.output_file}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        log.info(f"Saved summary statistics to {stats_file}")
    
    def save_diagnostics(self):
        """Save convergence diagnostic information."""
        # Convergence diagnostics
        if self.convergence_history:
            conv_diagnostics = {
                'iterations': [conv.iteration for conv in self.convergence_history],
                'logz_values': [conv.logz_dict.get('mean', np.nan) for conv in self.convergence_history],
                'logz_errors': [conv.delta for conv in self.convergence_history],
                'thresholds': [conv.threshold for conv in self.convergence_history],
                'converged_flags': [conv.converged for conv in self.convergence_history]
            }
            
            conv_file = f"{self.output_file}_convergence.npz"
            np.savez_compressed(conv_file, **conv_diagnostics)
            log.info(f"Saved convergence diagnostics to {conv_file}")
    
    def save_intermediate(self):
        """Save intermediate results for crash recovery."""
        intermediate = {
            'convergence_history': [conv.to_dict() for conv in self.convergence_history],
            'logz_evolution': self.logz_evolution,
            'start_time': self.start_time,
            'param_names': self.param_names,
            'param_labels': self.param_labels,
            'param_bounds': self.param_bounds.tolist(),
            'settings': self.settings
        }
        
        intermediate_file = f"{self.output_file}_intermediate.json"
        with open(intermediate_file, 'w') as f:
            json.dump(intermediate, f, indent=2)
    
    def get_getdist_samples(self) -> Optional['MCSamples']:
        """
        Convert results to GetDist MCSamples object.
        
        Returns:
            GetDist MCSamples object if GetDist is available, None otherwise
        """
        if not HAS_GETDIST:
            log.warning("GetDist not available, cannot create MCSamples object")
            return None
        
        if self.final_samples is None:
            log.warning("No final samples available")
            return None
        
        # Parameter ranges for GetDist
        # param_bounds is shape (2, d): [lower_bounds, upper_bounds]
        ranges = {name: [self.param_bounds[0, i], self.param_bounds[1, i]] 
                  for i, name in enumerate(self.param_names)}
        
        # Determine sampler method
        sampler_method = 'nested' if self.final_logz_dict else 'mcmc'
        
        samples = MCSamples(
            samples=self.final_samples,
            names=self.param_names,
            labels=self.param_labels,
            ranges=ranges,
            weights=self.final_weights,
            loglikes=self.final_loglikes,
            label='BOBE',
            sampler=sampler_method
        )
        
        return samples
    
    @classmethod
    def load_results(cls, output_file: str) -> 'BOBEResults':
        """
        Load results from saved files.
        
        Args:
            output_file: Base name of the output files
            
        Returns:
            BOBEResults object with loaded data
        """
        # Try to load from pickle first (most complete)
        pickle_file = f"{output_file}_results.pkl"
        if Path(pickle_file).exists():
            with open(pickle_file, 'rb') as f:
                results_dict = pickle.load(f)
            
            # Reconstruct BOBEResults object
            results = cls(
                output_file=output_file,
                param_names=results_dict['param_names'],
                param_labels=results_dict['param_labels'],
                param_bounds=results_dict['param_bounds'],
                settings=results_dict['run_info']['settings'],
                likelihood_name=results_dict['run_info']['likelihood_name']
            )
            
            # Restore data
            results.final_samples = results_dict['samples']
            results.final_weights = results_dict['weights']
            results.final_loglikes = results_dict['logl']
            results.final_logz_dict = results_dict['logz_bounds']
            results.converged = results_dict['converged']
            results.termination_reason = results_dict['termination_reason']
            
            # Restore convergence and evidence evolution
            if 'convergence_history' in results_dict:
                # Reconstruct ConvergenceInfo objects
                results.convergence_history = []
                for conv_dict in results_dict['convergence_history']:
                    conv_info = ConvergenceInfo(
                        iteration=conv_dict['iteration'],
                        logz_dict=conv_dict['logz_dict'],
                        converged=conv_dict['converged'],
                        delta=conv_dict['delta'],
                        threshold=conv_dict['threshold']
                    )
                    results.convergence_history.append(conv_info)
            
            if 'logz_history' in results_dict:
                results.logz_evolution = results_dict['logz_history']
            
            # Restore timing
            start_str = results_dict['run_info']['start_time']
            end_str = results_dict['run_info']['end_time']
            results.start_time = datetime.fromisoformat(start_str).timestamp()
            if end_str:
                results.end_time = datetime.fromisoformat(end_str).timestamp()
            
            log.info(f"Loaded complete results from {pickle_file}")
            return results
        
        else:
            raise FileNotFoundError(f"Results file not found: {pickle_file}")


def load_bobe_results(output_file: str) -> BOBEResults:
    """
    Convenience function to load BOBE results.
    
    Args:
        output_file: Base name of the output files
        
    Returns:
        BOBEResults object with loaded data
    """
    return BOBEResults.load_results(output_file)
