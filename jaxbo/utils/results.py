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

from .timing import BOBETimingCollector  
from .logging_utils import get_logger

log = get_logger("results")


def convert_jax_to_json_serializable(obj):
    """
    Convert JAX arrays and other non-JSON-serializable objects to JSON-serializable types.
    
    Args:
        obj: Object to convert (can be JAX array, numpy array, list, dict, etc.)
        
    Returns:
        JSON-serializable version of the object
    """
    if hasattr(obj, 'tolist'):  # JAX arrays and numpy arrays
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_jax_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_jax_to_json_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, '__array__'):  # Other array-like objects
        return np.asarray(obj).tolist()
    else:
        return obj


# Removed IterationInfo dataclass - not needed for simplified tracking


@dataclass
class ConvergenceInfo:
    """Information about convergence checks and nested sampling runs."""
    iteration: int
    logz_dict: Dict[str, float]
    converged: bool
    delta: float
    threshold: float
    dlogz_sampler: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'iteration': self.iteration,
            'logz_dict': self.logz_dict,
            'converged': bool(self.converged),
            'delta': float(self.delta),
            'threshold': float(self.threshold),
            'dlogz_sampler': float(self.dlogz_sampler),
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
                 likelihood_name: str = "unknown",
                 resume_from_existing: bool = False):
        """
        Initialize the results manager.
        
        Args:
            output_file: Base name for output files
            param_names: List of parameter names
            param_labels: List of parameter LaTeX labels
            param_bounds: Parameter bounds array [n_params, 2]
            settings: Dictionary of BOBE settings
            likelihood_name: Name of the likelihood function
            resume_from_existing: If True, try to load existing results and continue from there
        """
        self.output_file = output_file
        self.param_names = param_names
        self.param_labels = param_labels
        self.param_bounds = np.array(param_bounds)
        self.ndim = len(param_names)
        self.likelihood_name = likelihood_name
        
        # Store settings
        self.settings = settings or {}
        
        # Try to resume from existing results if requested
        if resume_from_existing:
            existing_results = self._load_existing_results(output_file)
            if existing_results:
                self._merge_existing_results(existing_results)
                log.info(f"Resumed from existing results with {len(self.convergence_history)} previous iterations")
            else:
                log.info("No existing results found, starting fresh")
                self._initialize_fresh()
        else:
            self._initialize_fresh()
        
        log.info(f"Initialized BOBE results manager for {self.ndim}D problem")
    
    def _initialize_fresh(self):
        """Initialize all tracking variables for a fresh run."""
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
        
        # GP hyperparameter tracking
        self.gp_iterations = []
        self.gp_lengthscales = []
        self.gp_kernel_variances = []
        
        # Best loglikelihood tracking 
        self.best_loglike_iterations = []
        self.best_loglike_values = []

        # Acquisition function tracking
        self.acquisition_iterations = []
        self.acquisition_values = []
        self.acquisition_functions = []
        
        # KL divergence tracking for convergence analysis
        self.kl_iterations = []
        self.kl_divergences = []  # List of dictionaries with KL results
        self.successive_kl = []   # KL between successive iterations
        
        # Final results
        self.final_samples = None
        self.final_weights = None
        self.final_loglikes = None
        self.final_logz_dict = None
        self.converged = False
        self.termination_reason = "Unknown"
        self.gp_info = {}  # Store GP and classifier information
    
    def _load_existing_results(self, output_file: str) -> Optional[Dict[str, Any]]:
        """
        Try to load existing results from previous runs.
        
        Args:
            output_file: Base name of the output files
            
        Returns:
            Dictionary of existing results if found, None otherwise
        """
        # First try to load from pickle (most complete)
        pickle_file = f"{output_file}_results.pkl"
        if Path(pickle_file).exists():
            try:
                with open(pickle_file, 'rb') as f:
                    results_dict = pickle.load(f)
                log.info(f"Found existing results in {pickle_file}")
                return results_dict
            except Exception as e:
                log.warning(f"Could not load existing pickle results: {e}")
        
        # Try to load from intermediate JSON
        intermediate_file = f"{output_file}_intermediate.json"
        if Path(intermediate_file).exists():
            try:
                with open(intermediate_file, 'r') as f:
                    intermediate_dict = json.load(f)
                log.info(f"Found existing intermediate results in {intermediate_file}")
                return intermediate_dict
            except Exception as e:
                log.warning(f"Could not load existing intermediate results: {e}")
        
        return None
    
    def _merge_existing_results(self, existing_results: Dict[str, Any]):
        """
        Merge existing results into this instance for resuming.
        
        Args:
            existing_results: Dictionary of existing results to merge
        """
        # Initialize fresh first
        self._initialize_fresh()
        
        # Restore convergence history
        if 'convergence_history' in existing_results:
            self.convergence_history = []
            for conv_dict in existing_results['convergence_history']:
                conv_info = ConvergenceInfo(
                    iteration=conv_dict['iteration'],
                    logz_dict=conv_dict['logz_dict'],
                    converged=conv_dict['converged'],
                    delta=conv_dict['delta'],
                    threshold=conv_dict['threshold']
                )
                self.convergence_history.append(conv_info)
        
        # Restore evidence evolution
        if 'logz_evolution' in existing_results:
            self.logz_evolution = existing_results['logz_evolution'].copy()
        elif 'logz_history' in existing_results:
            self.logz_evolution = existing_results['logz_history'].copy()
        
        # Restore acquisition function data if available
        if 'acquisition_data' in existing_results:
            acq_data = existing_results['acquisition_data']
            self.acquisition_iterations = acq_data.get('iterations', []).copy()
            self.acquisition_values = acq_data.get('values', []).copy()
            self.acquisition_functions = acq_data.get('functions', []).copy()
        
        # Restore GP hyperparameter data if available (from comprehensive results)
        if 'gp_hyperparams' in existing_results:
            gp_data = existing_results['gp_hyperparams']
            self.gp_iterations = gp_data.get('iterations', []).copy()
            self.gp_lengthscales = gp_data.get('lengthscales', []).copy()
            self.gp_kernel_variances = gp_data.get('kernel_variances', []).copy()
            # Backward compatibility: check for old 'outputscales' key
            if 'outputscales' in gp_data and not self.gp_kernel_variances:
                self.gp_kernel_variances = gp_data.get('outputscales', []).copy()
        
        # Restore best loglikelihood data if available
        if 'best_loglike_data' in existing_results:
            loglike_data = existing_results['best_loglike_data']
            self.best_loglike_iterations = loglike_data.get('iterations', []).copy()
            self.best_loglike_values = loglike_data.get('best_loglike', []).copy()
        
        # Restore KL divergence data if available
        if 'kl_data' in existing_results:
            kl_data = existing_results['kl_data']
            self.kl_iterations = kl_data.get('iterations', []).copy()
            self.kl_divergences = kl_data.get('kl_divergences', []).copy()
            self.successive_kl = kl_data.get('successive_kl', []).copy()
        # Also check legacy naming for backward compatibility
        elif 'kl_divergence_data' in existing_results:
            kl_data = existing_results['kl_divergence_data']
            self.kl_iterations = kl_data.get('iterations', []).copy()
            self.kl_divergences = kl_data.get('kl_divergences', []).copy()
            self.successive_kl = kl_data.get('successive_kl', []).copy()
        
        # Restore timing information (accumulate previous times)
        if 'timing' in existing_results and 'phase_times' in existing_results['timing']:
            for phase, prev_time in existing_results['timing']['phase_times'].items():
                if phase in self.phase_times:
                    self.phase_times[phase] = prev_time
        
        # Restore timing from phase_times if available (for backward compatibility)
        if 'phase_times' in existing_results:
            for phase, prev_time in existing_results['phase_times'].items():
                if phase in self.phase_times:
                    self.phase_times[phase] = prev_time
        
        # Restore GP info
        if 'gp_info' in existing_results:
            self.gp_info = existing_results['gp_info'].copy()
        
        # If this was a completed run, preserve final results
        if 'samples' in existing_results and existing_results['samples'] is not None:
            self.final_samples = np.array(existing_results['samples'])
            self.final_weights = np.array(existing_results['weights'])
            self.final_loglikes = np.array(existing_results['logl'])
            self.final_logz_dict = existing_results.get('logz_bounds', {})
            self.converged = existing_results.get('converged', False)
            self.termination_reason = existing_results.get('termination_reason', "Resumed run")
        
        # Update start time to preserve total runtime calculation
        if 'run_info' in existing_results and 'start_time' in existing_results['run_info']:
            start_str = existing_results['run_info']['start_time']
            try:
                self.start_time = datetime.fromisoformat(start_str).timestamp()
            except Exception:
                # If parsing fails, keep current start time
                pass

    def update_iteration(self, iteration: int, save_step: int, gp = None, **kwargs):
        """
        Simplified iteration update - only saves intermediate results periodically.
        
        Args:
            iteration: Current iteration number
            **kwargs: Additional arguments (ignored in simplified version)
        """
        # Save intermediate results periodically
        if iteration % save_step == 0:
            self.save_intermediate(gp=gp)

    def update_acquisition(self, iteration: int, acquisition_value: float, acquisition_function: str):
        """
        Track acquisition function values throughout iterations.
        
        Args:
            iteration: Current iteration number
            acquisition_value: Value of the acquisition function at the selected point
            acquisition_function: String name of the acquisition function used
        """
        self.acquisition_iterations.append(iteration)
        self.acquisition_values.append(float(acquisition_value))
        self.acquisition_functions.append(acquisition_function)

    def update_gp_hyperparams(self, iteration: int, lengthscales: list, kernel_variance: float):
        """
        Track GP hyperparameters evolution.
        
        Args:
            iteration: Current iteration number
            lengthscales: List of lengthscale values (can be JAX arrays)
            kernel_variance: Kernel variance value
        """
        self.gp_iterations.append(iteration)
        self.gp_lengthscales.append(lengthscales)
        self.gp_kernel_variances.append(float(kernel_variance))
    
    def update_best_loglike(self, iteration: int, best_loglike: float):
        """
        Track best loglikelihood evolution.
        
        Args:
            iteration: Current iteration number
            best_loglike: Current best loglikelihood value
        """
        self.best_loglike_iterations.append(iteration)
        self.best_loglike_values.append(best_loglike)
    
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
        delta = logz_dict['std'] #logz_dict.get('upper', 0) - logz_dict.get('lower', 0)
        
        conv_info = ConvergenceInfo(
            iteration=iteration,
            logz_dict=logz_dict.copy(),
            converged=converged,
            delta=delta,
            threshold=threshold,
            dlogz_sampler=logz_dict.get('dlogz_sampler', np.nan)
        )
        
        self.convergence_history.append(conv_info)
        
        # Track logz evolution
        self.logz_evolution.append({
            'iteration': iteration,
            'logz': logz_dict.get('mean', np.nan),
            'logz_upper': logz_dict.get('upper', np.nan),
            'logz_lower': logz_dict.get('lower', np.nan),
            'logz_err': delta,
            'logz_var': logz_dict.get('var', np.nan),
            'logz_std': logz_dict.get('std', np.nan),
            'dlogz_sampler': logz_dict.get('dlogz_sampler', np.nan)
        })
    
    def update_kl_divergences(self,
                             iteration: int,
                             successive_kl: Optional[Dict[str, float]] = None):
        """
        Update KL divergence tracking for convergence analysis.
        
        Args:
            iteration: Current iteration number
            successive_kl: Optional KL divergence between successive iterations
        """
        self.kl_iterations.append(iteration)
        
        if successive_kl is not None:
            self.successive_kl.append({
                'iteration': iteration,
                **successive_kl
            })
    
    def get_last_iteration(self) -> int:
        """
        Get the last iteration number from the results history.
        
        Returns:
            Last iteration number, or 0 if no iterations have been recorded
        """
        if self.convergence_history:
            return self.convergence_history[-1].iteration
        elif self.acquisition_iterations:
            return max(self.acquisition_iterations)
        elif self.gp_iterations:
            return max(self.gp_iterations)
        elif self.best_loglike_iterations:
            return max(self.best_loglike_iterations)
        else:
            return 0
    
    def is_resuming(self) -> bool:
        """
        Check if this is a resumed run (has existing data).
        
        Returns:
            True if this appears to be a resumed run
        """
        return (len(self.convergence_history) > 0 or 
                len(self.acquisition_iterations) > 0 or
                len(self.gp_iterations) > 0 or 
                len(self.best_loglike_iterations) > 0)
    
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
    
    def get_gp_data(self) -> Dict[str, list]:
        """
        Get GP hyperparameter evolution data for plotting.
        
        Returns:
            Dictionary with 'iterations', 'lengthscales', and 'kernel_variances' keys
        """
        return {
            'iterations': self.gp_iterations,
            'lengthscales': convert_jax_to_json_serializable(self.gp_lengthscales),
            'kernel_variances': convert_jax_to_json_serializable(self.gp_kernel_variances)
        }
    
    def get_acquisition_data(self) -> Dict[str, list]:
        """
        Get acquisition function evolution data for plotting.
        
        Returns:
            Dictionary with 'iterations', 'values', and 'functions' keys
        """
        return {
            'iterations': self.acquisition_iterations,
            'values': self.acquisition_values,
            'functions': self.acquisition_functions
        }
    
    def get_best_loglike_data(self) -> Dict[str, list]:
        """
        Get best loglikelihood evolution data for plotting.
        
        Returns:
            Dictionary with 'iterations' and 'best_loglike' keys
        """
        return {
            'iterations': self.best_loglike_iterations,
            'best_loglike': self.best_loglike_values
        }
    
    def finalize(self,
                 samples: np.ndarray,
                 weights: np.ndarray,
                 loglikes: np.ndarray,
                 logz_dict: Optional[Dict[str, float]] = None,
                 converged: bool = False,
                 termination_reason: str = "Max iterations reached",
                 gp_info: Optional[Dict[str, Any]] = None):
        """
        Finalize the results with final samples and metadata.
        
        Args:
            samples: Final parameter samples [n_samples, n_params]
            weights: Sample weights [n_samples]
            loglikes: Log-likelihood values [n_samples]
            logz_dict: Final evidence information
            converged: Whether the run converged
            termination_reason: Reason for termination
            gp_info: Dictionary containing GP and classifier information
        """
        self.end_time = time.time()
        
        self.final_samples = np.array(samples)
        self.final_weights = np.array(weights)
        self.final_loglikes = np.array(loglikes)
        
        # Use provided logz_dict, or fall back to the last convergence check
        if logz_dict is not None:
            self.final_logz_dict = logz_dict
        elif self.convergence_history:
            # Use the logz_dict from the last convergence check
            self.final_logz_dict = self.convergence_history[-1].logz_dict.copy()
        else:
            self.final_logz_dict = {}
        
        self.converged = converged
        self.termination_reason = termination_reason
        self.gp_info = gp_info or {}
        
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
            'dlogz_sampler': float(self.final_logz_dict.get('dlogz_sampler', np.nan)),
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
            
            # === GP AND CLASSIFIER INFORMATION ===
            'gp_info': self.gp_info,

            # === ACQUISITION FUNCTION TRACKING ===
            'acquisition_data': {
                'iterations': self.acquisition_iterations,
                'values': self.acquisition_values,
                'functions': self.acquisition_functions
            },

            # === GP HYPERPARAMETER TRACKING ===
            'gp_hyperparams': {
                'iterations': self.gp_iterations,
                'lengthscales': self.gp_lengthscales,
                'kernel_variances': self.gp_kernel_variances
            },

            # === BEST LOGLIKELIHOOD TRACKING ===
            'best_loglike_data': {
                'iterations': self.best_loglike_iterations,
                'best_loglike': self.best_loglike_values
            },

            # === KL DIVERGENCE TRACKING ===
            'kl_data': {
                'iterations': self.kl_iterations,
                'kl_divergences': self.kl_divergences,
                'successive_kl': self.successive_kl
            },

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
        
        # Timing data
        self.save_timing_data()
    
    def save_main_results(self):
        """Save main comprehensive results file."""
        results = self.get_results_dict()
        
        # Save as pickle for full Python object preservation
        pickle_file = f"{self.output_file}_results.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"Saved main results to {pickle_file}")
    
    def save_chain_files(self):
        """Save chain files in GetDist format using MCSamples.saveAsText method."""
        if len(self.final_samples) == 0:
            return
        
        if not HAS_GETDIST:
            log.warning("GetDist not available, cannot save chain files")
            return
        
        # Get MCSamples object
        samples = self.get_getdist_samples()
        if samples is None:
            log.warning("Could not create MCSamples object")
            return
        
        # Use GetDist's saveAsText method to save the chain files
        # This automatically creates .txt, .paramnames, and .ranges files
        samples.saveAsText(self.output_file)
        log.info(f"Saved GetDist format files using MCSamples.saveAsText to {self.output_file}")
        log.info("  - Created: .txt (chain), .paramnames (parameter info), .ranges (parameter bounds)")
        
    
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
                "logz_upper": float(self.final_logz_dict.get('upper', np.nan)),
                "dlogz_sampler": float(self.final_logz_dict.get('dlogz_sampler', np.nan))
            },
            "diagnostics": {
                "n_samples": int(len(self.final_samples)),
                "n_effective": int(np.sum(self.final_weights)**2 / np.sum(self.final_weights**2)) if len(self.final_weights) > 0 else 0,
                "runtime_hours": float((self.end_time - self.start_time) / 3600) if self.end_time else 0,
                "converged": bool(self.converged),
                "termination_reason": str(self.termination_reason)
            },
            "gp_info": self.gp_info,
            # "acquisition_function": {
            #     "iterations": [int(x) for x in self.acquisition_iterations],
            #     "values": [float(x) for x in self.acquisition_values],
            #     "functions": self.acquisition_functions
            # },
            "final_convergence": {
                "iteration": int(self.convergence_history[-1].iteration),
                "logz_value": float(self.convergence_history[-1].logz_dict.get('mean', np.nan)),
                "logz_error": float(self.convergence_history[-1].delta),
                "threshold": float(self.convergence_history[-1].threshold),
                "converged": bool(self.convergence_history[-1].converged),
                "dlogz_sampler": self.convergence_history[-1].logz_dict.get('dlogz_sampler', np.nan)

            } if self.convergence_history else {},
            "parameters": param_stats,

        }
        
        stats_file = f"{self.output_file}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        log.info(f"Saved summary statistics to {stats_file}")
    
    def save_intermediate(self,gp):
        """Save intermediate results for crash recovery and resuming."""
        intermediate = {
            'convergence_history': [conv.to_dict() for conv in self.convergence_history],
            'logz_evolution': self.logz_evolution,
            'acquisition_data': {
                'iterations': self.acquisition_iterations,
                'values': self.acquisition_values,
                'functions': self.acquisition_functions
            },
            'gp_hyperparams': {
                'iterations': self.gp_iterations,
                'lengthscales': convert_jax_to_json_serializable(self.gp_lengthscales),
                'kernel_variances': convert_jax_to_json_serializable(self.gp_kernel_variances)
            },
            'best_loglike_data': {
                'iterations': self.best_loglike_iterations,
                'best_loglike': self.best_loglike_values
            },
            'kl_data': {
                'iterations': self.kl_iterations,
                'kl_divergences': convert_jax_to_json_serializable(self.kl_divergences),
                'successive_kl': convert_jax_to_json_serializable(self.successive_kl)
            },
            'timing': self.get_timing_summary(),
            'gp_info': self.gp_info,
            'start_time': self.start_time,
            'param_names': self.param_names,
            'param_labels': self.param_labels,
            'param_bounds': self.param_bounds.tolist(),
            'settings': self.settings,
            'run_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'likelihood_name': self.likelihood_name,
                'output_file': self.output_file
            }
        }
        
        intermediate_file = f"{self.output_file}_intermediate.json"
        with open(intermediate_file, 'w') as f:
            # Convert the entire intermediate dictionary to ensure all JAX arrays are handled
            json_safe_intermediate = convert_jax_to_json_serializable(intermediate)
            json.dump(json_safe_intermediate, f, indent=2)
        log.info(f"Saved intermediate results to {intermediate_file}")

        if gp is not None:
            gp.save(outfile=f"{self.output_file}_gp")
    
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
        # param_bounds is shape (2, nparams)
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
                        threshold=conv_dict['threshold'],
                        dlogz_sampler=conv_dict.get('dlogz_sampler', np.nan)
                    )
                    results.convergence_history.append(conv_info)
            
            if 'logz_history' in results_dict:
                results.logz_evolution = results_dict['logz_history']
            
            # Restore GP hyperparameter tracking data
            if 'gp_hyperparams' in results_dict:
                gp_data = results_dict['gp_hyperparams']
                results.gp_iterations = gp_data.get('iterations', [])
                results.gp_lengthscales = gp_data.get('lengthscales', [])
                results.gp_kernel_variances = gp_data.get('kernel_variances', [])
                # Backward compatibility: check for old 'outputscales' key
                if 'outputscales' in gp_data and not results.gp_kernel_variances:
                    results.gp_kernel_variances = gp_data.get('outputscales', [])
            
            # Restore acquisition function tracking data
            if 'acquisition_data' in results_dict:
                acq_data = results_dict['acquisition_data']
                results.acquisition_iterations = acq_data.get('iterations', [])
                results.acquisition_values = acq_data.get('values', [])
                results.acquisition_functions = acq_data.get('functions', [])
            
            # Restore best loglikelihood tracking data
            if 'best_loglike_data' in results_dict:
                loglike_data = results_dict['best_loglike_data']
                results.best_loglike_iterations = loglike_data.get('iterations', [])
                results.best_loglike_values = loglike_data.get('best_loglike', [])
            
            # Restore KL divergence tracking data
            if 'kl_data' in results_dict:
                kl_data = results_dict['kl_data']
                results.kl_iterations = kl_data.get('iterations', [])
                results.kl_divergences = kl_data.get('kl_divergences', [])
                results.successive_kl = kl_data.get('successive_kl', [])
            
            # Restore GP and classifier info
            if 'gp_info' in results_dict:
                results.gp_info = results_dict['gp_info']
            
            # Restore timing information
            if 'timing' in results_dict and 'phase_times' in results_dict['timing']:
                for phase, prev_time in results_dict['timing']['phase_times'].items():
                    if phase in results.phase_times:
                        results.phase_times[phase] = prev_time
            
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


def create_resumable_results(output_file: str,
                            param_names: List[str],
                            param_labels: List[str],
                            param_bounds: np.ndarray,
                            settings: Optional[Dict[str, Any]] = None,
                            likelihood_name: str = "unknown") -> BOBEResults:
    """
    Create a BOBEResults manager that automatically resumes from existing results if available.
    
    Args:
        output_file: Base name for output files
        param_names: List of parameter names
        param_labels: List of parameter LaTeX labels
        param_bounds: Parameter bounds array [n_params, 2]
        settings: Dictionary of BOBE settings
        likelihood_name: Name of the likelihood function
        
    Returns:
        BOBEResults object, either fresh or resumed from existing data
    """
    return BOBEResults(
        output_file=output_file,
        param_names=param_names,
        param_labels=param_labels,
        param_bounds=param_bounds,
        settings=settings,
        likelihood_name=likelihood_name,
        resume_from_existing=True
    )
