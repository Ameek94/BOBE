"""
Results management system for BOBE sampler.

This module provides comprehensive result storage and formatting similar to 
typical nested samplers like Dynesty, PolyChord, MultiNest, etc.
"""

import os
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

from .log import get_logger

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
                 param_names: List[str],
                 param_labels: List[str],
                 param_bounds: np.ndarray,
                 output_file: str = 'results',
                 save_dir: Optional[str] = './',
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
        self.output_file = output_file or 'results'
        self.save_dir = save_dir or './'
        self.save_path = os.path.join(self.save_dir, output_file)
        self.param_names = param_names
        self.param_labels = param_labels
        self.param_bounds = np.array(param_bounds)
        self.ndim = len(param_names)
        self.likelihood_name = likelihood_name
        
        # Store settings
        self.settings = settings or {}
        
        self._initialize_fresh()
        log.info(f"Initialized BOBE results manager for {self.ndim}D problem")
    
    def _initialize_fresh(self):
        """Initialize all tracking variables for a fresh run."""
        # Initialize timing variables
        self.start_time = time.time()
        self.end_time = None
        self.previous_runtime = 0.0  # Track previously elapsed time from resumed runs
        
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
            'MCMC Sampling': 0.0,
        }

        if 'use_clf' in self.settings and self.settings['use_clf']:
            self.phase_times['Classifier Training'] = 0.0

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
        
        # Best point information (for getdist minimum files)
        self.best_point = None
        self.best_loglike = None
        self.best_iteration = None
    
    def restore_from_checkpoint(self, data: Dict[str, Any]):
        """Restore tracking state from a checkpoint dict (loaded from .pkl by bo._handle_resume)."""
        self._initialize_fresh()

        if 'convergence_history' in data:
            self.convergence_history = []
            for conv_dict in data['convergence_history']:
                self.convergence_history.append(ConvergenceInfo(
                    iteration=conv_dict['iteration'],
                    logz_dict=conv_dict['logz_dict'],
                    converged=conv_dict['converged'],
                    delta=conv_dict['delta'],
                    threshold=conv_dict['threshold'],
                    dlogz_sampler=conv_dict['dlogz_sampler'],
                ))

        if 'logz_evolution' in data:
            self.logz_evolution = list(data['logz_evolution'])

        acq = data.get('acquisition_data', {})
        self.acquisition_iterations = list(acq.get('iterations', []))
        self.acquisition_values     = list(acq.get('values', []))
        self.acquisition_functions  = list(acq.get('functions', []))

        gph = data.get('gp_hyperparams', {})
        self.gp_iterations      = list(gph.get('iterations', []))
        self.gp_lengthscales    = list(gph.get('lengthscales', []))
        self.gp_kernel_variances = list(gph.get('kernel_variances', []))

        bll = data.get('best_loglike_data', {})
        self.best_loglike_iterations = list(bll.get('iterations', []))
        self.best_loglike_values     = list(bll.get('best_loglike', []))

        kl = data.get('kl_data', {})
        self.kl_iterations  = list(kl.get('iterations', []))
        self.kl_divergences = list(kl.get('kl_divergences', []))
        self.successive_kl  = list(kl.get('successive_kl', []))

        pt = data.get('phase_times', {})
        for phase, val in pt.items():
            if phase in self.phase_times:
                self.phase_times[phase] = float(val)

        self.previous_runtime = float(data.get('previous_runtime', 0.0))
        self.gp_info = dict(data.get('gp_info', {}))

        log.info(f"Restored results state: {len(self.convergence_history)} convergence entries, "
                 f"{len(self.acquisition_values)} acq values")

    def save_checkpoint(self, run_state: Dict[str, Any]):
        """Merge BO run state with results tracking state and write a single .pkl checkpoint."""
        data = {'settings': self.settings, **run_state}
        data.update(self.get_state_dict())
        pkl_path = self.save_path + '.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.debug(f"Checkpoint written to {pkl_path}")

    def load_checkpoint(self, resume_file: str) -> Optional[Dict[str, Any]]:
        """
        Load a .pkl checkpoint, restore results tracking state, and return the raw dict
        so that bo._handle_resume can unpack GP/transform/counter fields.
        Returns None if the file is missing or unreadable (caller should start fresh).
        """
        pkl_path = resume_file + '.pkl'
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            self.restore_from_checkpoint(data)
            log.info(f"Loaded checkpoint from {pkl_path}")
            return data
        except FileNotFoundError:
            log.warning(f"{pkl_path} not found; starting fresh")
            return None
        except Exception as e:
            log.error(f"Failed to load checkpoint {pkl_path}: {e}; starting fresh")
            return None

    def get_last_convergence_state(self) -> Dict[str, Any]:
        """Return the converged/delta/threshold fields from the most recent convergence entry."""
        if self.convergence_history:
            last = self.convergence_history[-1]
            return {'converged': last.converged, 'delta': last.delta, 'threshold': last.threshold}
        return {'converged': False, 'delta': None, 'threshold': None}

    def get_state_dict(self) -> Dict[str, Any]:
        """Return all tracking lists/dicts as a serialisable dict for the checkpoint pkl."""
        timing = self.get_timing_summary()
        return {
            'convergence_history': [c.to_dict() for c in self.convergence_history],
            'logz_evolution':      self.logz_evolution,
            'acquisition_data': {
                'iterations': self.acquisition_iterations,
                'values':     self.acquisition_values,
                'functions':  self.acquisition_functions,
            },
            'gp_hyperparams': {
                'iterations':      self.gp_iterations,
                'lengthscales':    convert_jax_to_json_serializable(self.gp_lengthscales),
                'kernel_variances': convert_jax_to_json_serializable(self.gp_kernel_variances),
            },
            'best_loglike_data': {
                'iterations':  self.best_loglike_iterations,
                'best_loglike': self.best_loglike_values,
            },
            'kl_data': {
                'iterations':    self.kl_iterations,
                'kl_divergences': convert_jax_to_json_serializable(self.kl_divergences),
                'successive_kl':  convert_jax_to_json_serializable(self.successive_kl),
            },
            'phase_times':     timing['phase_times'],
            'previous_runtime': timing['total_runtime'],
            'gp_info':         self.gp_info,
        }

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
                          threshold: float,
                          delta: float = None):
        """
        Update convergence information from a nested sampling check.
        
        Args:
            iteration: Current iteration number
            logz_dict: Dictionary with logz information
            converged: Whether convergence was achieved
            threshold: Convergence threshold used
            delta: The convergence metric value (must match what was used for the
                   convergence check so that resume comparisons are correct).
                   Defaults to logz_dict['std'] if not provided.
        """
        if delta is None:
            delta = logz_dict.get('std', np.nan)
        
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
        current_session_runtime = (self.end_time or time.time()) - self.start_time
        total_runtime = self.previous_runtime + current_session_runtime
        
        # Calculate percentages
        percentages = {}
        if total_runtime > 0:
            for phase, time_spent in self.phase_times.items():
                percentages[phase] = (time_spent / total_runtime) * 100
        
        return {
            'phase_times': self.phase_times.copy(),
            'percentages': percentages,
            'total_runtime': total_runtime,
            'current_session_runtime': current_session_runtime,
            'previous_runtime': self.previous_runtime
        }
    
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
                 samples_dict: Dict[str, np.ndarray] = {},
                 logz_dict: Optional[Dict[str, float]] = None,
                 converged: bool = False,
                 termination_reason: str = "Max iterations reached",
                 gp_info: Optional[Dict[str, Any]] = None,
                 best_point: Optional[np.ndarray] = None,
                 best_loglike: Optional[float] = None,
                 best_iteration: Optional[int] = None):
        """
        Finalize the results with final samples and metadata.
        
        Args:
            samples_dict: Dictionary with 'x', 'weights', 'logl' keys for final samples
            logz_dict: Final evidence information
            converged: Whether the run converged
            termination_reason: Reason for termination
            gp_info: Dictionary containing GP and classifier information
            best_point: Best point found (physical parameter space)
            best_loglike: Best log-likelihood value
            best_iteration: Iteration where best point was found
        """
        self.end_time = time.time()
        
        self.final_samples = samples_dict.get('x', np.array([]))
        self.final_weights = samples_dict.get('weights', np.array([]))
        self.final_loglikes = samples_dict.get('logl', np.array([]))


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
        
        # Store best point information
        self.best_point = best_point
        self.best_loglike = best_loglike
        self.best_iteration = best_iteration

        log.info(f"Finalized BOBE results: converged={converged}, reason={termination_reason}")
        if best_point is not None and best_loglike is not None:
            log.info(f"Best point: logL={best_loglike:.6f} at iteration {best_iteration}")

        # Save all results
        self.save_chain_files()
        self.save_summary_json()
        self.save_minimum_files()
    
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
        
        # Runtime - use timing summary for accurate total runtime calculation
        timing_summary = self.get_timing_summary()
        runtime = timing_summary['total_runtime']
        
        results = {
            # === SAMPLES AND WEIGHTS ===
            'samples': self.final_samples,
            'weights': self.final_weights,
            'logl': self.final_loglikes,
            'logwt': np.log(self.final_weights+1e-300) if len(self.final_weights) > 0 else np.array([]),
            
            # === EVIDENCE INFORMATION ===
            'logz': self.final_logz_dict.get('mean', np.nan),
            'logzerr': self.final_logz_dict.get('std', self.final_logz_dict.get('upper', 0) - self.final_logz_dict.get('lower', 0)),
            'dlogz_sampler': float(self.final_logz_dict.get('dlogz_sampler', np.nan)),
            'final_logz_dict': self.final_logz_dict.copy(),  # Preserve full logz_dict including std
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
    
    def save_summary_json(self):
        """Save a single human-readable summary JSON at the end of the run."""
        if len(self.final_samples) == 0:
            return

        timing = self.get_timing_summary()

        # Parameter statistics
        param_stats = {}
        for i, name in enumerate(self.param_names):
            values  = self.final_samples[:, i]
            weights = self.final_weights
            mean = np.average(values, weights=weights)
            std  = np.sqrt(np.average((values - mean)**2, weights=weights))
            sorted_idx = np.argsort(values)
            cumsum = np.cumsum(weights[sorted_idx]) / weights.sum()
            def wp(p, _ci=cumsum, _vs=values[sorted_idx]):
                idx = min(np.searchsorted(_ci, p / 100.0), len(_vs) - 1)
                return float(_vs[idx])
            param_stats[name] = {
                'mean': float(mean), 'std': float(std),
                '16%': wp(16), '84%': wp(84), 'median': wp(50),
            }

        logz = self.final_logz_dict
        summary = {
            'run_info': {
                'likelihood_name': self.likelihood_name,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': (datetime.fromtimestamp(self.end_time).isoformat()
                             if self.end_time else None),
                'runtime_hours': timing['total_runtime'] / 3600,
            },
            'convergence': {
                'converged': bool(self.converged),
                'termination_reason': str(self.termination_reason),
                'n_iterations': len(self.acquisition_iterations),
            },
            'evidence': {
                'logz':       float(logz.get('mean', float('nan'))),
                'logz_err':   float(logz.get('std', float('nan'))),
                'logz_lower': float(logz.get('lower', float('nan'))),
                'logz_upper': float(logz.get('upper', float('nan'))),
            },
            'timing': timing['phase_times'],
            'gp_info': self.gp_info,
            'parameters': param_stats,
        }

        summary_file = f"{self.save_path}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        log.info(f"Saved summary to {summary_file}")

    def save_chain_files(self, samples_dict: Optional[Dict[str, np.ndarray]] = None, filename: Optional[str] = None):
        """Save chain files in GetDist format using MCSamples.saveAsText method."""
        
        if not HAS_GETDIST:
            log.warning("GetDist not available, cannot save chain files")
            return
        
        # Get MCSamples object
        getdist_samples = self.get_getdist_samples(samples_dict)
        if getdist_samples is None:
            log.warning("Could not create MCSamples object")
            return
        
        if filename is not None:
            output_file = os.path.join(self.save_dir, filename)
        else:
            output_file = self.save_path
        
        # Use GetDist's saveAsText method to save the chain files
        # This automatically creates .txt, .paramnames, and .ranges files
        getdist_samples.saveAsText(root=output_file, make_dirs=True)
        log.info(f"Saved GetDist format files to {output_file}")
        log.info("Created: .txt (chain), .paramnames (parameter info), .ranges (parameter bounds)")
    
    def save_minimum_files(self):
        """
        Save best point in GetDist minimum format.
        
        Creates two files:
        - .minimum.txt: Simple table with best point
        - .minimum: Formatted text with parameter details
        """
        if self.best_point is None or self.best_loglike is None:
            log.debug("No best point data available, skipping minimum files")
            return
        
        best_point = np.atleast_1d(self.best_point)
        
        if len(best_point) != self.ndim:
            log.warning(f"Best point dimension {len(best_point)} != {self.ndim}, skipping minimum files")
            return
        
        minuslogpost = -self.best_loglike
        chi_sq = 2.0 * minuslogpost
        
        # Write .minimum.txt file (simple table format)
        minimum_txt_file = f"{self.save_path}.minimum.txt"
        try:
            with open(minimum_txt_file, 'w') as f:
                # Header line
                header = "#        weight    minuslogpost"
                for param_name in self.param_names:
                    header += f"  {param_name:>13s}"
                f.write(header + "\n")
                
                # Data line (weight is always 1 for single best point)
                line = f"              1  {minuslogpost:13.7f}"
                for val in best_point:
                    line += f"  {val:13.8e}"
                f.write(line + "\n")
            
            log.info(f"Saved minimum table to {minimum_txt_file}")
        except Exception as e:
            log.warning(f"Failed to save .minimum.txt file: {e}")
        
        # Write .minimum file (formatted text with labels)
        minimum_file = f"{self.save_path}.minimum"
        try:
            with open(minimum_file, 'w') as f:
                # Header with likelihood info
                f.write(f" -log(Like) = {minuslogpost:.12f}\n")
                f.write(f"  chi-sq    = {chi_sq:.12f}\n")
                f.write("\n")
                
                # Parameter list with index, value, name, and LaTeX label
                for i, (param_name, param_label, val) in enumerate(zip(
                    self.param_names, self.param_labels, best_point), start=1):
                    # Format: index (right-aligned, width 5), value (scientific), name, label
                    f.write(f"{i:>5d}  {val:.9e}   {param_name:40s}  {param_label}\n")
            
            log.info(f"Saved minimum point details to {minimum_file}")
        except Exception as e:
            log.warning(f"Failed to save .minimum file: {e}")
        
    
    def get_getdist_samples(self, samples_dict = None) -> Optional['MCSamples']:
        """
        Convert results to GetDist MCSamples object.
        
        Returns:
            GetDist MCSamples object if GetDist is available, None otherwise
        """
        if not HAS_GETDIST:
            log.warning("GetDist not available, cannot create MCSamples object")
            return None
        
        if samples_dict is not None: # for checkpoint samples
            samples= samples_dict['x']
            weights = samples_dict['weights']
            loglikes = samples_dict['logl']
            sampler_method = samples_dict.get('method','mcmc')
        else: # for final samples
            if self.final_samples is None:
                log.warning("No final samples available")
                return None
            samples = self.final_samples
            weights = self.final_weights
            loglikes = self.final_loglikes
            # Determine sampler method
            sampler_method = 'nested' if self.final_logz_dict else 'mcmc'

        # Check if samples array is empty
        if len(samples) == 0:
            log.warning("Samples array is empty, cannot create MCSamples object")
            return None

        # Parameter ranges for GetDist
        # param_bounds is shape (2, nparams)
        ranges = {name: [self.param_bounds[0, i], self.param_bounds[1, i]] 
                  for i, name in enumerate(self.param_names)}
        
        
        getdist_samples = MCSamples(
            samples=samples,
            names=self.param_names,
            labels=self.param_labels,
            ranges=ranges,
            weights=weights,
            loglikes=loglikes,
            label='BOBE',
            sampler=sampler_method
        )

        return getdist_samples
    
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
            # Try new naming first, fall back to old naming for backward compatibility
            results.final_logz_dict = results_dict.get('final_logz_dict', results_dict.get('logz_bounds', {}))
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