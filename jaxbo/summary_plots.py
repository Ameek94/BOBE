"""
Summary plotting module for JaxBo BOBE runtime visualization.

This module provides comprehensive plotting capabilities for analyzing BOBE runs,
including evidence evolution, GP hyperparameters, timing information, and convergence diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings
from pathlib import Path
import json

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    warnings.warn("Seaborn not available. Using matplotlib defaults.")

try:
    import getdist
    from getdist import plots
    HAS_GETDIST = True
except ImportError:
    HAS_GETDIST = False
    warnings.warn("GetDist not available. Triangle plots will be limited.")

from .results import BOBEResults, load_bobe_results
from .logging_utils import get_logger

log = get_logger("[plots]")

# Set default plotting style
plt.style.use('default')
if HAS_SEABORN:
    sns.set_palette("husl")


class BOBESummaryPlotter:
    """
    Comprehensive plotting class for BOBE run analysis and diagnostics.
    """
    
    def __init__(self, results: Union[BOBEResults, str], figsize_scale: float = 1.0):
        """
        Initialize the plotter with BOBE results.
        
        Args:
            results: BOBEResults object or path to results file
            figsize_scale: Scale factor for figure sizes (default: 1.0)
        """
        if isinstance(results, str):
            self.results = load_bobe_results(results)
            self.output_file = results
        else:
            self.results = results
            self.output_file = results.output_file
            
        self.figsize_scale = figsize_scale
        self.param_names = self.results.param_names
        self.param_labels = self.results.param_labels
        self.ndim = self.results.ndim
        
        log.info(f"Initialized summary plotter for {self.ndim}D problem: {self.output_file}")
    
    def plot_evidence_evolution(self, ax: Optional[plt.Axes] = None, 
                               show_convergence: bool = True) -> plt.Axes:
        """
        Plot the evolution of log evidence (logZ) with error bounds.
        
        Args:
            ax: Matplotlib axes to plot on (creates new if None)
            show_convergence: Whether to mark convergence points
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8*self.figsize_scale, 5*self.figsize_scale))
        
        if not self.results.logz_evolution:
            ax.text(0.5, 0.5, 'No logZ evolution data available', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        # Extract data
        iterations = [entry['iteration'] for entry in self.results.logz_evolution]
        logz_values = [entry['logz'] for entry in self.results.logz_evolution]
        
        # Use actual upper and lower bounds if available, otherwise fall back to symmetric errors
        if self.results.logz_evolution and 'logz_upper' in self.results.logz_evolution[0]:
            logz_upper = [entry['logz_upper'] for entry in self.results.logz_evolution]
            logz_lower = [entry['logz_lower'] for entry in self.results.logz_evolution]
        else:
            # Fallback to symmetric errors for backwards compatibility
            logz_errors = [entry['logz_err'] for entry in self.results.logz_evolution]
            logz_errors = np.array(logz_errors)
            logz_values_arr = np.array(logz_values)
            logz_upper = logz_values_arr + logz_errors
            logz_lower = logz_values_arr - logz_errors
        
        # Convert to numpy arrays
        iterations = np.array(iterations)
        logz_values = np.array(logz_values)
        logz_upper = np.array(logz_upper)
        logz_lower = np.array(logz_lower)
        
        # Plot mean line
        ax.plot(iterations, logz_values, 'b-', linewidth=2, label='Mean log Z', alpha=0.9)
        
        # Plot upper and lower bounds
        ax.plot(iterations, logz_upper, 'r--', linewidth=1.5, alpha=0.7, label='Upper bound')
        ax.plot(iterations, logz_lower, 'g--', linewidth=1.5, alpha=0.7, label='Lower bound')
        
        # Shade the region between upper and lower bounds
        ax.fill_between(iterations, logz_lower, logz_upper, 
                       alpha=0.2, color='blue', label='Uncertainty region')
        
        # Mark convergence points if requested
        if show_convergence and self.results.convergence_history:
            conv_iterations = [conv.iteration for conv in self.results.convergence_history if conv.converged]
            if conv_iterations:
                for conv_iter in conv_iterations:
                    # Find corresponding logZ value
                    idx = np.searchsorted(iterations, conv_iter)
                    if idx < len(logz_values):
                        ax.axvline(conv_iter, color='red', linestyle='--', alpha=0.7)
                        ax.scatter(conv_iter, logz_values[idx], color='red', s=50, 
                                 marker='o', zorder=5, label='Convergence' if conv_iter == conv_iterations[0] else "")
        
        # Final logZ if available
        if self.results.final_logz_dict:
            final_logz = self.results.final_logz_dict.get('mean', np.nan)
            if not np.isnan(final_logz):
                ax.axhline(final_logz, color='green', linestyle='-', alpha=0.7, 
                          linewidth=2, label=f'Final log Z = {final_logz:.3f}')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('log Z')
        ax.set_title('Evidence Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_gp_lengthscales(self, gp_data: Optional[Dict] = None, 
                            ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot evolution of GP lengthscales only.
        
        Args:
            gp_data: Dictionary containing GP hyperparameter evolution data
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10*self.figsize_scale, 6*self.figsize_scale))
        
        if gp_data is None:
            ax.text(0.5, 0.5, 'No GP lengthscale data provided\n'
                   'Pass gp_data dictionary with evolution info', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        # Extract hyperparameter evolution
        if 'iterations' not in gp_data or 'lengthscales' not in gp_data:
            ax.text(0.5, 0.5, 'Invalid GP data format\n'
                   'Need "iterations" and "lengthscales" keys', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        iterations = np.array(gp_data['iterations'])
        lengthscales = np.array(gp_data['lengthscales'])  # Shape: [n_iterations, n_params]
        
        # Plot lengthscales for each parameter
        colors = plt.cm.Set1(np.linspace(0, 1, self.ndim))
        for i in range(self.ndim):
            if i < lengthscales.shape[1]:
                ax.plot(iterations, lengthscales[:, i], 
                       color=colors[i], linewidth=2, 
                       label=f'{self.param_labels[i]}')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Lengthscale')
        ax.set_title('GP Lengthscale Evolution')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_gp_outputscale(self, gp_data: Optional[Dict] = None, 
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot evolution of GP outputscale only.
        
        Args:
            gp_data: Dictionary containing GP hyperparameter evolution data
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10*self.figsize_scale, 6*self.figsize_scale))
        
        if gp_data is None:
            ax.text(0.5, 0.5, 'No GP outputscale data provided\n'
                   'Pass gp_data dictionary with evolution info', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        # Extract outputscale evolution
        if 'iterations' not in gp_data or 'outputscales' not in gp_data:
            ax.text(0.5, 0.5, 'Invalid GP data format\n'
                   'Need "iterations" and "outputscales" keys', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        iterations = np.array(gp_data['iterations'])
        outputscales = np.array(gp_data['outputscales'])
        
        # Plot outputscale
        ax.plot(iterations, outputscales, 'purple', linewidth=2, 
               label='Output scale', alpha=0.8)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Output Scale')
        ax.set_title('GP Output Scale Evolution')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_gp_hyperparameters(self, gp_data: Optional[Dict] = None, 
                               ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot evolution of GP hyperparameters (backward compatibility - now plots lengthscales only).
        
        Args:
            gp_data: Dictionary containing GP hyperparameter evolution data
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        # For backward compatibility, this now calls the lengthscales plot
        return self.plot_gp_lengthscales(gp_data=gp_data, ax=ax)
    
    def plot_best_loglike_evolution(self, best_loglike_data: Optional[Dict] = None,
                                   ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot evolution of the best log-likelihood found so far.
        
        Args:
            best_loglike_data: Dictionary with 'iterations' and 'best_loglike' keys
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8*self.figsize_scale, 5*self.figsize_scale))
        
        if best_loglike_data is None:
            ax.text(0.5, 0.5, 'No best log-likelihood data provided\n'
                   'Pass best_loglike_data dictionary', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        if 'iterations' not in best_loglike_data or 'best_loglike' not in best_loglike_data:
            ax.text(0.5, 0.5, 'Invalid best log-likelihood data format\n'
                   'Need "iterations" and "best_loglike" keys', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        iterations = np.array(best_loglike_data['iterations'])
        best_loglike = np.array(best_loglike_data['best_loglike'])
        
        # Plot evolution
        ax.plot(iterations, best_loglike, 'g-', linewidth=2, 
               label='Best log-likelihood', alpha=0.8)
        
        # Mark improvements
        improvements = np.diff(best_loglike) > 0
        if np.any(improvements):
            improve_iter = iterations[1:][improvements]
            improve_vals = best_loglike[1:][improvements]
            ax.scatter(improve_iter, improve_vals, color='red', s=30, 
                      marker='o', alpha=0.6, label='Improvements')
        
        # Final value
        if len(best_loglike) > 0:
            final_val = best_loglike[-1]
            ax.axhline(final_val, color='green', linestyle='--', alpha=0.5,
                      label=f'Final best = {final_val:.3f}')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best log-likelihood')
        ax.set_title('Best Log-likelihood Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_timing_breakdown(self, timing_data: Optional[Dict] = None,
                             ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot timing breakdown of different phases of the algorithm.
        
        Args:
            timing_data: Dictionary with timing information for different phases
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8*self.figsize_scale, 6*self.figsize_scale))
        
        if timing_data is None:
            # Use basic timing info from results
            total_time = (self.results.end_time - self.results.start_time) if self.results.end_time else 0
            timing_data = {'Total Runtime': total_time}
        
        # Handle new timing data structure from BOBEResults
        if isinstance(timing_data, dict) and 'phase_times' in timing_data:
            # Extract phase times from new structure
            phase_times = timing_data['phase_times']
            # Only include phases that have time > 0
            timing_data = {phase: time for phase, time in phase_times.items() if time > 0}
        
        # Create bar plot
        phases = list(timing_data.keys())
        times = list(timing_data.values())
        
        if len(phases) == 1:
            # Simple total time display
            ax.bar([0], times, color='skyblue', alpha=0.7)
            ax.set_xticks([0])
            ax.set_xticklabels(phases)
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Runtime Information')
        else:
            # Multiple phases
            colors = plt.cm.Set3(np.linspace(0, 1, len(phases)))
            bars = ax.bar(range(len(phases)), times, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                       f'{time_val:.1f}s', ha='center', va='bottom')
            
            ax.set_xticks(range(len(phases)))
            ax.set_xticklabels(phases, rotation=45, ha='right')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Runtime Breakdown')
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        return ax
    
    def plot_convergence_diagnostics(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot convergence diagnostics including thresholds and delta evolution.
        
        Args:
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8*self.figsize_scale, 5*self.figsize_scale))
        
        if not self.results.convergence_history:
            ax.text(0.5, 0.5, 'No convergence history available', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        # Extract convergence data
        iterations = [conv.iteration for conv in self.results.convergence_history]
        deltas = [conv.delta for conv in self.results.convergence_history]
        thresholds = [conv.threshold for conv in self.results.convergence_history]
        converged_flags = [conv.converged for conv in self.results.convergence_history]
        
        # Plot delta evolution
        ax.plot(iterations, deltas, 'b-', linewidth=2, label='Δlog Z', alpha=0.8)
        
        # Plot threshold
        ax.plot(iterations, thresholds, 'r--', linewidth=2, label='Threshold', alpha=0.7)
        
        # Mark convergence points
        conv_iterations = [it for it, conv in zip(iterations, converged_flags) if conv]
        conv_deltas = [delta for delta, conv in zip(deltas, converged_flags) if conv]
        
        if conv_iterations:
            ax.scatter(conv_iterations, conv_deltas, color='green', s=50, 
                      marker='o', zorder=5, label='Converged points')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Δlog Z')
        ax.set_title('Convergence Diagnostics')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_parameter_evolution(self, param_evolution_data: Optional[Dict] = None,
                                max_params: int = 4) -> plt.Figure:
        """
        Plot evolution of parameter values during optimization.
        
        Args:
            param_evolution_data: Dictionary with parameter evolution data
            max_params: Maximum number of parameters to plot
            
        Returns:
            The matplotlib figure object
        """
        n_plot = min(self.ndim, max_params)
        fig, axes = plt.subplots(n_plot, 1, figsize=(8*self.figsize_scale, 3*n_plot*self.figsize_scale))
        
        if n_plot == 1:
            axes = [axes]
        
        if param_evolution_data is None:
            for i, ax in enumerate(axes):
                ax.text(0.5, 0.5, f'No evolution data for {self.param_names[i]}', 
                       transform=ax.transAxes, ha='center', va='center')
            return fig
        
        for i in range(n_plot):
            param_name = self.param_names[i]
            if param_name in param_evolution_data:
                data = param_evolution_data[param_name]
                iterations = data.get('iterations', [])
                values = data.get('values', [])
                
                if iterations and values:
                    axes[i].plot(iterations, values, 'o-', linewidth=1, markersize=3, alpha=0.7)
                    axes[i].set_ylabel(f'{self.param_labels[i]}')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add parameter bounds if available
                    if hasattr(self.results, 'param_bounds') and i < len(self.results.param_bounds[0]):
                        lower = self.results.param_bounds[0, i]
                        upper = self.results.param_bounds[1, i]
                        axes[i].axhline(lower, color='red', linestyle=':', alpha=0.5, label='Bounds')
                        axes[i].axhline(upper, color='red', linestyle=':', alpha=0.5)
                        axes[i].legend()
                else:
                    axes[i].text(0.5, 0.5, f'No data for {param_name}', 
                               transform=axes[i].transAxes, ha='center', va='center')
            else:
                axes[i].text(0.5, 0.5, f'No data for {param_name}', 
                           transform=axes[i].transAxes, ha='center', va='center')
        
        axes[-1].set_xlabel('Iteration')
        plt.tight_layout()
        
        return fig
    
    def create_summary_dashboard(self, 
                                gp_data: Optional[Dict] = None,
                                best_loglike_data: Optional[Dict] = None,
                                timing_data: Optional[Dict] = None,
                                param_evolution_data: Optional[Dict] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive summary dashboard with all diagnostic plots.
        
        Args:
            gp_data: GP hyperparameter evolution data
            best_loglike_data: Best log-likelihood evolution data
            timing_data: Timing breakdown data
            param_evolution_data: Parameter evolution data (deprecated - not used)
            save_path: Path to save the figure (optional)
            
        Returns:
            The matplotlib figure object
        """
        # Create figure with subplots (2x3 grid)
        fig = plt.figure(figsize=(18*self.figsize_scale, 12*self.figsize_scale))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Evidence evolution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_evidence_evolution(ax=ax1)
        
        # GP lengthscales (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_gp_lengthscales(gp_data=gp_data, ax=ax2)
        
        # GP outputscale (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_gp_outputscale(gp_data=gp_data, ax=ax3)
        
        # Best log-likelihood (bottom left)
        ax4 = fig.add_subplot(gs[1, 0])
        self.plot_best_loglike_evolution(best_loglike_data=best_loglike_data, ax=ax4)
        
        # Convergence diagnostics (bottom middle)
        ax5 = fig.add_subplot(gs[1, 1])
        self.plot_convergence_diagnostics(ax=ax5)
        
        # Timing breakdown (bottom right)
        ax6 = fig.add_subplot(gs[1, 2])
        self.plot_timing_breakdown(timing_data=timing_data, ax=ax6)
        
        # Add overall title
        fig.suptitle(f'BOBE Summary Dashboard: {self.output_file}', 
                    fontsize=16*self.figsize_scale, y=0.95)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            log.info(f"Saved summary dashboard to {save_path}")
        
        return fig
    
    def plot_summary_stats(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot key summary statistics as text.
        
        Args:
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6*self.figsize_scale, 4*self.figsize_scale))
        
        # Collect summary statistics
        stats_text = []
        
        # Basic info
        stats_text.append(f"Problem: {self.ndim}D")
        stats_text.append(f"Likelihood: {self.results.likelihood_name}")
        
        # Samples info
        if self.results.final_samples is not None:
            n_samples = len(self.results.final_samples)
            stats_text.append(f"Final samples: {n_samples}")
            
            if len(self.results.final_weights) > 0:
                n_eff = int(np.sum(self.results.final_weights)**2 / np.sum(self.results.final_weights**2))
                stats_text.append(f"Effective samples: {n_eff}")
        
        # Evidence
        if self.results.final_logz_dict:
            logz = self.results.final_logz_dict.get('mean', np.nan)
            logz_err = (self.results.final_logz_dict.get('upper', 0) - 
                       self.results.final_logz_dict.get('lower', 0))
            if not np.isnan(logz):
                stats_text.append(f"log Z = {logz:.3f} ± {logz_err:.3f}")
        
        # Runtime
        if self.results.end_time:
            runtime = self.results.end_time - self.results.start_time
            runtime_str = f"{runtime/3600:.2f} hours" if runtime > 3600 else f"{runtime:.1f} seconds"
            stats_text.append(f"Runtime: {runtime_str}")
        
        # Convergence
        stats_text.append(f"Converged: {'Yes' if self.results.converged else 'No'}")
        stats_text.append(f"Termination: {self.results.termination_reason}")
        
        # Display text
        ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes, 
               fontsize=11*self.figsize_scale, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Run Summary')
        
        return ax
    
    def plot_parameter_traces(self, param_evolution_data: Dict, 
                             ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot parameter traces for all parameters in a single plot.
        
        Args:
            param_evolution_data: Dictionary with parameter evolution data
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12*self.figsize_scale, 4*self.figsize_scale))
        
        colors = plt.cm.Set1(np.linspace(0, 1, self.ndim))
        
        for i, param_name in enumerate(self.param_names):
            if param_name in param_evolution_data:
                data = param_evolution_data[param_name]
                iterations = data.get('iterations', [])
                values = data.get('values', [])
                
                if iterations and values:
                    # Normalize values to [0, 1] for display
                    if hasattr(self.results, 'param_bounds') and i < len(self.results.param_bounds[0]):
                        lower = self.results.param_bounds[0, i]
                        upper = self.results.param_bounds[1, i]
                        norm_values = (np.array(values) - lower) / (upper - lower)
                    else:
                        norm_values = np.array(values)
                    
                    ax.plot(iterations, norm_values, color=colors[i], 
                           linewidth=1, alpha=0.7, label=self.param_labels[i])
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Normalized Parameter Value')
        ax.set_title('Parameter Evolution (Normalized)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_final_parameter_summary(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot final parameter values and uncertainties.
        
        Args:
            ax: Matplotlib axes to plot on (creates new if None)
            
        Returns:
            The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10*self.figsize_scale, 4*self.figsize_scale))
        
        if self.results.final_samples is None:
            ax.text(0.5, 0.5, 'No final samples available', 
                   transform=ax.transAxes, ha='center', va='center')
            return ax
        
        # Calculate parameter statistics
        param_means = []
        param_stds = []
        
        for i in range(self.ndim):
            values = self.results.final_samples[:, i]
            weights = self.results.final_weights
            
            mean = np.average(values, weights=weights)
            var = np.average((values - mean)**2, weights=weights)
            std = np.sqrt(var)
            
            param_means.append(mean)
            param_stds.append(std)
        
        # Create error bar plot
        x_pos = np.arange(self.ndim)
        ax.errorbar(x_pos, param_means, yerr=param_stds, 
                   fmt='o', markersize=8, capsize=5, capthick=2)
        
        # Add parameter bounds if available
        if hasattr(self.results, 'param_bounds'):
            for i in range(self.ndim):
                lower = self.results.param_bounds[0, i]
                upper = self.results.param_bounds[1, i]
                ax.axhline(lower, color='red', linestyle=':', alpha=0.3)
                ax.axhline(upper, color='red', linestyle=':', alpha=0.3)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.param_labels)
        ax.set_ylabel('Parameter Value')
        ax.set_title('Final Parameter Estimates')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def save_all_plots(self, output_dir: Optional[str] = None, **data_kwargs):
        """
        Save all individual plots and the summary dashboard.
        
        Args:
            output_dir: Directory to save plots (uses output_file base if None)
            **data_kwargs: Data dictionaries for different plot types
        """
        if output_dir is None:
            output_dir = Path(self.output_file).parent
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        base_name = Path(self.output_file).stem
        
        # Individual plots
        plots_to_save = [
            ('evidence_evolution', self.plot_evidence_evolution),
            ('convergence_diagnostics', self.plot_convergence_diagnostics),
        ]
        
        # Optional plots with data
        if 'gp_data' in data_kwargs:
            plots_to_save.append(('gp_lengthscales', 
                                lambda ax: self.plot_gp_lengthscales(data_kwargs['gp_data'], ax=ax)))
            plots_to_save.append(('gp_outputscale', 
                                lambda ax: self.plot_gp_outputscale(data_kwargs['gp_data'], ax=ax)))
        
        if 'best_loglike_data' in data_kwargs:
            plots_to_save.append(('best_loglike_evolution', 
                                lambda ax: self.plot_best_loglike_evolution(data_kwargs['best_loglike_data'], ax=ax)))
        
        if 'timing_data' in data_kwargs:
            plots_to_save.append(('timing_breakdown', 
                                lambda ax: self.plot_timing_breakdown(data_kwargs['timing_data'], ax=ax)))
        
        # Save individual plots
        for plot_name, plot_func in plots_to_save:
            fig, ax = plt.subplots(1, 1, figsize=(8*self.figsize_scale, 6*self.figsize_scale))
            plot_func(ax=ax)
            plt.tight_layout()
            save_path = output_dir / f"{base_name}_{plot_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            log.info(f"Saved {plot_name} to {save_path}")
        
        # Save summary dashboard
        dashboard_fig = self.create_summary_dashboard(**data_kwargs)
        dashboard_path = output_dir / f"{base_name}_summary_dashboard.png"
        dashboard_fig.savefig(dashboard_path, dpi=150, bbox_inches='tight')
        plt.close(dashboard_fig)
        log.info(f"Saved summary dashboard to {dashboard_path}")


def create_summary_plots(results_file: str, 
                        gp_data: Optional[Dict] = None,
                        best_loglike_data: Optional[Dict] = None,
                        timing_data: Optional[Dict] = None,
                        param_evolution_data: Optional[Dict] = None,
                        output_dir: Optional[str] = None,
                        figsize_scale: float = 1.0) -> BOBESummaryPlotter:
    """
    Convenience function to create all summary plots for a BOBE run.
    
    Args:
        results_file: Path to BOBE results file (without extension)
        gp_data: GP hyperparameter evolution data
        best_loglike_data: Best log-likelihood evolution data  
        timing_data: Timing breakdown data
        param_evolution_data: Parameter evolution data
        output_dir: Directory to save plots
        figsize_scale: Scale factor for figure sizes
        
    Returns:
        BOBESummaryPlotter object
    """
    plotter = BOBESummaryPlotter(results_file, figsize_scale=figsize_scale)
    
    # Create and save all plots
    plotter.save_all_plots(
        output_dir=output_dir,
        gp_data=gp_data,
        best_loglike_data=best_loglike_data,
        timing_data=timing_data,
        param_evolution_data=param_evolution_data
    )
    
    return plotter


# Data format documentation for users
def get_data_format_examples() -> Dict[str, Dict]:
    """
    Return example data formats for the plotting functions.
    
    Returns:
        Dictionary with example data structures
    """
    examples = {
        'gp_data': {
            'iterations': [10, 20, 30, 40, 50],
            'lengthscales': [[1.0, 0.5], [0.8, 0.6], [0.7, 0.7], [0.6, 0.8], [0.5, 0.9]],
            'outputscales': [2.0, 1.8, 1.6, 1.4, 1.2]
        },
        'best_loglike_data': {
            'iterations': [1, 5, 10, 15, 20],
            'best_loglike': [-10.0, -8.5, -7.2, -6.8, -6.5]
        },
        'timing_data': {
            'GP Training': 45.2,
            'Nested Sampling': 120.8,
            'Optimization': 30.1,
            'I/O Operations': 5.3
        },
        'param_evolution_data': {
            'x1': {
                'iterations': [1, 5, 10, 15, 20],
                'values': [0.1, 0.3, 0.5, 0.4, 0.45]
            },
            'x2': {
                'iterations': [1, 5, 10, 15, 20], 
                'values': [-0.2, 0.0, 0.2, 0.15, 0.18]
            }
        }
    }
    
    return examples
