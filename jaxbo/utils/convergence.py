import numpy as np
import scipy.stats as stats

class ConvergenceChecker:

    def __init__(self, logz_threshold = 1., kl_div_threshold = 0.1):
        """
        Convergence checker using existing nested sampling results.

        Parameters:
        -----------
        gp_model : trained GP model with predict() method
        """
        # self.gp = gp_model
        # self.bounds = parameter_bounds
        # self.n_dims = len(parameter_bounds)
        
        # Default convergence thresholds
        self.logz_threshold = logz_threshold  # Coefficient of variation
        self.kl_div_threshold = kl_div_threshold  # KL divergence bound
        self.min_iterations = 10           # Minimum iterations before checking
        
        # History tracking
        self.history = []
        self.convergence_status = []
        self.prev_mean_samples = None
        self.prev_mean_weights = None   

    def _compute_all_convergence_metrics(self,gp,ns_samples):

        self.logz_results = self._compute_evidence_metrics()
        self.kl_results = self._compute_kl_divergences()

    def _compute_evidence_metrics(self, iteration_data):
        """Compute evidence-based convergence metrics."""
        logz_values = np.array([
            iteration_data['logz_lower'],
            iteration_data['logz_mean'],
            iteration_data['logz_upper']
        ])
        
        logz_std = np.std(logz_values)
        logz_mean_val = np.mean(logz_values)
        logz_cv = logz_std / (np.abs(logz_mean_val) + 1e-12)
        
        return {
            'logz_std': logz_std,
            'logz_cv': logz_cv,
            'logz_range': logz_values[2] - logz_values[0],  # upper - lower
            'logz_uncertainty': logz_std  # same as std
        }

    def _compute_kl_divergences(self, mean_ll, upper_ll, lower_ll, log_weights):
        """
        Compute KL divergences between all combinations.
        
        Returns:
        --------
        dict with KL divergence results
        """
        # Convert to probability distributions
        # Normalize using log weights
        weights = np.exp(log_weights - np.max(log_weights))
        weights = weights / np.sum(weights)
        
        # Normalize likelihoods relative to their max for numerical stability
        mean_norm = mean_ll - np.max(mean_ll)
        upper_norm = upper_ll - np.max(upper_ll)
        lower_norm = lower_ll - np.max(lower_ll)
        
        # Convert to probabilities
        p_mean = np.exp(mean_norm) * weights
        p_upper = np.exp(upper_norm) * weights
        p_lower = np.exp(lower_norm) * weights
        
        # Normalize probability distributions
        p_mean = p_mean / np.sum(p_mean)
        p_upper = p_upper / np.sum(p_upper)
        p_lower = p_lower / np.sum(p_lower)
        
        # Compute KL divergences
        kl_results = {}
        
        # KL between mean and bounds
        kl_results['mean_upper'] = stats.entropy(p_mean, p_upper)
        kl_results['mean_lower'] = stats.entropy(p_mean, p_lower)
        kl_results['upper_mean'] = stats.entropy(p_upper, p_mean)
        kl_results['lower_mean'] = stats.entropy(p_lower, p_mean)
        
        # KL between bounds
        kl_results['upper_lower'] = stats.entropy(p_upper, p_lower)
        kl_results['lower_upper'] = stats.entropy(p_lower, p_upper)
        
        # Symmetrized versions
        kl_results['sym_mean_upper'] = 0.5 * (kl_results['mean_upper'] + kl_results['upper_mean'])
        kl_results['sym_mean_lower'] = 0.5 * (kl_results['mean_lower'] + kl_results['lower_mean'])
        kl_results['sym_upper_lower'] = 0.5 * (kl_results['upper_lower'] + kl_results['lower_upper'])
        
        return kl_results
    
    def _compute_successive_kl(self, prev_loglike, curr_loglike, log_weights):
        """Compute KL divergence between successive iterations."""
        # Convert to probability distributions
        weights = np.exp(log_weights - np.max(log_weights))
        weights = weights / np.sum(weights)
        
        prev_norm = prev_loglike - np.max(prev_loglike)
        curr_norm = curr_loglike - np.max(curr_loglike)
        
        p_prev = np.exp(prev_norm) * weights
        p_curr = np.exp(curr_norm) * weights
        
        p_prev = p_prev / np.sum(p_prev)
        p_curr = p_curr / np.sum(p_curr)
        
        # Forward and reverse KL
        kl_forward = stats.entropy(p_prev, p_curr)
        kl_reverse = stats.entropy(p_curr, p_prev)
        kl_sym = 0.5 * (kl_forward + kl_reverse)
        
        return {
            'forward': kl_forward,
            'reverse': kl_reverse,
            'symmetric': kl_sym
        }


    def update_results(self, ns_results, iteration):
        """
        Update convergence checker with nested sampling results.
        
        Parameters:
        -----------
        ns_results : dict with 'samples', 'weights', 'logz_dict'
        iteration : current iteration number
        """
        # Extract nested sampling results
        samples = ns_results['samples']  # (n_samples, n_dims)
        log_weights = ns_results['weights']  # (n_samples,)
        log_evidence = ns_results['logz']  # scalar
        
        # Evaluate GP at nested sampling samples
        mean_loglike, var_pred = self.gp.predict(samples, return_std=False, return_var=True)
        std_pred = np.sqrt(var_pred)
        
        # GP bounds at samples
        upper_loglike = mean_loglike + std_pred
        lower_loglike = mean_loglike - std_pred
        
        # Compute evidence bounds using same samples/log_weights
        logz_upper = self._compute_evidence_from_samples(upper_loglike, log_weights)
        logz_lower = self._compute_evidence_from_samples(lower_loglike, log_weights)
        logz_mean = log_evidence  # From actual nested sampling on mean
        
        # Compute KL divergences
        kl_results = self._compute_kl_divergences(
            mean_loglike, upper_loglike, lower_loglike, log_weights
        )
        
        # If this isn't the first iteration, compute KL with previous iteration
        successive_kl = None
        if len(self.history) > 0:
            prev_mean_loglike = self.history[-1]['mean_loglike']
            successive_kl = self._compute_successive_kl(
                prev_mean_loglike, mean_loglike, log_weights
            )
        
        # Store results
        iteration_data = {
            'iteration': iteration,
            'samples': samples,
            'log_weights': log_weights,
            'mean_loglike': mean_loglike,
            'upper_loglike': upper_loglike,
            'lower_loglike': lower_loglike,
            'logz_mean': logz_mean,
            'logz_upper': logz_upper,
            'logz_lower': logz_lower,
            'kl_divergences': kl_results,
            'successive_kl': successive_kl
        }
        
        self.history.append(iteration_data)
        
        return iteration_data
    
    def check_convergence(self, iteration, **kwargs):
        """
        Main convergence checking function.
        
        Parameters:
        -----------
        iteration : current iteration number
        
        Returns:
        --------
        dict with convergence status and metrics
        """
        if iteration < self.min_iterations or len(self.history) == 0:
            return {
                'converged': False,
                'reason': 'minimum_iterations_not_reached' if iteration < self.min_iterations else 'no_history',
                'metrics': {}
            }
        
        # Get latest iteration data
        latest_data = self.history[-1]
        
        # Compute metrics
        evidence_metrics = self.compute_evidence_metrics(latest_data)
        kl_metrics = latest_data['kl_divergences']
        successive_kl = latest_data['successive_kl']
        
        # Combine all metrics
        all_metrics = {
            **evidence_metrics,
            'kl_metrics': kl_metrics,
            'successive_kl': successive_kl
        }
        
        # Check convergence criteria
        converged = False
        reasons = []
        
        # Evidence-based convergence
        if evidence_metrics['logz_cv'] < self.evidence_cv_threshold:
            converged = True
            reasons.append('evidence_cv')
        
        # KL divergence-based convergence (use symmetrized versions)
        if kl_metrics['sym_upper_lower'] < self.kl_div_threshold:
            converged = True
            reasons.append('kl_divergence_bounds')
        
        # Successive iteration KL divergence
        if successive_kl is not None and successive_kl['symmetric'] < 0.01:  # stricter threshold
            converged = True
            reasons.append('successive_kl')
        
        # Store convergence status
        conv_status = {
            'iteration': iteration,
            'converged': converged,
            'reasons': reasons,
            'all_metrics': all_metrics
        }
        
        self.convergence_status.append(conv_status)
        
        return conv_status
    
    def get_convergence_summary(self):
        """Get summary of convergence metrics across all iterations."""
        if len(self.history) == 0:
            return "No convergence data available."
        
        summary = {
            'iterations': [data['iteration'] for data in self.history],
            'evidence_cvs': [],
            'max_kl_divs': [],
            'successive_kls': [],
            'evidence_ranges': []
        }
        
        for data in self.history:
            # Evidence metrics
            ev_metrics = self.compute_evidence_metrics(data)
            summary['evidence_cvs'].append(ev_metrics['logz_cv'])
            summary['evidence_ranges'].append(ev_metrics['logz_range'])
            
            # Max KL divergence between any pair
            kl_vals = list(data['kl_divergences'].values())
            summary['max_kl_divs'].append(max(kl_vals))
            
            # Successive KL if available
            if data['successive_kl'] is not None:
                summary['successive_kls'].append(data['successive_kl']['symmetric'])
            else:
                summary['successive_kls'].append(None)
        
        return summary