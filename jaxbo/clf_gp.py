import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from .gp import GP, DSLP_GP, SAAS_GP, Uniform_GP
from .clf import train_svm, svm_predict_proba, train_nn, nn_predict_proba, train_ellipsoid, ellipsoid_predict_proba
import numpyro
# numpyro.set_host_device_count(4) ?
from numpyro.infer import MCMC, NUTS, SA, AIES
import numpyro.distributions as dist
from numpyro.infer.initialization import init_to_value, init_to_sample
from numpyro.util import enable_x64
enable_x64()
import logging
log = logging.getLogger(__name__)


available_classifiers = {
    'svm': {
        'train': train_svm,
    },
    'nn': {
        'train': train_nn,
    },
    'ellipsoid': {
        'train': train_ellipsoid,
     },
    # ... maybe add other classifiers
}

class ClassifierGP:
    def __init__(self, train_x=None, train_y=None, clf_flag=True,
                 clf_type='svm', clf_settings={},
                 clf_use_size=400, clf_update_step=5,
                 probability_threshold=0.5, minus_inf=-1e5,
                 clf_threshold=250, gp_threshold=1000,
                 noise=1e-8, kernel="rbf", optimizer="adam", 
                 outputscale_bounds=[-4, 4], lengthscale_bounds=[np.log10(0.05), 2],
                 lengthscale_priors='DSLP', lengthscales=None, outputscale=1.0,
                 ):
        """
        Generic Classifier-GP class combining a GP with a classifier. The GP is trained on the data points
        that are within the GP threshold of the maximum value of the GP.

        Arguments
        ---------
        train_x : array-like, shape (n_samples, n_dim)
            Initial training points.
        train_y : array-like, shape (n_samples,)
            Initial training values.
        clf_type : str, optional
            Type of classifier ('svm', 'nn', 'ellipsoid', etc.). Default is 'svm'.
        clf_params : dict, optional
            Parameters specific to the chosen classifier. Default is None.
        clf_use_size : int, optional
            Minimum number of points to start using the classifier. Default is 300.
        clf_update_step : int, optional
            Update classifier every `clf_update_step` points after `clf_use_size` is reached. Default is 5.
        probability_threshold : float, optional
            Threshold for classifier probability/score to consider a point feasible (important for nn, ellipsoid). Default is 0.5.
        minus_inf : float, optional
            Value used for infeasible predictions. Default is -1e5.
        clf_threshold : float, optional
            Threshold for initial classifier training labels (if used).
            If None, `gp_threshold` might be used or a default calculated.
        gp_threshold : float, optional
            Threshold for adding points to the GP training set. Default is 5000.
        noise, kernel, optimizer, outputscale_bounds, lengthscale_bounds,
        lengthscale_priors, lengthscales, outputscale:
            GP parameters (see DSLP_GP/SAAS_GP).
        """
        # Store Data and Classifier Settings
        self.train_x_clf = jnp.array(train_x)
        self.train_y_clf = jnp.array(train_y).reshape(-1, 1) # Ensure 2D
        self.clf_data_size = self.train_x_clf.shape[0]
        self.clf_use_size = clf_use_size
        self.clf_update_step = clf_update_step
        self.clf_type = clf_type.lower()
        self.clf_settings = clf_settings
        self.clf_params = None
        self.clf_metrics = {}
        self.probability_threshold = probability_threshold
        self.minus_inf = minus_inf
        self.clf_flag = clf_flag  # Whether to use classifier or not

        # Handle Thresholds
        self.clf_threshold = clf_threshold 
        self.gp_threshold = gp_threshold


        # Prepare GP Data
        mask_gp = self.train_y_clf.flatten() > (self.train_y_clf.max() - self.gp_threshold)
        train_x_gp = self.train_x_clf[mask_gp]
        train_y_gp = self.train_y_clf[mask_gp] 

        # Initialize GP 
        self.ndim = train_x_gp.shape[1] 
        if lengthscale_priors not in ['DSLP', 'SAAS', 'Uniform']:
            raise ValueError("lengthscale_priors must be either 'DSLP', 'SAAS' or 'Uniform'")
        if lengthscale_priors == 'DSLP':
            self.gp = DSLP_GP(train_x_gp, train_y_gp, noise, kernel, optimizer,
                              outputscale_bounds, lengthscale_bounds, lengthscales=lengthscales, outputscale=outputscale)
        elif lengthscale_priors == 'SAAS':
            self.gp = SAAS_GP(train_x_gp, train_y_gp, noise, kernel, optimizer,
                              outputscale_bounds, lengthscale_bounds, lengthscales=lengthscales, outputscale=outputscale)
        else:
             self.gp = Uniform_GP(train_x_gp, train_y_gp, noise, kernel, optimizer,
                              outputscale_bounds, lengthscale_bounds, lengthscales=lengthscales, outputscale=outputscale)

        self.train_x = self.gp.train_x
        self.train_y = self.gp.train_y
        self.noise = self.gp.noise

        # Initialize Classifier
        self.use_clf = (self.clf_data_size >= self.clf_use_size) and self.clf_flag
        self.clf_model_params = None
        self._clf_predict_func = None # Will hold the jitted prediction function

        if self.use_clf and self.clf_type in available_classifiers:
             self._train_classifier() # Initial training if enough data
        elif self.use_clf and self.clf_type not in available_classifiers:
             raise ValueError(f"Classifier type '{self.clf_type}' not supported. Available: {list(available_classifiers.keys())}")
        else:
             log.info(f"Not enough data ({self.clf_data_size}) to use classifier (need {self.clf_use_size} points), or classifier type not set.")


    def _train_classifier(self):
        """Trains the classifier based on clf_type."""

        start_time = time.time()

        # Determine labels for classifier training
        labels = np.where(
            self.train_y_clf.flatten() < self.train_y_clf.max() - self.clf_threshold,
            0, 1
        )

        # Add method to handle if only class is present
        if np.all(labels == labels[0]):
            # If all labels are the same, we make sure not to use the classifier
            log.info("All labels are identical. Not using classifier for the moment")
            self.use_clf = False
            return 

        # Get training function and parameters
        train_func = available_classifiers[self.clf_type]['train']

        # Call the specific training function
        # Training functions return (predict_func, model_params, metrics_dict)

        
        best_pt = self.train_x_clf[jnp.argmax(self.train_y_clf)]
        kwargs = {
            'best_pt': best_pt,
            'probability_threshold': self.probability_threshold,
        }
        self._clf_predict_func, self.clf_params, self.clf_metrics = train_func(self.train_x_clf, 
                                                                               labels, init_params = self.clf_params,
                                                                               **kwargs)

        log.info(f"Trained {self.clf_type.upper()} classifier on {self.clf_data_size} points in {time.time() - start_time:.2f}s")
        log.info(f"Classifier metrics: {self.clf_metrics}") # Use debug for detailed metrics

    def fit(self, lr=1e-2, maxiter=150, n_restarts=4):
        """Fits the GP hyperparameters."""
        return self.gp.fit(lr=lr, maxiter=maxiter, n_restarts=n_restarts)

    def predict_mean(self, x):
        """
        Predicts the GP mean, adjusted by the classifier.
        If classifier predicts infeasible (prob < threshold), return minus_inf.
        """
        gp_mean = self.gp.predict_mean(x)
        if not self.use_clf or self._clf_predict_func is None:
            return gp_mean

        clf_probs = self._clf_predict_func(x)
        res = jnp.where(clf_probs >= self.probability_threshold, gp_mean, self.minus_inf)
        return res
    
    def predict_var(self, x):
        """
        Predicts the GP variance, adjusted by the classifier.
        If classifier predicts infeasible (prob < threshold), return 0 variance.
        """
        var = self.gp.predict_var(x)
        if not self.use_clf or self._clf_predict_func is None:
            return var

        clf_probs = self._clf_predict_func(x)
        res = jnp.where(clf_probs >= self.probability_threshold, var, 0.0)
        return res

    def predict(self, x):
        """
        Predicts the mean and variance of the GP at x but does not unstandardize it adjusted by the classifier.
        If classifier predicts infeasible (prob < threshold), return 0 variance and minus_inf mean
        """
        mean, var = self.gp.predict(x)
        if not self.use_clf or self._clf_predict_func is None:
            return mean, var
        clf_probs = self._clf_predict_func(x)
        mean_res = jnp.where(clf_probs >= self.probability_threshold, mean, 0.0)
        var_res = jnp.where(clf_probs >= self.probability_threshold, var, 0.0)
        return mean_res, var_res

    def fantasy_var(self, x_new, mc_points):
        """
        Computes the fantasy variance, see gp.py for more details.
        Classifier logic could potentially be added here if needed.
        """
        return self.gp.fantasy_var(x_new, mc_points)

    def update(self, new_x, new_y, refit=True, lr=1e-2, maxiter=150, n_restarts=2, step=0):
        """
        Updates the classifier and GP training sets.
        Retrains classifier/GP based on thresholds and steps.
        """
        new_x = jnp.atleast_2d(new_x)
        new_y = jnp.atleast_2d(new_y)
        mll_val = np.nan
        if not self.clf_flag:
            gp_not_updated = self.gp.update(new_x, new_y, refit=refit, lr=lr, maxiter=maxiter, n_restarts=n_restarts)
        else:
            # Check for duplicates in classifier data
            is_duplicate = jnp.any(jnp.all(jnp.isclose(self.train_x_clf, new_x, atol=1e-6, rtol=1e-4), axis=1))
            if is_duplicate:
                log.info(f"Point already exists in the classifier training set, not updating.")
                return True, mll_val

            # Update classifier data
            self.train_x_clf = jnp.concatenate([self.train_x_clf, new_x], axis=0)
            self.train_y_clf = jnp.concatenate([self.train_y_clf, new_y], axis=0)
            self.clf_data_size += 1
            log.info(f"Added point to classifier data. New size: {self.clf_data_size}")

            # Update GP data if within threshold
            gp_not_updated = False
            if new_y.flatten()[0] > (self.train_y_clf.max() - self.gp_threshold):
                # Update GP
                _, mll_val = self.gp.update(new_x, new_y, refit=refit, lr=lr, maxiter=maxiter, n_restarts=n_restarts)
                self.train_x = self.gp.train_x
                self.train_y = self.gp.train_y
            else:
                log.info("Point not within GP threshold, not updating GP.")
                if refit:
                    mll_val = self.gp.fit(lr=lr, maxiter=maxiter, n_restarts=n_restarts) # Refit GP on existing data?
                gp_not_updated = True

            # Check if classifier data size has reached the threshold
            if not self.use_clf:
                if self.clf_data_size >= self.clf_use_size:
                    log.info(f"Classifier data size ({self.clf_data_size}) reached use size ({self.clf_use_size}). Will start using classifier.")
                    self.use_clf = True

            # Retrain classifier if conditions are met
            if self.use_clf and (step % self.clf_update_step == 0):
                self._train_classifier()

        # Return whether GP was updated, classifier is always updated
        return gp_not_updated, mll_val

    def get_random_point(self):
        pts_idx = self.train_y_clf.flatten() > self.train_y_clf.max() - self.clf_threshold/2.
        if not jnp.any(pts_idx):
            log.info("No points above threshold")
            return self.train_x_clf[jnp.argmax(self.train_y_clf)]

        # Sample a random point from the filtered points
        valid_indices = jnp.where(pts_idx)[0]
    
        # Use np.random for random selection
        chosen_index = np.random.choice(valid_indices, size=1)[0]
    
        result = self.train_x_clf[chosen_index]
        log.info(f"Random point sampled with value {self.train_y_clf[chosen_index]}")
    
        return result
    
    def save(self,outfile='gp'):
        """
        Saves the GP to a file

        Arguments
        ---------
        outfile: str
            The name of the file to save the GP to. Default is 'gp'.
        """
        np.savez(f'{outfile}.npz',train_x=self.train_x_clf,train_y=self.train_y_clf,noise=self.noise,
                 y_std = self.train_y_clf.std(),y_mean=self.train_y_clf.mean(),
                 clf_threshold=self.clf_threshold,gp_threshold=self.gp_threshold,
                lengthscales=self.gp.lengthscales,outputscale=self.gp.outputscale
                )
        
    def gp_numpyro_model(self):
        """
        Returns a numpyro model for the GP.
        This is used for sampling using GP surrogate using the mean as the target for NUTS or SA.
        """
        x = numpyro.sample('x', dist.Uniform(
                low=jnp.zeros(self.train_x_clf.shape[1]),
                high=jnp.ones(self.train_x_clf.shape[1])
            ))
            
        mean = self.predict_mean(x)
        numpyro.factor('y', mean)
        numpyro.deterministic('logp', mean)

    def sample_GP_NUTS(self, rng_key, warmup_steps=512, num_samples=512, progress_bar=True, thinning=8, verbose=True,
                       init_params=None, temp=1., restart_on_flat_logp=True):
        """
        Obtain samples from the posterior represented by the GP mean as the logprob.
        Optionally restarts MCMC if all logp values are the same or if HMC fails.
        """

        num_chains = 1
        # # init_params = jnp.array([self.get_random_point() for _ in range(num_chains-1)])#  if init_params is None else init_params
        # best_pt = self.train_x_clf[jnp.argmax(self.train_y_clf)]
        # # init_params = jnp.concatenate([init_params, best_pt.reshape(1, -1)], axis=0) #if init_params is not None else None
        # init_strategy = init_to_value(values = {'x': self.get_random_point()}) #values=[{'x': init_params[i]} for i in range(num_chains)]


        init_params = self.get_random_point() if init_params is None else init_params
        
        if self.use_clf and init_params is not None:
            init_strategy = init_to_value(values={'x': init_params})
        else:
            init_strategy = init_to_sample()

        start = time.time()
        
        # try:
        #     kernel = SA(self.gp_numpyro_model, init_strategy=init_strategy)
        #     mcmc = MCMC(kernel, num_warmup=warmup_steps, num_samples= num_samples,
        #                     num_chains=num_chains, progress_bar=False, thinning=thinning,
        #                     chain_method='sequential')
        #     mcmc.run(rng_key)
        # except Exception as e:
        #     if verbose:
        #         log.error(f"SA kernel also failed with error: {e}")
        #     raise e
        
        # First attempt with NUTS
        try:
            kernel = NUTS(self.gp_numpyro_model, dense_mass=False, max_tree_depth=5, init_strategy=init_strategy)
            mcmc = MCMC(kernel, num_warmup=warmup_steps, num_samples=num_samples,
                        num_chains=1, progress_bar=progress_bar, thinning=thinning)
            mcmc.run(rng_key)
            
            # Check if HMC ran successfully
            mc_samples = mcmc.get_samples()
            logp_vals = mc_samples['logp']
            hmc_success = True
            
        except Exception as e:
            if verbose:
                log.info(f"HMC failed with error: {e}. Falling back to SA kernel.")
            hmc_success = False
            logp_vals = None

        # Check if we need to restart due to flat logp or HMC failure
        should_restart = False
        
        if not hmc_success:
            should_restart = True
            if verbose:
                log.info("HMC failed. Restarting with SA kernel and best point as initial point.")
        elif restart_on_flat_logp and (jnp.any(logp_vals == self.minus_inf) or 
                                       jnp.allclose(logp_vals, logp_vals[0])):
            should_restart = True
            if verbose:
                log.info("All logp values are the same or contain invalid values. Restarting MCMC from best training point.")

        # Restart with SA if needed
        if should_restart:
            try:
                num_chains = 1
                best_pt = self.train_x_clf[jnp.argmax(self.train_y_clf)]
                init_strategy = init_to_value(values={'x': best_pt})
                log.info(f"Reinitializing MCMC with {num_chains} chains using SA kernel.")
                kernel = SA(self.gp_numpyro_model, init_strategy=init_strategy)
                mcmc = MCMC(kernel, num_warmup=warmup_steps, num_samples=2 * num_samples,
                            num_chains=num_chains, progress_bar=False, thinning=thinning)
                mcmc.run(rng_key)
            except Exception as e:
                if verbose:
                    log.error(f"SA kernel also failed with error: {e}")
                raise e

        if verbose:
            mcmc.print_summary(exclude_deterministic=False)
        log.info(f"Sampled parameters MCMC took {time.time() - start:.4f} s")

        mc_samples = mcmc.get_samples()
        samples = {
            'x': mc_samples['x'],
            'logp': mc_samples['logp'],
            'best': mc_samples['x'][jnp.argmax(mc_samples['logp'])]
        }

        print(f"shape of samples: {samples['x'].shape}")

        return samples
    

    def prune(self):
        """
        Every time a new maximum is found, we discard points from the GP which do now lie outside the threshold. 
        TO BE IMPLEMENTED
        """
        pass