import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from .gp import GP, DSLP_GP, SAAS_GP, safe_noise_floor
from .clf import train_svm, svm_predict_proba, train_nn, train_nn_multiple_restarts,train_ellipsoid_multiple_restarts, nn_predict_proba, train_ellipsoid, ellipsoid_predict_proba
from .utils.seed_utils import get_new_jax_key, get_numpy_rng
from .utils.logging_utils import get_logger
import numpyro
from numpyro.infer import MCMC, NUTS, SA, AIES
import numpyro.distributions as dist
from numpyro.infer.initialization import init_to_value, init_to_sample
from numpyro.util import enable_x64
enable_x64()
log = get_logger("[clf_gp]")


available_classifiers = {
    'svm': {
        'train': train_svm,
    },
    'nn': {
        'train': train_nn_multiple_restarts,
    },
    'ellipsoid': {
        'train': train_ellipsoid_multiple_restarts,
     },
    # ... maybe add other classifiers
}

class GPwithClassifier:
    def __init__(self, train_x=None, train_y=None, clf_flag=True,
                 clf_type='svm', clf_settings={},
                 clf_use_size=400, clf_update_step=5,
                 probability_threshold=0.5, minus_inf=-1e5,
                 clf_threshold=250, gp_threshold=1000,
                 noise=1e-8, kernel="rbf", optimizer="adam", 
                 outputscale_bounds=[-4, 4], lengthscale_bounds=[np.log10(0.05), 2],
                 lengthscale_priors='DSLP', lengthscales=None, outputscale=1.0,
                 tausq=None, tausq_bounds=[-4, 4]
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
        # self.clf_data_size = self.train_x_clf.shape[0]
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

        if lengthscale_priors.upper() == 'DSLP':
            self.gp = DSLP_GP(train_x_gp, train_y_gp, noise, kernel, optimizer,
                              outputscale_bounds, lengthscale_bounds, lengthscales=lengthscales, outputscale=outputscale)
        elif lengthscale_priors.upper() == 'SAAS':
            self.gp = SAAS_GP(train_x_gp, train_y_gp, noise, kernel, optimizer,
                              outputscale_bounds, lengthscale_bounds, tausq_bounds,
                              lengthscales=lengthscales, outputscale=outputscale, tausq=tausq)
        else:
            log.warning(f"Not using DSLP or SAAS priors (got '{lengthscale_priors}'), using default GP")
            self.gp = GP(train_x_gp, train_y_gp, noise, kernel, optimizer,
                          outputscale_bounds, lengthscale_bounds, lengthscales=lengthscales, outputscale=outputscale)

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

        log.info(f" Number of labels 0: {np.sum(labels == 0)}, 1: {np.sum(labels == 1)}")

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

    def fit(self, lr=5e-3, maxiter=300, n_restarts=4):
        """Fits the GP hyperparameters."""
        self.gp.fit(lr=lr, maxiter=maxiter, n_restarts=n_restarts)

    def predict_mean_single(self,x):
        gp_mean = self.gp.predict_mean_single(x)
        if not self.use_clf or self._clf_predict_func is None:
            return gp_mean

        clf_probs = self._clf_predict_func(x)
        return jnp.where(clf_probs >= self.probability_threshold, gp_mean, self.minus_inf)

    def predict_var_single(self,x):
        var  = self.gp.predict_var_single(x)
        if not self.use_clf or self._clf_predict_func is None:
            return var

        clf_probs = self._clf_predict_func(x)
        return jnp.where(clf_probs >= self.probability_threshold, var, safe_noise_floor)

    def predict_mean_batched(self,x):
        x = jnp.atleast_2d(x)
        return jax.vmap(self.predict_mean_single)(x)

    def predict_var_batched(self,x):
        x = jnp.atleast_2d(x)
        return jax.vmap(self.predict_var_single)(x)

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

    def predict_single(self,x):
        mean, var = self.gp.predict_single(x)
        if not self.use_clf or self._clf_predict_func is None:
            return mean, var

        clf_probs = self._clf_predict_func(x)
        mean = jnp.where(clf_probs >= self.probability_threshold, mean, self.minus_inf)
        var = jnp.where(clf_probs >= self.probability_threshold, var, safe_noise_floor)
        return mean, var

    # def batched_predict_mean(self,x):
    #     x = jnp.atleast_2d(x)
    #     return jax.vmap(self.predict_mean)(x)

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

    # def batched_predict_var(self,x):
    #     x = jnp.atleast_2d(x)
    #     return jax.vmap(self.predict_var)(x)

    def fantasy_var(self, new_x, mc_points,k_train_mc):
        """
        Computes the fantasy variance, see gp.py for more details.
        Classifier logic could potentially be added here if needed.
        """
        return self.gp.fantasy_var(new_x, mc_points,k_train_mc)

    def update(self, new_x, new_y, refit=True, lr=5e-3, maxiter=300, n_restarts=4, step=0):
        """
        Updates the classifier and GP training sets.
        Retrains classifier/GP based on thresholds and steps.
        """
        new_x = jnp.atleast_2d(new_x)
        new_y = jnp.atleast_2d(new_y)

        if not self.clf_flag:
            gp_not_updated = self.gp.update(new_x, new_y, refit=refit, lr=lr, maxiter=maxiter, n_restarts=n_restarts)
        else:
            # Check for duplicates in classifier data
            is_duplicate = jnp.any(jnp.all(jnp.isclose(self.train_x_clf, new_x, atol=1e-6, rtol=1e-4), axis=1))
            if is_duplicate:
                log.info(f"Point already exists in the classifier training set, not updating.")
                return True 
            
            # # can we not add point to GP if it is below threshold and correctly classified, only add to classifier data in that case?
            # if self.use_clf:
            #     correct_classification = (self.predict_mean(new_x) >= self.train_y_clf.max() - self.clf_threshold)
            #     predicted_classification = (self._clf_predict_func(new_x) >= self.probability_threshold)
            #     correctly_classified = correct_classification==predicted_classification

            # Update classifier data
            self.train_x_clf = jnp.concatenate([self.train_x_clf, new_x], axis=0)
            self.train_y_clf = jnp.concatenate([self.train_y_clf, new_y], axis=0)
            log.info(f"Added point to classifier data. New size: {self.clf_data_size}")

            # Update GP data if within threshold
            gp_not_updated = False
            if new_y.flatten()[0] > (self.train_y_clf.max() - self.gp_threshold):
                # Update GP - properties will automatically reflect the updated GP
                self.gp.update(new_x, new_y, refit=refit, lr=lr, maxiter=maxiter, n_restarts=n_restarts)
            else:
                log.info("Point not within GP threshold, not updating GP.")
                if refit:
                    self.gp.fit(lr=lr, maxiter=maxiter, n_restarts=n_restarts) # Refit GP on existing data?
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
        return gp_not_updated

    def kernel(self,x1,x2,lengthscales,outputscale,noise,include_noise=True):
        """
        Returns the kernel function used by the GP.
        """
        return self.gp.kernel(x1,x2,lengthscales,outputscale,noise,include_noise=include_noise)

    def get_random_point(self):

        if self.use_clf:
            pts_idx = self.train_y_clf.flatten() > self.train_y_clf.max() - self.clf_threshold
    
            # if not jnp.any(pts_idx):
            #     log.info("No points above threshold")
            #     return self.train_x_clf[jnp.argmax(self.train_y_clf)]

            # Sample a random point from the filtered points
            valid_indices = jnp.where(pts_idx)[0]
    
            # Use np.random for random selection
            chosen_index = np.random.choice(valid_indices, size=1)[0]
    
            pt = self.train_x_clf[chosen_index]
            log.info(f"Random point sampled with value {self.train_y_clf[chosen_index]}")
        else:
            log.info(f"Getting random point")

            pt = np.random.uniform(0, 1, size=self.ndim)

        return pt
    
    def save(self,outfile='gp'):
        """
        Saves the GPwithClassifier to a file

        Arguments
        ---------
        outfile: str
            The name of the file to save the GP to. Default is 'gp'.
        """
        # Save both classifier training data and GP training data
        save_dict = {
            'train_x_clf': self.train_x_clf,
            'train_y_clf': self.train_y_clf,
            'train_x_gp': self.gp.train_x,
            'train_y_gp': self.gp.train_y * self.gp.y_std + self.gp.y_mean,  # unstandardize
            'noise': self.noise,
            'clf_threshold': self.clf_threshold,
            'gp_threshold': self.gp_threshold,
            'lengthscales': self.gp.lengthscales,
            'outputscale': self.gp.outputscale,
            'hyperparam_priors': self.gp.hyperparam_priors,
            'clf_type': self.clf_type,
            'clf_use_size': self.clf_use_size,
            'clf_update_step': self.clf_update_step,
            'probability_threshold': self.probability_threshold,
            'minus_inf': self.minus_inf,
            'clf_flag': self.clf_flag,
            'use_clf': self.use_clf
        }
        
        # Add SAAS-specific parameters if applicable
        if hasattr(self.gp, 'tausq'):
            save_dict['tausq'] = self.gp.tausq
            save_dict['tausq_bounds'] = getattr(self.gp, 'tausq_bounds', [-4, 4])
            
        # Add classifier parameters if available
        if self.clf_params is not None:
            save_dict['clf_params'] = self.clf_params
        if self.clf_metrics:
            save_dict['clf_metrics'] = self.clf_metrics
        
        np.savez(f'{outfile}.npz', **save_dict)

    @classmethod
    def load(cls, filename, **kwargs):
        """
        Loads a GPwithClassifier from a file
        
        Arguments
        ---------
        filename: str
            The name of the file to load the GP from (with or without .npz extension)
        **kwargs: 
            Additional keyword arguments to pass to the GPwithClassifier constructor
            
        Returns
        -------
        gp_clf: GPwithClassifier
            The loaded GPwithClassifier object
        """
        if not filename.endswith('.npz'):
            filename += '.npz'
            
        try:
            data = np.load(filename, allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file {filename}")
        
        # Extract classifier training data
        train_x_clf = jnp.array(data['train_x_clf'])
        train_y_clf = jnp.array(data['train_y_clf'])
        
        # Extract GP settings
        clf_threshold = float(data['clf_threshold']) if 'clf_threshold' in data.files else 250
        gp_threshold = float(data['gp_threshold']) if 'gp_threshold' in data.files else 1000
        clf_type = str(data['clf_type']) if 'clf_type' in data.files else 'svm'
        clf_use_size = int(data['clf_use_size']) if 'clf_use_size' in data.files else 300
        clf_update_step = int(data['clf_update_step']) if 'clf_update_step' in data.files else 5
        probability_threshold = float(data['probability_threshold']) if 'probability_threshold' in data.files else 0.5
        minus_inf = float(data['minus_inf']) if 'minus_inf' in data.files else -1e5
        clf_flag = bool(data['clf_flag']) if 'clf_flag' in data.files else True
        noise = float(data['noise']) if 'noise' in data.files else 1e-8
        
        # Determine GP type and create lengthscale_priors
        lengthscale_priors = str(data['hyperparam_priors'].item()) if 'hyperparam_priors' in data.files else 'DSLP'
        lengthscales = jnp.array(data['lengthscales']) if 'lengthscales' in data.files else None
        outputscale = float(data['outputscale']) if 'outputscale' in data.files else None
        tausq = float(data['tausq']) if 'tausq' in data.files else None
        tausq_bounds = data['tausq_bounds'].tolist() if 'tausq_bounds' in data.files else [-4, 4]
        
        # Create GPwithClassifier instance
        gp_clf = cls(
            train_x=train_x_clf,
            train_y=train_y_clf,
            clf_flag=clf_flag,
            clf_type=clf_type,
            clf_use_size=clf_use_size,
            clf_update_step=clf_update_step,
            probability_threshold=probability_threshold,
            minus_inf=minus_inf,
            clf_threshold=clf_threshold,
            gp_threshold=gp_threshold,
            noise=noise,
            lengthscale_priors=lengthscale_priors,
            lengthscales=lengthscales,
            outputscale=outputscale,
            tausq=tausq,
            tausq_bounds=tausq_bounds,
            **kwargs
        )
        
        # Restore saved classifier parameters if available
        if 'clf_params' in data.files:
            gp_clf.clf_params = data['clf_params'].item()
        if 'clf_metrics' in data.files:
            gp_clf.clf_metrics = data['clf_metrics'].item()
            
        log.info(f"Loaded GPwithClassifier from {filename} with {train_x_clf.shape[0]} training points")
        return gp_clf
        
    def sample_GP_NUTS(self,warmup_steps=256,num_samples=512,progress_bar=True,thinning=8,verbose=True,
                       init_params=None,temp=1.,restart_on_flat_logp=True,num_chains=4):
        
        """
        Obtain samples from the posterior represented by the GP mean as the logprob.
        Optionally restarts MCMC if all logp values are the same or if HMC fails. (RESTART LOGIC TO BE IMPLEMENTED)
        """        

        rng_mcmc = get_numpy_rng()
        prob = rng_mcmc.uniform(0, 1)
        high_temp = rng_mcmc.uniform(1.,2.) ** 2
        temp = np.where(prob < 1/2, 1., high_temp) # Randomly choose temperature either 1 or high_temp
        seed_int = rng_mcmc.integers(0, 2**31 - 1)
        log.info(f"Running MCMC chains with temperature {temp:.4f}")

        def model():
            x = numpyro.sample('x', dist.Uniform(
                low=jnp.zeros(self.train_x_clf.shape[1]),
                high=jnp.ones(self.train_x_clf.shape[1])
            ))

            mean = self.predict_mean_batched(x)
            numpyro.factor('y', mean/temp)
            numpyro.deterministic('logp', mean)

        @jax.jit
        def run_single_chain(rng_key,init_x):
                init_strategy = init_to_value(values={'x': init_x})
                kernel = NUTS(model, dense_mass=False, max_tree_depth=5, init_strategy=init_strategy)
                mcmc = MCMC(kernel, num_warmup=warmup_steps, num_samples=num_samples,
                        num_chains=1, progress_bar=False, thinning=thinning)
                mcmc.run(rng_key)
                samples_x = mcmc.get_samples()['x']
                logps = mcmc.get_samples()['logp']
                return samples_x,logps
    

        num_devices = jax.device_count()
        rng_keys = jax.random.split(jax.random.PRNGKey(seed_int), num_chains) # handle properly the keys [get_new_jax_key() for _ in range(num_chains)] #
        if num_chains == 1: 
            inits = jnp.array([self.get_random_point()])
        else:
            inits = jnp.vstack([self.get_random_point() for _ in range(num_chains-1)])
            inits = jnp.vstack([inits, self.train_x_clf[jnp.argmax(self.train_y_clf)]])

        log.info(f"Running MCMC with {num_chains} chains on {num_devices} devices.")

        if (num_devices >= num_chains) and num_chains > 1:
            # if devices present run with pmap
            pmapped = jax.pmap(run_single_chain, in_axes=(0,0),out_axes=(0,0))
            samples_x, logps = pmapped(rng_keys,inits)
            # log.info(f"Xs shape: {samples_x.shape}, logps shape: {logps.shape}")
            # reshape to get proper shapes
            samples_x = jnp.concatenate(samples_x, axis=0)
            logps = jnp.reshape(logps, (samples_x.shape[0],))
            # log.info(f"Xs shape: {samples_x.shape}, logps shape: {logps.shape}")
        else:
            # if devices not available run sequentially
            samples_x = []
            logps = []
            for i in range(num_chains):
                samples_x_i, logps_i = run_single_chain(rng_keys[i], inits[i])
                samples_x.append(samples_x_i)
                logps.append(logps_i)

            samples_x = jnp.concatenate(samples_x)
            logps = jnp.concatenate(logps)

        samples_dict = {
            'x': samples_x,
            'logp': logps,
            'best': samples_x[jnp.argmax(logps)],
            'method': "MCMC"
        }

        log.info(f"Max logl found = {np.max(logps):.4f}")

        return samples_dict


    def reset_threshold_points(self):
        """
        Every 100 steps or so we discard points from the GP which do now lie outside the threshold. 
        """
        full_size = self.train_x_clf.shape[0]
        if full_size % 100 == 0:
            mask = self.train_y_clf.flatten() > (self.train_y_clf.max() - self.gp_threshold)
            train_x_gp = self.train_x_clf[mask]
            train_y_gp = self.train_y_clf[mask] * self.y_std + self.y_mean  # Rescale to original scale
            hyperparams_dict = self.gp.hyperparams
            self.gp.reset_train_data(train_x = train_x_gp, train_y = train_y_gp,)

    @property
    def lengthscales(self):
        """Access the underlying GP's lengthscales."""
        return self.gp.lengthscales
    
    @property
    def outputscale(self):
        """Access the underlying GP's outputscale."""
        return self.gp.outputscale
    
    @property
    def tausq(self):
        """Access the underlying GP's tausq if available."""
        return getattr(self.gp, 'tausq', None)
    
    @property
    def train_x(self):
        """Access the underlying GP's training inputs."""
        return self.gp.train_x
    
    @property
    def train_y(self):
        """Access the underlying GP's training outputs."""
        return self.gp.train_y
    
    @property
    def y_mean(self):
        """Access the underlying GP's y_mean."""
        return self.gp.y_mean
    
    @property
    def y_std(self):
        """Access the underlying GP's y_std."""
        return self.gp.y_std
    
    @property
    def noise(self):
        """Access the underlying GP's noise parameter."""
        return self.gp.noise
    
    @property
    def hyperparams(self): 
        return self.gp.hyperparams

    @property
    def clf_data_size(self):
        """Size of the classifier's training inputs."""
        return self.train_x_clf.shape[0]
    
    @property
    def npoints(self):
        return self.train_x_clf.shape[0]
    
    # def create_jitted_single_predict(self):

    #     @jax.jit
    #     def predict_one_mean(x):
    #         gp_mean = self.gp.jitted_single_predict_mean(x)
    #         if self.use_clf:
    #             clf_probs = self._clf_predict_func(x)
    #             return jnp.where(clf_probs >= self.probability_threshold, gp_mean, self.minus_inf)
    #         return gp_mean

    #     @jax.jit
    #     def predict_one_var(x):
    #         var = self.gp.jitted_single_predict_var(x)
    #         if self.use_clf:
    #             clf_probs = self._clf_predict_func(x)
    #             return jnp.where(clf_probs >= self.probability_threshold, var, safe_noise_floor)
    #         return var

    #     self.jitted_single_predict_mean = predict_one_mean
    #     self.jitted_single_predict_var = predict_one_var

    # def gp_numpyro_model(self,temp=1.):
    #     """
    #     Returns a numpyro model for the GP.
    #     This is used for sampling using GP surrogate using the mean as the target for NUTS or SA.
    #     """
    #     x = numpyro.sample('x', dist.Uniform(
    #             low=jnp.zeros(self.train_x_clf.shape[1]),
    #             high=jnp.ones(self.train_x_clf.shape[1])
    #         ))
            
    #     mean = self.predict_mean(x)
    #     numpyro.factor('y', mean/temp)
    #     numpyro.deterministic('logp', mean)

    # def sample_GP_NUTS_old(self, warmup_steps=512, num_samples=512, progress_bar=True, thinning=8, verbose=True,
    #                    init_params=None, temp=4., restart_on_flat_logp=True):
    #     """
    #     Obtain samples from the posterior represented by the GP mean as the logprob.
    #     Optionally restarts MCMC if all logp values are the same or if HMC fails.
    #     """
    #     start = time.time()

    #     rng_mcmc = get_numpy_rng()
    #     # high_temp = rng_mcmc.uniform(np.sqrt(2), ) ** 2
    #     prob = rng_mcmc.uniform(0, 1)
    #     # temp = np.where(prob < 1/3, 1., high_temp) # Randomly choose temperature either 1 or high_temp
    #     high_temp = rng_mcmc.uniform(1., 2.) ** 2
    #     temp = np.where(prob < 1/3, 1., high_temp) # Randomly choose temperature either 1 or high_temp
    #     log.info(f"Running MCMC chains with temperature {temp:.4f}")
        
    #     num_chains = 4

    #     samples_x = []
    #     samples_logp = []
        
    #     # temps = np.arange(1, num_chains+1, 1)

    #     rng_mcmc = get_numpy_rng()
    #     prob = rng_mcmc.uniform(0, 1)
    #     # temp = np.where(prob < 1/3, 1., high_temp) # Randomly choose temperature either 1 or high_temp
    #     high_temp = rng_mcmc.uniform(1., 2.) ** 2
    #     temp = np.where(prob < 1/3, 1., high_temp) # Randomly choose temperature either 1 or high_temp
    #     log.info(f"Running MCMC chains with temperature {temp:.4f}")
        
    #     def model():
    #         x = numpyro.sample('x', dist.Uniform(
    #             low=jnp.zeros(self.train_x_clf.shape[1]),
    #             high=jnp.ones(self.train_x_clf.shape[1])
    #         ))

    #         mean = self.jitted_single_predict_mean(x)
    #         numpyro.factor('y', mean/temp)
    #         numpyro.deterministic('logp', mean)

    #     rng_key = get_new_jax_key()

    #     for i in range(num_chains):
    #         if i== 0:
    #             init_params = self.train_x_clf[jnp.argmax(self.train_y_clf)]
    #         else:
    #             init_params = self.get_random_point() #if init_params is None else init_params
        
    #         if self.use_clf and init_params is not None:
    #             init_strategy = init_to_value(values={'x': init_params})
    #         else:
    #             init_strategy = init_to_sample()
        
    #     # First attempt with NUTS
    #         try:
    #             kernel = NUTS(model, dense_mass=False, max_tree_depth=5, init_strategy=init_strategy)
    #             mcmc = MCMC(kernel, num_warmup=warmup_steps, num_samples=num_samples,
    #                     num_chains=1, progress_bar=progress_bar, thinning=thinning)
    #             mcmc.run(rng_key)

    #             # Check if HMC ran successfully
    #             mc_samples = mcmc.get_samples()
    #             logp_vals = mc_samples['logp']
    #             hmc_success = True
            
    #         except Exception as e:
    #             if verbose:
    #                 log.error(f"HMC failed with error: {e}. Falling back to SA kernel.")
    #             hmc_success = False
    #             logp_vals = None

    #         # Check if we need to restart due to flat logp or HMC failure
    #         should_restart = False
        
    #         if not hmc_success:
    #             should_restart = True
    #             if verbose:
    #                 log.error("HMC failed. Restarting with SA kernel and best point as initial point.")
    #         elif restart_on_flat_logp and (jnp.any(logp_vals == self.minus_inf) or 
    #                                    jnp.allclose(logp_vals, logp_vals[0])):
    #             should_restart = True
    #             if verbose:
    #                 log.error("All logp values are the same or contain invalid values. Restarting MCMC from best training point.")

    #         # Restart with SA if needed
    #         if should_restart:
    #             try:
    #                 rng_key = get_new_jax_key()
    #                 num_chains = 1
    #                 best_pt = self.train_x_clf[jnp.argmax(self.train_y_clf)]
    #                 init_strategy = init_to_value(values={'x': best_pt})
    #                 log.info(f"Reinitializing MCMC with {num_chains} chains using SA kernel.")
    #                 kernel = SA(model, init_strategy=init_strategy)
    #                 mcmc = MCMC(kernel, num_warmup=warmup_steps, num_samples=2 * num_samples,
    #                         num_chains=num_chains, progress_bar=False, thinning=thinning)
    #                 mcmc.run(rng_key,)
    #             except Exception as e:
    #                 if verbose:
    #                     log.error(f"SA kernel also failed with error: {e}")
    #                 raise e
                
    #         samples = mcmc.get_samples()
    #         logp_vals = samples['logp']
    #         samples_x.append(samples['x'])
    #         samples_logp.append(logp_vals)

    #         if verbose:
    #             mcmc.print_summary(exclude_deterministic=False)
    
    #     log.info(f"Sampled parameters MCMC took {time.time() - start:.4f} s")

    #     samples_x = jnp.concatenate(samples_x, axis=0)
    #     samples_logp = jnp.concatenate(samples_logp, axis=0)

    #     samples = {'x': samples_x, 'logp': samples_logp, 'best': samples_x[jnp.argmax(samples_logp)]}

    #     print(f"shape of samples: {samples['x'].shape}")

    #     return samples
    

    # def prune(self):
    #     """
    #     Every time a new maximum is found, we discard points from the GP which do now lie outside the threshold. 
    #     TO BE IMPLEMENTED
    #     """
    #     pass


def load_clf_gp(filename, **kwargs):
    """
    Utility function to load a GPwithClassifier from a file
    
    Arguments
    ---------
    filename: str
        The name of the file to load the GP from (with or without .npz extension)
    **kwargs: 
        Additional keyword arguments to pass to the GPwithClassifier constructor
        
    Returns
    -------
    gp_clf: GPwithClassifier
        The loaded GPwithClassifier object
    """
    return GPwithClassifier.load(filename, **kwargs)