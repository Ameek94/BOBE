import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import copy
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from numpyro.infer.initialization import init_to_value
from numpyro.util import enable_x64
enable_x64()
from .gp import GP, safe_noise_floor
from .clf import train_svm, train_nn_multiple_restarts, train_ellipsoid_multiple_restarts
from .utils.seed_utils import get_new_jax_key, get_numpy_rng
from .utils.logging_utils import get_logger
log = get_logger("clf_gp")


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
}

class GPwithClassifier(GP):
    def __init__(self, train_x=None, train_y=None, clf_flag=True,
                 clf_type='svm', clf_settings={},
                 clf_use_size=400, clf_update_step=5,
                 probability_threshold=0.5, minus_inf=-1e5,
                 clf_threshold=250., gp_threshold=500.,
                 noise=1e-8, kernel="rbf", 
                 optimizer="optax", optimizer_kwargs={'lr': 5e-3, 'name': 'adam'},
                 kernel_variance_bounds = [-4, 8], lengthscale_bounds=[np.log10(0.01), 2],
                 lengthscale_priors='DSLP', lengthscales=None, kernel_variance=1.0,
                 tausq=None, tausq_bounds=[-4, 4], train_clf_on_init=True,
                 ):
        """
        Initializes a Gaussian Process model with a preceding classifier.

        This class combines a classifier with a GP. The classifier learns a feasibility region,
        and the GP models the objective function only within this region. This is useful for
        expensive objective functions where large parts of the parameter space are invalid or
        yield very low values.

        Arguments
        ---------
        train_x : jnp.ndarray, optional
            Initial training inputs, shape (N, D). Default is None.
        train_y : jnp.ndarray, optional
            Initial training outputs, shape (N,). Default is None.
        clf_flag : bool, optional
            If True, the classifier is used. If False, the model behaves like a standard GP. Default is True.
        clf_type : str, optional
            The type of classifier to use. Supported types are 'svm', 'nn', 'ellipsoid'. Default is 'svm'.
        clf_settings : dict, optional
            A dictionary of settings for the classifier. Default is {}.
        clf_use_size : int, optional
            The minimum number of training points required to start using the classifier. Default is 400.
        clf_update_step : int, optional
            The classifier is retrained every `clf_update_step` new points. Default is 5.
        probability_threshold : float, optional
            The probability threshold for the classifier to consider a point 'feasible'. Default is 0.5.
        minus_inf : float, optional
            The value to return for points predicted as 'infeasible' by the classifier. Default is -1e5.
        clf_threshold : float, optional
            The threshold used to label data for training the classifier. Points with y < max(y) - clf_threshold are labeled as infeasible (0), others as feasible (1). Default is 250.0.
        gp_threshold : float, optional
            The threshold used to select points for training the GP. Only points with y > max(y) - gp_threshold are used. Default is 500.0.
        noise : float, optional
            The noise level for the GP. Default is 1e-8.
        kernel : str, optional
            The kernel for the GP. Default is "rbf".
        optimizer : str, optional
            The optimizer to use for training the GP. Default is "optax".
        optimizer_kwargs : dict, optional
            Keyword arguments for the optimizer. Default is {'lr': 5e-3, 'name': 'adam'}.
        kernel_variance_bounds : list, optional
            Bounds for the kernel variance hyperparameter. Default is [-4, 8].
        lengthscale_bounds : list, optional
            Bounds for the lengthscale hyperparameter. Default is [log10(0.01), 2].
        lengthscale_priors : str, optional
            The prior to use for the lengthscales. Default is 'DSLP'.
        lengthscales : jnp.ndarray, optional
            Initial values for the lengthscales. If None, they are initialized automatically. Default is None.
        kernel_variance : float, optional
            Initial value for the kernel variance. If None, it is initialized automatically. Default is 1.0.
        tausq : float, optional
            Parameter for the SAAS prior, if used. Default is None.
        tausq_bounds : list, optional
            Bounds for the tausq parameter. Default is [-4, 4].
        train_clf_on_init : bool, optional
            If True, the classifier is trained during initialization. Default is True.
        """
        # Store Data and Classifier Settings
        self.train_x_clf = jnp.array(train_x)
        self.train_y_clf = jnp.array(train_y).reshape(-1, 1)
        self.clf_use_size = clf_use_size
        self.clf_update_step = clf_update_step
        self.clf_type = clf_type.lower()
        self.clf_settings = clf_settings
        self.clf_params = None
        self.clf_metrics = {}
        self.probability_threshold = probability_threshold
        self.minus_inf = minus_inf
        self.clf_flag = clf_flag

        self.clf_threshold = clf_threshold 
        self.gp_threshold = gp_threshold

        # Prepare GP Data
        mask_gp = self.train_y_clf.flatten() > (self.train_y_clf.max() - self.gp_threshold)
        train_x_gp = self.train_x_clf[mask_gp]
        train_y_gp = self.train_y_clf[mask_gp] 

        if lengthscale_priors.upper() not in ['DSLP', 'UNIFORM']:
             log.warning(f"Ignoring lengthscale_priors='{lengthscale_priors}' and using default GP (uniform priors) due to inheritance structure.")

        super().__init__(
            train_x=train_x_gp, train_y=train_y_gp, noise=noise, kernel=kernel, optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs, kernel_variance_bounds=kernel_variance_bounds,
            lengthscale_bounds=lengthscale_bounds, lengthscales=lengthscales, kernel_variance=kernel_variance
        )

        self.use_clf = (self.clf_data_size >= self.clf_use_size) and self.clf_flag
        self._clf_predict_func = None

        if self.use_clf and self.clf_type in available_classifiers:
             if train_clf_on_init:
                 self._train_classifier()
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

        log.debug(f" Number of labels 0: {np.sum(labels == 0)}, 1: {np.sum(labels == 1)}")

        # Add method to handle if only class is present
        if np.all(labels == labels[0]):
            # If all labels are the same, we make sure not to use the classifier
            log.info("All labels are identical. Not using classifier for the moment")
            self.use_clf = False
            return 

        # Get training function and parameters
        train_func = available_classifiers[self.clf_type]['train']
        best_pt = self.train_x_clf[jnp.argmax(self.train_y_clf)]
        kwargs = {'best_pt': best_pt, 'probability_threshold': self.probability_threshold}
        self._clf_predict_func, self.clf_params, self.clf_metrics = train_func(self.train_x_clf, labels, init_params=self.clf_params, **kwargs)
        log.debug(f"Trained {self.clf_type.upper()} classifier on {self.clf_data_size} points in {time.time() - start_time:.2f}s")
        log.debug(f"Classifier metrics: {self.clf_metrics}")

    def predict_mean_single(self,x):
        """
        Predicts the mean at a single point, applying the classifier filter.

        Arguments
        ---------
        x : jnp.ndarray
            A single input point, shape (D,).

        Returns
        -------
        jnp.ndarray
            The predicted mean value. Returns `minus_inf` if the classifier deems the point infeasible.
        """
        gp_mean = super().predict_mean_single(x)
        if not self.use_clf or self._clf_predict_func is None:
            return gp_mean
        clf_probs = self._clf_predict_func(x)
        return jnp.where(clf_probs >= self.probability_threshold, gp_mean, self.minus_inf)

    def predict_var_single(self,x):
        """
        Predicts the variance at a single point, applying the classifier filter.

        Arguments
        ---------
        x : jnp.ndarray
            A single input point, shape (D,).

        Returns
        -------
        jnp.ndarray
            The predicted variance. Returns a small noise value if the classifier deems the point infeasible.
        """
        var  = super().predict_var_single(x)
        if not self.use_clf or self._clf_predict_func is None:
            return var
        clf_probs = self._clf_predict_func(x)
        return jnp.where(clf_probs >= self.probability_threshold, var, safe_noise_floor)

    def predict_mean_batched(self,x):
        """
        Predicts the mean for a batch of points using `vmap`.

        Arguments
        ---------
        x : jnp.ndarray
            A batch of input points, shape (N, D).

        Returns
        -------
        jnp.ndarray
            An array of predicted mean values, shape (N,).
        """
        x = jnp.atleast_2d(x)
        return jax.vmap(self.predict_mean_single)(x)

    def predict_var_batched(self,x):
        """
        Predicts the variance for a batch of points using `vmap`.

        Arguments
        ---------
        x : jnp.ndarray
            A batch of input points, shape (N, D).

        Returns
        -------
        jnp.ndarray
            An array of predicted variances, shape (N,).
        """
        x = jnp.atleast_2d(x)
        return jax.vmap(self.predict_var_single)(x)

    def predict_mean(self,x):
        """
        Predicts the mean for a batch of points, applying the classifier filter.

        This method is an alternative to `predict_mean_batched` and may be more efficient
        for certain classifier types.

        Arguments
        ---------
        x : jnp.ndarray
            A batch of input points, shape (N, D).

        Returns
        -------
        jnp.ndarray
            An array of predicted mean values, shape (N,).
        """
        res = super().predict_mean_batched(x)
        if not self.use_clf or self._clf_predict_func is None:
            return res
        clf_probs = self._clf_predict_func(x)
        return jnp.where(clf_probs >= self.probability_threshold, res, self.minus_inf)
    
    def predict_var(self,x):
        """
        Predicts the variance for a batch of points, applying the classifier filter.

        This method is an alternative to `predict_var_batched` and may be more efficient
        for certain classifier types.

        Arguments
        ---------
        x : jnp.ndarray
            A batch of input points, shape (N, D).

        Returns
        -------
        jnp.ndarray
            An array of predicted variances, shape (N,).
        """
        var = super().predict_var_batched(x)
        if not self.use_clf or self._clf_predict_func is None:
            return var
        clf_probs = self._clf_predict_func(x)
        return jnp.where(clf_probs >= self.probability_threshold, var, safe_noise_floor)

    def predict_single(self,x):
        """
        Predicts the mean and variance at a single point, applying the classifier filter.

        Arguments
        ---------
        x : jnp.ndarray
            A single input point, shape (D,).

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            A tuple containing the predicted mean and variance.
        """
        mean, var = super().predict_single(x)
        if not self.use_clf or self._clf_predict_func is None:
            return mean, var
        clf_probs = self._clf_predict_func(x)
        mean = jnp.where(clf_probs >= self.probability_threshold, mean, self.minus_inf)
        var = jnp.where(clf_probs >= self.probability_threshold, var, safe_noise_floor)
        return mean, var

    def update(self, new_x, new_y, refit=True, maxiter=500, n_restarts=6):
        """
        Updates the model with new data points.

        This method adds new points to the classifier's training set. If the new points
        are within the `gp_threshold` of the current maximum, they are also added to the
        GP's training set. The classifier and GP are retrained as needed.

        Arguments
        ---------
        new_x : jnp.ndarray
            New input points, shape (N, D).
        new_y : jnp.ndarray
            New output values, shape (N,).
        refit : bool, optional
            If True, the GP is refitted after updating the training data. Default is True.
        maxiter : int, optional
            The maximum number of iterations for the GP optimizer. Default is 500.
        n_restarts : int, optional
            The number of restarts for the GP optimizer. Default is 6.

        Returns
        -------
        bool
            Returns True if the provided points already existed in the training set, False otherwise.
        """
        new_x = jnp.atleast_2d(new_x)
        new_y = jnp.atleast_2d(new_y)
        if not self.clf_flag:
            return super().update(new_x, new_y, refit=refit, maxiter=maxiter, n_restarts=n_restarts)
        
        new_pts_to_add = []
        new_vals_to_add = []
        for i in range(new_x.shape[0]):
            if not jnp.any(jnp.all(jnp.isclose(self.train_x_clf, new_x[i], atol=1e-6,rtol=1e-4), axis=1)):
                new_pts_to_add.append(new_x[i])
                new_vals_to_add.append(new_y[i])

        if not new_pts_to_add:
            log.info("Point(s) already exist in the training set, not updating.")
            return True

        new_pts_to_add = jnp.atleast_2d(jnp.array(new_pts_to_add))
        new_vals_to_add = jnp.atleast_2d(jnp.array(new_vals_to_add))
        self.train_x_clf = jnp.concatenate([self.train_x_clf, new_pts_to_add])
        self.train_y_clf = jnp.concatenate([self.train_y_clf, new_vals_to_add])
        log.info(f"Added point to classifier data. New size: {self.clf_data_size}")

        gp_updated = False
        for i in range(new_pts_to_add.shape[0]):
            if new_vals_to_add[i] > (self.train_y_clf.max() - self.gp_threshold):
                super().update(new_pts_to_add[i:i+1], new_vals_to_add[i:i+1], refit=False)
                gp_updated = True
        
        if refit and gp_updated:
            self.fit(maxiter=maxiter, n_restarts=n_restarts)

        if not self.use_clf and self.clf_data_size >= self.clf_use_size:
            self.use_clf = True
            log.info(f"Classifier data size ({self.clf_data_size}) reached use size ({self.clf_use_size}). Will start using classifier.")
        
        if self.use_clf:
            self._train_classifier()
        
        return False

    def get_random_point(self,rng=None):
        """
        Gets a random point from the feasible region defined by the classifier.

        If the classifier is not in use, it returns a random point from the unit cube.
        Otherwise, it samples a point from the training data that is above the `clf_threshold`.

        Arguments
        ---------
        rng : np.random.Generator, optional
            A numpy random number generator. If None, a new one is created. Default is None.

        Returns
        -------
        jnp.ndarray
            A random point, shape (D,).
        """

        rng = rng if rng is not None else get_numpy_rng()

        if self.use_clf:
            pts_idx = self.train_y_clf.flatten() > self.train_y_clf.max() - self.clf_threshold
    
            # Sample a random point from the filtered points
            valid_indices = jnp.where(pts_idx)[0]
    
            if valid_indices.shape[0] == 0:
                 log.debug("No points above classifier threshold, sampling from unit cube.")
                 return rng.uniform(0, 1, size=self.ndim)
            chosen_index = rng.choice(valid_indices)
            pt = self.train_x_clf[chosen_index]
            log.debug(f"Random point sampled with value {self.train_y_clf[chosen_index]}")
            return pt
        else:
            log.debug(f"Getting random point in unit cube")
            return rng.uniform(0, 1, size=self.ndim)
    
    def save(self,outfile='gp'):
        """
        Saves the state of the GPwithClassifier model to a .npz file.

        This includes the training data for both the classifier and the GP, hyperparameters,
        and classifier-specific parameters and metrics.

        Arguments
        ---------
        outfile : str, optional
            The base name for the output file. '.npz' will be appended. Default is 'gp'.
        """
        save_dict = {
            'train_x_clf': self.train_x_clf, 'train_y_clf': self.train_y_clf,
            'train_x_gp': self.train_x, 'train_y_gp': self.train_y * self.y_std + self.y_mean,
            'noise': self.noise, 'clf_threshold': self.clf_threshold, 'gp_threshold': self.gp_threshold,
            'lengthscales': self.lengthscales, 'kernel_variance': self.kernel_variance,
            'hyperparam_priors': self.hyperparam_priors, 'clf_type': self.clf_type,
            'clf_use_size': self.clf_use_size, 'clf_update_step': self.clf_update_step,
            'probability_threshold': self.probability_threshold, 'minus_inf': self.minus_inf,
            'clf_flag': self.clf_flag, 'use_clf': self.use_clf
        }
        if hasattr(self, 'tausq'):
            save_dict['tausq'] = self.tausq
        if self.clf_params is not None:
            save_dict['clf_params'] = self.clf_params
        if self.clf_metrics:
            save_dict['clf_metrics'] = self.clf_metrics
        np.savez(f'{outfile}.npz', **save_dict)

    @classmethod
    def load(cls, filename, **kwargs):
        """
        Loads a GPwithClassifier model from a .npz file.

        Arguments
        ---------
        filename : str
            The path to the .npz file.
        **kwargs
            Additional keyword arguments to override the loaded parameters.

        Returns
        -------
        GPwithClassifier
            An instance of the loaded model.
        """
        if not filename.endswith('.npz'):
            filename += '.npz'
        try:
            data = np.load(filename, allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file {filename}")
        
        init_kwargs = {
            'train_x': jnp.array(data['train_x_clf']),
            'train_y': jnp.array(data['train_y_clf']),
            'clf_threshold': float(data['clf_threshold']) if 'clf_threshold' in data.files else 250,
            'gp_threshold': float(data['gp_threshold']) if 'gp_threshold' in data.files else 1000,
            'clf_type': str(data['clf_type']) if 'clf_type' in data.files else 'svm',
            'clf_use_size': int(data['clf_use_size']) if 'clf_use_size' in data.files else 300,
            'clf_update_step': int(data['clf_update_step']) if 'clf_update_step' in data.files else 5,
            'probability_threshold': float(data['probability_threshold']) if 'probability_threshold' in data.files else 0.5,
            'minus_inf': float(data['minus_inf']) if 'minus_inf' in data.files else -1e5,
            'clf_flag': bool(data['clf_flag']) if 'clf_flag' in data.files else True,
            'noise': float(data['noise']) if 'noise' in data.files else 1e-8,
            'lengthscale_priors': str(data['hyperparam_priors'].item()) if 'hyperparam_priors' in data.files else 'DSLP',
            'lengthscales': jnp.array(data['lengthscales']) if 'lengthscales' in data.files else None,
            'kernel_variance': float(data['kernel_variance']) if 'kernel_variance' in data.files else None,
            'tausq': float(data['tausq']) if 'tausq' in data.files else None,
            'tausq_bounds': data['tausq_bounds'].tolist() if 'tausq_bounds' in data.files else [-4, 4],
        }
        init_kwargs.update(kwargs)
        gp_clf = cls(**init_kwargs)
        
        if 'clf_params' in data.files:
            gp_clf.clf_params = data['clf_params'].item()
        if 'clf_metrics' in data.files:
            gp_clf.clf_metrics = data['clf_metrics'].item()
            
        log.info(f"Loaded GPwithClassifier from {filename} with {gp_clf.clf_data_size} training points")
        return gp_clf
        
    def sample_GP_NUTS(self,warmup_steps=256,num_samples=512,progress_bar=True,thinning=8,verbose=True,
                       init_params=None,temp=1.,restart_on_flat_logp=True,num_chains=4,np_rng=None, rng_key=None):
        """
        Samples from the GP posterior using the No-U-Turn Sampler (NUTS).

        This method uses MCMC to draw samples from the GP's posterior predictive distribution,
        conditioned by the classifier.

        Arguments
        ---------
        warmup_steps : int, optional
            The number of warmup steps for the NUTS sampler. Default is 256.
        num_samples : int, optional
            The number of samples to draw. Default is 512.
        progress_bar : bool, optional
            Whether to display a progress bar. Default is True.
        thinning : int, optional
            The thinning factor for the MCMC chain. Default is 8.
        verbose : bool, optional
            Whether to print verbose output. Default is True.
        init_params : dict, optional
            Initial parameters for the sampler. Default is None.
        temp : float, optional
            The temperature for the MCMC sampling. Default is 1.0.
        restart_on_flat_logp : bool, optional
            Whether to restart if the log probability becomes flat. Default is True.
        num_chains : int, optional
            The number of MCMC chains to run in parallel. Default is 4.
        np_rng : np.random.Generator, optional
            A numpy random number generator. Default is None.
        rng_key : jax.random.PRNGKey, optional
            A JAX random key. Default is None.

        Returns
        -------
        dict
            A dictionary containing the samples ('x'), log probabilities ('logp'),
            the best sample ('best'), and the sampling method ('method').
        """
        rng_mcmc = np_rng if np_rng is not None else get_numpy_rng()
        prob = rng_mcmc.uniform(0, 1)
        high_temp = rng_mcmc.uniform(1.5,6.) 
        temp = np.where(prob < 1/3, 1., high_temp)
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
                return mcmc.get_samples()['x'], mcmc.get_samples()['logp']
    
        num_devices = jax.device_count()
        num_chains = min(num_devices,num_chains)
        rng_key = rng_key if rng_key is not None else get_new_jax_key()
        rng_keys = jax.random.split(rng_key, num_chains)
        
        inits = jnp.vstack([self.get_random_point(rng=np_rng) for _ in range(num_chains-1)])
        inits = jnp.vstack([inits, self.train_x_clf[jnp.argmax(self.train_y_clf)]])

        log.info(f"Running MCMC with {num_chains} chains on {num_devices} devices.")

        if (num_devices >= num_chains) and num_chains > 1:
            pmapped = jax.pmap(run_single_chain, in_axes=(0,0),out_axes=(0,0))
            samples_x, logps = pmapped(rng_keys,inits)
            samples_x = jnp.concatenate(samples_x, axis=0)
            logps = jnp.reshape(logps, (samples_x.shape[0],))
        else:
            samples_x, logps = run_single_chain(rng_keys[0], inits[0])

        samples_dict = {
            'x': samples_x, 'logp': logps, 'best': samples_x[jnp.argmax(logps)], 'method': "MCMC"
        }
        log.info(f"Max logl found = {np.max(logps):.4f}")
        return samples_dict

    def copy(self):
        """
        Creates a deep copy of the GPwithClassifier instance.

        Returns
        -------
        GPwithClassifier
            A new instance that is a deep copy of the current one.
        """
        gp_clf_copy = GPwithClassifier(
            train_x=np.array(self.train_x_clf),
            train_y=np.array(self.train_y_clf),
            clf_flag=self.clf_flag, clf_type=self.clf_type, clf_settings=copy.deepcopy(self.clf_settings),
            clf_use_size=self.clf_use_size, clf_update_step=self.clf_update_step,
            probability_threshold=self.probability_threshold, minus_inf=self.minus_inf,
            clf_threshold=self.clf_threshold, gp_threshold=self.gp_threshold,
            noise=self.noise, kernel=self.kernel_name, optimizer=self.optimizer_method,
            optimizer_kwargs=self.optimizer_kwargs, kernel_variance_bounds=self.kernel_variance_bounds,
            lengthscale_bounds=self.lengthscale_bounds, lengthscales=np.array(self.lengthscales),
            kernel_variance=float(self.kernel_variance), train_clf_on_init=False,
        )
        gp_clf_copy.alphas = jnp.array(self.alphas, copy=True)
        gp_clf_copy.cholesky = jnp.array(self.cholesky, copy=True)
        gp_clf_copy.fitted = self.fitted
        gp_clf_copy.clf_params = copy.deepcopy(self.clf_params)
        gp_clf_copy.clf_metrics = copy.deepcopy(self.clf_metrics)
        gp_clf_copy.use_clf = self.use_clf
        gp_clf_copy._clf_predict_func = self._clf_predict_func
        return gp_clf_copy

    @property
    def tausq(self):
        return getattr(self, 'tausq', None)
    
    @property
    def clf_data_size(self):
        """Returns the number of data points available for the classifier."""
        return self.train_x_clf.shape[0]
    
    @property
    def npoints(self):
        """Returns the total number of data points (same as `clf_data_size`)."""
        return self.train_x_clf.shape[0]

def load_clf_gp(filename, **kwargs):
    """
    A convenience function to load a GPwithClassifier model.

    This is an alias for `GPwithClassifier.load`.

    Arguments
    ---------
    filename : str
        The path to the .npz file.
    **kwargs
        Additional keyword arguments to override the loaded parameters.

    Returns
    -------
    GPwithClassifier
        An instance of the loaded model.
    """
    return GPwithClassifier.load(filename, **kwargs)