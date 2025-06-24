import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from .gp import GP, DSLP_GP, SAAS_GP, Uniform_GP
from sklearn.svm import SVC
import logging
log = logging.getLogger("[SVM-GP]")


@jax.jit
def svm_predict(x, support_vectors, dual_coef, intercept, gamma):
    """
    Compute the decision function for SVM with RBF kernel.
    
    Arguments:
      x: Input data point, shape (n_features,)
      support_vectors: JAX array of support vectors, shape (n_sv, n_features)
      dual_coef: JAX array of dual coefficients, shape (n_sv,)
      intercept: Scalar bias term.
      gamma: RBF kernel gamma parameter.
      
    Returns:
      Decision function value (scalar). Sign of this value gives the predicted class.
    """
    # Compute squared Euclidean distances between x and each support vector.
    diff = support_vectors - x  # shape (n_sv, n_features)
    norm_sq = jnp.sum(diff ** 2, axis=1)  # shape (n_sv,)
    # Compute RBF kernel values.
    kernel_vals = jnp.exp(-gamma * norm_sq)  # shape (n_sv,)
    # Compute the decision function.
    decision = jnp.sum(dual_coef * kernel_vals) + intercept
    return decision

def svm_predict_batch(x, support_vectors, dual_coef, intercept, gamma,batch_size=200):
    """
    Compute the decision function for SVM with RBF kernel for a batch of inputs.    
    """
    batched_predict = lambda x: svm_predict(x, support_vectors, dual_coef, intercept, gamma)
    return jax.lax.map(batched_predict, x, batch_size=batch_size)

def train_svm(x,y,gamma="scale",C=1e7):
    """
    Train the SVM on the data.
    """
    clf = SVC(kernel='rbf', gamma=gamma, C=C)
    clf.fit(x, y) 
    support_vectors = clf.support_vectors_
    dual_coef = clf.dual_coef_[0]  # convert to 1D array
    intercept = float(clf.intercept_[0])
    gamma_eff = float(clf._gamma) # note: this is the effective gamma value used by scikit-learn
    # convert to jax arrays
    support_vectors = jnp.array(support_vectors)
    dual_coef = jnp.array(dual_coef)
    return support_vectors, dual_coef, intercept, gamma_eff

def svm_volume_estimate(train_x,labels,seed=0,n_samples=10000):
    """
    Returns an estimate of the volume of the SVM decision boundary using Monte Carlo sampling.
    
    Arguments
    ---------
    seed: int
        The random seed for the Monte Carlo sampling. Default is 0.
    n_samples: int
        The number of samples to use for the Monte Carlo sampling. Default is 10000.

    Returns
    -------
    vol: float
        The estimated volume of the SVM decision boundary.
    err: float
        The estimated error of the volume estimate.
    """
    ndim = train_x.shape[1]
    support_vectors, dual_coef, intercept, gamma_eff = train_svm(train_x,labels)

    rng = np.random.RandomState(seed)
    # Uniformly sample in [0,1]^d
    samples = rng.rand(n_samples, ndim)
    x = jnp.array(samples)
    preds = svm_predict_batch(x, support_vectors,dual_coef,intercept, gamma_eff)
    n_inside = np.sum(preds >= 0)
    # Estimate volume and standard error
    vol = n_inside / n_samples
    err = np.sqrt(vol * (1 - vol) / n_samples)
    return vol, err

def set_svm_threshold(train_x,train_y,max_val,dlogz=0.01,method='volume'):
    if method == 'volume':
        # when using the SVM, the logZ is effectively bounded between 
        # logZ_min =  \log_svm_volume + (max_val - threshold) and logZ_max = log_svm_volume + max_val
        # since the points outside the SVM boundary are treated as minus_inf
        # we want that their contribution relative to logZ_min is tiny
        log10_th_min = 0.
        log10_th_max = 5.

        threshold = 0.
    elif method == 'chi2':
        threshold = 250
    else:
        raise ValueError("Method must be either 'volume' or 'chi2'")
    return threshold

class SVM_GP:

    def __init__(self,support_vectors=None, dual_coef=None, intercept=None, gamma_eff=None,
                 svm_use_size=400,svm_update_step=5,minus_inf=-1e5,svm_threshold = 250,gp_threshold = 10000,
                 train_x=None,train_y=None,noise=1e-8,kernel="rbf",optimizer="adam",
                 outputscale_bounds = [-4,4],lengthscale_bounds = [np.log10(0.05),2],lengthscale_priors='DSLP'):
        """
        SVM-GP class that combines a GP with an SVM classifier. The GP is trained on the data points
        that are within the GP threshold of the maximum value of the GP. The SVM classification is trained on the data
        points that are within the SVM threshold of the maximum value of the SVM. The SVM is used to classify the point as
        either inside or outside the GP threshold, if outside the GP prediction is replaced by minus_inf, otherwise the GP prediction is used.

        Arguments
        ---------
        support_vectors: JAX array of support vectors, shape (n_sv, n_features)
            The support vectors of the SVM. You can pass None to train the SVM from scratch.
        dual_coef: JAX array of dual coefficients, shape (n_sv,)
            The dual coefficients of the SVM. You can pass None to train the SVM from scratch.
        intercept: Scalar bias term.
            The intercept of the SVM. You can pass None to train the SVM from scratch.
        gamma_eff: Scalar RBF kernel gamma parameter.
            The gamma parameter of the SVM. You can pass None to train the SVM from scratch.
        svm_use_size: Minimum number of points to use the SVM
            Default is 400. If the number of points is less than this, the SVM is not used.
        svm_update_step: Number of points to update the SVM
            Default is 5. The SVM is updated every svm_update_step points.
        minus_inf: Value to replace minus infinity
            Default is -1e5. This is also the value used to replace the GP mean if the point is outside the SVM threshold.
        svm_threshold: Threshold for the SVM
            This is the threshold used to train the SVM. Default is auto. The threshold is automatically determined based on the dimension of the parameter space.
            If a point has a likelihood below this threshold, it is classified as 'bad' and the GP prediction is replace by minus_inf.
        gp_threshold: Threshold for the GP
            This is the threshold for the GP. Default is 20 times the svm_threshold. 
            If a point is below this threshold, the it is not added to the GP training set but only to the SVM training set.
        train_x: JAX array of training points, shape (n_samples, n_features)
            The initial training points for the GP.
        train_y: JAX array of training values, shape (n_samples,)
            The initial training values for the GP.
        noise: Scalar noise level for the GP
            Default is 1e-8. This is the noise level for the GP.
        kernel: Kernel type for the GP
            Default is 'rbf'. This is the kernel type for the GP. Can be 'rbf' or 'matern'.
        optimizer: Optimizer type for the GP
            Default is 'adam'. This is the optimizer type for the GP.
        outputscale_bounds: Bounds for the output scale of the GP (in log10 space) 
            Default is [-4,4]. These are the bounds for the output scale of the GP.
        lengthscale_bounds: Bounds for the length scale of the GP (in log10 space) 
            Default is [np.log10(0.05),2]. These are the bounds for the length scale of the GP.
        lengthscale_priors: Lengthscale priors for the GP
            Default is 'DSLP'. This is the lengthscale priors for the GP. Can be 'DSLP' or 'SAAS'. See the GP class for more details.
        """


        self.train_x_svm = train_x
        self.train_y_svm = train_y
        self.svm_threshold = svm_threshold
        self.gp_threshold = gp_threshold

        train_y_gp = train_y[train_y > train_y.max() - self.gp_threshold]
        train_y_gp = jnp.reshape(train_y_gp, (-1, 1))
        train_x_gp = train_x[train_y.flatten() > train_y.max() - self.gp_threshold]
        # print(f"training data shapes: {train_x_gp.shape}, {train_y_gp.shape}")

        if lengthscale_priors not in ['DSLP','SAAS', 'Uniform']:
            raise ValueError("lengthscale_priors must be either 'DSLP', 'SAAS' or 'Uniform'")

        if lengthscale_priors == 'DSLP':
            self.gp = DSLP_GP(train_x_gp,train_y_gp,noise,kernel,optimizer,outputscale_bounds,lengthscale_bounds)
        else:
            self.gp = SAAS_GP(train_x_gp,train_y_gp,noise,kernel,optimizer,outputscale_bounds,lengthscale_bounds)

        self.train_x = train_x_gp
        self.train_y = train_y_gp
        self.ndim = self.gp.ndim
        self.noise = self.gp.noise

        self.svm_data_size = 0
        self.svm_use_size = svm_use_size
        self.svm_update_step = svm_update_step
        self.minus_inf = minus_inf
        self.use_svm = self.train_x_svm.shape[0] > self.svm_use_size
        labels = np.where(self.train_y_svm.flatten() < self.train_y_svm.max() - self.svm_threshold,0, 1)
        self.support_vectors, self.dual_coef, self.intercept, self.gamma_eff = train_svm(self.train_x_svm,labels)

        log.info(f" Use SVM {self.use_svm}")

    def fit(self,lr=1e-2,maxiter=250,n_restarts=2):
        """
        Fits the GP using maximum likelihood hyperparameters with the optax adam optimizer
        """
        self.gp.fit(lr=lr,maxiter=maxiter,n_restarts=n_restarts)
    
    def predict_mean(self, x):
        """
        Predicts the mean of the GP at x and unstandardizes it if x is within the boundary of the SVM, else returns -inf
        """
        gp_mean = self.gp.predict_mean(x)

        def w_classifier():
            decision = svm_predict(x,self.support_vectors, self.dual_coef, self.intercept, self.gamma_eff)
            res = jnp.where(decision >= 0, gp_mean, self.minus_inf)
            return res
        
        def no_classifer():
            res = gp_mean
            return res
        
        res = jax.lax.cond(self.use_svm, w_classifier, no_classifer)

        return res
    
    def predict_var(self, x):
        """
        Predicts the variance of the GP at x and unstandardizes it if x is within the boundary of the SVM, else returns 0.
        """
        var =  self.gp.predict_var(x)
        
        def w_classifier():
            decision = svm_predict(x,self.support_vectors, self.dual_coef, self.intercept, self.gamma_eff)
            res = jnp.where(decision >= 0, var, 0.)
            return res
        
        def no_classifer():
            res = var
            return res
        
        res = jax.lax.cond(self.use_svm, w_classifier, no_classifer)

        return res
    
    def fantasy_var(self,x_new,mc_points):
        """
        Computes the variance of the GP at the mc_points assuming x_new is added to the training set
        """
        return self.gp.fantasy_var(x_new,mc_points)
    
    def update(self,new_x,new_y,refit=True,lr=1e-2,maxiter=250,n_restarts=2):
        """
        Updates the SVM training set and the GP with new training points and refits the GP if refit is True

        Arguments
        ---------
        new_x: JAX array of new training points, shape (n_samples, n_features)
            The new training points for the GP.
        new_y: JAX array of new training values, shape (n_samples,1)
            The new training values for the GP.
        refit: bool
            Whether to refit the GP hyperparameters. Default is True.
        lr: float
            The learning rate for the optax optimizer. Default is 1e-2.
        maxiter: int
            The maximum number of iterations for the optax optimizer. Default is 250.
        n_restarts: int
            The number of restarts for the optax optimizer. Default is 2.

        Returns
        -------
        repeat: bool
            Whether the point new_x, new_y already exists in the training set.
        """
        # first check if new point does not already exist in the training set
        if jnp.any(jnp.all(jnp.isclose(self.train_x_svm, new_x, atol=1e-6,rtol=1e-4), axis=1)):
            log.info(f" Point {new_x} already exists in the training set, not updating")
            repeat = True

        else:
            # update the SVM data
            self.train_x_svm = jnp.concatenate((self.train_x_svm,new_x))
            self.train_y_svm = jnp.concatenate((self.train_y_svm,new_y))
        
            # update the GP if point is within the GP threshold
            if new_y > self.train_y_svm.flatten().max() - self.gp_threshold:
                self.gp.update(new_x,new_y,refit=refit,lr=lr,maxiter=maxiter,n_restarts=n_restarts)
                self.train_x = self.gp.train_x
                self.train_y = self.gp.train_y
                log.info(f" Updated GP with new point ") #{new_x} and value {new_y}
                repeat = False
            else:
                log.info(f" Point not within GP threshold, not updating GP")
                if refit:
                    self.gp.fit(lr=lr,maxiter=maxiter,n_restarts=n_restarts)
                repeat = True
            # update the SVM data size, if size>svm_use_size, train the SVM
            self.svm_data_size = self.train_x_svm.shape[0]
            if (self.svm_data_size > self.svm_use_size) and (self.svm_data_size % self.svm_update_step == 0):
                self.update_svm()
            return repeat
        
    def update_svm(self):
        """
        Updates the SVM with the new training points
        """
        # train the SVM
        labels = np.where(self.train_y_svm.flatten() < self.train_y_svm.max() - self.svm_threshold,0, 1)
        self.support_vectors, self.dual_coef, self.intercept, self.gamma_eff = train_svm(self.train_x_svm,labels)
        self.use_svm = True
        log.info(f" Trained SVM with {self.svm_data_size} points")

    
    def prune(self):
        """
        Every time a new maximum is found, we discard points from the GP which do now lie outside the threshold. 
        TO BE IMPLEMENTED
        """
        pass

    def get_initial_point(self):
        """
        Get a random point from the training set within the svm boundary 
        TO BE IMPLEMENTED
        """
        pass

    def save(self,outfile='gp'):
        """
        Saves the GP to a file

        Arguments
        ---------
        outfile: str
            The name of the file to save the GP to. Default is 'gp'.
        """
        np.savez(f'{outfile}.npz',train_x=self.train_x_svm,train_y=self.train_y_svm,noise=self.noise,
                 y_std = self.train_y_svm.std(),y_mean=self.train_y_svm.mean(),
                 svm_threshold=self.svm_threshold,gp_threshold=self.gp_threshold,
                lengthscales=self.gp.lengthscales,outputscale=self.gp.outputscale
                ,support_vectors=self.support_vectors,dual_coef=self.dual_coef,intercept=self.intercept,gamma_eff=self.gamma_eff,
                )
