import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from .gp import GP, DSLP_GP, SAAS_GP
from sklearn.svm import SVC
import logging
log = logging.getLogger("[SVM-GP]")


@jax.jit
def svm_predict(x, support_vectors, dual_coef, intercept, gamma):
    """
    Compute the decision function for SVM with RBF kernel.
    
    Parameters:
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

def svm_predict_batch(x, support_vectors, dual_coef, intercept, gamma):
    batched_predict = lambda x: svm_predict(x, support_vectors, dual_coef, intercept, gamma)
    return jax.lax.map(batched_predict, x, batch_size=400)

def train_svm(x,y,gamma="scale",C=1e7):
    """
    Train SVM on the data
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


class SVM_GP:

    def __init__(self,support_vectors=None, dual_coef=None, intercept=None, gamma_eff=None,
                 svm_use_size=400,svm_update_step=5,minus_inf=-1e5,svm_threshold = 250,gp_threshold = 10000,
                 train_x=None,train_y=None,noise=1e-8,kernel="rbf",optimizer="adam",
                 outputscale_bounds = [-4,4],lengthscale_bounds = [np.log10(0.05),2],lengthscale_priors='DSLP'):
        self.train_x_svm = train_x
        self.train_y_svm = train_y
        self.svm_threshold = svm_threshold
        self.gp_threshold = gp_threshold

        train_y_gp = train_y[train_y > train_y.max() - self.gp_threshold]
        train_y_gp = jnp.reshape(train_y_gp, (-1, 1))
        train_x_gp = train_x[train_y.flatten() > train_y.max() - self.gp_threshold]
        # print(f"training data shapes: {train_x_gp.shape}, {train_y_gp.shape}")

        if lengthscale_priors not in ['DSLP','SAAS']:
            raise ValueError("lengthscale_priors must be either 'DSLP' or 'SAAS'")

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
        # add svm gate to the fantasy variance?
        return self.gp.fantasy_var(x_new,mc_points)
    
    def update(self,new_x,new_y,refit=True,lr=1e-2,maxiter=250,n_restarts=2):
        """
        Updates the SVM training set and the GP with new training points and refits the GP if refit is True
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
        # train the SVM
        labels = np.where(self.train_y_svm.flatten() < self.train_y_svm.max() - self.svm_threshold,0, 1)
        self.support_vectors, self.dual_coef, self.intercept, self.gamma_eff = train_svm(self.train_x_svm,labels)
        self.use_svm = True
        log.info(f" Trained SVM with {self.svm_data_size} points")

    def svm_volume_estimate(self,seed=0,n_samples=10000):
        """
        Returns the volume of the SVM decision boundary
        """
        rng = np.random.RandomState(seed)
        # Uniformly sample in [0,1]^d
        samples = rng.rand(n_samples, self.ndim)
        x = jnp.array(samples)
        preds = svm_predict_batch(x, self.support_vectors, self.dual_coef, self.intercept, self.gamma_eff)
        n_inside = np.sum(preds >= 0)
        # Estimate volume and standard error
        vol = n_inside / n_samples
        err = np.sqrt(vol * (1 - vol) / n_samples)
        return vol, err
    
    def prune(self):
        pass

    def save(self,outfile='gp'):
        """
        Saves the GP to a file
        """
        np.savez(f'{outfile}.npz',train_x=self.train_x_svm,train_y=self.train_y_svm,noise=self.noise,
                 svm_threshold=self.svm_threshold,gp_threshold=self.gp_threshold,
                lengthscales=self.gp.lengthscales,outputscale=self.gp.outputscale
                ,support_vectors=self.support_vectors,dual_coef=self.dual_coef,intercept=self.intercept,gamma_eff=self.gamma_eff,
                )
