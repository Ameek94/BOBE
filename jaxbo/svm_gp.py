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
                 svm_use_size=400,minus_inf=-1e5,svm_threshold = 250,gp_threshold = 10000,
                 train_x=None,train_y=None,noise=1e-8,kernel="rbf",optimizer="adam",
                 outputscale_bounds = [-4,4],lengthscale_bounds = [np.log10(0.05),2],lengthscale_priors='DSLP'):
        self.train_x_svm = train_x
        self.train_y_svm = train_y
        self.svm_threshold = svm_threshold
        self.gp_threshold = gp_threshold

        train_y_gp = train_y[train_y > train_y.max() - self.gp_threshold]
        train_x_gp = train_x[train_y > train_y.max() - self.gp_threshold]

        if lengthscale_priors not in ['DSLP','SAAS']:
            raise ValueError("lengthscale_priors must be either 'DSLP' or 'SAAS'")

        if lengthscale_priors == 'DSLP':
            self.gp = DSLP_GP(train_x_gp,train_y_gp,noise,kernel,optimizer,outputscale_bounds,lengthscale_bounds)
        else:
            self.gp = SAAS_GP(train_x_gp,train_y_gp,noise,kernel,optimizer,outputscale_bounds,lengthscale_bounds)

        self.train_x = train_x_gp
        self.train_y = train_y_gp

        # self.support_vectors = jnp.array(support_vectors)
        # self.dual_coef = jnp.array(dual_coef)
        # self.intercept = intercept
        # self.gamma_eff = gamma_eff
        self.svm_data_size = 0
        self.svm_use_size = svm_use_size
        self.minus_inf = minus_inf
        self.use_svm = False

    def fit(self,lr=1e-2,maxiter=250,n_restarts=1):
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
    
    def update(self,new_x,new_y,refit=True,lr=1e-2,maxiter=250,n_restarts=1):
        """
        Updates the SVM training set and the GP with new training points and refits the GP if refit is True
        """
        # first check if new point does not already exist in the training set
        if jnp.any(jnp.all(jnp.isclose(self.train_x_svm, new_x, atol=1e-6,rtol=1e-4), axis=1)):
            log.info(f"Point {new_x} already exists in the training set, not updating")
            return True

        else:
            # first update the SVM data
            self.train_x_svm = jnp.concatenate((self.train_x_svm,new_x))
            self.train_y_svm = jnp.concatenate((self.train_y_svm,new_y))
        
            # update the GP if point is within the GP threshold
            if new_y > self.train_y_svm.max() - self.gp_threshold:
                self.gp.update(new_x,new_y,refit=refit,lr=lr,maxiter=maxiter,n_restarts=n_restarts)
                self.train_x = self.gp.train_x
                self.train_y = self.gp.train_y
                log.info(f"Updated GP with new point {new_x} and value {new_y}")
            else:
                log.info(f"Point not within GP threshold, not updating GP")
            # update the SVM data size, if size>svm_use_size, train the SVM
            self.svm_data_size = self.train_x_svm.shape[0]
            if self.svm_data_size > self.svm_use_size:
                # train the SVM
                labels = np.where(self.train_y_svm.flatten() < self.train_y_svm.max() - self.svm_threshold,0, 1)
                self.support_vectors, self.dual_coef, self.intercept, self.gamma_eff = train_svm(self.train_x_svm,labels)
                self.use_svm = True
                log.info(f"Trained SVM with {self.svm_data_size} points")
            return False

    def save(self,outfile='gp'):
        """
        Saves the GP to a file
        """
        np.savez(f'{outfile}.npz',train_x=self.train_x,train_y=self.train_y,noise=self.noise,
         y_mean=self.y_mean,y_std=self.y_std,lengthscales=self.lengthscales,outputscale=self.outputscale
         ,support_vectors=self.support_vectors,dual_coef=self.dual_coef,intercept=self.intercept,gamma_eff=self.gamma_eff,
         svm_train_x=self.train_x_svm,svm_train_y=self.train_y_svm)
