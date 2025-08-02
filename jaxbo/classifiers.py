# interfaces and routines for some classifiers
# SVM, Neural Networks, k-Nearest Neighbors, etc.

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from .gp import GP, DSLP_GP, SAAS_GP
from sklearn.svm import SVC
import logging
log = logging.getLogger("[CLF]")


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
    # Add method to handle case where data has only 1 label
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

class SVMClassifier:
    """
    SVM Classifier using JAX for prediction.
    """
    def __init__(self, support_vectors, dual_coef, intercept, gamma):
        self.support_vectors = support_vectors
        self.dual_coef = dual_coef
        self.intercept = intercept
        self.gamma = gamma

    def predict(self, x):
        return svm_predict_batch(x, self.support_vectors, self.dual_coef, self.intercept, self.gamma)
    
    def __call__(self, x):
        return self.predict(x)