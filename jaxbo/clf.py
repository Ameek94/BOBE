# interfaces and routines for some classifiers
# SVM, Neural Networks, Ellipsoidal bound, etc.

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sklearn.svm import SVC
from flax import linen as nn
from flax.core import freeze
import optax
from typing import Callable, Dict, Any, Union, List, Optional, Tuple
from functools import partial
from .logging_utils import get_logger 
from .seed_utils import get_numpy_rng, get_new_jax_key
log = get_logger("[clf]")

# -----------------------------------------------------------------------------
# Scikit-learn SVM Classifier using JAX for prediction.
# -----------------------------------------------------------------------------

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

def svm_predict_proba(x, support_vectors, dual_coef, intercept, gamma):
    decision = svm_predict(x, support_vectors, dual_coef, intercept, gamma)
    return jnp.where(decision >= 0, 1.0, 0.0)  # Binary classification: 1 if decision >= 0, else 0

def svm_predict_batch(x, support_vectors, dual_coef, intercept, gamma,batch_size=200):
    """
    Compute the decision function for SVM with RBF kernel for a batch of inputs.    
    """
    batched_predict = lambda x: svm_predict(x, support_vectors, dual_coef, intercept, gamma)
    return jax.lax.map(batched_predict, x, batch_size=batch_size)

def train_svm(x,y, svm_settings = {} ,gamma="scale",C=1e7, init_params=None, **kwargs):
    """
    Train the SVM on the data.
    """

    gamma = svm_settings.get('gamma', "scale")
    C = svm_settings.get('C', 1e7)
    kernel = svm_settings.get('kernel', 'rbf')


    # Currently missing method to handle case where data has only 1 label

    clf = SVC(kernel=kernel, gamma=gamma, C=C)
    clf.fit(x, y) 
    support_vectors = clf.support_vectors_
    dual_coef = clf.dual_coef_[0]  # convert to 1D array
    intercept = float(clf.intercept_[0])
    gamma_eff = float(clf._gamma) # note: this is the effective gamma value used by scikit-learn
    
    # convert to jax arrays
    support_vectors = jnp.array(support_vectors)
    dual_coef = jnp.array(dual_coef)
    metrics = {
        'n_support_vectors': len(support_vectors),
        'gamma': f"{gamma_eff:.2e}",
        'C': f"{C:.2e}",
        'intercept': f"{intercept:.2e}",
    }
    predict_fn = jax.jit(partial(svm_predict_proba, support_vectors=support_vectors, dual_coef=dual_coef, intercept=intercept, gamma=gamma_eff))
    params = {
        'support_vectors': support_vectors,
        'dual_coef': dual_coef,
        'intercept': intercept,
        'gamma_eff': gamma_eff
    }
    return predict_fn, params, metrics


# The methods below are currently in development
# -----------------------------------------------------------------------------
# Neural Network Classifier with train/validation split
# -----------------------------------------------------------------------------

# Common training utilities
def create_train_val_split(x: jnp.ndarray, y: jnp.ndarray, val_frac: float, seed: int):
    """Create fixed train/validation split"""
    N = x.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    split = int(N * (1 - val_frac))
    train_idx, val_idx = perm[:split], perm[split:]
    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx])

def train_with_restarts(
    train_fn: Callable,
    x: jnp.ndarray,
    y: jnp.ndarray,
    n_restarts: int = 4,
    seed_offset: int = 0,
    split_seed: int = 42,
    init_params = None,  # Add this parameter
    **train_kwargs
) -> Tuple[Callable, Dict, Dict]:
    """
    Train model with multiple restarts using consistent train/val split.
    
    Args:
        train_fn: Training function that returns (predict_fn, params, metrics)
        x: (N, d) features
        y: (N,) labels
        n_restarts: number of random restarts
        seed_offset: offset for training seed generation
        split_seed: fixed seed for train/val split consistency
        init_params: initial parameters for first restart
        **train_kwargs: passed to train_fn
    """
    # Create consistent train/val split
    (x_train, y_train), (x_val, y_val) = create_train_val_split(x, y, 
                                                               train_kwargs.get('val_frac', 0.2), 
                                                               split_seed)
    
    best_val_loss = jnp.inf
    best_predict_fn = None
    best_params = None
    best_metrics = {}

    try:
        rng = get_numpy_rng()
    except Exception as e:
        log.error(f"{e} - falling back to default RNG")
        rng = np.random.default_rng()

    for i in range(n_restarts):
        current_seed = rng.integers(0, 2**32 - 1)
        log.info(f"[Restart {i+1}/{n_restarts}] Starting training with seed {current_seed}")
        
        # Use initial params for first restart, None for others
        restart_init_params = init_params if i == 0 else None
        
        if i == 0 and init_params is not None:
            log.info(f"[Restart {i+1}/{n_restarts}] Using provided initial parameters")
        elif i > 0:
            log.info(f"[Restart {i+1}/{n_restarts}] Using random initialization")

        # Pass the pre-split data to avoid re-splitting
        predict_fn, params, metrics = train_fn(
            x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val,
            seed=current_seed,
            init_params=restart_init_params,  # Pass init_params to training function
            **train_kwargs
        )

        val_loss = float(metrics['val_loss'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_predict_fn = predict_fn
            best_params = params
            best_metrics = metrics
            log.info(f"[Restart {i+1}/{n_restarts}] New best val_loss: {val_loss:.4e}")

    log.info(f"[Training] Best model selected with val_loss = {best_val_loss:.4e}")
    return best_predict_fn, best_params, best_metrics


# Neural Network Classifier
class FeasibilityMLP(nn.Module):
    hidden_dims: list
    dropout_rate: float

    @nn.compact
    def __call__(self, x, train: bool = False):
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(1)(x)
        return x

def train_nn(
    x_train: jnp.ndarray, y_train: jnp.ndarray,
    x_val: jnp.ndarray, y_val: jnp.ndarray,
    hidden_dims=(64,64),
    dropout_rate=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    n_epochs=2000,
    batch_size=128,
    early_stop_patience=100,
    seed=0,
    init_params=None,  # Add this parameter
    **kwargs
):
    """Simplified NN training with pre-split data"""
    N, d = x_train.shape
    model = FeasibilityMLP(hidden_dims=list(hidden_dims), dropout_rate=dropout_rate)
    
    # Handle initialization
    if init_params is not None:
        params = init_params
    else:
        key = jax.random.PRNGKey(seed)
        params = model.init(key, jnp.ones((1, d)), train=True)

    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, batch_x, batch_y, rng):
        logits = model.apply(params, batch_x, train=True, rngs={"dropout": rng})
        return optax.sigmoid_binary_cross_entropy(logits.squeeze(-1), batch_y).mean()

    @jax.jit
    def compute_val_loss(params):
        logits = model.apply(params, x_val, train=False)
        return optax.sigmoid_binary_cross_entropy(logits.squeeze(-1), y_val).mean()

    @jax.jit
    def train_step(params, opt_state, batch_x, batch_y, rng):
        grads = jax.grad(loss_fn)(params, batch_x, batch_y, rng)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    best_params = params
    best_val_loss = jnp.inf
    x_np, y_np = np.array(x_train), np.array(y_train)
    steps = max(1, x_train.shape[0] // batch_size)

    rng_opt = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    for epoch in range(n_epochs):
        perm_train = rng_opt.permutation(x_train.shape[0])
        for i in range(steps):
            idx = perm_train[i*batch_size:(i+1)*batch_size]
            bx = jnp.array(x_np[idx])
            by = jnp.array(y_np[idx])
            key, subkey = jax.random.split(key)
            params, opt_state = train_step(params, opt_state, bx, by, subkey)
            
        val_loss = compute_val_loss(params)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            patience = early_stop_patience
        else:
            patience -= 1
            if patience <= 0:
                log.info(f"[NN] early stopping at epoch {epoch}")
                break

    apply_fn = lambda xx: model.apply(best_params, xx, train=False)
    metrics = {
        'train_loss': f"{float(loss_fn(best_params, x_train, y_train, key)):.2e}",
        'val_loss': f"{float(best_val_loss):.2e}",
        'epochs': epoch + 1,
    }

    predict_fn = jax.jit(lambda x: jax.nn.sigmoid(apply_fn(x).squeeze(-1)))
    return predict_fn, best_params, metrics

def train_nn_multiple_restarts(x: jnp.ndarray, y: jnp.ndarray, **kwargs):
    """Wrapper for NN training with restarts"""
    return train_with_restarts(train_nn, x, y, **kwargs)

# Prediction functions

def nn_predict(x: jnp.ndarray, apply_fn):
    """
    Single-point prediction: returns logit.
    """
    logit = apply_fn(x[None, :]).squeeze()
    return logit

def nn_predict_batch(x: jnp.ndarray, apply_fn):
    """
    Batch prediction: returns logits for all x.
    """
    return apply_fn(x).squeeze(-1)

def nn_predict_proba(x,  apply_fn):
    return jax.nn.sigmoid(nn_predict(x,  apply_fn))

def nn_predict_proba_batch(x, apply_fn):
    return jax.nn.sigmoid(nn_predict_batch(x,  apply_fn))

# Ellipsoid Classifier
class EllipsoidClassifier(nn.Module):
    d: int
    mu: jnp.ndarray
    init_scale: float = 1.0

    def setup(self):
        tril = self.d * (self.d + 1) // 2
        self.flat_L = self.param("flat_L", nn.initializers.normal(self.init_scale), (tril,))
        self.alpha = self.param("alpha", nn.initializers.ones, ())
        self.beta = self.param("beta", nn.initializers.zeros, ())

    def _unpack_L(self):
        L_matrix = jnp.zeros((self.d, self.d))
        tril_indices = jnp.tril_indices(self.d)
        rows, cols = tril_indices
        diagonal_mask = rows == cols
        flat_L_processed = jnp.where(diagonal_mask, nn.softplus(self.flat_L) + 1e-4, self.flat_L)
        return L_matrix.at[tril_indices].set(flat_L_processed)

    @nn.compact
    def __call__(self, x, train: bool = False):
        L = self._unpack_L()
        diff = x - self.mu
        md2 = jnp.einsum("...i,ij,...j->...", diff, L @ L.T, diff)
        logit = -self.alpha * md2 + self.beta
        return logit

def train_ellipsoid(
    x_train: jnp.ndarray, y_train: jnp.ndarray,
    x_val: jnp.ndarray, y_val: jnp.ndarray,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    n_epochs: int = 2000,
    batch_size: int = 32,
    seed: int = 0,
    patience: int = 250,
    init_params=None,  # Add this parameter
    **kwargs
):
    """Simplified ellipsoid training with pre-split data"""
    mu = kwargs.get('best_pt', 0.5 * jnp.ones(x_train.shape[1]))
    model = EllipsoidClassifier(d=x_train.shape[1], mu=mu)
    
    # Handle initialization
    if init_params is not None:
        params = init_params
    else:
        key = jax.random.PRNGKey(seed)
        params = model.init(key, x_train)
    
        
    optimizer = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, batch_x, batch_y):
        logits = model.apply(params, batch_x, train=False)
        return optax.sigmoid_binary_cross_entropy(logits, batch_y).mean()
    
    @jax.jit
    def train_step(params, opt_state, bx, by):
        grads = jax.grad(loss_fn)(params, bx, by)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    best_params = params
    best_val_loss = jnp.inf
    x_np, y_np = np.array(x_train), np.array(y_train)
    steps = max(1, x_train.shape[0] // batch_size)
    patience_counter = 0
    
    rng = np.random.RandomState(seed)

    for epoch in range(n_epochs):
        perm_train = rng.permutation(x_train.shape[0])
        for i in range(steps):
            idx = perm_train[i*batch_size:(i+1)*batch_size]
            bx = jnp.array(x_np[idx])
            by = jnp.array(y_np[idx])
            params, opt_state = train_step(params, opt_state, bx, by)

        val_loss = loss_fn(params, x_val, y_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    train_loss = loss_fn(best_params, x_train, y_train)
    metrics = {
        'train_loss': f"{float(train_loss):.2e}",
        'val_loss': f"{float(best_val_loss):.2e}",
        'epochs': epoch + 1,
    }

    apply_fn = lambda p, xx: model.apply(p, xx, train=False)
    predict_fn = jax.jit(lambda x: jax.nn.sigmoid(apply_fn(best_params, x).squeeze()))
    return predict_fn, best_params, metrics

def train_ellipsoid_multiple_restarts(x: jnp.ndarray, y: jnp.ndarray, **kwargs):
    """Wrapper for ellipsoid training with restarts"""
    return train_with_restarts(train_ellipsoid, x, y, **kwargs)

def ellipsoid_predict(x: jnp.ndarray, params: Dict, apply_fn: Callable) -> jnp.ndarray:
    """Predict logits for single or batch input"""
    return apply_fn(params, x if x.ndim > 1 else x[None, :]).squeeze()

def ellipsoid_predict_proba(x: jnp.ndarray, params: Dict, apply_fn: Callable) -> jnp.ndarray:
    """Predict probabilities for single or batch input"""
    logits = ellipsoid_predict(x, params, apply_fn)
    return jax.nn.sigmoid(logits)