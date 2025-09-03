# interfaces and routines for some classifiers
# SVM, Neural Networks, Ellipsoidal bound, etc.

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sklearn.svm import SVC
import optax
from typing import Callable, Dict, Any, Union, List, Optional, Tuple
from functools import partial
from .utils.logging_utils import get_logger 
from .utils.seed_utils import get_numpy_rng
log = get_logger("clf")

try:
    from flax import linen as nn
except ImportError:
    log.warning("Flax is not available. Only SVM classifier will be available.")

# -----------------------------------------------------------------------------
# Scikit-learn SVM Classifier using JAX for prediction.
# -----------------------------------------------------------------------------

class SVMClassifier:

    def __init__(self, gamma: str = "scale", C: float = 1e7, kernel: str = 'rbf'):
        self.gamma = gamma
        self.C = C
        self.kernel = kernel

    def train(self,X,Y,init_params=None):
        clf = SVC(kernel=self.kernel, gamma=self.gamma, C=self.C)
        clf.fit(X, Y) 
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
            'C': f"{self.C:.2e}",
            'intercept': f"{intercept:.2e}",}
        params = {
            'support_vectors': support_vectors,
            'dual_coef': dual_coef,
            'intercept': intercept,
            'gamma_eff': gamma_eff
        }
        return params, metrics       

    def get_predict_proba_fn(self,params):
        support_vectors = params['support_vectors']
        dual_coef = params['dual_coef']
        intercept = params['intercept']
        gamma = params['gamma_eff']
        return jax.jit(partial(svm_predict_proba, support_vectors=support_vectors, dual_coef=dual_coef, intercept=intercept, gamma=gamma))

def svm_predict(x: jnp.ndarray, support_vectors: jnp.ndarray, dual_coef: jnp.ndarray, intercept: float, gamma: float):
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

def svm_predict_proba(x: jnp.ndarray, support_vectors: jnp.ndarray, dual_coef: jnp.ndarray, intercept: float, gamma: float):
    decision = svm_predict(x, support_vectors, dual_coef, intercept, gamma)
    return jnp.where(decision >= 0, 1.0, 0.0)  # Binary classification: 1 if decision >= 0, else 0


# -----------------------------------------------------------------------------
# Neural Network Classifiers with train/validation split
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
    n_restarts: int = 2,
    seed_offset: int = 0,
    split_seed: int = 42,
    init_params = None,  # Add this parameter
    **train_kwargs
) -> Tuple[Dict, Dict]:
    """
    Train model with multiple restarts using consistent train/val split.
    
    Args:
        train_fn: Training function that returns (params, metrics)
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
    best_params = None
    best_metrics = {}

    try:
        rng = get_numpy_rng()
    except Exception as e:
        log.error(f"{e} - falling back to default RNG")
        rng = np.random.default_rng()

    for i in range(n_restarts):
        current_seed = rng.integers(0, 2**32 - 1)
        log.debug(f"[Restart {i+1}/{n_restarts}] Starting training with seed {current_seed}")
        
        # Use initial params for first restart, None for others
        restart_init_params = init_params if i == 0 else None
        
        if i == 0 and init_params is not None:
            log.debug(f"[Restart {i+1}/{n_restarts}] Using provided initial parameters")
        elif i > 0:
            log.debug(f"[Restart {i+1}/{n_restarts}] Using random initialization")

        # Pass the pre-split data to avoid re-splitting
        params, metrics = train_fn(
            x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val,
            seed=current_seed,
            init_params=restart_init_params,  # Pass init_params to training function
            **train_kwargs
        )

        val_loss = float(metrics['val_loss'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            best_metrics = metrics
            log.debug(f"[Restart {i+1}/{n_restarts}] New best val_loss: {val_loss:.4e}")

    log.debug(f"[Training] Best model selected with val_loss = {best_val_loss:.4e}")
    return best_params, best_metrics


# Neural Network Classifier
class MLPClassifier(nn.Module):
    hidden_dims: list = (64, 64)
    dropout_rate: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 1000
    batch_size: int = 128
    early_stop_patience: int = 50
    n_restarts: int = 2
    val_frac: float = 0.2
    seed_offset: int = 0
    split_seed: int = 42

    @nn.compact
    def __call__(self, x, train: bool = False):
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(1)(x)
        return x

    def train(self, X, Y, init_params=None, **kwargs):
        params, metrics = train_nn_multiple_restarts(
            model=self,
            x=X, y=Y,
            init_params=init_params
        )
        return params, metrics

    def get_predict_proba_fn(self, params):
        def predict_proba_fn(x):
            logits = self.apply(params, x, train=False)
            return jax.nn.sigmoid(logits.squeeze(-1))
        return jax.jit(predict_proba_fn)

def train_nn(
    model: MLPClassifier,
    x_train: jnp.ndarray, y_train: jnp.ndarray,
    x_val: jnp.ndarray, y_val: jnp.ndarray,
    seed=0,
    init_params=None,  # Add this parameter
    **kwargs
):
    """Simplified NN training with pre-split data"""
    N, d = x_train.shape
    
    # Handle initialization
    if init_params is not None:
        params = init_params
    else:
        key = jax.random.PRNGKey(seed)
        params = model.init(key, jnp.ones((1, d)), train=True)

    optimizer = optax.adamw(model.lr, weight_decay=model.weight_decay)
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
    steps = max(1, x_train.shape[0] // model.batch_size)

    rng_opt = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    for epoch in range(model.n_epochs):
        perm_train = rng_opt.permutation(x_train.shape[0])
        for i in range(steps):
            idx = perm_train[i*model.batch_size:(i+1)*model.batch_size]
            bx = jnp.array(x_np[idx])
            by = jnp.array(y_np[idx])
            key, subkey = jax.random.split(key)
            params, opt_state = train_step(params, opt_state, bx, by, subkey)
            
        val_loss = compute_val_loss(params)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            patience = model.early_stop_patience
        else:
            patience -= 1
            if patience <= 0:
                log.debug(f"[NN] early stopping at epoch {epoch}")
                break

    metrics = {
        'train_loss': f"{float(loss_fn(best_params, x_train, y_train, key)):.2e}",
        'val_loss': f"{float(best_val_loss):.2e}",
        'epochs': epoch + 1,
    }

    return best_params, metrics

def train_nn_multiple_restarts(model: MLPClassifier, x: jnp.ndarray, y: jnp.ndarray, **kwargs):
    """Wrapper for NN training with restarts"""
    train_kwargs = {
        'val_frac': model.val_frac,
    }
    return train_with_restarts(partial(train_nn, model), x, y, 
                               n_restarts=model.n_restarts, 
                               seed_offset=model.seed_offset, 
                               split_seed=model.split_seed, **train_kwargs)

# Ellipsoid Classifier with center at best fit point
class EllipsoidClassifier(nn.Module):
    d: int
    mu: jnp.ndarray
    init_scale: float = 1.0
    lr: float = 1e-2
    weight_decay: float = 1e-4
    n_epochs: int = 1000
    batch_size: int = 32
    patience: int = 50
    n_restarts: int = 2
    val_frac: float = 0.2
    seed_offset: int = 0
    split_seed: int = 42

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

    def train(self, X, Y, init_params=None, **kwargs):
        self.mu = kwargs.get('best_pt', self.mu)
        params, metrics = train_ellipsoid_multiple_restarts(
            model=self,
            x=X, y=Y,
            init_params=init_params,
        )
        return params, metrics

    def get_predict_proba_fn(self, params):
        def predict_proba_fn(x):
            logits = self.apply(params, x, train=False)
            return jax.nn.sigmoid(logits.squeeze(-1))
        return jax.jit(predict_proba_fn)

def train_ellipsoid(
    model: EllipsoidClassifier,
    x_train: jnp.ndarray, y_train: jnp.ndarray,
    x_val: jnp.ndarray, y_val: jnp.ndarray,
    seed: int = 0,
    init_params=None,  # Add this parameter
    **kwargs
):
    """Simplified ellipsoid training with pre-split data"""
    # Handle initialization
    if init_params is not None:
        params = init_params
    else:
        key = jax.random.PRNGKey(seed)
        params = model.init(key, x_train)
    
        
    optimizer = optax.adamw(model.lr, weight_decay=model.weight_decay)
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
    steps = max(1, x_train.shape[0] // model.batch_size)
    patience_counter = 0
    
    rng = np.random.RandomState(seed)

    for epoch in range(model.n_epochs):
        perm_train = rng.permutation(x_train.shape[0])
        for i in range(steps):
            idx = perm_train[i*model.batch_size:(i+1)*model.batch_size]
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
            if patience_counter > model.patience:
                log.debug(f"Early stopping at epoch {epoch}")
                break

    train_loss = loss_fn(best_params, x_train, y_train)
    metrics = {
        'train_loss': f"{float(train_loss):.2e}",
        'val_loss': f"{float(best_val_loss):.2e}",
        'epochs': epoch + 1,
    }

    return best_params, metrics

def train_ellipsoid_multiple_restarts(model: EllipsoidClassifier, x: jnp.ndarray, y: jnp.ndarray, **kwargs):
    """Wrapper for ellipsoid training with restarts"""
    train_kwargs = {
        'val_frac': model.val_frac,
    }
    return train_with_restarts(partial(train_ellipsoid, model), x, y, 
                               n_restarts=model.n_restarts, 
                               seed_offset=model.seed_offset, 
                               split_seed=model.split_seed, **train_kwargs)