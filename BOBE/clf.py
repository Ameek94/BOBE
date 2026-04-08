# interfaces and routines for some classifiers
# SVM, Neural Networks, Ellipsoidal bound, etc.

import os
import tempfile
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sklearn.svm import SVC
from typing import Callable, Dict, Any, Union, List, Optional, Tuple
from functools import partial
from .utils.log import get_logger
from .utils.seed import get_numpy_rng
log = get_logger("clf")

try:
    import optax
    OPTAX_AVAILABLE = True
except ImportError:
    OPTAX_AVAILABLE = False
    optax = None
    log.debug("optax is not available. NN and Ellipsoid classifiers will require it.")

try:
    import equinox as eqx
    EQX_AVAILABLE = True
except ImportError:
    EQX_AVAILABLE = False
    eqx = None
    log.debug("Equinox is not available. Only SVM classifier will be available.")


# -----------------------------------------------------------------------------
# Standalone training and prediction functions for classifiers
# -----------------------------------------------------------------------------

def train_svm_classifier(X, Y, settings = {}, init_params=None, **kwargs):
    """Train SVM classifier and return parameters, metrics, and predict function."""
    gamma = settings.get('gamma', 'scale')
    C = settings.get('C', 1e7)
    kernel = settings.get('kernel', 'rbf')

    clf = SVC(kernel=kernel, gamma=gamma, C=C)
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
        'C': f"{C:.2e}",
        'intercept': f"{intercept:.2e}",
    }
    params = {
        'support_vectors': support_vectors,
        'dual_coef': dual_coef,
        'intercept': intercept,
        'gamma_eff': gamma_eff
    }

    # Create predict function
    predict_fn = jax.jit(partial(svm_predict_proba, support_vectors=support_vectors,
                                dual_coef=dual_coef, intercept=intercept, gamma=gamma_eff))

    return params, metrics, predict_fn

def get_svm_predict_proba_fn(params, settings=None, **kwargs):
    """Get prediction function for SVM classifier from parameters (for loading from file)."""
    support_vectors = params['support_vectors']
    dual_coef = params['dual_coef']
    intercept = params['intercept']
    gamma = params['gamma_eff']
    return jax.jit(partial(svm_predict_proba, support_vectors=support_vectors,
                          dual_coef=dual_coef, intercept=intercept, gamma=gamma))


# -----------------------------------------------------------------------------
# Neural Network Classifier

def train_nn_classifier(X, Y, settings = {}, init_params=None, **kwargs):
    """Train neural network classifier and return parameters, metrics, and predict function."""
    if not EQX_AVAILABLE or not OPTAX_AVAILABLE:
        raise ImportError("Equinox and optax are required for NN classifier. "
            "Install with: pip install 'BOBE[nn]'"
        )
    d = X.shape[1]
    label_size = X.shape[0]

    # Create a local copy so we don't mutate the caller's dict
    nn_settings = dict(settings)
    if label_size < 500:
        nn_settings['hidden_dims'] = [64, 64]
        nn_settings['batch_size'] = 64
    else:
        nn_settings['hidden_dims'] = [64, 64]
        nn_settings['batch_size'] = 128

    template_key = jax.random.PRNGKey(0)
    model = MLPClassifier(input_dim=d, key=template_key, **nn_settings)

    # Train with multiple restarts
    trained_model, metrics = train_nn_multiple_restarts(
        model=model,
        x=X, y=Y,
        init_params=init_params
    )

    # Create predict function (no key → inference, dropout disabled)
    predict_fn = jax.jit(lambda x: jax.nn.sigmoid(trained_model(x).squeeze(-1)))

    return trained_model, metrics, predict_fn

def get_nn_predict_proba_fn(model, settings=None, **kwargs):
    """Get prediction function for NN classifier from equinox model (for loading from file)."""
    def predict_proba_fn(x):
        return jax.nn.sigmoid(model(x).squeeze(-1))
    return jax.jit(predict_proba_fn)

def train_ellipsoid_classifier(X, Y, settings = {}, init_params=None, **kwargs):
    """Train ellipsoid classifier and return parameters, metrics, and predict function."""
    if not OPTAX_AVAILABLE:
        raise ImportError(
            "optax is required for Ellipsoid classifier. "
            "Install with: pip install 'BOBE[nn]'"
        )
    d = X.shape[1]
    mu = kwargs.get('best_pt', 0.5 * jnp.ones(d))

    # Filter out 'd' and 'mu' keys if present to avoid constructor conflicts
    constructor_settings = {k: v for k, v in settings.items() if k not in ('d', 'mu')}
    model = EllipsoidClassifier(d=d, mu=mu, **constructor_settings)

    # Train with multiple restarts
    trained_params, metrics = train_ellipsoid_multiple_restarts(
        model=model,
        x=X, y=Y,
        init_params=init_params,
    )

    predict_fn = jax.jit(lambda x: jax.nn.sigmoid(_ellipsoid_forward(trained_params, x).squeeze()))

    return trained_params, metrics, predict_fn

def get_ellipsoid_predict_proba_fn(params):
    """Get prediction function for ellipsoid classifier from params dict (for loading from file)."""
    return jax.jit(lambda x: jax.nn.sigmoid(_ellipsoid_forward(params, x).squeeze()))

# Dictionary mapping classifier types to their functions
CLASSIFIER_REGISTRY = {
    'svm': {
        'train_fn': train_svm_classifier,
        'predict_fn': get_svm_predict_proba_fn,
    },
    'nn': {
        'train_fn': train_nn_classifier,
        'predict_fn': get_nn_predict_proba_fn,
    },
    'ellipsoid': {
        'train_fn': train_ellipsoid_classifier,
        'predict_fn': get_ellipsoid_predict_proba_fn,
    }
}

# -----------------------------------------------------------------------------
# SVM prediction functions
# -----------------------------------------------------------------------------

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
# Neural Network Classifiers
# -----------------------------------------------------------------------------

# Common training utilities
def train_with_restarts(
    train_fn: Callable,
    x: jnp.ndarray,
    y: jnp.ndarray,
    n_restarts: int = 2,
    seed_offset: int = 0,
    split_seed: int = 42,
    init_params = None,
    **train_kwargs
) -> Tuple[Dict, Dict]:
    """
    Train model with multiple restarts using the entire dataset.

    Args:
        train_fn: Training function that returns (params, metrics)
        x: (N, d) features
        y: (N,) labels
        n_restarts: number of random restarts
        seed_offset: offset for training seed generation
        split_seed: fixed seed for train/val split consistency (unused now)
        init_params: initial parameters for first restart
        **train_kwargs: passed to train_fn
    """
    best_train_loss = jnp.inf
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

        # Use entire dataset for training
        params, metrics = train_fn(
            x_train=x, y_train=y,
            seed=current_seed,
            init_params=restart_init_params,
            **train_kwargs
        )

        train_loss = float(metrics['train_loss'])

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_params = params
            best_metrics = metrics
            log.debug(f"[Restart {i+1}/{n_restarts}] New best train_loss: {train_loss:.4e}")

    log.debug(f"[Training] Best model selected with train_loss = {best_train_loss:.4e}")
    return best_params, best_metrics


# -----------------------------------------------------------------------------
# MLPClassifier (Equinox)
# -----------------------------------------------------------------------------

if EQX_AVAILABLE:
    class MLPClassifier(eqx.Module):
        """MLP binary classifier implemented as an equinox module.

        The model contains both its architecture (as static fields) and
        its trainable weights (as JAX array leaves).  Calling with
        ``key=None`` disables dropout (inference mode).
        """
        linear_layers: list
        input_dim: int = eqx.field(static=True)
        hidden_dims: tuple = eqx.field(static=True)
        dropout_rate: float = eqx.field(static=True)
        lr: float = eqx.field(static=True)
        weight_decay: float = eqx.field(static=True)
        n_epochs: int = eqx.field(static=True)
        batch_size: int = eqx.field(static=True)
        early_stop_patience: int = eqx.field(static=True)
        n_restarts: int = eqx.field(static=True)
        val_frac: float = eqx.field(static=True)
        seed_offset: int = eqx.field(static=True)
        split_seed: int = eqx.field(static=True)

        def __init__(self, input_dim, hidden_dims=(32, 32), dropout_rate=0.1,
                     lr=1e-3, weight_decay=1e-4, n_epochs=1000, batch_size=128,
                     early_stop_patience=50, n_restarts=2, val_frac=0.2,
                     seed_offset=0, split_seed=42, *, key):
            self.input_dim = input_dim
            self.hidden_dims = tuple(hidden_dims)
            self.dropout_rate = dropout_rate
            self.lr = lr
            self.weight_decay = weight_decay
            self.n_epochs = n_epochs
            self.batch_size = batch_size
            self.early_stop_patience = early_stop_patience
            self.n_restarts = n_restarts
            self.val_frac = val_frac
            self.seed_offset = seed_offset
            self.split_seed = split_seed

            dims = [input_dim] + list(hidden_dims) + [1]
            keys = jax.random.split(key, len(dims) - 1)
            self.linear_layers = [
                eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
                for i in range(len(dims) - 1)
            ]

        def __call__(self, x, key=None):
            """Forward pass.  Pass ``key`` during training to enable dropout."""
            for linear in self.linear_layers[:-1]:
                x = jax.nn.relu(linear(x))
                if key is not None and self.dropout_rate > 0.0:
                    key, subkey = jax.random.split(key)
                    mask = jax.random.bernoulli(subkey, 1.0 - self.dropout_rate, x.shape)
                    x = jnp.where(mask, x / (1.0 - self.dropout_rate), 0.0)
            return self.linear_layers[-1](x)

        def reinit(self, key):
            """Return a fresh copy with the same hyperparams but resampled weights."""
            return MLPClassifier(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                dropout_rate=self.dropout_rate,
                lr=self.lr,
                weight_decay=self.weight_decay,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                early_stop_patience=self.early_stop_patience,
                n_restarts=self.n_restarts,
                val_frac=self.val_frac,
                seed_offset=self.seed_offset,
                split_seed=self.split_seed,
                key=key,
            )
else:
    MLPClassifier = None


def train_nn(
    model,
    x_train: jnp.ndarray, y_train: jnp.ndarray,
    seed=0,
    init_params=None,
    **kwargs
):
    """Train equinox MLPClassifier with early stopping on a held-out val split."""
    N, d = x_train.shape

    # Initialise weights: warm-start from init_params or draw fresh weights
    if init_params is not None:
        current_model = init_params
    else:
        key = jax.random.PRNGKey(seed)
        current_model = model.reinit(key)

    optimizer = optax.adamw(current_model.lr, weight_decay=current_model.weight_decay)
    opt_state = optimizer.init(eqx.filter(current_model, eqx.is_array))

    # --- loss (with dropout, used during training) ---
    @eqx.filter_jit
    def loss_fn(m, batch_x, batch_y, key):
        keys = jax.random.split(key, batch_x.shape[0])
        logits = jax.vmap(lambda xi, ki: m(xi, key=ki))(batch_x, keys)
        return optax.sigmoid_binary_cross_entropy(logits.squeeze(-1), batch_y).mean()

    # --- loss without dropout (for validation and final metric) ---
    @eqx.filter_jit
    def inference_loss(m, x, y):
        logits = jax.vmap(m)(x)
        return optax.sigmoid_binary_cross_entropy(logits.squeeze(-1), y).mean()

    @eqx.filter_jit
    def train_step(m, opt_state, batch_x, batch_y, key):
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(m, batch_x, batch_y, key)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(m, eqx.is_array)
        )
        new_m = eqx.apply_updates(m, updates)
        return new_m, new_opt_state, loss_val

    x_np, y_np = np.array(x_train), np.array(y_train)

    # Train/val split for early stopping
    val_frac = current_model.val_frac
    patience = current_model.early_stop_patience
    use_early_stopping = patience > 0 and val_frac > 0.0 and N > 4

    if use_early_stopping:
        rng_split = np.random.default_rng(kwargs.get('split_seed', 42))
        n_val = max(1, int(N * val_frac))
        val_idx = rng_split.choice(N, size=n_val, replace=False)
        train_idx = np.setdiff1d(np.arange(N), val_idx)
        x_tr, y_tr = jnp.array(x_np[train_idx]), jnp.array(y_np[train_idx])
        x_val, y_val = jnp.array(x_np[val_idx]), jnp.array(y_np[val_idx])
        N_tr = len(train_idx)
    else:
        x_tr, y_tr = jnp.array(x_np), jnp.array(y_np)
        N_tr = N

    steps = max(1, N_tr // current_model.batch_size)
    rng_opt = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    best_val_loss = jnp.inf
    best_model = current_model
    patience_counter = 0
    epochs_trained = 0

    for epoch in range(current_model.n_epochs):
        perm = rng_opt.permutation(N_tr)
        for i in range(steps):
            idx = perm[i * current_model.batch_size:(i + 1) * current_model.batch_size]
            bx = x_tr[idx]
            by = y_tr[idx]
            key, subkey = jax.random.split(key)
            current_model, opt_state, _ = train_step(current_model, opt_state, bx, by, subkey)

        epochs_trained = epoch + 1

        if use_early_stopping:
            val_loss = float(inference_loss(current_model, x_val, y_val))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = current_model
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log.debug(f"Early stopping at epoch {epochs_trained} (patience={patience})")
                    break
        else:
            best_model = current_model

    final_loss = float(inference_loss(best_model, x_train, y_train))
    metrics = {
        'train_loss': f"{final_loss:.2e}",
        'epochs': epochs_trained,
    }
    if use_early_stopping:
        metrics['val_loss'] = f"{best_val_loss:.2e}"
    return best_model, metrics


def train_nn_multiple_restarts(model, x: jnp.ndarray, y: jnp.ndarray, **kwargs):
    """Wrapper for NN training with restarts."""
    return train_with_restarts(
        partial(train_nn, model), x, y,
        n_restarts=model.n_restarts,
        seed_offset=model.seed_offset,
        split_seed=model.split_seed,
        **kwargs,
    )


# -----------------------------------------------------------------------------
# EllipsoidClassifier — plain Python (no equinox needed)
# -----------------------------------------------------------------------------

class EllipsoidClassifier:
    """Hyperparameter container for the ellipsoidal classifier.

    NOT a JAX pytree.  Trainable parameters (flat_L, alpha, beta) and the
    fixed centre (mu) live in a separate plain dict returned by training.
    """
    def __init__(self, d, mu, init_scale=0.1, lr=1e-2, weight_decay=1e-4,
                 n_epochs=1000, batch_size=64, patience=25, n_restarts=2,
                 val_frac=0.1, seed_offset=0, split_seed=42):
        self.d = d
        self.mu = jnp.array(mu).flatten()
        self.init_scale = init_scale
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.n_restarts = n_restarts
        self.val_frac = val_frac
        self.seed_offset = seed_offset
        self.split_seed = split_seed


def _ellipsoid_unpack_L(flat_L, d):
    """Build lower-triangular L from a flat representation (softplus diagonal)."""
    rows, cols = np.tril_indices(d)
    diagonal_mask = jnp.array(rows == cols)
    flat_L_processed = jnp.where(
        diagonal_mask,
        jax.nn.softplus(flat_L) + 1e-4,
        flat_L,
    )
    return jnp.zeros((d, d)).at[rows, cols].set(flat_L_processed)


def _ellipsoid_forward(params, x):
    """Compute the ellipsoid logit for a single point x.

    params keys: 'flat_L', 'alpha', 'beta', 'mu', 'd'
    """
    d = int(params['d'])
    L = _ellipsoid_unpack_L(jnp.array(params['flat_L']), d)
    mu = jnp.array(params['mu'])
    diff = x - mu
    md2 = jnp.einsum("i,ij,j->", diff, L @ L.T, diff)
    return -jnp.array(params['alpha']) * md2 + jnp.array(params['beta'])


def _make_ellipsoid_params(d, mu, init_scale, seed):
    """Initialise a fresh ellipsoid params dict."""
    key = jax.random.PRNGKey(seed)
    tril = d * (d + 1) // 2
    return {
        'flat_L': jax.random.normal(key, (tril,)) * init_scale,
        'alpha': jnp.ones(()),
        'beta': jnp.zeros(()),
        'mu': jnp.array(mu).flatten(),
        'd': d,
    }


def train_ellipsoid(
    model,
    x_train: jnp.ndarray, y_train: jnp.ndarray,
    seed: int = 0,
    init_params=None,
    **kwargs
):
    """Train EllipsoidClassifier using plain JAX + optax.

    mu is naturally frozen because it is not part of the grad-tracked dict.
    Returns a plain params dict and metrics.
    """
    mu = jnp.array(model.mu).flatten()
    d = int(model.d)

    if init_params is not None:
        # Warm-start: reuse weights, update mu/d to the current model
        trainable = {k: jnp.array(init_params[k]) for k in ('flat_L', 'alpha', 'beta')}
    else:
        p0 = _make_ellipsoid_params(d, mu, model.init_scale, seed)
        trainable = {k: p0[k] for k in ('flat_L', 'alpha', 'beta')}

    optimizer = optax.adamw(model.lr, weight_decay=model.weight_decay)
    opt_state = optimizer.init(trainable)

    def _loss(trainable, batch_x, batch_y):
        params = {**trainable, 'mu': mu, 'd': d}
        logits = jax.vmap(partial(_ellipsoid_forward, params))(batch_x)
        return optax.sigmoid_binary_cross_entropy(logits, batch_y).mean()

    @jax.jit
    def train_step(trainable, opt_state, bx, by):
        loss_val, grads = jax.value_and_grad(_loss)(trainable, bx, by)
        updates, new_opt_state = optimizer.update(grads, opt_state, trainable)
        return optax.apply_updates(trainable, updates), new_opt_state, loss_val

    x_np, y_np = np.array(x_train), np.array(y_train)
    steps = max(1, x_train.shape[0] // model.batch_size)
    rng = np.random.RandomState(seed)

    for epoch in range(model.n_epochs):
        perm = rng.permutation(x_train.shape[0])
        for i in range(steps):
            idx = perm[i * model.batch_size:(i + 1) * model.batch_size]
            bx = jnp.array(x_np[idx])
            by = jnp.array(y_np[idx])
            trainable, opt_state, _ = train_step(trainable, opt_state, bx, by)

    final_loss = _loss(trainable, x_train, y_train)
    metrics = {
        'train_loss': f"{float(final_loss):.2e}",
        'epochs': epoch + 1,
    }
    return {**trainable, 'mu': mu, 'd': d}, metrics


def train_ellipsoid_multiple_restarts(model, x: jnp.ndarray, y: jnp.ndarray, **kwargs):
    """Wrapper for ellipsoid training with restarts."""
    return train_with_restarts(
        partial(train_ellipsoid, model), x, y,
        n_restarts=model.n_restarts,
        seed_offset=model.seed_offset,
        split_seed=model.split_seed,
        **kwargs,
    )


# -----------------------------------------------------------------------------
# Equinox model serialisation helpers
# -----------------------------------------------------------------------------

def serialize_eqx_clf(model) -> np.ndarray:
    """Serialise an equinox classifier model to a uint8 numpy array.

    Uses :func:`equinox.tree_serialise_leaves` so the result is robust to
    pickle-format changes across Python versions.
    """
    with tempfile.NamedTemporaryFile(suffix='.eqx', delete=False) as f:
        tmp = f.name
    try:
        eqx.tree_serialise_leaves(tmp, model)
        with open(tmp, 'rb') as f:
            return np.frombuffer(f.read(), dtype=np.uint8).copy()
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def deserialize_eqx_clf(model_bytes: np.ndarray, template) -> object:
    """Deserialise an equinox classifier model from a uint8 numpy array.

    Parameters
    ----------
    model_bytes : np.ndarray
        Byte array produced by :func:`serialize_eqx_clf`.
    template :
        An equinox model with the **same structure** as the saved model.
        Leaf values are ignored — only the pytree structure matters.
    """
    with tempfile.NamedTemporaryFile(suffix='.eqx', delete=False) as f:
        tmp = f.name
    try:
        with open(tmp, 'wb') as f:
            f.write(bytes(model_bytes))
        return eqx.tree_deserialise_leaves(tmp, template)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
