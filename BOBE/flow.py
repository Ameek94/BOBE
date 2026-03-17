"""
Normalising Flow implementation for BOBE using JAX + Flax (linen).

Architecture: Real NVP (Dinh et al. 2017) — alternating affine coupling layers.

Each coupling layer splits dimensions with a binary mask:
  - Masked dims pass through unchanged.
  - Unmasked dims are affinely transformed: y = x * exp(s(x_masked)) + t(x_masked).
The Jacobian is triangular, so log|det J| = sum(s) over unmasked dims.

Main external API
-----------------
NormalisingFlow
    .fit(x, weights=None, ...)   – train on samples
    .to_latent(x)                – data → z ~ N(0, I)
    .to_data(z)                  – z   → data space
    .log_prob(x)                 – log p(x) under the flow
    .state_dict() / .from_state_dict(state)  – serialisation

The preprocessing step centres / scales the data by empirical mean/std before
passing it to the coupling network, improving training stability.
"""

import pickle
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

jax.config.update("jax_enable_x64", True)

from .utils.log import get_logger

log = get_logger("flow")

_LOG_2PI = float(np.log(2.0 * np.pi))

# -------------------------------------------------------------------------
# Flax modules
# -------------------------------------------------------------------------


class _ST_Net(nn.Module):
    """Scale-translate MLP used inside each coupling layer.

    Parameters
    ----------
    hidden_dim : int
    output_dim : int  (= ndim of the full data)
    """

    hidden_dim: int
    output_dim: int

    def setup(self):
        self.fc1 = nn.Dense(self.hidden_dim,
                            param_dtype=jnp.float64, dtype=jnp.float64)
        self.fc2 = nn.Dense(self.hidden_dim,
                            param_dtype=jnp.float64, dtype=jnp.float64)
        self.fc_out = nn.Dense(2 * self.output_dim,
                               param_dtype=jnp.float64, dtype=jnp.float64)

    def __call__(self, x):
        h = jnp.tanh(self.fc1(x))
        h = jnp.tanh(self.fc2(h))
        out = self.fc_out(h)
        s, t = jnp.split(out, 2, axis=-1)
        # Bound scale to (-2, 2) for numerical stability (exp stays in a safe range)
        s = 2.0 * jnp.tanh(s)
        return s, t


class _CouplingLayer(nn.Module):
    """Affine coupling layer (Real NVP style).

    Parameters
    ----------
    mask : tuple of int (0 or 1), length == ndim
        Dimensions where mask == 1 pass through unchanged.
        Dimensions where mask == 0 are affinely transformed.
    hidden_dim : int
    """

    mask: tuple
    hidden_dim: int

    def setup(self):
        ndim = len(self.mask)
        self.st_net = _ST_Net(hidden_dim=self.hidden_dim, output_dim=ndim)

    def __call__(self, x):
        """Forward pass: x → y, returns (y, log|det J|)."""
        mask = jnp.array(self.mask, dtype=x.dtype)
        inv_mask = 1.0 - mask

        x_m = x * mask                  # pass-through part
        s, t = self.st_net(x_m)         # network sees only masked part
        s = s * inv_mask                # zero out scale on masked dims
        t = t * inv_mask                # zero out translate on masked dims

        y = x_m + inv_mask * (x * jnp.exp(s) + t)
        log_det = jnp.sum(s, axis=-1)   # shape (batch,)
        return y, log_det

    def inverse(self, y):
        """Exact inverse: y → x (single forward pass through network)."""
        mask = jnp.array(self.mask, dtype=y.dtype)
        inv_mask = 1.0 - mask

        y_m = y * mask
        s, t = self.st_net(y_m)
        s = s * inv_mask
        t = t * inv_mask

        x = y_m + inv_mask * ((y - t) * jnp.exp(-s))
        return x


class _RealNVP(nn.Module):
    """Stack of alternating affine coupling layers.

    Parameters
    ----------
    ndim : int
    n_layers : int
    hidden_dim : int
    """

    ndim: int
    n_layers: int = 8
    hidden_dim: int = 64

    def setup(self):
        # Alternate checkerboard masks across layers
        masks = [
            tuple(int((j + i) % 2) for j in range(self.ndim))
            for i in range(self.n_layers)
        ]
        self.layers = [
            _CouplingLayer(mask=masks[i], hidden_dim=self.hidden_dim)
            for i in range(self.n_layers)
        ]

    def __call__(self, x):
        """Forward: x → z ~ N(0, I).  Returns (z, sum_log_det)."""
        z = x
        log_det = jnp.zeros(x.shape[0], dtype=x.dtype)
        for layer in self.layers:
            z, ld = layer(z)
            log_det = log_det + ld
        return z, log_det

    def inverse(self, z):
        """Inverse: z → x (data space)."""
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x


# -------------------------------------------------------------------------
# Python wrapper
# -------------------------------------------------------------------------


class NormalisingFlow:
    """
    Normalising flow (Real NVP) with training and inference.

    The flow maps data → z ~ N(0, I) via:
      1. Centering / scaling  : x_norm = (x - mu) / sigma   (fitted to data)
      2. Coupling network     : z = flow(x_norm)

    Parameters
    ----------
    ndim : int
    n_layers : int
        Number of coupling layers. Default 8. For low-D data 6–8 is enough.
    hidden_dim : int
        Hidden layer width. Default 64.
    n_sigma : float
        When used inside NormalisingFlowTransform, z is expected to lie in
        [-n_sigma, n_sigma] per dimension.  Default 5.
    seed : int
        JAX PRNG seed for parameter initialisation and training shuffle.
    """

    def __init__(
        self,
        ndim: int,
        n_layers: int = 8,
        hidden_dim: int = 64,
        n_sigma: float = 5.0,
        seed: int = 42,
    ):
        self.ndim = ndim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_sigma = n_sigma
        self._seed = seed

        self._model = _RealNVP(ndim=ndim, n_layers=n_layers, hidden_dim=hidden_dim)
        self._params: Optional[dict] = None
        self._pre_mean = np.zeros(ndim, dtype=np.float64)
        self._pre_std = np.ones(ndim, dtype=np.float64)
        self._trained = False

    # ------------------------------------------------------------------
    # Internal preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, x: np.ndarray) -> jnp.ndarray:
        return jnp.array((x - self._pre_mean) / self._pre_std, dtype=jnp.float64)

    def _unpreprocess(self, x_norm) -> np.ndarray:
        return np.array(x_norm, dtype=np.float64) * self._pre_std + self._pre_mean

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        weights: Optional[np.ndarray] = None,
        n_epochs: int = 2000,
        lr: float = 3e-4,
        batch_size: Optional[int] = 512,
        verbose: bool = True,
        seed: Optional[int] = None,
        patience: int = 100,
        early_stop_delta: float = 1e-4,
    ):
        """Train the flow to map x → N(0, I).

        Parameters
        ----------
        x : (N, D) array_like
            Training samples in physical / data space.
        weights : (N,) array_like, optional
            Positive sample weights (e.g. from importance sampling).
        n_epochs : int
            Training epochs. Default 2000.
        lr : float
            Peak Adam learning rate. A cosine decay schedule is applied.
        batch_size : int or None
            Mini-batch size.  None = full batch.
        verbose : bool
            Log NLL every 200 epochs.
        seed : int, optional
            Override the instance seed.
        patience : int
            Number of consecutive epochs without an NLL improvement of at least
            ``early_stop_delta`` before training is halted. Default 100.
        early_stop_delta : float
            Minimum absolute NLL decrease that resets the patience counter.
            Default 1e-4.
        """
        if seed is not None:
            self._seed = seed

        x = np.asarray(x, dtype=np.float64)
        N, D = x.shape
        if D != self.ndim:
            raise ValueError(f"Expected {self.ndim}-D data, got {D}")

        # Compute whitening statistics
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            w = np.clip(w, 0.0, None)
            w /= w.sum()
            self._pre_mean = np.average(x, weights=w, axis=0)
            diff = x - self._pre_mean
            self._pre_std = np.sqrt(np.average(diff ** 2, weights=w, axis=0))
        else:
            self._pre_mean = x.mean(axis=0)
            self._pre_std = x.std(axis=0)
        self._pre_std = np.maximum(self._pre_std, 1e-6)

        x_norm = self._preprocess(x)  # (N, D) float64 jax array

        # Initialise model parameters
        key = jax.random.PRNGKey(self._seed)
        key, init_key = jax.random.split(key)
        self._params = self._model.init(init_key, x_norm[:2])

        bs = N if batch_size is None else min(batch_size, N)
        n_full = max(1, N // bs)   # number of complete batches per epoch

        # Optax: cosine LR decay + gradient clipping
        total_steps = n_epochs * n_full
        sched = optax.cosine_decay_schedule(lr, total_steps, alpha=0.05)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(sched),
        )
        opt_state = optimizer.init(self._params)

        D_float = float(D)

        def _loss(params, batch):
            z, log_det = self._model.apply(params, batch)
            log_pz = -0.5 * jnp.sum(z * z, axis=-1) - 0.5 * D_float * _LOG_2PI
            return -(log_pz + log_det).mean()

        def _scan_step(carry, batch):
            """Single gradient step used as the lax.scan body over batches."""
            params, opt_state = carry
            loss, grads = jax.value_and_grad(_loss)(params, batch)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return (new_params, new_opt_state), loss

        @jax.jit
        def _epoch(params, opt_state, x_batches):
            """Run one full epoch over (n_full, bs, D) batches via lax.scan.

            All gradient steps execute inside a single XLA program, keeping
            params and opt_state on-device between batches and allowing XLA
            to fuse memory operations across steps.
            """
            (params, opt_state), losses = jax.lax.scan(
                _scan_step, (params, opt_state), x_batches
            )
            return params, opt_state, losses.mean()

        best_nll = float("inf")
        no_improve = 0

        for epoch in range(n_epochs):
            key, shuffle_key = jax.random.split(key)
            perm = jax.random.permutation(shuffle_key, N)
            # Truncate to n_full complete batches and reshape to (n_full, bs, D)
            x_batches = x_norm[perm[: n_full * bs]].reshape(n_full, bs, D)

            self._params, opt_state, avg_loss = _epoch(
                self._params, opt_state, x_batches
            )
            epoch_nll = float(avg_loss)

            if verbose and (epoch + 1) % 200 == 0:
                log.info(
                    f"Flow epoch {epoch + 1:4d}/{n_epochs}  "
                    f"NLL = {epoch_nll:.4f}"
                )

            # Early stopping
            if epoch_nll < best_nll - early_stop_delta:
                best_nll = epoch_nll
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                log.info(
                    f"Flow early stop at epoch {epoch + 1}/{n_epochs}  "
                    f"best NLL = {best_nll:.4f}"
                )
                break

        self._trained = True
        log.info("Flow training complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def to_latent(self, x: np.ndarray) -> np.ndarray:
        """Map data x → standard Normal latent z.

        Parameters
        ----------
        x : (..., D) array_like  —  physical / data space.

        Returns
        -------
        z : same shape as x  —  latent space (approximately N(0, I)).
        """
        self._check_trained()
        x = np.asarray(x, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[np.newaxis]
        z, _ = self._model.apply(self._params, self._preprocess(x))
        z = np.array(z, dtype=np.float64)
        return z[0] if single else z

    def to_data(self, z: np.ndarray) -> np.ndarray:
        """Map standard Normal z → data / physical space.

        Parameters
        ----------
        z : (..., D) array_like  —  latent space.

        Returns
        -------
        x : same shape as z  —  data space.
        """
        self._check_trained()
        z = np.asarray(z, dtype=np.float64)
        single = z.ndim == 1
        if single:
            z = z[np.newaxis]
        x_norm = self._model.apply(
            self._params, jnp.array(z, dtype=jnp.float64),
            method=self._model.inverse,
        )
        x = self._unpreprocess(x_norm)
        return x[0] if single else x

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """Log probability density log p(x) under the flow model.

        Parameters
        ----------
        x : (..., D) array_like

        Returns
        -------
        log_p : scalar (if single point) or (N,) array.
        """
        self._check_trained()
        x = np.asarray(x, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[np.newaxis]
        x_norm = self._preprocess(x)
        z, log_det = self._model.apply(self._params, x_norm)
        log_pz = -0.5 * jnp.sum(z * z, axis=-1) - 0.5 * self.ndim * _LOG_2PI
        # Jacobian from whitening transform: -sum(log(pre_std))
        log_p = log_pz + log_det - float(np.sum(np.log(self._pre_std)))
        log_p = np.array(log_p, dtype=np.float64)
        return float(log_p[0]) if single else log_p

    def forward_with_logdet(self, x: np.ndarray):
        """Return (z, log_det_jacobian) for x → z.

        log_det_jacobian here is the *full* change-of-variables term including
        the preprocessing Jacobian: log |dz/dx| = log_det_net - sum(log(pre_std)).
        """
        self._check_trained()
        x = np.asarray(x, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[np.newaxis]
        x_norm = self._preprocess(x)
        z, log_det_net = self._model.apply(self._params, x_norm)
        log_det = np.array(log_det_net, dtype=np.float64) - float(
            np.sum(np.log(self._pre_std))
        )
        z = np.array(z, dtype=np.float64)
        if single:
            return z[0], float(log_det[0])
        return z, log_det

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return a serialisable dict (params stored as nested dict of arrays)."""
        return {
            "ndim": self.ndim,
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "n_sigma": self.n_sigma,
            "seed": self._seed,
            "trained": self._trained,
            "pre_mean": self._pre_mean.copy(),
            "pre_std": self._pre_std.copy(),
            "params": self._params,       # JAX pytree — must be pickled separately
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "NormalisingFlow":
        obj = cls(
            ndim=int(state["ndim"]),
            n_layers=int(state["n_layers"]),
            hidden_dim=int(state["hidden_dim"]),
            n_sigma=float(state["n_sigma"]),
            seed=int(state["seed"]),
        )
        obj._trained = bool(state["trained"])
        obj._pre_mean = np.array(state["pre_mean"], dtype=np.float64)
        obj._pre_std = np.array(state["pre_std"], dtype=np.float64)
        obj._params = state["params"]
        return obj

    def save(self, path: str):
        """Pickle the full flow state to ``path``."""
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)
        log.info(f"NormalisingFlow saved to {path}")

    @classmethod
    def load(cls, path: str) -> "NormalisingFlow":
        """Load a saved NormalisingFlow from ``path``."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        log.info(f"NormalisingFlow loaded from {path}")
        return cls.from_state_dict(state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_trained(self):
        if not self._trained or self._params is None:
            raise RuntimeError(
                "NormalisingFlow has not been trained. Call .fit() first."
            )

    def __repr__(self):
        status = "trained" if self._trained else "untrained"
        return (
            f"NormalisingFlow(ndim={self.ndim}, n_layers={self.n_layers}, "
            f"hidden_dim={self.hidden_dim}, {status})"
        )
