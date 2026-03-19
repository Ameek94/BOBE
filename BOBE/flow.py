"""
Normalising Flow implementation for BOBE using flowjax.

Architecture: Masked Autoregressive Flow (MAF) with Rational Quadratic Spline
transformers, built on top of the ``flowjax`` library.

Bijection convention (flowjax ``invert=True``, the default):
  - ``flow.bijection.transform(z)``  : latent z → data x
  - ``flow.bijection.inverse(x)``    : data x → latent z

Main external API
-----------------
NormalisingFlow
    .fit(x, weights=None, ...)   – train on samples
    .to_latent(x)                – data → z ~ N(0, I)
    .to_data(z)                  – z   → data space
    .log_prob(x)                 – log p(x) under the flow
    .state_dict() / .from_state_dict(state)  – serialisation (via equinox)

The preprocessing step centres / scales the data by empirical mean/std before
passing it to the flow, improving training stability.

Requires: ``pip install BOBE[nn]``  →  flowjax, equinox, optax
"""

import os
import tempfile
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .utils.log import get_logger

log = get_logger("flow")


# -------------------------------------------------------------------------
# Python wrapper
# -------------------------------------------------------------------------


class NormalisingFlow:
    """
    Normalising flow (MAF via flowjax) with training and inference.

    The flow maps data → z ~ N(0, I) via:
      1. Centering / scaling  : x_norm = (x - mu) / sigma   (fitted to data)
      2. MAF bijection        : z = flow.bijection.inverse(x_norm)

    Parameters
    ----------
    ndim : int
    n_layers : int
        Number of MAF layers. Default 8.
    hidden_dim : int
        Hidden layer width in each autoregressive network. Default 64.
    nn_depth : int
        Depth of each autoregressive network. Default 2.
    n_sigma : float
        Whitened coordinates are expected to lie in [-n_sigma, n_sigma] per 
        dimension after pre-whitening transform. Default 5.
    seed : int
        JAX PRNG seed for parameter initialisation and training shuffle.
    """

    def __init__(
        self,
        ndim: int,
        n_layers: int = 8,
        hidden_dim: int = 64,
        nn_depth: int = 2,
        n_sigma: float = 5.0,
        seed: int = 42,
    ):
        self.ndim = ndim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.nn_depth = nn_depth
        self.n_sigma = n_sigma
        self._seed = seed

        self._flow = None          # flowjax Transformed distribution
        self._pre_mean = jnp.zeros(ndim, dtype=jnp.float64)
        self._pre_std = jnp.ones(ndim, dtype=jnp.float64)
        self._log_pre_std_sum = 0.0  # precomputed sum(log(std)) for efficiency
        self._trained = False
        
        # Calibration: offset to align flow log_prob with true log-likelihood
        self._calibration_offset = 0.0
        self._calibrated = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_flow(self, key):
        """Construct an untrained flowjax MAF with the configured architecture."""
        from flowjax.flows import masked_autoregressive_flow
        from flowjax.distributions import Normal

        base = Normal(jnp.zeros(self.ndim), jnp.ones(self.ndim))
        return masked_autoregressive_flow(
            key=key,
            base_dist=base,
            flow_layers=self.n_layers,
            nn_width=self.hidden_dim,
            nn_depth=self.nn_depth,
        )

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
            Positive sample weights (e.g. from importance sampling). Used only
            for computing the whitening statistics; flowjax ``fit_to_data`` does
            not support weighted NLL natively.
        n_epochs : int
            Maximum training epochs. Default 2000.
        lr : float
            Peak Adam learning rate. Default 3e-4.
        batch_size : int or None
            Mini-batch size.  None = full batch.
        verbose : bool
            Log val NLL every 200 epochs.
        seed : int, optional
            Override the instance seed for this call.
        patience : int
            ``max_patience`` passed to ``fit_to_data`` (early stopping).
        early_stop_delta : float
            Unused (kept for API compatibility; flowjax uses its own criterion).
        """
        from flowjax.train import fit_to_data

        if seed is not None:
            self._seed = seed

        x = np.asarray(x, dtype=np.float64)
        N, D = x.shape
        if D != self.ndim:
            raise ValueError(f"Expected {self.ndim}-D data, got {D}")

        # Whitening statistics (optionally weighted)
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            w = np.clip(w, 0.0, None)
            w /= w.sum()
            pre_mean = np.average(x, weights=w, axis=0)
            diff = x - pre_mean
            pre_std = np.sqrt(np.average(diff ** 2, weights=w, axis=0))
        else:
            pre_mean = x.mean(axis=0)
            pre_std = x.std(axis=0)
        pre_std = np.maximum(pre_std, 1e-6)
        # Store as JAX arrays for JAX-compatible log_prob
        self._pre_mean = jnp.asarray(pre_mean)
        self._pre_std = jnp.asarray(pre_std)
        self._log_pre_std_sum = float(jnp.sum(jnp.log(self._pre_std)))

        x_norm = self._preprocess(x)

        key = jax.random.key(self._seed)
        key, init_key = jax.random.split(key)
        key, fit_key = jax.random.split(key)

        flow = self._make_flow(init_key)

        bs = N if batch_size is None else min(batch_size, N)

        flow, losses = fit_to_data(
            key=fit_key,
            dist=flow,
            data=x_norm,
            learning_rate=lr,
            max_epochs=n_epochs,
            max_patience=patience,
            batch_size=bs,
            show_progress=False,
        )

        if verbose:
            val_losses = losses.get("val", [])
            if val_losses:
                log.info(
                    f"Flow training complete. "
                    f"Final val NLL = {val_losses[-1]:.4f} "
                    f"(over {len(val_losses)} epochs)"
                )
            else:
                train_losses = losses.get("train", [])
                log.info(
                    f"Flow training complete. "
                    f"Final train NLL = {train_losses[-1]:.4f} "
                    f"(over {len(train_losses)} epochs)"
                )

        self._flow = flow
        self._trained = True

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
        x_norm = self._preprocess(x)
        z = np.array(
            jax.vmap(self._flow.bijection.inverse)(x_norm), dtype=np.float64
        )
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
        x_norm = np.array(
            jax.vmap(self._flow.bijection.transform)(jnp.array(z, dtype=jnp.float64)),
            dtype=np.float64,
        )
        x = self._unpreprocess(x_norm)
        return x[0] if single else x

    def sample(self, n_samples: int, key=None) -> np.ndarray:
        """Draw samples from the flow distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        key : JAX PRNGKey, optional
            JAX random key. If None, uses the instance seed.

        Returns
        -------
        samples : (n_samples, D) array
            Samples in physical / data space.
        """
        self._check_trained()
        if key is None:
            key = jax.random.key(self._seed)
            self._seed += 1  # Increment for next call
        
        # Sample from flowjax distribution (returns normalized samples)
        samples_norm = self._flow.sample(key, (n_samples,))
        
        # Un-preprocess to get back to physical space
        samples = self._unpreprocess(samples_norm)
        return samples

    def log_prob_single(self, x):
        """Log probability density log p(x) for a single point.

        JAX-compatible: can be called inside JIT-compiled functions.

        Parameters
        ----------
        x : (D,) array_like — single point

        Returns
        -------
        log_p : scalar
        """
        self._check_trained()
        x = jnp.atleast_1d(x)
        x_norm = (x - self._pre_mean) / self._pre_std
        return self._flow.log_prob(x_norm) - self._log_pre_std_sum

    def log_prob(self, x):
        """Log probability density log p(x) under the flow model.

        JAX-compatible: can be called inside JIT-compiled functions.

        Parameters
        ----------
        x : (D,) or (N, D) array_like

        Returns
        -------
        log_p : scalar (if single point) or (N,) array.
        """
        self._check_trained()
        x = jnp.asarray(x)
        if x.ndim == 1:
            return self.log_prob_single(x)
        return jax.vmap(self.log_prob_single)(x)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, best_pt: np.ndarray, best_logl: float):
        """Calibrate the flow so that calibrated_log_prob(best_pt) = best_logl.

        Parameters
        ----------
        best_pt : (D,) array_like
            The best-fit point in physical parameter space.
        best_logl : float
            The true log-likelihood value at the best-fit point.
        """
        self._check_trained()
        flow_logp = float(self.log_prob(best_pt))
        self._calibration_offset = best_logl - flow_logp
        self._calibrated = True
        log.info(
            f"Flow calibrated: offset = {self._calibration_offset:.4f} "
            f"(flow_logp={flow_logp:.4f}, target={best_logl:.4f})"
        )

    def calibrated_log_prob_single(self, x):
        """Return log_prob_single(x) + calibration offset for a single point.

        JAX-compatible: can be called inside JIT-compiled functions.

        Parameters
        ----------
        x : (D,) array_like — single point

        Returns
        -------
        log_p : scalar
        """
        if not self._calibrated:
            raise RuntimeError(
                "Flow has not been calibrated. Call .calibrate() first."
            )
        return self.log_prob_single(x) + self._calibration_offset

    def calibrated_log_prob(self, x):
        """Return log_prob(x) + calibration offset.

        JAX-compatible: can be called inside JIT-compiled functions.
        This aligns the flow's log probability density with the true
        log-likelihood scale, enabling its use as a GP mean function.

        Parameters
        ----------
        x : (D,) or (N, D) array_like

        Returns
        -------
        log_p : scalar (if single point) or (N,) array.
        """
        if not self._calibrated:
            raise RuntimeError(
                "Flow has not been calibrated. Call .calibrate() first."
            )
        x = jnp.asarray(x)
        if x.ndim == 1:
            return self.calibrated_log_prob_single(x)
        return jax.vmap(self.calibrated_log_prob_single)(x)

    @property
    def is_calibrated(self) -> bool:
        """Return True if the flow has been calibrated."""
        return self._calibrated

    # ------------------------------------------------------------------
    # Serialisation  (uses equinox for the flow pytree)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return a serialisable dict.

        The flowjax model cannot be pickled directly; the pytree leaves are
        saved separately via :func:`equinox.tree_serialise_leaves` to a
        temporary file and embedded as raw bytes.
        """
        import equinox as eqx

        state: dict = {
            "ndim": self.ndim,
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "nn_depth": self.nn_depth,
            "n_sigma": self.n_sigma,
            "seed": self._seed,
            "trained": self._trained,
            "pre_mean": np.asarray(self._pre_mean),
            "pre_std": np.asarray(self._pre_std),
            "log_pre_std_sum": self._log_pre_std_sum,
            "calibration_offset": self._calibration_offset,
            "calibrated": self._calibrated,
            "flow_bytes": None,
        }
        if self._trained and self._flow is not None:
            with tempfile.NamedTemporaryFile(suffix=".eqx", delete=False) as f:
                tmp = f.name
            try:
                eqx.tree_serialise_leaves(tmp, self._flow)
                with open(tmp, "rb") as f:
                    state["flow_bytes"] = f.read()
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        return state

    @classmethod
    def from_state_dict(cls, state: dict) -> "NormalisingFlow":
        import equinox as eqx

        obj = cls(
            ndim=int(state["ndim"]),
            n_layers=int(state["n_layers"]),
            hidden_dim=int(state["hidden_dim"]),
            nn_depth=int(state.get("nn_depth", 2)),
            n_sigma=float(state["n_sigma"]),
            seed=int(state["seed"]),
        )
        obj._trained = bool(state["trained"])
        obj._pre_mean = jnp.asarray(state["pre_mean"], dtype=jnp.float64)
        obj._pre_std = jnp.asarray(state["pre_std"], dtype=jnp.float64)
        obj._log_pre_std_sum = float(state.get("log_pre_std_sum", jnp.sum(jnp.log(obj._pre_std))))
        
        # Restore calibration state
        obj._calibration_offset = float(state.get("calibration_offset", 0.0))
        obj._calibrated = bool(state.get("calibrated", False))

        flow_bytes = state.get("flow_bytes")
        if obj._trained and flow_bytes is not None:
            # Reconstruct template and deserialise leaves
            template = obj._make_flow(jax.random.key(obj._seed))
            with tempfile.NamedTemporaryFile(suffix=".eqx", delete=False) as f:
                tmp = f.name
            try:
                with open(tmp, "wb") as f:
                    f.write(flow_bytes)
                obj._flow = eqx.tree_deserialise_leaves(tmp, template)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        return obj

    def save(self, path: str):
        """Save the full flow state to ``path`` via pickle of state_dict."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)
        log.info(f"NormalisingFlow saved to {path}")

    @classmethod
    def load(cls, path: str) -> "NormalisingFlow":
        """Load a saved NormalisingFlow from ``path``."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        log.info(f"NormalisingFlow loaded from {path}")
        return cls.from_state_dict(state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_trained(self):
        if not self._trained or self._flow is None:
            raise RuntimeError(
                "NormalisingFlow has not been trained. Call .fit() first."
            )

    def __repr__(self):
        status = "trained" if self._trained else "untrained"
        return (
            f"NormalisingFlow(ndim={self.ndim}, n_layers={self.n_layers}, "
            f"hidden_dim={self.hidden_dim}, nn_depth={self.nn_depth}, {status})"
        )
