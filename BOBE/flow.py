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
        nn_depth: int = 2,
        n_sigma: float = 5.0,
        seed: int = 42,
        use_rotation_precon: bool = False,
    ):
        self.ndim = ndim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.nn_depth = nn_depth
        self.n_sigma = n_sigma
        self._seed = seed
        self.use_rotation_precon = use_rotation_precon

        self._flow = None          # flowjax Transformed distribution
        self._pre_mean = np.zeros(ndim, dtype=np.float64)
        self._pre_std = np.ones(ndim, dtype=np.float64)
        # Rotation preconditioning (PCA whitening before the MAF)
        self._use_rotation_precon = False
        self._pre_eigvecs = None   # (D, D) columns = eigenvectors, descending λ
        self._pre_eigvals = None   # (D,) eigenvalues, descending
        self._trained = False

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

    def _preprocess(self, x) -> jnp.ndarray:
        """Preprocess x → normalised space fed to the MAF.

        When ``_use_rotation_precon`` is True this performs PCA whitening:
            x_white = (x - mean) @ V / sqrt(λ)
        so the MAF sees data with Cov ≈ I and only needs to learn non-Gaussian
        residuals.  Otherwise falls back to axis-wise standardisation.
        """
        if self._use_rotation_precon:
            centered = jnp.array(x, dtype=jnp.float64) - jnp.array(self._pre_mean)
            return (centered @ jnp.array(self._pre_eigvecs)) / jnp.array(self._pre_std)
        return jnp.array((x - self._pre_mean) / self._pre_std, dtype=jnp.float64)

    def _unpreprocess(self, x_norm) -> np.ndarray:
        if self._use_rotation_precon:
            # Undo scaling then un-rotate: x = x_white * sqrt(λ) @ V^T + mean
            return (
                np.array(x_norm, dtype=np.float64)
                * self._pre_std
            ) @ self._pre_eigvecs.T + self._pre_mean
        return np.array(x_norm, dtype=np.float64) * self._pre_std + self._pre_mean

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        weights: Optional[np.ndarray] = None,
        pre_covariance: Optional[np.ndarray] = None,
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

        # --- Preprocessing statistics ---
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            w = np.clip(w, 0.0, None)
            w /= w.sum()
            self._pre_mean = np.average(x, weights=w, axis=0)
            diff = x - self._pre_mean
            sample_cov = (diff * w[:, None]).T @ diff
            axis_std = np.sqrt(np.diag(sample_cov))
        else:
            self._pre_mean = x.mean(axis=0)
            diff = x - self._pre_mean
            sample_cov = np.cov(x.T) if N > 1 else np.eye(D)
            axis_std = x.std(axis=0)

        # Decide whether to use rotation preconditioning
        _cov_for_rot = None
        if pre_covariance is not None:
            _cov_for_rot = np.asarray(pre_covariance, dtype=np.float64)
        elif self.use_rotation_precon:
            _cov_for_rot = sample_cov

        if _cov_for_rot is not None:
            # PCA whitening: eigenvectors in descending eigenvalue order
            eigvals, eigvecs = np.linalg.eigh(_cov_for_rot)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            self._pre_eigvals = np.maximum(eigvals, 1e-12)
            self._pre_eigvecs = eigvecs
            # _pre_std = sqrt(λ) so log_prob Jacobian formula is unchanged
            self._pre_std = np.sqrt(self._pre_eigvals)
            self._use_rotation_precon = True
            log.info(
                f"Rotation preconditioning enabled. "
                f"Condition number: {self._pre_eigvals[0]/self._pre_eigvals[-1]:.2e}"
            )
        else:
            self._pre_std = np.maximum(axis_std, 1e-6)
            self._use_rotation_precon = False
            self._pre_eigvecs = None
            self._pre_eigvals = None

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
        log_p = np.array(
            jax.vmap(self._flow.log_prob)(x_norm)
            - float(np.sum(np.log(self._pre_std))),
            dtype=np.float64,
        )
        return float(log_p[0]) if single else log_p

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
            "use_rotation_precon": self.use_rotation_precon,
            "active_rotation_precon": self._use_rotation_precon,
            "pre_mean": self._pre_mean.copy(),
            "pre_std": self._pre_std.copy(),
            "pre_eigvecs": self._pre_eigvecs.copy() if self._pre_eigvecs is not None else None,
            "pre_eigvals": self._pre_eigvals.copy() if self._pre_eigvals is not None else None,
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
        obj.use_rotation_precon = bool(state.get("use_rotation_precon", False))
        obj._use_rotation_precon = bool(state.get("active_rotation_precon", False))
        obj._pre_mean = np.array(state["pre_mean"], dtype=np.float64)
        obj._pre_std = np.array(state["pre_std"], dtype=np.float64)
        _ev = state.get("pre_eigvecs")
        obj._pre_eigvecs = np.array(_ev, dtype=np.float64) if _ev is not None else None
        _el = state.get("pre_eigvals")
        obj._pre_eigvals = np.array(_el, dtype=np.float64) if _el is not None else None

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
