"""
Normalising-flow parameter space transform for BOBE.

FlowTransform uses a flowjax coupling flow to map from physical parameter
space to the unit cube [0, 1]^D via:

  θ (physical)  →  θ_std = (θ - μ) / σ              (standardise with train mean/std)
                →  z = flow.bijection.inverse(θ_std)  ∈ ℝ^D  (≈ N(0, I) after training)
                →  u = (z + n_sigma) / (2 * n_sigma)  ∈ [0, 1]^D  (linear scaling)

Inverse:
  u  →  z = u * (2 * n_sigma) - n_sigma       (invert linear scaling)
     →  θ_std = flow.bijection.transform(z)    (flow decode)
     →  θ = θ_std * σ + μ                      (unstandardise)

This matches exactly how the covariance-rotation transform works: bounds in
latent (z) space are set to ±n_sigma (default 5), and a simple affine map
translates z ∈ [-n_sigma, n_sigma]^D  ↔  u ∈ [0, 1]^D.  Points outside
±n_sigma are clipped, so there are no singularities at the cube walls.

Until ``train_flow()`` has been called the class behaves as a plain linear
transform (identical to ``ParameterTransform`` with no rotation).

Serialisation note
------------------
The flowjax model cannot be embedded in a plain numpy ``.npz`` file.
``state_dict()`` marks the transform type but does **not** embed the model
weights.  A separate pickle file is written alongside the ``.npz`` when
``save_flow()`` is called.  ``bo.py._save_transform()`` orchestrates this;
``_load_transform()`` calls ``load_flow()`` to restore the model.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp

from .transforms import ParameterTransform
from .log import get_logger

log = get_logger("flow_transform")


class FlowTransform(ParameterTransform):
    """
    Normalising-flow parameter space transform.

    Until ``train_flow()`` is called this behaves as a simple linear transform
    (inheriting the linear-scaling logic from ``ParameterTransform``).  After
    training, ``to_unit`` and ``from_unit`` use the learned flow.

    Parameters
    ----------
    param_bounds : array-like, shape (2, D)
        Physical parameter bounds ``[[min_1, …], [max_1, …]]``.
        Used for the linear fallback before the flow is trained.

    Attributes
    ----------
    is_flow : bool
        Always ``True`` for ``FlowTransform``.
    is_flow_trained : bool
        ``True`` after a successful call to ``train_flow()``.
    """

    def __init__(self, param_bounds, n_sigma: float = 5.0):
        # Initialise via the parent's linear-scaling path (no rotation).
        super().__init__(
            param_bounds=param_bounds,
            rotation_matrix=None,
        )
        self._n_sigma = float(n_sigma)
        # Flow-specific state
        self._flow = None
        self._flow_trained = False
        self._encode = None   # jit-vmapped  θ_std → z
        self._decode = None   # jit-vmapped  z → θ_std
        # Architecture hyper-parameters (stored so we can rebuild for deserialisation)
        self._flow_arch = {}
        # z-space bounds (set after training)
        self._z_min = None
        self._z_max = None
        self._z_range = None
        # Standardisation statistics (set after training)
        self._train_mean = None
        self._train_std = None

    # ------------------------------------------------------------------
    # Property overrides
    # ------------------------------------------------------------------

    @property
    def is_flow(self) -> bool:
        return True

    @property
    def is_flow_trained(self) -> bool:
        return self._flow_trained

    # ------------------------------------------------------------------
    # Flow training
    # ------------------------------------------------------------------

    def train_flow(
        self,
        samples_phys: np.ndarray,
        flow_layers: int = 8,
        nn_width: int = 64,
        nn_depth: int = 2,
        learning_rate: float = 5e-4,
        max_epochs: int = 600,
        batch_size: int = 256,
        max_patience: int = 40,
        seed: int = 42,
    ) -> None:
        """
        Train a flowjax coupling flow on physical-space samples.

        After training, ``to_unit`` / ``from_unit`` use the flow.

        Parameters
        ----------
        samples_phys : ndarray, shape (N, D)
            Physical-space parameter samples to train on
            (e.g. MCMC or nested-sampling chain in physical coordinates).
        flow_layers : int
            Number of coupling layers. Default 8.
        nn_width : int
            Width of each coupling network. Default 64.
        nn_depth : int
            Depth of each coupling network. Default 2.
        learning_rate : float
            Adam learning rate. Default 5e-4.
        max_epochs : int
            Maximum training epochs. Default 600.
        batch_size : int
            Mini-batch size. Default 256.
        max_patience : int
            Early-stopping patience (epochs without val improvement). Default 40.
        seed : int
            JAX PRNG seed. Default 42.
        """
        try:
            import flowjax.flows as fl
            import flowjax.distributions as fdist
            import flowjax.train as tr
        except ImportError as exc:
            raise ImportError(
                "flowjax is required for FlowTransform. "
                "Install it with:  pip install flowjax"
            ) from exc

        samples = np.asarray(samples_phys, dtype=np.float64)
        N, D = samples.shape
        if D != self.ndim:
            raise ValueError(
                f"samples_phys has {D} columns but transform has ndim={self.ndim}"
            )
        if N < 2 * D:
            log.warning(
                f"Only {N} training samples for a {D}-D flow — results may be poor."
            )

        # Standardise samples: z_std = (θ - μ) / σ
        train_mean = np.mean(samples, axis=0)
        train_std  = np.std(samples, axis=0)
        train_std  = np.where(train_std < 1e-10, 1.0, train_std)  # avoid divide-by-zero
        samples_std = (samples - train_mean) / train_std
        log.info(
            f"Standardising training data: mean={train_mean}, std={train_std}"
        )

        jkey = jax.random.PRNGKey(seed)
        jkey, subkey = jax.random.split(jkey)

        log.info(
            f"Building coupling flow: D={D}, layers={flow_layers}, "
            f"nn_width={nn_width}, nn_depth={nn_depth}"
        )
        flow = fl.coupling_flow(
            subkey,
            base_dist=fdist.StandardNormal((D,)),
            flow_layers=flow_layers,
            nn_width=nn_width,
            nn_depth=nn_depth,
            invert=True,  # flow.bijection.inverse: data → latent
        )

        log.info(f"Training flow on {N} standardised samples …")
        jkey, subkey = jax.random.split(jkey)
        flow, losses = tr.fit_to_data(
            subkey,
            flow,
            jnp.array(samples_std, dtype=jnp.float64),
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=min(batch_size, N),
            max_patience=max_patience,
            show_progress=False,
        )

        train_losses = losses.get("train", [])
        val_losses = losses.get("val", [])
        log.info(
            f"Flow trained for {len(train_losses)} epoch(s); "
            f"final NLL train={train_losses[-1]:.4f}, val={val_losses[-1]:.4f}"
            if train_losses and val_losses
            else f"Flow trained for {len(train_losses)} epoch(s)."
        )

        # Round-trip sanity check on a small subsample (in standardised space)
        try:
            idx = np.random.choice(N, size=min(64, N), replace=False)
            z_check = np.asarray(jax.jit(jax.vmap(flow.bijection.inverse))(
                jnp.array(samples_std[idx])
            ))
            theta_std_check = np.asarray(jax.jit(jax.vmap(flow.bijection.transform))(
                jnp.array(z_check)
            ))
            rt_err = np.abs(theta_std_check - samples_std[idx]).max()
            log.info(f"Round-trip max |err| on {len(idx)} samples: {rt_err:.3e}")
            if rt_err > 1.0:
                log.warning("Large round-trip error — flow may not have converged.")
        except Exception as e:
            log.warning(f"Round-trip check failed: {e}")

        # Store model and compiled functions
        self._flow = flow
        self._encode = jax.jit(jax.vmap(flow.bijection.inverse))   # θ_std → z
        self._decode = jax.jit(jax.vmap(flow.bijection.transform))  # z → θ_std
        self._flow_trained = True
        self._flow_arch = dict(
            flow_layers=flow_layers,
            nn_width=nn_width,
            nn_depth=nn_depth,
        )
        # Standardisation statistics
        self._train_mean = train_mean
        self._train_std  = train_std
        # Linear z-space bounds: flow base is N(0,I), so ±n_sigma covers all directions
        self._z_min   = -self._n_sigma * np.ones(D)
        self._z_max   = +self._n_sigma * np.ones(D)
        self._z_range =  self._z_max - self._z_min   # = 2 * n_sigma * ones
        log.info(
            f"FlowTransform is now active (standardised input, linear z-scaling, n_sigma={self._n_sigma})."
        )

    # ------------------------------------------------------------------
    # Core transforms
    # ------------------------------------------------------------------

    def to_unit(self, theta, clip=True):
        """
        Map physical parameters θ → unit cube u ∈ [0, 1]^D.

        Pipeline: θ → θ_std = (θ - μ) / σ          (standardise)
                    → z = flow⁻¹(θ_std) ≈ N(0,I)
                    → u = (z - z_min) / z_range      (linear, identical to rotation code)

        z-space bounds are ±n_sigma (default 5), so u is in [0,1] for all
        points within n_sigma standard deviations of the posterior centre.
        Points beyond ±n_sigma are clipped to [0,1].

        If the flow is not yet trained, falls back to linear scaling.

        Parameters
        ----------
        theta : array-like, shape (D,) or (N, D)
        clip : bool
            If True (default), clip output to [0, 1].

        Returns
        -------
        u : ndarray, shape (D,) or (N, D)
        """
        if not self._flow_trained:
            return super().to_unit(theta, clip=clip)

        theta = np.asarray(theta, dtype=np.float64)
        single = theta.ndim == 1
        if single:
            theta = theta.reshape(1, -1)

        if np.any(np.isnan(theta)):
            log.warning("NaN in input to FlowTransform.to_unit()")

        # Standardise before encoding
        theta_std = (theta - self._train_mean) / self._train_std
        z = np.asarray(self._encode(jnp.array(theta_std)))   # (N, D)
        u = (z - self._z_min) / self._z_range                 # linear scaling

        if clip:
            u = np.clip(u, 0.0, 1.0)

        return u[0] if single else u

    def from_unit(self, u):
        """
        Map unit cube u ∈ [0, 1]^D → physical parameters θ.

        Pipeline: u → z = z_min + u * z_range   (invert linear scaling)
                    → θ_std = flow(z)            (flow decode)
                    → θ = θ_std * σ + μ          (unstandardise)

        If the flow is not yet trained, falls back to linear scaling.

        Parameters
        ----------
        u : array-like, shape (D,) or (N, D)

        Returns
        -------
        theta : ndarray, shape (D,) or (N, D)
        """
        if not self._flow_trained:
            return super().from_unit(u)

        u = np.asarray(u, dtype=np.float64)
        single = u.ndim == 1
        if single:
            u = u.reshape(1, -1)

        if np.any(np.isnan(u)):
            log.warning("NaN in input to FlowTransform.from_unit()")

        z = self._z_min + np.clip(u, 0.0, 1.0) * self._z_range   # (N, D)
        theta_std = np.asarray(self._decode(jnp.array(z)))         # (N, D)

        # Unstandardise: θ = θ_std * σ + μ
        theta = theta_std * self._train_std + self._train_mean

        # Soft clip to physical bounds to prevent wild extrapolations
        theta = np.clip(theta, self.original_bounds[0], self.original_bounds[1])

        return theta[0] if single else theta

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def state_dict(self):
        """Return a numpy-serialisable state dict (flow weights stored separately)."""
        state = super().state_dict()
        state["transform_type"] = "flow"
        state["flow_trained"] = self._flow_trained
        state["n_sigma"] = self._n_sigma
        if self._train_mean is not None:
            state["train_mean"] = self._train_mean
        if self._train_std is not None:
            state["train_std"] = self._train_std
        if self._flow_arch:
            for k, v in self._flow_arch.items():
                state[f"flow_arch_{k}"] = v
        return state

    @classmethod
    def from_state_dict(cls, state):
        """
        Restore a FlowTransform from a serialised state dict.

        The flow model weights are **not** embedded in the state dict.
        Call ``load_flow(path)`` after this to restore the model; until
        then the instance operates in linear-fallback mode.
        """
        bounds = np.array(state["original_bounds"])
        n_sigma = float(state.get("n_sigma", 5.0))
        obj = cls(param_bounds=bounds, n_sigma=n_sigma)
        obj._flow_trained = False   # model must be reloaded separately
        # Restore standardisation statistics if present
        if "train_mean" in state:
            obj._train_mean = np.array(state["train_mean"])
        if "train_std" in state:
            obj._train_std = np.array(state["train_std"])
        log.info(
            "FlowTransform restored from state dict without flow model "
            "(linear fallback until load_flow() is called)."
        )
        return obj

    def save_flow(self, base_path: str) -> None:
        """
        Serialise the flowjax model to disk using equinox.

        Two files are written:
          - ``{base_path}.eqx``       — model leaf arrays (via equinox)
          - ``{base_path}_arch.json`` — architecture hyper-parameters

        Parameters
        ----------
        base_path : str
            Base file path (without extension).  The suffix ``.pkl`` is stripped
            automatically if present, so legacy callers that pass ``foo.pkl``
            will still get ``foo.eqx`` + ``foo_arch.json``.
        """
        if not self._flow_trained:
            log.warning("save_flow() called but flow is not trained — nothing saved.")
            return

        import equinox as eqx
        import json

        # Strip legacy .pkl suffix so paths stay clean
        base = base_path.rstrip('.pkl').rstrip('_pkl').replace('.pkl', '')

        eqx_path  = base + '.eqx'
        arch_path = base + '_arch.json'

        eqx.tree_serialise_leaves(eqx_path, self._flow)
        arch_data = dict(self._flow_arch)
        if self._train_mean is not None:
            arch_data["train_mean"] = self._train_mean.tolist()
        if self._train_std is not None:
            arch_data["train_std"] = self._train_std.tolist()
        with open(arch_path, 'w') as fh:
            json.dump(arch_data, fh)
        log.debug(f"Saved flow model to {eqx_path} + {arch_path}")

    def load_flow(self, base_path: str) -> bool:
        """
        Restore the flowjax model from files written by ``save_flow()``.

        Parameters
        ----------
        base_path : str
            Base file path (without extension) — same value that was passed to
            ``save_flow()``.  A ``.pkl`` suffix is stripped automatically.

        Returns
        -------
        bool
            ``True`` if loading succeeded, ``False`` otherwise.
        """
        try:
            import equinox as eqx
            import json
            import flowjax.flows as fl
            import flowjax.distributions as fdist

            base = base_path.rstrip('.pkl').rstrip('_pkl').replace('.pkl', '')
            eqx_path  = base + '.eqx'
            arch_path = base + '_arch.json'

            if not (os.path.exists(eqx_path) and os.path.exists(arch_path)):
                log.debug(f"Flow model files not found at {eqx_path} / {arch_path}")
                return False

            with open(arch_path) as fh:
                arch = json.load(fh)

            # Rebuild a template flow with the same architecture to act as the
            # equinox "like" structure for deserialisation.
            jkey = jax.random.PRNGKey(0)
            template = fl.coupling_flow(
                jkey,
                base_dist=fdist.StandardNormal((self.ndim,)),
                flow_layers=arch.get('flow_layers', 8),
                nn_width=arch.get('nn_width', 64),
                nn_depth=arch.get('nn_depth', 2),
                invert=True,
            )
            flow = eqx.tree_deserialise_leaves(eqx_path, like=template)

            self._flow      = flow
            # Separate flow architecture from standardisation stats
            self._flow_arch = {
                k: v for k, v in arch.items()
                if k not in ('train_mean', 'train_std')
            }
            self._encode    = jax.jit(jax.vmap(flow.bijection.inverse))
            self._decode    = jax.jit(jax.vmap(flow.bijection.transform))
            self._flow_trained = True
            # Restore standardisation statistics
            if "train_mean" in arch:
                self._train_mean = np.array(arch["train_mean"])
            if "train_std" in arch:
                self._train_std  = np.array(arch["train_std"])
            # Restore z-space bounds
            D = self.ndim
            self._z_min   = -self._n_sigma * np.ones(D)
            self._z_max   = +self._n_sigma * np.ones(D)
            self._z_range =  self._z_max - self._z_min
            log.info(f"Flow model restored from {eqx_path} (arch={self._flow_arch}, n_sigma={self._n_sigma}).")
            return True
        except Exception as e:
            log.warning(f"Failed to load flow model from {base_path!r}: {e}")
            return False

    def __repr__(self) -> str:
        status = "trained" if self._flow_trained else "untrained (linear fallback)"
        return (
            f"FlowTransform(ndim={self.ndim}, status={status}, "
            f"n_sigma={self._n_sigma}, arch={self._flow_arch})"
        )
