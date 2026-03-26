"""
Parameter space transforms for BOBE.

Class hierarchy
---------------
BaseTransform       -- abstract interface shared by all transforms
IdentityTransform   -- simple linear scaling between physical bounds and [0,1]^D; no update
RotationTransform   -- eigenspace rotation with automatic update from MC samples

The ParameterTransform factory is kept for backward compatibility.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

from .utils.log import get_logger

log = get_logger("transforms")


class BaseTransform(ABC):
    """Abstract base class for all parameter-space transforms."""

    @abstractmethod
    def to_unit(self, theta, clip=False):
        """Map physical parameters theta -> unit cube u in [0,1]^r."""

    @abstractmethod
    def from_unit(self, u):
        """Map unit cube u in [0,1]^r -> physical parameters theta."""

    def update(self, gp, mc_samples, acq_val, acq_threshold,
               best_pt, all_x_phys, all_y_raw, step=0):
        """Attempt to learn and apply a new transform. Returns True if updated.
        After a successful update, self.updated_mc_samples holds the remapped dict."""
        return False

    def logz_correction(self, log_V_phys: float) -> float:
        """Additive correction to convert unit-cube log-evidence to physical-space log-evidence.

        logz_phys = logz_unit_cube + logz_correction(log_V_phys)

        where log_V_phys = Σᵢ log(θ_max_i − θ_min_i) is the log-volume of the
        physical prior box.

        The base implementation returns 0.0 (identity/linear mapping, where the
        unit-cube Jacobian exactly cancels the prior volume).  Subclasses with a
        constant non-trivial Jacobian (e.g. RotationTransform) override this.
        Transforms whose Jacobian varies per point (e.g. FlowTransform) leave the
        default, since the correction cannot be expressed as a single constant.
        """
        return 0.0

    @property
    @abstractmethod
    def ndim(self): ...

    @property
    @abstractmethod
    def rank(self): ...

    @property
    @abstractmethod
    def effective_bounds(self): ...

    @abstractmethod
    def state_dict(self): ...

    @classmethod
    @abstractmethod
    def from_state_dict(cls, state): ...

    def save(self, path):
        """Save transform state to {path}_transform.npz."""
        fname = path + "_transform.npz"
        state = self.state_dict()
        saveable = {}
        for k, v in state.items():
            if v is None:
                saveable[k] = np.array(False)
            elif isinstance(v, bool):
                saveable[k] = np.array(v)
            elif isinstance(v, (int, float)):
                saveable[k] = np.array(v)
            else:
                saveable[k] = np.asarray(v)
        np.savez(fname, **saveable)
        log.debug(f"Saved transform state to {fname}")

    @staticmethod
    def _prep_input(arr, expected_ndim):
        arr = np.asarray(arr, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr.reshape(1, -1)
        if arr.shape[-1] != expected_ndim:
            raise ValueError(
                f"Expected last dimension {expected_ndim}, got {arr.shape[-1]}")
        return arr, single

    def __repr__(self):
        return f"{self.__class__.__name__}(ndim={self.ndim})"


class IdentityTransform(BaseTransform):
    """
    Linear scaling between physical bounds and the unit cube.

    u = (theta - theta_min) / (theta_max - theta_min)
    theta = theta_min + u * (theta_max - theta_min)
    """

    _TYPE_KEY = "identity"

    def __init__(self, param_bounds):
        bounds = np.asarray(param_bounds, dtype=np.float64)
        if bounds.shape[0] != 2:
            raise ValueError("param_bounds must have shape (2, D)")
        self._original_bounds = bounds
        self._ndim = bounds.shape[1]
        self._theta_min = bounds[0].copy()
        self._theta_max = bounds[1].copy()
        self._theta_range = self._theta_max - self._theta_min
        self._effective_bounds = bounds.copy()
        log.info(f"IdentityTransform: ndim={self._ndim}")
        log.info(f"Physical bounds: {list(zip(self._theta_min, self._theta_max))}")

    def to_unit(self, theta, clip=False):
        theta = jnp.asarray(theta)
        single = theta.ndim == 1
        if single:
            theta = theta[jnp.newaxis]
        u = (theta - jnp.array(self._theta_min)) / jnp.array(self._theta_range)
        if clip:
            u = jnp.clip(u, 0.0, 1.0)
        return u[0] if single else u

    def from_unit(self, u):
        u, single = self._prep_input(u, self._ndim)
        if np.any(np.isnan(u)):
            log.warning("NaN detected in from_unit() input")
        theta = self._theta_min + u * self._theta_range
        return theta[0] if single else theta

    @property
    def ndim(self):
        return self._ndim

    @property
    def rank(self):
        return self._ndim

    @property
    def effective_bounds(self):
        return self._effective_bounds

    def state_dict(self):
        return {
            "type": self._TYPE_KEY,
            "original_bounds": self._original_bounds,
            "ndim": self._ndim,
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["original_bounds"])

    def __repr__(self):
        vol = float(np.prod(self._theta_range))
        return f"IdentityTransform(ndim={self._ndim}, phys_vol={vol:.2e})"


class RotationTransform(BaseTransform):
    """
    Eigenspace rotation with optional automatic update from MC samples.

    Before the first successful update() call (or if no initial covariance is
    provided), this transform falls back to simple linear scaling (identity mode).

    After each update():
    - A weighted sample covariance is estimated from current MC samples.
    - The new rotation (eigenvectors of covariance) is applied.
    - GP training points are remapped; out-of-bounds go to an internal dropped
      pool and are reconsidered at the next update.
    - MC samples are remapped and stored in self.updated_mc_samples.
    - The GP is warm-start refitted via pool.gp_fit().
    """

    _TYPE_KEY = "rotation"

    def __init__(
        self,
        param_bounds,
        covariance=None,
        center=None,
        is_fisher=False,
        n_sigma=5.0,
        regularize_eps=0.0,
        rank=None,
        kl_threshold=1.0,
        max_updates=10,
        update_step=None,
    ):
        """
        Parameters
        ----------
        param_bounds : array-like (2, D)
        covariance : array-like (D, D) or None
            Initial covariance (or Fisher if is_fisher=True). If None, starts
            in identity mode and learns rotation on first update() call.
        center : array-like (D,) or None
        is_fisher : bool
        n_sigma : float
            Bounds in rotated space are +/- n_sigma * sqrt(eigenvalue).
        regularize_eps : float
        rank : int or None
        kl_threshold : float
            Min symmetric KL between current and proposed Gaussian before an
            update fires. Updates are skipped when KL < kl_threshold.
        max_updates : int
        update_step : int or None
            Min BO iterations between consecutive updates (after the first).
            None disables further updates after the first.
        """
        bounds = np.asarray(param_bounds, dtype=np.float64)
        if bounds.shape[0] != 2:
            raise ValueError("param_bounds must have shape (2, D)")
        self._original_bounds = bounds
        self._ndim = bounds.shape[1]

        self.n_sigma = float(n_sigma)
        self.regularize_eps = float(regularize_eps)
        self._rank_cfg = rank
        self.kl_threshold = float(kl_threshold)
        self.max_updates = int(max_updates)
        self.update_step = update_step

        self.update_count = 0
        self.last_update_ii = None
        self.last_update_acq_val = None
        self.updated_mc_samples = None

        if covariance is not None:
            self._use_rotation = True
            self._setup_rotation(
                np.asarray(covariance, dtype=np.float64),
                center, is_fisher, n_sigma, regularize_eps, rank,
            )
        else:
            self._use_rotation = False
            self._setup_linear()

        log.info(repr(self))

    def _setup_linear(self):
        self._theta_min = self._original_bounds[0].copy()
        self._theta_max = self._original_bounds[1].copy()
        self._theta_range = self._theta_max - self._theta_min
        self._effective_bounds = self._original_bounds.copy()
        self._r = self._ndim
        log.info(f"RotationTransform (linear mode): ndim={self._ndim}")
        log.info(f"Physical bounds: {list(zip(self._theta_min, self._theta_max))}")

    def _setup_rotation(self, cov_input, center, is_fisher,
                        n_sigma, regularize_eps, rank):
        """(Re-)initialise the internal rotation state from a covariance matrix."""
        if cov_input.shape != (self._ndim, self._ndim):
            raise ValueError(
                f"covariance must be ({self._ndim},{self._ndim}), "
                f"got {cov_input.shape}"
            )
        if is_fisher:
            log.info("Inverting Fisher matrix to obtain covariance.")
            try:
                cov_phys = np.linalg.inv(cov_input)
            except np.linalg.LinAlgError:
                raise ValueError("Fisher matrix is singular.")
        else:
            cov_phys = cov_input.copy()

        if regularize_eps > 0.0:
            cov_phys = cov_phys + regularize_eps * np.eye(self._ndim)
        cov_phys = 0.5 * (cov_phys + cov_phys.T)
        min_eval = np.min(np.linalg.eigvalsh(cov_phys))
        if min_eval <= 0:
            raise ValueError(
                f"Covariance not positive definite (min eigenvalue {min_eval:.3e}). "
                "Increase regularize_eps."
            )
        self._covariance_phys = cov_phys

        eigvals, eigvecs = np.linalg.eigh(cov_phys)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        r = self._ndim if rank is None else int(rank)
        if not (1 <= r <= self._ndim):
            raise ValueError(f"rank must be in 1..{self._ndim}")
        self._r = r

        lambdas_r = np.maximum(eigvals[:r], 0.0)
        V_r = eigvecs[:, :r]
        sqrt_lambdas = np.sqrt(lambdas_r)
        self._z_min = -n_sigma * sqrt_lambdas
        self._z_max = +n_sigma * sqrt_lambdas
        self._z_range = self._z_max - self._z_min
        self._V_r = V_r

        if center is not None:
            center = np.asarray(center, dtype=np.float64).flatten()
            if center.shape != (self._ndim,):
                raise ValueError(f"center must have shape ({self._ndim},)")
            self._theta_center = center
        else:
            self._theta_center = 0.5 * (
                self._original_bounds[0] + self._original_bounds[1]
            )
            log.info("No center provided; using midpoint of param_bounds")

        theta_min_implied = self._theta_center + np.sum(
            np.minimum(V_r * self._z_min, V_r * self._z_max), axis=1
        )
        theta_max_implied = self._theta_center + np.sum(
            np.maximum(V_r * self._z_min, V_r * self._z_max), axis=1
        )
        self._effective_bounds = np.vstack([theta_min_implied, theta_max_implied])

        cond = (np.max(lambdas_r) / np.min(lambdas_r)
                if np.min(lambdas_r) > 0 else np.inf)
        log.info(f"RotationTransform enabled: rank={r}, n_sigma={n_sigma}")
        log.info(f"Eigenvalues (top {min(6, r)}): {lambdas_r[:min(6, r)]}")
        log.info(f"Condition number: {cond:.2e}")
        log.info(f"z bounds: {list(zip(self._z_min, self._z_max))}")
        log.info("Implied physical bounds (axis-aligned):")
        for i in range(self._ndim):
            log.info(
                f"  theta[{i}]: "
                f"[{theta_min_implied[i]:.4g}, {theta_max_implied[i]:.4g}]"
            )

    # -----------------------------------------------------------------
    # Core transforms
    # -----------------------------------------------------------------

    def to_unit(self, theta, clip=False):
        theta = jnp.asarray(theta)
        single = theta.ndim == 1
        if single:
            theta = theta[jnp.newaxis]
        if self._use_rotation:
            dtheta = theta - jnp.array(self._theta_center)
            z = dtheta @ jnp.array(self._V_r)
            u = (z - jnp.array(self._z_min)) / jnp.array(self._z_range)
        else:
            u = (theta - jnp.array(self._theta_min)) / jnp.array(self._theta_range)
        if clip:
            u = jnp.clip(u, 0.0, 1.0)
        return u[0] if single else u

    def from_unit(self, u):
        u, single = self._prep_input(u, self._r)
        if np.any(np.isnan(u)):
            log.warning("NaN detected in from_unit() input")
        if self._use_rotation:
            z = self._z_min + u * self._z_range
            theta = self._theta_center + z @ self._V_r.T
        else:
            theta = self._theta_min + u * self._theta_range
        return theta[0] if single else theta

    def logz_correction(self, log_V_phys: float) -> float:
        """Σᵢ log(z_range_i) − log_V_phys.

        The Jacobian ∂θ/∂u = V_r @ diag(z_range) has |det J| = ∏ᵢ z_range_i
        (V_r is orthogonal so |det V_r| = 1).  Adding this constant inside the
        nested-sampler loglike converts the unit-cube log-evidence into the
        physical-space log-evidence with a uniform prior over the original
        parameter bounds (volume exp(log_V_phys)).

        Returns 0.0 when the rotation is not yet active (linear fallback mode).
        """
        if not self._use_rotation:
            return 0.0
        return float(np.sum(np.log(self._z_range))) - log_V_phys

    # -----------------------------------------------------------------
    # Automatic update
    # -----------------------------------------------------------------

    def update(self, gp, mc_samples, acq_val, acq_threshold,
               best_pt, all_x_phys, all_y_raw, step=0):
        """
        Re-estimate covariance from MC samples and, if warranted, rebuild the
        rotation + GP training set in the new unit-cube space.

        Parameters
        ----------
        all_x_phys : ndarray (N, D)
            All physical training points accumulated by BOBE (canonical store).
        all_y_raw : ndarray (N,)
            Raw log-likelihood values corresponding to all_x_phys.

        The update fires when:
          - acq_val <= acq_threshold
          - update_count < max_updates
          - First update (no step-gap check), or subsequent update with
            step - last_update_ii >= update_step AND acq_val improved.

        After a successful update, self.updated_mc_samples holds the remapped
        MC sample dict for the caller to absorb.

        Returns True if the transform was updated.
        """
        self.updated_mc_samples = None

        # trigger guards
        if acq_val > acq_threshold:
            return False
        if self.update_count >= self.max_updates:
            log.info("[Rotation update] max_updates reached; skipping.")
            return False

        is_first = (self.update_count == 0)
        if not is_first:
            if self.update_step is None:
                return False
            if (step - self.last_update_ii) < self.update_step:
                return False
            if acq_val >= self.last_update_acq_val:
                return False

        # convert MC samples to physical space
        mc_x_unit = np.array(mc_samples["x"])
        mc_x_phys = self.from_unit(mc_x_unit)
        N = mc_x_phys.shape[0]

        if N < self._ndim + 2:
            log.warning(
                f"[Rotation update] Too few MC samples ({N}); skipping."
            )
            return False

        # weighted sample covariance
        weights = mc_samples.get("weights", None)
        if weights is not None:
            w = np.clip(np.array(weights, dtype=np.float64), 0.0, None)
            if w.sum() <= 0.0:
                w = np.ones(N)
            w /= w.sum()
            sample_mean = np.average(mc_x_phys, weights=w, axis=0)
            diff = mc_x_phys - sample_mean
            new_cov = (diff * w[:, None]).T @ diff
        else:
            new_cov = np.cov(mc_x_phys.T)
        new_cov = 0.5 * (new_cov + new_cov.T)
        new_cov += 1e-10 * np.eye(self._ndim)

        new_center = np.asarray(best_pt, dtype=np.float64).flatten()

        # KL divergence check
        if self._use_rotation:
            from .utils.core import kl_divergence_gaussian
            try:
                kl_dict = kl_divergence_gaussian(
                    self._theta_center, self._covariance_phys,
                    new_center, new_cov,
                )
                kl_sym = float(kl_dict["symmetric"])
            except Exception as e:
                log.warning(
                    f"[Rotation update] KL computation failed: {e}; skipping."
                )
                return False

            log.info(
                f"[Rotation update {self.update_count + 1}] "
                f"Symmetric KL = {kl_sym:.4f} "
                f"(threshold = {self.kl_threshold:.4f})"
            )
            if kl_sym < self.kl_threshold:
                log.info("[Rotation update] KL below threshold; skipping.")
                return False
        else:
            log.info(
                f"[Rotation update {self.update_count + 1}] "
                "No existing rotation -- establishing from sample covariance "
                "(KL check bypassed)."
            )

        # preview candidate rotation: compute in-bounds without committing
        try:
            _cov = new_cov.copy()
            if self.regularize_eps > 0:
                _cov += self.regularize_eps * np.eye(self._ndim)
            _cov = 0.5 * (_cov + _cov.T)
            _evals, _evecs = np.linalg.eigh(_cov)
            _idx = np.argsort(_evals)[::-1]
            _evals = _evals[_idx]
            _evecs = _evecs[:, _idx]
            _r = self._ndim if self._rank_cfg is None else int(self._rank_cfg)
            _lam_r = np.maximum(_evals[:_r], 0.0)
            _V_r_new = _evecs[:, :_r]
            _sqrt_lam = np.sqrt(_lam_r)
            _z_min_new = -self.n_sigma * _sqrt_lam
            _z_range_new = 2.0 * self.n_sigma * _sqrt_lam
        except np.linalg.LinAlgError as e:
            log.warning(
                f"[Rotation update] Eigendecomposition failed: {e}; skipping."
            )
            return False

        is_clf = hasattr(gp, "train_x_clf")

        dtheta = all_x_phys - new_center
        z_cand = dtheta @ _V_r_new
        u_cand = (z_cand - _z_min_new) / _z_range_new
        in_bounds = np.all((u_cand >= 0.0) & (u_cand <= 1.0), axis=1)

        n_dropped = int((~in_bounds).sum())
        if n_dropped > 0:
            log.info(
                f"[Rotation update] {n_dropped}/{len(all_x_phys)} points "
                "outside new unit cube (excluded from GP; retained in all_train_x_phys)."
            )

        x_ib = u_cand[in_bounds]
        y_ib = all_y_raw[in_bounds]

        if x_ib.shape[0] < self._ndim + 2:
            log.warning(
                f"[Rotation update] Only {x_ib.shape[0]} points remain "
                "after filtering -- too few; skipping."
            )
            return False

        # pre-check: skip if all MC samples would fall outside the new unit cube
        mc_dtheta = mc_x_phys - new_center
        mc_z_cand = mc_dtheta @ _V_r_new
        mc_u_cand_pre = (mc_z_cand - _z_min_new) / _z_range_new
        n_mc_in = int(np.all((mc_u_cand_pre >= 0.0) & (mc_u_cand_pre <= 1.0), axis=1).sum())
        if n_mc_in == 0:
            log.warning(
                f"[Rotation update] All {N} MC samples would fall outside new "
                "unit cube; skipping update."
            )
            return False

        # hand new data to GP (before committing transform)
        if is_clf:
            gp.remap_from_full_dataset(x_ib, y_ib)
        else:
            gp.remap_from_raw(x_ib, y_ib)

        # commit the new rotation
        self._setup_rotation(
            new_cov, new_center, False,
            self.n_sigma, self.regularize_eps, self._rank_cfg,
        )
        self._use_rotation = True

        # remap MC samples into new unit cube
        mc_x_new_unit = self.to_unit(mc_x_phys, clip=False)
        in_bounds_mc = np.all(
            (mc_x_new_unit >= 0.0) & (mc_x_new_unit <= 1.0), axis=1
        )
        n_dropped_mc = int((~in_bounds_mc).sum())
        if n_dropped_mc > 0:
            log.info(
                f"[Rotation update] Dropping {n_dropped_mc}/{N} MC samples "
                "outside new unit cube."
            )
        mc_x_new_unit = mc_x_new_unit[in_bounds_mc]
        new_mc = {
            "x": mc_x_new_unit,
            "method": mc_samples.get("method", "unknown"),
        }
        for key in ("weights", "logl", "logp"):
            if key in mc_samples:
                new_mc[key] = np.array(mc_samples[key])[in_bounds_mc]
        if "best" in mc_samples:
            new_mc["best"] = mc_samples["best"]
        self.updated_mc_samples = new_mc

        # update tracking
        self.update_count += 1
        self.last_update_ii = step
        self.last_update_acq_val = acq_val
        log.info(
            f"[Rotation update] Complete. "
            f"count={self.update_count}. Transform: {self!r}"
        )
        return True

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def ndim(self):
        return self._ndim

    @property
    def rank(self):
        return self._r

    @property
    def effective_bounds(self):
        return self._effective_bounds

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def state_dict(self):
        state = {
            "type": self._TYPE_KEY,
            "original_bounds": self._original_bounds,
            "ndim": self._ndim,
            "r": self._r,
            "use_rotation": self._use_rotation,
            "effective_bounds": self._effective_bounds,
            "n_sigma": self.n_sigma,
            "regularize_eps": self.regularize_eps,
            "rank_cfg": (self._rank_cfg if self._rank_cfg is not None else -1),
            "kl_threshold": self.kl_threshold,
            "max_updates": self.max_updates,
            "update_step": (self.update_step if self.update_step is not None else -1),
            "update_count": self.update_count,
            "last_update_ii": (self.last_update_ii
                               if self.last_update_ii is not None else -1),
            "last_update_acq_val": (self.last_update_acq_val
                                    if self.last_update_acq_val is not None
                                    else np.nan),
        }
        if self._use_rotation:
            state.update({
                "covariance_phys": self._covariance_phys,
                "theta_center": self._theta_center,
                "V_r": self._V_r,
                "z_min": self._z_min,
                "z_max": self._z_max,
                "z_range": self._z_range,
            })
        else:
            state.update({
                "theta_min": self._theta_min,
                "theta_max": self._theta_max,
                "theta_range": self._theta_range,
            })
        return state

    @classmethod
    def from_state_dict(cls, state):
        obj = cls.__new__(cls)
        obj._original_bounds = np.array(state["original_bounds"])
        obj._ndim = int(state["ndim"])
        obj._r = int(state["r"])
        obj._use_rotation = bool(state["use_rotation"])
        obj._effective_bounds = np.array(state["effective_bounds"])

        obj.n_sigma = float(state.get("n_sigma", 5.0))
        obj.regularize_eps = float(state.get("regularize_eps", 0.0))
        _rc = state.get("rank_cfg", -1)
        obj._rank_cfg = None if int(_rc) == -1 else int(_rc)
        obj.kl_threshold = float(state.get("kl_threshold", 1.0))
        obj.max_updates = int(state.get("max_updates", 10))
        _us = state.get("update_step", -1)
        obj.update_step = None if int(_us) == -1 else int(_us)

        obj.update_count = int(state.get("update_count", 0))
        _lii = state.get("last_update_ii", -1)
        obj.last_update_ii = None if int(_lii) == -1 else int(_lii)
        _lav = state.get("last_update_acq_val", np.nan)
        obj.last_update_acq_val = (
            None if np.isnan(float(_lav)) else float(_lav)
        )
        obj.updated_mc_samples = None

        if obj._use_rotation:
            obj._covariance_phys = np.array(state["covariance_phys"])
            obj._theta_center = np.array(state["theta_center"])
            obj._V_r = np.array(state["V_r"])
            obj._z_min = np.array(state["z_min"])
            obj._z_max = np.array(state["z_max"])
            obj._z_range = np.array(state["z_range"])
        else:
            obj._theta_min = np.array(state["theta_min"])
            obj._theta_max = np.array(state["theta_max"])
            obj._theta_range = np.array(state["theta_range"])

        return obj

    def __repr__(self):
        if self._use_rotation:
            vol = float(np.prod(self._z_range))
            return (
                f"RotationTransform(ndim={self._ndim}, rank={self._r}, "
                f"updates={self.update_count}/{self.max_updates}, "
                f"z_vol={vol:.2e})"
            )
        vol = float(np.prod(self._theta_max - self._theta_min))
        return (
            f"RotationTransform(ndim={self._ndim}, "
            f"mode=linear->rotation, phys_vol={vol:.2e})"
        )


class NormalisingFlowTransform(BaseTransform):
    """
    Normalising Flow transform with automatic update from MC samples.

    Maps physical space → standard Normal latent z → unit cube [0,1]^D:
        to_unit(theta):  z = flow(theta),  u = (z + n_sigma) / (2 * n_sigma)
        from_unit(u):    z = u * 2*n_sigma - n_sigma,  theta = flow^{-1}(z)

    Before the first successful ``update()`` call, falls back to linear
    (identity) scaling like ``IdentityTransform``.

    Requires flax and optax (``pip install BOBE[nn]``).
    """

    _TYPE_KEY = "normalising_flow"

    def __init__(
        self,
        param_bounds,
        n_sigma: float = 5.0,
        kl_threshold: float = 1.0,
        max_updates: int = 10,
        update_step=None,
        # flow architecture
        n_layers: int = 8,
        hidden_dim: int = 64,
        nn_depth: int = 2,
        flow_n_epochs: int = 2000,
        flow_lr: float = 3e-4,
        flow_batch_size: int = 512,
        seed: int = 42,
        use_rotation_precon: bool = True,
    ):
        """
        Parameters
        ----------
        param_bounds : array-like (2, D)
            Physical parameter bounds.
        n_sigma : float
            Latent-space bound: unit cube [0,1]^D maps to z in [-n_sigma, n_sigma]^D.
        kl_threshold : float
            Minimum symmetric KL (Gaussian approximation) between old and new
            posteriors before an update fires.
        max_updates : int
        update_step : int or None
            Minimum BO iterations between consecutive updates. None disables
            secondary updates.
        n_layers : int
            Number of MAF flow layers.
        hidden_dim : int
            Hidden layer width in each autoregressive network.
        nn_depth : int
            Depth of each autoregressive network. Default 2.
        flow_n_epochs : int
            Maximum training epochs for each flow fit.
        flow_lr : float
            Peak Adam learning rate for flow training.
        flow_batch_size : int
            Mini-batch size for flow training.
        seed : int
            Base random seed (incremented by update_count at each update).
        """
        try:
            from .flow import NormalisingFlow as _NF  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "NormalisingFlowTransform requires flowjax and equinox. "
                "Install with:  pip install BOBE[nn]"
            ) from exc

        bounds = np.asarray(param_bounds, dtype=np.float64)
        if bounds.shape[0] != 2:
            raise ValueError("param_bounds must have shape (2, D)")
        self._original_bounds = bounds
        self._ndim = bounds.shape[1]

        self.n_sigma = float(n_sigma)
        self.kl_threshold = float(kl_threshold)
        self.max_updates = int(max_updates)
        self.update_step = update_step
        self.n_layers = int(n_layers)
        self.hidden_dim = int(hidden_dim)
        self.nn_depth = int(nn_depth)
        self.flow_n_epochs = int(flow_n_epochs)
        self.flow_lr = float(flow_lr)
        self.flow_batch_size = int(flow_batch_size)
        self.seed = int(seed)
        self.use_rotation_precon = bool(use_rotation_precon)

        # runtime state
        self._use_flow = False
        self._flow = None          # NormalisingFlow instance
        self._cached_to_u_fn = None  # stable JAX closure for to_unit (rebuilt after each flow fit)
        self._prev_mc_x_phys = None  # physical MC samples from last flow training (for KL check)
        self._prev_mc_logl = None    # GP logL at those samples at training time (for KL check)
        self.update_count = 0
        self.last_update_ii = None
        self.last_update_acq_val = None
        self.updated_mc_samples = None

        self._setup_linear()
        log.info(repr(self))

    # -----------------------------------------------------------------
    # Linear fallback setup
    # -----------------------------------------------------------------

    def _setup_linear(self):
        self._theta_min = self._original_bounds[0].copy()
        self._theta_max = self._original_bounds[1].copy()
        self._theta_range = self._theta_max - self._theta_min
        self._effective_bounds = self._original_bounds.copy()

    def _build_to_u_fn(self):
        """Build and cache the JAX-traceable per-point unit transform function.

        Called once whenever ``self._flow`` is replaced (pretrain, update, resume)
        so that ``to_unit()`` reuses a single stable closure instead of rebuilding
        a new one—with fresh ``jnp.array`` allocations and a new equinox method
        reference—on every call.  A stable closure lets JAX cache its trace across
        calls and avoids spurious recompilation inside JIT-compiled samplers.
        """
        assert self._flow is not None, "_build_to_u_fn() requires a trained flow"
        _mean = jnp.array(self._flow._pre_mean)
        _std = jnp.array(self._flow._pre_std)
        _n_sigma = self.n_sigma
        _bij_inv = self._flow._flow.bijection.inverse
        if self._flow._use_rotation_precon:
            _eigvecs = jnp.array(self._flow._pre_eigvecs)
            def _fn(t):
                z = _bij_inv((t - _mean) @ _eigvecs / _std)
                return (z + _n_sigma) / (2.0 * _n_sigma)
        else:
            def _fn(t):
                z = _bij_inv((t - _mean) / _std)
                return (z + _n_sigma) / (2.0 * _n_sigma)
        self._cached_to_u_fn = _fn

    # -----------------------------------------------------------------
    # Core transforms
    # -----------------------------------------------------------------

    def to_unit(self, theta, clip=False):
        theta = jnp.asarray(theta)
        single = theta.ndim == 1
        if self._use_flow:
            # Use the pre-built cached closure (see _build_to_u_fn).
            # Avoids rebuilding a new function object with jnp.array allocations
            # on every call, which would force JAX to retrace within JIT contexts.
            u = self._cached_to_u_fn(theta) if single else jax.vmap(self._cached_to_u_fn)(theta)
        else:
            if single:
                theta = theta[jnp.newaxis]
            u = (theta - jnp.array(self._theta_min)) / jnp.array(self._theta_range)
            if clip:
                u = jnp.clip(u, 0.0, 1.0)
            return u[0] if single else u
        if clip:
            u = jnp.clip(u, 0.0, 1.0)
        return u

    def from_unit(self, u):
        u, single = self._prep_input(u, self._ndim)
        if self._use_flow:
            z = u * 2.0 * self.n_sigma - self.n_sigma
            theta = self._flow.to_data(z)
        else:
            theta = self._theta_min + u * self._theta_range
        return theta[0] if single else theta

    # -----------------------------------------------------------------
    # Pre-training on external samples
    # -----------------------------------------------------------------

    def pretrain(self, x_phys, weights=None):
        """Train the normalising flow on external samples and activate the transform.

        After calling this the flow is live (``_use_flow = True``) and
        ``update_count`` is set to ``max_updates`` so that no further
        automatic updates are triggered during the BOBE run.

        Parameters
        ----------
        x_phys : array-like (N, D)
            Physical-space training samples (e.g. from reference MCMC chains).
        weights : array-like (N,) or None
            Optional importance weights for the samples.
        """
        from .flow import NormalisingFlow

        x_phys = np.asarray(x_phys, dtype=np.float64)
        N = x_phys.shape[0]
        log.info(
            f"[Flow pretrain] Training flow on {N} external samples "
            f"({'rotation precon + ' if self.use_rotation_precon else ''}MAF) ..."
        )

        flow = NormalisingFlow(
            ndim=self._ndim,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
            nn_depth=self.nn_depth,
            n_sigma=self.n_sigma,
            seed=self.seed,
            use_rotation_precon=self.use_rotation_precon,
        )
        flow.fit(
            x_phys,
            weights=weights,
            n_epochs=self.flow_n_epochs,
            lr=self.flow_lr,
            batch_size=self.flow_batch_size,
            verbose=True,
        )
        self._flow = flow
        self._use_flow = True
        self._build_to_u_fn()
        # Lock out automatic updates for the rest of the BOBE run.
        self.update_count = self.max_updates

        # Diagnostics
        z_diag = self._flow.to_latent(x_phys[:min(N, 1000)])
        u_diag = (z_diag + self.n_sigma) / (2.0 * self.n_sigma)
        frac_in = float(np.mean(np.all((u_diag >= 0.0) & (u_diag <= 1.0), axis=1)))
        log.info(
            f"[Flow pretrain] External samples in unit cube: {frac_in * 100:.1f}% "
            f"(n_sigma={self.n_sigma})"
        )
        rt_err = np.max(np.abs(flow.to_data(flow.to_latent(x_phys[:5])) - x_phys[:5]))
        log.info(f"[Flow pretrain] Roundtrip max error: {rt_err:.2e}")
        log.info(f"[Flow pretrain] Complete. Transform: {self!r}")

    # -----------------------------------------------------------------
    # Automatic update
    # -----------------------------------------------------------------

    def update(self, gp, mc_samples, acq_val, acq_threshold,
               best_pt, all_x_phys, all_y_raw, step=0):
        """Train a new flow from MC samples and rebuild the unit-cube mapping.

        Parameters
        ----------
        all_x_phys : ndarray (N, D)  – all physical training points.
        all_y_raw  : ndarray (N,)   – corresponding raw log-likelihood values.

        Returns True if the transform was updated.
        """
        self.updated_mc_samples = None

        # When the flow is already fitted, evaluate sample-based KL BEFORE the
        # acquisition-value gate.  A large KL means the GP posterior has shifted
        # enough that the flow geometry needs updating regardless of how good the
        # current acquisition value looks.
        _force_kl_update = False
        if self._use_flow and self._prev_mc_x_phys is not None and self._prev_mc_logl is not None:
            try:
                from .utils.core import kl_divergence_samples as _kl_samples
                prev_u = np.asarray(self.to_unit(self._prev_mc_x_phys))
                in_bounds = np.all((prev_u >= 0.0) & (prev_u <= 1.0), axis=1)
                prev_u_valid = jnp.array(prev_u[in_bounds], dtype=jnp.float64)
                prev_logl_valid = self._prev_mc_logl[in_bounds]
                if len(prev_u_valid) >= 4:
                    current_gp_logl = np.array(
                        jax.lax.map(
                            gp.predict_mean_single, prev_u_valid, batch_size=100
                        )
                    )
                    kl_dict = _kl_samples(prev_logl_valid, current_gp_logl)
                    kl_sym = float(kl_dict["symmetric"])
                    log.info(
                        f"[Flow update {self.update_count + 1}] "
                        f"KL(GP samples) = {kl_sym:.4f} "
                        f"(threshold = {self.kl_threshold:.4f})"
                    )
                    if kl_sym >= self.kl_threshold:
                        _force_kl_update = True
                        log.info(
                            "[Flow update] KL above threshold — "
                            "forcing retraining regardless of acquisition value."
                        )
                    else:
                        log.info("[Flow update] KL below threshold; skipping.")
                        return False
                else:
                    log.info(
                        "[Flow update] Too few prev samples in current bounds; "
                        "skipping KL check."
                    )
            except Exception as exc:
                log.warning(f"[Flow update] KL check failed: {exc}; proceeding.")
        elif self._use_flow:
            log.info(
                f"[Flow update {self.update_count + 1}] "
                "No previous flow samples stored — KL check bypassed."
            )

        if self.update_count >= self.max_updates:
            log.info("[Flow update] max_updates reached; skipping.")
            return False
        # Acquisition-value gate: skip only when not forced by a large KL.
        if not _force_kl_update and acq_val > acq_threshold:
            return False

        is_first = (self.update_count == 0)
        if not is_first:
            if self.update_step is None:
                return False
            if (step - self.last_update_ii) < self.update_step:
                return False
            # Require acq improvement only when not forced by KL.
            if not _force_kl_update and acq_val >= self.last_update_acq_val:
                return False

        # MC samples in physical space.
        # mc_samples["x"] is always in the current unit cube; from_unit() maps it
        # back to physical coordinates using whichever transform is currently active
        # (flow or linear fallback).  The new flow is then trained on mc_x_phys.
        mc_x_unit = np.array(mc_samples["x"])
        mc_x_phys = self.from_unit(mc_x_unit)
        N = mc_x_phys.shape[0]
        log.debug(
            f"[Flow update] mc_x_phys range: "
            f"min={np.min(mc_x_phys, axis=0).round(4)}, "
            f"max={np.max(mc_x_phys, axis=0).round(4)}"
        )

        if N < self._ndim + 2:
            log.warning(f"[Flow update] Too few MC samples ({N}); skipping.")
            return False

        weights = mc_samples.get("weights", None)

        # Compute new posterior Gaussian approximation (also reused for KL check)
        if weights is not None:
            w = np.clip(np.array(weights, dtype=np.float64), 0.0, None)
            w /= w.sum()
            new_mean = np.average(mc_x_phys, weights=w, axis=0)
            diff = mc_x_phys - new_mean
            new_cov = (diff * w[:, None]).T @ diff
        else:
            new_mean = mc_x_phys.mean(axis=0)
            new_cov = np.cov(mc_x_phys.T) if N > 1 else np.eye(self._ndim)
        new_cov = 0.5 * (new_cov + new_cov.T) + 1e-10 * np.eye(self._ndim)
        # (KL check was already performed above the trigger guards.)

        # Pre-viability check using Gaussian approximation: verify enough training
        # points fall within n_sigma of the new posterior centre before spending
        # time on the full flow training.  Mirrors RotationTransform's cheap preview.
        try:
            _evals, _evecs = np.linalg.eigh(new_cov)
            _idx = np.argsort(_evals)[::-1]
            _sqrt_lam = np.sqrt(np.maximum(_evals[_idx], 0.0))
            _V = _evecs[:, _idx]
            _z_min_pre = -self.n_sigma * _sqrt_lam
            _z_rng_pre = 2.0 * self.n_sigma * _sqrt_lam
            _u_pre = (((all_x_phys - new_mean) @ _V) - _z_min_pre) / _z_rng_pre
            n_pre_covered = int(np.all((_u_pre >= 0.0) & (_u_pre <= 1.0), axis=1).sum())
            if n_pre_covered < self._ndim + 2:
                log.warning(
                    f"[Flow update] Pre-check: only {n_pre_covered}/{len(all_x_phys)} "
                    "training points within Gaussian approx. bounds — "
                    "skipping flow training."
                )
                return False
            log.info(
                f"[Flow update] Pre-check: {n_pre_covered}/{len(all_x_phys)} "
                "training points covered; proceeding to flow training."
            )
        except np.linalg.LinAlgError:
            pass  # eigendecomposition failed; proceed to flow training anyway

        # Train new flow
        from .flow import NormalisingFlow
        log.info(
            f"[Flow update {self.update_count + 1}] "
            f"Training flow on {N} samples "
            f"({'rotation precon + ' if self.use_rotation_precon else ''}MAF) …"
        )
        flow = NormalisingFlow(
            ndim=self._ndim,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
            nn_depth=self.nn_depth,
            n_sigma=self.n_sigma,
            seed=self.seed + self.update_count,
            use_rotation_precon=self.use_rotation_precon,
        )
        flow.fit(
            mc_x_phys,
            weights=weights,
            n_epochs=self.flow_n_epochs,
            lr=self.flow_lr,
            batch_size=self.flow_batch_size,
            verbose=True,
        )
        old_flow = self._flow   # save for rollback before committing candidate
        self._flow = flow

        # --- Diagnostic: how well does the new transform cover MC samples? ---
        # Compute directly via to_latent() to avoid any dependency on _use_flow.
        _diag_x = mc_x_phys[:min(N, 1000)]
        _z_diag = self._flow.to_latent(_diag_x)
        u_diag = (_z_diag + self.n_sigma) / (2.0 * self.n_sigma)
        frac_in = float(np.mean(np.all((u_diag >= 0.0) & (u_diag <= 1.0), axis=1)))
        log.info(
            f"[Flow update] MC samples in unit cube: {frac_in*100:.1f}% "
            f"(n_sigma={self.n_sigma})"
        )
        log.info(
            f"[Flow update] u mean: {np.round(u_diag.mean(0), 3)}, "
            f"u std: {np.round(u_diag.std(0), 3)}"
        )
        # Roundtrip check on a small batch
        _x5 = mc_x_phys[:5]
        _rt_err = np.max(np.abs(flow.to_data(flow.to_latent(_x5)) - _x5))
        log.info(f"[Flow update] Roundtrip max error: {_rt_err:.2e}")

        # Compute new unit-cube coordinates for all GP training points directly
        # using the new flow (mirrors RotationTransform: compute with new params
        # before committing, so the GP always receives correct coordinates).
        z_cand = self._flow.to_latent(all_x_phys)
        u_cand = (z_cand + self.n_sigma) / (2.0 * self.n_sigma)
        in_bounds = np.all((u_cand >= 0.0) & (u_cand <= 1.0), axis=1)

        n_dropped = int((~in_bounds).sum())
        if n_dropped > 0:
            log.info(
                f"[Flow update] {n_dropped}/{len(all_x_phys)} training points "
                "outside new unit cube (excluded from GP)."
            )

        x_ib = u_cand[in_bounds]
        y_ib = all_y_raw[in_bounds]

        if x_ib.shape[0] < self._ndim + 2:
            log.warning(
                f"[Flow update] Only {x_ib.shape[0]} points remain after "
                "filtering — too few; rolling back."
            )
            self._flow = old_flow   # restore previous flow (not None) to avoid crash
            return False

        # pre-check: skip if all MC samples would fall outside the new unit cube
        mc_z_cand_pre = self._flow.to_latent(mc_x_phys)
        mc_u_cand_pre = (mc_z_cand_pre + self.n_sigma) / (2.0 * self.n_sigma)
        n_mc_in = int(np.all((mc_u_cand_pre >= 0.0) & (mc_u_cand_pre <= 1.0), axis=1).sum())
        if n_mc_in == 0:
            log.warning(
                f"[Flow update] All {N} MC samples would fall outside new "
                "unit cube; rolling back."
            )
            self._flow = old_flow
            return False

        # Update GP training data BEFORE committing the new transform, mirroring
        # RotationTransform ordering: GP receives correctly-transformed coordinates
        # first, then the transform is activated so subsequent to_unit() calls use
        # the new flow.  This closes the window where classifier retraining inside
        # remap_from_full_dataset could call to_unit() with an inconsistent state.
        is_clf = hasattr(gp, "train_x_clf")
        if is_clf:
            gp.remap_from_full_dataset(x_ib, y_ib)
        else:
            gp.remap_from_raw(x_ib, y_ib)

        # Commit the new transform and rebuild the cached to_unit closure.
        self._use_flow = True
        self._build_to_u_fn()

        # Remap MC samples into the new unit cube
        mc_x_new_unit = self.to_unit(mc_x_phys, clip=False)
        in_bounds_mc = np.all(
            (mc_x_new_unit >= 0.0) & (mc_x_new_unit <= 1.0), axis=1
        )
        n_dropped_mc = int((~in_bounds_mc).sum())
        if n_dropped_mc > 0:
            log.info(
                f"[Flow update] Dropping {n_dropped_mc}/{N} MC samples "
                "outside new unit cube."
            )
        mc_x_new_unit = mc_x_new_unit[in_bounds_mc]
        new_mc = {
            "x": mc_x_new_unit,
            "method": mc_samples.get("method", "unknown"),
        }
        for key in ("weights", "logl", "logp"):
            if key in mc_samples:
                new_mc[key] = np.array(mc_samples[key])[in_bounds_mc]
        if "best" in mc_samples:
            new_mc["best"] = mc_samples["best"]
        self.updated_mc_samples = new_mc

        # Store the MC samples (physical coords) and their GP logL values used
        # for this flow training.  The next update() call uses these to compute
        # a sample-based KL divergence against the then-current GP predictions,
        # giving a direct measure of posterior shift since the last flow fit.
        logl_key = 'logp' if 'logp' in mc_samples else ('logl' if 'logl' in mc_samples else None)
        n_store = min(N, 2000)
        self._prev_mc_x_phys = mc_x_phys[:n_store]
        self._prev_mc_logl = np.array(mc_samples[logl_key])[:n_store] if logl_key else None

        self.update_count += 1
        self.last_update_ii = step
        self.last_update_acq_val = acq_val
        log.info(
            f"[Flow update] Complete. "
            f"count={self.update_count}. Transform: {self!r}"
        )
        return True

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def ndim(self):
        return self._ndim

    @property
    def rank(self):
        return self._ndim

    @property
    def effective_bounds(self):
        return self._original_bounds

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def state_dict(self):
        state = {
            "type": self._TYPE_KEY,
            "original_bounds": self._original_bounds,
            "ndim": self._ndim,
            "n_sigma": self.n_sigma,
            "kl_threshold": self.kl_threshold,
            "max_updates": self.max_updates,
            "update_step": (self.update_step if self.update_step is not None else -1),
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "nn_depth": self.nn_depth,
            "flow_n_epochs": self.flow_n_epochs,
            "flow_lr": self.flow_lr,
            "flow_batch_size": self.flow_batch_size,
            "seed": self.seed,
            "use_flow": self._use_flow,
            "use_rotation_precon": self.use_rotation_precon,
            "update_count": self.update_count,
            "last_update_ii": (
                self.last_update_ii if self.last_update_ii is not None else -1
            ),
            "last_update_acq_val": (
                self.last_update_acq_val
                if self.last_update_acq_val is not None
                else np.nan
            ),
        }
        if not self._use_flow:
            state.update({
                "theta_min": self._theta_min,
                "theta_max": self._theta_max,
                "theta_range": self._theta_range,
            })
        return state

    def save(self, path):
        """Save state to ``{path}_transform.npz`` and flow to ``{path}_flow.pkl``."""
        import pickle as _pickle

        fname = path + "_transform.npz"
        state = self.state_dict()
        saveable = {}
        for k, v in state.items():
            if v is None:
                saveable[k] = np.array(False)
            elif isinstance(v, bool):
                saveable[k] = np.array(v)
            elif isinstance(v, (int, float)):
                saveable[k] = np.array(v)
            else:
                try:
                    saveable[k] = np.asarray(v)
                except Exception:
                    pass   # skip non-serialisable entries (pytrees, etc.)
        np.savez(fname, **saveable)

        if self._use_flow and self._flow is not None:
            flow_path = path + "_flow.pkl"
            with open(flow_path, "wb") as f:
                _pickle.dump(self._flow.state_dict(), f)

        log.debug(f"Saved NormalisingFlowTransform to {fname}")

    @classmethod
    def from_state_dict(cls, state):
        obj = cls.__new__(cls)
        obj._original_bounds = np.array(state["original_bounds"])
        obj._ndim = int(state["ndim"])
        obj.n_sigma = float(state.get("n_sigma", 5.0))
        obj.kl_threshold = float(state.get("kl_threshold", 1.0))
        obj.max_updates = int(state.get("max_updates", 10))
        _us = state.get("update_step", -1)
        obj.update_step = None if int(_us) == -1 else int(_us)
        obj.n_layers = int(state.get("n_layers", 8))
        obj.hidden_dim = int(state.get("hidden_dim", 64))
        obj.nn_depth = int(state.get("nn_depth", 2))
        obj.flow_n_epochs = int(state.get("flow_n_epochs", 2000))
        obj.flow_lr = float(state.get("flow_lr", 3e-4))
        obj.flow_batch_size = int(state.get("flow_batch_size", 512))
        obj.seed = int(state.get("seed", 42))
        obj.use_rotation_precon = bool(state.get("use_rotation_precon", True))
        obj._use_flow = bool(state.get("use_flow", False))
        obj.update_count = int(state.get("update_count", 0))
        _lii = state.get("last_update_ii", -1)
        obj.last_update_ii = None if int(_lii) == -1 else int(_lii)
        _lav = state.get("last_update_acq_val", np.nan)
        obj.last_update_acq_val = (
            None if np.isnan(float(_lav)) else float(_lav)
        )
        obj.updated_mc_samples = None
        obj._flow = None
        obj._cached_to_u_fn = None
        obj._prev_mc_x_phys = None
        obj._prev_mc_logl = None

        if not obj._use_flow:
            if "theta_min" in state:
                obj._theta_min = np.array(state["theta_min"])
                obj._theta_max = np.array(state["theta_max"])
                obj._theta_range = np.array(state["theta_range"])
            else:
                obj._setup_linear()
        else:
            obj._setup_linear()   # fallback linear attrs (unused in flow mode)

        return obj

    def __repr__(self):
        if self._use_flow:
            return (
                f"NormalisingFlowTransform(ndim={self._ndim}, "
                f"updates={self.update_count}/{self.max_updates}, flow_trained)"
            )
        return (
            f"NormalisingFlowTransform(ndim={self._ndim}, linear_mode)"
        )


def ParameterTransform(
    param_bounds,
    rotation_matrix=None,
    rotation_center=None,
    rotation_is_fisher=False,
    n_sigma=5.0,
    regularize_eps=0.0,
    rank=None,
):
    """
    Factory function retained for backward compatibility.
    Returns IdentityTransform when rotation_matrix is None, RotationTransform otherwise.
    """
    if rotation_matrix is None:
        return IdentityTransform(param_bounds)
    return RotationTransform(
        param_bounds,
        covariance=rotation_matrix,
        center=rotation_center,
        is_fisher=rotation_is_fisher,
        n_sigma=n_sigma,
        regularize_eps=regularize_eps,
        rank=rank,
    )


def load_transform(path):
    """
    Load a BaseTransform from {path}_transform.npz.
    Correct subclass is detected from the 'type' key.
    """
    fname = path + "_transform.npz"
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Transform file not found: {fname}")

    data = np.load(fname, allow_pickle=True)
    state = {}
    for key in data.files:
        val = data[key]
        if isinstance(val, np.ndarray) and val.shape == ():
            state[key] = val.item()
        else:
            state[key] = val

    type_key = state.get("type", "rotation")
    if type_key == IdentityTransform._TYPE_KEY:
        return IdentityTransform.from_state_dict(state)
    elif type_key == RotationTransform._TYPE_KEY:
        return RotationTransform.from_state_dict(state)
    elif type_key == NormalisingFlowTransform._TYPE_KEY:
        obj = NormalisingFlowTransform.from_state_dict(state)
        # Attempt to restore flow params from companion pickle
        flow_pkl = path + "_flow.pkl"
        if obj._use_flow and os.path.exists(flow_pkl):
            import pickle as _pickle
            from .flow import NormalisingFlow
            with open(flow_pkl, "rb") as f:
                flow_state = _pickle.load(f)
            obj._flow = NormalisingFlow.from_state_dict(flow_state)
        return obj
    else:
        raise ValueError(f"Unknown transform type key: {type_key!r}")
