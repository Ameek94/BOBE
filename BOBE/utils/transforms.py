"""
Parameter space transforms for BOBE.

This module provides the ParameterTransform class which manages all coordinate
transformations between physical parameter space and the unit cube [0,1]^D
used by the GP, acquisition functions, and nested sampler.

Two modes of operation:
1. **No rotation** (rotation_matrix=None): Simple linear scaling between physical
   bounds and unit cube. u = (θ - θ_min) / (θ_max - θ_min).

2. **With rotation** (rotation_matrix provided): The fundamental space is the 
   rotated eigenspace z defined by covariance eigenvectors.
   - z bounds are set to ± n_sigma * sqrt(eigenvalues) (default n_sigma=5).
   - Unit cube u ∈ [0,1]^r maps to z via simple affine scaling.
   - Physical parameters θ ∈ R^D are obtained via rotation: θ = θ_* + V_r @ z.

Transforms:
  - to_unit(θ) → u: physical → unit cube
  - from_unit(u) → θ: unit cube → physical
"""

import numpy as np
from abc import ABC, abstractmethod
from .log import get_logger

log = get_logger("transforms")


class BaseParameterTransform(ABC):
    """
    Abstract base class for all parameter space transforms in BOBE.

    Defines the interface that all transforms must implement so that the
    Gaussian Process, acquisition functions, and nested sampler can use
    any transform interchangeably.

    Subclasses
    ----------
    ParameterTransform : linear scaling or covariance-rotation transform.
    FlowTransform      : normalising-flow-based transform (BOBE/utils/flow.py).
    """

    # Subclasses must set these as instance attributes in __init__:
    #   self.ndim           -- physical parameter dimensionality (int)
    #   self._r             -- GP / unit-cube dimensionality (int)
    #   self.original_bounds -- (2, D) array
    #   self.effective_bounds -- (2, D) array

    @abstractmethod
    def to_unit(self, theta, clip=True):
        """Map physical parameters θ → unit cube u ∈ [0,1]^r."""

    @abstractmethod
    def from_unit(self, u):
        """Map unit cube u ∈ [0,1]^r → physical parameters θ."""

    @abstractmethod
    def state_dict(self):
        """Serialise transform state to a dict of numpy-serialisable values."""

    @classmethod
    @abstractmethod
    def from_state_dict(cls, state):
        """Restore a transform from a serialised state dict."""

    @property
    def rank(self):
        """Dimensionality of unit cube space."""
        return self._r

    @property
    def uses_rotation(self):
        """Whether a covariance rotation is active (False for FlowTransform)."""
        return getattr(self, '_use_rotation', False)

    @property
    def is_flow(self):
        """Whether this transform is a normalising flow (False for ParameterTransform)."""
        return False


class ParameterTransform(BaseParameterTransform):
    """
    Manage physical <-> unit-cube transformations.

    Supports two modes:
    1. Simple linear scaling (no rotation): u = (θ - θ_min) / range
    2. Rotated eigenspace (with rotation_matrix): via eigenvector projection

    Public methods:
      - to_unit(theta) -> u: physical → unit cube
      - from_unit(u) -> theta: unit cube → physical
      - unit_to_rotated(u) -> z: unit cube → rotated space (rotation mode only)
      - rotated_to_unit(z) -> u: rotated space → unit cube (rotation mode only)
    """

    def __init__(self, param_bounds, rotation_matrix=None, rotation_center=None,
                 rotation_is_fisher=False, n_sigma=5.0, regularize_eps=0.0,
                 rank=None):
        """
        Parameters
        ----------
        param_bounds : array-like (2, D)
            Physical parameter bounds [[min1, min2, ...], [max1, max2, ...]].
            Used directly when rotation_matrix is None.
        rotation_matrix : array-like (D, D) or None
            Covariance matrix in physical space, or Fisher (if rotation_is_fisher=True).
            If None, uses simple linear scaling with param_bounds.
        rotation_center : array-like (D,) or None
            Physical-space center where covariance was computed (e.g. MAP/MLE point).
            If None, defaults to center of param_bounds.
        rotation_is_fisher : bool
            If True, invert `rotation_matrix` to obtain covariance (cov = inv(F)).
        n_sigma : float
            Number of standard deviations for bounds in rotated space (default 5.0).
        regularize_eps : float
            Small diagonal regularizer added to covariance for numerical stability.
        rank : int or None
            Truncate rotation to `rank` leading eigenvectors; if None, use full rank D.
        """
        self.original_bounds = np.asarray(param_bounds, dtype=np.float64)
        if self.original_bounds.shape[0] != 2:
            raise ValueError("param_bounds must have shape (2, D)")
        self.ndim = self.original_bounds.shape[1]

        if rotation_matrix is None:
            # Simple linear scaling mode
            self._use_rotation = False
            self._setup_linear()
        else:
            # Rotated eigenspace mode
            self._use_rotation = True
            self._setup_rotation(
                rotation_matrix, rotation_center, rotation_is_fisher, 
                n_sigma, regularize_eps, rank
            )

    # -----------------------------------------------------------------
    # Linear scaling setup (no rotation)
    # -----------------------------------------------------------------
    def _setup_linear(self):
        """
        Setup simple linear scaling between physical bounds and unit cube.
        
        u = (θ - θ_min) / (θ_max - θ_min)
        θ = θ_min + u * (θ_max - θ_min)
        """
        self._theta_min = self.original_bounds[0]
        self._theta_max = self.original_bounds[1]
        self._theta_range = self._theta_max - self._theta_min
        
        # For linear mode, effective bounds equal original bounds
        self.effective_bounds = self.original_bounds.copy()
        self._r = self.ndim  # Full dimensionality
        
        log.info(f"Linear transform: ndim={self.ndim}")
        log.info(f"Physical bounds: {list(zip(self._theta_min, self._theta_max))}")

    # -----------------------------------------------------------------
    # Rotation setup
    # -----------------------------------------------------------------
    def _setup_rotation(self, rotation_matrix, rotation_center,
                        rotation_is_fisher, n_sigma, regularize_eps, rank):
        """
        Setup coordinate system based on eigendecomposition of covariance.

        Steps:
         1. Get covariance in physical space (invert Fisher if needed)
         2. Eigendecompose: cov_phys = V Λ V^T
         3. Truncate to rank r (keep top r eigenvectors/eigenvalues)
         4. Set z bounds: [-n_sigma*√λ_i, +n_sigma*√λ_i] for i=1..r
         5. Store V_r (D×r eigenvectors) and center θ_*

        The transforms are then:
         - θ → z: z = V_r^T @ (θ - θ_*)
         - z → θ: θ = θ_* + V_r @ z
         - u → z: z = z_min + u * z_range  (simple scaling)
         - z → u: u = (z - z_min) / z_range
        """
        rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
        if rotation_matrix.shape != (self.ndim, self.ndim):
            raise ValueError(f"rotation_matrix must be ({self.ndim},{self.ndim})")

        # 1) Get covariance in physical space
        if rotation_is_fisher:
            log.info("Inverting Fisher matrix to obtain covariance.")
            try:
                cov_phys = np.linalg.inv(rotation_matrix)
            except np.linalg.LinAlgError:
                raise ValueError("Fisher matrix is singular and cannot be inverted.")
        else:
            cov_phys = rotation_matrix.copy()

        # Regularize if requested
        if regularize_eps > 0.0:
            cov_phys = cov_phys + regularize_eps * np.eye(self.ndim)
            log.info(f"Applied regularization: eps={regularize_eps}")

        # Ensure symmetry
        cov_phys = 0.5 * (cov_phys + cov_phys.T)

        # Check positive definiteness
        eigvals_check = np.linalg.eigvalsh(cov_phys)
        min_eval = np.min(eigvals_check)
        if min_eval <= 0:
            raise ValueError(
                f"Covariance not positive definite (min eigenvalue {min_eval:.3e}). "
                "Increase regularize_eps."
            )

        self._covariance_phys = cov_phys

        # 2) Eigendecompose covariance
        eigvals, eigvecs = np.linalg.eigh(cov_phys)
        # Sort descending by eigenvalue
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # 3) Determine rank
        if rank is None:
            r = self.ndim
        else:
            r = int(rank)
            if r <= 0 or r > self.ndim:
                raise ValueError("rank must be in 1..D")
        self._r = r

        # Truncate to rank r
        lambdas_r = eigvals[:r].copy()
        lambdas_r[lambdas_r < 0] = 0.0  # numerical guard
        V_r = eigvecs[:, :r]  # D × r

        # 4) Set z bounds as ± n_sigma * sqrt(eigenvalues)
        sqrt_lambdas = np.sqrt(lambdas_r)
        self._z_min = -n_sigma * sqrt_lambdas
        self._z_max = +n_sigma * sqrt_lambdas
        self._z_range = self._z_max - self._z_min

        # 5) Store eigenvectors and center
        self._V_r = V_r  # D × r
        if rotation_center is not None:
            rotation_center = np.asarray(rotation_center, dtype=np.float64).flatten()
            if rotation_center.shape != (self.ndim,):
                raise ValueError(f"rotation_center must have shape ({self.ndim},)")
            self._theta_center = rotation_center
        else:
            # Default to center of original bounds
            self._theta_center = 0.5 * (self.original_bounds[0] + self.original_bounds[1])
            log.info("No rotation_center provided; using center of param_bounds")

        # Compute implied physical bounds (for diagnostics)
        theta_min_implied = self._theta_center + np.sum(
            np.minimum(V_r * self._z_min, V_r * self._z_max), axis=1
        )
        theta_max_implied = self._theta_center + np.sum(
            np.maximum(V_r * self._z_min, V_r * self._z_max), axis=1
        )
        self.effective_bounds = np.vstack([theta_min_implied, theta_max_implied])

        # Remember n_sigma for later rotation updates
        self._n_sigma = n_sigma

        # Log diagnostics
        cond = np.max(lambdas_r) / np.min(lambdas_r) if np.min(lambdas_r) > 0 else np.inf
        log.info(f"Rotation enabled: rank={r}, n_sigma={n_sigma}")
        log.info(f"Eigenvalues (top {min(6, r)}): {lambdas_r[:min(6, r)]}")
        log.info(f"Std devs (top {min(6, r)}): {sqrt_lambdas[:min(6, r)]}")
        log.info(f"Condition number: {cond:.2e}")
        log.info(f"z bounds: {list(zip(self._z_min, self._z_max))}")
        log.info(f"Implied physical bounds (axis-aligned):")
        for i in range(self.ndim):
            log.info(f"  θ[{i}]: [{theta_min_implied[i]:.4g}, {theta_max_implied[i]:.4g}]")

    # -----------------------------------------------------------------
    # Core forward transform: physical -> unit cube
    # -----------------------------------------------------------------
    def to_unit(self, theta, clip=True):
        """
        Map physical parameters θ → unit cube u ∈ [0,1]^r.

        Parameters
        ----------
        theta : array-like, shape (D,) or (N, D)
            Physical parameters
        clip : bool
            If True (default), clip output to [0, 1].

        Returns
        -------
        u : ndarray, shape (r,) or (N, r)
            Unit cube coordinates
        """
        theta = np.asarray(theta, dtype=np.float64)
        single = False
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
            single = True
        if theta.shape[-1] != self.ndim:
            raise ValueError(f"Expected last dimension {self.ndim}, got {theta.shape[-1]}")

        # Check for NaN in input
        if np.any(np.isnan(theta)):
            log.warning(f"NaN detected in input to to_unit(). Affected rows: {np.sum(np.any(np.isnan(theta), axis=1))}")

        if self._use_rotation:
            # z = V_r^T @ (θ - θ_*) = (θ - θ_*) @ V_r
            dtheta = theta - self._theta_center
            z = dtheta @ self._V_r  # (N, D) @ (D, r) = (N, r)
            # u = (z - z_min) / z_range
            u = (z - self._z_min) / self._z_range
        else:
            # Simple linear: u = (θ - θ_min) / θ_range
            u = (theta - self._theta_min) / self._theta_range

        if clip:
            u = np.clip(u, 0.0, 1.0)

        if single:
            return u[0]
        return u

    # -----------------------------------------------------------------
    # Map unit cube -> rotated z and inverse (rotation mode only)
    # -----------------------------------------------------------------
    def unit_to_rotated(self, u):
        """
        Map unit cube u ∈ [0,1]^r → rotated coordinates z.

        Only available in rotation mode.

        Parameters
        ----------
        u : array-like, shape (r,) or (N, r)
            Unit cube coordinates

        Returns
        -------
        z : ndarray, shape (r,) or (N, r)
            Rotated eigenspace coordinates
        """
        if not self._use_rotation:
            raise RuntimeError("unit_to_rotated() only available in rotation mode")
            
        u = np.asarray(u, dtype=np.float64)
        single = False
        if u.ndim == 1:
            u = u.reshape(1, -1)
            single = True
        if u.shape[-1] != self._r:
            raise ValueError(f"Expected last dimension {self._r}, got {u.shape[-1]}")

        z = self._z_min + u * self._z_range

        if single:
            return z[0]
        return z

    def rotated_to_unit(self, z, clip=True):
        """
        Map rotated coordinates z → unit cube u.

        Only available in rotation mode.

        Parameters
        ----------
        z : array-like, shape (r,) or (N, r)
            Rotated eigenspace coordinates
        clip : bool
            If True, clip result to [0, 1]

        Returns
        -------
        u : ndarray, shape (r,) or (N, r)
            Unit cube coordinates
        """
        if not self._use_rotation:
            raise RuntimeError("rotated_to_unit() only available in rotation mode")
            
        z = np.asarray(z, dtype=np.float64)
        single = False
        if z.ndim == 1:
            z = z.reshape(1, -1)
            single = True
        if z.shape[-1] != self._r:
            raise ValueError(f"Expected last dimension {self._r}, got {z.shape[-1]}")

        u = (z - self._z_min) / self._z_range

        if clip:
            u = np.clip(u, 0.0, 1.0)

        if single:
            return u[0]
        return u

    # -----------------------------------------------------------------
    # Canonical inverse: unit cube -> physical
    # -----------------------------------------------------------------
    def from_unit(self, u):
        """
        Map unit cube u ∈ [0,1]^r → physical parameters θ ∈ R^D.

        Parameters
        ----------
        u : array-like, shape (r,) or (N, r)
            Unit cube coordinates

        Returns
        -------
        theta : ndarray, shape (D,) or (N, D)
            Physical parameters
        """
        u = np.asarray(u, dtype=np.float64)
        single = False
        if u.ndim == 1:
            u = u.reshape(1, -1)
            single = True
        if u.shape[-1] != self._r:
            raise ValueError(f"Expected last dimension {self._r}, got {u.shape[-1]}")

        # Check for NaN in input
        if np.any(np.isnan(u)):
            log.warning(f"NaN detected in input to from_unit(). Affected rows: {np.sum(np.any(np.isnan(u), axis=1))}")

        if self._use_rotation:
            # z = z_min + u * z_range
            z = self._z_min + u * self._z_range
            # θ = θ_* + V_r @ z^T = θ_* + z @ V_r^T
            theta = self._theta_center + z @ self._V_r.T  # (N, r) @ (r, D) = (N, D)
        else:
            # Simple linear: θ = θ_min + u * θ_range
            theta = self._theta_min + u * self._theta_range

        if single:
            return theta[0]
        return theta

    # -----------------------------------------------------------------
    # Utility methods
    # -----------------------------------------------------------------
    def in_physical_bounds(self, theta):
        """
        Check if physical parameters are within effective bounds.
        """
        theta = np.asarray(theta)
        bounds_min = self.effective_bounds[0]
        bounds_max = self.effective_bounds[1]
        return np.all((theta >= bounds_min) & (theta <= bounds_max), axis=-1)

    @property
    def logprior_vol(self):
        """
        Log volume of prior.

        For linear mode: sum(log(θ_range))
        For rotation mode: sum(log(z_range))
        """
        if self._use_rotation:
            return np.sum(np.log(self._z_range))
        else:
            return np.sum(np.log(self._theta_range))

    @property
    def rank(self):
        """Dimensionality of unit cube space."""
        return self._r

    @property
    def uses_rotation(self):
        """Whether rotation mode is active."""
        return self._use_rotation

    def state_dict(self):
        """Serialize transform state for saving/loading."""
        state = {
            'original_bounds': self.original_bounds,
            'ndim': self.ndim,
            'r': self._r,
            'use_rotation': self._use_rotation,
            'effective_bounds': self.effective_bounds,
        }
        if self._use_rotation:
            state.update({
                'covariance_phys': self._covariance_phys,
                'theta_center': self._theta_center,
                'V_r': self._V_r,
                'z_min': self._z_min,
                'z_max': self._z_max,
                'z_range': self._z_range,
            })
        else:
            state.update({
                'theta_min': self._theta_min,
                'theta_max': self._theta_max,
                'theta_range': self._theta_range,
            })
        return state

    @classmethod
    def from_state_dict(cls, state):
        """Restore transform from serialized state."""
        obj = cls.__new__(cls)
        obj.original_bounds = np.array(state['original_bounds'])
        obj.ndim = int(state['ndim'])
        obj._r = int(state['r'])
        obj._use_rotation = state['use_rotation']
        obj.effective_bounds = np.array(state['effective_bounds'])
        
        if obj._use_rotation:
            obj._covariance_phys = np.array(state['covariance_phys'])
            obj._theta_center = np.array(state['theta_center'])
            obj._V_r = np.array(state['V_r'])
            obj._z_min = np.array(state['z_min'])
            obj._z_max = np.array(state['z_max'])
            obj._z_range = np.array(state['z_range'])
        else:
            obj._theta_min = np.array(state['theta_min'])
            obj._theta_max = np.array(state['theta_max'])
            obj._theta_range = np.array(state['theta_range'])
        return obj

    def __repr__(self):
        if self._use_rotation:
            vol = np.prod(self._z_range)
            return (f"ParameterTransform(ndim={self.ndim}, rank={self._r}, "
                    f"mode='rotation', z_vol={vol:.2e})")
        else:
            vol = np.prod(self._theta_range)
            return (f"ParameterTransform(ndim={self.ndim}, mode='linear', "
                    f"phys_vol={vol:.2e})")
