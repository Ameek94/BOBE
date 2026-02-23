"""
Parameter space transforms for BOBE (fixed).

This module provides the ParameterTransform class which manages all coordinate
transformations between physical parameter space and the unit cube [0,1]^D
used by the GP, acquisition functions, and nested sampler.

Design principles / changes from previous version:
- The affine map from unit cube to physical space is strictly linear and
  preserved: theta = theta_min + theta_range * u.
- Rotations/whitening are applied as a separate linear map from unit-cube
  deviations to a rotated feature space z:
      z = M @ (u - u_center),   M = L_unit.T,  L_unit = V * sqrt(lambda)
  where V, lambda are eigenvectors/eigenvalues of the covariance in unit space.
- We provide explicit unit <-> rotated conversions: unit_to_rotated, rotated_to_unit.
- to_unit/from_unit remain the canonical physical <-> unit-cube transforms.
  from_unit accepts either u (unit-space) or z (rotated-space) when rotation is active.
"""

import numpy as np
from .log import get_logger

log = get_logger("transforms")


class ParameterTransform:
    """
    Manage physical <-> unit-cube transformations, plus optional linear rotation
    based on a covariance or Fisher matrix.

    Public methods:
      - to_unit(theta) -> u
      - from_unit(u_or_z) -> theta  # accepts unit u; if rotation active and shape matches r, accepts z
      - unit_to_rotated(u) -> z
      - rotated_to_unit(z) -> u

    See docstrings on each method for details.
    """

    def __init__(self, param_bounds, rotation_matrix=None, rotation_center=None,
                 rotation_is_fisher=False, n_sigma=0.0, regularize_eps=0.0,
                 rank=None):
        """
        Parameters
        ----------
        param_bounds : array-like (2, D)
            Physical parameter bounds: [theta_min, theta_max]
        rotation_matrix : array-like (D,D) or None
            Covariance matrix in physical space, or Fisher (if rotation_is_fisher=True).
        rotation_center : array-like (D,) or None
            Physical-space center where covariance was computed (e.g. MAP).
            If None, defaults to center of bounds.
        rotation_is_fisher : bool
            If True, invert `rotation_matrix` to obtain covariance (cov = inv(F)).
        regularize_eps : float
            Small diagonal regularizer added to covariance for stability.
        rank : int or None
            Truncate rotation to `rank` leading eigenvectors; if None, use full rank D.
        """
        self.original_bounds = np.asarray(param_bounds, dtype=np.float64)
        if self.original_bounds.shape[0] != 2:
            raise ValueError("param_bounds must have shape (2, D)")
        self.ndim = self.original_bounds.shape[1]

        self._theta_min = self.original_bounds[0].astype(np.float64)
        self._theta_max = self.original_bounds[1].astype(np.float64)
        self._theta_range = self._theta_max - self._theta_min
        if np.any(self._theta_range <= 0.0):
            raise ValueError("All parameter bounds must satisfy upper > lower")

        # Backwards-compatible attributes
        self.has_rotation = False
        self._L_unit = None     # D x r (columns are scaled eigenvectors)
        self._M = None          # r x D mapping from unit deviations to z
        self._M_pinv = None     # D x r pseudo-inverse: maps z -> (u - u_center)
        self._u_center = None   # center in unit coordinates (u_star)
        self._r = None

        if rotation_matrix is not None:
            self._setup_rotation(rotation_matrix, rotation_center,
                                 rotation_is_fisher, regularize_eps,
                                 rank)
        else:
            # No rotation, effective_bounds refer to unit-cube == original physical bounds scaled
            self.effective_bounds = self.original_bounds.copy()
            self._range = self._theta_range.copy()

    # -----------------------------------------------------------------
    # Rotation setup
    # -----------------------------------------------------------------
    def _setup_rotation(self, rotation_matrix, rotation_center,
                        rotation_is_fisher, regularize_eps, rank):
        """
        Compute L_unit (D x r) from covariance in unit coordinates and
        build linear maps M = L_unit^T (r x D) and its pseudo-inverse.

        Steps:
         - If rotation_is_fisher: invert provided Fisher to get cov_phys.
         - Transform to unit-space covariance: cov_unit = D_inv @ cov_phys @ D_inv,
           where D_inv = diag(1 / theta_range).
         - Eigendecompose cov_unit = V Λ V^T.
         - Build L_unit = V[:, :r] * sqrt(Λ[:r]).
         - Set M = L_unit^T; M maps (u - u_center) -> z.
         - Store pseudo-inverse for inverse mapping.
        """
        rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
        if rotation_matrix.shape != (self.ndim, self.ndim):
            raise ValueError(f"rotation_matrix must be ({self.ndim},{self.ndim})")

        # 1) covariance in physical space
        if rotation_is_fisher:
            log.info("Inverting Fisher matrix to obtain covariance (physical space).")
            try:
                cov_phys = np.linalg.inv(rotation_matrix)
            except np.linalg.LinAlgError:
                raise ValueError("Fisher matrix is singular and cannot be inverted.")
        else:
            cov_phys = rotation_matrix.copy()

        # regularize if requested
        if regularize_eps > 0.0:
            cov_phys = cov_phys + regularize_eps * np.eye(self.ndim)

        # ensure symmetry
        cov_phys = 0.5 * (cov_phys + cov_phys.T)

        # quick PD-check (eigenvalues)
        eigvals_phys = np.linalg.eigvalsh(cov_phys)
        min_eval = np.min(eigvals_phys)
        if min_eval <= 0:
            raise ValueError(
                f"Physical covariance not positive definite (min eigenvalue {min_eval:.3e}). "
                "Increase regularize_eps."
            )

        self._covariance_phys = cov_phys

        # 2) transform covariance to unit cube coordinates:
        D_inv = 1.0 / self._theta_range
        # Elementwise scaling for symmetric matrix: Σ_unit[i,j] = Σ_phys[i,j] / (θ_range[i]*θ_range[j])
        cov_unit = cov_phys * np.outer(D_inv, D_inv)
        # Numeric symmetrize
        cov_unit = 0.5 * (cov_unit + cov_unit.T)
        self._covariance_unit = cov_unit

        # 3) eigendecompose cov_unit
        eigvals_unit, eigvecs_unit = np.linalg.eigh(cov_unit)
        # sort descending
        idx = np.argsort(eigvals_unit)[::-1]
        eigvals_unit = eigvals_unit[idx]
        eigvecs_unit = eigvecs_unit[:, idx]

        # default rank
        if rank is None:
            r = self.ndim
        else:
            r = int(rank)
            if r <= 0 or r > self.ndim:
                raise ValueError("rank must be in 1..D")
        self._r = r

        # Build L_unit = V_r * sqrt(Lambda_r)
        lambdas_r = eigvals_unit[:r].copy()
        # numerical guard: ensure nonnegative
        lambdas_r[lambdas_r < 0] = 0.0
        sqrt_lambdas = np.sqrt(lambdas_r)
        V_r = eigvecs_unit[:, :r]    # D x r
        # scale each column of V_r by sqrt(lambda)
        L_unit = V_r * sqrt_lambdas.reshape((1, -1))  # D x r
        self._L_unit = L_unit

        # M maps (u - u_center) -> z
        M = L_unit.T  # r x D
        self._M = M
        # pseudo-inverse maps z -> (u - u_center)
        # use stable pinv (r x D) -> D x r
        self._M_pinv = np.linalg.pinv(M)

        # store unit-space center (u_star) corresponding to rotation_center (physical)
        if rotation_center is not None:
            rotation_center = np.asarray(rotation_center, dtype=np.float64).flatten()
            if rotation_center.shape != (self.ndim,):
                raise ValueError(f"rotation_center must have shape ({self.ndim},)")
            # map to unit cube
            self._u_center = (rotation_center - self._theta_min) / self._theta_range
        else:
            # default center is geometric center in unit-space
            self._u_center = 0.5 * np.ones(self.ndim, dtype=np.float64)

        # effective bounds: we keep these for diagnostic purposes.
        # z = M @ (u - u_center), u in [0,1]^D -> z lives in some parallelepiped in R^r.
        # We can compute axis-aligned bounds for z by extrema of linear map: z_i min/max over corners.
        # But we won't rescale z into [0,1] (we keep z in R^r).
        # Compute z bounds by evaluating M @ (corner - u_center) for all 2^D corners is expensive for large D,
        # so approximate axis-aligned z bounds using column-wise sums:
        # For each row i of M: z_i_min = sum_j min(M[i,j]*(0-u_center[j]), M[i,j]*(1-u_center[j]))
        #                 z_i_max = sum_j max(...)
        M_mat = self._M
        u0 = np.zeros(self.ndim)
        u1 = np.ones(self.ndim)
        # compute contribution extremes per column
        z_min = np.sum(np.minimum(M_mat * (0.0 - self._u_center), M_mat * (1.0 - self._u_center)), axis=1)
        z_max = np.sum(np.maximum(M_mat * (0.0 - self._u_center), M_mat * (1.0 - self._u_center)), axis=1)
        self._z_bounds = np.vstack([z_min, z_max])  # shape (2, r)

        self.has_rotation = True
        # For backwards compatibility set effective_bounds to unit bounds (theta bounds) but document rotated separately
        self.effective_bounds = self.original_bounds.copy()
        self._range = self._theta_range.copy()

        # Logging diagnostics
        cond = np.inf
        try:
            cond = np.max(eigvals_unit) / np.min(eigvals_unit)
        except Exception:
            pass
        log.info(f"Rotation enabled: rank={r}, unit-cov eigenvalues head={eigvals_unit[:min(6, len(eigvals_unit))]}")
        log.info(f"Unit-space covariance condition estimate: {cond:.2e}")
        log.info(f"z-space approx bounds (per-axis): min={z_min}, max={z_max}")

    # -----------------------------------------------------------------
    # Core forward transform: physical -> unit cube (always linear)
    # -----------------------------------------------------------------
    def to_unit(self, theta):
        """
        Map physical parameters theta -> unit cube u in [0,1]^D.

        This is always the simple affine map:
            u = (theta - theta_min) / theta_range

        Parameters
        ----------
        theta : array-like, shape (D,) or (N, D)
        Returns
        -------
        u : ndarray, same shape as theta
        """
        theta = np.asarray(theta, dtype=np.float64)
        single = False
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
            single = True
        if theta.shape[-1] != self.ndim:
            raise ValueError(f"Expected last dimension {self.ndim}")

        u = (theta - self._theta_min) / self._theta_range

        if single:
            return u[0]
        return u

    # -----------------------------------------------------------------
    # Map unit cube -> rotated z and inverse
    # -----------------------------------------------------------------
    def unit_to_rotated(self, u):
        """
        Map unit-cube points u (in [0,1]^D) to rotated coordinates z in R^r:
            z = M @ (u - u_center)

        Parameters
        ----------
        u : array-like, shape (D,) or (N, D)
        Returns
        -------
        z : ndarray, shape (r,) or (N, r)
        """
        if not self.has_rotation:
            raise RuntimeError("Rotation not configured.")

        u = np.asarray(u, dtype=np.float64)
        single = False
        if u.ndim == 1:
            u = u.reshape(1, -1)
            single = True
        if u.shape[-1] != self.ndim:
            raise ValueError(f"Expected last dim {self.ndim}")

        du = (u - self._u_center)  # (N, D)
        # z = (N, r) = du @ M.T? M is r x D, so du (N,D) @ M.T (D,r) gives (N,r) wrong.
        # Instead compute z = (M) @ du.T -> (r,N) then transpose. Simpler: z = du @ M.T
        z = du @ self._M.T  # (N, r)

        if single:
            return z[0]
        return z

    def rotated_to_unit(self, z, clip=True):
        """
        Map rotated coordinates z (shape r or (N,r)) back to unit-cube u via:
            u = M_pinv @ z + u_center
        Optionally clip to [0,1].

        Returns u in [0,1]^D (if clip=True).
        """
        if not self.has_rotation:
            raise RuntimeError("Rotation not configured.")

        z = np.asarray(z, dtype=np.float64)
        single = False
        if z.ndim == 1:
            z = z.reshape(1, -1)
            single = True
        if z.shape[-1] != self._r:
            raise ValueError(f"Expected last dim {self._r} for rotated space")

        # u_offset = (D, N) = M_pinv (D x r) @ z.T (r x N)
        u_offset = (self._M_pinv @ z.T).T  # (N, D)
        u = u_offset + self._u_center
        if clip:
            u = np.clip(u, 0.0, 1.0)

        if single:
            return u[0]
        return u

    # -----------------------------------------------------------------
    # Canonical inverse: unit cube -> physical
    # -----------------------------------------------------------------
    def from_unit(self, u_or_z):
        """
        Map either (a) unit-cube coordinates u (shape D) -> physical theta,
        or (b) rotated coordinates z (shape r) -> physical theta (if rotation active).

        - If rotation is not active: expect u input and return theta = theta_min + u * theta_range.
        - If rotation is active and input last-dim == r: treated as z -> converts to u via pseudo-inverse,
          then maps to theta (clipping u to [0,1] first).
        - Otherwise expects u and maps to theta.

        Returns theta in physical bounds.
        """
        arr = np.asarray(u_or_z, dtype=np.float64)
        single = False
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
            single = True

        last_dim = arr.shape[-1]
        if self.has_rotation and last_dim == self._r:
            # Input is rotated z
            u = self.rotated_to_unit(arr, clip=True)
        else:
            # Treat as unit-cube u
            if last_dim != self.ndim:
                raise ValueError(f"Input last-dim must be {self.ndim} (unit) or {self._r} (rotated)")
            u = arr
            # enforce [0,1] safety
            u = np.clip(u, 0.0, 1.0)

        theta = self._theta_min + u * self._theta_range

        if single:
            return theta[0]
        return theta

    # -----------------------------------------------------------------
    # Utility methods
    # -----------------------------------------------------------------
    def in_physical_bounds(self, theta):
        theta = np.asarray(theta)
        return np.all((theta >= self._theta_min) & (theta <= self._theta_max), axis=-1)

    @property
    def logprior_vol(self):
        return np.sum(np.log(self._theta_range))

    def state_dict(self):
        state = {
            'original_bounds': self.original_bounds,
            'has_rotation': self.has_rotation,
            'ndim': self.ndim,
        }
        if self.has_rotation:
            state.update({
                'covariance_phys': self._covariance_phys,
                'covariance_unit': self._covariance_unit,
                'rotation_center_unit': self._u_center,
                'L_unit': self._L_unit,
                'M': self._M,
                'M_pinv': self._M_pinv,
                'z_bounds': self._z_bounds,
                'r': self._r,
            })
        return state

    @classmethod
    def from_state_dict(cls, state):
        obj = cls.__new__(cls)
        obj.original_bounds = np.array(state['original_bounds'])
        obj.ndim = int(state['ndim'])
        obj._theta_min = obj.original_bounds[0]
        obj._theta_max = obj.original_bounds[1]
        obj._theta_range = obj._theta_max - obj._theta_min

        obj.has_rotation = bool(state['has_rotation'])
        obj._L_unit = None
        obj._M = None
        obj._M_pinv = None
        obj._u_center = None
        obj._r = None

        if obj.has_rotation:
            obj._covariance_phys = np.array(state['covariance_phys'])
            obj._covariance_unit = np.array(state['covariance_unit'])
            obj._u_center = np.array(state['rotation_center_unit'])
            obj._L_unit = np.array(state['L_unit'])
            obj._M = np.array(state['M'])
            obj._M_pinv = np.array(state['M_pinv'])
            obj._z_bounds = np.array(state['z_bounds'])
            obj._r = int(state['r'])
            obj.effective_bounds = obj.original_bounds.copy()
            obj._range = obj._theta_range.copy()
        else:
            obj.effective_bounds = obj.original_bounds.copy()
            obj._range = obj._theta_range.copy()

        return obj

    def __repr__(self):
        if self.has_rotation:
            vol_est = np.prod(self._z_bounds[1] - self._z_bounds[0])
            return (f"ParameterTransform(ndim={self.ndim}, rotation=True, rank={self._r}, "
                    f"z_vol_approx={vol_est:.2e})")
        return f"ParameterTransform(ndim={self.ndim}, rotation=False)"
