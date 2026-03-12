from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
import jax.numpy as jnp
from BOBE.utils.log import get_logger
import jax

log = get_logger("transforms")


class InputTransform(ABC):
    """
    Base interface for any input transform.
    Regularised space <-> GP Space
    """

    name: str = "base"

    @abstractmethod
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        regularised space -> GP space. Supports (D,) and (N,D)
        """
        raise NotImplementedError
    
    @abstractmethod
    def inverse(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        GP space -> regularised space. Supports (D,) and (N,D)
        """
        raise NotImplementedError
    

class FisherPrincipalAxesTransform(InputTransform):
    """
    Linear transform form Fisher metric:
       Rotate: z = Q^T (x - x0)
       whiten: z = diag(sqrt(lambda)) Q^T (x - x0)
    Stores A and Ainv so forward/inverse are cheap and clean
    """

    name: str = "fisher_principal_axes"

    def __init__(self, rotation_matrix: jnp.ndarray, rotation_center: jnp.ndarray, rotation_is_fisher: bool = False, mode="rotate", eig_floor: float = 1e-10):
        # if mode == 'whiten':
        #     raise NotImplementedError("Whitening not yet implemented for FisherPrincipalAxesTransform")
        self.mode = mode
        self.eig_floor = eig_floor
        if rotation_center is None:
            raise ValueError("rotation_center must be provided for FisherPrincipalAxesTransform")
        self.rotation_center = jnp.array(rotation_center)
        self.rotation_matrix = jnp.array(rotation_matrix)

        self.update(rotation_matrix=rotation_matrix, rotation_center=self.rotation_center, rotation_is_fisher=rotation_is_fisher)

        log.info(f"Initialized FisherPrincipalAxesTransform with mode {self.mode}")
    
    def update(self, rotation_matrix: jnp.ndarray, rotation_center: jnp.ndarray | None = None, rotation_is_fisher: bool = False, regularise_eps: float = 0.0, gp=None):
        if self.mode not in ("rotate", "whiten"):
            raise ValueError(f"Invalid mode {self.mode}, must be 'rotate' or 'whiten'")
        
        if rotation_center is None:
            raise ValueError("x0 must be provided for FisherPrincipalAxesTransform")
        self.rotation_center = jnp.array(rotation_center)

        if rotation_matrix is None:
            raise ValueError("rotation_matrix must be provided for FisherPrincipalAxesTransform")
        self.rotation_matrix = jnp.asarray(rotation_matrix)

        if rotation_is_fisher:
            F = (self.rotation_matrix + self.rotation_matrix.T) / 2.0 # ensure symmetry
            cov = jnp.linalg.inv(F)
            cov = (cov + cov.T) / 2.0  # ensure symmetry
        else:
            cov = (self.rotation_matrix + self.rotation_matrix.T) / 2.0  # ensure symmetry
        
        if regularise_eps > 0.0:
            cov += regularise_eps * jnp.eye(cov.shape[0], dtype=cov.dtype)

        eigvals_check = jnp.linalg.eigvalsh(cov)
        min_eval = jnp.min(eigvals_check)
        if min_eval <= 0:
            raise ValueError(
                f"Covariance not positive definite (min eigenvalue {min_eval:.3e}). "
                "Increase regularize_eps."
            )

        eigvals, eigvecs = jnp.linalg.eigh(cov)

        # sort descending
        idx = jnp.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        V = eigvecs[:, idx]

        # store
        self.Q = V
        self.lam = eigvals

        # Cache linear maps for speed/clarity
        if self.mode == "rotate":
            self.A = V
            self.Ainv = V.T
        else:
            sqrt_lam = jnp.sqrt(jnp.maximum(eigvals, self.eig_floor))
            self.A = V * sqrt_lam[None, :]
            self.Ainv = (V.T / sqrt_lam[:, None])

        log.info(f"Forward Transform test: fisher_MAP -> GP Space: {self.forward(self.rotation_center)}")
        
    
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return (x - self.rotation_center) @ self.A
    
    def inverse(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.rotation_center + z @ self.Ainv
    
    def state_dict(self):
        return {
            "type": "FisherPrincipalAxesTransform",
            "mode": self.mode,
            "rotation_center": self.rotation_center,
            "rotation_is_fisher": False, # We don't need this to reconstruct the transform as we already have the transforms themselves.
            "Q": self.Q,
            "lam": self.lam,
            "A": self.A,
            "Ainv": self.Ainv,
        }
    
    @classmethod
    def from_state_dict(cls, state_dict: dict) -> FisherPrincipalAxesTransform:
        obj = cls.__new__(cls)  # Create an uninitialized instance
        obj.name = "fisher_principal_axes"
        obj.mode = state_dict["mode"]
        obj.eig_floor = 1e-10  # default value; not stored in state_dict
        obj.rotation_center = state_dict["rotation_center"]
        obj.Q = state_dict["Q"]
        obj.lam = state_dict["lam"]
        obj.A = state_dict["A"]
        obj.Ainv = state_dict["Ainv"]
        return obj