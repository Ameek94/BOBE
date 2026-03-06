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


        ### DEBUG ###
        if gp is not None:
            D = int(self.rotation_center.shape[0])

            # ---- basic sanity -------------------------------------------------
            # centre should map to exactly 0 (up to float eps)
            z0 = self.forward(self.rotation_center)
            print(f"[tform] D={D}  ||forward(center)||_inf = {float(jnp.max(jnp.abs(z0))):.3e}")

            # orthonormality check (rotation mode should be orthonormal)
            # (A^T A ≈ I) and (Ainv ≈ A^T)
            ATA = self.A.T @ self.A
            I = jnp.eye(D, dtype=ATA.dtype)
            ortho_err = jnp.max(jnp.abs(ATA - I))
            inv_err = jnp.max(jnp.abs(self.Ainv - self.A.T))
            print(f"[tform] ortho max|A^T A - I| = {float(ortho_err):.3e}   max|Ainv - A^T| = {float(inv_err):.3e}")

            # eigenvalue / conditioning info (in the same convention as the other impl)
            lam = self.lam
            lam_min = float(jnp.min(lam))
            lam_max = float(jnp.max(lam))
            cond = lam_max / lam_min if lam_min > 0 else jnp.inf
            print(f"[tform] cov eig min/max = {lam_min:.3e} / {lam_max:.3e}   cond(cov)={float(cond):.3e}")

            # ---- inverse consistency -----------------------------------------
            # random probe in unit cube neighbourhood to make sure inverse(forward(x)) is identity
            # (use GP context if provided; else just use center + small jitter)
            key = jax.random.PRNGKey(0)
            eps = 1e-3
            probe = self.rotation_center + eps * jax.random.normal(key, (D,), dtype=self.rotation_center.dtype)
            recon = self.inverse(self.forward(probe))
            recon_err = float(jnp.max(jnp.abs(recon - probe)))
            print(f"[tform] max|inverse(forward(probe)) - probe| = {recon_err:.3e}")

            # ---- dataset geometry diagnostics (if you pass gp) ----------------
            if gp is not None:
                X = gp.train_x
                if X is None:
                    print("[tform] gp.train_x is None; skipping dataset diagnostics")
                    return
                X = jnp.asarray(X)
                n = int(X.shape[0])
                if n < 3:
                    print(f"[tform] n_train={n}; skipping dataset diagnostics")
                    return

                Xr = self.forward(X)

                # correlation reduction (use covariance->corr; avoid numpy)
                def corrcoef_jax(Y):
                    Y = Y - jnp.mean(Y, axis=0, keepdims=True)
                    C = (Y.T @ Y) / Y.shape[0]
                    d = jnp.sqrt(jnp.clip(jnp.diag(C), 1e-30, None))
                    return C / (d[:, None] * d[None, :])

                C0 = corrcoef_jax(X)
                C1 = corrcoef_jax(Xr)

                off0 = jnp.mean(jnp.abs(C0 - jnp.eye(D, dtype=C0.dtype)))
                off1 = jnp.mean(jnp.abs(C1 - jnp.eye(D, dtype=C1.dtype)))
                print(f"[tform] n_train={n}  mean|corr offdiag| raw={float(off0):.3e}  rot={float(off1):.3e}")

                # distance geometry (this is the one that matters for an RBF)
                # Compare mean squared Euclidean distance before/after rotation.
                # NOTE: rotation about center preserves Euclidean distances exactly
                #       if A is orthonormal. So this should be ~identical for rotate-mode.
                def mean_sqdist(Y):
                    # sample a few pairs cheaply
                    m = min(n, 256)
                    idx = jnp.arange(m)
                    Y0 = Y[idx]
                    # pair with a rolled version
                    Y1 = jnp.roll(Y0, shift=1, axis=0)
                    return float(jnp.mean(jnp.sum((Y0 - Y1) ** 2, axis=1)))

                d0 = mean_sqdist(X)
                d1 = mean_sqdist(Xr)
                print(f"[tform] mean sqdist (paired, cheap) raw={d0:.6e} rot={d1:.6e}")

                # if you want a *non-invariance* diagnostic (often the smoking gun):
                # compare distances AFTER lengthscale scaling, because your kernel uses x/ell.
                # This WILL differ if the rotation aligns data with ell updates.
                ell = getattr(gp.kernel, "lengthscales", None)
                if ell is not None:
                    ell = jnp.asarray(ell)
                    Xs0 = X / ell
                    Xs1 = Xr / ell
                    ds0 = mean_sqdist(Xs0)
                    ds1 = mean_sqdist(Xs1)
                    print(f"[tform] mean sqdist after /lengthscales raw={ds0:.6e} rot={ds1:.6e}")
                    b = jnp.asarray(gp.kernel.bounds_spec["lengthscales"])
                    if b.shape == (2,):
                        lo, hi = b[0], b[1]
                        at_lo = float(jnp.mean(ell <= lo * 1.001))
                        at_hi = float(jnp.mean(ell >= hi * 0.999))
                        print(f"[tform] mean(log ell)={float(jnp.mean(jnp.log(ell))):.3f}  frac@lo={at_lo:.2f}  frac@hi={at_hi:.2f}")
                    else:
                        # per-dim bounds
                        lo = b[:,0]; hi = b[:,1]
                        at_lo = float(jnp.mean(ell <= lo * 1.001))
                        at_hi = float(jnp.mean(ell >= hi * 0.999))
                        print(f"[tform] mean(log ell)={float(jnp.mean(jnp.log(ell))):.3f}  frac@lo={at_lo:.2f}  frac@hi={at_hi:.2f}")
        
    
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