from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
import jax.numpy as jnp
from BOBE.utils.log import get_logger
from BOBE.utils.core import renormalise_log_weights
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
    

class PrincipalAxesTransform(InputTransform):
    """
    Linear transform form Fisher metric:
       Rotate: z = Q^T (x - x0)
       whiten: z = diag(sqrt(lambda)) Q^T (x - x0)
    Stores A and Ainv so forward/inverse are cheap and clean
    """

    name: str = "principal_axes"

    def __init__(self, 
                 rotation_matrix: jnp.ndarray = None, 
                 rotation_centre: jnp.ndarray = None, 
                 rotation_is_fisher: bool = False, 
                 rotation_samples: jnp.ndarray = None,
                 rotation_logwt: jnp.ndarray = None,
                 rotation_logl: jnp.ndarray = None,
                 rotation_top_frac: float = 1.0,
                 mode="rotate", 
                 learn_rotation=False, 
                 eig_floor: float = 1e-10):
        """
        Initialise PrincipalAxesTransform class (probably need to change name to covariance principal axes transform).

        Parameters
        ----------
        rotation_matrix: array, optional
            User defined rotation matrix. Requires rotation_centre be defined
        rotation_centre: array, optional
            User defined centre of rotation. Requires rotation_matrix be defined
        rotation_is_fisher: bool
            Indicates whether rotation_matrix is a fisher matrix (True) or covariance matrix (False)
        rotation_samples: array, optional
            User defined samples to calculate a rotation matrix and rotation centre from
        rotation_logwt: array, optional
            User defined log weights to use in calculation of rotation matrix and rotation centre
        rotation_logl: array, optional
            User defined log likelihood values to use in calculation of rotation matrix and rotation centre
        mode: str
            Defines what type of transform to apply (rotate or whiten) [DEPRECATED]
        learn_rotation: bool
            Tells the rotation class whether it should learn the rotation and update 
        """
        self.mode = mode
        self.eig_floor = eig_floor

        self.rotation_matrix = None
        self.rotation_centre = None
        self.rotation_is_fisher = False

        self.is_active = False
        self.learn_rotation = learn_rotation
        self.Q = None
        self.lam = None
        self.A = None
        self.Ainv = None
        
        self.samples = None
        self.weights = None
        self.logwt = None
        self.logl = None
        self.top_frac = rotation_top_frac

        log.info(f"Weighting? :{'False' if rotation_logwt is None  else 'True'}, Top Frac: {rotation_top_frac}")

        user_matrix_rotation = rotation_matrix is not None or rotation_centre is not None
        user_sample_rotation = rotation_samples is not None

        if user_matrix_rotation and user_sample_rotation:
            raise ValueError("Provide either rotation_matrix/rotation_centre or rotation_samples, not both")
        
        elif user_matrix_rotation:
            if rotation_matrix is None or rotation_centre is None:
                raise ValueError("Must provide both rotation_matrix and rotation_centre, or neither")
            self.rotation_matrix = rotation_matrix
            self.rotation_centre = rotation_centre
            self.rotation_is_fisher = rotation_is_fisher
            #self.update(
                #rotation_matrix=rotation_matrix, 
                #rotation_centre=rotation_centre, 
                #rotation_is_fisher=rotation_is_fisher,
            #)

        elif user_sample_rotation:
            if rotation_logl is None:
                raise ValueError("rotation_logl must be provided with rotation_samples")
            self.samples=rotation_samples
            self.logwt=rotation_logwt
            self.logl=rotation_logl
            self.top_frac=rotation_top_frac
            #self.update(
                # samples=rotation_samples,
                # logwt=rotation_logwt,
                # logl=rotation_logl,
                # top_frac=rotation_top_frac,
            #)
    
    def _install(self, cov, centre):
        self.rotation_matrix = cov
        self.rotation_centre = centre

        eigvals, eigvecs = jnp.linalg.eigh(cov)
        idx = jnp.argsort(eigvals)[::-1]

        self.Q = eigvecs[:, idx]
        self.lam = eigvals[idx]

        if self.mode == "rotate":
            self.A = self.Q
            self.Ainv = self.Q.T
        else:
            sqrt_lam = jnp.sqrt(jnp.maximum(self.lam, self.eig_floor))
            self.A = self.Q / sqrt_lam[None, :]
            self.Ainv = self.Q.T * sqrt_lam[:, None]
        self.is_active = True

    def update(self, 
            #    samples=None, 
            #    rotation_matrix=None,
            #    rotation_centre=None,
            #    rotation_is_fisher=False,
            #    weights=None, 
            #    logwt=None, 
            #    logl=None, 
            #    top_frac=None, 
                centre='mean', 
                regularise_eps=1e-10,
                centre_tol=None,
                rot_tol=None,
                cov_tol=None,
                ):
        """
        Build + Optionally update rotation and centre

        Parameters
        ----------
        samples : (N, D), optional
            Samples in GP input space used to build a covariance rotation.
        rotation_matrix : (D, D), optional
            User-provided covariance or Fisher matrix.
        rotation_centre : (D,), optional
            Centre associated with rotation_matrix.
        rotation_is_fisher : bool, optional
            If True, interpret rotation_matrix as a Fisher matrix and invert it.
        weights : (N,), optional
            Sample weights.
        logwt : (N,), optional
            Nested-sampling log weights.
        logl : (N,), optional
            Log-likelihood values.
        top_frac : float, optional
            Keep only the top fraction of samples by logl.
        centre : {"map", "mean"}, optional
            How to define the centre from samples.
        regularise_eps : float, optional
            Diagonal regularisation added to covariance.
        centre_tol, rot_tol, cov_tol : float, optional
            Thresholds for accepting a new rotation candidate.

        Returns
        -------
        dict
        Contains:
            did_update : bool
            initialised : bool
            delta_centre : float, optional
            delta_rot : float, optional
            delta_cov : float, optional
        """
        if self.rotation_matrix is None and self.samples is None:
            raise ValueError("No stored rotation source available for installation")


        # --- Build Candidate --- #
        if self.rotation_matrix is not None:
            if self.rotation_centre is None:
                raise ValueError("rotaiton_centre required with rotation_matrix")
            
            cov = self.rotation_matrix
            if self.rotation_is_fisher:
                cov = jnp.linalg.inv(cov)
                cov = 0.5 * (cov + cov.T)
            
            if regularise_eps > 0:
                cov += regularise_eps * jnp.eye(cov.shape[0])
            x0 = self.rotation_centre
            
        else:
            cov, x0 = self.rotation_from_samples(
                samples=self.samples,
                weights=self.weights,
                logwt=self.logwt,
                logl=self.logl,
                top_frac=self.top_frac,
                centre=centre,
                regularise_eps=regularise_eps
            )

        # --- First time initialisation --- #
        if not self.is_active:
            self._install(cov, x0)
            return {"did_update": True, "initialised": True}
        
        # --- Always update if no thresholds --- #
        if centre_tol is None and rot_tol is None and cov_tol is None:
            self._install(cov, x0)
            return {"did_update": True, "initialised": False}
        
        # --- Check Metrics --- #
        metrics = self.rotation_metrics(cov, x0)

        did_update = (
            (centre_tol is not None and metrics["delta_centre"] > centre_tol)
            or (rot_tol is not None and metrics["delta_rot"] > rot_tol)
            or (cov_tol is not None and metrics["delta_cov"] > cov_tol)
        )

        if did_update:
            self._install(cov, x0)
        
        return {"did_update": did_update, "initialised": False, **metrics}

    def rotation_from_samples(self, 
                              samples=None,  
                              weights=None, 
                              logwt=None, 
                              logl=None, 
                              top_frac=None, 
                              centre="mean", 
                              regularise_eps=1e-12):
        """
        Build covariance rotation + centre from samples.

        Parameters
        ----------
        samples: (N, D)
            Samples in GP input space
        weights: (N,) optional
            Direct sample weights (e.g. from MCMC importance weights)
        logwt: (N,), optional
            Nested-sampling log-weights
        logl: (N,), optional
            Log-likelihood values, used for MAP centre or top-fraction filtering.
        top_frac: float, optional
            Keep only the top fraction of samples by logl before computing covariance.
        centre: str
            "mean" for weighted mean centre of rotation, "map" for highest-logl sample.
        regularise_eps: float
            Diagnoal jitter added to covariance.


        Returns
        -------
        cov: (D, D)
        x0: (D,)
        weights_used: (N_used,)
        used_idx: (N_used,)
        """

        if samples is None:
            raise ValueError("Samples must be provided")
        
        n = samples.shape[0]

        if samples.ndim != 2:
            raise ValueError(f"samples must have shape (N, D), got {samples.shape}")
        
        if logwt is not None:
            weights = renormalise_log_weights(logwt)
        elif weights is not None:
            weights = weights / jnp.sum(weights)
        else:
            weights = jnp.ones(n)/n

        # # Top Fraction
        if top_frac is not None:
            if logl is None:
                raise ValueError("top_frac requires logl")

            n_keep = max(2, int(jnp.ceil(top_frac * n)))
            keep = jnp.argsort(logl)[-n_keep:]

            samples = samples[keep]
            weights = weights[keep]
            weights = weights / jnp.sum(weights)
            logl = logl[keep]

        # Centre
        if centre == "mean":
            x0 = jnp.sum(weights[:, None] * samples, axis=0)
        elif centre == "map":
            if logl is None:
                raise ValueError("centre='map' requires logl")
            x0 = samples[jnp.argmax(logl)]
        else:
            raise ValueError("centre must be 'mean' or 'map'")

        # Covariance
        xc = samples - x0
        cov = (xc * weights[:, None]).T @ xc
        cov = 0.5 * (cov + cov.T)

        if regularise_eps > 0:
            cov += regularise_eps * jnp.eye(cov.shape[0])

        return cov, x0        

    
    def rotation_metrics(self, cov_new, centre_new):
        """
        Compare a candidate rotation against the current transform

        Parameters
        ----------
        cov_new:
            Candidate covariance matrix
        centre_new:
            Candidate rotation centre
        
        Returns
        -------
        dict with:
            delta_centre
            delta_rot
            delta_cov
        """

        if not self.is_active:
            return {
                "delta_centre": jnp.inf,
                "delta_rot": jnp.inf,
                "delta_cov": jnp.inf
            }
        
        cov_old = self.rotation_matrix

        # Centre Shift
        delta_centre = jnp.linalg.norm(centre_new - self.rotation_centre)

        # Rotation change via principal axes overlap
        eigvals_new, eigvecs_new = jnp.linalg.eigh(cov_new)
        Q_new = eigvecs_new[:, jnp.argsort(eigvals_new)[::-1]]

        M = self.Q.T @ Q_new
        delta_rot = 1.0 - jnp.mean(jnp.abs(jnp.diag(M)))

        # Covariance change
        denom = jnp.linalg.norm(cov_old, ord="fro")
        delta_cov = jnp.linalg.norm(cov_new - cov_old, ord="fro") / jnp.maximum(denom, 1e-15)

        return {
            "delta_centre": delta_centre,
            "delta_rot": delta_rot,
            "delta_cov": delta_cov,
        }
        
    
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return (x - self.rotation_centre) @ self.A
    
    def inverse(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.rotation_centre + z @ self.Ainv
    
    def state_dict(self):
        return {
            "type": "PrincipalAxesTransform",
            "mode": self.mode,
            "rotation_centre": self.rotation_centre,
            "rotation_matrix": self.rotation_matrix,
            "samples": self.samples,
            "weights": self.weights,
            "logwt": self.logwt,
            "logl": self.logl,
            "top_frac": self.top_frac,
            "rotation_is_fisher": self.rotation_is_fisher,
            "Q": self.Q,
            "lam": self.lam,
            "A": self.A,
            "Ainv": self.Ainv,
            "learn_rotation": self.learn_rotation,
            "is_active": self.is_active,
        }
    
    @classmethod
    def from_state_dict(cls, state_dict: dict) -> PrincipalAxesTransform:
        obj = cls.__new__(cls)  # Create an uninitialized instance
        obj.name = "fisher_principal_axes"
        obj.mode = state_dict["mode"]
        obj.eig_floor = 1e-10  # default value; not stored in state_dict
        obj.rotation_centre = state_dict["rotation_centre"]
        obj.rotation_matrix = state_dict["rotation_matrix"]
        obj.samples = state_dict.get("samples", None)
        obj.weights = state_dict.get("weights", None)
        obj.logwt = state_dict.get("logwt", None)
        obj.logl = state_dict.get("logl", None)
        obj.top_frac = state_dict.get("top_frac", 1.0)
        obj.rotation_is_fisher = state_dict.get("rotation_is_fisher", False)
        obj.Q = state_dict["Q"]
        obj.lam = state_dict["lam"]
        obj.A = state_dict["A"]
        obj.Ainv = state_dict["Ainv"]
        obj.learn_rotation = state_dict["learn_rotation"]
        obj.is_active = state_dict["is_active"]
        return obj