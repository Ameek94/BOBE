"""
Parameter space transforms for BOBE.

Class hierarchy
---------------
BaseTransform       -- abstract interface shared by all transforms
BoxTransform        -- linear scaling between physical bounds and [0,1]^D

The ParameterTransform factory is kept for backward compatibility.
"""

import os
import numpy as np
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

    @property
    @abstractmethod
    def ndim(self): ...

    @property
    @abstractmethod
    def rank(self): ...

    @property
    @abstractmethod
    def effective_bounds(self): ...

    @property
    def logprior_vol(self):
        raise NotImplementedError

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

    def __repr__(self):
        return f"{self.__class__.__name__}(ndim={self.ndim})"


class BoxTransform(BaseTransform):
    """
    Linear scaling between physical bounds and the unit cube.

    u = (theta - theta_min) / (theta_max - theta_min)
    theta = theta_min + u * (theta_max - theta_min)

    All methods are JAX-compatible for use inside JIT-compiled functions.
    """

    _TYPE_KEY = "box"

    def __init__(self, param_bounds):
        bounds = np.asarray(param_bounds, dtype=np.float64)
        if bounds.shape[0] != 2:
            raise ValueError("param_bounds must have shape (2, D)")
        self._original_bounds = bounds
        self._ndim = bounds.shape[1]
        # Store as JAX arrays for compatibility with JIT-compiled functions
        self._theta_min = jnp.asarray(bounds[0])
        self._theta_max = jnp.asarray(bounds[1])
        self._theta_range = self._theta_max - self._theta_min
        self._effective_bounds = bounds.copy()
        log.info(f"BoxTransform: ndim={self._ndim}")
        log.info(f"Physical bounds: {list(zip(bounds[0], bounds[1]))}")

    def to_unit(self, theta, clip=False):
        """Map physical parameters theta -> unit cube u in [0,1]^D. JAX-compatible."""
        theta = jnp.atleast_2d(theta)
        single = theta.shape[0] == 1
        u = (theta - self._theta_min) / self._theta_range
        if clip:
            u = jnp.clip(u, 0.0, 1.0)
        return jnp.squeeze(u) if single else u

    def from_unit(self, u):
        """Map unit cube u in [0,1]^D -> physical parameters theta. JAX-compatible."""
        u = jnp.atleast_2d(u)
        single = u.shape[0] == 1
        theta = self._theta_min + u * self._theta_range
        return jnp.squeeze(theta) if single else theta

    def in_physical_bounds(self, theta):
        """Check if theta is within physical bounds. JAX-compatible."""
        theta = jnp.atleast_1d(theta)
        return jnp.all(
            (theta >= self._theta_min) & (theta <= self._theta_max),
            axis=-1,
        )

    @property
    def ndim(self):
        return self._ndim

    @property
    def rank(self):
        return self._ndim

    @property
    def effective_bounds(self):
        return self._effective_bounds

    @property
    def logprior_vol(self):
        return float(jnp.sum(jnp.log(self._theta_range)))

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
        vol = float(jnp.prod(self._theta_range))
        return f"BoxTransform(ndim={self._ndim}, phys_vol={vol:.2e})"


def ParameterTransform(param_bounds, **kwargs):
    """
    Factory function retained for backward compatibility.
    Always returns BoxTransform (rotation transforms have been removed).
    """
    return BoxTransform(param_bounds)


def load_transform(path):
    """
    Load a BaseTransform from {path}_transform.npz.
    Only BoxTransform is supported.
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

    type_key = state.get("type", "box")
    # Support legacy "identity" key for backward compatibility
    if type_key in (BoxTransform._TYPE_KEY, "identity"):
        return BoxTransform.from_state_dict(state)
    else:
        raise ValueError(f"Unknown transform type key: {type_key!r}")


# Backward compatibility alias
IdentityTransform = BoxTransform
