"""
Normalizing Flow transform for BOBE parameter transformations.

Implements a Neural Spline Flow (rational quadratic splines in coupling layers)
that maps physical parameters θ to a unit-cube space u ∈ [0,1]^D that is
GP-friendly.

Transform pipeline (forward: θ → u):
  1. Linear pre-scaling:  v = (θ - θ_min) / θ_range  ∈ [0,1]^D
  2. Logit gate:          w = logit(clip(v, ε, 1-ε))  ∈ R^D
  3. Coupling layers:      z = flow(w)                  ∈ R^D   (z ≈ N(0,I) after training)
  4. Normal CDF squash:   u = Φ(z)                     ∈ [0,1]^D

The log-likelihood the GP emulates in u-space has the form
  log L(θ(u)) ≈ C - ½‖Φ⁻¹(u)‖² + smooth_residual
which is a smooth quadratic bowl — GP-friendly.

Inverse (u → θ):
  u → z = Φ⁻¹(u) → w = flow⁻¹(z) → v = σ(w) → θ = θ_min + v * θ_range

When the flow is not yet trained, both directions fall back to plain linear
scaling (step 1 only), so behaviour before the first flow update is identical
to the existing ParameterTransform linear mode.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import ndtri as jax_probit   # Φ⁻¹ (standard normal quantile)
from jax.scipy.stats.norm import cdf as jax_normal_cdf  # Φ

jax.config.update("jax_enable_x64", True)

import optax

from .log import get_logger

log = get_logger("flow")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FlowConfig:
    """
    Hyperparameters for the Neural Spline Flow.

    Attributes
    ----------
    n_layers : int
        Number of coupling layers (more = more expressive, slower to train).
    hidden_features : int
        Width of the coupling conditioner network.
    n_bins : int
        Number of spline bins per dimension per layer.
    tail_bound : float
        Spline operates on [-tail_bound, +tail_bound]; values outside are
        passed through linearly (identity beyond the tails).
    learning_rate : float
        Adam learning rate.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    clip_grad_norm : float
        Global gradient norm clipping value.
    min_samples_to_train : int
        Minimum number of physical-space samples required to (re-)train.
    """
    n_layers: int = 6
    hidden_features: int = 64
    n_bins: int = 8
    tail_bound: float = 4.0
    learning_rate: float = 1e-3
    n_epochs: int = 200
    batch_size: int = 128
    clip_grad_norm: float = 5.0
    min_samples_to_train: int = 50


# ---------------------------------------------------------------------------
# Rational Quadratic Spline (element-wise, operates on a single dimension)
# ---------------------------------------------------------------------------

def _rqs_forward(
    x: jnp.ndarray,
    widths: jnp.ndarray,
    heights: jnp.ndarray,
    derivatives: jnp.ndarray,
    tail_bound: float,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Forward rational quadratic spline for a single dimension.

    Parameters
    ----------
    x           : (N,)  inputs
    widths      : (N, K) unnormalised bin widths
    heights     : (N, K) unnormalised bin heights
    derivatives : (N, K+1) unnormalised derivatives at knots
    tail_bound  : scalar, spline domain = [-B, B]

    Returns
    -------
    y          : (N,) outputs
    log_det    : (N,) per-sample log |dy/dx|
    """
    B = tail_bound
    K = widths.shape[-1]

    # Normalise to get valid widths / heights / derivatives
    W = jax.nn.softmax(widths, axis=-1)
    W = min_bin_width + (1.0 - min_bin_width * K) * W
    H = jax.nn.softmax(heights, axis=-1)
    H = min_bin_height + (1.0 - min_bin_height * K) * H
    D = min_derivative + jax.nn.softplus(derivatives)

    # Cumulative knot positions
    cumW = jnp.concatenate([jnp.zeros((*W.shape[:-1], 1)), jnp.cumsum(W, axis=-1)], axis=-1)
    cumW = cumW * (2 * B) - B   # scale to [-B, B]
    cumH = jnp.concatenate([jnp.zeros((*H.shape[:-1], 1)), jnp.cumsum(H, axis=-1)], axis=-1)
    cumH = cumH * (2 * B) - B

    # Bin index for each input
    bin_idx = jnp.sum(x[:, None] >= cumW[:, :-1], axis=-1) - 1
    bin_idx = jnp.clip(bin_idx, 0, K - 1)

    # Gather per-sample bin parameters
    N = x.shape[0]
    idx = jnp.arange(N)

    w_k  = cumW[idx, bin_idx + 1] - cumW[idx, bin_idx]   # bin width  (Δx)
    h_k  = cumH[idx, bin_idx + 1] - cumH[idx, bin_idx]   # bin height (Δy)
    d_k  = D[idx, bin_idx]
    d_k1 = D[idx, bin_idx + 1]
    s_k  = h_k / w_k                                       # slope

    xi = (x - cumW[idx, bin_idx]) / w_k                   # fractional position in bin
    xi1 = 1.0 - xi
    xi_xi1 = xi * xi1

    # RQS formula
    num = h_k * (s_k * xi**2 + d_k * xi_xi1)
    den = s_k + (d_k + d_k1 - 2.0 * s_k) * xi_xi1
    y_in = cumH[idx, bin_idx] + num / den

    # Log derivative
    d_num = s_k**2 * (d_k1 * xi**2 + 2.0 * s_k * xi_xi1 + d_k * xi1**2)
    log_det_in = jnp.log(d_num) - 2.0 * jnp.log(jnp.abs(den))

    # Outside interval: identity (linear; log_det = 0)
    inside = (x >= -B) & (x <= B)
    y = jnp.where(inside, y_in, x)
    log_det = jnp.where(inside, log_det_in, 0.0)

    return y, log_det


def _rqs_inverse(
    y: jnp.ndarray,
    widths: jnp.ndarray,
    heights: jnp.ndarray,
    derivatives: jnp.ndarray,
    tail_bound: float,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Inverse rational quadratic spline (y → x)."""
    B = tail_bound
    K = widths.shape[-1]

    W = jax.nn.softmax(widths, axis=-1)
    W = min_bin_width + (1.0 - min_bin_width * K) * W
    H = jax.nn.softmax(heights, axis=-1)
    H = min_bin_height + (1.0 - min_bin_height * K) * H
    D = min_derivative + jax.nn.softplus(derivatives)

    cumW = jnp.concatenate([jnp.zeros((*W.shape[:-1], 1)), jnp.cumsum(W, axis=-1)], axis=-1)
    cumW = cumW * (2 * B) - B
    cumH = jnp.concatenate([jnp.zeros((*H.shape[:-1], 1)), jnp.cumsum(H, axis=-1)], axis=-1)
    cumH = cumH * (2 * B) - B

    bin_idx = jnp.sum(y[:, None] >= cumH[:, :-1], axis=-1) - 1
    bin_idx = jnp.clip(bin_idx, 0, K - 1)

    N = y.shape[0]
    idx = jnp.arange(N)

    w_k  = cumW[idx, bin_idx + 1] - cumW[idx, bin_idx]
    h_k  = cumH[idx, bin_idx + 1] - cumH[idx, bin_idx]
    d_k  = D[idx, bin_idx]
    d_k1 = D[idx, bin_idx + 1]
    s_k  = h_k / w_k

    # Solve quadratic: a·ξ² + b·ξ + c = 0
    y_c = y - cumH[idx, bin_idx]
    a = h_k * (s_k - d_k) + y_c * (d_k + d_k1 - 2.0 * s_k)
    b = h_k * d_k - y_c * (d_k + d_k1 - 2.0 * s_k)
    c = -s_k * y_c
    discriminant = jnp.clip(b**2 - 4.0 * a * c, a_min=0.0)
    xi = (2.0 * c) / (-b - jnp.sqrt(discriminant))
    xi = jnp.clip(xi, 0.0, 1.0)
    xi1 = 1.0 - xi
    xi_xi1 = xi * xi1

    x_in = xi * w_k + cumW[idx, bin_idx]

    # Log det of *forward* evaluated at this xi, then negate
    den = s_k + (d_k + d_k1 - 2.0 * s_k) * xi_xi1
    d_num = s_k**2 * (d_k1 * xi**2 + 2.0 * s_k * xi_xi1 + d_k * xi1**2)
    log_det_in = jnp.log(d_num) - 2.0 * jnp.log(jnp.abs(den))

    inside = (y >= -B) & (y <= B)
    x = jnp.where(inside, x_in, y)
    log_det = jnp.where(inside, -log_det_in, 0.0)   # negative for inverse

    return x, log_det


# ---------------------------------------------------------------------------
# Coupling layer (operates on all D dims simultaneously; conditions on half)
# ---------------------------------------------------------------------------

def _conditioner_forward(params: Dict, x_cond: jnp.ndarray) -> jnp.ndarray:
    """2-hidden-layer MLP mapping x_cond → spline params."""
    h = jax.nn.tanh(jnp.dot(x_cond, params['w1']) + params['b1'])
    h = jax.nn.tanh(jnp.dot(h, params['w2']) + params['b2'])
    return jnp.dot(h, params['w3']) + params['b3']


def _coupling_forward(
    layer: Dict,
    x: jnp.ndarray,    # (N, D)
    n_bins: int,
    tail_bound: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward pass of one coupling layer."""
    # c_idx / t_idx are numpy int arrays — static to JAX JIT
    c_idx = layer['c_idx']   # conditioning (pass-through) dim indices
    t_idx = layer['t_idx']   # transformed dim indices
    D_t   = len(t_idx)

    # Condition only on pass-through dims; static numpy indexing is safe in JIT
    if len(c_idx) > 0:
        x_cond = x[:, c_idx]              # (N, D_cond)
    else:
        x_cond = jnp.zeros((x.shape[0], 1))  # dummy for D=1 edge case
    raw = _conditioner_forward(layer['net'], x_cond)  # (N, D_t * (3K+1))

    n_params = 3 * n_bins + 1
    raw = raw.reshape(x.shape[0], D_t, n_params)

    W  = raw[..., :n_bins]
    H  = raw[..., n_bins:2 * n_bins]
    Dv = raw[..., 2 * n_bins:]    # (N, D_t, K+1)

    x_t = x[:, t_idx]             # (N, D_t)

    # Vectorise spline over D_t  (loop index d is a concrete Python int via vmap)
    y_t, log_dets = jax.vmap(
        lambda d: _rqs_forward(x_t[:, d], W[:, d, :], H[:, d, :], Dv[:, d, :], tail_bound)
    )(jnp.arange(D_t))
    # y_t : (D_t, N), log_dets : (D_t, N)
    y_t     = y_t.T                     # (N, D_t)
    log_det = log_dets.sum(axis=0)      # (N,)

    y = x.at[:, t_idx].set(y_t)
    return y, log_det


def _coupling_inverse(
    layer: Dict,
    y: jnp.ndarray,    # (N, D)
    n_bins: int,
    tail_bound: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Inverse of one coupling layer."""
    c_idx = layer['c_idx']
    t_idx = layer['t_idx']
    D_t   = len(t_idx)

    if len(c_idx) > 0:
        y_cond = y[:, c_idx]
    else:
        y_cond = jnp.zeros((y.shape[0], 1))
    raw = _conditioner_forward(layer['net'], y_cond)

    n_params = 3 * n_bins + 1
    raw = raw.reshape(y.shape[0], D_t, n_params)

    W  = raw[..., :n_bins]
    H  = raw[..., n_bins:2 * n_bins]
    Dv = raw[..., 2 * n_bins:]

    y_t = y[:, t_idx]

    y_t_arr, log_dets = jax.vmap(
        lambda d: _rqs_inverse(y_t[:, d], W[:, d, :], H[:, d, :], Dv[:, d, :], tail_bound)
    )(jnp.arange(D_t))
    y_t_arr = y_t_arr.T
    log_det  = log_dets.sum(axis=0)

    x = y.at[:, t_idx].set(y_t_arr)
    return x, log_det


# ---------------------------------------------------------------------------
# Helper: initialise a conditioner MLP
# ---------------------------------------------------------------------------

def _init_conditioner(D_cond: int, D_t: int, hidden: int, n_bins: int,
                      key: jax.Array) -> Dict:
    n_out = D_t * (3 * n_bins + 1)
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / D_cond)
    scale2 = jnp.sqrt(2.0 / hidden)
    scale3 = jnp.sqrt(2.0 / hidden)
    return {
        'w1': random.normal(k1, (D_cond, hidden)) * scale1,
        'b1': jnp.zeros(hidden),
        'w2': random.normal(k2, (hidden, hidden)) * scale2,
        'b2': jnp.zeros(hidden),
        'w3': random.normal(k3, (hidden, n_out)) * scale3,
        'b3': jnp.zeros(n_out),
    }


# ---------------------------------------------------------------------------
# FlowTransform — drop-in replacement for ParameterTransform
# ---------------------------------------------------------------------------

_EPS = 1e-6   # clip margin for logit gate


class FlowTransform:
    """
    Normalizing-flow-based parameter transform for BOBE.

    Exposes the same interface as ``ParameterTransform``:
      - ``to_unit(theta, clip=True)``  → u ∈ [0,1]^D
      - ``from_unit(u)``               → theta ∈ R^D
      - ``uses_rotation``  property    → True (always, once trained)
      - ``state_dict()``
      - ``FlowTransform.from_state_dict(state)``

    Transform pipeline (forward θ → u):
      θ --[linear]--> v ∈ [0,1]^D --[logit]--> w ∈ R^D
        --[coupling layers]--> z ≈ N(0,I) --[Φ(·)]--> u ∈ [0,1]^D

    Before the flow has been trained, ``to_unit`` / ``from_unit`` fall back to
    plain linear scaling (identical to ParameterTransform linear mode).

    Parameters
    ----------
    param_bounds : array (2, D)
        Physical parameter bounds [[lo_1, ...], [hi_1, ...]].
    config : FlowConfig or None
        Flow hyper-parameters; defaults to FlowConfig().
    seed : int
        JAX PRNGKey seed.
    """

    def __init__(
        self,
        param_bounds: np.ndarray,
        config: Optional[FlowConfig] = None,
        seed: int = 0,
    ):
        self.original_bounds = np.asarray(param_bounds, dtype=np.float64)
        if self.original_bounds.shape[0] != 2:
            raise ValueError("param_bounds must have shape (2, D)")
        self.ndim = self.original_bounds.shape[1]
        self._r = self.ndim   # unit-cube dimensionality (same as D)

        self._theta_min   = self.original_bounds[0]
        self._theta_max   = self.original_bounds[1]
        self._theta_range = self._theta_max - self._theta_min
        # Guard against zero-range parameters
        zero_range = self._theta_range == 0.0
        if np.any(zero_range):
            self._theta_range = np.where(zero_range, 1.0, self._theta_range)
            log.warning(f"Zero-range parameter(s) at dims {np.where(zero_range)[0].tolist()}; "
                        "set range=1 to avoid division by zero.")

        self.effective_bounds = self.original_bounds.copy()

        self.config = config if config is not None else FlowConfig()
        self._key = random.PRNGKey(seed)

        # Build alternating masks and initialise coupling-layer parameters
        self._layers: List[Dict] = self._init_layers()

        self.flow_trained = False
        self.training_losses: List[float] = []

        log.info(f"FlowTransform initialised: D={self.ndim}, "
                 f"layers={self.config.n_layers}, bins={self.config.n_bins}, "
                 f"hidden={self.config.hidden_features}")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_layers(self) -> List[Dict]:
        layers = []
        for i in range(self.config.n_layers):
            self._key, sk = random.split(self._key)
            # Alternating mask: even layers pass dims 0,2,4…; odd layers pass 1,3,5…
            mask = np.zeros(self.ndim, dtype=bool)
            if i % 2 == 0:
                mask[::2] = True
            else:
                mask[1::2] = True
            D_cond = int(mask.sum())
            D_t    = self.ndim - D_cond
            if D_t == 0:          # edge case: D=1 — transform everything
                D_t = self.ndim
                D_cond = 0
                mask[:] = False
            # Precompute index arrays as numpy so JAX treats them as static in JIT
            c_idx = np.where(mask)[0].astype(np.int32)    # conditioning dims
            t_idx = np.where(~mask)[0].astype(np.int32)   # transformed dims
            net = _init_conditioner(max(D_cond, 1),
                                    D_t, self.config.hidden_features,
                                    self.config.n_bins, sk)
            layers.append({'c_idx': c_idx, 't_idx': t_idx, 'net': net})
        return layers

    # ------------------------------------------------------------------
    # Forward pipeline helpers (all in JAX for JIT)
    # ------------------------------------------------------------------

    def _theta_to_w(self, theta_np: np.ndarray) -> jnp.ndarray:
        """Physical → logit space (no flow yet)."""
        v = (theta_np - self._theta_min) / self._theta_range
        v = np.clip(v, _EPS, 1.0 - _EPS)
        return jnp.array(np.log(v / (1.0 - v)), dtype=jnp.float64)   # logit

    def _w_to_theta(self, w: jnp.ndarray) -> np.ndarray:
        """Logit space → physical."""
        v = jax.scipy.special.expit(w)  # sigmoid
        return np.array(v) * self._theta_range + self._theta_min

    def _flow_forward(self, w: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply coupling layers: w → z, (N,) log_det."""
        log_det = jnp.zeros(w.shape[0])
        x = w
        for layer in self._layers:
            x, ld = _coupling_forward(layer, x,
                                      self.config.n_bins, self.config.tail_bound)
            log_det = log_det + ld
        return x, log_det

    def _flow_inverse(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply inverse coupling layers: z → w."""
        log_det = jnp.zeros(z.shape[0])
        x = z
        for layer in reversed(self._layers):
            x, ld = _coupling_inverse(layer, x,
                                      self.config.n_bins, self.config.tail_bound)
            log_det = log_det + ld
        return x, log_det

    # ------------------------------------------------------------------
    # Public interface: to_unit / from_unit
    # ------------------------------------------------------------------

    def to_unit(self, theta: np.ndarray, clip: bool = True) -> np.ndarray:
        """
        Map physical θ → unit cube u ∈ [0,1]^D.

        Pipeline: θ → v (linear) → w (logit) → z (flow) → u = Φ(z).
        Falls back to plain linear scaling when flow is not yet trained.

        Parameters
        ----------
        theta : array (D,) or (N, D)
        clip : bool  — clip final u to [0,1] (default True)

        Returns
        -------
        u : array same shape as theta
        """
        theta = np.asarray(theta, dtype=np.float64)
        single = theta.ndim == 1
        if single:
            theta = theta.reshape(1, -1)
        if theta.shape[-1] != self.ndim:
            raise ValueError(f"Expected {self.ndim} dims, got {theta.shape[-1]}")

        if not self.flow_trained:
            # Fallback: simple linear
            u = (theta - self._theta_min) / self._theta_range
        else:
            v = np.clip(theta, self._theta_min + _EPS * self._theta_range,
                        self._theta_max - _EPS * self._theta_range)
            v = (v - self._theta_min) / self._theta_range
            v = np.clip(v, _EPS, 1.0 - _EPS)
            w = np.log(v / (1.0 - v))          # logit: matches training transform
            z, _ = self._flow_forward(jnp.array(w, dtype=jnp.float64))
            # Normal CDF squash z → u ∈ [0,1]
            u = np.array(jax_normal_cdf(z))

        if clip:
            u = np.clip(u, 0.0, 1.0)
        return u[0] if single else u

    def log_abs_det_jacobian(self, theta: np.ndarray) -> np.ndarray:
        """
        Log absolute value of the Jacobian determinant  log|∂u/∂θ|  at θ.

        All four stages of the forward pipeline contribute:

          log|∂v/∂θ|  = -sum(log(θ_range))          (linear scaling, constant)
          log|∂w/∂v|  = sum(-log v - log(1-v))       (logit, per dim)
          log|∂z/∂w|  = flow log-det                 (coupling layers)
          log|∂u/∂z|  = sum(log φ(z_d))              (Φ squash, per dim)

        When the flow is untrained only the constant linear term is returned.

        Parameters
        ----------
        theta : array (D,) or (N, D) — physical parameter values

        Returns
        -------
        log_det : array (N,) — one scalar per sample
        """
        theta = np.asarray(theta, dtype=np.float64)
        single = theta.ndim == 1
        if single:
            theta = theta.reshape(1, -1)
        N = theta.shape[0]

        # --- constant linear term: log|∂v/∂θ| = -sum(log θ_range) ---
        log_det_linear = -np.sum(np.log(self._theta_range))  # scalar

        if not self.flow_trained:
            return np.full(N, log_det_linear)

        # --- logit term: log|∂w/∂v|, w = logit(v) ---
        v = np.clip(theta, self._theta_min + _EPS * self._theta_range,
                    self._theta_max - _EPS * self._theta_range)
        v = (v - self._theta_min) / self._theta_range
        v = np.clip(v, _EPS, 1.0 - _EPS)
        # d(logit)/dv = 1/(v(1-v))
        log_det_logit = np.sum(-np.log(v) - np.log(1.0 - v), axis=-1)  # (N,)

        # --- flow coupling layers: log|∂z/∂w| ---
        w_jnp = jnp.array(np.log(v / (1.0 - v)), dtype=jnp.float64)
        z, log_det_flow = self._flow_forward(w_jnp)
        log_det_flow = np.array(log_det_flow)                           # (N,)

        # --- Φ squash: log|∂u/∂z| = sum_d log φ(z_d) where φ is the standard normal PDF ---
        z_np = np.array(z)
        log_det_phi = np.sum(-0.5 * z_np**2 - 0.5 * np.log(2.0 * np.pi), axis=-1)  # (N,)

        log_det = log_det_linear + log_det_logit + log_det_flow + log_det_phi
        return log_det[0] if single else log_det

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        """
        Map unit cube u ∈ [0,1]^D → physical θ.

        Pipeline: u → z = Φ⁻¹(u) → w (inv-flow) → v = σ(w) → θ.
        Falls back to plain linear scaling when flow is not yet trained.

        Parameters
        ----------
        u : array (D,) or (N, D)

        Returns
        -------
        theta : array same shape as u
        """
        u = np.asarray(u, dtype=np.float64)
        single = u.ndim == 1
        if single:
            u = u.reshape(1, -1)
        if u.shape[-1] != self.ndim:
            raise ValueError(f"Expected {self.ndim} dims, got {u.shape[-1]}")

        if not self.flow_trained:
            theta = self._theta_min + u * self._theta_range
        else:
            u_safe = np.clip(u, _EPS, 1.0 - _EPS)
            z = np.array(jax_probit(jnp.array(u_safe, dtype=jnp.float64)))  # Φ⁻¹
            w, _ = self._flow_inverse(jnp.array(z))
            v = np.array(jax.scipy.special.expit(w))   # sigmoid → [0,1]
            theta = self._theta_min + v * self._theta_range

        return theta[0] if single else theta

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        samples_physical: np.ndarray,
        verbose: bool = True,
        reinit: bool = False,
    ) -> float:
        """
        (Re-)train the normalizing flow to map p(θ) → N(0,I).

        Maximises log p_flow(w) on transformed samples w = logit((θ - θ_min)/θ_range).

        Parameters
        ----------
        samples_physical : array (N, D)
            Training samples in physical parameter space.
        verbose : bool
            Log training progress every 20 epochs.
        reinit : bool
            If True, re-initialise network weights before training (cold start).
            Default False = warm start (continue from current weights).

        Returns
        -------
        final_loss : float
        """
        N = samples_physical.shape[0]
        if N < self.config.min_samples_to_train:
            log.warning(f"[FlowTransform] Only {N} samples — need at least "
                        f"{self.config.min_samples_to_train}; skipping training.")
            return float('inf')

        if reinit:
            self._layers = self._init_layers()
            log.info("[FlowTransform] Re-initialised network weights (cold start).")

        log.info(f"[FlowTransform] Training on {N} physical-space samples "
                 f"({'warm' if not reinit else 'cold'} start, "
                 f"{self.config.n_epochs} epochs)...")

        # Pre-process: physical → logit space  (done in numpy for speed)
        theta = np.asarray(samples_physical, dtype=np.float64)
        v = np.clip((theta - self._theta_min) / self._theta_range, _EPS, 1.0 - _EPS)
        w_np = np.log(v / (1.0 - v))   # logit, shape (N, D)
        w_jnp = jnp.array(w_np, dtype=jnp.float64)

        # Separate differentiable network weights from static index arrays.
        # JAX's grad cannot handle integer/bool leaves, so we pass only `nets`
        # to value_and_grad and capture indices in the loss closure.
        nets    = [l['net']   for l in self._layers]   # list of float dicts — differentiable
        c_idxs  = [l['c_idx'] for l in self._layers]   # numpy int arrays — static
        t_idxs  = [l['t_idx'] for l in self._layers]   # numpy int arrays — static

        # Setup optimizer (fresh state, but parameters warm-started)
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.clip_grad_norm),
            optax.adam(self.config.learning_rate),
        )
        opt_state = optimizer.init(nets)

        # Loss: negative mean log-prob under N(0,I) after flow transform.
        n_bins_    = self.config.n_bins
        tail_bound = self.config.tail_bound
        ndim_      = self.ndim

        def loss_fn(nets_, batch):
            log_det = jnp.zeros(batch.shape[0])
            x = batch
            for net_, c_idx_, t_idx_ in zip(nets_, c_idxs, t_idxs):
                layer_ = {'c_idx': c_idx_, 't_idx': t_idx_, 'net': net_}
                x, ld = _coupling_forward(layer_, x, n_bins_, tail_bound)
                log_det = log_det + ld
            # z = x; log p(z) under N(0,I)
            log_pz = (-0.5 * jnp.sum(x**2, axis=-1)
                      - 0.5 * ndim_ * jnp.log(2.0 * jnp.pi))
            return -jnp.mean(log_pz + log_det)

        loss_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

        losses = []

        for epoch in range(self.config.n_epochs):
            # Shuffle
            self._key, permkey = random.split(self._key)
            perm   = np.array(random.permutation(permkey, N))
            w_shuf = w_jnp[perm]

            n_batches  = max(1, N // self.config.batch_size)
            epoch_loss = 0.0
            for b in range(n_batches):
                batch = w_shuf[b * self.config.batch_size:(b + 1) * self.config.batch_size]
                loss_val, grads = loss_grad_fn(nets, batch)
                updates, opt_state = optimizer.update(grads, opt_state)
                nets       = optax.apply_updates(nets, updates)
                epoch_loss += float(loss_val)
            epoch_loss /= n_batches
            losses.append(epoch_loss)

            if verbose and (epoch + 1) % 20 == 0:
                log.info(f"  [Flow] Epoch {epoch+1}/{self.config.n_epochs}  "
                         f"loss={epoch_loss:.4f}")

        # Merge updated nets back into self._layers
        for i, net in enumerate(nets):
            self._layers[i]['net'] = net

        self.training_losses.extend(losses)
        self.flow_trained = True

        final_loss = losses[-1]
        log.info(f"[FlowTransform] Training complete. Final loss: {final_loss:.4f}")
        return final_loss

    # ------------------------------------------------------------------
    # Properties (match ParameterTransform interface)
    # ------------------------------------------------------------------

    @property
    def uses_rotation(self) -> bool:
        """Always True — FlowTransform always acts as a non-linear rotation."""
        return True

    @property
    def rank(self) -> int:
        return self._r

    @property
    def logprior_vol(self) -> float:
        """Log prior volume (physical space, for nested sampling)."""
        return float(np.sum(np.log(self._theta_range)))

    def in_physical_bounds(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta)
        return np.all((theta >= self._theta_min) & (theta <= self._theta_max), axis=-1)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Serialise the full flow state (parameters + config)."""
        # Convert JAX arrays in layers to numpy for storage
        def _to_np(x):
            if isinstance(x, jnp.ndarray):
                return np.array(x)
            return x

        def _layer_to_np(layer):
            net = {k: _to_np(v) for k, v in layer['net'].items()}
            return {'c_idx': layer['c_idx'], 't_idx': layer['t_idx'], 'net': net}

        return {
            'original_bounds': self.original_bounds,
            'ndim': self.ndim,
            'r': self._r,
            'config': {
                'n_layers': self.config.n_layers,
                'hidden_features': self.config.hidden_features,
                'n_bins': self.config.n_bins,
                'tail_bound': self.config.tail_bound,
                'learning_rate': self.config.learning_rate,
                'n_epochs': self.config.n_epochs,
                'batch_size': self.config.batch_size,
                'clip_grad_norm': self.config.clip_grad_norm,
                'min_samples_to_train': self.config.min_samples_to_train,
            },
            'layers': [_layer_to_np(l) for l in self._layers],
            'flow_trained': self.flow_trained,
            'training_losses': list(self.training_losses),
            # Linear fallback info
            'theta_min': self._theta_min,
            'theta_max': self._theta_max,
            'theta_range': self._theta_range,
            'effective_bounds': self.effective_bounds,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> 'FlowTransform':
        """Reconstruct from a serialised state dict."""
        cfg_dict  = state['config']
        config    = FlowConfig(**cfg_dict)
        bounds    = np.array(state['original_bounds'])
        obj       = cls.__new__(cls)

        obj.original_bounds  = bounds
        obj.ndim             = int(state['ndim'])
        obj._r               = int(state['r'])
        obj.config           = config
        obj._key             = random.PRNGKey(0)  # not used after training
        obj.flow_trained     = bool(state['flow_trained'])
        obj.training_losses  = list(state.get('training_losses', []))
        obj._theta_min       = np.array(state['theta_min'])
        obj._theta_max       = np.array(state['theta_max'])
        obj._theta_range     = np.array(state['theta_range'])
        obj.effective_bounds = np.array(state['effective_bounds'])

        def _layer_from_np(raw):
            c_idx = np.array(raw['c_idx'], dtype=np.int32)
            t_idx = np.array(raw['t_idx'], dtype=np.int32)
            net   = {k: jnp.array(v) for k, v in raw['net'].items()}
            return {'c_idx': c_idx, 't_idx': t_idx, 'net': net}

        obj._layers = [_layer_from_np(l) for l in state['layers']]
        log.info(f"FlowTransform restored: D={obj.ndim}, trained={obj.flow_trained}, "
                 f"layers={config.n_layers}")
        return obj

    def __repr__(self) -> str:
        status = f"trained (loss={self.training_losses[-1]:.3f})" \
                 if self.flow_trained else "untrained (linear fallback)"
        return (f"FlowTransform(ndim={self.ndim}, layers={self.config.n_layers}, "
                f"bins={self.config.n_bins}, {status})")
