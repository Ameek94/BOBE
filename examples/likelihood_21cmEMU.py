import os
from dataclasses import dataclass

import numpy as np
from py21cmemu import Emulator

PARAMETER_NAMES = [
    "F_STAR10",
    "ALPHA_STAR",
    "F_ESC10",
    "ALPHA_ESC",
    "M_TURN",
    "t_STAR",
    "L_X",
    "NU_X_THRESH",
]

PARAMETER_LABELS = [
    r"\log_{10}(f_{*,10})",
    r"\alpha_*",
    r"\log_{10}(f_{\mathrm{esc},10})",
    r"\alpha_{\mathrm{esc}}",
    r"\log_{10}(M_{\mathrm{turn}})",
    r"t_*",
    r"\log_{10}\left(\frac{L_{X<2\,\mathrm{keV}}}{\mathrm{SFR}}\right)",
    r"E_0",
]

FIDUCIAL_THETA = np.array([
    -1.30,
     0.50,
    -1.00,
    -0.50,
     8.70,
     0.50,
    40.50,
   500.0,
], dtype=float)

SIGMA_LO = np.array([0.02, 0.02, 0.02, 0.015, 0.02, 0.02, 0.0015, 0.95], dtype=float)
SIGMA_HI = np.array([0.02, 0.02, 0.02, 0.015, 0.02, 0.02, 0.0015, 1.40], dtype=float)

Z_MIN = 6.0
Z_MAX = 30.0
K_MIN = 0.1
K_MAX = 1.0


def default_param_bounds(nsigma: float = 5.0) -> np.ndarray:
    return np.array([
        FIDUCIAL_THETA - nsigma * SIGMA_LO,
        FIDUCIAL_THETA + nsigma * SIGMA_HI,
    ])


@dataclass
class Likelihood21cmEMU:
    include_norm: bool = False
    x_ray_spec_index: float = 1.0
    nsigma_bounds: float = 5.0
    param_bounds: np.ndarray | None = None
    emu_instance: object | None = None

    # Temporary placeholder observational model
    fractional_sigma: float = 0.10

    # Active parameter control
    varying_names: list[str] | None = None

    def __post_init__(self):
        self.full_param_names = PARAMETER_NAMES
        self.full_param_labels = PARAMETER_LABELS
        self.full_fiducial_theta = FIDUCIAL_THETA.copy()

        if self.param_bounds is None:
            self.full_param_bounds = default_param_bounds(self.nsigma_bounds)
        else:
            self.full_param_bounds = np.asarray(self.param_bounds, dtype=float)
            if self.full_param_bounds.shape != (2, 8):
                raise ValueError(
                    f"param_bounds must have shape (2, 8), got {self.full_param_bounds.shape}"
                )

        if self.varying_names is None:
            self.varying_names = list(self.full_param_names)

        unknown = [name for name in self.varying_names if name not in self.full_param_names]
        if unknown:
            raise ValueError(f"Unknown varying parameter names: {unknown}")

        self.varying_indices = [self.full_param_names.index(name) for name in self.varying_names]

        self.param_names = [self.full_param_names[i] for i in self.varying_indices]
        self.param_labels = [self.full_param_labels[i] for i in self.varying_indices]
        self.fiducial_theta = self.full_fiducial_theta[self.varying_indices]
        self.param_bounds = self.full_param_bounds[:, self.varying_indices]

        self.ndim = len(self.varying_indices)

        self.emu = Emulator() if self.emu_instance is None else self.emu_instance

        self.z_full = np.asarray(self.emu.properties.PS_zs, dtype=float)
        self.k_full = np.asarray(self.emu.properties.PS_ks, dtype=float)

        self.z_mask = (self.z_full > Z_MIN) & (self.z_full < Z_MAX)
        self.k_mask = (self.k_full > K_MIN) & (self.k_full < K_MAX)

        self.z = self.z_full[self.z_mask]
        self.k = self.k_full[self.k_mask]

        self.data_ps = self._compute_model_ps_full(self.full_fiducial_theta)

        self.sigma_ps = self.fractional_sigma * np.maximum(np.abs(self.data_ps), 1e-12)
        self.mask = np.isfinite(self.data_ps) & np.isfinite(self.sigma_ps) & (self.sigma_ps > 0)

    def expand_theta(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.ndim,):
            raise ValueError(f"Expected theta with shape ({self.ndim},), got {theta.shape}")

        theta_full = self.full_fiducial_theta.copy()
        theta_full[self.varying_indices] = theta
        return theta_full

    def _theta_to_input_dict(self, theta_full: np.ndarray) -> dict:
        return {
            "F_STAR10": float(theta_full[0]),
            "ALPHA_STAR": float(theta_full[1]),
            "F_ESC10": float(theta_full[2]),
            "ALPHA_ESC": float(theta_full[3]),
            "M_TURN": float(theta_full[4]),
            "t_STAR": float(theta_full[5]),
            "L_X": float(theta_full[6]),
            "NU_X_THRESH": float(theta_full[7]),
            "X_RAY_SPEC_INDEX": float(self.x_ray_spec_index),
        }

    def _compute_model_ps_full(self, theta_full: np.ndarray) -> np.ndarray:
        theta_full = np.asarray(theta_full, dtype=float)
        if theta_full.shape != (8,):
            raise ValueError(f"Expected full theta with shape (8,), got {theta_full.shape}")

        _, output, _ = self.emu.predict(self._theta_to_input_dict(theta_full))

        ps_full = np.asarray(output["PS"], dtype=float)
        expected_shape = (len(self.z_full), len(self.k_full))
        if ps_full.shape != expected_shape:
            raise ValueError(f"Unexpected PS shape {ps_full.shape}, expected {expected_shape}")

        return ps_full[np.ix_(self.z_mask, self.k_mask)]

    def _compute_model_ps(self, theta: np.ndarray) -> np.ndarray:
        theta_full = self.expand_theta(theta)
        return self._compute_model_ps_full(theta_full)

    def gaussian_loglike(self, model_ps: np.ndarray) -> float:
        if model_ps.shape != self.data_ps.shape or model_ps.shape != self.sigma_ps.shape:
            raise ValueError(
                f"Shape mismatch: model {model_ps.shape}, data {self.data_ps.shape}, sigma {self.sigma_ps.shape}"
            )

        resid = model_ps[self.mask] - self.data_ps[self.mask]
        var = self.sigma_ps[self.mask] ** 2
        chi2 = np.sum((resid ** 2) / var)

        if self.include_norm:
            return -0.5 * (chi2 + np.sum(np.log(2.0 * np.pi * var)))
        return -0.5 * chi2

    def __call__(self, theta: np.ndarray) -> float:
        model_ps = self._compute_model_ps(theta)
        return self.gaussian_loglike(model_ps)

    def loglike(self, theta: np.ndarray) -> float:
        return self(theta)