"""
Module to estimate Fisher matrix from a quick BOBE run.

The Fisher Information Matrix (FIM) is computed as the negative Hessian of
the GP mean surrogate expressed directly in physical parameter space:

    F = -∇²_θ μ_GP(θ)|_{θ*}

using JAX automatic differentiation.

Fisher run phases (see fisher.md for the design doc):
  Phase I  — EI acquisition to locate the peak θ*.
             Terminates when EI < ei_goal for 2 consecutive batches.
  Phase II — Local WIPStd refinement within V_peak = θ* ± α·ℓ (unit-cube),
             converging when the FIM KL divergence is small for kl_n_iters
             consecutive batches.  Also tracks the WIPStd integral < 0.5
             as a secondary quality metric.
  Phase III — Exact FIM via jax.hessian of μ_GP(θ) in physical space.
"""

import jax
import numpy as np
import jax.numpy as jnp
from scipy.stats import qmc

from .bo import BOBE
from .acquisition import WIPStd
from .utils.core import kl_divergence_gaussian
from .utils.log import get_logger

log = get_logger("fisher")


class Fisher(BOBE):
    """
    Estimate the Fisher Information Matrix (FIM) from a BOBE run.

    Workflow
    --------
    Phase I  — EI acquisition until improvement < *ei_goal* for 2 consecutive
               batches (via :meth:`BOBE.run` with ``acq='ei'``).
    Phase II — Local WIPStd refinement restricted to a hyper-rectangle V_peak
               defined by GP lengthscales around θ*.  Stops when the symmetric
               KL divergence between consecutive FIM-defined Gaussians drops
               below *kl_tol* for *kl_n_iters* successive batches, or the
               WIPStd integral drops below 0.5.
    Phase III — Exact Hessian via JAX AD directly in physical parameter space
               (no Jacobian chain-rule needed).

    Attributes
    ----------
    fisher_matrix_physical : ndarray, shape (ndim, ndim)
        FIM in physical space.  Pass as ``rotation_matrix`` to a full BOBE
        run with ``rotation_is_fisher=True`` to whiten the parameter space.
    fisher_peak_physical : ndarray, shape (ndim,)
        Expansion point (best-fit) in physical space.
    fisher_peak_u : ndarray, shape (r,)
        Expansion point in unit-cube / rotated space.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fisher_matrix_physical = None
        self.fisher_peak_u = None
        self.fisher_peak_physical = None
        self.fisher_matrix_u = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self,
            acq='logei',
            ei_goal: float = 1e-2,
            phase2_batches: int = 20,
            kl_tol: float = 1e-3,
            kl_n_iters: int = 2,
            local_alpha: float = 0.25,
            **ei_kwargs):
        """
        Run Fisher estimation (Phases I – III).

        Parameters
        ----------
        ei_goal : float
            EI convergence threshold for Phase I.  Stops when EI < *ei_goal*
            for 2 consecutive batches.  Default 0.5 (per the design doc).
        phase2_batches : int
            Maximum WIPStd batches in Phase II.  Default 20.
        kl_tol : float
            Symmetric KL tolerance for FIM convergence in Phase II.  Default 1e-3.
        kl_n_iters : int
            Consecutive batches below *kl_tol* required for Phase II convergence.
            Default 2.
        local_alpha : float
            V_peak half-width as a multiple of GP lengthscales.  Default 0.1.
        **ei_kwargs
            Forwarded to :meth:`BOBE.run` (e.g. ``max_evals``, ``min_evals``).
        """
        # Phase I: EI peak-finding.  BOBE.run() closes the pool on exit.
        super().run(acq=acq, ei_goal=ei_goal, convergence_n_iters=2, **ei_kwargs)

        if not self.is_main:
            return

        # Phase II: local WIPStd refinement (likelihood called directly,
        # bypassing the MPI pool which is closed after Phase I).
        self._run_local_wipstd(
            max_batches=phase2_batches,
            kl_tol=kl_tol,
            kl_n_iters=kl_n_iters,
            local_alpha=local_alpha,
        )

        # Phase III: Fisher extraction
        self.fisher_matrix_physical = self.estimate_fisher_matrix()
        self.fisher_peak_physical = self.transform.from_unit(self.fisher_peak_u)
        log.info("Fisher matrix estimation complete.")
        log.info(f"Peak (physical): {self.fisher_peak_physical}")
        log.info(f"Fisher matrix (physical):\n{self.fisher_matrix_physical}")

        np.savetxt(f"{self.save_dir}/{self.loglikelihood.name}_cov_matrix.txt", np.linalg.inv(self.fisher_matrix_physical))
        np.savetxt(f"{self.save_dir}/{self.loglikelihood.name}_cov_peak.txt", self.fisher_peak_physical)

        return {
            'fisher_matrix': self.fisher_matrix_physical,
            'fisher_peak': self.fisher_peak_physical,
            'gp': self.gp,
            # 'results': self.results_manager,
            'likelihood': self.loglikelihood,   }

    # ------------------------------------------------------------------
    # Phase II: local WIPStd loop
    # ------------------------------------------------------------------

    def _run_local_wipstd(self, max_batches: int, kl_tol: float,
                          kl_n_iters: int, local_alpha: float):
        """
        Phase II: Local WIPStd refinement around the likelihood peak.

        MC samples for WIPStd are drawn uniformly within:
            V_peak = [best_u - alpha·ℓ, best_u + alpha·ℓ] ∩ [0,1]^r

        Convergence metrics (per design doc §4):
          1. Integrated Uncertainty: WIPStd integral < 0.5.
          2. FIM KL Divergence: D_KL(G_t || G_{t-1}) < kl_tol for kl_n_iters
             consecutive batches, where G_t = N(θ*, F_t^{-1}).
        """
        log.info("Phase II: Local WIPStd refinement around the peak")
        acq = WIPStd(optimizer=self.optimizer)
        prev_F = None
        kl_count = 0
        self.mc_points_size = 256

        for step in range(1, max_batches + 1):
            

            # Get current Fisher
            self.fisher_peak_u = np.array(self.gp.train_x[int(jnp.argmax(self.gp.train_y))].flatten())
            self.fisher_matrix_u = self.estimate_fisher_matrix_u()

            # ---- local bounds in unit-cube space ------------------------
            best_u, (lo, hi) = self._get_local_bounds(local_alpha)

            # # ---- MC samples uniform inside V_peak -----------------------
            # ndim_u = len(lo)
            # sobol = qmc.Sobol(ndim_u, scramble=True, rng=self.np_rng).random(512)
            # mc_x = lo + sobol * (hi - lo)
            # mc_samples = {'x': mc_x, 'weights': np.ones(len(mc_x)), 'method': 'uniform'}

            # MC samples from drawn from Fisher Gaussian approximation (for better convergence metric stability)
            samples = np.random.multivariate_normal(
                mean=self.fisher_peak_u,
                cov=np.linalg.inv(self.fisher_matrix_u + 1e-12 * np.eye(len(self.fisher_peak_u))),
                size=self.mc_points_size,)
            samples = np.clip(samples, lo, hi)
            mc_samples = {'x': samples, 'weights': np.ones(len(samples))}


            # ---- WIPStd next point restricted to V_peak -----------------
            acq_kwargs = {'mc_samples': mc_samples}
            new_pts_u, acq_vals = acq.get_next_batch(
                gp=self.gp, n_batch=4,
                acq_kwargs=acq_kwargs,
                n_restarts=1, maxiter=100, early_stop_patience=10,
            )
            new_pts_u = jnp.atleast_2d(jnp.array(new_pts_u))

            # ---- evaluate likelihood directly (pool closed after Phase I)
            new_pts = np.asarray(self.transform.from_unit(new_pts_u))
            new_vals = jnp.array(
                [[self.loglikelihood(pt) for pt in new_pts]], dtype=jnp.float64
            ).reshape(-1, 1)

            # ---- update GP with new data and refit locally ---------------
            self.gp.update(new_pts_u, new_vals)
            fit_result = self.gp.fit()   # single-restart local fit, no MPI pool needed
            # Apply best hyperparams with concrete values to flush any leaked JAX tracers
            # that neg_mll stores in self.kernel during JIT-traced optimization.
            self.gp.update_hyperparams(fit_result['params'])

            # ---- Metric 1: WIPStd integral (Integrated Uncertainty) ------
            wipstd_val = float(np.mean(np.abs(acq_vals)))
            log.info(f"Phase II step {step}/{max_batches}: WIPStd integral = {wipstd_val:.4f}")
            if wipstd_val < 5.:
                log.info("  Integrated Uncertainty < 0.5: emulator smooth enough for Hessian")

                # ---- Metric 2: FIM KL divergence ----------------------------
                if prev_F is not None:
                    kl_sym = self._fim_kl(prev_F, prev_best, self.fisher_matrix_u, self.fisher_peak_u)
                    log.info(f"  FIM KL (symmetric) = {kl_sym:.6f}, tol = {kl_tol}")
                    if kl_sym < kl_tol:
                        kl_count += 1
                        if kl_count >= kl_n_iters:
                            log.info(f"Phase II converged after {step} steps "
                                    f"(KL < {kl_tol} for {kl_n_iters} consecutive batches)")
                            break
                    else:
                        kl_count = 0

                prev_F = self.fisher_matrix_u
                prev_best = self.fisher_peak_u

    # ------------------------------------------------------------------
    # Phase III: Fisher extraction
    # ------------------------------------------------------------------

    def estimate_fisher_matrix_u(self) -> np.ndarray:
        """
        Compute the FIM as the negative Hessian of the GP mean in unit-cube space.

        Returns
        -------
        F_u : ndarray, shape (r, r)
            Fisher Information Matrix in unit-cube space.
        """
        idx_best = int(jnp.argmax(self.gp.train_y))
        best_u = jnp.array(self.gp.train_x[idx_best].flatten())

        def gp_mean_unit(u):
            """GP mean as a function of unit-cube parameters u."""
            return self.gp.predict_mean_single(u)

        hessian_func = jax.hessian(gp_mean_unit)
        hessian = hessian_func(best_u)   # (r, r)
        return np.array(-hessian)

    def estimate_fisher_matrix(self) -> np.ndarray:
        """
        Compute the FIM as the negative Hessian of the GP mean in physical space.

        Returns
        -------
        F_phys : ndarray, shape (ndim, ndim)
            Fisher Information Matrix in physical parameter space.
        """
 
        fisher_u = self.estimate_fisher_matrix_u()  # FIM in unit-cube space

        # transfrom to physical space via Jacobian

        bounds = self.transform._theta_range

        fisher_phys = fisher_u / np.outer(bounds, bounds)  # F_phys = J^T F_u J, where J = diag(bounds)

        return np.array(fisher_phys)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_unit_jax(self, theta):
        """
        JAX-compatible physical θ → unit-cube u transform.

        Mirrors :meth:`ParameterTransform.to_unit` but uses jnp instead of np
        so that jax.hessian can differentiate through it.
        """
        t = self.transform
        if not t._use_rotation:
            return (theta - jnp.array(t._theta_min)) / jnp.array(t._theta_range)
        else:
            z = (theta - jnp.array(t._theta_center)) @ jnp.array(t._V_r)
            return (z - jnp.array(t._z_min)) / jnp.array(t._z_range)

    def _get_local_bounds(self, alpha: float):
        """
        V_peak bounds in unit-cube space centred on the current best point.

        Width in dimension i is  alpha × ℓ_i  (GP lengthscale), clipped to [0,1].

        Returns
        -------
        best_u : ndarray, shape (r,)
        (lo, hi) : tuple of ndarray, shape (r,)
        """
        idx_best = int(jnp.argmax(self.gp.train_y))
        best_u = np.array(self.gp.train_x[idx_best].flatten())
        half_w = alpha * np.array(self.gp.lengthscales)
        lo = np.clip(best_u - half_w, 0.0, 1.0)
        hi = np.clip(best_u + half_w, 0.0, 1.0)
        return best_u, (lo, hi)

    def _fim_kl(self, F_prev: np.ndarray, prev_best, F_curr: np.ndarray, curr_best) -> float:
        """
        Symmetric KL divergence between N(θ*, F_prev^{-1}) and N(θ*, F_curr^{-1}).

        D_KL = ½ [tr(F_t F_{t-1}^{-1}) - D + log(det F_{t-1} / det F_t)]
             + reverse term

        Uses :func:`kl_divergence_gaussian` from BOBE's utils.
        """
        eps = 1e-8 * np.eye(len(F_prev))
        try:
            C_prev = np.linalg.inv(F_prev + eps)
            C_curr = np.linalg.inv(F_curr + eps)
        except np.linalg.LinAlgError:
            return np.inf
        kl = kl_divergence_gaussian(prev_best, C_prev, curr_best, C_curr)
        return float(kl.get('symmetric', np.inf))