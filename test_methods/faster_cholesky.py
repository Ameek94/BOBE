import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp

# Enable 64‑bit for better numerical stability
jax.config.update("jax_enable_x64", True)

# ---- 1) RBF kernel ----
def rbf_kernel(X, Y, lengthscale=1.0, variance=1.0):
    """
    Compute RBF kernel matrix between X (n×d) and Y (m×d) → (n×m).
    """
    sqdist = jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    return variance * jnp.exp(-0.5 * sqdist / lengthscale**2)

# ---- 2) Full Cholesky recompute ----
@jax.jit
def full_cholesky(K):
    return jnp.linalg.cholesky(K)

# ---- 3) Fast “grow” update via block‐Cholesky ----
@jax.jit
def cholesky_append(L, k, k_self):
    """
    Given L @ L^T = K_n, k = K_n[:, new], and k_self = K(new,new),
    returns the (n+1)x(n+1) Cholesky factor of:
        [K_n   k]
        [k^T k_self].
    """
    # solve L v = k  →  v has shape (n,)
    v = jsp.solve_triangular(L, k, lower=True)

    # new diagonal entry
    diag = jnp.sqrt(k_self - jnp.dot(v, v))

    # build a zero (n+1)x(n+1) and fill blocks
    n = L.shape[0]
    L_big = jnp.zeros((n+1, n+1), dtype=L.dtype)
    L_big = L_big.at[:n, :n].set(L)      # top-left
    L_big = L_big.at[n, :n].set(v)       # bottom-left
    L_big = L_big.at[n, n].set(diag)     # bottom-right
    return L_big

# ---- 4) Benchmark function ----
def benchmark(n=1000, iters=20, dim=20):
    rng = np.random.default_rng(0)

    # 4.1 Generate initial data and its Cholesky
    X = rng.standard_normal((n, dim))
    K = rbf_kernel(X, X)
    L = jnp.linalg.cholesky(K)

    # 4.2 One new point → cross‐covariances
    new_x = rng.standard_normal((1, dim))
    k = rbf_kernel(X, new_x).flatten()           # shape (n,)
    k_self = float(rbf_kernel(new_x, new_x)[0, 0])  # scalar

    # 4.3 JIT warm‑up
    _ = cholesky_append(L, k, k_self).block_until_ready()
    K_big = jnp.block([
        [K,           k[:, None]],
        [k[None, :],  jnp.array([[k_self]])]
    ])
    _ = full_cholesky(K_big).block_until_ready()

    # 4.4 Time the *fast* append (always feeding the same L,k)
    t0 = time.time()
    for _ in range(iters):
        _ = cholesky_append(L, k, k_self).block_until_ready()
    t_grow = (time.time() - t0) / iters

    # 4.5 Time the *full* recompute
    t0 = time.time()
    for _ in range(iters):
        K_big = jnp.block([
            [K,           k[:, None]],
            [k[None, :],  jnp.array([[k_self]])]
        ])
        _ = full_cholesky(K_big).block_until_ready()
    t_full = (time.time() - t0) / iters

    # 4.6 Report
    print(f"Average per-step over {iters} iterations:")
    print(f" • Fast append (n→n+1):      {t_grow*1e3:.3f} ms")
    print(f" • Full cholesky (n+1)×(n+1): {t_full*1e3:.3f} ms")

    # Check equal results
    L_grow = cholesky_append(L, k, k_self)
    L_full = full_cholesky(K_big)
    assert jnp.allclose(L_grow, L_full), "Results differ between fast append and full recompute!"


if __name__ == "__main__":
    benchmark()
