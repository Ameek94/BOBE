from typing import Optional, Union, List, Tuple, Callable, Any
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import minimize 
from .utils.core_utils import scale_to_unit, scale_from_unit, split_vmap 
from .utils.logging_utils import get_logger
from .utils.seed_utils import get_new_jax_key, get_global_seed

log = get_logger("optim")

def _get_optimizer(optimizer_name: str, learning_rate: float = 1e-3, optimizer_kwargs: Optional[dict] = {}) -> optax.GradientTransformation:
    """Get the optax optimizer."""
    optimizer_name = optimizer_name.lower()
    if 'learning_rate' not in optimizer_kwargs:
        optimizer_kwargs['learning_rate'] = learning_rate

    if optimizer_name == "adam":
        return optax.adam(**optimizer_kwargs)
    elif optimizer_name == "sgd":
        return optax.sgd(**optimizer_kwargs)
    elif optimizer_name == "lbfgs":
        return optax.lbfgs(**optimizer_kwargs)
    else:
        try:
            optimizer_fn = getattr(optax, optimizer_name)
            return optimizer_fn(**optimizer_kwargs)
        except AttributeError:
            raise ValueError(f"Optimizer '{optimizer_name}' not found in optax library")

def _setup_bounds(bounds: Optional[Union[List, Tuple, jnp.ndarray]], num_params: int) -> Optional[jnp.ndarray]:
    """
    Setup parameter bounds.

    Parameters
    ----------
    bounds : None, list, tuple, or jnp.ndarray
        If None, returns None (unbounded optimization).
        If array-like, must be shape (2,) for shared bounds or (2, num_params).
    num_params : int
        Number of parameters.

    Returns
    -------
    Optional[jnp.ndarray]
        Bounds array with shape (2, num_params), or None if unbounded.
    """
    if bounds is None:
        return None

    bounds = jnp.array(bounds)

    if bounds.shape == (2,):  # same bounds for all dimensions
        bounds = jnp.tile(bounds.reshape(1, 2), (num_params, 1)).T
    elif bounds.shape != (2, num_params):
        raise ValueError(f"Bounds shape {bounds.shape} incompatible with {num_params} parameters")

    return bounds

def optimize_optax(
    fun: Callable,
    fun_args: Optional[Tuple] = (),
    fun_kwargs: Optional[dict] = {},
    num_params: int = 1,
    bounds: Optional[Union[List, Tuple, jnp.ndarray]] = None,
    x0: jnp.ndarray = None,
    optimizer_options: Optional[dict] = {"name": "adam", "lr": 1e-3, "early_stop_patience": 25},
    maxiter: int = 200,
    n_restarts: int = 1,
    verbose: bool = False,
    ) -> Tuple[jnp.ndarray, float]:
    """
    SEQUENTIAL OPTIMIZER: Minimize a function using JAX + optax,
    iterating through restarts with a Python for-loop.
    """


    if x0 is None:
        raise ValueError("x0 must be provided (shape: (n_restarts, num_params))")

    x0 = jnp.atleast_2d(x0)
    if x0.shape[0] < n_restarts:
        raise ValueError(f"x0 provided with {x0.shape[0]} restarts but n_restarts={n_restarts}")
    x0 = x0[:n_restarts]

    bounds_arr = _setup_bounds(bounds, num_params)
    if bounds_arr is not None:
        scaled_func = lambda x: fun(scale_from_unit(x, bounds_arr), *fun_args, **fun_kwargs) 
    else:
        scaled_func = lambda x: fun(x, *fun_args, **fun_kwargs)

    early_stop_patience = optimizer_options.pop("early_stop_patience", 25)
    lr = optimizer_options.pop("lr", 1e-3)
    optimizer_name = optimizer_options.pop("name", "adam")
    optimizer = _get_optimizer(optimizer_name, lr, optimizer_options)

    log.info(f"Starting optax optimization with {optimizer_name}, restarts: {n_restarts}, maxiter: {maxiter}")

    @jax.jit
    def step(params, opt_state):
        val, grad = jax.value_and_grad(scaled_func)(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        if bounds_arr is not None:
            params = jnp.clip(params, 0.0, 1.0)
        return params, opt_state, val

    global_best_f = np.inf
    global_best_params = None

    for i, x_init in enumerate(x0):
        try:
            val = scaled_func(x_init)
            if np.isfinite(val) and val < global_best_f:
                global_best_f = val
                global_best_params = x_init
                log.debug(f"  Initial point {i+1}/{n_restarts}: New best found -> {val:.4e}")
        except Exception as e:
            log.warning(f"  Initial point {i+1}/{n_restarts}: Failed with an error: {e}")

    for restart_idx in range(n_restarts):
        log.debug(f"Starting restart {restart_idx + 1}/{n_restarts}")
        current_params = x0[restart_idx]
        opt_state = optimizer.init(current_params)
        best_f_for_restart = float("inf")
        patience_counter = early_stop_patience

        for iter_idx in range(maxiter):
            current_params, opt_state, current_value = step(current_params, opt_state)
            if current_value < best_f_for_restart:
                best_f_for_restart = current_value
                patience_counter = early_stop_patience
            else:
                patience_counter -= 1
                if patience_counter == 0:
                    if verbose: log.debug(f"Early stopping at iter {iter_idx} (restart {restart_idx+1})")
                    break
            if verbose and iter_idx % 10 == 0: log.debug(f"Restart {restart_idx+1}, iter {iter_idx}, best={float(best_f_for_restart):.4e}")

        if best_f_for_restart < global_best_f:
            global_best_f = best_f_for_restart
            global_best_params = current_params
        if verbose: log.debug(f"Restart {restart_idx+1} done. Best={float(best_f_for_restart):.4e}")

    best_params_original = scale_from_unit(global_best_params, bounds_arr) if bounds_arr is not None else global_best_params
    return jnp.array(best_params_original), float(global_best_f)


def optimize_optax_vmap(
    fun: Callable,
    fun_args: Optional[Tuple] = (),
    fun_kwargs: Optional[dict] = {},
    num_params: int = 1,
    bounds: Optional[Union[List, Tuple, jnp.ndarray]] = None,
    x0: jnp.ndarray = None,
    optimizer_options: Optional[dict] = {"name": "adam", "lr": 1e-3, "early_stop_patience": 25},
    maxiter: int = 200,
    n_restarts: int = 1,
    verbose: bool = False,
) -> Tuple[jnp.ndarray, float]:
    """
    VECTORIZED OPTIMIZER: Minimize a function using JAX + optax,
    vectorizing over restarts with jax.vmap for parallel execution. Only to be used with EI. 
    """


    if x0 is None:
        raise ValueError("x0 must be provided (shape: (n_restarts, num_params))")

    x0 = jnp.atleast_2d(x0)
    if x0.shape[0] != n_restarts:
        raise ValueError(f"x0 has {x0.shape[0]} restarts but n_restarts was set to {n_restarts}")

    bounds_arr = _setup_bounds(bounds, num_params)
    if bounds_arr is not None:
        scaled_func = lambda x: fun(scale_from_unit(x, bounds_arr), *fun_args, **fun_kwargs) 
    else:
        scaled_func = lambda x: fun(x, *fun_args, **fun_kwargs)

    early_stop_patience = optimizer_options.pop("early_stop_patience", 25)
    lr = optimizer_options.pop("lr", 1e-3)
    optimizer_name = optimizer_options.pop("name", "adam")
    optimizer = _get_optimizer(optimizer_name, lr, optimizer_options)

    log.info(f"Starting vectorized optax optimization with {optimizer_name}, restarts: {n_restarts}, maxiter: {maxiter}")

    def step(params, opt_state):
        val, grad = jax.value_and_grad(scaled_func)(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        if bounds_arr is not None:
            params = jnp.clip(params, 0.0, 1.0)
        return params, opt_state, val

    v_step = jax.jit(jax.vmap(step, in_axes=(0, 0), out_axes=(0, 0, 0)))

    params = x0
    opt_state = jax.vmap(optimizer.init)(params)
    best_vals = jnp.full(n_restarts, jnp.inf)
    best_params = jnp.zeros_like(params)
    patience_counters = jnp.full(n_restarts, early_stop_patience, dtype=jnp.int32)
    
    for iter_idx in range(maxiter):
        params, opt_state, current_vals = v_step(params, opt_state)
        improved_mask = current_vals < best_vals
        best_vals = jnp.where(improved_mask, current_vals, best_vals)
        best_params = jnp.where(improved_mask[:, None], params, best_params)
        patience_counters = jnp.where(improved_mask, early_stop_patience, patience_counters - 1)
        
        if jnp.all(patience_counters <= 0):
            if verbose: log.debug(f"All restarts stopped early at iteration {iter_idx}")
            break
        if verbose and iter_idx % 10 == 0: log.debug(f"Iter {iter_idx}, best overall={float(jnp.min(best_vals)):.4e}")

    best_restart_idx = jnp.argmin(best_vals)
    global_best_f = best_vals[best_restart_idx]
    global_best_params_scaled = best_params[best_restart_idx]

    if bounds_arr is not None:
        best_params_original = scale_from_unit(global_best_params_scaled, bounds_arr)
    else:
        best_params_original = global_best_params_scaled
    return jnp.array(best_params_original), float(global_best_f)


def optimize_scipy(
    fun: Callable,
    fun_args: Optional[Tuple] = (),
    fun_kwargs: Optional[dict] = {},
    num_params: int = 1,
    bounds: Optional[Union[List, Tuple, jnp.ndarray]] = None,
    x0: Optional[jnp.ndarray] = None,
    optimizer_options: Optional[dict] = {"method": "L-BFGS-B", "ftol": 1e-6, "gtol": 1e-6},
    maxiter: int = 200,
    n_restarts: int = 4,
    verbose: bool = False,
) -> Tuple[jnp.ndarray, float]:
    """
    Standalone method to minimize a function using scipy.optimize.minimize.

    Arguments
    ---------
    fun : Callable
        The objective function to minimize
    fun_args : tuple, optional
        Additional arguments to pass to the function
    fun_kwargs : dict, optional
        Additional keyword arguments to pass to the function
    num_params: int
        Number of parameters.
    bounds: Optional[Union[List, Tuple, jnp.ndarray]]
        Parameter bounds in shape (2, num_params).
    x0: Optional[jnp.ndarray]
        Initial guess for the parameters, rescaled to unit space.
    optimizer_kwargs: Optional[dict]
        Additional keyword arguments to pass to the optimizer.
    maxiter: int
        Maximum number of iterations.
    n_restarts: int
        Number of restarts for the optimization.
    verbose: bool
        Whether to print progress messages.
    Returns
    -------
    Tuple[jnp.ndarray, float]
        The best parameters found and the corresponding function value.
    """

    optimizer_options.update({'maxiter': maxiter})

    method = optimizer_options.pop("method", "L-BFGS-B")

    log.info(f"Starting scipy optimization with method: {method}, restarts: {n_restarts}, maxiter: {maxiter}")


    bounds_arr = _setup_bounds(bounds, num_params)
    log.debug(f"Function bounds: {bounds_arr}")
    # Create scipy bounds format: list of (min, max) tuples
    if bounds_arr is None:
        scipy_bounds = None
    else:   
        scipy_bounds = [(float(bounds_arr[0, i]), float(bounds_arr[1, i])) for i in range(num_params)]

    # JIT-compiled function that computes both value and gradient
    @jax.jit
    def value_and_grad_func(x):
        return jax.value_and_grad(fun)(x, *fun_args, **fun_kwargs)
    
    if x0 is None:
        raise ValueError("x0 must be provided (shape: (n_restarts, num_params) or (num_params,))")

    x0 = jnp.atleast_2d(x0)
    if x0.shape[0] < n_restarts:
        raise ValueError(f"x0 provided with {x0.shape[0]} restarts but n_restarts={n_restarts}")
    elif x0.shape[0] > n_restarts:
        x0 = x0[:n_restarts]

    # Global best across all restarts
    global_best_f = np.inf
    global_best_params = None

    # check values at initial points
    for i, x_init in enumerate(x0):
        try:
            val, _ = value_and_grad_func(x_init)
            if np.isfinite(val) and val < global_best_f:
                global_best_f = val
                global_best_params = x_init
                log.debug(f"  Initial point {i+1}/{n_restarts}: New best found -> {val:.4e}")
        except Exception as e:
            log.warning(f"  Initial point {i+1}/{n_restarts}: Failed with an error: {e}")

    for i, x_init in enumerate(x0):
        try:
            result = minimize(value_and_grad_func, x_init, method=method, jac=True, bounds=scipy_bounds, options=optimizer_options)

            # Check if the result is acceptable and an improvement
            is_acceptable = result.success or "ITERATIONS REACHED LIMIT" in result.message.upper()
            is_valid = np.isfinite(result.fun)

            if is_acceptable and is_valid and result.fun < global_best_f:
                global_best_f = result.fun
                global_best_params = result.x
                status = "success" if result.success else "max iterations"
                log.debug(f"  Restart {i+1}/{n_restarts}: New best found ({status}) -> {result.fun:.4e}")
            else:
                log.debug(f"  Restart {i+1}/{n_restarts}: Finished but not improved. Status: {result.message}")

        except Exception as e:
            if verbose:
                log.warning(f"  Restart {i+1}/{n_restarts}: Failed with an error: {e}")
            continue
    
    log.info(f"Scipy optimization ({method}) completed with {n_restarts} restarts: "
            f"Final best_f = {float(global_best_f):.4e}")
    
    return jnp.array(global_best_params), float(global_best_f)


    # if global_best_params is None:
    #     log.warning("No valid optimization results found, falling back to best initial point")
    #     log.warning("This indicates all optimizations had serious failures (not just max iterations)")
        
    #     # Fallback to best initial point if all optimizations failed
    #     best_init_idx = 0
    #     best_init_val = np.inf
    #     for i, x_init in enumerate(x0_list):
    #         try:
    #             val, _ = value_and_grad_func(x_init)
    #             if np.isfinite(val) and val < best_init_val:
    #                 best_init_val = val
    #                 best_init_idx = i
    #         except Exception:
    #             continue
        
    #     if not np.isfinite(best_init_val):
    #         log.error("All initial points have non-finite function values!")
        
    #     global_best_params = x0_list[best_init_idx]
    #     global_best_f = best_init_val
    #     log.warning(f"Using initial point {best_init_idx} with value {best_init_val:.6e}")

    # # Run optimization with multiple restarts
    # for restart_idx in range(n_restarts):
    #     initial_x = x0_list[restart_idx]
        
    #     try:
    #         # Evaluate function at initial point
    #         initial_val, initial_grad = value_and_grad_func(initial_x)
            
    #         # Check for NaN/inf in initial evaluation
    #         if not np.isfinite(initial_val) or not np.all(np.isfinite(initial_grad)):
    #             log.warning(f"Restart {restart_idx + 1}: Initial point has non-finite values, skipping")
    #             continue
            
    #         result = minimize(
    #             value_and_grad_func,
    #             initial_x,
    #             method=method,
    #             jac=True,
    #             bounds=scipy_bounds,
    #             options={
    #                 'maxiter': maxiter,
    #                 'ftol': optimizer_kwargs.get("ftol", 1e-6),
    #                 'gtol': optimizer_kwargs.get("gtol", 1e-6),
    #             }
    #         )
            
    #         # Check final function value
    #         if not np.isfinite(result.fun):
    #             log.warning(f"Restart {restart_idx + 1}: Final function value is not finite")
    #             continue  # Skip this restart if function value is not finite
            
    #         # Check if this is a good result even if not "successful"
    #         # L-BFGS-B can reach max iterations but still find good solutions
    #         is_max_iter = "TOTAL NO. OF ITERATIONS REACHED LIMIT" in result.message
    #         is_valid_result = np.isfinite(result.fun) and np.all(np.isfinite(result.x))
            
    #         if (result.success or is_max_iter) and is_valid_result and result.fun < global_best_f:
    #             global_best_f = result.fun
    #             global_best_params = result.x
    #             if result.success:
    #                 log.debug(f"New best found in restart {restart_idx + 1}: {result.fun:.6e}")
    #             else:
    #                 log.debug(f"New best found in restart {restart_idx + 1} (max iter): {result.fun:.6e}")
    #         elif not result.success and not is_max_iter:
    #             # Only treat as failure if it's not just hitting max iterations
    #             log.debug(f"Restart {restart_idx + 1} failed: {result.message}")
    #         elif is_max_iter:
    #             # Max iterations reached but not better than current best
    #             log.debug(f"Restart {restart_idx + 1} reached max iterations with value {result.fun:.6e} (not better than current best {global_best_f:.6e})")
                
    #     except Exception as e:
    #         log.warning(f"Restart {restart_idx + 1} failed with error: {e}")
    #         continue