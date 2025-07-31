from typing import Optional, Union, List, Tuple, Callable, Any
import jax
import jax.numpy as jnp
import numpy as np
from .utils import scale_to_unit, scale_from_unit, split_vmap 
import optax
from .logging_utils import get_logger
from .seed_utils import get_new_jax_key, get_global_seed

log = get_logger("[Opt]")

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


def _setup_bounds(bounds: Optional[Union[List, Tuple, jnp.ndarray]], ndim: int) -> jnp.ndarray:
    """Setup parameter bounds."""
    if bounds is None:
        return jnp.array([[0., 1.]] * ndim)
    bounds = jnp.array(bounds)
    if bounds.shape == (2,):  # Same bounds for all dimensions
        bounds = jnp.tile(bounds.reshape(1, 2), (ndim, 1))
    elif bounds.shape != (ndim, 2):
        raise ValueError(f"Bounds shape {bounds.shape} incompatible with {ndim} dimensions")
    return bounds

def optimize(
    func: Callable,
    ndim: int,
    bounds: Optional[Union[List, Tuple, jnp.ndarray]] = None,
    x0: Optional[jnp.ndarray] = None,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    optimizer_kwargs: Optional[dict] = {},
    maxiter: int = 200,
    n_restarts: int = 4,
    minimize: bool = True,
    verbose: bool = True,
    rng_key: Optional[jax.random.PRNGKey] = None, # Add rng_key parameter
    split_vmap_batch_size: int = 10,
    func_kwargs: dict = {},
) -> Tuple[jnp.ndarray, float]:
    """
    Standalone optimization function.
    # ... (docstring) ...
    rng_key: jax.random.PRNGKey, optional
             Random key for JAX randomness. If None, a default key is used (less ideal for reproducibility).
    # ... (rest of docstring) ...
    """
    # --- 1. Handle Random Key ---
    # Prioritize the passed rng_key
    if rng_key is not None:
        key = rng_key
    else:
        # Fallback to global seed management if no key provided
        try:
            key = get_new_jax_key()
        except RuntimeError:
            # No global seed set, use default
            key = jax.random.PRNGKey(0)
            log.warning("No rng_key provided and no global seed set for optimization, using default key")


    # --- 2. Setup and JIT Compile Core Functions ---
    bounds_arr = _setup_bounds(bounds, ndim)
    scaled_func = lambda x: func(scale_from_unit(x, bounds_arr), **func_kwargs)

    @jax.jit
    def val_grad(x):
        return jax.value_and_grad(scaled_func)(x)

    optimizer = _get_optimizer(optimizer_name, lr, optimizer_kwargs)

    @jax.jit
    def step(params, opt_state):
        (val, grad) = val_grad(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        # Clipping is typically done on the original scale, but here on [0,1]
        params = jnp.clip(params, 0., 1.) 
        return params, opt_state, val

    # --- 3. Generate Initial Points (No noise later) ---
    if x0 is not None:
        # If x0 is provided, tile it for all restarts
        x0_array = jnp.atleast_2d(x0) # Handle 1D input
        if x0_array.shape[0] == 1:
             # Single point provided, tile it
             init_params_unit = jnp.tile(scale_to_unit(x0_array, bounds_arr), (n_restarts, 1))
        elif x0_array.shape[0] == n_restarts:
             # Exactly n_restarts points provided
             init_params_unit = scale_to_unit(x0_array, bounds_arr)
        else:
             # Mismatch, fallback to random
             log.warning(f"x0 provided with {x0_array.shape[0]} points, expected 1 or {n_restarts}. Using random initials.")
             keys = jax.random.split(key, n_restarts)
             init_params_unit = jax.vmap(
                 lambda k: jax.random.uniform(k, shape=(ndim,), minval=0., maxval=1.)
             )(keys)
    else:
        # Generate n_restarts random initial points in [0, 1] space
        keys = jax.random.split(key, n_restarts)
        init_params_unit = jax.vmap(
            lambda k: jax.random.uniform(k, shape=(ndim,), minval=0., maxval=1.)
        )(keys)

    # --- 4. Initialize Optimizer States ---
    opt_states = jax.vmap(optimizer.init)(init_params_unit)

    # --- 5. Define the Body Function for lax.fori_loop ---
    def body_fun(i, carry):
        params, opt_states, best_f, best_params_unit = carry

        # --- Use split_vmap for the optimization step ---
        # This replaces: params_next, opt_states_next, values = jax.vmap(step)(params, opt_states)
        params_next, opt_states_next, values = split_vmap(
            step, 
            [params, opt_states], 
            batch_size=split_vmap_batch_size
        )
        # -----------------------------------------------

        # Find best among all parallel restarts at this step
        if minimize:
            current_best_idx = jnp.argmin(values)
            current_best_val = values[current_best_idx]
            is_better = current_best_val < best_f
        else:
            current_best_idx = jnp.argmax(values)
            current_best_val = values[current_best_idx]
            is_better = current_best_val > best_f

        # Update best found so far using jax.lax.cond for functional style
        new_best_f = jnp.where(is_better, current_best_val, best_f)
        new_best_params_unit = jax.lax.cond(
            is_better,
            lambda operands: operands[0][operands[1]], # params_next[current_best_idx]
            lambda operands: operands[2],              # best_params_unit
            (params_next, current_best_idx, best_params_unit)
        )
        
        # Logging inside the loop (can be tricky, consider moving outside or using host_callback)
        # For simplicity, we'll rely on the final logging or omit per-iteration logs if inside scan/loop
        # is problematic. The original verbose flag is handled after the loop in this structure.
        
        return (params_next, opt_states_next, new_best_f, new_best_params_unit)

    # --- 6. Set Initial Best Values ---
    # Initialize best values based on the first iteration's results
    # We need to run the step once to get initial values for comparison
    init_params_tmp, init_opt_states_tmp, init_values_tmp = split_vmap(
        step, [init_params_unit, opt_states], batch_size=split_vmap_batch_size
    )
    
    if minimize:
        initial_best_idx = jnp.argmin(init_values_tmp)
        initial_best_f = init_values_tmp[initial_best_idx]
    else:
        initial_best_idx = jnp.argmax(init_values_tmp)
        initial_best_f = init_values_tmp[initial_best_idx]
        
    initial_best_params_unit = init_params_tmp[initial_best_idx]
    
    # --- 7. Run the Optimization Loop using lax.fori_loop ---
    final_params, final_opt_states, final_best_f, final_best_params_unit = jax.lax.fori_loop(
        1, # Start from 1 as we already did iteration 0 above
        maxiter,
        body_fun,
        (init_params_tmp, init_opt_states_tmp, initial_best_f, initial_best_params_unit)
    )

    # --- 8. Final Processing and Return ---
    # Convert best parameters back to original space
    best_params_original = scale_from_unit(final_best_params_unit, bounds_arr)
    best_f_original = final_best_f if minimize else -final_best_f

    # --- 9. Final Logging (if verbose was True) ---
    # Since logging inside lax.fori_loop is tricky, we log the final result here if needed.
    # Note: Per-iteration logging from the original loop is lost with this structure.
    # If per-iteration logging is crucial, consider using host_callback or a different approach.
    if verbose:
        mode_str = "min" if minimize else "max"
        desc = f'Completed {maxiter} steps ({optimizer_name}, {mode_str})'
        display_val = float(final_best_f) if minimize else float(-final_best_f)
        log.info(f"{desc}: Final best_f = {display_val}")

    return best_params_original, best_f_original
