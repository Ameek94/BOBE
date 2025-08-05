from typing import Optional, Union, List, Tuple, Callable, Any
import jax
import jax.numpy as jnp
import numpy as np
from .utils import scale_to_unit, scale_from_unit, split_vmap 
import optax
from .logging_utils import get_logger
from .seed_utils import get_new_jax_key, get_global_seed

log = get_logger(__name__)

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
        return jnp.array([[0., 1.]] * ndim).T
    bounds = jnp.array(bounds)
    if bounds.shape == (2,):  # Same bounds for all dimensions
        bounds = jnp.tile(bounds.reshape(1, 2), (ndim, 1)).T
    elif bounds.shape != (2, ndim):
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
    verbose: bool = True,
    split_vmap_batch_size: int = 4,
    func_kwargs: dict = {},
) -> Tuple[jnp.ndarray, float]:
    """
    Standalone method to minimize a function using JAX and optax.
    # ... (docstring) ...
    rng_key: jax.random.PRNGKey, optional
             Random key for JAX randomness. If None, a default key is used (less ideal for reproducibility).
    # ... (rest of docstring) ...
    """


    bounds_arr = _setup_bounds(bounds, ndim)
    scaled_func = lambda x: func(scale_from_unit(x, bounds_arr), **func_kwargs)

    def func_val_grad(x):
        return jax.value_and_grad(scaled_func)(x)

    optimizer = _get_optimizer(optimizer_name, lr, optimizer_kwargs)

    @jax.jit
    def step(u_params, opt_state):
        val, grad = func_val_grad(u_params)
        updates, opt_state = optimizer.update(grad, opt_state)
        u_params = optax.apply_updates(u_params, updates)
        u_params = jnp.clip(u_params, 0., 1.) 
        return u_params, opt_state, val

    # Get initial points (can handle x0 of shapes (ndim) or (1,ndim) or (n_restarts, ndim))
    if x0 is not None:
        x0_array = jnp.atleast_2d(x0) # Handle 1D input
        if x0_array.shape[0] == 1:
            # if only one point provided, add n_restarts-1 random points
            init_params_unit = np.random.uniform(size=(n_restarts-1, ndim))
            init_params_unit = jnp.concatenate([scale_to_unit(x0_array, bounds_arr), init_params_unit], axis=0)
        elif x0_array.shape[0] == n_restarts:
             init_params_unit = scale_to_unit(x0_array, bounds_arr)
        else:
             # Mismatch, fallback to random
             log.warning(f"x0 provided with {x0_array.shape[0]} points, expected 1 or {n_restarts}. Using random initials.")
             init_params_unit = np.random.uniform(size=(n_restarts, ndim))
    else:
        # Generate n_restarts random initial points in [0, 1] space
        init_params_unit = np.random.uniform(size=(n_restarts, ndim))

    init_params_unit = jnp.array(init_params_unit) 

    # Initialize optimizer states
    opt_states = jax.vmap(optimizer.init)(init_params_unit)

    # Define the body function for the optimization loop
    def body_fun(i, carry):
        params, opt_states, best_f, best_params_unit = carry

        # This replaces: params_next, opt_states_next, values = jax.vmap(step)(params, opt_states)
        params_next, opt_states_next, values = split_vmap(
            step, 
            [params, opt_states], 
            batch_size=split_vmap_batch_size
        )

        # Find best among all parallel n_restarts at this step
        current_best_idx = jnp.argmin(values)
        current_best_val = values[current_best_idx]
        is_better = current_best_val < best_f

        # Update best found so far using jax.lax.cond for functional style
        new_best_f = jnp.where(is_better, current_best_val, best_f)
        new_best_params_unit = jax.lax.cond(
            is_better,
            lambda operands: operands[0][operands[1]], # params_next[current_best_idx]
            lambda operands: operands[2],              # best_params_unit
            (params_next, current_best_idx, best_params_unit)
        )
        
        # Cant have logging here, (maybe using host_callback?)
        
        return (params_next, opt_states_next, new_best_f, new_best_params_unit)

    # Initialize best values based on the first step results
    init_params_tmp, init_opt_states_tmp, init_values_tmp = split_vmap(
        step, (init_params_unit, opt_states), batch_size=split_vmap_batch_size
    )
    
    initial_best_idx = jnp.argmin(init_values_tmp)
    initial_best_f = init_values_tmp[initial_best_idx]        
    initial_best_params_unit = init_params_tmp[initial_best_idx]
    
    # Using lax.fori_loop
    final_params, final_opt_states, final_best_f, final_best_params_unit = jax.lax.fori_loop(
        1, 
        maxiter,
        body_fun,
        (init_params_tmp, init_opt_states_tmp, initial_best_f, initial_best_params_unit)
    )

    # Convert best parameters back to original space
    best_params_original = scale_from_unit(final_best_params_unit, bounds_arr)
    best_f_original = final_best_f

    # Final logging
    if verbose:
        desc = f'Completed {maxiter} steps ({optimizer_name}) with {n_restarts} restarts'
        display_val = float(final_best_f)
        log.info(f"{desc}: Final best_f = {display_val}")

    return best_params_original, best_f_original
