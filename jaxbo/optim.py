from typing import Optional, Union, List, Tuple, Callable, Any
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
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

def _setup_initial_points(
    x0: Optional[jnp.ndarray], 
    n_restarts: int, 
    ndim: int, 
) -> jnp.ndarray:
    """
    Generates initial points for optimization restarts in the unit cube.

    Parameters:
    -----------
    x0 : array-like, optional
        Initial guess(es), assumed to be in the unit cube.
    n_restarts : int
        Number of restarts.
    ndim : int
        Number of dimensions.

    Returns:
    --------
    jnp.ndarray
        An array of initial points in the unit cube.
    """
    if x0 is None:
        # Generate all points randomly in the unit cube
        x0_arr = np.random.uniform(low=0.0, high=1.0, size=(n_restarts, ndim))
    else:
        # Use provided points and add more if needed
        x0_arr = jnp.atleast_2d(x0)
        n_x0 = x0_arr.shape[0]
        
        if n_x0 < n_restarts:
            needed = n_restarts - n_x0
            new_points = np.random.uniform(low=0.0, high=1.0, size=(needed, ndim))
            x0_arr = jnp.concatenate([x0_arr, new_points], axis=0)
        elif n_x0 > n_restarts:
            x0_arr = x0_arr[:n_restarts]
            
    return jnp.array(x0_arr)

def optimize_optax(
    fun: Callable,
    fun_args: Optional[Tuple] = (),
    fun_kwargs: Optional[dict] = {},
    ndim: int = 1,
    bounds: Optional[Union[List, Tuple, jnp.ndarray]] = None,
    x0: Optional[jnp.ndarray] = None,
    optimizer_kwargs: Optional[dict] = {"name": "adam", "lr": 1e-3, "early_stop_patience": 25},
    maxiter: int = 200,
    n_restarts: int = 4,
    verbose: bool = False,
) -> Tuple[jnp.ndarray, float]:
    """
    Standalone method to minimize a function using JAX and optax.
    Runs multiple restarts sequentially with early stopping per restart.
    """
    bounds_arr = _setup_bounds(bounds, ndim)
    
    # Scaled function: operates in unit space [0,1], then maps to real bounds
    def scaled_func(x):
        return fun(scale_from_unit(x, bounds_arr), *fun_args, **fun_kwargs)
    # scaled_func = lambda x: func(scale_from_unit(x, bounds_arr), *fun_args, **fun_kwargs)

    # Get optimizer
    lr = optimizer_kwargs.get("lr", 1e-3)
    optimizer_name = optimizer_kwargs.get("name", "adam")
    optimizer = _get_optimizer(optimizer_name, lr, optimizer_kwargs)
    early_stop_patience = optimizer_kwargs.get("early_stop_patience", 25)

    # JIT the step function for performance
    @jax.jit
    def step(u_params, opt_state):
        val, grad = jax.value_and_grad(scaled_func)(u_params)
        updates, opt_state = optimizer.update(grad, opt_state)
        u_params = optax.apply_updates(u_params, updates)
        u_params = jnp.clip(u_params, 0.0, 1.0)  # Stay in unit cube
        return u_params, opt_state, val
    
    init_params_unit = _setup_initial_points(
        x0, n_restarts, ndim,)
    
    # Global best across all restarts
    global_best_f = np.inf
    global_best_params_unit = None
    
    # Run each restart independently with its own early stopping
    for restart_idx in range(n_restarts):
        log.debug(f"Starting restart {restart_idx + 1}/{n_restarts}")
            
        # Initialize this restart
        current_params = init_params_unit[restart_idx]
        opt_state = optimizer.init(current_params)
        
        # Initialize early stopping for this restart
        best_f_for_restart = float('inf')
        patience_counter = early_stop_patience
        
        # Initial evaluation
        current_params, opt_state, current_value = step(current_params, opt_state)
        
        if current_value < best_f_for_restart:
            best_f_for_restart = current_value
            
        # Local optimization loop for this restart
        restart_progress = tqdm.tqdm(range(maxiter), desc=f'Restart {restart_idx + 1}', leave=False)
        
        for iter_idx in restart_progress:
            # Take optimization step
            current_params, opt_state, current_value = step(current_params, opt_state)
            
            # Check for improvement in this restart
            if current_value < best_f_for_restart:
                best_f_for_restart = current_value
                patience_counter = early_stop_patience  # Reset patience
            else:
                patience_counter -= 1
                if patience_counter == 0:
                    if verbose:
                        log.debug(f"Early stopping for restart {restart_idx + 1} at iteration {iter_idx}")
                    break
            
            restart_progress.set_postfix({"best_value": float(best_f_for_restart)})
        
        # Update global best if this restart found a better solution
        if best_f_for_restart < global_best_f:
            global_best_f = best_f_for_restart
            global_best_params_unit = current_params
            
        if verbose:
            log.debug(f"Restart {restart_idx + 1} completed. Best value: {float(best_f_for_restart):.4e}")
    
    # Final best in original space
    best_params_original = scale_from_unit(global_best_params_unit, bounds_arr)
    best_f_original = global_best_f

    if verbose:
        desc = f'Completed optimization with {n_restarts} restarts ({optimizer_name})'
        log.info(f"{desc}: Final best_f = {float(best_f_original):.4e}")

    return best_params_original, float(best_f_original)


def optimize_scipy(
    fun: Callable,
    fun_args: Optional[Tuple] = (),
    fun_kwargs: Optional[dict] = {},
    ndim: int = 1,
    bounds: Optional[Union[List, Tuple, jnp.ndarray]] = None,
    x0: Optional[jnp.ndarray] = None,
    optimizer_kwargs: Optional[dict] = {"method": "L-BFGS-B", "ftol": 1e-6, "gtol": 1e-6},
    maxiter: int = 200,
    n_restarts: int = 4,
    verbose: bool = False,
) -> Tuple[jnp.ndarray, float]:
    """
    Standalone method to minimize a function using scipy.optimize.minimize.
    
    Parameters:
    -----------
    fun : Callable
        The objective function to minimize
    fun_args : tuple, optional
        Additional arguments to pass to the function
    fun_kwargs : dict, optional
        Additional keyword arguments to pass to the function
    ndim : int, default=1
        Number of dimensions
    bounds : array-like, optional
        Bounds for optimization. If None, uses [0, 1] for all dimensions
    x0 : array-like, optional
        Initial guess(es) in unit cube. If None, random points are generated
    method : str, default="L-BFGS-B"
        Scipy optimization method
    maxiter : int, default=200
        Maximum number of iterations
    n_restarts : int, default=4
        Number of random restarts
    verbose : bool, default=False
        Whether to print verbose output
    ftol : float, default=1e-6
        Function tolerance for convergence
    gtol : float, default=1e-6
        Gradient tolerance for convergence
        
    Returns:
    --------
    best_params : jnp.ndarray
        Best parameters found
    best_value : float
        Best function value found
    """
    bounds_arr = _setup_bounds(bounds, ndim)
    
    # Create scipy bounds format: list of (min, max) tuples
    scipy_bounds = [(bounds_arr[0, i], bounds_arr[1, i]) for i in range(ndim)]
    
    # JIT-compiled function that computes both value and gradient
    @jax.jit
    def value_and_grad_func(x):
        return jax.value_and_grad(fun)(x, *fun_args, **fun_kwargs)
    
    # Generate initial points
    x0_list = _setup_initial_points(
        x0, n_restarts, ndim, bounds_arr, in_unit_space=False
    )

    x0_list = scale_from_unit(x0_list, bounds_arr)  # Convert to numpy for scipy


    # Global best across all restarts
    global_best_f = np.inf
    global_best_params = None
    
    # Run optimization with multiple restarts
    for restart_idx in range(n_restarts):
        if verbose:
            log.debug(f"Starting scipy restart {restart_idx + 1}/{n_restarts}")
        
        try:
            result = minimize(
                value_and_grad_func,
                x0_list[restart_idx],
                method=optimizer_kwargs.get("method", "L-BFGS-B"),
                jac=True,
                bounds=scipy_bounds,
                options={
                    'maxiter': maxiter,
                    'ftol': optimizer_kwargs.get("ftol", 1e-6),
                    'gtol': optimizer_kwargs.get("gtol", 1e-6),
                }
            )
            
            if result.success and result.fun < global_best_f:
                global_best_f = result.fun
                global_best_params = result.x
                
            if verbose:
                status = "success" if result.success else "failed"
                log.debug(f"Restart {restart_idx + 1} {status}. Value: {result.fun:.4e}")
                
        except Exception as e:
            if verbose:
                log.warning(f"Restart {restart_idx + 1} failed with error: {e}")
            continue
    
    if global_best_params is None:
        # Fallback to best initial point if all optimizations failed
        best_init_idx = 0
        best_init_val = np.inf
        for i, x_init in enumerate(x0_list):
            val, grad = value_and_grad_func(x_init)
            if val < best_init_val:
                best_init_val = val
                best_init_idx = i
        global_best_params = x0_list[best_init_idx]
        global_best_f = best_init_val
        if verbose:
            log.warning("All optimizations failed, returning best initial point")
    
    if verbose:
        log.info(f"Scipy optimization ({method}) completed with {n_restarts} restarts: "
                f"Final best_f = {float(global_best_f):.4e}")
    
    return jnp.array(global_best_params), float(global_best_f)


# # Main optimize function without JAX control flow
# def optimize(
#     func: Callable,
#     fun_args: Optional[Tuple] = (),
#     fun_kwargs: Optional[dict] = {},
#     ndim: int = 1,
#     bounds: Optional[Union[List, Tuple, jnp.ndarray]] = None,
#     x0: Optional[jnp.ndarray] = None,
#     optimizer_name: str = "adam",
#     lr: float = 1e-3,
#     optimizer_kwargs: Optional[dict] = {},
#     maxiter: int = 200,
#     n_restarts: int = 4,
#     verbose: bool = True,
#     early_stop_patience: int = 100,
#     split_vmap_batch_size: int = 4,
# ) -> Tuple[jnp.ndarray, float]:
#     """
#     Standalone method to minimize a function using JAX and optax.
#     Uses only Python for loops (no jax.lax control flow).
#     """
#     bounds_arr = _setup_bounds(bounds, ndim)
    
#     # Scaled function: operates in unit space [0,1], then maps to real bounds
#     scaled_func = lambda x: func(scale_from_unit(x, bounds_arr), *fun_args, **fun_kwargs)

#     # Get optimizer
#     optimizer = _get_optimizer(optimizer_name, lr, optimizer_kwargs)

#     # JIT the step function for performance (still allowed, not control flow)
#     @jax.jit
#     def step(u_params, opt_state):
#         val, grad = jax.value_and_grad(scaled_func)(u_params)
#         updates, opt_state = optimizer.update(grad, opt_state)
#         u_params = optax.apply_updates(u_params, updates)
#         u_params = jnp.clip(u_params, 0.0, 1.0)  # Stay in unit cube
#         return u_params, opt_state, val
    
#     if x0 is None:
#         x0 = np.random.uniform(size=(n_restarts, ndim))
#     else:
#         x0 = jnp.atleast_2d(x0)  # Ensure x0 is at least 2D
#         n_x0 = x0.shape[0]
#         if n_x0 < n_restarts:
#             needed_x0 = n_restarts - n_x0
#             added_x0 = np.random.uniform(size=(needed_x0, ndim))
#             x0 = jnp.concatenate([x0, added_x0], axis=0)

#     init_params_unit = jnp.array(x0)

    
#     opt_states = [optimizer.init(init_params_unit[i]) for i in range(n_restarts)]

#     # Evaluate initial step to set best_f and best_params
#     params_list = []
#     values_list = []
#     new_opt_states = []

#     for i in range(n_restarts):
#         p, os, v = step(init_params_unit[i], opt_states[i])
#         params_list.append(p)
#         new_opt_states.append(os)
#         values_list.append(v)

#     current_params = params_list  # List of arrays of shape (ndim,)
#     current_opt_states = new_opt_states
#     values = jnp.array(values_list)

#     # Initialize best
#     best_idx = jnp.argmin(values)
#     best_f = values[best_idx]
#     best_params_unit = current_params[best_idx]

#     # --- Optimization loop over maxiter ---
#     r = np.arange(maxiter)
#     progress_bar = tqdm.tqdm(r,desc=f'Optimizing')

#     for iter_idx in progress_bar:
#         next_params = []
#         next_opt_states = []
#         step_values = []

#         # Step each restart independently
#         for i in range(n_restarts):
#             p_next, os_next, v_next = step(current_params[i], current_opt_states[i])
#             next_params.append(p_next)
#             next_opt_states.append(os_next)
#             step_values.append(v_next)

#         step_values = jnp.array(step_values)
#         current_best_idx = jnp.argmin(step_values)
#         current_best_val = step_values[current_best_idx]

#         # Update global best if improved
#         if current_best_val < best_f:
#             best_f = current_best_val
#             best_params_unit = next_params[current_best_idx]
#         else:
#             early_stop_patience -= 1
#             if early_stop_patience == 0:
#                 log.info(f"Early stopping at iteration {iter_idx}")
#                 break

#         progress_bar.set_postfix({"best_value": float(best_f)})

#         # Update current params and states
#         current_params = next_params
#         current_opt_states = next_opt_states

#         # Optional verbose logging during loop
#         # if verbose and (iter_idx % 50 == 0 or iter_idx == maxiter - 1):
#         #     log.info(f"Step {iter_idx}: current best_f = {float(best_f):.6f}")

#     # Final best in original space
#     best_params_original = scale_from_unit(best_params_unit, bounds_arr)
#     best_f_original = best_f

#     if verbose:
#         desc = f'Completed {maxiter} steps ({optimizer_name}) with {n_restarts} restarts'
#         log.info(f"{desc}: Final best_f = {float(best_f_original):.6f}")

#     return best_params_original, float(best_f_original)


    # # --- Handle initial points ---
    # if x0 is not None:
    #     x0_array = jnp.atleast_2d(x0)
    #     if x0_array.shape[0] == 1:
    #         # One point given: use it + random others
    #         init_params_unit = np.random.uniform(size=(n_restarts - 1, ndim))
    #         x0_unit = scale_to_unit(x0_array[0], bounds_arr)
    #         init_params_unit = np.concatenate([x0_unit[None, :], init_params_unit], axis=0)
    #     elif x0_array.shape[0] == n_restarts:
    #         init_params_unit = scale_to_unit(x0_array, bounds_arr)
    #     else:
    #         log.warning(f"x0 provided with {x0_array.shape[0]} points, expected 1 or {n_restarts}. Using random initials.")
    #         init_params_unit = np.random.uniform(size=(n_restarts, ndim))
    # else:
    #     init_params_unit = np.random.uniform(size=(n_restarts, ndim))

    # init_params_unit = jnp.array(init_params_unit)

    # Initialize optimizer states per restart


# Jax control flow version
# def optimize(
#     func: Callable,
#     func_args: Optional[Tuple] = (),
#     func_kwargs: Optional[dict] = {},
#     ndim: int = 1,
#     bounds: Optional[Union[List, Tuple, jnp.ndarray]] = None,
#     x0: Optional[jnp.ndarray] = None,
#     optimizer_name: str = "adam",
#     lr: float = 1e-3,
#     optimizer_kwargs: Optional[dict] = {},
#     maxiter: int = 200,
#     n_restarts: int = 4,
#     verbose: bool = True,
#     split_vmap_batch_size: int = 4,
# ) -> Tuple[jnp.ndarray, float]:
#     """
#     Standalone method to minimize a function using JAX and optax.
#     # ... (docstring) ...
#     rng_key: jax.random.PRNGKey, optional
#              Random key for JAX randomness. If None, a default key is used (less ideal for reproducibility).
#     # ... (rest of docstring) ...
#     """


#     bounds_arr = _setup_bounds(bounds, ndim)
#     scaled_func = lambda x: func(scale_from_unit(x, bounds_arr), *func_args, **func_kwargs)

#     optimizer = _get_optimizer(optimizer_name, lr, optimizer_kwargs)

#     @jax.jit
#     def step(u_params, opt_state):
#         val, grad = jax.value_and_grad(scaled_func)(u_params)
#         updates, opt_state = optimizer.update(grad, opt_state)
#         u_params = optax.apply_updates(u_params, updates)
#         u_params = jnp.clip(u_params, 0., 1.) 
#         return u_params, opt_state, val

#     # Get initial points (can handle x0 of shapes (ndim) or (1,ndim) or (n_restarts, ndim))
#     if x0 is not None:
#         x0_array = jnp.atleast_2d(x0) # Handle 1D input
#         if x0_array.shape[0] == 1:
#             # if only one point provided, add n_restarts-1 random points
#             init_params_unit = np.random.uniform(size=(n_restarts-1, ndim))
#             init_params_unit = jnp.concatenate([scale_to_unit(x0_array, bounds_arr), init_params_unit], axis=0)
#         elif x0_array.shape[0] == n_restarts:
#              init_params_unit = scale_to_unit(x0_array, bounds_arr)
#         else:
#              # Mismatch, fallback to random
#              log.warning(f"x0 provided with {x0_array.shape[0]} points, expected 1 or {n_restarts}. Using random initials.")
#              init_params_unit = np.random.uniform(size=(n_restarts, ndim))
#     else:
#         # Generate n_restarts random initial points in [0, 1] space
#         init_params_unit = np.random.uniform(size=(n_restarts, ndim))

#     init_params_unit = jnp.array(init_params_unit) 

#     # Initialize optimizer states
#     opt_states = jax.vmap(optimizer.init)(init_params_unit)

#     # Define the body function for the optimization loop
#     def body_fun(i, carry):
#         params, opt_states, best_f, best_params_unit = carry

#         # This replaces: params_next, opt_states_next, values = jax.vmap(step)(params, opt_states)
#         params_next, opt_states_next, values = split_vmap(
#             step, 
#             [params, opt_states], 
#             batch_size=split_vmap_batch_size
#         )

#         # Find best among all parallel n_restarts at this step
#         current_best_idx = jnp.argmin(values)
#         current_best_val = values[current_best_idx]
#         is_better = current_best_val < best_f

#         # Update best found so far using jax.lax.cond for functional style
#         new_best_f = jnp.where(is_better, current_best_val, best_f)
#         new_best_params_unit = jax.lax.cond(
#             is_better,
#             lambda operands: operands[0][operands[1]], # params_next[current_best_idx]
#             lambda operands: operands[2],              # best_params_unit
#             (params_next, current_best_idx, best_params_unit)
#         )
        
#         # Cant have logging here, (maybe using host_callback?)
        
#         return (params_next, opt_states_next, new_best_f, new_best_params_unit)

#     # Initialize best values based on the first step results
#     init_params_tmp, init_opt_states_tmp, init_values_tmp = split_vmap(
#         step, (init_params_unit, opt_states), batch_size=split_vmap_batch_size
#     )
    
#     initial_best_idx = jnp.argmin(init_values_tmp)
#     initial_best_f = init_values_tmp[initial_best_idx]        
#     initial_best_params_unit = init_params_tmp[initial_best_idx]
    
#     # Using lax.fori_loop
#     final_params, final_opt_states, final_best_f, final_best_params_unit = jax.lax.fori_loop(
#         1, 
#         maxiter,
#         body_fun,
#         (init_params_tmp, init_opt_states_tmp, initial_best_f, initial_best_params_unit)
#     )

#     # Convert best parameters back to original space
#     best_params_original = scale_from_unit(final_best_params_unit, bounds_arr)
#     best_f_original = final_best_f

#     # Final logging
#     if verbose:
#         desc = f'Completed {maxiter} steps ({optimizer_name}) with {n_restarts} restarts'
#         display_val = float(final_best_f)
#         log.info(f"{desc}: Final best_f = {display_val}")

#     return best_params_original, best_f_original
