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

def _setup_bounds(bounds: Optional[Union[List, Tuple, jnp.ndarray]], num_params: int) -> jnp.ndarray:
    """Setup parameter bounds."""
    if bounds is None:
        return jnp.array([[0., 1.]] * num_params).T
    bounds = jnp.array(bounds)
    if bounds.shape == (2,):  # Same bounds for all dimensions
        bounds = jnp.tile(bounds.reshape(1, 2), (num_params, 1)).T
    elif bounds.shape != (2, num_params):
        raise ValueError(f"Bounds shape {bounds.shape} incompatible with {num_params} dimensions")
    return bounds

def _setup_initial_points(
    x0: Optional[jnp.ndarray], 
    n_restarts: int, 
    num_params: int, 
) -> jnp.ndarray:
    """
    Generates initial points for optimization restarts in the unit cube.

    Parameters:
    -----------
    x0 : array-like, optional
        Initial guess(es), assumed to be in the unit cube.
    n_restarts : int
        Number of restarts.
    num_params : int
        Number of dimensions.

    Returns:
    --------
    jnp.ndarray
        An array of initial points in the unit cube.
    """
    if x0 is None:
        # Generate all points randomly in the unit cube
        x0_arr = np.random.uniform(low=0.0, high=1.0, size=(n_restarts, num_params))
    else:
        # Use provided points and add more if needed
        x0_arr = jnp.atleast_2d(x0)
        n_x0 = x0_arr.shape[0]
        
        if n_x0 < n_restarts:
            needed = n_restarts - n_x0
            new_points = np.random.uniform(low=0.0, high=1.0, size=(needed, num_params))
            x0_arr = jnp.concatenate([x0_arr, new_points], axis=0)
        elif n_x0 > n_restarts:
            x0_arr = x0_arr[:n_restarts]
            
    return jnp.array(x0_arr)

def optimize_optax(
    fun: Callable,
    fun_args: Optional[Tuple] = (),
    fun_kwargs: Optional[dict] = {},
    num_params: int = 1,
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

    Arguments
    ---------
    fun: Callable
        The objective function to minimize.
    fun_args: Optional[Tuple]
        Positional arguments to pass to the objective function.
    fun_kwargs: Optional[dict]
        Keyword arguments to pass to the objective function.
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
    bounds_arr = _setup_bounds(bounds, num_params)
    
    # Scaled function: operates in unit space [0,1], then maps to real bounds
    def scaled_func(x):
        return fun(scale_from_unit(x, bounds_arr), *fun_args, **fun_kwargs)
    # scaled_func = lambda x: func(scale_from_unit(x, bounds_arr), *fun_args, **fun_kwargs)

    # Get optimizer
    early_stop_patience = optimizer_kwargs.pop("early_stop_patience", 25)
    lr = optimizer_kwargs.pop("lr", 1e-3)
    optimizer_name = optimizer_kwargs.pop("name", "adam")
    optimizer = _get_optimizer(optimizer_name, lr, optimizer_kwargs)

    # JIT the step function for performance
    @jax.jit
    def step(u_params, opt_state):
        val, grad = jax.value_and_grad(scaled_func)(u_params)
        updates, opt_state = optimizer.update(grad, opt_state)
        u_params = optax.apply_updates(u_params, updates)
        u_params = jnp.clip(u_params, 0.0, 1.0)  # Stay in unit cube
        return u_params, opt_state, val
    
    init_params_unit = _setup_initial_points(
        x0, n_restarts, num_params)

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
    num_params: int = 1,
    bounds: Optional[Union[List, Tuple, jnp.ndarray]] = None,
    x0: Optional[jnp.ndarray] = None,
    optimizer_kwargs: Optional[dict] = {"method": "L-BFGS-B", "ftol": 1e-6, "gtol": 1e-6},
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

    method = optimizer_kwargs.get("method", "L-BFGS-B")

    bounds_arr = _setup_bounds(bounds, num_params)
    
    # Create scipy bounds format: list of (min, max) tuples
    scipy_bounds = [(float(bounds_arr[0, i]), float(bounds_arr[1, i])) for i in range(num_params)]

    # JIT-compiled function that computes both value and gradient
    @jax.jit
    def value_and_grad_func(x):
        return jax.value_and_grad(fun)(x, *fun_args, **fun_kwargs)
    
    # Generate initial points
    x0_list = _setup_initial_points(
        x0, n_restarts, num_params,)

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
                method=method,
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