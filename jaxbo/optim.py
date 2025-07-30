from typing import Optional, Union, List, Tuple, Callable, Any
import jax
import jax.numpy as jnp
import numpy as np
from jaxbo.bo_utils import input_standardize, input_unstandardize
import logging
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import optax
from .logging_utils import get_logger

log = get_logger("[Opt]")


def _get_optimizer(optimizer_name: str, learning_rate: float = 1e-3, **optimizer_kwargs) -> optax.GradientTransformation:
    """
    Get the optax optimizer based on the specified name.

    Args:
        optimizer_name: Name of the optax optimizer ("adam", "sgd", "lbfgs")
        learning_rate: Learning rate for the optimizer
        **optimizer_kwargs: Additional arguments for the optimizer

    Returns:
        optax.GradientTransformation: The configured optimizer
    """
    optimizer_name = optimizer_name.lower()
    
    # Set default learning rate if not provided in kwargs
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
    """
    Setup parameter bounds, defaulting to [0, 1] if not provided.

    Args:
        bounds: Parameter bounds as [(low1, high1), (low2, high2), ...] or None
        ndim: Number of dimensions

    Returns:
        jnp.ndarray: Bounds array of shape (ndim, 2)
    """
    if bounds is None:
        return jnp.array([[0., 1.]] * ndim)

    bounds = jnp.array(bounds)
    if bounds.shape == (2,):  # Same bounds for all dimensions
        bounds = jnp.tile(bounds.reshape(1, 2), (ndim, 1))
    elif bounds.shape != (ndim, 2):
        raise ValueError(f"Bounds shape {bounds.shape} incompatible with {ndim} dimensions")

    return bounds


def _scale_to_unit(x: jnp.ndarray, bounds: jnp.ndarray) -> jnp.ndarray:
    """Scale parameters to unit bounds [0, 1]."""
    return input_standardize(x, bounds)


def _scale_from_unit(x: jnp.ndarray, bounds: jnp.ndarray) -> jnp.ndarray:
    """Scale parameters from unit bounds [0, 1] back to original bounds."""
    return input_unstandardize(x, bounds)


def optimize(
    func: Callable,
    ndim: int,
    bounds: Optional[Union[List, Tuple, jnp.ndarray]] = None,
    x0: Optional[jnp.ndarray] = None,
    lr: float = 1e-3,
    maxiter: int = 200,
    n_restarts: int = 4,
    minimize: bool = True,
    verbose: bool = True,
    random_seed: Optional[int] = None,
    optimizer_name: str = "adam",
    **func_kwargs
) -> Tuple[jnp.ndarray, float]:
    """
    Standalone optimization function for arbitrary functions supporting multiple optax optimizers.
    
    Supports parameter bounds with automatic rescaling to [0,1] during optimization.
    Works with: adam, sgd, lbfgs, and other optax optimizers.
    
    Args:
        func: Function to optimize
        ndim: Number of dimensions
        bounds: Parameter bounds as [(low1, high1), (low2, high2), ...] or None
        x0: Initial guess (optional)
        lr: Learning rate
        maxiter: Maximum iterations
        n_restarts: Number of restarts
        minimize: Whether to minimize (True) or maximize (False)
        verbose: Whether to print progress
        random_seed: Random seed for reproducibility
        optimizer_name: Name of the optax optimizer ("adam", "sgd", "lbfgs")
        **func_kwargs: Additional arguments passed to the function
        
    Returns:
        Tuple of (best_params, best_value)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        key = jax.random.PRNGKey(random_seed)
    else:
        from .seed_utils import get_new_jax_key, get_global_seed
        if get_global_seed() is not None:
            key = get_new_jax_key()
        else:
            key = jax.random.PRNGKey(0)

    # Setup bounds
    bounds = _setup_bounds(bounds, ndim)

    # Create scaled objective function
    scaled_func = lambda x: func(_scale_from_unit(x, bounds), **func_kwargs)

    # JIT compile value and gradient
    @jax.jit
    def val_grad(x):
        return jax.value_and_grad(scaled_func)(x)

    # JIT step function
    optimizer = _get_optimizer(optimizer_name, lr)

    @jax.jit
    def step(params, opt_state):
        (val, grad) = val_grad(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, 0., 1.)
        return params, opt_state, val

    # Generate initial parameters for all restarts
    if x0 is None:
        keys = jax.random.split(key, n_restarts)
        init_params = jax.vmap(
            lambda k: jax.random.uniform(k, shape=(ndim,), minval=0., maxval=1.)
        )(keys)
    else:
        init_params = jnp.tile(_scale_to_unit(jnp.array(x0), bounds)[None, :], (n_restarts, 1))

    # Initialize optimizer states for all restarts
    opt_states = jax.vmap(optimizer.init)(init_params)

    best_f = jnp.inf if minimize else -jnp.inf
    best_params_unit = None

    # Outer loop for fixed number of optimization steps
    for iteration in range(maxiter):
        # Run one step of optimization for all restarts in parallel
        params_next, opt_states_next, values = jax.vmap(step)(init_params, opt_states)
        
        # Update parameters and states for next iteration
        init_params = params_next
        opt_states = opt_states_next

        # Find best among all parallel restarts at this step
        if minimize:
            current_best_idx = jnp.argmin(values)
            current_best_val = values[current_best_idx]
            is_better = current_best_val < best_f
        else:
            current_best_idx = jnp.argmax(values)
            current_best_val = values[current_best_idx]
            is_better = current_best_val > best_f

        if is_better:
            best_f = current_best_val
            best_params_unit = init_params[current_best_idx]

        # Optional: Add noise to parameters for next iteration (except last)
        if iteration < maxiter - 1:
            noise_keys = jax.random.split(jax.random.fold_in(key, iteration), n_restarts)
            noise = 0.1 * jax.vmap(jax.random.normal)(noise_keys, shape=(n_restarts, ndim))
            init_params = jnp.clip(init_params + noise, 0., 1.)

        if verbose:
            mode_str = "min" if minimize else "max"
            desc = f'Step {iteration+1}/{maxiter} ({optimizer_name}, {mode_str})'
            display_val = float(best_f) if minimize else float(-best_f)
            log.info(f"{desc}: best_f = {display_val}")

    # Convert best parameters back to original space
    best_params_original = _scale_from_unit(best_params_unit, bounds)
    best_f_original = best_f if minimize else -best_f

    return best_params_original, best_f_original