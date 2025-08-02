from typing import Optional, Union, List, Tuple, Callable, Any
import jax
import jax.numpy as jnp
import numpy as np
from jaxbo.bo_utils import input_standardize, input_unstandardize
import logging
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import optax

log = logging.getLogger("[Opt]")


class FunctionOptimizer:
    """
    A flexible optimizer class for arbitrary functions supporting multiple optax optimizers.
    
    Supports parameter bounds with automatic rescaling to [0,1] during optimization.
    Works with: adam, sgd, lbfgs, and other optax optimizers.
    """

    def __init__(self, optimizer_name: str = "adam", **optimizer_kwargs):
        """
        Initialize the optimizer.

        Args:
            optimizer_name: Name of the optax optimizer ("adam", "sgd", "lbfgs")
            **optimizer_kwargs: Additional arguments for the optimizer
        """
        self.optimizer_name = optimizer_name.lower()
        self.optimizer_kwargs = optimizer_kwargs

    def _get_optimizer(self, learning_rate: float = 1e-3) -> optax.GradientTransformation:
        """
        Get the optax optimizer based on the specified name.

        Args:
            learning_rate: Learning rate for the optimizer

        Returns:
            optax.GradientTransformation: The configured optimizer
        """
        # Set default learning rate if not provided in kwargs
        if 'learning_rate' not in self.optimizer_kwargs:
            self.optimizer_kwargs['learning_rate'] = learning_rate

        if self.optimizer_name == "adam":
            return optax.adam(**self.optimizer_kwargs)
        elif self.optimizer_name == "sgd":
            return optax.sgd(**self.optimizer_kwargs)
        elif self.optimizer_name == "lbfgs":
            return optax.lbfgs(**self.optimizer_kwargs)
        else:
            try:
                optimizer_fn = getattr(optax, self.optimizer_name)
                return optimizer_fn(**self.optimizer_kwargs)
            except AttributeError:
                raise ValueError(f"Optimizer '{self.optimizer_name}' not found in optax library")

    def _setup_bounds(self, bounds: Optional[Union[List, Tuple, jnp.ndarray]], ndim: int) -> jnp.ndarray:
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

    def _scale_to_unit(self, x: jnp.ndarray, bounds: jnp.ndarray) -> jnp.ndarray:
        return input_standardize(x, bounds)

    def _scale_from_unit(self, x: jnp.ndarray, bounds: jnp.ndarray) -> jnp.ndarray:
        return input_unstandardize(x, bounds)

    def optimize(
        self,
        func: Callable,
        ndim: int,
        bounds: Optional[Union[List, Tuple, jnp.ndarray]] = None,
        x0: Optional[jnp.ndarray] = None,
        lr: float = 5e-3,
        maxiter: int = 100,
        n_restarts: int = 4,
        minimize: bool = True,
        verbose: bool = True,
        random_seed: Optional[int] = None,
        **func_kwargs
    ) -> Tuple[jnp.ndarray, float]:
        """
        Run optimization with fixed number of steps, evaluating n_restarts in parallel at each step.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            key = jax.random.PRNGKey(random_seed)
        else:
            key = jax.random.PRNGKey(0)

        # Setup bounds
        bounds = self._setup_bounds(bounds, ndim)

        # Create scaled objective function
        scaled_func = lambda x: func(self._scale_from_unit(x, bounds), **func_kwargs)

        # JIT compile value and gradient
        @jax.jit
        def val_grad(x):
            return jax.value_and_grad(scaled_func)(x)

        # JIT step function
        optimizer = self._get_optimizer(lr)

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
            init_params = jnp.tile(self._scale_to_unit(jnp.array(x0), bounds)[None, :], (n_restarts, 1))

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
                desc = f'Step {iteration+1}/{maxiter} ({self.optimizer_name}, {mode_str})'
                display_val = float(best_f) if minimize else float(-best_f)
                print(f"{desc}: best_f = {display_val}")

        # Convert best parameters back to original space
        best_params_original = self._scale_from_unit(best_params_unit, bounds)
        best_f_original = best_f if minimize else -best_f

        return best_params_original, best_f_original