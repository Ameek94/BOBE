import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jaxbo.bo_utils import input_standardize, input_unstandardize
import logging
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from optax import adam, apply_updates

log = logging.getLogger("[Opt]")

# in progress
def optim_optax(func, ndim,x0=None, param_bounds=None, lr=5e-3, maxiter=150, n_restarts=4):

    param_bounds = np.array(param_bounds) if param_bounds is not None else np.array(ndim*[[0,1]]).T # type: ignore

    mins = param_bounds[:,0]
    maxs = param_bounds[:,1]
    scales = maxs - mins

    optimizer = adam(lr)

    init_params = x0 if x0 is not None else np.random.uniform(0, 1, size=(ndim,))

    @jax.jit
    def step(carry):
        """
        Step function for the optimizer
        """
        u_params, opt_state = carry
        params = input_unstandardize(u_params, param_bounds.T)
        loss, grads = jax.value_and_grad(func)(params)
        grads = grads * scales
        updates, opt_state = optimizer.update(grads, opt_state)
        u_params = apply_updates(u_params, updates)
        u_params = jnp.clip(u_params, 0., 1.)
        carry = u_params, opt_state
        return carry, loss

    best_f = jnp.inf
    best_params = None

    u_params = input_standardize(init_params, param_bounds.T)

    # display with progress bar
    r = jnp.arange(maxiter)
    for n in range(n_restarts):
        opt_state = optimizer.init(u_params)
        progress_bar = tqdm.tqdm(r, desc=f'Training GP')
        with logging_redirect_tqdm():
            for i in progress_bar:
                (u_params, opt_state), fval = step((u_params, opt_state))
                progress_bar.set_postfix({"fval": float(fval)})
                if fval < best_f:
                    best_f = fval
                    best_params = u_params
        u_params = jnp.clip(u_params + 0.25 * np.random.normal(size=init_params.shape), 0, 1)
    params = input_unstandardize(best_params, param_bounds.T)
    return params, best_f