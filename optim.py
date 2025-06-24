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

def optimise_vmap(gp, acq, mc_points, x0=None, lr=5e-3, maxiter=150, n_restarts_optimiser=4):
    min_delta=1e-7 #Minimum amount by which loss function must improve to be considered progressing
    patience=5 #Number of steps without progress before restart is stopped
    
    f = lambda x: acq(x=x, gp=gp, mc_points=mc_points)

    @jax.jit
    def acq_val_grad(x):
        return jax.value_and_grad(f)(x)
        
    key = jax.random.PRNGKey(0)
    params = jax.random.uniform(key, (n_restarts_optimiser, mc_points.shape[1]), minval=0, maxval=1) if x0 is None else jnp.tile(x0, (n_restarts_optimiser, 1))
    optimizer = adam(learning_rate=lr)
    opt_state = jax.vmap(optimizer.init)(params)
    
    @jax.jit
    def scan_step(carry):
        params, opt_state, best_params, best_values, no_improve_steps, active_mask = carry

        values, grads = acq_val_grad(params)
        
        # Zero out gradients for inactive restarts
        grads = jnp.where(active_mask[:, None], grads, jnp.zeros_like(grads))
        updates, opt_state = jax.vmap(optimizer.update)(grads, opt_state, params)
        new_params = apply_updates(params, updates)

        is_better = values < (best_values - min_delta)
        best_params = jnp.where(is_better[:, None], new_params, best_params)
        best_values = jnp.where(is_better, values, best_values)
        no_improve_steps = jnp.where(is_better, 0, no_improve_steps + 1)

        # Deactivate finished restarts
        active_mask = jnp.where(no_improve_steps >= patience, False, active_mask)

        # Keep params unchanged if inactive
        new_params = jnp.where(active_mask[:, None], new_params, params)

        return (new_params, opt_state, best_params, best_values, no_improve_steps, active_mask)
        
    best_values, _ = jax.vmap(acq_val_grad)(params)
    active_mask = jnp.ones(n_restarts_optimiser, dtype=bool)
    no_improve_steps = jnp.zeros(n_restarts_optimiser, dtype=jnp.int32)

    carry = (params, opt_state, params, best_values, no_improve_steps, active_mask)

    pbar = tqdm_(range(maxiter), desc="Acq Optimisation (parallel with per-restart early stop)")
    for i in pbar:
        carry = scan_step(carry)
        _, _, best_params, best_values, no_improve_steps, active_mask = carry

        pbar.set_postfix(
            best_val=jnp.min(best_values).item(),
            #active_restarts=active_mask.sum().item(),
        )

        if jnp.logical_not(jnp.any(active_mask)):
            #print(f"\n All restarts stopped early after {i+1} steps.")
            break
    best_param = best_params[jnp.argmin(best_values)]
    best_f = jnp.min(best_values)
    
    return jnp.atleast_2d(best_param), best_f