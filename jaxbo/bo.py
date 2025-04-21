import sys
import numpy as np
from scipy.stats import qmc
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from numpyro.util import enable_x64
enable_x64()
from .acquisition import WIPV, EI #, logEI
from .gp import DSLP_GP, SAAS_GP, sample_GP_NUTS
from .svm_gp import SVM_GP
from .loglike import external_loglike,cobaya_loglike
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from .bo_utils import input_standardize, input_unstandardize
from optax import adam, apply_updates
from .nested_sampler import nested_sampling_Dy, nested_sampling_jaxns
from getdist import plots, MCSamples, loadMCSamples
import tqdm
import time
import logging

# log = logging.getLogger("[BO]")

# 1) Filter class: only allow exactly INFO
class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO

# 2) Create and configure the stdout handler
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)       # accept INFO and above...
stdout_handler.addFilter(InfoFilter())      # ...but filter down to only INFO
stdout_fmt = logging.Formatter('%(asctime)s %(levelname)s:%(name)s: %(message)s')
stdout_handler.setFormatter(stdout_fmt)

# 3) Create and configure the stderr handler
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING)    # accept WARNING and above
stderr_fmt = logging.Formatter('%(asctime)s %(levelname)s:%(name)s: %(message)s')
stderr_handler.setFormatter(stderr_fmt)

# 4) Get your logger, clear defaults, add both handlers
log = logging.getLogger(name="[BO]")
# stop this logger “bubbling” messages up to root
log.propagate = False  
log.handlers.clear()
log.setLevel(logging.INFO)               # ensure INFO+ get processed
log.addHandler(stdout_handler)
log.addHandler(stderr_handler)

# Acquisition optimizer
def optimize_acq(gp, acq, mc_points, x0=None, lr=5e-3, maxiter=150, n_restarts_optimizer=4):
    
    f = lambda x: acq(x=x, gp=gp, mc_points=mc_points)

    @jax.jit
    def acq_val_grad(x):
        return jax.value_and_grad(f)(x)

    params = jnp.array(np.random.uniform(0, 1, size=mc_points.shape[1])) if x0 is None else x0
    optimizer = adam(learning_rate=lr)

    @jax.jit
    def step(carry):
        params, opt_state = carry
        (val, grad) = acq_val_grad(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = apply_updates(params, updates)
        return (jnp.clip(params, 0., 1.), opt_state), val

    best_f, best_params = jnp.inf, None
    r = jnp.arange(maxiter)
    for n in range(n_restarts_optimizer):
        opt_state = optimizer.init(params)
        progress_bar = tqdm.tqdm(r,desc=f'ACQ Optimization restart {n+1}')
        for i in progress_bar:
            (params, opt_state), fval = step((params, opt_state))
            progress_bar.set_postfix({"fval": float(fval)})
            if fval < best_f:
                best_f, best_params = fval, params
        # Perturb for next restart
        params = jnp.clip(best_params + 0.5 * jnp.array(np.random.normal(size=params.shape)), 0., 1.)

    # print(f"Best params: {best_params}, fval: {best_f}")
    return jnp.atleast_2d(best_params), best_f

# Utility functions

# def get_point_with_large_value(train_x,train_y, n_points=1):
#     """
#     Get a point with large value from the training data
#     """
#     idx = jnp.argsort(train_y.flatten())[-n_points:]
#     return train_x[idx].flatten()

def get_mc_samples(gp, rng_key, warmup_steps=512, num_samples=512, thinning=1,method="NUTS",init_params=None):
    if method=='NUTS':
        try:
            mc_samples = sample_GP_NUTS(
            gp=gp, rng_key=rng_key, warmup_steps=warmup_steps,
            num_samples=num_samples, thinning=thinning
            )
        except:
            log.error("Error in sampling GP NUTS")
            mc_samples, _ = nested_sampling_Dy(gp, gp.ndim, maxcall=int(1e6)
                                            , dynamic=False, dlogz=0.1,equal_weights=True,
            )
    elif method=='NS':
        mc_samples, _ = nested_sampling_Dy(gp, gp.ndim, maxcall=int(1e6)
                                            , dynamic=False, dlogz=0.1,equal_weights=True,
        )
    elif method=='uniform':
        mc_samples = {}
        points = qmc.Sobol(gp.ndim, scramble=True).random(num_samples)
        mc_samples['x'] = points
    else:
        raise ValueError(f"Unknown method {method} for sampling GP")
    return mc_samples


def get_mc_points(mc_samples, mc_points_size=64):
    mc_size = max(mc_samples['x'].shape[0], mc_points_size)
    idxs = np.random.choice(mc_size, size=mc_points_size, replace=False)
    return mc_samples['x'][idxs]


class BOBE:

    def __init__(self,
                 n_cobaya_init=4,
                 n_sobol_init=32,
                 miniters=200,
                 maxiters=1500,
                 max_gp_size=1200,
                 loglikelihood=None,
                 resume=False,
                 resume_file=None,
                 save=True,
                 fit_step=10,
                 update_mc_step=10,
                 ns_step=10,
                 num_hmc_warmup=512,
                 num_hmc_samples=512,
                 mc_points_size=64,
                 mc_points_method='NUTS',
                 lengthscale_priors='DSLP',
                 use_svm=True,
                 svm_use_size = 300,
                 svm_update_step=5,
                 logz_threshold=1.0,
                 minus_inf=-1e5):
        
        if not isinstance(loglikelihood, external_loglike):
            raise ValueError("loglikelihood must be an instance of external_loglike")

        self.loglikelihood = loglikelihood

        self.param_list = self.loglikelihood.param_list
        self.param_bounds = self.loglikelihood.param_bounds
        self.param_labels = self.loglikelihood.param_labels
        self.ndim = len(self.param_list)


        if resume:
            assert resume_file is not None, "resume_file must be provided if resume is True"
            log.info(f" Resuming from file {resume_file}")
            data = np.load(resume_file)
            self.train_x = jnp.array(data['svm_train_x'])
            self.train_y = jnp.array(data['svm_train_y'])
        else:
            init_points, init_vals = self.loglikelihood.get_initial_points(n_cobaya_init=n_cobaya_init,
                                    n_init_sobol=n_sobol_init)
            self.train_x = jnp.array(input_standardize(init_points, self.param_bounds))
            self.train_y = jnp.array(init_vals)

        # Best point so far
        idx_best = jnp.argmax(self.train_y)
        self.best_pt = input_unstandardize(self.train_x[idx_best],self.param_bounds).flatten()
        self.best_f = float(self.train_y.max())
        self.best = {name: f"{float(val):.4f}" for name, val in zip(self.param_list, self.best_pt)}
        log.info(f" Initial best point {self.best} with value = {self.best_f:.4f}")

        # GP setup
        if use_svm:
            gp = SVM_GP(
                train_x=self.train_x, train_y=self.train_y,
                minus_inf=minus_inf, lengthscale_priors=lengthscale_priors,
                kernel='rbf',svm_use_size=svm_use_size,svm_update_step=svm_update_step,
            )
        else:
            gp = {
                'DSLP': DSLP_GP,
                'SAAS': SAAS_GP
            }[lengthscale_priors](
                train_x=self.train_x, train_y=self.train_y,
                noise=1e-8, kernel='rbf'
            )
        gp.fit()
        self.gp = gp

        # Store settings
        self.maxiters = maxiters
        self.miniters = miniters
        self.max_gp_size = max_gp_size
        self.fit_step = fit_step
        self.update_mc_step = update_mc_step
        self.ns_step = ns_step
        self.num_hmc_warmup = num_hmc_warmup
        self.num_hmc_samples = num_hmc_samples
        self.mc_points_size = mc_points_size
        self.minus_inf = minus_inf
        self.output_file = self.loglikelihood.name
        self.mc_points_method = mc_points_method
        self.save = save

        # Convergence control
        self.logz_threshold = logz_threshold
        self.converged = False
        self.termination_reason = "Max iterations reached"



    def check_convergence(self, step, logz_dict, threshold=2.0):
        """
        Check if the nested sampling has converged.
        """
        delta = logz_dict['upper'] - logz_dict['lower']
        if delta < threshold and step > self.miniters:
            log.info(f" Convergence check: delta = {delta:.4f}, step = {step}")
            log.info(" Converged")
            return True
        else:
            return False

    def run(self):
        """
        Execute the iterative BO process.
        """

        # Monte Carlo points for acquisition function
        self.mc_samples = get_mc_samples(self.gp, rng_key=jax.random.PRNGKey(0),
            warmup_steps=self.num_hmc_warmup, num_samples=self.num_hmc_samples, thinning=1,method=self.mc_points_method)
        self.mc_points = get_mc_points(self.mc_samples, self.mc_points_size)

        for i in range(self.maxiters):
            ii = i + 1
            refit = (ii % self.fit_step == 0)
            update_mc = (ii % self.update_mc_step == 0)
            ns_flag = (ii % self.ns_step == 0)

            print("\n")
            log.info(f" Iteration {ii}/{self.maxiters}, refit={refit}, update_mc={update_mc}, ns={ns_flag}")
            x0 = self.gp.train_x[jnp.argmax(self.gp.train_y)]

            new_pt_u, acq_val = optimize_acq(
                self.gp, WIPV, self.mc_points, x0=x0)
            
            new_pt = input_unstandardize(new_pt_u, self.param_bounds) #.flatten()

            log.info(f" Acquisition value {acq_val:.4e} at new point")
            new_val = self.loglikelihood(
                new_pt, logp_args=(), logp_kwargs={}
            )

            new = {name: f"{float(val):.4f}" for name, val in zip(self.param_list, new_pt.flatten())}
            log.info(f" New point {new} with value = {new_val.item():.4f}")

            pt_exists = self.gp.update(new_pt_u, new_val, refit=refit)
            if pt_exists:
                update_mc = True
            if update_mc:
                self.mc_samples = get_mc_samples(
                    self.gp, rng_key=jax.random.PRNGKey(ii*10),
                    warmup_steps=self.num_hmc_warmup, num_samples=self.num_hmc_samples,
                    thinning=1, method=self.mc_points_method,init_params=x0
                )
            self.mc_points = get_mc_points(self.mc_samples, self.mc_points_size)

            if float(new_val) > self.best_f:
                self.best_f = float(new_val)
                self.best_pt = new_pt
                self.best = {name: f"{float(val):.4f}" for name, val in zip(self.param_list, self.best_pt.flatten())}
            log.info(f" Current best point {self.best} with value = {self.best_f:.4f}")

            if i % 4 == 0 and i > 0:
                jax.clear_caches()

            if (ii % 50 == 0) and self.save:
                log.info(" Saving GP to file")
                self.gp.save(outfile=self.output_file)

            if ns_flag:
                log.info(" Running Nested Sampling")
                ns_samples, logz_dict = nested_sampling_Dy(
                    self.gp, self.ndim, maxcall=int(1e6), dynamic=False, dlogz=0.05
                )
                log.info(" LogZ info: " + ", ".join([f"{k}={v:.4f}" for k,v in logz_dict.items()]))
                if self.check_convergence(i, logz_dict,threshold=self.logz_threshold):
                    self.converged = True
                    self.termination_reason = "LogZ converged"
                    break

            if self.gp.train_x.shape[0] > self.max_gp_size:
                self.termination_reason = "Max GP size exceeded"
                log.info(f" {self.termination_reason}")
                break

        self.gp.fit()

        # Save and final nested sampling
        if self.save:
            self.gp.save(outfile=self.output_file)
        log.info(f" Sampling stopped: {self.termination_reason}")

        log.info(" Final Nested Sampling")
        ns_samples, logz_dict = nested_sampling_Dy(
            self.gp, self.ndim, maxcall=int(5e6), dynamic=True, dlogz=0.01
        )
        log.info(" Final LogZ: " + ", ".join([f"{k}={v:.4f}" for k,v in logz_dict.items()]))

        return self.gp, ns_samples, logz_dict


