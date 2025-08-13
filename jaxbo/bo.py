import sys
import numpy as np
from scipy.stats import qmc
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from numpyro.util import enable_x64
enable_x64()
from typing import Optional, Union, Tuple, Dict, Any
from .acquisition import WIPV, EI #, logEI
from .gp import DSLP_GP, SAAS_GP, Uniform_GP #, sample_GP_NUTS
from .svm_gp import SVM_GP
from .clf_gp import ClassifierGP
from .loglike import ExternalLikelihood, CobayaLikelihood
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from .utils import scale_from_unit, scale_to_unit
from .seed_utils import set_global_seed, get_global_seed, get_jax_key, split_jax_key, ensure_reproducibility
from optax import adam, apply_updates
from .nested_sampler import nested_sampling_Dy#, nested_sampling_jaxns
from getdist import plots, MCSamples, loadMCSamples
import tqdm
import time
import logging

# log = logging.getLogger("[BO]")

# 1) Filter class: only allow exactly INFO
class InfoFilter(logging.Filter):
    """
    """
    def filter(self, record):
        """
        """
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
def optimize_acq(gp, acq, mc_points, x0=None, lr=5e-3, maxiter=250, n_restarts_optimizer=8):
    if mc_points is not None:
        f = lambda x: acq(x=x, gp=gp, mc_points=mc_points)
    else:
        f = lambda x: acq(x=x, gp=gp)
    
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
    opt_path = []
    for n in range(n_restarts_optimizer):
        curr_opt_path = []
        opt_state = optimizer.init(params)
        progress_bar = tqdm.tqdm(r,desc=f'ACQ Optimization restart {n+1}')
        for i in progress_bar:
            curr_opt_path.append(params)
            (params, opt_state), fval = step((params, opt_state))
            progress_bar.set_postfix({"fval": float(fval)})
            if fval < best_f:
                best_f, best_params = fval, params
        # Perturb for next restart
        opt_path.append(curr_opt_path)
        params = jnp.clip(best_params + 0.5 * jnp.array(np.random.normal(size=params.shape)), 0., 1.)

    #print(f"Best params: {best_params}, fval: {best_f}")
    return jnp.atleast_2d(best_params), best_f, opt_path

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
            mc_samples = gp.sample_GP_NUTS(rng_key=rng_key, warmup_steps=warmup_steps,
            num_samples=num_samples, thinning=thinning
            )
        except Exception as e:
            log.error(f"Error in sampling GP NUTS: {e}")
            mc_samples, logz, success = nested_sampling_Dy(gp, gp.ndim, maxcall=int(2e6)
                                            , dynamic=False, dlogz=0.5,equal_weights=True,
            )
    elif method=='NS':
        mc_samples, logz, success = nested_sampling_Dy(gp, gp.ndim, maxcall=int(2e6)
                                            , dynamic=False, dlogz=0.5,equal_weights=False,
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
    return {'x': mc_samples['x'][idxs], 'weights': mc_samples['weights'][idxs]}


class BOBE:

    def __init__(self,
                loglikelihood=None,
                 n_cobaya_init=4,
                 n_sobol_init=32,
                 miniters=200,
                 maxiters=1500,
                 max_gp_size=1200,
                 resume=False,
                 resume_file=None,
                 save=True,
                 save_dir=None,
                 fit_step=10,
                 update_mc_step=10,
                 ns_step=10,
                 num_hmc_warmup=512,
                 num_hmc_samples=512,
                 mc_points_size=64,
                 mc_points_method='NUTS',
                 lengthscale_priors='DSLP',
                 acq = 'WIPV',
                 use_clf=True,
                 clf_type = "svm",
                 clf_use_size = 300,
                 clf_update_step=5,
                 clf_threshold=250,
                 gp_threshold=5000,
                 logz_threshold=1.0,
                 minus_inf=-1e5,
                 do_final_ns=True,
                 return_getdist_samples=False,
                 seed: Optional[int] = None):
        """
        Initialize the BOBE sampler class.

        Arguments
        ---------
        loglikelihood : external_loglike
            The loglikelihood function to be used. Must be an instance of external_loglike.
        n_cobaya_init : int
            Number of initial points from the cobaya reference distirbution when starting a run. 
            Is only used when the likelihood is an instance of cobaya_loglike, otherwise ignored.
        n_sobol_init : int
            Number of initial Sobol points for sobol when starting a run. 
        miniters : int
            Minimum number of iterations before checking convergence.
        maxiters : int
            Maximum number of iterations.
        max_gp_size : int
            Maximum number of points used to train the GP. 
            If using SVM, this is not the same as the number of points used to train the SVM.
        resume : bool
            If True, resume from a previous run. The resume_file argument must be provided.
        resume_file : str
            The file to resume from. Must be a .npz file containing the training data.
        save : bool
            If True, save the GP training data to a file so that it can be resumed from later.
        fit_step : int
            Number of iterations between GP refits.
        update_mc_step : int
            Number of iterations between MC point updates.
        ns_step : int
            Number of iterations between nested sampling runs.
        num_hmc_warmup : int
            Number of warmup steps for HMC sampling.
        num_hmc_samples : int
            Number of samples to draw from the GP.
        mc_points_size : int
            Number of points to use for the weighted integrated posterior variance acquisition function.
        mc_points_method : str
            Method to use for generating the MC points. Options are 'NUTS', 'NS', or 'uniform'. 
            Recommend to use 'NUTS' for most cases, 'NS' can be a good choice if the underlying likelihood has a highly complex structure.
        lengthscale_priors : str
            Lengthscale priors to use. Options are 'DSLP' or 'SAAS'. See the GP class for more details.
        use_clf : bool
            If True, use SVM to filter the GP predictions. 
            This is only required for high dimensional problems and when the scale of variation of the likelihood is extremely large. 
            For cosmological likelihoods with nuisance parameters, this is highly recommended.
        clf_use_size : int
            Minimum size of the classifier training set before the classifier filter is used in the GP.
        clf_update_step : int
            Number of iterations between classifier updates.
        logz_threshold : float
            Threshold for convergence of the nested sampling logz. 
            If the difference between the upper and lower bounds of logz is less than this value, the sampling will end.
        minus_inf : float
            Value to use for minus infinity. This is used to set the lower bound of the loglikelihood.
        """


        set_global_seed(seed)

        if not isinstance(loglikelihood, ExternalLikelihood):
            raise ValueError("loglikelihood must be an instance of ExternalLikelihood")

        self.loglikelihood = loglikelihood
        self.ndim = len(self.loglikelihood.param_list)

        ### Debug Plot Variables #######################################################################
        self.timing = {'GP': [], 'MC': [], 'ACQ': [], 'NS': [], 'LH': [], 'Step': []}                  #
        self.logz_data = []                                                                            #
        self.hyperparameter_data = {'lengthscales': [], 'outputscale': [], 'mll': []}                  #
        self.acq_data = {'xs': [], 'ys': [], 'mc_points': [], 'pt_and_val': [],  'opt_path': []}       #
        self.sample_point_data = {'del_val': [], 'variance': []}                                       #
        self.n_sobol_init = n_sobol_init                                                               #
        self.mll_val = np.nan                                                                          #
        self.acq_val = None                                                                            #
        self.use_clf = use_clf                                                                         #   
        ################################################################################################

        if resume and resume_file is not None:
            # assert resume_file is not None, "resume_file must be provided if resume is True"
            log.info(f" Resuming from file {resume_file}")
            data = np.load(resume_file)
            self.train_x = jnp.array(data['train_x'])
            self.train_y = jnp.array(data['train_y'])
            lengthscales = data['lengthscales']
            outputscale = data['outputscale']
        else:
            init_points, init_vals = self.loglikelihood.get_initial_points(n_cobaya_init=n_cobaya_init,
                                    n_init_sobol=n_sobol_init)
            self.train_x = jnp.array(scale_to_unit(init_points, self.loglikelihood.param_bounds))
            self.train_y = jnp.array(init_vals)
            lengthscales = None
            outputscale = None

        # Best point so far
        idx_best = jnp.argmax(self.train_y)
        self.best_pt = scale_from_unit(self.train_x[idx_best],self.loglikelihood.param_bounds).flatten()
        self.best_f = float(self.train_y.max())
        self.best = {name: f"{float(val):.4f}" for name, val in zip(self.loglikelihood.param_list, self.best_pt)}
        log.info(f" Initial best point {self.best} with value = {self.best_f:.4f}")

        # GP setup
        if use_clf:
            self.gp = ClassifierGP(
                train_x=self.train_x, train_y=self.train_y,
                minus_inf=minus_inf, lengthscale_priors=lengthscale_priors,
                clf_type=clf_type, clf_use_size=clf_use_size, clf_update_step=clf_update_step,
                clf_threshold=clf_threshold, gp_threshold=gp_threshold,
                lengthscales=lengthscales, outputscale=outputscale
            )
            # gp = SVM_GP(
            #     train_x=self.train_x, train_y=self.train_y,
            #     minus_inf=minus_inf, lengthscale_priors=lengthscale_priors,
            #     kernel='rbf',svm_use_size=svm_use_size,svm_update_step=svm_update_step,
            #     svm_threshold=svm_threshold,gp_threshold=svm_gp_threshold,lengthscales=lengthscales,
            #     outputscale=outputscale
            # )
        else:
            self.gp = {
                'DSLP': DSLP_GP,
                'SAAS': SAAS_GP,
                'Uniform': Uniform_GP
            }[lengthscale_priors](
                train_x=self.train_x, train_y=self.train_y,
                noise=1e-8, kernel='rbf',lengthscales=lengthscales,outputscale=outputscale
            )


        if resume:
            self.gp.fit(maxiter=100,n_restarts=2) # if resuming need not spend too much time on fitting
        else:
            # Fit the GP to the initial points
            self.gp.fit(maxiter=150,n_restarts=4)


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
        self.output_file = f"{save_dir}{self.loglikelihood.name}"
        self.mc_points_method = mc_points_method
        self.save = save
        self.return_getdist_samples = return_getdist_samples
        self.do_final_ns = do_final_ns

        # Convergence control
        self.logz_threshold = logz_threshold
        self.converged = False
        self.termination_reason = "Max iterations reached"

    def check_convergence(self, step, logz_dict, threshold=2.0,ndim=1):
        """
        Check if the nested sampling has converged.
        """
        if ndim > 10:
            delta = logz_dict['mean'] - logz_dict['lower'] # for now just to speed up results
        else:
            delta = logz_dict['upper'] - logz_dict['lower']
        mean = logz_dict['mean']
        if (delta < threshold and mean>self.minus_inf+10)  and step > self.miniters:
            log.info(f" Convergence check: delta = {delta:.4f}, step = {step}")
            log.info(" Converged")
            return True
        else:
            return False

    def run(self):
        """
        Run the iterative Bayesian Optimization loop.

        Arguments
        ---------
        None

        Returns
        ---------
        gp : GP object
            The fitted GP object.
        ns_samples : MCSamples | Nested sampling samples
            The samples from the final nested sampling run. This is a either a getdist MCSamples instance or a dictionary with the following keys ['x','weights','logl'].
        logz_dict : dict
            The logz dictionary from the nested sampling run. This contains the upper and lower bounds of the logz.
        """


        # Monte Carlo points for acquisition function
        self.mc_samples = get_mc_samples(self.gp, rng_key=jax.random.PRNGKey(0),
            warmup_steps=self.num_hmc_warmup, num_samples=self.num_hmc_samples, thinning=1,method=self.mc_points_method)
        self.mc_samples['method'] = 'MCMC'        
        self.mc_points = get_mc_points(self.mc_samples, self.mc_points_size)
        ns_samples = None
        logz_dict = None

        best_pt_iteration = 0

        ii = 0

        for i in range(self.maxiters):

            ii = i + 1
            refit = (ii % self.fit_step == 0)
            ns_flag = (ii % self.ns_step == 0) #and ii >= self.miniters
            update_mc = (ii % self.update_mc_step == 0) and not ns_flag

            print("\n")
            log.info(f" Iteration {ii}/{self.maxiters}, refit={refit}, update_mc={update_mc}, ns={ns_flag}")
            
            #start acq from mc_point with max var or max value
           
            
            ### ACQ ###
            #x0_acq = self.mc_samples['best']
            start_acq = time.time()
            if ii > self.n_sobol_init:
                f = lambda x: WIPV(x=x, gp=self.gp, mc_points=self.mc_points)
                mc_points_var = jax.vmap(self.gp.predict_var)(self.mc_points['x']) #jax.lax.map(self.gp.predict_var,self.mc_points['x'])
                x0_acq = self.mc_points['x'][jnp.argmax(mc_points_var)]
                new_pt_u, acq_val, opt_path = optimize_acq(
                self.gp, WIPV, self.mc_points, x0=x0_acq)
            else:
                acq = EI(gp=self.gp, zeta=0.01)
                f = lambda x: acq(x=x, gp=self.gp, mc_points=None)
                x0_acq =  self.gp.train_x[jnp.argmax(self.gp.train_y)]
                new_pt_u, acq_val, opt_path = optimize_acq(
                self.gp, EI(gp=self.gp, zeta=0.01), None, x0=x0_acq)
            
            
            new_pt = scale_from_unit(new_pt_u, self.loglikelihood.param_bounds) #.flatten()
            end_acq = time.time()
            self.timing['ACQ'].append(end_acq-start_acq)
            ############

            ### ACQ DEBUG ###
            self.acq_data['opt_path'].append(opt_path)
            if self.ndim == 1:
                acq_plot_xs = np.linspace(0, 1, 100)
                acq_plot_ys = [f(x) for x in acq_plot_xs]
            if self.ndim == 2:
                acq_plot_x1 = acq_plot_x2 = np.linspace(0, 1, 50)
                acq_plot_X1, acq_plot_X2 = np.meshgrid(acq_plot_x1, acq_plot_x2)
                acq_plot_xs = np.stack([acq_plot_X1.ravel(), acq_plot_X2.ravel()], axis=-1)
                acq_plot_ys = [f(x) for x in acq_plot_xs]
            if self.ndim == 1 or self.ndim == 2:
                self.acq_data['xs'].append(acq_plot_xs)
                self.acq_data['ys'].append(acq_plot_ys)
                self.acq_data['mc_points'].append(self.mc_points)
                acq_pt = new_pt_u.tolist()[0]
                #log.info(f"{acq_pt}")
                self.acq_data['pt_and_val'].append([*acq_pt, acq_val])
            #################

            log.info(f" Acquisition value {acq_val:.4e} at new point")
            ### LH ###
            lh_start = time.time()
            new_val = self.loglikelihood(
                new_pt, logp_args=(), logp_kwargs={}
            )
            lh_end = time.time()
            self.timing['LH'].append(lh_end-lh_start)
            ##########

            new_pt_vals = {name: f"{float(val):.4f}" for name, val in zip(self.loglikelihood.param_list, new_pt.flatten())}
            log.info(f" New point {new_pt_vals}")
            suprise_factor = (np.abs(new_val.item() - self.gp.predict_mean(new_pt_u).item()))/np.sqrt(self.gp.predict_var(new_pt_u).item())
            log.info(f" Objective function value = {new_val.item():.4f}, GP predicted value = {self.gp.predict_mean(new_pt_u).item():.4f}, Suprise Factor = {suprise_factor:.4f}")
            self.sample_point_data['del_val'].append((np.abs(new_val.item() - self.gp.predict_mean(new_pt_u).item())))
            self.sample_point_data['variance'].append(self.gp.predict_var(new_pt_u).item())

            ### GP FIT ###
            start_gp = time.time()
            pt_exists_or_below_threshold, mll_val = self.gp.update(new_pt_u, new_val, refit=refit,step=ii,n_restarts=8)
            # x0_acq =  self.gp.train_x[jnp.argmax(self.gp.train_y)]
            end_gp = time.time()
            self.timing['GP'].append(end_gp - start_gp)
            if mll_val != np.nan:
                self.mll_val = mll_val
            if self.use_clf:
                self.hyperparameter_data['lengthscales'].append(self.gp.gp.lengthscales.tolist())
                self.hyperparameter_data['outputscale'].append(self.gp.gp.outputscale.tolist())
                self.hyperparameter_data['mll'].append(self.mll_val)
            else:
                self.hyperparameter_data['lengthscales'].append(self.gp.lengthscales.tolist())
                self.hyperparameter_data['outputscale'].append(self.gp.outputscale.tolist())
                self.hyperparameter_data['mll'].append(self.mll_val)
            
            if (pt_exists_or_below_threshold and self.mc_points_method == 'NUTS') and (self.mc_samples['method'] == 'MCMC'):
                update_mc = True
                
            ### Update MC ###
            mc_start = time.time()
            if update_mc:
                x0_hmc = self.gp.train_x[jnp.argmax(self.gp.train_y)]
                self.mc_samples = get_mc_samples(
                    self.gp, rng_key=jax.random.PRNGKey(ii*10),
                    warmup_steps=self.num_hmc_warmup, num_samples=self.num_hmc_samples,
                    thinning=1, method=self.mc_points_method,init_params=x0_hmc
                )
                self.mc_samples['method'] = 'MCMC'
            mc_end = time.time()
            self.timing['MC'].append(mc_end-mc_start)
            #################

            if float(new_val) > self.best_f:
                self.best_f = float(new_val)
                self.best_pt = new_pt
                self.best = {name: f"{float(val):.4f}" for name, val in zip(self.loglikelihood.param_list, self.best_pt.flatten())}
                best_pt_iteration = ii
            log.info(f" Current best point {self.best} with value = {self.best_f:.4f}, found at iteration {best_pt_iteration}")

            if i % 4 == 0 and i > 0:
                jax.clear_caches()

            if (ii % 10 == 0) and self.save:
                log.info(" Saving GP to file")
                self.gp.save(outfile=self.output_file)
            ### NS ###
            ns_start = time.time()
            if ns_flag:
                log.info(" Running Nested Sampling")
                ns_samples, logz_dict, ns_success = nested_sampling_Dy(
                    self.gp, self.ndim, maxcall=int(5e6), dynamic=False, dlogz=0.1
                )
                log.info(" LogZ info: " + ", ".join([f"{k}={v:.4f}" for k,v in logz_dict.items()]))
                self.logz_data.append(logz_dict)
                if ns_success and self.check_convergence(i, logz_dict,threshold=self.logz_threshold,ndim=self.ndim):
                    self.converged = True
                    self.termination_reason = "LogZ converged"
                    ns_end = time.time()
                    self.timing['NS'].append(ns_end-ns_start)
                    break
            
                self.mc_samples = ns_samples
                self.mc_samples['method'] = 'NS'
                
            ns_end = time.time()
            self.timing['NS'].append(ns_end-ns_start)
            self.mc_points = get_mc_points(self.mc_samples, self.mc_points_size)

            if self.gp.train_x.shape[0] >= self.max_gp_size:
                self.termination_reason = "Max GP size reached"
                log.info(f" {self.termination_reason}")
                break
            if self.gp.train_x.shape[0] > 1600:
                self.ns_step = 25

        if not self.converged:
            self.gp.fit()

        # Save and final nested sampling
        if self.save:
            self.gp.save(outfile=self.output_file)
        log.info(f" Sampling stopped: {self.termination_reason}")

        log.info(f" Final GP training set size: {self.gp.train_x.shape[0]}, max size: {self.max_gp_size}")
        log.info(f" Number of iterations: {ii}, max iterations: {self.maxiters}")


        # Prepare final results 

        if self.do_final_ns and not self.converged:
            log.info(" Final Nested Sampling")
            ns_samples, logz_dict, ns_success = nested_sampling_Dy(
                self.gp, self.ndim, maxcall=int(1e7), dynamic=True, dlogz=0.01
            )
            log.info(" Final LogZ: " + ", ".join([f"{k}={v:.4f}" for k,v in logz_dict.items()]))
            self.logz_data.append(logz_dict)
        if ns_samples is None:
            log.info("No nested sampling results found, MC samples from HMC/MCMC will be used instead.")
            samples = self.mc_samples['x']
            weights = self.mc_samples['weights']
            loglikes = self.mc_samples['logl']
        else:
            samples = ns_samples['x']
            weights = ns_samples['weights']
            loglikes = ns_samples['logl']
            log.info(f"Using nested sampling results")

        samples = scale_from_unit(samples, self.loglikelihood.param_bounds)
        samples_dict = {
            'x': samples,
            'weights': weights,
            'loglikes': loglikes
        }

        if self.save:
            # np.savez(f'{self.output_file}_samples.npz',
            #          samples=samples,param_bounds=self.loglikelihood.param_bounds,
            #          weights=weights,loglikes=loglikes)
            np.savez(f'{self.output_file}_data.npz',
                     samples=samples,
                     param_bounds=self.loglikelihood.param_bounds,
                     weights=weights,
                     loglikes=loglikes,
                     logz_data=self.logz_data,
                     train_x=self.gp.train_x,
                     train_y=self.gp.train_y,
                     train_y_unstd=self.gp.train_y*self.gp.y_std + self.gp.y_mean,
                     lengthscales=self.hyperparameter_data['lengthscales'],
                     outputscales=self.hyperparameter_data['outputscale'],
                     mll=self.hyperparameter_data['mll'],
                     acq_data_x=self.acq_data['xs'],
                     acq_data_y=self.acq_data['ys'],
                     acq_mcpoints = self.acq_data['mc_points'],
                     acq_point_val = self.acq_data['pt_and_val'],
                     acq_opt_path = self.acq_data['opt_path'],
                     sobol_samples=self.n_sobol_init,
                     allow_pickle=True)

        if self.return_getdist_samples:
            sampler_method = 'nested' if ns_samples is not None else 'mcmc'
            ranges = dict(zip(self.loglikelihood.param_list,self.loglikelihood.param_bounds.T))
            gd_samples = MCSamples(samples=samples, names=self.loglikelihood.param_list, labels=self.loglikelihood.param_labels, 
                                ranges=ranges, weights=weights,loglikes=loglikes,label='GP',sampler=sampler_method)
            output_samples = gd_samples
            log.info(f"Returning getdist samples with method {sampler_method}")
        else:
            output_samples = samples_dict
                
        return self.gp, output_samples, logz_dict

