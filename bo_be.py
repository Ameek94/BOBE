# The main sampler class

import functools
import pandas as pd
import numpy as np
from typing import Any, Callable, List,Optional, Tuple, Dict
from fb_gp import saas_fbgp, sample_GP_NUTS
import time
import jax
import jax.numpy as jnp
from jax import random,vmap, grad
from acquisition import EI, WIPV, Acquisition, optim_bobyqa, optim_scipy_bh 
from scipy.stats import qmc
#from jaxns import NestedSampler
from nested_sampler import NestedSampler, JaxNS, DynestyNS #nested_sampling_jaxns, nested_sampling_Dy
from bo_utils import input_standardize,input_unstandardize, output_standardize, output_unstandardize, plot_gp
import logging
from ext_loglike import external_loglike
from init import input_settings
log = logging.getLogger("[BO]")
np.set_printoptions(precision=4,suppress=False,floatmode='fixed')
jnp.set_printoptions(precision=4,suppress=False,floatmode='fixed')

# todo
# 1. save and resume
# 2. initialize everything from input yaml file and clean up this module
# 3. ?

class sampler:

    # BO settings
    max_steps: int
    acq_goal: float
    acq_batch_size: int
    feedback_lvl: int
    init_strat: str
    ninit: int
    gpfit_step: int
    save: bool
    save_file: str
    save_step: int
    ns_step: int
    seed: int
    Acq: Acquisition
    gp: saas_fbgp

    def __init__(self,
            ndim: Optional[int] = None,
            settings_file: Optional[str] = None,
            cobaya_model = True,
            cobaya_start: int = 8,
            cobaya_input_file: Optional[str] = None,
            resume_from_file=False,
            resume_file = None,
            objfun =  None, # for external functions this should be the logposterior 
            feedback_lvl: int = 1,
            seed: int = 0,) -> None:
        
        # input file with all settings
        if settings_file is not None:
            set_from_file=True
        else:
            set_from_file = False
        inputs = input_settings(set_from_file=set_from_file
                                ,file=settings_file
                                ,input_dict=None)
        self.settings = inputs.settings
        log.info(" Loaded input settings")
        
        self.timing = {'GP': [], 'NS': [], 'ACQ': [], 'Likelihood': []}
        self.integral_accuracy = {'mean': [np.nan], 'upper': [np.nan], 'lower': [np.nan], 'dlogz sampler': [np.nan]}
        self.plot_data = {'acq_check': [], 'outputscale': [], 'lengthscales': [], 'mean': [], 'mll': []} #To store data for summary plot

        # initialize BO settings
        self.init_run_settings()
        # then initialize the model, params, bounds
        if cobaya_model:
            points, lls = self._cobaya_init(cobaya_input_file,cobaya_start=cobaya_start) # type: ignore
        else: # for external model, to clean up
            if isinstance(objfun,external_loglike):
                self.objfun = objfun
                self.param_list = self.objfun.param_list
                self.param_bounds = self.objfun.param_bounds
                self.param_labels = self.objfun.param_labels
                self.logp  = lambda x: self.objfun(input_unstandardize(x,self.param_bounds)) #functools.partial(self._ext_logp,loglike=objfun)
                log.info(f" Using external function {self.objfun.name}")
            else: 
                raise ValueError("Loglike Error")
        self.ndim = len(self.param_list)
        self.bounds_dict = dict(zip(self.param_list,self.param_bounds.T)) # type: ignore
        log.info(f" Running the sampler with the following params {self.param_list}")
        log.info(f" Parameter lower bounds = {dict(zip(self.param_list,self.param_bounds[0]))}")
        log.info(f" Parameter upper bounds = {dict(zip(self.param_list,self.param_bounds[1]))}")
        log.info(f" Parameter labels  = {self.param_labels}")
        self.feedback = feedback_lvl # unused for now

        # set up gp
        rng_key, _ = random.split(random.PRNGKey(seed), 2)
        self.gp_method = self.settings["GP"]["method"]
        self.gp_settings = self.settings["GP"][self.gp_method]
        if resume_from_file:
            assert resume_file is not None
            with np.load(resume_file) as run_data:
                self.train_x = jnp.array(run_data['train_x'])
                self.train_y = jnp.array(run_data['train_y'])
                self.param_bounds = run_data['param_bounds']
                lengthscales = jnp.array(run_data["lengthscales"])
                outputscales = jnp.array(run_data["outputscales"])
            self.gp = saas_fbgp(self.train_x,self.train_y,
                                sample_lengthscales=lengthscales,sample_outputscales=outputscales,
                                num_chains=jax.device_count(), **self.gp_settings) 
            log.info(f"Resuming from file {resume_file} with {self.train_x.shape[0]} previous points")
        else:
            self.train_x = qmc.Sobol(self.ndim, scramble=True,seed=seed).random(self.ninit)
            self.train_y = np.reshape(self.logp(self.train_x),(self.ninit,1))
            if cobaya_model:
                for pt,ll in zip(points,lls): 
                    pt_unit = input_standardize(np.reshape(pt,(1,self.ndim)),self.param_bounds)
                    if pt_unit not in self.train_x:
                        log.info(" Adding cobaya point")
                        log.info("".join(f" {key} = {val:.4f}, " for key,val in dict(zip(self.param_list,pt)).items()) )
                        self.train_x = np.concatenate([self.train_x,pt_unit])
                        self.train_y = np.concatenate([self.train_y,np.atleast_2d(ll)])
            log.info(f" Initial loglikes \n{self.train_y.T}")
            log.info(f" Sampler will start with {len(self.train_y)} points and run for a maximum of {self.max_steps} steps")
            self.gp = saas_fbgp(self.train_x,self.train_y,
                                num_chains=jax.device_count(), **self.gp_settings)             
            self.gp.fit(rng_key)
            log.info(f" Initialized {self.gp_method} GP with settings {self.gp_settings}")

        # once GP is initialized, we can now initialize the acquisition function and its optimizers
        self.init_aquisition(rng_key)

        # initialise NS settings
        self.init_nestedSampler()
        self.run_nested_sampler = -1
        self.final_ns_samples = None

    def run(self):
        """
        Run the BO loop until convergence or max steps reached. 
        """        
        num_step = 0
        self.converged = False
        start = time.time()
        while not self.converged:
            seed = num_step
            rng_key, _ = random.split(random.PRNGKey(seed), 2)
            ###  Acquisition Function  ###
            start_a = time.time()
            max_idx = np.argmax(self.train_y)
            x0 = self.train_x[max_idx]
            pt,val = self.acq.optimize(x0)
            self.acq_val = abs(val)
            self.plot_data['acq_check'].append(self.acq_val)
            next_x = jnp.atleast_2d(pt)
            end_a = time.time()
            self.timing['ACQ'].append(end_a-start_a)
            log.info(f" Acquisition Optimisation took {self.timing['ACQ'][-1]:.4f} s")
            log.info(f" Next point at x = {self._print_point(pt)} with acquisition function value = {val:.4e}")
            #############################
            ###   Evaluate New Point  ###
            start_l = time.time()
            next_y = self.logp(next_x)
            end_l = time.time()
            self.timing['Likelihood'].append(end_l-start_l)
            log.info(f" Likelihood evaluation took {self.timing['Likelihood'][-1]:.4f} s")
            log.info(f" Current best loglike = {self.train_y[max_idx]} at {self._print_point(self.train_x[max_idx])} \nLoglike at new point = {next_y}")
            #############################
            if (num_step%4==0 and num_step>0):
                jax.clear_caches() # hack for managing memory until I implement GP pytree 
            ###       Train GP        ###
            start_gp = time.time()
            self.train_x = jnp.concatenate([self.train_x,next_x])
            self.train_y = jnp.concatenate([self.train_y,next_y])
            if (num_step%self.gpfit_step==0):
                self.gp.update(next_x,next_y,rng_key)
                self.acq.update_mc_points(rng_key)
            else:
                self.gp.quick_update(next_x,next_y)
            end_gp = time.time()
            self.timing['GP'].append(end_gp-start_gp)
            self.plot_data['mll'].append(self.gp.samples['minus_log_prob']) 
            self.plot_data['lengthscales'].append(self.gp.get_median_lengthscales()) 
            self.plot_data['outputscale'].append(self.gp.get_median_outputscales())
            log.info(f" GP Training took {self.timing['GP'][-1]:.4f} s")
            #############################
            ###    Nested Sampling    ###
            start_ns = time.time()
            if self.run_nested_sampler > 0:
                if (num_step - self.run_nested_sampler)%self.ns_step==0:
                    _, logz_dict = self.NestedSampler.run(final_run=False)
                    log.info(f"Current evidence estimate: {logz_dict['mean']:.4f} ± {(logz_dict['upper'] - logz_dict['lower'])/2 + logz_dict['dlogz sampler']:.4f}")
                    log.info(f"Mean: {logz_dict['mean']:.4f}, Upper Bound: {logz_dict['upper']:.4f}, Lower Bound: {logz_dict['lower']:.4f}")
                else:
                    logz_dict = {'mean': self.integral_accuracy['mean'][-1], 'upper': self.integral_accuracy['upper'][-1], 'lower': self.integral_accuracy['lower'][-1], 'dlogz sampler': self.integral_accuracy['dlogz sampler'][-1]}
            else:
                logz_dict = {'mean': self.integral_accuracy['mean'][-1], 'upper': self.integral_accuracy['upper'][-1], 'lower': self.integral_accuracy['lower'][-1], 'dlogz sampler': self.integral_accuracy['dlogz sampler'][-1]}
                
            self.integral_accuracy = {key: self.integral_accuracy[key] + [logz_dict[key]] for key in self.integral_accuracy.keys()}
            end_ns = time.time()
            self.timing['NS'].append(end_ns-start_ns)
            log.info(f" Nested Sampling took {self.timing['NS'][-1]:.4f} s")
            #############################
                
            log.info(f" --------------------Step {num_step+1} completed in {sum(values[-1] for values in self.timing.values()):.4f}s-------------------\n")
            
            # check convergence
            num_step+=1
            self.converged = self._check_converged(num_step)

            # save if needed
            if ((num_step%self.save_step==0 and self.save) or self.converged):
                self.gp.save(self.save_file)
                log.info(f" Run training data and hyperparameters saved at step {num_step}")

        samples, logz_dict, self.final_ns_samples = self.NestedSampler.run(final_run=True)
        log.info(f" Final LogZ info: "+"".join(f"{key} = {value:.4f}, " for key, value in logz_dict.items()))
        log.info(" Run Completed")
        final_logz = logz_dict['mean']
        final_dlogz = (logz_dict['upper'] - logz_dict['lower'])/2
        final_dlogz_err = logz_dict['dlogz sampler']
        log.info(f" BO took {time.time() - start:.2f}s took {self.ninit+self.acq_batch_size*num_step} samples for a final evidence of {final_logz:.4f} ± {final_dlogz:.4f} ± {final_dlogz_err:.4f}")
        samples = input_unstandardize(samples,self.param_bounds)
        np.savez(self.save_file+'_samples.npz',*samples)


    def _check_converged(self,num_step):
        acq = (self.acq_val < self.acq_goal)
        ns_converged = (self.integral_accuracy['upper'][-1] - self.integral_accuracy['lower'][-1] < self.precision_goal)
        steps = (num_step >= self.max_steps)
        if acq:
            self.run_nested_sampler = num_step
            if ns_converged:
                log.info(" Acquisition goal reached")
        if steps:
            log.info(" Max steps reached")
        return ns_converged or steps

    def _ext_logp(self,x,loglike):
        x  = input_unstandardize(x,self.param_bounds)
        return jnp.atleast_2d(loglike(x))

    def _cobaya_logp(self, x): #logposterior for cobaya likelihoods
        # X should be a N x DIM  with parameters in the same order as the param_list in range [0,1]^DIM
        pdf = []
        x =  input_unstandardize(x,self.param_bounds)
        for point in x: # can parallelize evaluation of likelihood by splitting x into nproc parts
            logpost = self.cobaya_model.logpost(point,make_finite=True) 
            pdf.append(logpost)
        return np.array(pdf).reshape(-1,1)

    def _cobaya_init(self,input_file,cobaya_start=8):
        try:
            from cobaya.yaml import yaml_load
            from cobaya.model import get_model
            self.logp = self._cobaya_logp
            assert input_file is not None
            info = yaml_load(input_file)
            self.cobaya_model = get_model(info) # type: ignore  #note that model is already a cobaya object
            rootlogger = logging.getLogger() 
            rootlogger.handlers.pop()    
            self.param_list = list(self.cobaya_model.parameterization.sampled_params()) # how do we deal with derived parameters
            self.param_bounds = np.array(self.cobaya_model.prior.bounds(confidence_for_unbounded=0.95)).T
            self.param_labels = [self.cobaya_model.parameterization.labels()[key] for key in self.param_list]
            points = []
            logpost = []
            for i in range(cobaya_start):
                res = self.cobaya_model.get_valid_point(100, ignore_fixed_ref=False
                                                             , logposterior_as_dict=True
                                                             , random_state=None)
                points.append(res[0])
                logpost.append(res[1]['logpost']) # type: ignore
            return np.array(points), np.array(logpost)
        except ModuleNotFoundError:
            log.error(" Cobaya not found")

    
    def _print_point(self,x):
        x = input_unstandardize(x,self.param_bounds)
        return dict(zip(self.param_list,np.array(x)))

    def init_run_settings(self):
        method = self.settings['BO']['method']
        for key,val in self.settings['BO'][method].items():
            setattr(self,key,val)
            
    def init_nestedSampler(self):
        ns_method = self.settings['NS']['method']
        ns_kwargs = self.settings['NS'][ns_method]
        if ns_method=="jaxns":
            self.NestedSampler = JaxNS(gp=self.gp
                                      ,ndim=self.ndim,
                                       ns_kwargs=ns_kwargs)
        elif ns_method=="dynesty":
            self.NestedSampler = DynestyNS(gp=self.gp
                                      ,ndim=self.ndim,
                                       ns_kwargs=ns_kwargs)

        else:
            raise ValueError('Not a valid Nested Sampler')

    def init_aquisition(self,rng_key):
        # acqs = {EI: "EI", WIPV: "WIPV"}
        acq_method = self.settings['ACQ']['method']
        acq_settings = self.settings['ACQ'][acq_method]
        optimizer_settings = self.settings['optimizer']
        if acq_method=="WIPV":
            self.acq = WIPV(gp=self.gp
                            ,rng_key=rng_key
                            ,ndim=self.ndim
                            ,batch_size=self.acq_batch_size
                            ,optimizer_settings=optimizer_settings
                            ,mcmc_kwargs=acq_settings)
        elif acq_method=="EI":
             self.acq = EI(gp=self.gp
                            ,rng_key=rng_key
                            ,ndim=self.ndim
                            ,batch_size=self.acq_batch_size
                            ,optimizer_settings=optimizer_settings
                            ,ei_kwargs=acq_settings)       
        else:
            raise ValueError('Not a valid acquisition function')
            
    def plot_summary(self, save_file=None, posterior=False, posterior_save_file=None):
        import matplotlib
        import matplotlib.pyplot as plt
        from plot_utils import acq_check_metric_plot, FBGP_hyperparameter_plot, FBGP_mll_plot, integral_accuracy_plot, integral_metrics_plot, timing_plot
        import getdist
        from getdist import MCSamples, plots
        fig,ax = plt.subplots(3, 3, figsize=(25, 30))
        #fig.delaxes(ax[3,0])
        #fig.delaxes(ax[3,2])
        num_step = self.train_x.shape[0] - self.ninit  #Add batch size
        fig.suptitle(f"Iteration: {num_step}, #Samples: {self.ninit+self.train_x.shape[0]}")
        self.ax = ax
        self.fig = fig
        self.ax = acq_check_metric_plot(self, num_step)
        self.ax = FBGP_hyperparameter_plot(self, num_step)
        self.ax = FBGP_mll_plot(self, num_step)
        if self.run_nested_sampler:
            self.ax = integral_accuracy_plot(self, num_step)
            self.ax = integral_metrics_plot(self, num_step)
        self.ax = timing_plot(self, num_step)
        plt.tight_layout()
        plt.subplots_adjust(
                    top=0.95,
                    wspace=0.2, 
                    hspace=0.2)
        if save_file != None:
            log.info(f" Saving Plot to {save_file}.png")
            fig.savefig(f"{save_file}.png")    
        else:
            plt.show()
        plt.close(fig)
        
        if posterior:
            fig = plt.figure()
            g = MCSamples(samples=self.final_ns_samples) #, labels=self.param_labels)
            g.plot_triangle(show_titles=True, contour_colors='blue', filled=True)
            if posterior_save_file != None:
                plt.savefig(f"{posterior_save_file}.png")
            else:
                g.show()
        plt.close(fig)
        
    