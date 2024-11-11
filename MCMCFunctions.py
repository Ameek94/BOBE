### Acquisition Functions and Optimisers ###
from GPUtils import *
import torch
from torch import Tensor
import numpy as np
from typing import Optional, Dict, Callable
import functools
from dynesty import NestedSampler,DynamicNestedSampler
import pybobyqa
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

def mcpoints_for_acquisition(samples, nested_samples_size, curr_step) -> Tensor:
        """ 
        Convert the full mc samples to torch and thin them, 
        later can also shuffle them to reuse - avoids giving the same points to the acquisition even when not regenerating mc_points
        """
        size = np.shape(samples)[0]
        with torch.no_grad(): 
            if size>=nested_samples_size:
                step = int(size/nested_samples_size)
                start_idx = 2*curr_step % size # shuffle mc_points when reusing + make sure to keep within array bounds
                return torch.tensor(samples[start_idx::step],**tkwargs)
            else:
                return torch.tensor(samples,**tkwargs)


def likelihood_for_mcmc(gp, 
                        x: np.ndarray | list[float], 
                        #temp: float = 1, 
                        interp_logp: bool = True, 
                        upper_lower: bool = True):
        """
		Wrapper function to allow nested sampler to evalute on the surrogate posterior of the FBGP
		Allows for error checking (ensuring no Nans or Infs) as well as calculation of standard deviation
		"""
        with torch.no_grad():
            y = gp.posterior(torch.tensor(np.atleast_2d(x),**tkwargs))                                          #Evaluate posterior of FBGP at given parameter values
            res = y.mixture_mean.squeeze(-1).squeeze(-1).cpu().numpy()                                          #Get the mixture mean value at these points
            result = np.array(res,dtype="float64") #temp*                                                         #Ensure our result is given with as much precision as possible
        if interp_logp:                                                                                         #If the likelihood is given as a loglikelihood
            if upper_lower:                                                                                     #If we want to calculate upper and lower bounds at each point
                logl_upper = y.mvn.confidence_region()[1].mean(dim=-2).squeeze(0).cpu().numpy() #temp*           #Get upper confidence region (Could be replaced with: + 1.96 * std_FBGP)
                logl_lower = y.mvn.confidence_region()[0].mean(dim=-2).squeeze(0).cpu().numpy() #temp*            #Get lower confidence region (Could be replaced with: - 1.96 * std_FBGP)
                del y
                blob = np.zeros(2)
                blob[0] = result + (logl_lower - result)/2 # get 1-sigma region
                blob[1] = result - (result - logl_upper)/2
                del logl_lower,logl_upper 
                return result, blob # return only stdev to reduce computational cost
            else:
                del y
                return result 
        else:
            # if GPR on exp(loglike), then need to return log(res)
            result[result<0]= 1e-300 # 
            return np.log(result)

# This is the dynesty function for computing logz and associated quantities, we only need logz
def compute_integrals(logl=None, logvol=None, reweight=None):
        """
        Computes value of logz (mean, upper and lower bounds)
        """
        assert logl is not None
        assert logvol is not None
        loglstar_pad = np.concatenate([[-1.e300], logl])
        # we want log(exp(logvol_i)-exp(logvol_(i+1)))
        # assuming that logvol0 = 0
        # log(exp(LV_{i})-exp(LV_{i+1})) =
        # = LV{i} + log(1-exp(LV_{i+1}-LV{i}))
        # = LV_{i+1} - (LV_{i+1} -LV_i) + log(1-exp(LV_{i+1}-LV{i}))
        dlogvol = np.diff(logvol, prepend=0)
        logdvol = logvol - dlogvol + np.log1p(-np.exp(dlogvol))
        # logdvol is log(delta(volumes)) i.e. log (X_i-X_{i-1})
        logdvol2 = logdvol + math.log(0.5)
        # These are log(1/2(X_(i+1)-X_i))
        dlogvol = -np.diff(logvol, prepend=0)
        # this are delta(log(volumes)) of the run
        # These are log((L_i+L_{i_1})*(X_i+1-X_i)/2)
        saved_logwt = np.logaddexp(loglstar_pad[1:], loglstar_pad[:-1]) + logdvol2
        if reweight is not None:
            saved_logwt = saved_logwt + reweight
        saved_logz = np.logaddexp.accumulate(saved_logwt)
        return saved_logz


def nested_sampling_Dy(gp
                       ,ndim: int = 2
                       #,temp: float = 1
                       ,dlogz: float = 0.001
                       ,dynamic: bool = True
                       ,interp_logp: bool = True
                       ,upper_lower: bool = True
                       ,bounds: np.ndarray = None
                       ,maxcall: Optional[int] = None
                      ,nthreads: int = 12) -> tuple[np.ndarray,Dict]:
	
    if maxcall is None:
        if ndim<=4:
            maxcall = int(100000*ndim)
        else:
            maxcall = max(int(6000*ndim),60000)
    else:
         maxcall = int(maxcall)
    
    #loglike = functools.partial(likelihood_for_mcmc, gp=gp, temp=temp, interp_logp=interp_logp, upper_lower=upper_lower)
    loglike = lambda x: likelihood_for_mcmc(gp, x=x, interp_logp=interp_logp,upper_lower=upper_lower) #temp=temp,
    if dynamic:
        sampler = DynamicNestedSampler(loglike,prior_transform,ptform_kwargs={'bounds': bounds},ndim=ndim,blob=upper_lower, bootstrap=0)#, pool=pool, queue_size=nthreads, bootstrap=0)
        sampler.run_nested(print_progress=True,dlogz_init=dlogz,maxcall=maxcall) #tune? ,maxcall=20000
    else:
        sampler = NestedSampler(loglike, prior_transform, ptform_kwargs={'bounds': bounds},ndim=ndim,blob=upper_lower, bootstrap=0)#, pool=pool, queue_size=nthreads, bootstrap=0)
        sampler.run_nested(print_progress=True,dlogz=dlogz)#,maxcall=maxcall
       
    res = sampler.results  # grab our results
    logl = res['logl']
   
    logl_lower,logl_upper = res['blob'].T
    logvol = res['logvol']
    logl = res['logl']
	
    upper = compute_integrals(logl=logl_upper,
							  logvol=logvol)
    
    lower = compute_integrals(logl=logl_lower,
							  logvol=logvol)
    
    mean = np.float64(res['logz'][-1])
    logz_err = res['logzerr'][-1]
    logz_dict = {'mean': mean,'dlogz sampler': logz_err, 'upper': upper[-1], 'lower': lower[-1]}
    samples = res.samples_equal()
    return samples, logz_dict
    
# optimize the acquisition function with Py-BOBYQA
def BOBYQA_optim(acq_func: Callable[[Tensor], Tensor]
                 ,x0: np.ndarray
                 ,dim: int = 1
                 ,batch_size: int = 1) -> tuple[Tensor,Tensor]:
    # set up bounds for the solver
    upper = np.ones(dim*batch_size)
    lower = np.zeros_like(upper)
    # convert acq_func output to numpy output
    def wrap_acq_func(x):
        with torch.no_grad():
            x = np.array(x).reshape(batch_size,dim) # optimizer cannot do joint optimization but can get batched points by converting to batch_size*ndim shaped arrays
            X = torch.tensor(x,**tkwargs)        
            Y = -acq_func(X)
            #print("Acq debug: ", Y)
            del X
            y = Y.view(-1).double().numpy()
            return y
    # run the solver
    soln = pybobyqa.solve(wrap_acq_func,x0,bounds=(lower,upper)
                          ,seek_global_minimum=True,print_progress=False,do_logging=False, maxfun=10000)
    #print(soln)
    xs = soln.x # xs is 1D array of size dim*batch_size
    best_x  = np.array(xs).reshape(batch_size,dim)
    best_x = torch.tensor(best_x,**tkwargs)
    best_val = torch.tensor(soln.f,**tkwargs)
    return best_x,best_val
    
