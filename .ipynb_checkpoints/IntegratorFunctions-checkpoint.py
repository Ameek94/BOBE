import numpy as np
import torch
from dynesty import NestedSampler,DynamicNestedSampler
from dynesty import utils as dyfunc
import GPUtils
import MCMCFunctions
from scipy import integrate
import functools

tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}
import logging
log = logging.getLogger("[INT UTILS]")

def dblquad_likelihood(FBGP, x, *args):
    FBGP = args[-1]
    X = torch.tensor([x, *args[0:-1]], **tkwargs)
    y = FBGP.posterior(torch.tensor(X,**tkwargs).unsqueeze(0))
    output = np.exp(y.mixture_mean.squeeze(-1).squeeze(-1).detach().numpy())
    return output

def calc_prior_fac(bounds):
    prior_fac = 1
    for i in range(len(bounds)):
        prior_fac *= np.abs(bounds[i][0] - bounds[i][1])
    return np.log(prior_fac)

def integrate_likelihood(x, *args, logp): #x = None, y = None
    if len(args) != 0 or type(x).__name__ == 'float': #and len(x) != 0:        
        X = np.array([x, *args])
        X = np.expand_dims(X, -2)
        output = np.exp(logp(X))[0]
    elif len(args) == 0: #and len([x]) != 0:
        x = np.array(x)
        #print(x)
        output = logp(x).squeeze(-1)
    else:
        print("Number of args: ", len(args))
        print("Args: ", args)
    return output


def dynesty_true_integral(bounds, ndim, dlogz, logp, prior_fac):
    ### Dynesty ###
    ptform_kwargs={'bounds': bounds}
    integrate_likelihood_dy = functools.partial(integrate_likelihood, logp=logp)
    sampler = NestedSampler(integrate_likelihood_dy, GPUtils.prior_transform, ptform_kwargs=ptform_kwargs,
                       nlive=1000, ndim=ndim, 
                       blob=False )#, pool=pool, queue_size=nthreads)
    sampler.run_nested(print_progress=True, dlogz=1e-4)
    
    res = sampler.results
    logz_dy = res.logz[-1] + 2*prior_fac #ln(z_dynesty) + ln(prior_fac) = z_dynesty*prior_fac = Evidence -> Why do we have to do 2xprior_fac?
    logzerr_dy = res.logzerr[-1]
    computed_integrals = dyfunc.compute_integrals(res.logl, res.logvol)
    #log.info(computed_integrals[1][-1], computed_integrals[2][-1])
    log.info(f"Logz from Dynesty = {logz_dy} +- {logzerr_dy} with {len(res.ncall)} samples")

    return logz_dy, logzerr_dy


def dblquad_true_integral(bounds, prior_fac, logp):
    ### Dbl Quad ### 
    #We add a factor of prior_fac here as well as dbl quad thinks it's integrating over the unit bounds#
    #z_dbl, zerr_dbl = integrate.dblquad(integrate_likelihood,0,1,0,1,epsrel=1e-6)
    integrate_likelihood_nquad = functools.partial(integrate_likelihood, logp=logp)
    z_dbl, zerr_dbl, out_dict = integrate.nquad(integrate_likelihood_nquad, ranges=bounds,  opts={'epsrel': 1e-6}, full_output=True)
    logz_dbl = np.log(z_dbl)
    logz_dbl += prior_fac
    logzerr_dbl = zerr_dbl/z_dbl #Because y=ln(x): ∆y = dy/dx*∆x = ∆x/x
    log.info(f"LogZ from direct integration = {logz_dbl} +- {logzerr_dbl} with {out_dict['neval']} samples")
    return logz_dbl, logzerr_dbl
################