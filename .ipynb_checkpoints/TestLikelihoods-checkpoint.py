### Likelihood and associated parameters ###
import torch
from torch import Tensor
from botorch.utils.transforms import normalize, unnormalize
import numpy as np
import jax.numpy as jnp

tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

def ext_logp(X, loglike, interp_logp, bounds): # logposterior for external likelihoods, takes input in [0,1] then calls the user defined function 
        # Internally X should be a N x DIM tensor with parameters in the same order as the param_list in range [0,1]^DIM
        # So we need to unnormalize x if physical range is not in [0,1]
        x =  unnormalize(X, bounds.t())
        logpdf = np.expand_dims(loglike(x), -1) #output should be N x 1    
        if interp_logp: # if GPR on loglikelihood
            return logpdf 
        else: # if GPR on exp(loglikelihood)
            return np.exp(logpdf)


def gaussian(X, dynesty=False, plot=False):
        # X is a N x DIM shaped tensor, output is N tensor
        mean = jnp.array(0.5) #len(param_list)*
        sigma = jnp.array(0.1) #len(param_list)*
        if dynesty:
            return -0.5*jnp.sum((X-mean)**2/sigma**2, axis=-1, keepdims=False)
        elif plot:
            return -0.5*jnp.sum((X-mean)**2/sigma**2, axis=-1, keepdims=False)
        else:
            return (-0.5*jnp.sum((X-mean)**2/sigma**2, axis=-1, keepdims=False))
        

def gaussian_ring_torch(X: Tensor) -> Tensor: #Bounds: [[-1, 1]]
    mean_r = 0.4
    scale = 0.08
    r2 = X[:, 0]**2 + X[:, 1]**2
    r = np.sqrt(r2)
    return torch.tensor(-0.5*((r-mean_r)/scale)**2)

def gaussian_ring(X): #Bounds: [[-1, 1]]
    mean_r = 0.4
    scale = 0.08
    r2 = X[..., 0]**2 + X[..., 1]**2
    r = np.sqrt(r2)
    return -0.5*((r-mean_r)/scale)**2
    

def eggbox_torch(X: Tensor) -> Tensor: #Bounds: [[0, 1]]
    scale_fac = 4.
    scale = scale_fac*np.pi
    pow = 3
    
    y = 2 + torch.cos(scale*X[...,0])*torch.cos(scale*X[...,1]) 
    return y**(pow)

def eggbox(X): #Bounds: [[0, 1]]
    scale_fac = 4.
    scale = scale_fac*np.pi
    pow = 3
    
    y = 2 + np.cos(scale*X[...,0])*np.cos(scale*X[...,1]) 
    return y**(pow)

def banana(X): #[[-1, 1], [-1, 2]]
    #a= 0.2
    #b= 20
    #logpdf = (a - X[:, 0])**2 + b*(X[:, 1] - X[:, 0]**2)**2
    logpdf = -0.25*(5*(0.2-X[...,0]))**2 - (20*(X[...,1]/4 - X[...,0]**4))**2
    return logpdf


def himmelblau(X: Tensor) -> Tensor:
    afac= 0.1
    r1 = (X[:,0] + X[:,1]**2 -7)**2
    r2 = (X[:,0]**2 + X[:,1]-11)**2
    return -0.5*(afac*r1 + r2)

def ackley(X: Tensor) -> Tensor:
    r1 = -20*np.exp(-0.2*np.sqrt(0.5*(X[:, 0]**2 + X[:, 1]**2)))
    r2 = -np.exp(0.5*(np.cos(2*(np.pi*X[:, 0])) + np.cos(2*(np.pi*X[:, 1])))) + np.e + 20
    return np.log((r1 + r2)+1)

def beale(X: Tensor) -> Tensor:
    r1 = (1.5 - X[:, 0] + X[:, 0]*X[:, 1])**2
    r2 = (2.25 - X[:, 0] + X[:, 0]*X[:, 1]**2)**2
    r3 = (2.625 - X[:, 0] + X[:, 0]*X[:, 1]**3)**2
    return np.log(r1 + r2 + r3)




test_fnc_param_bounds = {'gaussian': [[0, 1], [0, 1]], #
                         'gaussian_ring': [[-1, 1], [-1, 1]], 
                         'eggbox': [[0, 1], [0, 1]], 
                         'banana': [[-1, 1], [-1, 2]], 
                         'himmelblau': [[-6, 6], [-6, 6]], 
                         'ackley': [[-4, 4], [-4, 4]]}

