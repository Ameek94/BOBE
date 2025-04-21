import time
from typing import Any,List, Optional
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm
from jax import config
config.update("jax_enable_x64", True)
import logging
log = logging.getLogger("[AQ]")

#------------------The acqusition functions-------------------------

def EI(x,gp,zeta=0.):
    mu, var = gp.predict(x)
    std = jnp.sqrt(var)
    best_f = gp.train_y.max()
    z = (mu - best_f - zeta) / std
    ei = std * (z*norm.cdf(z) + norm.pdf(z))
    return jnp.reshape(-ei,()) 

def WIPV(x, gp, mc_points):
    var = gp.fantasy_var(x, mc_points=mc_points)
    return jnp.mean(var)

# def logEI