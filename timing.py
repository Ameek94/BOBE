from fb_gp import saas_fbgp
import numpy as np
import time
import pyro
import torch
import sys
import jax.numpy as jnp
from jax import random


np.random.seed(10004118) # fixed for reproducibility
train_x = np.random.uniform(0,1,4).reshape(-1, 1)
f = lambda x: -0.5*10*(x-0.5)**2 #-0.5*25*jnp.cos(20*(x-0.5)**2)  # 

train_y = f(train_x)
test_x =  np.linspace(0,1,1000).reshape(-1, 1)
test_y = f(test_x)

train_yvar = 1e-6*jnp.ones_like(train_y)


warmup_steps = int(sys.argv[1])
num_steps = int(sys.argv[2])
thinning = int(sys.argv[3])

print(f"MC points size = {int(num_steps/thinning)}, test points size = {test_x.shape[0]}")


print("Testing lightweight implementation")

gp = saas_fbgp(train_x,train_y)
seed = 0
rng_key, _ = random.split(random.PRNGKey(seed), 2)
gp.fit(rng_key,warmup_steps=warmup_steps,num_samples=num_steps,thinning=16)

start = time.time()
mu, var = gp.predict(test_x,single=False)

print(f"predicting took {time.time() - start:.4f}s\n")
