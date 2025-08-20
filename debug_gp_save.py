#!/usr/bin/env python3
"""
Debug script to check what gets saved in the GP files
"""
import sys
sys.path.insert(0, '/Users/amkpd/cosmocodes/JaxBo')

import numpy as np
import jax.numpy as jnp
from jaxbo.clf_gp import GPwithClassifier

# Create GPwithClassifier with SAAS
train_x = jnp.array(np.random.uniform(0, 1, (50, 3)))
train_y = jnp.array(np.random.normal(0, 1, (50, 1)))

gp_clf = GPwithClassifier(
    train_x=train_x, 
    train_y=train_y, 
    lengthscale_priors='SAAS',
    clf_type='svm',
    clf_use_size=30,
    lengthscales=jnp.ones(3), 
    outputscale=1.0,
    tausq=2.0
)

print(f"GP type: {type(gp_clf.gp).__name__}")
print(f"GP hyperparam_priors: {repr(gp_clf.gp.hyperparam_priors)}")

# Save it
gp_clf.save("debug_clf_gp")

# Check what got saved
data = np.load("debug_clf_gp.npz", allow_pickle=True)
print("\nSaved data keys:", list(data.files))
if 'hyperparam_priors' in data.files:
    print(f"Saved hyperparam_priors: {repr(data['hyperparam_priors'])}")
    print(f"Type: {type(data['hyperparam_priors'])}")
if 'tausq' in data.files:
    print(f"Saved tausq: {data['tausq']}")
