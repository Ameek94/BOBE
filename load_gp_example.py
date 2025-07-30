#!/usr/bin/env python3
"""
Example script demonstrating how to save and load GP models
"""

import jax.numpy as jnp
import numpy as np
from jaxbo.gp import DSLP_GP, SAAS_GP, load_gp
from jaxbo.svm_gp import SVM_GP
from jaxbo.logging_utils import get_logger

log = get_logger("[Load GP Example]")

def create_sample_data(n_samples=50, n_dims=2, seed=42):
    """Create sample training data"""
    np.random.seed(seed)
    X = np.random.uniform(-1, 1, (n_samples, n_dims))
    # Simple test function: sum of squares
    y = -np.sum(X**2, axis=1, keepdims=True) + 0.1 * np.random.randn(n_samples, 1)
    return jnp.array(X), jnp.array(y)

def example_dslp_gp():
    """Example of saving and loading a DSLP_GP"""
    log.info("=== DSLP_GP Example ===")
    
    # Create sample data
    X, y = create_sample_data()
    
    # Create and fit DSLP_GP
    log.info("Creating and fitting DSLP_GP...")
    gp = DSLP_GP(X, y)
    gp.fit(maxiter=50)  # Reduce iterations for quick example
    
    # Save the GP
    log.info("Saving DSLP_GP...")
    gp.save('example_dslp_gp')
    
    # Load the GP
    log.info("Loading DSLP_GP...")
    loaded_gp = DSLP_GP.load('example_dslp_gp.npz')
    
    # Verify they produce the same predictions
    test_x = jnp.array([[0.5, 0.5]])
    orig_mean = gp.predict_mean(test_x)
    loaded_mean = loaded_gp.predict_mean(test_x)
    
    log.info(f"Original GP prediction: {orig_mean}")
    log.info(f"Loaded GP prediction: {loaded_mean}")
    log.info(f"Difference: {jnp.abs(orig_mean - loaded_mean)}")

def example_saas_gp():
    """Example of saving and loading a SAAS_GP"""
    log.info("=== SAAS_GP Example ===")
    
    # Create sample data
    X, y = create_sample_data()
    
    # Create and fit SAAS_GP
    log.info("Creating and fitting SAAS_GP...")
    gp = SAAS_GP(X, y)
    gp.fit(maxiter=50)  # Reduce iterations for quick example
    
    # Save the GP
    log.info("Saving SAAS_GP...")
    gp.save('example_saas_gp')
    
    # Load the GP
    log.info("Loading SAAS_GP...")
    loaded_gp = SAAS_GP.load('example_saas_gp.npz')
    
    # Verify they produce the same predictions
    test_x = jnp.array([[0.5, 0.5]])
    orig_mean = gp.predict_mean(test_x)
    loaded_mean = loaded_gp.predict_mean(test_x)
    
    log.info(f"Original GP prediction: {orig_mean}")
    log.info(f"Loaded GP prediction: {loaded_mean}")
    log.info(f"Difference: {jnp.abs(orig_mean - loaded_mean)}")

def example_auto_load():
    """Example of using the auto-loading utility function"""
    log.info("=== Auto-Load Example ===")
    
    # Load GPs using the auto-detection utility
    log.info("Auto-loading DSLP_GP...")
    dslp_gp = load_gp('example_dslp_gp.npz')
    log.info(f"Loaded GP type: {type(dslp_gp).__name__}")
    
    log.info("Auto-loading SAAS_GP...")
    saas_gp = load_gp('example_saas_gp.npz')
    log.info(f"Loaded GP type: {type(saas_gp).__name__}")

def example_svm_gp():
    """Example of saving and loading an SVM_GP"""
    log.info("=== SVM_GP Example ===")
    
    # Create larger sample data for SVM_GP (needs more points)
    X, y = create_sample_data(n_samples=500, seed=42)
    
    # Create SVM_GP
    log.info("Creating SVM_GP...")
    svm_gp = SVM_GP(train_x=X, train_y=y, svm_use_size=400, lengthscale_priors='DSLP')
    
    # Fit the underlying GP
    log.info("Fitting SVM_GP...")
    svm_gp.fit(maxiter=50)
    
    # Save the SVM_GP  
    log.info("Saving SVM_GP...")
    svm_gp.save('example_svm_gp')
    
    # Load the SVM_GP
    log.info("Loading SVM_GP...")
    loaded_svm_gp = SVM_GP.load('example_svm_gp.npz', lengthscale_priors='DSLP')
    
    # Verify they produce the same predictions
    test_x = jnp.array([[0.5, 0.5]])
    orig_mean = svm_gp.predict_mean(test_x)
    loaded_mean = loaded_svm_gp.predict_mean(test_x)
    
    log.info(f"Original SVM_GP prediction: {orig_mean}")
    log.info(f"Loaded SVM_GP prediction: {loaded_mean}")
    log.info(f"Difference: {jnp.abs(orig_mean - loaded_mean)}")

if __name__ == "__main__":
    log.info("GP Save/Load Examples")
    log.info("====================")
    
    # Run examples
    example_dslp_gp()
    example_saas_gp()
    example_auto_load()
    example_svm_gp()
    
    log.info("All examples completed!")
