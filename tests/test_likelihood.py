"""
Tests for Likelihood classes.

Tests include:
- ExternalLikelihood initialization and evaluation
- CobayaLikelihood initialization (if Cobaya available)
- Batch evaluation
- Initial point generation with Sobol
- Error handling (NaN, inf, exceptions)
- Parameter bounds validation
"""

import numpy as np
import pytest
import sys
from jaxbo.likelihood import BaseLikelihood, ExternalLikelihood
from jaxbo.utils.pool import MPI_Pool


def simple_loglike(x):
    """Simple quadratic log-likelihood."""
    return -np.sum((x - 0.5)**2)


def nan_loglike(x):
    """Log-likelihood that returns NaN."""
    return np.nan


def exception_loglike(x):
    """Log-likelihood that raises an exception."""
    raise ValueError("Test exception")


def test_external_likelihood_initialization():
    """Test ExternalLikelihood initialization."""
    print("\n" + "="*80)
    print("TEST: ExternalLikelihood Initialization")
    print("="*80)
    
    pool = MPI_Pool()
    
    likelihood = ExternalLikelihood(
        loglikelihood=simple_loglike,
        param_list=['x', 'y', 'z'],
        param_bounds=np.array([(0, 1), (-5, 5), (0, 10)]).T,
        param_labels=['X', 'Y', 'Z'],
        pool=pool,
        name="test_likelihood"
    )
    
    assert likelihood.ndim == 3, f"Expected ndim=3, got {likelihood.ndim}"
    assert likelihood.name == "test_likelihood"
    assert len(likelihood.param_list) == 3
    assert likelihood.param_bounds.shape == (2, 3)
    
    print("✓ ExternalLikelihood initialization successful")
    print(f"  Parameters: {likelihood.param_list}")
    print(f"  Bounds shape: {likelihood.param_bounds.shape}")


def test_external_likelihood_single_evaluation():
    """Test single point evaluation."""
    print("\n" + "="*80)
    print("TEST: ExternalLikelihood Single Point Evaluation")
    print("="*80)
    
    pool = MPI_Pool()
    
    likelihood = ExternalLikelihood(
        loglikelihood=simple_loglike,
        param_list=['x', 'y'],
        param_bounds=[(0, 1), (0, 1)],
        pool=pool
    )
    
    x = np.array([0.5, 0.5])
    result = likelihood(x)
    
    assert result.shape == (1, 1), f"Expected shape (1, 1), got {result.shape}"
    assert np.isfinite(result[0, 0]), "Result should be finite"
    
    # Should be maximum at [0.5, 0.5]
    expected = -0.0
    assert np.abs(result[0, 0] - expected) < 1e-10, f"Expected {expected}, got {result[0, 0]}"
    
    print(f"✓ Single point evaluation successful: {result[0, 0]:.6f}")


def test_external_likelihood_batch_evaluation():
    """Test batch evaluation."""
    print("\n" + "="*80)
    print("TEST: ExternalLikelihood Batch Evaluation")
    print("="*80)
    
    pool = MPI_Pool()
    
    likelihood = ExternalLikelihood(
        loglikelihood=simple_loglike,
        param_list=['x', 'y'],
        param_bounds=np.array([(0, 1), (0, 1)]).T,
        pool=pool
    )
    
    X = np.array([
        [0.5, 0.5],
        [0.0, 0.0],
        [1.0, 1.0]
    ])
    
    results = likelihood(X)
    
    assert results.shape == (3, 1), f"Expected shape (3, 1), got {results.shape}"
    assert all(np.isfinite(results)), "All results should be finite"
    
    # First point should be maximum
    assert results[0, 0] > results[1, 0], "Center should have higher value than corners"
    assert results[0, 0] > results[2, 0], "Center should have higher value than corners"
    
    print(f"✓ Batch evaluation successful")
    print(f"  Results: {results.flatten()}")


def test_external_likelihood_initial_points():
    """Test initial Sobol point generation."""
    print("\n" + "="*80)
    print("TEST: ExternalLikelihood Initial Points")
    print("="*80)
    
    pool = MPI_Pool()
    
    likelihood = ExternalLikelihood(
        loglikelihood=simple_loglike,
        param_list=['x', 'y', 'z'],
        param_bounds=np.array([(0, 1), (0, 1), (0, 1)]).T,
        pool=pool
    )
    
    if pool.is_master:
        n_init = 16
        X_init, y_init = likelihood.get_initial_points(n_sobol_init=n_init, rng=42)
        
        assert X_init.shape == (n_init, 3), f"Expected shape ({n_init}, 3), got {X_init.shape}"
        assert y_init.shape == (n_init, 1), f"Expected shape ({n_init}, 1), got {y_init.shape}"
        
        # Check bounds
        assert np.all(X_init >= 0) and np.all(X_init <= 1), "Points should be within [0, 1]"
        
        # Check all evaluations are finite
        assert all(np.isfinite(y_init)), "All initial evaluations should be finite"
        
        print(f"✓ Initial points generation successful")
        print(f"  Generated {n_init} points")
        print(f"  Value range: [{y_init.min():.4f}, {y_init.max():.4f}]")


def test_external_likelihood_nan_handling():
    """Test handling of NaN values."""
    print("\n" + "="*80)
    print("TEST: ExternalLikelihood NaN Handling")
    print("="*80)
    
    pool = MPI_Pool()
    
    likelihood = ExternalLikelihood(
        loglikelihood=nan_loglike,
        param_list=['x'],
        param_bounds=np.array([(0, 1)]).T,
        pool=pool,
        minus_inf=-1e5
    )
    
    x = np.array([0.5])
    result = likelihood(x)
    
    assert result.shape == (1, 1)
    assert result[0, 0] == -1e5, f"NaN should be replaced with minus_inf={-1e5}"
    
    print(f"✓ NaN handling successful: replaced with {result[0, 0]}")


def test_external_likelihood_exception_handling():
    """Test handling of exceptions."""
    print("\n" + "="*80)
    print("TEST: ExternalLikelihood Exception Handling")
    print("="*80)
    
    pool = MPI_Pool()
    
    likelihood = ExternalLikelihood(
        loglikelihood=exception_loglike,
        param_list=['x'],
        param_bounds=np.array([(0, 1)]).T,
        pool=pool,
        minus_inf=-1e5
    )
    
    x = np.array([0.5])
    result = likelihood(x)
    
    assert result.shape == (1, 1)
    assert result[0, 0] == -1e5, f"Exception should result in minus_inf={-1e5}"
    
    print(f"✓ Exception handling successful: replaced with {result[0, 0]}")


def test_external_likelihood_bounds_validation():
    """Test parameter bounds validation."""
    print("\n" + "="*80)
    print("TEST: ExternalLikelihood Bounds Validation")
    print("="*80)
    
    pool = MPI_Pool()
    
    likelihood = ExternalLikelihood(
        loglikelihood=simple_loglike,
        param_list=['x', 'y'],
        param_bounds=np.array([(-10, 10), (0, 100)]).T,
        pool=pool
    )
    
    # Test with points at boundaries
    X_boundary = np.array([
        [-10, 0],
        [10, 100],
        [0, 50]
    ])
    
    results = likelihood(X_boundary)
    
    assert results.shape == (3, 1)
    assert all(np.isfinite(results)), "Boundary evaluations should be finite"
    
    print(f"✓ Bounds validation successful")
    print(f"  Boundary results: {results.flatten()}")


def test_external_likelihood_noise():
    """Test noise parameter handling."""
    print("\n" + "="*80)
    print("TEST: ExternalLikelihood Noise Parameter")
    print("="*80)
    
    pool = MPI_Pool()
    
    likelihood = ExternalLikelihood(
        loglikelihood=simple_loglike,
        param_list=['x', 'y'],
        param_bounds=np.array([(0, 1), (0, 1)]).T,
        pool=pool,
        noise_std=0.1
    )
    
    assert likelihood.noise_std == 0.1, f"Expected noise_std=0.1, got {likelihood.noise_std}"
    
    print(f"✓ Noise parameter set correctly: {likelihood.noise_std}")


def test_external_likelihood_dimension_mismatch():
    """Test error handling for dimension mismatch."""
    print("\n" + "="*80)
    print("TEST: ExternalLikelihood Dimension Mismatch")
    print("="*80)
    
    pool = MPI_Pool()
    
    likelihood = ExternalLikelihood(
        loglikelihood=simple_loglike,
        param_list=['x', 'y', 'z'],
        param_bounds=np.array([(0, 1), (0, 1), (0, 1)]).T,
        pool=pool
    )
    
    # Try to evaluate with wrong dimension
    x_wrong = np.array([0.5, 0.5])  # 2D instead of 3D
    
    try:
        likelihood(x_wrong)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "does not match ndim" in str(e)
        print(f"✓ Dimension mismatch correctly caught: {str(e)}")


if __name__ == '__main__':
    print("\nRunning Likelihood Tests")
    print("="*80)
    
    test_external_likelihood_initialization()
    test_external_likelihood_single_evaluation()
    test_external_likelihood_batch_evaluation()
    test_external_likelihood_initial_points()
    test_external_likelihood_nan_handling()
    test_external_likelihood_exception_handling()
    test_external_likelihood_bounds_validation()
    test_external_likelihood_noise()
    test_external_likelihood_dimension_mismatch()
    
    print("\n" + "="*80)
    print("All Likelihood Tests Passed!")
    print("="*80)
