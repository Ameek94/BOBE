"""
Tests for Gaussian Process (GP) functionality.

Tests include:
- Base GP initialization and setup
- Fitting hyperparameters
- Adding random points
- Evaluating predictions (mean and variance)
- Random point generation
- State save/load
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import sys
from BOBE.gp import GP


def generate_test_data(n_samples=50, d=2, seed=42):
    """Generate synthetic test data for GP."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n_samples, d))
    # Simple function: sum of squares
    y = -np.sum((X - 0.5)**2, axis=1).reshape(-1, 1)
    return X, y


def test_gp_initialization():
    """Test basic GP initialization."""
    print("\n" + "="*80)
    print("TEST: GP Initialization")
    print("="*80)
    
    X, y = generate_test_data(n_samples=20, d=3)
    
    gp = GP(
        train_x=X,
        train_y=y,
        noise=1e-6,
        kernel="rbf",
        lengthscale_bounds=[0.01, 10],
        kernel_variance_bounds=[1e-4, 1e4]
    )
    
    assert gp.ndim == 3, f"Expected ndim=3, got {gp.ndim}"
    assert gp.train_x.shape[0] == 20, f"Expected 20 training points, got {gp.train_x.shape[0]}"
    assert gp.kernel_name == "rbf", f"Expected rbf kernel, got {gp.kernel_name}"
    
    print(f"✓ GP initialized with {gp.npoints} points in {gp.ndim}D")
    print(f"✓ Kernel: {gp.kernel_name}")
    print(f"✓ Initial lengthscales: {gp.lengthscales}")
    print(f"✓ Initial kernel variance: {gp.kernel_variance}")
    print(f"✓ Y mean: {gp.y_mean:.4f}, Y std: {gp.y_std:.4f}")
    

def test_gp_fitting():
    """Test GP hyperparameter fitting."""
    print("\n" + "="*80)
    print("TEST: GP Hyperparameter Fitting")
    print("="*80)
    
    X, y = generate_test_data(n_samples=30, d=2)
    
    gp = GP(
        train_x=X,
        train_y=y,
        noise=1e-6,
        kernel="matern",
        optimizer="scipy",
        lengthscale_prior="DSLP"
    )
    
    print(f"Initial hyperparameters:")
    print(f"  Lengthscales: {gp.lengthscales}")
    print(f"  Kernel variance: {gp.kernel_variance:.4f}")
    
    # Fit the GP
    result = gp.fit(maxiter=200, x0=None)
    
    print(f"\nAfter fitting:")
    print(f"  MLL: {result['mll']:.4f}")
    print(f"  Lengthscales: {gp.lengthscales}")
    print(f"  Kernel variance: {gp.kernel_variance:.4f}")
    
    # Check that hyperparameters were updated
    assert result['mll'] is not None, "MLL should not be None after fitting"
    print(f"\n✓ GP fitted successfully with MLL = {result['mll']:.4f}")


def test_gp_predictions():
    """Test GP mean and variance predictions."""
    print("\n" + "="*80)
    print("TEST: GP Predictions")
    print("="*80)
    
    X, y = generate_test_data(n_samples=25, d=2)
    
    gp = GP(train_x=X, train_y=y, noise=1e-6)
    
    # Single point prediction
    test_point = jnp.array([0.5, 0.5])
    mean_single = gp.predict_mean_single(test_point)
    var_single = gp.predict_var_single(test_point)
    
    print(f"Single point prediction at [0.5, 0.5]:")
    print(f"  Mean: {mean_single:.4f}")
    print(f"  Variance: {var_single:.4e}")
    
    assert mean_single.shape == (), f"Expected scalar mean, got shape {mean_single.shape}"
    assert var_single.shape == (), f"Expected scalar variance, got shape {var_single.shape}"
    assert var_single > 0, "Variance must be positive"
    
    # Batched prediction
    test_points = jnp.array([[0.2, 0.3], [0.7, 0.8], [0.5, 0.5]])
    means_batch = gp.predict_mean_batched(test_points)
    vars_batch = gp.predict_var_batched(test_points)
    
    print(f"\nBatch prediction for {test_points.shape[0]} points:")
    print(f"  Means: {means_batch}")
    print(f"  Variances: {vars_batch}")
    
    assert means_batch.shape == (3,), f"Expected shape (3,), got {means_batch.shape}"
    assert vars_batch.shape == (3,), f"Expected shape (3,), got {vars_batch.shape}"
    assert jnp.all(vars_batch > 0), "All variances must be positive"
    
    # Check that prediction at training point has low variance
    train_point = X[0]
    mean_at_train = gp.predict_mean_single(train_point)
    var_at_train = gp.predict_var_single(train_point)
    
    print(f"\nPrediction at training point:")
    print(f"  True value: {y[0,0]:.4f}")
    print(f"  Predicted mean: {mean_at_train:.4f}")
    print(f"  Predicted variance: {var_at_train:.4e}")
    
    # Variance at training point should be close to noise level
    assert var_at_train < 1e-3, f"Variance at training point too high: {var_at_train}"
    
    print(f"\n✓ All prediction tests passed")


def test_gp_update():
    """Test adding new points to GP."""
    print("\n" + "="*80)
    print("TEST: GP Update with New Points")
    print("="*80)
    
    X, y = generate_test_data(n_samples=15, d=2)
    
    gp = GP(train_x=X, train_y=y, noise=1e-6)
    initial_size = gp.npoints
    
    print(f"Initial GP size: {initial_size} points")
    
    # Add new points
    new_X = jnp.array([[0.8, 0.2], [0.3, 0.9]])
    new_y = -jnp.sum((new_X - 0.5)**2, axis=1, keepdims=True)
    
    gp.update(new_X, new_y, refit=False)
    
    print(f"After update: {gp.npoints} points")
    
    assert gp.npoints == initial_size + 2, f"Expected {initial_size + 2} points, got {gp.npoints}"
    
    # Try adding duplicate point
    gp.update(new_X[0:1], new_y[0:1], refit=False)
    assert gp.npoints == initial_size + 2, "Duplicate point should not be added"
    
    print(f"✓ GP update successful, duplicates correctly handled")


def test_gp_random_point():
    """Test random point generation."""
    print("\n" + "="*80)
    print("TEST: Random Point Generation")
    print("="*80)
    
    X, y = generate_test_data(n_samples=20, d=3)
    gp = GP(train_x=X, train_y=y)
    
    # Generate multiple random points
    rng = np.random.default_rng(42)
    points = []
    for i in range(10):
        pt = gp.get_random_point(rng=rng)
        points.append(pt)
        print(f"  Point {i+1}: {pt}")
    
    points = np.array(points)
    
    assert points.shape == (10, 3), f"Expected shape (10, 3), got {points.shape}"
    assert np.all(points >= 0) and np.all(points <= 1), "Points should be in unit cube [0,1]^d"
    
    # Check that points are different (not all the same)
    assert not np.allclose(points, points[0]), "Random points should be different"
    
    print(f"\n✓ Random point generation successful")


def test_gp_state_dict():
    """Test GP state save/load."""
    print("\n" + "="*80)
    print("TEST: GP State Save/Load")
    print("="*80)
    
    X, y = generate_test_data(n_samples=20, d=2)
    
    gp1 = GP(
        train_x=X,
        train_y=y,
        noise=1e-6,
        kernel="rbf",
        lengthscales=jnp.array([0.5, 0.3]),
        kernel_variance=2.0
    )
    
    # Get state dict
    state = gp1.state_dict()
    
    print(f"State dict keys: {list(state.keys())}")
    
    # Create new GP from state
    gp2 = GP.from_state_dict(state)
    
    # Check that GPs are identical
    assert gp2.ndim == gp1.ndim, "Dimensions don't match"
    assert gp2.npoints == gp1.npoints, "Number of points don't match"
    assert jnp.allclose(gp2.lengthscales, gp1.lengthscales), "Lengthscales don't match"
    assert jnp.isclose(gp2.kernel_variance, gp1.kernel_variance), "Kernel variance doesn't match"
    assert jnp.allclose(gp2.train_x, gp1.train_x), "Training X doesn't match"
    
    # Check predictions are the same
    test_point = jnp.array([0.5, 0.5])
    mean1 = gp1.predict_mean_single(test_point)
    mean2 = gp2.predict_mean_single(test_point)
    
    print(f"\nPrediction comparison at [0.5, 0.5]:")
    print(f"  Original GP mean: {mean1:.4f}")
    print(f"  Loaded GP mean: {mean2:.4f}")
    print(f"  Difference: {abs(mean1 - mean2):.4e}")
    
    assert jnp.isclose(mean1, mean2, rtol=1e-6), "Predictions don't match"
    
    print(f"\n✓ State save/load successful")


def test_gp_copy():
    """Test GP copy functionality."""
    print("\n" + "="*80)
    print("TEST: GP Copy")
    print("="*80)
    
    X, y = generate_test_data(n_samples=15, d=2)
    
    gp1 = GP(train_x=X, train_y=y, noise=1e-6)
    gp2 = gp1.copy()
    
    # Check that copy is independent
    new_X = jnp.array([[0.9, 0.1]])
    new_y = jnp.array([[-0.5]])
    
    gp2.update(new_X, new_y, refit=False)
    
    print(f"Original GP size: {gp1.npoints}")
    print(f"Copied GP size: {gp2.npoints}")
    
    assert gp1.npoints != gp2.npoints, "Copy should be independent"
    assert gp2.npoints == gp1.npoints + 1, "Copied GP should have one more point"
    
    print(f"✓ GP copy is independent")


def test_gp_different_kernels():
    """Test GP with different kernel types."""
    print("\n" + "="*80)
    print("TEST: Different Kernel Types")
    print("="*80)
    
    X, y = generate_test_data(n_samples=20, d=2)
    
    # RBF kernel
    gp_rbf = GP(train_x=X, train_y=y, kernel="rbf")
    mean_rbf = gp_rbf.predict_mean_single(jnp.array([0.5, 0.5]))
    
    # Matern kernel
    gp_matern = GP(train_x=X, train_y=y, kernel="matern")
    mean_matern = gp_matern.predict_mean_single(jnp.array([0.5, 0.5]))
    
    print(f"RBF kernel prediction: {mean_rbf:.4f}")
    print(f"Matern kernel prediction: {mean_matern:.4f}")
    
    # Both should give reasonable predictions (not too different, but not identical)
    assert not jnp.isclose(mean_rbf, mean_matern, rtol=0.01), "Different kernels should give different predictions"
    
    print(f"✓ Both kernel types work correctly")


def run_all_tests():
    """Run all GP tests."""
    print("\n" + "="*80)
    print("RUNNING ALL GP TESTS")
    print("="*80)
    
    tests = [
        test_gp_initialization,
        test_gp_fitting,
        test_gp_predictions,
        test_gp_update,
        test_gp_random_point,
        test_gp_state_dict,
        test_gp_copy,
        test_gp_different_kernels,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ TEST FAILED: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ TEST ERROR: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print(f"\n✗ {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
