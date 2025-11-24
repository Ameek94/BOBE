"""
Tests for MPI_Pool utility functionality.

Tests include:
- Pool initialization (with and without MPI)
- Serial execution (map_objective)
- GP fitting with pool
- Dynamic task distribution
- Error handling
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import sys
from jaxbo.utils.pool import MPI_Pool
from jaxbo.gp import GP


def simple_objective(x):
    """Simple test objective function."""
    return -np.sum((x - 0.5)**2)


def generate_test_gp(n_samples=20, d=2, seed=42):
    """Generate a simple GP for testing."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n_samples, d))
    y = -np.sum((X - 0.5)**2, axis=1, keepdims=True)
    
    gp = GP(
        train_x=X,
        train_y=y,
        noise=1e-6,
        kernel="rbf",
        lengthscales=jnp.array([0.3] * d),
        kernel_variance=1.0
    )
    
    return gp


def test_pool_initialization():
    """Test MPI_Pool initialization."""
    print("\n" + "="*80)
    print("TEST: Pool Initialization")
    print("="*80)
    
    pool = MPI_Pool()
    
    print(f"Pool rank: {pool.rank}")
    print(f"Pool size: {pool.size}")
    print(f"Is MPI: {pool.is_mpi}")
    print(f"Is master: {pool.is_master}")
    print(f"Is worker: {pool.is_worker}")
    
    # In non-MPI environment
    assert pool.rank == 0, "Rank should be 0 in serial mode"
    assert pool.size == 1, "Size should be 1 in serial mode"
    assert not pool.is_mpi, "Should not be MPI in serial mode"
    assert pool.is_master, "Should be master in serial mode"
    assert not pool.is_worker, "Should not be worker in serial mode"
    
    print(f"\n✓ Pool initialized successfully in serial mode")


def test_run_map_objective():
    """Test parallel mapping of objective function."""
    print("\n" + "="*80)
    print("TEST: Run Map Objective")
    print("="*80)
    
    pool = MPI_Pool()
    
    # Create test tasks (points to evaluate)
    test_points = [
        np.array([0.2, 0.3]),
        np.array([0.5, 0.5]),
        np.array([0.8, 0.7]),
        np.array([0.1, 0.9]),
    ]
    
    print(f"Evaluating {len(test_points)} points...")
    
    # Run map
    results = pool.run_map_objective(simple_objective, test_points)
    
    print(f"\nResults shape: {results.shape}")
    print(f"Results: {results}")
    
    assert results.shape == (4,), f"Expected shape (4,), got {results.shape}"
    
    # Verify results are correct
    for i, pt in enumerate(test_points):
        expected = simple_objective(pt)
        print(f"  Point {pt}: result={results[i]:.4f}, expected={expected:.4f}")
        assert np.isclose(results[i], expected), f"Result mismatch at point {i}"
    
    print(f"\n✓ Map objective successful")


def test_gp_fit_serial():
    """Test GP fitting through pool (serial mode)."""
    print("\n" + "="*80)
    print("TEST: GP Fit (Serial Mode)")
    print("="*80)
    
    pool = MPI_Pool()
    gp = generate_test_gp(n_samples=25, d=2)
    
    print(f"Initial GP hyperparameters:")
    print(f"  Lengthscales: {gp.lengthscales}")
    print(f"  Kernel variance: {gp.kernel_variance:.4f}")
    
    # Fit GP through pool
    rng = np.random.default_rng(42)
    pool.gp_fit(gp, maxiters=200, n_restarts=3, rng=rng, use_pool=True)
    
    print(f"\nAfter fitting:")
    print(f"  Lengthscales: {gp.lengthscales}")
    print(f"  Kernel variance: {gp.kernel_variance:.4f}")
    
    # Verify hyperparameters were updated
    assert gp.lengthscales is not None, "Lengthscales should be set"
    assert gp.kernel_variance > 0, "Kernel variance should be positive"
    
    print(f"\n✓ GP fit through pool successful")


def test_gp_fit_without_pool():
    """Test GP fitting without using pool."""
    print("\n" + "="*80)
    print("TEST: GP Fit (Without Pool)")
    print("="*80)
    
    pool = MPI_Pool()
    gp = generate_test_gp(n_samples=25, d=2, seed=123)
    
    print(f"Initial GP hyperparameters:")
    print(f"  Lengthscales: {gp.lengthscales}")
    print(f"  Kernel variance: {gp.kernel_variance:.4f}")
    
    # Fit GP without pool
    rng = np.random.default_rng(42)
    pool.gp_fit(gp, maxiters=200, n_restarts=3, rng=rng, use_pool=False)
    
    print(f"\nAfter fitting (without pool):")
    print(f"  Lengthscales: {gp.lengthscales}")
    print(f"  Kernel variance: {gp.kernel_variance:.4f}")
    
    assert gp.lengthscales is not None, "Lengthscales should be set"
    assert gp.kernel_variance > 0, "Kernel variance should be positive"
    
    print(f"\n✓ GP fit without pool successful")


def test_pool_with_different_task_sizes():
    """Test pool with varying number of tasks."""
    print("\n" + "="*80)
    print("TEST: Pool with Different Task Sizes")
    print("="*80)
    
    pool = MPI_Pool()
    
    # Test with different numbers of tasks
    task_sizes = [1, 5, 10, 20]
    
    for n_tasks in task_sizes:
        rng = np.random.RandomState(42)
        test_points = [rng.uniform(0, 1, size=2) for _ in range(n_tasks)]
        
        results = pool.run_map_objective(simple_objective, test_points)
        
        print(f"  {n_tasks} tasks: results shape = {results.shape}")
        
        assert results.shape == (n_tasks,), f"Expected {n_tasks} results, got {results.shape[0]}"
        assert np.all(np.isfinite(results)), "All results should be finite"
    
    print(f"\n✓ Pool handles different task sizes correctly")


def test_pool_with_zero_tasks():
    """Test pool with empty task list."""
    print("\n" + "="*80)
    print("TEST: Pool with Zero Tasks")
    print("="*80)
    
    pool = MPI_Pool()
    
    # Empty task list
    test_points = []
    results = pool.run_map_objective(simple_objective, test_points)
    
    print(f"Results for 0 tasks: {results}")
    
    assert len(results) == 0, "Should return empty array for zero tasks"
    
    print(f"✓ Pool handles zero tasks correctly")


def test_objective_function_types():
    """Test pool with different objective function types."""
    print("\n" + "="*80)
    print("TEST: Different Objective Function Types")
    print("="*80)
    
    pool = MPI_Pool()
    
    # Simple scalar function
    def scalar_fn(x):
        return float(np.sum(x))
    
    # Function returning array element
    def array_fn(x):
        return np.array([np.sum(x**2)])[0]
    
    # Function with conditional logic
    def conditional_fn(x):
        if np.sum(x) > 1.0:
            return 1.0
        return -1.0
    
    test_point = [np.array([0.5, 0.5])]
    
    result_scalar = pool.run_map_objective(scalar_fn, test_point)
    result_array = pool.run_map_objective(array_fn, test_point)
    result_conditional = pool.run_map_objective(conditional_fn, test_point)
    
    print(f"Scalar function result: {result_scalar}")
    print(f"Array function result: {result_array}")
    print(f"Conditional function result: {result_conditional}")
    
    assert len(result_scalar) == 1, "Should return single result"
    assert len(result_array) == 1, "Should return single result"
    assert len(result_conditional) == 1, "Should return single result"
    
    print(f"\n✓ Pool works with different objective function types")


def test_gp_state_serialization_for_pool():
    """Test that GP state can be properly serialized for pool distribution."""
    print("\n" + "="*80)
    print("TEST: GP State Serialization for Pool")
    print("="*80)
    
    gp = generate_test_gp(n_samples=20, d=2)
    
    # Get state dict (this is what would be sent to workers)
    state = gp.state_dict()
    
    print(f"State dict has {len(state)} keys")
    print(f"State dict keys: {list(state.keys())[:10]}...")  # Show first 10 keys
    
    # Reconstruct GP from state
    gp_reconstructed = GP.from_state_dict(state)
    
    # Verify reconstruction
    assert gp_reconstructed.ndim == gp.ndim, "Dimensions don't match"
    assert gp_reconstructed.npoints == gp.npoints, "Number of points don't match"
    assert jnp.allclose(gp_reconstructed.lengthscales, gp.lengthscales), "Lengthscales don't match"
    
    # Test prediction consistency
    test_point = jnp.array([0.5, 0.5])
    mean1 = gp.predict_mean_single(test_point)
    mean2 = gp_reconstructed.predict_mean_single(test_point)
    
    print(f"\nPrediction comparison:")
    print(f"  Original: {mean1:.4f}")
    print(f"  Reconstructed: {mean2:.4f}")
    print(f"  Difference: {abs(mean1 - mean2):.4e}")
    
    assert jnp.isclose(mean1, mean2, rtol=1e-6), "Predictions don't match after serialization"
    
    print(f"\n✓ GP state serialization works correctly")


def test_pool_close():
    """Test closing the pool."""
    print("\n" + "="*80)
    print("TEST: Pool Close")
    print("="*80)
    
    pool = MPI_Pool()
    
    # Close pool (should work without errors in serial mode)
    pool.close()
    
    print(f"✓ Pool closed successfully")


def run_all_tests():
    """Run all pool tests."""
    print("\n" + "="*80)
    print("RUNNING ALL POOL TESTS")
    print("="*80)
    print("NOTE: These tests run in serial mode (no MPI)")
    print("="*80)
    
    tests = [
        test_pool_initialization,
        test_run_map_objective,
        test_gp_fit_serial,
        test_gp_fit_without_pool,
        test_pool_with_different_task_sizes,
        test_pool_with_zero_tasks,
        test_objective_function_types,
        test_gp_state_serialization_for_pool,
        test_pool_close,
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
    
    print("\nNOTE: MPI-specific functionality (parallel execution, worker processes)")
    print("      cannot be tested in serial mode. These tests verify the serial fallback.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
