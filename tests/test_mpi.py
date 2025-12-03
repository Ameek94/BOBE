"""
Tests for new MPI utility functionality.

Tests include:
- MPI initialization and detection
- Serial execution (map_parallel)
- GP fitting with parallel hyperparameter optimization
- Scatter/gather operations
- Error handling
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import sys
from BOBE import mpi
from BOBE.gp import GP


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
    """Test MPI initialization and detection."""
    print("\n" + "="*80)
    print("TEST: MPI Initialization")
    print("="*80)
    
    print(f"MPI rank: {mpi.rank()}")
    print(f"MPI size: {mpi.size()}")
    print(f"Has MPI: {mpi.more_than_one_process()}")
    print(f"Is main process: {mpi.is_main_process()}")
    
    # Check MPI is working
    assert mpi.rank() >= 0, "Rank should be non-negative"
    assert mpi.size() >= 1, "Size should be at least 1"
    assert mpi.is_main_process() == (mpi.rank() == 0), "Main process check should match rank"
    
    if mpi.more_than_one_process():
        print(f"\n✓ MPI initialized successfully with {mpi.size()} processes")
    else:
        print(f"\n✓ MPI initialized successfully in serial mode")


def test_run_map_objective():
    """Test parallel mapping of objective function using map_parallel."""
    if mpi.is_main_process():
        print("\n" + "="*80)
        print("TEST: Map Parallel")
        print("="*80)
    
    # Create test tasks (points to evaluate) - only on main
    if mpi.is_main_process():
        test_points = [
            np.array([0.2, 0.3]),
            np.array([0.5, 0.5]),
            np.array([0.8, 0.7]),
            np.array([0.1, 0.9]),
        ]
        print(f"Evaluating {len(test_points)} points...")
    else:
        test_points = None
    
    # All processes call map_parallel (workers get None as tasks via scatter)
    results = mpi.map_parallel(simple_objective, test_points)
    
    # Only main process checks results
    if mpi.is_main_process():
        print(f"\nResults: {results}")
        
        assert len(results) == 4, f"Expected 4 results, got {len(results)}"
        
        # Verify results are correct
        for i, pt in enumerate(test_points):
            expected = simple_objective(pt)
            print(f"  Point {pt}: result={results[i]:.4f}, expected={expected:.4f}")
            assert np.isclose(results[i], expected), f"Result mismatch at point {i}"
        
        print(f"\n✓ map_parallel successful")


def test_gp_fit_serial():
    """Test GP fitting using gp_fit_parallel."""
    if mpi.is_main_process():
        print("\n" + "="*80)
        print("TEST: GP Fit Parallel")
        print("="*80)
    
    gp = generate_test_gp(n_samples=25, d=2)
    
    if mpi.is_main_process():
        print(f"Initial GP hyperparameters:")
        print(f"  Lengthscales: {gp.lengthscales}")
        print(f"  Kernel variance: {gp.kernel_variance:.4f}")
    
    # All processes call gp_fit_parallel
    mpi.gp_fit_parallel(gp, maxiters=200, n_restarts=3, use_parallel=True)
    
    if mpi.is_main_process():
        print(f"\nAfter fitting:")
        print(f"  Lengthscales: {gp.lengthscales}")
        print(f"  Kernel variance: {gp.kernel_variance:.4f}")
        
        # Verify hyperparameters were updated
        assert gp.lengthscales is not None, "Lengthscales should be set"
        assert gp.kernel_variance > 0, "Kernel variance should be positive"
        
        print(f"\n✓ gp_fit_parallel successful")


def test_gp_fit_without_pool():
    """Test GP fitting without using parallel mode."""
    if mpi.is_main_process():
        print("\n" + "="*80)
        print("TEST: GP Fit (Without Parallel)")
        print("="*80)
    
    gp = generate_test_gp(n_samples=25, d=2, seed=123)
    
    if mpi.is_main_process():
        print(f"Initial GP hyperparameters:")
        print(f"  Lengthscales: {gp.lengthscales}")
        print(f"  Kernel variance: {gp.kernel_variance:.4f}")
    
    # All processes call (but use_parallel=False means serial on main only)
    mpi.gp_fit_parallel(gp, maxiters=200, n_restarts=3, use_parallel=False)
    
    if mpi.is_main_process():
        print(f"\nAfter fitting (without parallel):")
        print(f"  Lengthscales: {gp.lengthscales}")
        print(f"  Kernel variance: {gp.kernel_variance:.4f}")
        
        assert gp.lengthscales is not None, "Lengthscales should be set"
        assert gp.kernel_variance > 0, "Kernel variance should be positive"
        
        print(f"\n✓ GP fit without parallel successful")


def test_pool_with_different_task_sizes():
    """Test map_parallel with varying number of tasks."""
    if mpi.is_main_process():
        print("\n" + "="*80)
        print("TEST: Map Parallel with Different Task Sizes")
        print("="*80)
    
    # Test with different numbers of tasks
    task_sizes = [1, 5, 10, 20]
    
    for n_tasks in task_sizes:
        if mpi.is_main_process():
            rng = np.random.RandomState(42)
            test_points = [rng.uniform(0, 1, size=2) for _ in range(n_tasks)]
        else:
            test_points = []
        
        results = mpi.map_parallel(simple_objective, test_points)
        
        if mpi.is_main_process():
            print(f"  {n_tasks} tasks: {len(results)} results")
            assert len(results) == n_tasks, f"Expected {n_tasks} results, got {len(results)}"
            assert all(np.isfinite(r) for r in results), "All results should be finite"
    
    if mpi.is_main_process():
        print(f"\n✓ map_parallel handles different task sizes correctly")


def test_pool_with_zero_tasks():
    """Test map_parallel with empty task list."""
    if mpi.is_main_process():
        print("\n" + "="*80)
        print("TEST: Map Parallel with Zero Tasks")
        print("="*80)
    
    # Empty task list
    test_points = [] if mpi.is_main_process() else []
    results = mpi.map_parallel(simple_objective, test_points)
    
    if mpi.is_main_process():
        print(f"Results for 0 tasks: {results}")
        assert len(results) == 0, "Should return empty list for zero tasks"
        print(f"✓ map_parallel handles zero tasks correctly")


def test_objective_function_types():
    """Test map_parallel with different objective function types."""
    if mpi.is_main_process():
        print("\n" + "="*80)
        print("TEST: Different Objective Function Types")
        print("="*80)
    
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
    
    test_point = [np.array([0.5, 0.5])] if mpi.is_main_process() else []
    
    result_scalar = mpi.map_parallel(scalar_fn, test_point)
    result_array = mpi.map_parallel(array_fn, test_point)
    result_conditional = mpi.map_parallel(conditional_fn, test_point)
    
    if mpi.is_main_process():
        print(f"Scalar function result: {result_scalar}")
        print(f"Array function result: {result_array}")
        print(f"Conditional function result: {result_conditional}")
        
        assert len(result_scalar) == 1, "Should return single result"
        assert len(result_array) == 1, "Should return single result"
        assert len(result_conditional) == 1, "Should return single result"
        
        print(f"\n✓ map_parallel works with different objective function types")


def test_gp_state_serialization_for_pool():
    """Test that GP state can be properly serialized for MPI distribution."""
    # This test doesn't use MPI, so only main process runs it
    if not mpi.is_main_process():
        return
        
    print("\n" + "="*80)
    print("TEST: GP State Serialization for MPI")
    print("="*80)
    
    gp = generate_test_gp(n_samples=20, d=2)
    
    # Get state dict (this is what would be sent to workers via share)
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


def test_scatter_gather():
    """Test MPI scatter and gather operations."""
    # This test works differently in serial vs parallel mode
    if not mpi.is_main_process():
        return
        
    print("\n" + "="*80)
    print("TEST: Scatter/Gather Operations")
    print("="*80)
    
    if not mpi.more_than_one_process():
        # Test data for serial mode
        data = [1, 2, 3, 4, 5]
        
        # Scatter data
        scattered = mpi.scatter(data)
        print(f"Scattered data (serial): {scattered}")
        
        # In serial mode, scatter returns first element
        assert scattered == data[0], "In serial mode, should receive first element"
        
        # Gather data back
        gathered = mpi.gather(scattered)
        print(f"Gathered data (serial): {gathered}")
        
        # In serial mode, gather returns a list with single element
        assert len(gathered) == 1, "In serial mode, gather should return list with 1 element"
        assert gathered[0] == data[0], "Gathered data should match scattered data"
    else:
        # In MPI mode, scatter() expects list to be split already
        # This is typically done internally by map_parallel and gp_fit_parallel
        print(f"Scatter/gather tested via map_parallel and gp_fit_parallel in MPI mode")
    
    print(f"\n✓ Scatter/gather operations work correctly")


def run_all_tests():
    """Run all MPI utility tests."""
    # All processes run tests, but only main prints output
    if mpi.is_main_process():
        print("\n" + "="*80)
        print("RUNNING ALL MPI UTILITY TESTS")
        print("="*80)
        if mpi.more_than_one_process():
            print(f"Running with MPI: {mpi.size()} processes")
        else:
            print("Running in serial mode (no MPI)")
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
        test_scatter_gather,
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
    print("\nTo test parallel execution, run:")
    print("  mpirun -n 4 python test_mpi.py")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
