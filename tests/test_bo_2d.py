"""
Test BOBE on 2D test functions.

This test runs both EI and WIPStd acquisition functions on simple 2D optimization problems
to verify the basic functionality of the Bayesian Optimization loop.
"""

import numpy as np
import sys
from BOBE.bo import BOBE
from BOBE.likelihood import Likelihood


def rosenbrock_loglike(x):
    """
    Negative Rosenbrock function as log-likelihood.
    Global minimum at (1, 1) with value 0.
    """
    return -((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2)


def himmelblau_loglike(x):
    """
    Negative Himmelblau function as log-likelihood.
    Has four identical local minima.
    """
    return -((x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2)


def test_bobe_ei_2d():
    """Test BOBE with EI acquisition on 2D Rosenbrock function."""
    print("\n" + "="*80)
    print("TEST: BOBE with EI on 2D Rosenbrock")
    print("="*80)
    
    param_bounds = np.array([[-2, 2], [-2, 2]]).T
    param_list = ['x', 'y']
    
    likelihood = Likelihood(
        loglikelihood=rosenbrock_loglike,
        param_list=param_list,
        param_bounds=param_bounds,
        name="rosenbrock_test"
    )
    
    bobe = BOBE(
        loglikelihood=likelihood,
        n_sobol_init=10,
        n_cobaya_init=0,
        save=False,
        use_clf=False,
        seed=42,
        verbosity='WARNING'
    )
    
    # Run with EI
    results = bobe.run(
        acq='EI',
        min_evals=15,
        max_evals=40,
        max_gp_size=40,
        ei_goal=1e-6,
        fit_n_points=5
    )
    
    # Check results structure
    assert 'gp' in results, "Results missing 'gp' key"
    assert 'likelihood' in results, "Results missing 'likelihood' key"
    assert 'results_manager' in results, "Results missing 'results_manager' key"
    assert 'best_val' in results, "Results missing 'best_val' key"
    assert 'best_pt' in results, "Results missing 'best_pt' key"
    assert 'termination_reason' in results, "Results missing 'termination_reason' key"
    
    # EI doesn't generate samples or logz - should be empty dicts
    assert 'samples' in results, "Results missing 'samples' key"
    assert 'logz' in results, "Results missing 'logz' key"
    assert results['samples'] == {}, "EI should have empty samples dict"
    assert results['logz'] == {}, "EI should have empty logz dict"
    
    # Check GP was trained (should have at least initial points)
    assert results['gp'].train_x.shape[0] >= 10, "GP has fewer training points than expected"
    assert results['gp'].train_x.shape[1] == 2, "GP has wrong dimensionality"
    
    # Check best point is reasonable (should be close to global optimum at [1, 1])
    best_pt = results['best_pt']
    best_val = results['best_val']
    
    print(f"\nBest point found: x={best_pt[0]:.4f}, y={best_pt[1]:.4f}")
    print(f"Best value: {best_val:.4f}")
    print(f"Distance from optimum (1, 1): {np.linalg.norm(best_pt - np.array([1, 1])):.4f}")
    print(f"Termination reason: {results['termination_reason']}")
    print(f"Total evaluations: {results['gp'].train_x.shape[0]}")
    
    # Check that best value is better than initial random samples (should be negative and large)
    assert best_val > -1000, f"Best value {best_val} is unexpectedly poor"
    
    print("\n✓ EI test passed")
    return results


def test_bobe_wipstd_2d():
    """Test BOBE with WIPStd acquisition on 2D Himmelblau function."""
    print("\n" + "="*80)
    print("TEST: BOBE with WIPStd on 2D Himmelblau")
    print("="*80)
    
    param_bounds = np.array([[-5, 5], [-5, 5]]).T
    param_list = ['x', 'y']
    
    likelihood = Likelihood(
        loglikelihood=himmelblau_loglike,
        param_list=param_list,
        param_bounds=param_bounds,
        name="himmelblau_test"
    )
    
    bobe = BOBE(
        loglikelihood=likelihood,
        n_sobol_init=15,
        n_cobaya_init=0,
        save=False,
        use_clf=False,
        seed=123,
        verbosity='WARNING'
    )
    
    # Run with WIPStd
    results = bobe.run(
        acq='WIPStd',
        min_evals=25,
        max_evals=60,
        max_gp_size=60,
        logz_threshold=0.5,
        convergence_n_iters=2,
        fit_n_points=8,
        ns_n_points=15,
        batch_size=1,  # Use batch size 1 to avoid shape issues
        mc_points_method='uniform'
    )
    
    # Check results structure
    assert 'gp' in results, "Results missing 'gp' key"
    assert 'likelihood' in results, "Results missing 'likelihood' key"
    assert 'results_manager' in results, "Results missing 'results_manager' key"
    assert 'best_val' in results, "Results missing 'best_val' key"
    assert 'best_pt' in results, "Results missing 'best_pt' key"
    assert 'termination_reason' in results, "Results missing 'termination_reason' key"
    
    # Check GP was trained
    assert results['gp'].train_x.shape[0] >= 30, "GP has fewer than min_evals training points"
    assert results['gp'].train_x.shape[1] == 2, "GP has wrong dimensionality"
    
    # Himmelblau has 4 minima, all with value 0
    # Check best point is reasonable
    best_pt = results['best_pt']
    best_val = results['best_val']
    
    print(f"\nBest point found: x={best_pt[0]:.4f}, y={best_pt[1]:.4f}")
    print(f"Best value: {best_val:.4f}")
    print(f"Termination reason: {results['termination_reason']}")
    print(f"Total evaluations: {results['gp'].train_x.shape[0]}")
    
    # Check that best value is reasonable (negative Himmelblau min is 0, so best should be close to 0)
    assert best_val > -500, f"Best value {best_val} is unexpectedly poor"
    
    # Check samples were generated (for WIPStd)
    if 'samples' in results and results['samples']:
        print(f"Generated {len(results['samples']['x'])} samples")
        assert len(results['samples']['x']) > 0, "No samples generated"
    
    print("\n✓ WIPStd test passed")
    return results


def test_bobe_with_classifier():
    """Test BOBE with classifier on 2D function."""
    print("\n" + "="*80)
    print("TEST: BOBE with Classifier on 2D Rosenbrock")
    print("="*80)
    
    param_bounds = np.array([[-2, 2], [-2, 2]]).T
    param_list = ['x', 'y']
    
    likelihood = Likelihood(
        loglikelihood=rosenbrock_loglike,
        param_list=param_list,
        param_bounds=param_bounds,
        name="rosenbrock_clf_test"
    )
    
    bobe = BOBE(
        loglikelihood=likelihood,
        n_sobol_init=12,
        n_cobaya_init=0,
        save=False,
        use_clf=True,
        clf_type='svm',
        clf_use_size=10,
        seed=456,
        verbosity='WARNING'
    )
    
    # Run with WIPStd and classifier
    results = bobe.run(
        acq='WIPStd',
        min_evals=20,
        max_evals=50,
        max_gp_size=50,
        logz_threshold=0.5,
        fit_n_points=6,
        ns_n_points=12,
        batch_size=1  # Use batch size 1
    )
    
    # Check results
    assert 'gp' in results
    assert hasattr(results['gp'], 'use_clf'), "GP should be GPwithClassifier"
    
    best_pt = results['best_pt']
    best_val = results['best_val']
    
    print(f"\nBest point found: x={best_pt[0]:.4f}, y={best_pt[1]:.4f}")
    print(f"Best value: {best_val:.4f}")
    print(f"Termination reason: {results['termination_reason']}")
    print(f"Total evaluations: {results['gp'].train_x.shape[0]}")
    
    print("\n✓ Classifier test passed")
    return results


def run_all_tests():
    """Run all 2D tests."""
    print("\n" + "="*80)
    print("RUNNING ALL BOBE 2D TESTS")
    print("="*80)
    
    try:
        test_bobe_ei_2d()
        test_bobe_wipstd_2d()
        test_bobe_with_classifier()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80 + "\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
