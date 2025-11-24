"""
Tests for GPwithClassifier functionality.

Tests include:
- Initialization with different classifier types (SVM, NN, Ellipsoid)
- Classifier training and thresholds
- Predictions with classifier filtering
- Update mechanism
- State save/load for classifier+GP
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import sys
from jaxbo.clf_gp import GPwithClassifier


def generate_test_data_with_outliers(n_good=30, n_bad=20, d=2, seed=42):
    """Generate test data with good and bad regions for classifier testing."""
    rng = np.random.RandomState(seed)
    
    # Good region: near [0.5, 0.5]
    X_good = rng.uniform(0.3, 0.7, size=(n_good, d))
    y_good = -np.sum((X_good - 0.5)**2, axis=1, keepdims=True)
    
    # Bad region: corners
    X_bad = rng.uniform(0, 1, size=(n_bad, d))
    X_bad = np.where(X_bad < 0.5, X_bad * 0.4, 0.6 + X_bad * 0.4)
    y_bad = -10 - np.sum((X_bad - 0.5)**2, axis=1, keepdims=True)
    
    X = np.vstack([X_good, X_bad])
    y = np.vstack([y_good, y_bad])
    
    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def test_clf_gp_initialization_svm():
    """Test GPwithClassifier initialization with SVM."""
    print("\n" + "="*80)
    print("TEST: GPwithClassifier Initialization (SVM)")
    print("="*80)
    
    X, y = generate_test_data_with_outliers(n_good=40, n_bad=30, d=3)
    
    gp_clf = GPwithClassifier(
        train_x=X,
        train_y=y,
        clf_type='svm',
        clf_settings={'gamma': 'scale', 'C': 1e5},
        clf_use_size=50,
        clf_threshold=5.0,
        gp_threshold=10.0,
        noise=1e-6
    )
    
    print(f"✓ Classifier data size: {gp_clf.clf_data_size}")
    print(f"✓ GP data size: {gp_clf.npoints}")
    print(f"✓ Classifier type: {gp_clf.clf_type}")
    print(f"✓ Using classifier: {gp_clf.use_clf}")
    
    assert gp_clf.clf_type == 'svm', "Classifier type should be SVM"
    assert gp_clf.clf_data_size == len(X), "Classifier should have all data"
    assert gp_clf.npoints <= len(X), "GP should have subset of data"
    
    if gp_clf.use_clf:
        print(f"✓ Classifier metrics: {gp_clf.clf_metrics}")


def test_clf_gp_initialization_nn():
    """Test GPwithClassifier initialization with Neural Network."""
    print("\n" + "="*80)
    print("TEST: GPwithClassifier Initialization (Neural Network)")
    print("="*80)
    
    X, y = generate_test_data_with_outliers(n_good=35, n_bad=25, d=2)
    
    gp_clf = GPwithClassifier(
        train_x=X,
        train_y=y,
        clf_type='nn',
        clf_settings={'hidden_dims': [32, 32], 'n_epochs': 500, 'n_restarts': 1},
        clf_use_size=50,
        clf_threshold=5.0,
        gp_threshold=10.0,
        noise=1e-6
    )
    
    print(f"✓ Classifier data size: {gp_clf.clf_data_size}")
    print(f"✓ GP data size: {gp_clf.npoints}")
    print(f"✓ Classifier type: {gp_clf.clf_type}")
    print(f"✓ Using classifier: {gp_clf.use_clf}")
    
    assert gp_clf.clf_type == 'nn', "Classifier type should be NN"
    
    if gp_clf.use_clf:
        print(f"✓ Classifier metrics: {gp_clf.clf_metrics}")


def test_clf_gp_initialization_ellipsoid():
    """Test GPwithClassifier initialization with Ellipsoid."""
    print("\n" + "="*80)
    print("TEST: GPwithClassifier Initialization (Ellipsoid)")
    print("="*80)
    
    X, y = generate_test_data_with_outliers(n_good=35, n_bad=25, d=2)
    
    gp_clf = GPwithClassifier(
        train_x=X,
        train_y=y,
        clf_type='ellipsoid',
        clf_settings={'n_epochs': 500, 'n_restarts': 1},
        clf_use_size=50,
        clf_threshold=5.0,
        gp_threshold=10.0,
        noise=1e-6
    )
    
    print(f"✓ Classifier data size: {gp_clf.clf_data_size}")
    print(f"✓ GP data size: {gp_clf.npoints}")
    print(f"✓ Classifier type: {gp_clf.clf_type}")
    print(f"✓ Using classifier: {gp_clf.use_clf}")
    
    assert gp_clf.clf_type == 'ellipsoid', "Classifier type should be Ellipsoid"
    
    if gp_clf.use_clf:
        print(f"✓ Classifier metrics: {gp_clf.clf_metrics}")


def test_clf_gp_predictions():
    """Test predictions with classifier filtering."""
    print("\n" + "="*80)
    print("TEST: GPwithClassifier Predictions")
    print("="*80)
    
    X, y = generate_test_data_with_outliers(n_good=40, n_bad=30, d=2)
    
    gp_clf = GPwithClassifier(
        train_x=X,
        train_y=y,
        clf_type='svm',
        clf_use_size=50,
        clf_threshold=5.0,
        gp_threshold=10.0,
        probability_threshold=0.5,
        minus_inf=-1e5,
        noise=1e-6
    )
    
    # Test prediction in good region
    good_point = jnp.array([0.5, 0.5])
    mean_good = gp_clf.predict_mean_single(good_point)
    var_good = gp_clf.predict_var_single(good_point)
    
    print(f"Prediction in good region [0.5, 0.5]:")
    print(f"  Mean: {mean_good:.4f}")
    print(f"  Variance: {var_good:.4e}")
    
    # Test prediction in bad region
    bad_point = jnp.array([0.05, 0.05])
    mean_bad = gp_clf.predict_mean_single(bad_point)
    var_bad = gp_clf.predict_var_single(bad_point)
    
    print(f"\nPrediction in bad region [0.05, 0.05]:")
    print(f"  Mean: {mean_bad:.4f}")
    print(f"  Variance: {var_bad:.4e}")
    
    if gp_clf.use_clf:
        # In bad region, mean should be very negative (minus_inf)
        assert mean_bad < mean_good, "Bad region should have lower predicted mean"
        print(f"\n✓ Classifier successfully filters bad regions")
    else:
        print(f"\n✓ Classifier not active (not enough data)")
    
    # Batched predictions
    test_points = jnp.array([[0.5, 0.5], [0.05, 0.05], [0.6, 0.4]])
    means = gp_clf.predict_mean_batched(test_points)
    vars = gp_clf.predict_var_batched(test_points)
    
    print(f"\nBatch predictions:")
    for i, pt in enumerate(test_points):
        print(f"  Point {pt}: mean={means[i]:.4f}, var={vars[i]:.4e}")
    
    assert means.shape == (3,), "Batched means should have correct shape"
    assert vars.shape == (3,), "Batched variances should have correct shape"


def test_clf_gp_update():
    """Test updating GPwithClassifier with new points."""
    print("\n" + "="*80)
    print("TEST: GPwithClassifier Update")
    print("="*80)
    
    X, y = generate_test_data_with_outliers(n_good=25, n_bad=15, d=2)
    
    gp_clf = GPwithClassifier(
        train_x=X,
        train_y=y,
        clf_type='svm',
        clf_use_size=30,
        clf_threshold=5.0,
        gp_threshold=10.0,
        noise=1e-6
    )
    
    initial_clf_size = gp_clf.clf_data_size
    initial_gp_size = gp_clf.npoints
    
    print(f"Initial sizes - Classifier: {initial_clf_size}, GP: {initial_gp_size}")
    
    # Add new points in good region
    new_X = jnp.array([[0.55, 0.45], [0.48, 0.52]])
    new_y = -jnp.sum((new_X - 0.5)**2, axis=1, keepdims=True)
    
    gp_clf.update(new_X, new_y, refit=False)
    
    print(f"After update - Classifier: {gp_clf.clf_data_size}, GP: {gp_clf.npoints}")
    
    assert gp_clf.clf_data_size == initial_clf_size + 2, "Classifier should have 2 more points"
    # GP might have 0, 1, or 2 more points depending on threshold
    assert gp_clf.npoints >= initial_gp_size, "GP should have at least as many points"
    
    print(f"✓ Update successful")


def test_clf_gp_classifier_training():
    """Test classifier training mechanism."""
    print("\n" + "="*80)
    print("TEST: Classifier Training Mechanism")
    print("="*80)
    
    # Start with not enough data
    X_small, y_small = generate_test_data_with_outliers(n_good=15, n_bad=10, d=2)
    
    gp_clf = GPwithClassifier(
        train_x=X_small,
        train_y=y_small,
        clf_type='svm',
        clf_use_size=50,  # Higher than initial data
        clf_threshold=5.0,
        noise=1e-6
    )
    
    print(f"Initial state - Data: {gp_clf.clf_data_size}, Use classifier: {gp_clf.use_clf}")
    assert not gp_clf.use_clf, "Should not use classifier with insufficient data"
    
    # Add more data to reach threshold
    X_add, y_add = generate_test_data_with_outliers(n_good=25, n_bad=15, d=2)
    gp_clf.update(X_add, y_add, refit=False)
    
    # Manually trigger classifier training
    gp_clf.train_classifier()
    
    print(f"After adding data - Data: {gp_clf.clf_data_size}, Use classifier: {gp_clf.use_clf}")
    
    if gp_clf.clf_data_size >= gp_clf.clf_use_size:
        assert gp_clf.use_clf, "Should use classifier after reaching threshold"
        print(f"✓ Classifier activated with metrics: {gp_clf.clf_metrics}")
    else:
        print(f"✓ Still not enough data ({gp_clf.clf_data_size} < {gp_clf.clf_use_size})")


def test_clf_gp_random_point():
    """Test random point generation with classifier."""
    print("\n" + "="*80)
    print("TEST: Random Point Generation with Classifier")
    print("="*80)
    
    X, y = generate_test_data_with_outliers(n_good=40, n_bad=30, d=2)
    
    gp_clf = GPwithClassifier(
        train_x=X,
        train_y=y,
        clf_type='svm',
        clf_use_size=50,
        clf_threshold=5.0,
        noise=1e-6
    )
    
    rng = np.random.default_rng(42)
    points = []
    
    for i in range(5):
        pt = gp_clf.get_random_point(rng=rng)
        points.append(pt)
        print(f"  Random point {i+1}: {pt}")
    
    points = np.array(points)
    
    assert points.shape == (5, 2), f"Expected shape (5, 2), got {points.shape}"
    assert np.all(points >= 0) and np.all(points <= 1), "Points should be in unit cube"
    
    print(f"✓ Random point generation successful")


def test_clf_gp_state_dict():
    """Test GPwithClassifier state save/load."""
    print("\n" + "="*80)
    print("TEST: GPwithClassifier State Save/Load")
    print("="*80)
    
    X, y = generate_test_data_with_outliers(n_good=35, n_bad=25, d=2)
    
    gp_clf1 = GPwithClassifier(
        train_x=X,
        train_y=y,
        clf_type='svm',
        clf_settings={'gamma': 'scale', 'C': 1e5},
        clf_use_size=50,
        clf_threshold=5.0,
        gp_threshold=10.0,
        noise=1e-6
    )
    
    # Get state dict
    state = gp_clf1.state_dict()
    
    print(f"State dict has {len(state)} keys")
    print(f"Classifier type in state: {state.get('clf_type')}")
    
    # Create new instance from state
    gp_clf2 = GPwithClassifier.from_state_dict(state)
    
    # Verify attributes match
    assert gp_clf2.clf_type == gp_clf1.clf_type, "Classifier type doesn't match"
    assert gp_clf2.clf_data_size == gp_clf1.clf_data_size, "Data size doesn't match"
    assert gp_clf2.use_clf == gp_clf1.use_clf, "Classifier usage flag doesn't match"
    assert jnp.allclose(gp_clf2.train_x_clf, gp_clf1.train_x_clf), "Training data doesn't match"
    
    # Test predictions match
    test_point = jnp.array([0.5, 0.5])
    mean1 = gp_clf1.predict_mean_single(test_point)
    mean2 = gp_clf2.predict_mean_single(test_point)
    
    print(f"\nPrediction at [0.5, 0.5]:")
    print(f"  Original: {mean1:.4f}")
    print(f"  Loaded: {mean2:.4f}")
    print(f"  Difference: {abs(mean1 - mean2):.4e}")
    
    assert jnp.isclose(mean1, mean2, rtol=1e-5), "Predictions don't match"
    
    print(f"✓ State save/load successful")


def test_clf_gp_copy():
    """Test GPwithClassifier copy functionality."""
    print("\n" + "="*80)
    print("TEST: GPwithClassifier Copy")
    print("="*80)
    
    X, y = generate_test_data_with_outliers(n_good=30, n_bad=20, d=2)
    
    gp_clf1 = GPwithClassifier(
        train_x=X,
        train_y=y,
        clf_type='svm',
        clf_use_size=40,
        noise=1e-6
    )
    
    gp_clf2 = gp_clf1.copy()
    
    # Modify copy
    new_X = jnp.array([[0.6, 0.4]])
    new_y = jnp.array([[-0.02]])
    gp_clf2.update(new_X, new_y, refit=False)
    
    print(f"Original size: {gp_clf1.clf_data_size}")
    print(f"Copy size: {gp_clf2.clf_data_size}")
    
    assert gp_clf1.clf_data_size != gp_clf2.clf_data_size, "Copy should be independent"
    assert gp_clf2.clf_data_size == gp_clf1.clf_data_size + 1, "Copy should have one more point"
    
    print(f"✓ Copy is independent")


def run_all_tests():
    """Run all clf_gp tests."""
    print("\n" + "="*80)
    print("RUNNING ALL CLASSIFIER+GP TESTS")
    print("="*80)
    
    tests = [
        test_clf_gp_initialization_svm,
        test_clf_gp_initialization_nn,
        test_clf_gp_initialization_ellipsoid,
        test_clf_gp_predictions,
        test_clf_gp_update,
        test_clf_gp_classifier_training,
        test_clf_gp_random_point,
        test_clf_gp_state_dict,
        test_clf_gp_copy,
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
