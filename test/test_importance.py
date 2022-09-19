"""
Test the module importance.py
"""

import numpy as np
from sklearn.datasets import make_sparse_uncorrelated
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from rfgroove.importance import grouped_importance, permute_group
from rfgroove.dataset_generation import gaussian_multidimensional

def test_grouped_importance():
    """
    Test the function grouped_importance
    """
    # Dataset 1
    n_samples = 100
    n_features = 10

    X, y = make_sparse_uncorrelated(n_samples=n_samples, n_features=n_features, random_state=0)

    estimators = [
        ExtraTreesRegressor(n_estimators=1000, bootstrap=True, oob_score=True, max_samples=.1),
        RandomForestRegressor(n_estimators=1000, bootstrap=True, oob_score=True, max_samples=.1)
    ]
    for regr in estimators:
        regr.fit(X, y)

        # TEST 1 / Importance with groups of size 1
        groups = [[k] for k in range(n_features)]
        imp = grouped_importance(regr, X, y, groups)

        # Test the size of the output
        assert len(imp) == len(groups)
        
        # Test if the actual relevant features have a larger importance than the other
        assert min(imp[:4]) > max(imp[4:])

    # TEST 2 / Group the four most relevant features together
    groups = [[0, 1, 2, 3]] + [[k] for k in range(4, n_features)]
    imp = grouped_importance(regr, X, y, groups)

    # Test the size of the output
    assert len(imp) == len(groups)
    
    # Test if the actual relevant features have a larger importance than the other
    assert imp[0] > max(imp[1:])

    # TEST 3 / Dataset 2
    X, y = gaussian_multidimensional(c=.5, size=1000, n_features_per_groups=5)
    regr = RandomForestRegressor(n_estimators=100, oob_score=True, max_samples=.1)
    regr.fit(X, y)

    groups = [list(range(5)), list(range(5, 10))] + [[k] for k in range(10, 15)]
    imp = grouped_importance(regr, X, y, groups)
   
    # Test the size of the output
    assert len(imp) == len(groups)
    
    # Test if the actual relevant features have a larger importance than the other
    assert imp[0] > imp[1] > max(imp[2:])

def test_permute_group():
    """
    Test the function permute_group
    """
    X = np.arange(40).reshape((10, 4))
    new_x = permute_group(X, [1], n_repeats=5)
    assert np.array_equal(X[:, [0, 2, 3]], new_x[:, [0, 2, 3]])

    new_x = permute_group(X, [1, 3], n_repeats=5)
    assert np.array_equal(X[:, [0, 2]], new_x[:, [0, 2]])
    
    # Test if the order is the same for features 1 and 3
    assert np.array_equal(new_x[:, 1] + 2, new_x[:, 3])

    new_x = permute_group(X, [], n_repeats=5)
    assert np.array_equal(X, new_x)
