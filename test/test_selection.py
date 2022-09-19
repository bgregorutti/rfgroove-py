"""
Test the module selection.py
"""
from sklearn.ensemble import RandomForestRegressor
from rfgroove.dataset_generation import gaussian_multidimensional
from rfgroove.selection import RFE

def test_RFE():
    """
    Test the class RFE
    """

    # Sample two groups of 5 correlated features and 5 indepedant features
    # The two groups are relevant for the prediction of the outcome
    X, y = gaussian_multidimensional(c=.5, size=1000, n_features_per_groups=5)
    groups = [list(range(5)), list(range(5, 10))] + [[k] for k in range(10, 15)]

    base = RandomForestRegressor(n_estimators=1000, bootstrap=True, oob_score=True, max_samples=.1)
    selector = RFE(base, groups, n_jobs=-1)
    selector.fit(X, y)
