"""
Recursive Feature Elimination algorithm based on the grouped importance measure defined in https://arxiv.org/abs/1411.4170
"""

import numpy as np

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y

from rfgroove.evaluation import error
from rfgroove.importance import grouped_importance

class RFE():
    def __init__(self, estimator, groups, test_size=.2, n_repeats=5, n_jobs=None, verbose=False):
        """
        Constructor

        Args:
            estimator: the base estimator, e.g. RandomForestRegressor or ExtraTreesRegressor
            groups: a list of lists, the indices of the groupes of features
            test_size: the proportion of the data that will be used in the test sample, default: .2
            n_repeats: number of repeats in the permutation, default: 5
            n_jobs: the number of jobs to run in parallel. None means no parallel computing and -1 means using all processors. Default: None
            verbose: print some comments during the main loop, default: False
        """
        self.estimator = estimator
        self.groups = groups
        self.test_size = test_size
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.verbose = verbose

        if test_size < 0 or test_size > 1:
            raise ValueError("The argumetn 'test_size' should be between 0 and 1")

        if not estimator.max_samples:
            print("The attribute 'max_samples' should not be None")
    
    def fit(self, X, y):
        """
        Run the selection algorithm

        Args:
            X: features data
            y: target data
        """

        X, y = check_X_y(X, y)

        # Initialization
        n_features = X.shape[1]
        support = np.ones(n_features, dtype=bool)

        # Train/Test split for computing the error
        X, X_test, y, y_test = train_test_split(X, y, test_size=self.test_size)

        # Store
        self.history = []

        # Elimination
        while np.sum(support):

            # Remaining features
            features = np.arange(n_features)[support]

            # Clone the estimator and train the model using the activated features, via the support
            estimator = clone(self.estimator)
            estimator.fit(X[:, features], y)

            # Compute the error
            predictions = estimator.predict(X_test[:, features])
            error_rate = error(estimator, y_test, predictions)

            # Store the error
            self.history.append({"n_features": np.sum(support), "n_groups": len(self.groups), "error": error_rate, "support": list(support)})

            if self.verbose > 0:
                print(f"Fitting estimator with {np.sum(support):2} features. Error rate: {error_rate:.4f}")

            # Compute the importance of each group of features
            importances = grouped_importance(model=estimator,
                                             X=X,
                                             y=y,
                                             groups=self.groups,
                                             n_repeats=self.n_repeats,
                                             n_jobs=self.n_jobs)
            ranks = np.argsort(importances)

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            # Eliminate the features given by the worst group
            indices = self.groups[ranks[0]]
            support[indices] = False
            self.groups.pop(ranks[0])

        # Find the lower error rate and the corresponding support
        errors = [item.get("error") for item in self.history]
        optimum = np.argmin(errors)
        support = self.history[optimum].get("support")
        features = np.arange(n_features)[support]

        # Build the final model
        self.estimator = clone(self.estimator)
        self.estimator.fit(X[:, features], y)
        self.n_features = self.history[optimum].get("n_features")
        self.support = support

        return self

if __name__ == "__main__":

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_sparse_uncorrelated
    from rfgroove.dataset_generation import gaussian_multidimensional

    # X, y = make_sparse_uncorrelated(n_samples=100, n_features=10, random_state=0)
    # groups = groups = [[0, 1, 2, 3]] + [[k] for k in range(4, 10)]

    X, y = gaussian_multidimensional(c=.5, size=1000, n_features_per_groups=5)
    groups = [list(range(5)), list(range(5, 10))] + [[k] for k in range(10, 15)]

    base = RandomForestRegressor(n_estimators=1000, bootstrap=True, oob_score=True, max_samples=.1)
    selector = RFE(base, groups, n_jobs=-1)
    selector.fit(X, y)

