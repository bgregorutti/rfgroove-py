"""
Grouped permutation importance module
"""

from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

def error(model, true, pred):
    """
    Compute the error: MSE for regression and class error rate for classification

    Args:
        model: an object from scikit-learn ensemble estimator
        true: the true values
        pred: the predicted values
    
    Returns:
        The model error
    """
    if hasattr(model, "classes_"):
        return 1 - accuracy_score(true, pred)
    else:
        return mean_squared_error(true, pred)

def get_oob_samples(n, random_state, max_samples):
    """
    Get the Out-Of-Bag indices

    Args:
        n: the number of samples in the full dataset
        random_state: the randomness of the bootstrap samples
        max_samples: maximum number of samples in the bootstrap sets
    
    Returns:
        Numpy array of indices indicates tha OOB sample
    """
    # Get the inbag sample indexes
    seed = np.random.RandomState(random_state)
    nr_samples = max_samples if max_samples > 1 else int(max_samples * n)
    inbag = seed.randint(0, n, nr_samples)

    # The get the OOB indexes
    sample_counts = np.bincount(inbag, minlength=n)
    oob_mask = sample_counts == 0
    
    return np.arange(n)[oob_mask]

def permute_group(X, group, n_repeats):
    """
    Permute the values of a group of features together

    Args:
        X: features data
        group: a list of lists, the indices of the groupes of features
        n_repeats: number of repeats in the permutation
    
    Returns
        Numpy array with permuted values within a group of features
    """
    if not group:
        print("group argument is empty. No features will be permuted")
    n = X.shape[0]
    rng = np.random.default_rng()
    permuted_indexes = rng.permuted(range(n))
    
    # Repeat the permutation if needed
    for _ in range(n_repeats):
        permuted_indexes = rng.permuted(permuted_indexes)

    cpy = X.copy()
    for idx in group:
        cpy[:, idx] = cpy[permuted_indexes, idx]
    return cpy

def grouped_importance(model, X, y, groups, n_repeats=5, n_jobs=None):
    """
    Grouped permutation importance

    Args:
        model: an object from scikit-learn ensemble estimator
        X: features data
        y: target data
        groups: a list of lists, the indices of the groupes of features
        n_repeats: number of repeats in the permutation, default: 5
        n_jobs: the number of jobs to run in parallel. None means no parallel computing and -1 means using all processors. Default: None
    
    Returns:
        Numpy array of the importance of each group
    """

    if not n_jobs or n_jobs == 1:
        importances = np.array([grouped_importance_tree(tree, X, y, groups, n_repeats, model.max_samples) for tree in model.estimators_])
    else:
        importances = Parallel(n_jobs=n_jobs, verbose=1)(delayed(grouped_importance_tree)(tree, X, y, groups, n_repeats, model.max_samples) for tree in model.estimators_)

    return np.mean(importances, axis=0)

def grouped_importance_tree(tree, X, y, groups, n_repeats, max_samples):
    """
    Grouped permutation importance of a tree

    Args:
        model: an object from scikit-learn ensemble estimator
        X: features data
        y: target data
        groups: a list of lists, the indices of the groupes of features
        n_repeats: number of repeats in the permutation, default: 5
        max_samples: maximum number of samples in the bootstrap sets
    
    Returns:
        Numpy array of the importance of each group for the current tree
    """
    # Get the OOB samples
    oob_indexes = get_oob_samples(X.shape[0], tree.random_state, max_samples)
    X_oob = X[oob_indexes]
    y_oob = y[oob_indexes]

    # Predict with the current tree and compute the error
    pred = tree.predict(X_oob)
    err = error(tree, y_oob, pred)

    # Loop over the groups
    importance_tree = []
    for group in groups:
        # Permute the group, predict and compute the error again
        X_permuted = permute_group(X_oob, group, n_repeats=n_repeats)
        pred = tree.predict(X_permuted)
        err_permuted = error(tree, y_oob, pred)
        
        # Compute the error increase for the current tree
        importance_tree.append(err_permuted - err)
    return importance_tree
