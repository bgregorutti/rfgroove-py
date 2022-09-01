"""
Generate datasets for testing
"""
import numpy as np

def gaussian_3d(c, tau, size):
    """
    3D Gaussian sampling corresponding to (X_1, X_2, Y) with correlations
    """
    cov = np.array([[1, c, tau], [c, 1, tau], [tau, tau, 1]])
    mean = np.zeros(3)
    M = np.random.multivariate_normal(mean, cov, size)
    return M[:, :2], M[:, 2]

def gaussian_multidimensional(c, size, n_features_per_groups=5):
    """
    n-dimensional Gaussian sampling corresponding to (X, Y), Two blocks of correlated features and one block of independent features.
    The features with a group have the same correlation with the target

    Args:
        c: the intra block correlation
        size: the size of the dataset
    
    Returns
        X and y numpy arrays
    """
    cov = np.eye(n_features_per_groups*3+1)
    
    # Add the blocks
    cov[:n_features_per_groups, :n_features_per_groups] = correlation_matrix(n_features_per_groups, c)
    cov[n_features_per_groups:2*n_features_per_groups, n_features_per_groups:2*n_features_per_groups] = correlation_matrix(n_features_per_groups, c)

    # Add the correlation with the target for group 1
    cov[-1, :n_features_per_groups] = .7
    cov[:n_features_per_groups, -1] = .7
    
    # Add the correlation with the target for group 2
    cov[-1, n_features_per_groups:2*n_features_per_groups] = .5
    cov[n_features_per_groups:2*n_features_per_groups, -1] = .5

    mean = np.zeros(n_features_per_groups*3+1)
    M = np.random.multivariate_normal(mean, cov, size)
    return M[:, :-1], M[:, -1]

def correlation_matrix(n, c):
    """
    Return a correlation matrix

    Args:
        n: the nimber of features
        c: the correlation between the features
    
    Returns:
        Numpy array of shape (n, n)
    """
    mat = np.eye(n)
    mat[mat==0] = c
    return mat
