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
