from deel.puncc.metrics import regression_sharpness, regression_mean_coverage

# Function to evaluate the sharpness and coverage of a conformal prediction model
def evaluate_cp(X_test, y_test, model_cp, alpha):
    """
    Evaluate the performance of a model using conformal prediction.

    Parameters:
    - X_test : The input features for the test set.
    - y_test : The true labels for the test set.
    - model_cp : The conformal prediction model.
    - alpha : The maximum risk level for the prediction intervals.

    Returns:
    - sharpness : The average width of the prediction intervals.
    - coverage : The average coverage of the prediction intervals.
    """
    y_pred, y_pred_lower, y_pred_upper = model_cp.predict(X_test, alpha=alpha)
    sharpness = regression_sharpness(y_pred_lower, y_pred_upper)
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    return sharpness, coverage
