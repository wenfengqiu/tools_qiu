import numpy as np


class FakeModel:
    """
    A fake model with fit/predict method
    used in double machine learning/residualizing for t
    when t is already randomized
    to save resources
    """

    def __init__(self, prob_treatment):
        self.is_fitted = False
        self.prob_treatment = prob_treatment

    def fit(self, X, y):
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() before predict().")

        return np.zeros(len(X))

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() before predict().")

        array_0 = np.repeat(1 - self.prob_treatment, len(X))
        array_1 = np.repeat(self.prob_treatment, len(X))
        result = np.column_stack((array_0, array_1))

        return result
