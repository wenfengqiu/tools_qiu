import numpy as np


class FakeModel:
    """
    A fake model with fit/predict method
    used in double machine learning/residualizing for t
    when t is already randomized
    to save resources
    """

    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y):
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() before predict().")

        return np.zeros(len(X))
