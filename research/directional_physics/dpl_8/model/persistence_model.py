import numpy as np


class PersistenceModel:
    def __init__(self, alpha=1e-3):
        self.alpha = alpha
        self.W = None

    def fit(self, X, y):
        XTX = X.T @ X
        reg = self.alpha * np.eye(X.shape[1])
        self.W = np.linalg.inv(XTX + reg) @ X.T @ y

    def predict(self, X):
        if self.W is None:
            return np.zeros(X.shape[0])
        return X @ self.W
