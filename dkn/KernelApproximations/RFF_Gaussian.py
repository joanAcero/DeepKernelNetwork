# Author: Joan Acero Pousa
# Date: 05/05/2024
# Reference: Random Fourier Features for Kernel Approximation, Rahimi and Recht, 2007
# Adapted from: Scikit-learn


import numpy as np


class RFF_Gaussian():
    """
    Generates a kernel approximation using Random Fourier Features (RFF)
    for the RBF (Gaussian) kernel.

    The approximation is:
        phi(x) = sqrt(2/D) * cos(x @ w.T + b)
    where w ~ N(0, 2*gamma*I) and b ~ U[0, 2*pi].

    This satisfies E[phi(x) @ phi(y)] ≈ exp(-gamma * ||x - y||^2).
    """

    def __init__(self, n_components: int = 1000, gamma: float = 1.0,
                 random_state=None):
        self.gamma        = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.fitted       = False

    def fit(self, X: np.ndarray) -> "RFF_Gaussian":
        if self.random_state is not None:
            np.random.seed(self.random_state)

        try:
            self.n_features = X.shape[1]
        except IndexError:
            self.n_features = 1

        self.w = np.random.multivariate_normal(
            mean=np.zeros(self.n_features),
            cov=2 * self.gamma * np.eye(self.n_features),
            size=self.n_components,
        )
        self.b      = np.random.uniform(0, 2 * np.pi, self.n_components)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("RFF_Gaussian has not been fitted yet.")
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return (2.0 / self.n_components) ** 0.5 * np.cos(X @ self.w.T + self.b)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def get_kernel(self):
        D = self.n_components
        def ker(x, y):
            z1 = np.cos(x @ self.w.T + self.b)
            z2 = np.cos(y @ self.w.T + self.b)
            return np.sqrt(2 / D) * (z1 @ z2.T)
        return ker
