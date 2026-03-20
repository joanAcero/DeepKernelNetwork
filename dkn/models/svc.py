"""
SVC baseline — Gaussian RBF kernel SVM via scikit-learn.

Provides the standard single-layer kernel SVM reference point.
Scales as O(n²) in memory and O(n³) in training time, so use
only on datasets where n ≤ ~20 000.
"""

import numpy as np
from sklearn.svm import SVC as _SVC
from models.base import BaseModel


class SVCBaseline(BaseModel):
    """
    Single-layer kernel SVM (Gaussian RBF).

    Parameters
    ----------
    C     : regularisation parameter
    gamma : kernel coefficient ('scale', 'auto', or float)
    """

    def __init__(self, C: float = 1.0, gamma: str | float = "scale"):
        self.C     = C
        self.gamma = gamma
        self.clf_  = _SVC(kernel="rbf", C=C, gamma=gamma)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVCBaseline":
        self.clf_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf_.predict(X)

    def get_params(self) -> dict:
        return dict(C=self.C, gamma=self.gamma)
