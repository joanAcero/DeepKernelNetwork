"""
MLP baseline — Multi-Layer Perceptron via scikit-learn.

Same depth as the DKN models being compared, with ReLU activations
and the same regularisation strength (alpha = 1/(2C)).
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from models.base import BaseModel


class MLPBaseline(BaseModel):
    """
    Multi-Layer Perceptron trained by backpropagation (Adam).

    Parameters
    ----------
    hidden_layer_sizes : tuple of ints — neurons per hidden layer
    alpha              : L2 regularisation (sklearn convention)
    max_iter           : maximum training epochs
    learning_rate_init : initial Adam step size
    seed               : random state
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (256, 256),
        alpha: float = 1e-4,
        max_iter: int = 500,
        learning_rate_init: float = 1e-3,
        seed: int = 42,
    ):
        self.hidden_layer_sizes  = hidden_layer_sizes
        self.alpha               = alpha
        self.max_iter            = max_iter
        self.learning_rate_init  = learning_rate_init
        self.seed                = seed

        self.clf_ = MLPClassifier(
            hidden_layer_sizes  = hidden_layer_sizes,
            activation          = "relu",
            solver              = "adam",
            alpha               = alpha,
            max_iter            = max_iter,
            learning_rate_init  = learning_rate_init,
            early_stopping      = True,
            validation_fraction = 0.1,
            random_state        = seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPBaseline":
        self.clf_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf_.predict(X)

    def get_params(self) -> dict:
        return dict(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            max_iter=self.max_iter,
        )
