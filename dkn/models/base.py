from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Abstract interface that every model in this project must implement.
    Mirrors the scikit-learn API so models can be dropped into the
    benchmark loop without modification.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """
        Train the model on (X, y).

        Parameters
        ----------
        X : (n, d) float array
        y : (n,) int array — class labels starting at 0

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted class labels for X.

        Parameters
        ----------
        X : (n, d) float array

        Returns
        -------
        y_pred : (n,) int array
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy on (X, y). Override if you need a different metric."""
        return float(np.mean(self.predict(X) == y))

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v}" for k, v in self.get_params().items()
        )
        return f"{self.__class__.__name__}({params})"

    def get_params(self) -> dict:
        """Return constructor hyperparameters for logging. Override as needed."""
        return {}
