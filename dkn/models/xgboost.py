"""
XGBoost baseline — gradient-boosted trees.

State-of-the-art on tabular benchmarks (Grinsztajn et al. 2022).
Serves as the strongest non-neural reference point.

Requires: xgboost (pip install xgboost)
"""

import numpy as np
from models.base import BaseModel

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


class XGBoostBaseline(BaseModel):
    """
    XGBoost gradient-boosted tree ensemble.

    Parameters
    ----------
    n_estimators  : number of boosting rounds
    max_depth     : maximum tree depth
    learning_rate : shrinkage factor per round
    subsample     : row subsampling ratio per tree
    seed          : random state
    """

    def __init__(
        self,
        n_estimators: int  = 500,
        max_depth: int     = 6,
        learning_rate: float = 0.05,
        subsample: float   = 0.8,
        seed: int          = 42,
    ):
        if not _XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is not installed. Run: pip install xgboost"
            )
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.subsample     = subsample
        self.seed          = seed

        self.clf_ = XGBClassifier(
            n_estimators   = n_estimators,
            max_depth      = max_depth,
            learning_rate  = learning_rate,
            subsample      = subsample,
            use_label_encoder = False,
            eval_metric    = "logloss",
            random_state   = seed,
            verbosity      = 0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
        self.clf_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf_.predict(X)

    def get_params(self) -> dict:
        return dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
        )
