"""
ML-SVM — wraps the original MLSVC / MLSVR from Acero-Pousa & Belanche
(ESANN 2025) behind the BaseModel interface.

The original logic (LinearSVC per layer, decision_function as inter-layer
signal) is preserved exactly.  The only algorithmic change is the gamma
estimation: the hand-crafted heuristic 1/(d·Var(X)) has been replaced by
sigest_gamma, which estimates the kernel bandwidth from the empirical
pairwise distance distribution of the data (Caputo et al., 2002 /
kernlab).  This provides a principled, data-adaptive bandwidth at the
same O(n) cost for the default frac=1 setting.

Bottleneck reminder: between layers the representation is compressed to
P dimensions (decision_function output), which is the structural problem
that DKN-AGOP and DKN-Alignment are designed to fix.
"""

import numpy as np
from sklearn.svm import LinearSVC
from models.base import BaseModel
from KernelApproximations.RFF_Gaussian import RFF_Gaussian
from KernelApproximations.sigest import sigest_gamma


class MLSVM(BaseModel):
    """
    Multi-Layer SVM classifier.

    Wraps the original MLSVC with the BaseModel interface.
    Uses RFF_Gaussian for kernel approximation and LinearSVC per layer.
    Gamma is estimated by sigest_gamma (median of the empirical pairwise
    distance distribution), replacing the previous 1/(d·Var(X)) heuristic.

    Parameters
    ----------
    n_layers    : number of SVM layers (>=1)
    n_components: RFF dimension per layer (D)
    sigest_frac : fraction of n to sample for sigest bandwidth estimation.
                  1.0 uses all pairs (recommended for n < 5000); reduce
                  for larger datasets.
    seed        : random seed passed to sigest for reproducibility.
    """

    def __init__(
        self,
        n_layers: int     = 2,
        n_components: int = 1000,
        sigest_frac: float = 1.0,
        seed: int = 42,
    ):
        self.n_layers      = n_layers
        self.n_components  = n_components
        self.sigest_frac   = sigest_frac
        self.seed          = seed

        # Filled during fit
        self.rffs_: list = [None] * n_layers
        self.clfs_: list = [None] * n_layers

    # ---------------------------------------------------------------- #
    #  Internal helpers                                                 #
    # ---------------------------------------------------------------- #

    def _compute_gamma(self, X: np.ndarray, layer: int) -> float:
        """
        Estimate gamma for an RBF kernel on the given representation X.

        Uses sigest_gamma (median quantile) with a layer-varied seed so
        that each layer's sampling is independent.
        """
        return sigest_gamma(
            X,
            frac=self.sigest_frac,
            scaled=False,   # X is already standardised by the outer benchmark loop
            quantile=0.5,
            seed=self.seed + layer,
        )

    def _make_rff(self, X: np.ndarray, layer: int) -> RFF_Gaussian:
        """Build and fit an RFF_Gaussian with sigest-estimated gamma."""
        rff = RFF_Gaussian(
            gamma=self._compute_gamma(X, layer),
            n_components=self.n_components,
        )
        return rff

    # ---------------------------------------------------------------- #
    #  fit                                                              #
    # ---------------------------------------------------------------- #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLSVM":
        """
        Greedy layer-wise training (mirrors original MLSVC.fit).

        Before the loop: X -> RFF (gamma estimated by sigest)
        Each hidden layer: RFF(H) -> LinearSVC -> decision_function -> H'
        Final layer:       RFF(H) -> LinearSVC  (kept for predict)
        """
        H = X.astype(np.float64)

        # First RFF (applied before the layer loop, as in original)
        rff = self._make_rff(H, layer=0)
        H   = rff.fit_transform(H)
        self.rffs_[0] = rff

        for layer in range(self.n_layers):
            clf = LinearSVC(max_iter=2000)
            clf.fit(H, y)
            self.clfs_[layer] = clf

            if layer < self.n_layers - 1:
                # Bottleneck: H collapses to P (or 1) dimensions
                H = clf.decision_function(H)
                if H.ndim == 1:
                    H = H[:, np.newaxis]

                rff = self._make_rff(H, layer=layer + 1)
                H   = rff.fit_transform(H)
                self.rffs_[layer + 1] = rff

        return self

    # ---------------------------------------------------------------- #
    #  predict                                                          #
    # ---------------------------------------------------------------- #

    def predict(self, X: np.ndarray) -> np.ndarray:
        H = X.astype(np.float64)
        H = self.rffs_[0].transform(H)

        for layer in range(self.n_layers):
            if layer < self.n_layers - 1:
                H = self.clfs_[layer].decision_function(H)
                if H.ndim == 1:
                    H = H[:, np.newaxis]
                H = self.rffs_[layer + 1].transform(H)
            else:
                return self.clfs_[layer].predict(H)

    # ---------------------------------------------------------------- #
    #  metadata                                                         #
    # ---------------------------------------------------------------- #

    def get_params(self) -> dict:
        return dict(
            n_layers=self.n_layers,
            n_components=self.n_components,
            sigest_frac=self.sigest_frac,
        )
