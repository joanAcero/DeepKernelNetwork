"""
DKN-AGOP — Deep Kernel Network trained with greedy AGOP updates.

Architecture (inference path, no bottleneck):
    x  →  [W¹ → RFF¹]  →  [W² → RFF²]  →  ...  →  [Wᴸ → RFFᴸ]  →  SVM_final  →  ŷ

Training procedure (layer by layer, backprop-free):
    For each layer k:
        1. Forward through block k:  Phi = RFF(W·H)   [RFF re-fitted at each step]
        2. Fit oracle LS-SVM on (Phi, y)  — convex, closed form
        3. Compute AGOP of that SVM:   M = (1/n) Σ Jᵢᵀ Jᵢ  (Jacobian, per class)
        4. Set W^(k+1) = top-r eigenvectors of M
        5. Discard oracle SVM — it is never used at inference time
    Final layer: fit LS-SVM on last representation — kept for inference.

Key property: each sub-problem is convex; no bottleneck in the
inference path; backpropagation is not required.

Changelog vs. original:
    FIX 1 — RFF re-fitted at every AGOP iteration.
        The original code fitted the RFF once on the initial W·H and then
        kept it frozen while W was updated across agop_steps iterations.
        After step t=0, W changes but the RFF frequencies ω were calibrated
        to the old W·H, so the kernel bandwidth becomes inconsistent with
        the new projection. The fix re-fits the RFF at the start of each
        AGOP step, so ω is always calibrated to the current W·H.

    FIX 2 — Correct multiclass AGOP (full Jacobian outer product).
        The original code collapsed the P-class coefficient matrix to a
        single vector by summing over classes before forming the weighted
        gradient, which is only correct for binary classification.  For
        P > 2 class contributions can cancel, producing a biased AGOP.
        The fix computes M = (1/n) Σᵢ Jᵢᵀ Jᵢ correctly via einsum.

    FIX 3 — sigest_gamma replaces the 1/(d·Var(X)) heuristic.
        Gamma is now estimated from the empirical pairwise distance
        distribution at each RFF construction, providing a principled,
        data-adaptive bandwidth consistent with the kernlab standard.
"""

import numpy as np
from sklearn.linear_model import RidgeClassifier
from models.base import BaseModel
from KernelApproximations.RFF_Gaussian import RFF_Gaussian
from KernelApproximations.sigest import sigest_gamma


def _fit_lssvm(Phi: np.ndarray, y: np.ndarray, C: float) -> RidgeClassifier:
    """
    Fit a Least-Squares SVM (Ridge classifier) on RFF features Phi.

    The LS-SVM primal is equivalent to Ridge regression on Phi:
        min_{w}  (1/2)||w||² + C·Σ ξᵢ²
    which becomes sklearn's RidgeClassifier with alpha = 1/(2C).
    """
    clf = RidgeClassifier(alpha=1.0 / (2.0 * C), fit_intercept=True)
    clf.fit(Phi, y)
    return clf


def _compute_agop(
    H: np.ndarray,
    W: np.ndarray,
    omega: np.ndarray,
    bias: np.ndarray,
    coef: np.ndarray,
) -> np.ndarray:
    """
    Compute the Average Gradient Outer Product (AGOP) of the oracle SVM
    with respect to the block input H.

    The oracle predictor is:  f(H) = Phi(W·H) @ coef.T
    where  Phi(Z) = sqrt(2/D) · cos(Z·omega.T + bias).

    For a P-class predictor, f: ℝ^{d_in} → ℝ^P and the Jacobian at sample i
    is Jᵢ ∈ ℝ^{P × d_in}.  The AGOP is:

        M = (1/n) Σᵢ Jᵢᵀ Jᵢ = (1/n) Σᵢ Σ_p (∂f_p/∂Hᵢ)ᵀ (∂f_p/∂Hᵢ)

    Parameters
    ----------
    H     : (n, d_in)         input to this block
    W     : (rank, d_in)     current weight matrix
    omega : (D, rank)        RFF frequencies  (rff.w)
    bias  : (D,)              RFF phases       (rff.b)
    coef  : (P, D) or (1, D)  oracle SVM coefficients (clf.coef_)

    Returns
    -------
    M : (d_in, d_in)  AGOP matrix (symmetric, positive semi-definite)
    """
    n, d_in = H.shape
    D = omega.shape[0]

    # sklearn's RidgeClassifier returns coef_ with shape (D,) for binary
    # problems (single regression target) and (P, D) for multiclass.
    # Normalise to (P, D) so the broadcasting below is always correct.
    coef = np.atleast_2d(coef)                  # (P, D) in all cases

    WH = H @ W.T                                # (n, rank)
    Z  = WH @ omega.T + bias                    # (n, D)

    # dPhi/dZ element-wise:  -sqrt(2/D) * sin(Z)
    dPhi_dZ = -np.sqrt(2.0 / D) * np.sin(Z)    # (n, D)

    # FIX 2: correct multiclass AGOP via full Jacobian outer product.
    # weighted[p, i, d] = dPhi_dZ[i, d] * coef[p, d]
    weighted = coef[:, np.newaxis, :] * dPhi_dZ[np.newaxis, :, :]  # (P, n, D)

    # Chain rule through Z = WH · omega.T  →  grad w.r.t. WH
    grad_WH = weighted @ omega                   # (P, n, rank)

    # Chain rule through WH = H · W.T  →  grad w.r.t. H
    grad_H = grad_WH @ W                        # (P, n, d_in)

    # AGOP: accumulate outer products over samples (n) and classes (P).
    M = np.einsum('pni,pnj->ij', grad_H, grad_H) / n   # (d_in, d_in)
    return M


class DKN_AGOP(BaseModel):
    """
    Deep Kernel Network — AGOP training regime.

    Parameters
    ----------
    n_layers    : number of W→RFF blocks
    D           : RFF dimension per block
    rank        : number of top AGOP eigenvectors kept as W rows.
                  W^(k) ∈ R^(rank × d_in) at every layer; the RFF at each
                  block therefore operates in a rank-dimensional projected
                  space.  d_k has been removed as a separate parameter —
                  it was redundant and the previous d_k > rank design silently
                  zero-padded W, wasting RFF capacity and compounding noise
                  in deep architectures.
    C           : LS-SVM regularisation per oracle and final SVM
    agop_steps  : number of AGOP update steps per layer (1 is theoretically
                  grounded by Gan & Poggio 2024)
    sigest_frac : fraction of n to sample for sigest bandwidth estimation
    seed        : RNG seed
    """

    def __init__(
        self,
        n_layers: int = 2,
        D: int = 500,
        rank: int = 50,
        C: float = 1.0,
        agop_steps: int = 1,
        sigest_frac: float = 1.0,
        seed: int = 42,
    ):
        self.n_layers     = n_layers
        self.D            = D
        self.rank         = rank
        self.C            = C
        self.agop_steps   = agop_steps
        self.sigest_frac  = sigest_frac
        self.seed         = seed

        # Set during fit
        self.weights_: list[np.ndarray]   = []
        self.rffs_:    list[RFF_Gaussian]  = []
        self.final_clf_: RidgeClassifier   = None

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _compute_gamma(self, X: np.ndarray, layer: int, step: int) -> float:
        """
        FIX 3: estimate gamma from the empirical pairwise distance
        distribution of X using sigest (median quantile).

        scaled=False because X (which is W·H) is a linear projection of
        an already-standardised input; re-scaling introduces distortion.
        The seed varies by (layer, step) for reproducibility.
        """
        return sigest_gamma(
            X,
            frac=self.sigest_frac,
            scaled=False,
            quantile=0.5,
            seed=self.seed + layer * 1000 + step,
        )

    def _make_rff(self, X: np.ndarray, layer: int, step: int) -> RFF_Gaussian:
        """
        Build and fit an RFF_Gaussian.
        Gamma is estimated by sigest on X = W·H (current projection).
        """
        rff = RFF_Gaussian(
            gamma=self._compute_gamma(X, layer, step),
            n_components=self.D,
            random_state=self.seed + layer * 1000 + step,
        )
        rff.fit(X)
        return rff

    def _init_W(self, d_in: int) -> np.ndarray:
        """
        Initialise W ∈ ℝ^(rank × d_in) as an identity-like matrix.

        rank plays the role of d_k — there is no separate d_k parameter.
        The identity initialisation copies the first min(rank, d_in)
        dimensions unchanged into the projected space.
        """
        r = min(self.rank, d_in)
        W = np.zeros((self.rank, d_in))
        W[:r, :r] = np.eye(r)
        return W

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DKN_AGOP":
        """
        Greedy AGOP training.

        For each layer k, repeat agop_steps times:
            1. WH   = H @ W.T
            2. rff  = RFF fitted on WH          [FIX 1 + FIX 3: re-fitted
                                                  each step with sigest gamma]
            3. Phi  = rff.transform(WH)
            4. SVM  = fit_lssvm(Phi, y)
            5. M    = AGOP(SVM, H, W, rff)      [FIX 2: full Jacobian]
            6. W    = top-rank eigvecs of M

        The rff stored for each layer is the one from the final step,
        ensuring consistency between the stored W and the stored RFF at
        inference time.
        """
        H    = X.astype(np.float64)
        d_in = H.shape[1]

        self.weights_ = []
        self.rffs_    = []

        W = self._init_W(d_in)

        for k in range(self.n_layers):
            rff = None

            for t in range(self.agop_steps):
                # FIX 1 + FIX 3: re-fit RFF at every step with sigest gamma
                WH  = H @ W.T                                       # (n, d_k)
                rff = self._make_rff(WH, layer=k, step=t)
                Phi = rff.transform(WH)                             # (n, D)

                oracle = _fit_lssvm(Phi, y, self.C)

                M = _compute_agop(H, W, rff.w, rff.b, oracle.coef_)

                eigvals, eigvecs = np.linalg.eigh(M)                # ascending
                r      = min(self.rank, eigvecs.shape[1])
                W      = eigvecs[:, -r:].T                          # (r, d_in)
                # No padding: W is exactly (r, d_in).  When r < rank
                # (can only happen if d_in < rank), W is simply smaller.

            # After the AGOP loop, W may have fewer rows than self.rank
            # (this always happens when rank > d_in, since the AGOP matrix M
            # is d_in × d_in and can yield at most d_in eigenvectors).
            # The rff from the last loop iteration was fitted on H @ W_old.T
            # and no longer matches the final W.  Re-fit with the definitive
            # W so the stored (W, rff) pair is consistent for forward
            # propagation here and at inference time.
            WH_final = H @ W.T
            rff      = self._make_rff(WH_final, layer=k, step=self.agop_steps)

            self.weights_.append(W)
            self.rffs_.append(rff)

            # Propagate representation forward with the consistent (W, rff)
            H    = rff.transform(WH_final)
            d_in = H.shape[1]
            W    = self._init_W(d_in)

        # Final SVM — kept for inference
        self.final_clf_ = _fit_lssvm(H, y, self.C)
        return self

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def _forward(self, X: np.ndarray) -> np.ndarray:
        H = X.astype(np.float64)
        for W, rff in zip(self.weights_, self.rffs_):
            H = rff.transform(H @ W.T)
        return H

    def predict(self, X: np.ndarray) -> np.ndarray:
        H = self._forward(X)
        return self.final_clf_.predict(H)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        H = self._forward(X)
        return self.final_clf_.decision_function(H)

    def get_params(self) -> dict:
        return dict(
            n_layers=self.n_layers,
            D=self.D,
            rank=self.rank,
            C=self.C,
            agop_steps=self.agop_steps,
            sigest_frac=self.sigest_frac,
        )
