"""
DKN-RFM-AGOP — Deep Kernel Network with Scalable RFM kernel adaptation
and AGOP-based inter-layer compression.

Architecture per layer k (fully backprop-free):
─────────────────────────────────────────────────────────────────────
  1. AGOP phase  (T steps, convex oracle at each step):
       M_0 ← I_{d_in}
       for t = 1..T:
           ω ~ N(0, 2·M_{t-1})        anisotropic frequencies (Bochner)
           Φ = RFF(h; ω, b)           (n, D)
           w ← LS-SVM(Φ, y)           convex, closed form — discarded after
           M_t ← AGOP(f, h)           (d_in, d_in)
       Store final (ω_T, b_T).

  2. Kernel phase:
       h_rff = RFF(h; ω_T, b_T)      (n, D)  — converged anisotropic kernel

  3. Compression phase (AGOP of linear SVM in kernel feature space):
       w ← LS-SVM(h_rff, y)          (P, D) coefficient matrix
       M_compress = coef⊤ coef        (D, D)  AGOP of linear predictor on h_rff
       W = top-r eigvecs(M_compress)  (r, D)
       h ← h_rff @ W⊤                (n, r)  — next layer input
─────────────────────────────────────────────────────────────────────
Final layer: LS-SVM on h^(L) — kept for inference.

Relationship to existing models
--------------------------------
DKN-AGOP   : W is before the RFF (in input space); isotropic kernel;
             uses M only for eigenvector directions (eigenvalues discarded).
DKN-Align  : W is before the RFF (in input space); isotropic kernel;
             W learned by alignment, no AGOP phase.
DKN-RFM-AGOP (this): W is after the RFF (in kernel space); anisotropic
             kernel driven by converged AGOP M; W learned by AGOP on
             linear SVM in RFF feature space; entirely backprop-free.

Relationship to Radhakrishnan et al. (2022)
--------------------------------------------
The AGOP phase is a scalable RFF approximation of the RFM procedure.
RFM uses an exact kernel machine at each step — O(n²) storage, O(n³)
solve.  Here the exact kernel is replaced by RFF features — O(nD) per
step — making the AGOP convergence tractable for large n.

Known limitation
----------------
The AGOP-based compression W = top eigvecs(coef⊤ coef) is rank-P in
the kernel feature space.  For binary classification (P=1) this is
rank-1: all r rows of W collapse to the same direction.  For binary
tasks, DKN_RFM_Align (alignment-based W) is preferable.
"""

import numpy as np
from sklearn.linear_model import RidgeClassifier
from models.base import BaseModel


# ================================================================== #
#  Module-level helpers (shared with dkn_rfm_align.py)              #
# ================================================================== #

def _fit_lssvm(Phi: np.ndarray, y: np.ndarray, C: float) -> RidgeClassifier:
    """Fit a Least-Squares SVM via Ridge regression on Phi."""
    clf = RidgeClassifier(alpha=1.0 / (2.0 * C), fit_intercept=True)
    clf.fit(Phi, y)
    return clf


def _sample_anisotropic_rff(
    M:    np.ndarray,
    d_in: int,
    D:    int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample D RFF frequencies from N(0, 2·M_reg) and uniform random phases.

    The Mahalanobis kernel k(x,x') = exp(-(x-x')⊤M(x-x')) has spectral
    density N(0, 2M) by Bochner's theorem.  For M = γI this reduces
    exactly to the standard isotropic Gaussian RBF.

    M is regularised as M_reg = M + ε·(tr(M)/d)·I to ensure PD.

    Parameters
    ----------
    M    : (d_in, d_in) PSD AGOP matrix
    d_in : input dimension
    D    : number of random features
    seed : RNG seed (varied per layer and step for independence)

    Returns
    -------
    omega : (D, d_in)  anisotropic RFF frequencies
    bias  : (D,)       uniform random phases in [0, 2π]
    """
    rng = np.random.default_rng(seed)

    # Regularise: absolute eps relative to matrix scale
    trace = np.trace(M)
    eps   = 1e-6 * (trace / max(d_in, 1))
    M_reg = M + eps * np.eye(d_in)

    # Eigendecomposition: M_reg = U Λ U⊤
    eigvals, eigvecs = np.linalg.eigh(M_reg)            # ascending; cols of eigvecs
    sqrt_ev          = np.sqrt(np.maximum(eigvals, 0.0)) # (d_in,)

    # Sample ω_j = sqrt(2) · U · diag(sqrt(Λ)) · z_j,  z_j ~ N(0, I_d)
    # => Cov(ω_j) = 2 · U Λ U⊤ = 2 M_reg  ✓
    z     = rng.standard_normal((D, d_in))               # (D, d_in)
    omega = np.sqrt(2.0) * (z * sqrt_ev) @ eigvecs.T    # (D, d_in)
    bias  = rng.uniform(0.0, 2.0 * np.pi, D)             # (D,)

    return omega, bias


def _apply_rff(H: np.ndarray, omega: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    RFF feature map: Φ(H) = sqrt(2/D) · cos(H·ω⊤ + b).

    H     : (n, d_in)
    omega : (D, d_in)
    bias  : (D,)

    Returns Phi : (n, D)
    """
    D = omega.shape[0]
    return np.sqrt(2.0 / D) * np.cos(H @ omega.T + bias)


def _compute_agop_direct(
    H:     np.ndarray,
    omega: np.ndarray,
    bias:  np.ndarray,
    coef:  np.ndarray,
) -> np.ndarray:
    """
    Compute the AGOP of f(H) = coef⊤ Φ(H; ω, b) w.r.t. H directly.

    No W matrix — the predictor maps H through the RFF without an
    intermediate linear projection.  Compare with _compute_agop in
    dkn_agop.py which threads the gradient through a W matrix first.

    Gradient of class p w.r.t. sample H_i (chain rule):
        ∂f_p/∂H_i = (−sqrt(2/D)·sin(H_i·ω⊤+b) ⊙ coef_p) · ω   ∈ R^{d_in}

    AGOP = (1/n) Σ_i Σ_p (∂f_p/∂H_i)⊤ (∂f_p/∂H_i)   ∈ R^{d_in × d_in}

    Parameters
    ----------
    H     : (n, d_in)
    omega : (D, d_in)  RFF frequencies
    bias  : (D,)       RFF phases
    coef  : (P, D) or (D,)  oracle SVM coefficients — normalised to (P, D)

    Returns
    -------
    M : (d_in, d_in)  PSD AGOP matrix
    """
    coef    = np.atleast_2d(coef)               # (P, D)
    n, d_in = H.shape
    D       = omega.shape[0]

    Z       = H @ omega.T + bias                # (n, D)
    dPhi_dZ = -np.sqrt(2.0 / D) * np.sin(Z)    # (n, D)

    # weighted[p, i, d] = dPhi_dZ[i, d] * coef[p, d]
    weighted = coef[:, np.newaxis, :] * dPhi_dZ[np.newaxis, :, :]  # (P, n, D)

    # Chain rule: ∂f/∂H = weighted · ω    (no W in the path)
    grad_H = weighted @ omega                   # (P, n, d_in)

    M = np.einsum('pni,pnj->ij', grad_H, grad_H) / n   # (d_in, d_in)
    return M


# ================================================================== #
#  Model                                                             #
# ================================================================== #

class DKN_RFM_AGOP(BaseModel):
    """
    Deep Kernel Network — Scalable RFM kernel + AGOP-based compression.

    Parameters
    ----------
    n_layers   : number of (AGOP-phase → RFF → compress) blocks
    D          : RFF dimension (number of random features per layer)
    rank       : compression rank r; W ∈ R^(r × D)
    C          : LS-SVM regularisation for oracle SVMs and final layer
    agop_steps : AGOP iterations per layer (T).  Each step refines M.
                 Analogous to the number of RFM steps in Radhakrishnan
                 et al. (2022).  Typically 3–10; more does not always help.
    seed       : base RNG seed (varied per layer and step internally)
    """

    def __init__(
        self,
        n_layers:   int   = 2,
        D:          int   = 500,
        rank:       int   = 50,
        C:          float = 1.0,
        agop_steps: int   = 5,
        seed:       int   = 42,
    ):
        self.n_layers   = n_layers
        self.D          = D
        self.rank       = rank
        self.C          = C
        self.agop_steps = agop_steps
        self.seed       = seed

        # Filled during fit
        self.omegas_:    list[np.ndarray] = []
        self.biases_:    list[np.ndarray] = []
        self.Ws_:        list[np.ndarray] = []
        self.final_clf_: RidgeClassifier  = None

    # ---------------------------------------------------------------- #
    #  Step 1: AGOP phase                                               #
    # ---------------------------------------------------------------- #

    def _agop_phase(
        self,
        H:     np.ndarray,
        y:     np.ndarray,
        layer: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Iteratively refine M using anisotropic RFF-based oracle SVMs.

        Initialise M = I and run agop_steps iterations of:
            ω ~ N(0, 2M)  →  Φ = RFF(H; ω)  →  oracle SVM  →  M ← AGOP

        Starting from isotropy (I) and converging toward the task-adaptive
        Mahalanobis metric is analogous to RFM's kernel adaptation, but
        with RFF replacing the exact kernel matrix at each step.

        Returns
        -------
        M     : (d_in, d_in)  converged AGOP matrix
        omega : (D, d_in)     RFF frequencies from the final step
        bias  : (D,)          RFF phases from the final step
        """
        d_in = H.shape[1]
        M    = np.eye(d_in)        # isotropic initialisation

        omega = bias = None
        for t in range(self.agop_steps):
            omega, bias = _sample_anisotropic_rff(
                M, d_in, self.D,
                seed = self.seed + layer * 10_000 + t,
            )
            Phi    = _apply_rff(H, omega, bias)
            oracle = _fit_lssvm(Phi, y, self.C)
            M      = _compute_agop_direct(H, omega, bias, oracle.coef_)
            # oracle is discarded — training artifact only

        return M, omega, bias

    # ---------------------------------------------------------------- #
    #  Step 3: AGOP-based compression                                  #
    # ---------------------------------------------------------------- #

    def _agop_compress(
        self,
        h_rff: np.ndarray,
        y:     np.ndarray,
    ) -> np.ndarray:
        """
        Learn W by computing the AGOP of a linear SVM on h_rff.

        For the linear predictor f_p(h_rff) = w_p⊤ h_rff, the Jacobian
        is constant: ∂f_p/∂h_rff = w_p.  The full AGOP across all P
        classes is:

            M_compress = (1/n) Σ_i Σ_p w_p w_p⊤ = coef⊤ coef ∈ R^{D×D}

        W = top-r eigenvectors of M_compress.

        Note: for binary classification (P=1), coef⊤ coef is rank-1,
        and all r rows of W will point in the same direction.  In binary
        tasks, use DKN_RFM_Align instead, where W is unconstrained.

        Returns
        -------
        W : (r, D)
        """
        oracle = _fit_lssvm(h_rff, y, self.C)
        coef   = np.atleast_2d(oracle.coef_)   # (P, D)

        # AGOP of linear predictor in RFF feature space
        M_compress = coef.T @ coef              # (D, D),  rank = min(P, D)

        eigvals, eigvecs = np.linalg.eigh(M_compress)      # ascending
        r = min(self.rank, eigvecs.shape[1])
        W = eigvecs[:, -r:].T                              # (r, D)
        return W

    # ---------------------------------------------------------------- #
    #  Training                                                         #
    # ---------------------------------------------------------------- #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DKN_RFM_AGOP":
        """
        Greedy layer-wise training.

        For each layer k:
            1. AGOP phase → converged M and (ω_T, b_T)
            2. Apply converged anisotropic RFF: h_rff = RFF(H; ω_T, b_T)
            3. AGOP-based compression: W = top eigvecs(coef⊤ coef)
            4. Propagate: H ← h_rff @ W⊤
        Then fit final LS-SVM on H.
        """
        H = X.astype(np.float64)

        self.omegas_ = []
        self.biases_ = []
        self.Ws_     = []

        for k in range(self.n_layers):
            # Step 1 — AGOP phase
            _, omega, bias = self._agop_phase(H, y, layer=k)

            # Step 2 — apply converged anisotropic kernel
            h_rff = _apply_rff(H, omega, bias)              # (n, D)

            # Step 3 — AGOP-based compression
            W = self._agop_compress(h_rff, y)               # (r, D)

            self.omegas_.append(omega)
            self.biases_.append(bias)
            self.Ws_.append(W)

            H = h_rff @ W.T                                 # (n, r)

        # Final SVM — kept for inference
        self.final_clf_ = _fit_lssvm(H, y, self.C)
        return self

    # ---------------------------------------------------------------- #
    #  Inference                                                        #
    # ---------------------------------------------------------------- #

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Run X through all stored (ω, b, W) blocks."""
        H = X.astype(np.float64)
        for omega, bias, W in zip(self.omegas_, self.biases_, self.Ws_):
            H = _apply_rff(H, omega, bias) @ W.T
        return H

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.final_clf_.predict(self._forward(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.final_clf_.decision_function(self._forward(X))

    def get_params(self) -> dict:
        return dict(
            n_layers   = self.n_layers,
            D          = self.D,
            rank       = self.rank,
            C          = self.C,
            agop_steps = self.agop_steps,
        )
