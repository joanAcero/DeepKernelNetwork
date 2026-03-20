"""
DKN-RFM-Align — Deep Kernel Network with Scalable RFM kernel adaptation
and alignment-based inter-layer compression.

Architecture per layer k:
─────────────────────────────────────────────────────────────────────
  1. AGOP phase  (T steps, fully backprop-free — shared with DKN-RFM-AGOP):
       M_0 ← I_{d_in}
       for t = 1..T:
           ω ~ N(0, 2·M_{t-1})        anisotropic frequencies
           Φ = RFF(h; ω, b)           (n, D)
           w ← LS-SVM(Φ, y)           convex, closed form — discarded
           M_t ← AGOP(f, h)           (d_in, d_in)
       Store final (ω_T, b_T).

  2. Kernel phase:
       h_rff = RFF(h; ω_T, b_T)      (n, D)  — converged anisotropic kernel
       h_rff is FIXED for step 3: no gradient flows through the RFF.

  3. Compression phase (alignment-based, within-layer Adam):
       W ∈ R^{r × D}  initialised as top-r PCA of h_rff
       Optimise W to maximise kernel-target alignment of W·h_rff with y:

           A(W) = ‖(W h_rff⊤) Y‖²_F / ‖W h_rff⊤ h_rff W⊤‖_F

       Gradient flows through W only (h_rff is constant — no RFF inside).
       Adam, minibatch, within-layer only.  No global backprop.
       h ← h_rff @ W⊤   (n, r)  — next layer input
─────────────────────────────────────────────────────────────────────
Final layer: LS-SVM on h^(L) — kept for inference.

Why this differs from DKN-Align
---------------------------------
DKN-Align:   W before isotropic RFF.  Alignment of RFF(W·h) with y —
             W is inside the non-linear RFF map; gradient is non-trivial.
             No AGOP phase; gamma from sigest.
DKN-RFM-Align:  W after anisotropic RFF.  Alignment of W·h_rff with y —
             h_rff is FIXED; gradient through W is purely linear.
             Explicit AGOP convergence phase drives the kernel metric.

The alignment objective is also structurally different:
  DKN-Align:     A(Φ(W·h), y)   — kernel is a function of W
  DKN-RFM-Align: A(W·h_rff, y)  — kernel is fixed; W compresses features

Advantage over DKN-RFM-AGOP for binary tasks
----------------------------------------------
DKN-RFM-AGOP's AGOP-based W = eigvecs(coef⊤coef) is rank-1 for binary
classification (P=1) — all rows of W collapse to one direction.  Here W
is an unconstrained (r × D) matrix optimised by alignment, which is
well-defined for any number of classes and any rank r.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelBinarizer
from models.base import BaseModel

# Import shared helpers from dkn_rfm_agop
# (If running standalone, copy the helper functions here directly.)
from models.dkn_rfm_agop import (
    _fit_lssvm,
    _sample_anisotropic_rff,
    _apply_rff,
    _compute_agop_direct,
)


# ================================================================== #
#  Alignment objective (operates on fixed h_rff)                    #
# ================================================================== #

def _alignment_loss_linear(
    compressed: torch.Tensor,
    Y:          torch.Tensor,
) -> torch.Tensor:
    """
    Negative kernel-target alignment for a linearly compressed representation.

        A(W) = ‖compressed⊤ Y‖²_F / ‖compressed compressed⊤‖_F

    where compressed = h_rff_batch @ W⊤  (batch, r).

    h_rff is fixed; gradients flow only through W.

    This is mathematically equivalent to DKN-Align's alignment objective,
    but the kernel matrix is computed from a linear projection of fixed
    RFF features, not from RFF(W·h).  The gradient w.r.t. W is therefore
    purely linear (no trigonometric terms), making optimisation simpler.

    Parameters
    ----------
    compressed : (batch, r)  W·h_rff for the current minibatch
    Y          : (batch, P)  one-hot label matrix (float)
    """
    numerator   = (compressed.T @ Y).pow(2).sum()        # ‖comp⊤ Y‖²_F
    KK          = compressed @ compressed.T              # (batch, batch) kernel
    denominator = torch.norm(KK, p="fro") + 1e-8
    return -numerator / denominator


# ================================================================== #
#  Model                                                             #
# ================================================================== #

class DKN_RFM_Align(BaseModel):
    """
    Deep Kernel Network — Scalable RFM kernel + alignment-based compression.

    Parameters
    ----------
    n_layers   : number of (AGOP-phase → RFF → compress) blocks
    D          : RFF dimension per layer
    rank       : compression rank r; W ∈ R^(r × D)
    C          : LS-SVM regularisation for oracle SVMs and final layer
    agop_steps : AGOP iterations per layer to converge M
    lr         : Adam learning rate for W optimisation
    epochs     : gradient steps for W per layer
    batch_size : minibatch size for alignment estimation
    seed       : base RNG seed
    device     : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        n_layers:   int   = 2,
        D:          int   = 500,
        rank:       int   = 50,
        C:          float = 1.0,
        agop_steps: int   = 5,
        lr:         float = 1e-3,
        epochs:     int   = 300,
        batch_size: int   = 512,
        seed:       int   = 42,
        device:     str   = "cpu",
    ):
        self.n_layers   = n_layers
        self.D          = D
        self.rank       = rank
        self.C          = C
        self.agop_steps = agop_steps
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.seed       = seed
        self.device     = device

        # Filled during fit
        self.omegas_:    list[np.ndarray] = []
        self.biases_:    list[np.ndarray] = []
        self.Ws_:        list[np.ndarray] = []
        self.final_clf_: RidgeClassifier  = None

    # ---------------------------------------------------------------- #
    #  Step 1: AGOP phase (identical to DKN-RFM-AGOP)                 #
    # ---------------------------------------------------------------- #

    def _agop_phase(
        self,
        H:     np.ndarray,
        y:     np.ndarray,
        layer: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Iteratively refine M via anisotropic-RFF oracle SVMs.
        See DKN_RFM_AGOP._agop_phase for full documentation.

        Returns
        -------
        M     : (d_in, d_in)  converged AGOP matrix  (not stored — only ω,b needed)
        omega : (D, d_in)     anisotropic RFF frequencies from final step
        bias  : (D,)          RFF phases from final step
        """
        d_in = H.shape[1]
        M    = np.eye(d_in)

        omega = bias = None
        for t in range(self.agop_steps):
            omega, bias = _sample_anisotropic_rff(
                M, d_in, self.D,
                seed = self.seed + layer * 10_000 + t,
            )
            Phi    = _apply_rff(H, omega, bias)
            oracle = _fit_lssvm(Phi, y, self.C)
            M      = _compute_agop_direct(H, omega, bias, oracle.coef_)

        return M, omega, bias

    # ---------------------------------------------------------------- #
    #  Step 3: alignment-based compression                             #
    # ---------------------------------------------------------------- #

    def _align_compress(
        self,
        h_rff:  np.ndarray,
        y:      np.ndarray,
        layer:  int,
    ) -> np.ndarray:
        """
        Learn W ∈ R^(r × D) by maximising kernel-target alignment of
        W·h_rff with y, using Adam gradient descent.

        h_rff is FIXED (the converged anisotropic RFF features).
        Gradients flow through W only — no backprop through the RFF.

        Initialisation: W is set to the top-r PCA directions of h_rff,
        which provides a better starting point than random initialisation
        and typically halves the number of epochs needed.

        Parameters
        ----------
        h_rff : (n, D)  fixed RFF features from the AGOP-converged kernel
        y     : (n,)    class labels
        layer : layer index (for seeding)

        Returns
        -------
        W : (r, D)  learned compression matrix (numpy)
        """
        dev = torch.device(self.device)
        torch.manual_seed(self.seed + layer)

        n, D = h_rff.shape
        r    = min(self.rank, D)

        # Binarise labels for alignment
        lb   = LabelBinarizer()
        Y_np = lb.fit_transform(y).astype(np.float32)          # (n, P)
        Y_t  = torch.tensor(Y_np, device=dev)

        h_t  = torch.tensor(h_rff.astype(np.float32), device=dev)   # (n, D), no grad

        # Initialise W with top-r PCA of h_rff for a better starting point
        _, _, Vt = np.linalg.svd(h_rff - h_rff.mean(0), full_matrices=False)
        W_init   = torch.tensor(Vt[:r].astype(np.float32), device=dev)  # (r, D)
        W_param  = nn.Parameter(W_init.clone())

        opt       = torch.optim.Adam([W_param], lr=self.lr)
        rng_batch = torch.Generator()
        rng_batch.manual_seed(self.seed + layer)

        for epoch in range(self.epochs):
            idx  = torch.randperm(n, generator=rng_batch)[:self.batch_size]
            h_b  = h_t[idx]                                    # (batch, D), no grad
            Y_b  = Y_t[idx]                                    # (batch, P)

            opt.zero_grad()
            compressed = h_b @ W_param.T                       # (batch, r)
            loss       = _alignment_loss_linear(compressed, Y_b)
            loss.backward()
            opt.step()

        return W_param.detach().cpu().numpy()                  # (r, D)

    # ---------------------------------------------------------------- #
    #  Training                                                         #
    # ---------------------------------------------------------------- #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DKN_RFM_Align":
        """
        Greedy layer-wise training.

        For each layer k:
            1. AGOP phase → converged (ω_T, b_T)                [backprop-free]
            2. Apply converged anisotropic RFF: h_rff = RFF(H; ω_T, b_T)
            3. Alignment-based compression: W via Adam on fixed h_rff [within-layer]
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

            # Step 2 — apply converged anisotropic kernel (fixed for step 3)
            h_rff = _apply_rff(H, omega, bias)                  # (n, D)

            # Step 3 — alignment-based compression
            W = self._align_compress(h_rff, y, layer=k)         # (r, D)

            self.omegas_.append(omega)
            self.biases_.append(bias)
            self.Ws_.append(W)

            H = h_rff @ W.T                                     # (n, r)

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
            lr         = self.lr,
            epochs     = self.epochs,
            batch_size = self.batch_size,
        )
