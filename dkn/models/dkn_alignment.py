"""
DKN-Alignment — Deep Kernel Network trained by maximising kernel-target
alignment at each layer.

Architecture (inference path, identical to DKN-AGOP):
    x  →  [W¹ → RFF¹]  →  [W² → RFF²]  →  ...  →  [Wᴸ → RFFᴸ]  →  SVM_final  →  ŷ

Training procedure (layer by layer):
    For each layer k:
        Optimise W^(k) by gradient descent on the negative kernel-target
        alignment objective:

            A(W) = ||Φᵀ Y||²_F / ||ΦΦᵀ||_F

        where Φ = RFF(W·H) and Y is the one-hot label matrix.

        Cortes et al. (2012) prove that maximising A minimises an upper
        bound on the SVM generalisation error — giving a direct theoretical
        grounding for this objective without solving an inner QP.

        W^(k) is learned via Adam; RFF frequencies are frozen.
        After training, W^(k) is frozen and H is updated.
    Final layer: LS-SVM on last H — kept for inference.

Changelog vs. original:
    FIX — sigest_gamma replaces the 1/(d·Var(X)) heuristic in _RFFBlock.
        Gamma is now estimated from the empirical pairwise distance
        distribution of the block input W·H, providing a principled,
        data-adaptive bandwidth consistent with kernlab and the other
        DKN models.  Note that in DKN-Alignment the RFF frequencies are
        frozen after block construction and W is learned by Adam, so the
        gamma is set once per layer (unlike DKN-AGOP where it is re-set
        per AGOP step).

Requires: torch (pip install torch)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelBinarizer
from models.base import BaseModel
from KernelApproximations.RFF_Gaussian import RFF_Gaussian as NumpyRFF
from KernelApproximations.sigest import sigest_gamma


# ------------------------------------------------------------------ #
#  PyTorch block (W is a learnable parameter)                         #
# ------------------------------------------------------------------ #

class _RFFBlock(nn.Module):
    """
    Single learnable block:  Phi = sqrt(2/D) · cos(H·Wᵀ·omegaᵀ + bias)

    W is an nn.Parameter.
    omega and bias are initialised using the same RFF convention:
        omega ~ N(0, 2*gamma*I),  bias ~ U[0, 2*pi]

    Gamma is estimated by sigest_gamma on the initial block input H,
    replacing the previous 1/(d·Var(X)) heuristic.
    """

    def __init__(
        self,
        d_in: int,
        d_k: int,
        D: int,
        H_numpy: np.ndarray,
        seed: int,
        sigest_frac: float = 1.0,
    ):
        super().__init__()

        # FIX: use sigest to estimate gamma from the actual input distribution.
        # scaled=False: H_numpy is already a projection of a standardised input.
        gamma = sigest_gamma(
            H_numpy,
            frac=sigest_frac,
            scaled=False,
            quantile=0.5,
            seed=seed,
        )

        rng = torch.Generator()
        rng.manual_seed(seed)

        # Learnable W, identity-like initialisation
        W_init = torch.zeros(d_k, d_in)
        r = min(d_k, d_in)
        W_init[:r, :r] = torch.eye(r)
        self.W = nn.Parameter(W_init)

        # Frozen RFF: omega ~ N(0, 2*gamma*I)
        std   = (2.0 * gamma) ** 0.5
        omega = torch.normal(0.0, std, (D, d_k), generator=rng)
        bias  = torch.rand(D, generator=rng) * 2 * torch.pi
        self.register_buffer("omega", omega)
        self.register_buffer("bias",  bias)
        self.D = D

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """H: (n, d_in)  →  Phi: (n, D)"""
        Z = H @ self.W.T                        # (n, d_k)
        Z = Z @ self.omega.T + self.bias        # (n, D)
        return (2.0 / self.D) ** 0.5 * torch.cos(Z)


# ------------------------------------------------------------------ #
#  Alignment objective                                                #
# ------------------------------------------------------------------ #

def _alignment_loss(Phi: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Negative kernel-target alignment:

        A(Phi) = ||Phi.T @ Y||²_F / ||Phi @ Phi.T||_F

    We minimise -A (i.e. maximise A).

    Parameters
    ----------
    Phi : (n, D) — RFF features
    Y   : (n, P) — one-hot label matrix (float)
    """
    numerator   = (Phi.T @ Y).pow(2).sum()           # ||Phi.T Y||²_F
    KK          = Phi @ Phi.T                         # (n, n) kernel matrix
    denominator = torch.norm(KK, p="fro") + 1e-8
    return -numerator / denominator


# ------------------------------------------------------------------ #
#  LS-SVM (NumPy, for the final inference layer)                      #
# ------------------------------------------------------------------ #

def _fit_lssvm(Phi: np.ndarray, y: np.ndarray, C: float) -> RidgeClassifier:
    clf = RidgeClassifier(alpha=1.0 / (2.0 * C), fit_intercept=True)
    clf.fit(Phi, y)
    return clf


# ------------------------------------------------------------------ #
#  DKN-Alignment                                                      #
# ------------------------------------------------------------------ #

class DKN_Alignment(BaseModel):
    """
    Deep Kernel Network — kernel-target alignment training regime.

    Each layer's weight matrix W^(k) is learned by gradient descent
    on the negative kernel-target alignment of the RFF features it
    produces.  The RFF bandwidth is set by sigest_gamma on the block
    input at construction time.

    Parameters
    ----------
    n_layers    : number of W→RFF blocks
    d_k         : hidden dimension after each W transform
    D           : RFF dimension per block
    C           : LS-SVM regularisation for the final layer
    lr          : Adam learning rate for W optimisation
    epochs      : gradient descent steps per layer
    batch_size  : minibatch size for alignment estimation
    sigest_frac : fraction of n to sample for sigest bandwidth estimation
    seed        : RNG seed (used for RFF, torch, and sigest)
    device      : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        n_layers: int = 2,
        d_k: int = 200,
        D: int = 500,
        C: float = 1.0,
        lr: float = 1e-3,
        epochs: int = 300,
        batch_size: int = 512,
        sigest_frac: float = 1.0,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.n_layers     = n_layers
        self.d_k          = d_k
        self.D            = D
        self.C            = C
        self.lr           = lr
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.sigest_frac  = sigest_frac
        self.seed         = seed
        self.device       = device

        # Set during fit
        self.blocks_:    list[_RFFBlock]  = []
        self.final_clf_: RidgeClassifier  = None

    # ---------------------------------------------------------------- #
    #  Training                                                         #
    # ---------------------------------------------------------------- #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DKN_Alignment":
        """
        Greedy alignment training.

        For each layer k:
            1. Build _RFFBlock (W is nn.Parameter, omega/bias frozen).
               Gamma is estimated by sigest on the current H (numpy).
            2. Optimise W by minimising _alignment_loss via Adam.
            3. Freeze W; propagate H = Phi(W·H) forward.
        Then fit final LS-SVM on the last H.
        """
        torch.manual_seed(self.seed)
        dev = torch.device(self.device)

        lb   = LabelBinarizer()
        Y_np = lb.fit_transform(y).astype(np.float32)  # (n, P)
        Y    = torch.tensor(Y_np, device=dev)

        H = torch.tensor(X.astype(np.float32), device=dev)
        self.blocks_ = []

        for k in range(self.n_layers):
            d_in    = H.shape[1]
            H_numpy = H.cpu().numpy()   # needed for sigest (numpy-based)

            block = _RFFBlock(
                d_in        = d_in,
                d_k         = self.d_k,
                D           = self.D,
                H_numpy     = H_numpy,
                seed        = self.seed + k,
                sigest_frac = self.sigest_frac,
            ).to(dev)

            opt = torch.optim.Adam(block.parameters(), lr=self.lr)

            block.train()
            n         = H.shape[0]
            rng_batch = torch.Generator()
            rng_batch.manual_seed(self.seed + k)

            for epoch in range(self.epochs):
                idx  = torch.randperm(n, generator=rng_batch)[:self.batch_size]
                H_b  = H[idx]
                Y_b  = Y[idx]

                opt.zero_grad()
                Phi  = block(H_b)
                loss = _alignment_loss(Phi, Y_b)
                loss.backward()
                opt.step()

                # Progress printing removed — output is controlled
                # at the benchmark level via evaluate.py's ETA tracker.

            # Freeze and propagate
            block.eval()
            with torch.no_grad():
                H = block(H)

            self.blocks_.append(block)

        # Final LS-SVM — kept for inference
        H_np = H.detach().cpu().numpy().astype(np.float64)
        self.final_clf_ = _fit_lssvm(H_np, y, self.C)
        return self

    # ---------------------------------------------------------------- #
    #  Inference                                                        #
    # ---------------------------------------------------------------- #

    def _forward(self, X: np.ndarray) -> np.ndarray:
        dev = torch.device(self.device)
        H   = torch.tensor(X.astype(np.float32), device=dev)

        for block in self.blocks_:
            block.eval()
            with torch.no_grad():
                H = block(H)

        return H.cpu().numpy().astype(np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        H = self._forward(X)
        return self.final_clf_.predict(H)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        H = self._forward(X)
        return self.final_clf_.decision_function(H)

    def get_params(self) -> dict:
        return dict(
            n_layers=self.n_layers,
            d_k=self.d_k,
            D=self.D,
            C=self.C,
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            sigest_frac=self.sigest_frac,
        )
