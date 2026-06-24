"""
mlsvm_interlayer.py — ablation subclass that WIDENS the inter-layer representation.

Diagnosis (from Experiment 3): each block outputs X_next = Phi @ W of width
m * cols_per_svm. For BINARY problems cols_per_svm = 1, so the next layer receives
only m near-collinear SVM decision scores ~ one effective dimension, leaving depth
nothing to recompose. This is why depth helped multiclass (CoverType, width m*K) but
not the binary physics datasets (HIGGS, SUSY, MiniBooNE).

This subclass adds two structural fixes, selectable via `interlayer_mode`, applied
identically in fit and inference so the two stay consistent:

  'baseline'   : X_next as-is (reproduces DiverseMLMSVM; control).
  'skip'       : X_next <- concat(X_next, X_curr)   — residual/skip connection: carry
                 the previous representation forward so a block ADDS to rather than
                 REPLACES the signal (the reason depth works in ResNets).
  'phi_concat' : X_next <- concat(X_next, R(Phi))   — inject a compressed view of the
                 block's own feature map Phi via a fixed random projection P->q, so the
                 next layer sees raw feature structure alongside the decision scores.
  'both'       : concat(X_next, X_curr, R(Phi)).

Only the inter-layer signal changes; feeding strategy, kernel, solver, C's are untouched,
so any change in the depth gain is attributable to the inter-layer width alone.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
from sklearn.preprocessing import StandardScaler

from mlsvm_extensions import DiverseMLMSVM, _InputSubspaceBlock


@dataclass
class _InterLayerExtra:
    """Per-block state needed to reproduce the inter-layer augmentation at inference."""
    mode: str
    R: Optional[np.ndarray]          # fixed P->q random projection for phi_concat (or None)
    scaler2: Optional[StandardScaler] # scales the AUGMENTED X_next (arc-cosine only)


class InterLayerMLMSVM(DiverseMLMSVM):
    def __init__(self, *args, interlayer_mode: str = "baseline",
                 phi_proj_dim: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self.interlayer_mode = interlayer_mode
        self.phi_proj_dim = phi_proj_dim
        self._extras = []   # one _InterLayerExtra per block, in fit order

    # ---- helpers -------------------------------------------------------------
    def _augment(self, X_next, X_curr, Phi, rng, fit, extra=None):
        """Apply the inter-layer augmentation. On fit, build & return the extra state;
        on inference, consume the stored extra."""
        mode = self.interlayer_mode
        parts = [X_next]
        R = (extra.R if extra is not None else None)
        if mode in ("phi_concat", "both"):
            if fit:
                q = min(self.phi_proj_dim, Phi.shape[1])
                R = rng.standard_normal((Phi.shape[1], q)) / np.sqrt(q)
            parts_phi = Phi @ R
            parts.append(parts_phi)
        if mode in ("skip", "both"):
            parts.append(X_curr)
        aug = np.hstack(parts) if len(parts) > 1 else X_next
        # standardise the augmented representation (arc-cosine), as the base does for X_next
        scaler2 = (extra.scaler2 if extra is not None else None)
        if self.kernel == "arc_cosine" and self.normalize_inter_layer:
            if fit:
                scaler2 = StandardScaler(); aug = scaler2.fit_transform(aug)
            elif scaler2 is not None:
                aug = scaler2.transform(aug)
        if fit:
            return aug, _InterLayerExtra(mode=mode, R=R, scaler2=scaler2)
        return aug

    # ---- fit: wrap the parent block, then augment ----------------------------
    def _fit_block(self, X, y_enc, C_list, rng):
        if self.interlayer_mode == "baseline":
            block, X_next = super()._fit_block(X, y_enc, C_list, rng)
            self._extras.append(None)
            return block, X_next
        X_curr = X
        block, X_next = super()._fit_block(X, y_enc, C_list, rng)
        # reconstruct THIS block's Phi for phi_concat (same Omega/b the block stored)
        Phi = self._block_phi(block, X_curr)
        aug, extra = self._augment(X_next, X_curr, Phi, rng, fit=True)
        self._extras.append(extra)
        return block, aug

    def _block_phi(self, block, X_curr):
        """Recompute Phi for a block given its input (handles both block types)."""
        if isinstance(block, _InputSubspaceBlock):
            cols0, Omega0, b0, _ = block.sub_models[0]
            # use the concatenation of each sub-SVM's featmap as 'Phi' surrogate
            mats = [self._feature_map(X_curr[:, c], Om, bb,
                        kernel=block.kernel, arc_cosine_degree=block.arc_cosine_degree)
                    for (c, Om, bb, _) in block.sub_models]
            return np.hstack(mats)
        return self._feature_map(X_curr, block.Omega, block.b,
                                 kernel=block.kernel, arc_cosine_degree=block.arc_cosine_degree)

    # ---- inference: mirror fit exactly ---------------------------------------
    def _forward_pass(self, X):
        if self.interlayer_mode == "baseline":
            return super()._forward_pass(X)
        X_curr = X
        for block, extra in zip(self.blocks_, self._extras):
            # base per-block transform (without the base scaler double-applying:
            # we recompute the raw projection then apply the base scaler ourselves)
            if isinstance(block, _InputSubspaceBlock):
                proj = []
                for (cols, Omega, b, Wj) in block.sub_models:
                    Phi_j = self._feature_map(X_curr[:, cols], Omega, b,
                                kernel=block.kernel, arc_cosine_degree=block.arc_cosine_degree)
                    proj.append(Phi_j @ Wj)
                X_next = np.hstack(proj)
            else:
                Phi_b = self._feature_map(X_curr, block.Omega, block.b,
                            kernel=block.kernel, arc_cosine_degree=block.arc_cosine_degree)
                X_next = Phi_b if block.W is None else Phi_b @ block.W
            if block.scaler is not None:
                X_next = block.scaler.transform(X_next)
            if extra is None:
                X_curr = X_next; continue
            Phi = self._block_phi(block, X_curr)
            X_curr = self._augment(X_next, X_curr, Phi, rng=None, fit=False, extra=extra)
        return X_curr
