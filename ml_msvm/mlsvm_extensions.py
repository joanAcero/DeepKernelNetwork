"""
mlsvm_extensions.py
===================
Research extensions to ML_MSVMClassifier — subclasses that add new
diversity and RFF-quality modes WITHOUT touching the original ml_msvm.py.

Classes
-------
DiverseMLMSVM      Adds diversity_mode={'c_spread','same_c','bootstrap','feature_subset'}
                   to study what actually makes m>1 beneficial.
QMC_MLMSVMClassifier  Adds rff_mode={'standard','orf','qmc'} to study whether
                   better random feature sampling closes the gap to the exact SVM.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

# ── auto-locate ml_msvm.py (same pattern as utils.import_ml_msvm) ───────────
def _import_base():
    """Return (ML_MSVMClassifier, module) using the same discovery logic as utils.py."""
    experiments_dir = Path(__file__).parent.resolve()
    for base in [experiments_dir.parent, experiments_dir.parent.parent, experiments_dir]:
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        for mod_path in ("mlsvm.ml_msvm", "ml_msvm.ml_msvm", "ml_msvm"):
            try:
                mod = importlib.import_module(mod_path)   # registers in sys.modules automatically
                cls = getattr(mod, "ML_MSVMClassifier", None)
                if cls is None or not (isinstance(cls, type) and hasattr(cls, "fit")):
                    continue
                return cls, mod
            except (ImportError, ModuleNotFoundError, AttributeError):
                pass
    # Filesystem fallback (same as utils.py)
    for f in sorted(experiments_dir.parent.rglob("ml_msvm.py")):
        spec = importlib.util.spec_from_file_location("ml_msvm_found", f)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules["ml_msvm_found"] = mod   # register BEFORE exec so @dataclass works
        spec.loader.exec_module(mod)
        cls = getattr(mod, "ML_MSVMClassifier", None)
        if cls:
            print(f"[mlsvm_extensions] Loaded ML_MSVMClassifier from {f}")
            return cls, mod
    raise ImportError("Cannot find ML_MSVMClassifier. Run: find ~ -name ml_msvm.py")

ML_MSVMClassifier, _base_mod = _import_base()
_Block                        = _base_mod._Block


from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class _InputSubspaceBlock:
    """Block where each SVM uses its own input-column subset and own Omega.
    sub_models: list of (cols, Omega, b, W_block) — one per SVM.
    The forward pass concatenates each SVM's projection featmap(X[:,cols]@Omega^T) @ W_block.
    """
    sub_models: list
    kernel: str
    arc_cosine_degree: int
    scaler: object = None
    cols_per_svm: int = 1
    W: object = None     # present so generic code that checks block.W is None still works


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DiverseMLMSVM
# ══════════════════════════════════════════════════════════════════════════════

class DiverseMLMSVM(ML_MSVMClassifier):
    """
    ML-MSVM with pluggable block-diversity strategies.

    diversity_mode : str
        'c_spread'        — logspace C values (original design, shown by exp6 to hurt).
        'same_c'          — all block SVMs at block_C, same training data. Rank-1 W for K=2.
        'bootstrap'       — each SVM trains on a random bootstrap_ratio subsample (replace=True).
        'feature_subset'  — each SVM uses floor(P/m) random features (C=block_C).
        'disjoint'        — TRUE BAGGING: training data split into m DISJOINT partitions of
                            n/m samples each; SVM j sees only partition j, with high block_C
                            so each overfits its own slice. Decorrelation by data partition.
        'disjoint_featsub'— disjoint data partition AND each SVM restricted to a random subset
                            of feature_frac*P features (default sqrt(P)). Double decorrelation
                            (data + features), random-subspace style (Ho 1998).
        'disjoint_featpart'— disjoint data partition AND P features split into m disjoint
                            blocks of P/m each (every feature used once, union = all P).
        'featpart_fulldata'— all data to every SVM; P features split into m disjoint P/m blocks.

        --- INPUT-SUBSPACE modes (the true Random-Forest analogue; Ho 1998) ---
        These subset the ORIGINAL INPUT COLUMNS of X (the d measured variables), not the
        P random features. Each SVM gets its OWN random projection Omega_j drawn in its
        input sub-space, so it is genuinely blind to the other input variables — forcing
        information-level decorrelation, exactly like RF trees on column subsets. The m
        per-SVM arc-cosine feature maps are concatenated as the block output X_next.
        'input_subspace_sqrt' — each SVM sees round(sqrt(d)) random input columns.
        'input_subspace_dm'   — input columns split into m disjoint blocks of d/m each
                                (requires m <= d; larger-m cells are skipped upstream).

    block_C : float          Regularisation for every block SVM (default 1.0; set 100 to overfit).
    bootstrap_ratio : float  Fraction of rows per bootstrap SVM.
    feature_frac : str|float 'sqrt' → round(sqrt(P)) RANDOM-FEATURE columns per SVM; or float in (0,1].
    input_subspace_full_data : bool  For input_subspace_* modes, whether every SVM sees all rows
                            (True, default) or a disjoint n/m data partition too (False).
    """

    def __init__(
        self,
        num_layers: int = 2,
        svms_per_block: int = 4,
        C_values: Optional[Sequence[float]] = None,
        rff_features: int = 1000,
        final_C: float = 1.0,
        kernel: str = "arc_cosine",
        arc_cosine_degree: int = 1,
        median_heuristic_subsample: Optional[int] = 1000,
        random_state: Optional[int] = None,
        normalize_inter_layer: bool = True,
        block_tol: float = 1e-2,
        block_max_iter: int = 1000,
        diversity_mode: str = "c_spread",
        bootstrap_ratio: float = 0.80,
        block_C: float = 1.0,
        feature_frac="sqrt",
        aggregate_W: bool = False,
        input_subspace_full_data: bool = True,
        input_subspace_full_P: bool = False,
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            svms_per_block=svms_per_block,
            C_values=C_values,
            rff_features=rff_features,
            final_C=final_C,
            kernel=kernel,
            arc_cosine_degree=arc_cosine_degree,
            median_heuristic_subsample=median_heuristic_subsample,
            random_state=random_state,
            normalize_inter_layer=normalize_inter_layer,
            block_tol=block_tol,
            block_max_iter=block_max_iter,
        )
        self.diversity_mode  = diversity_mode
        self.bootstrap_ratio = bootstrap_ratio
        self.block_C         = block_C
        self.feature_frac    = feature_frac
        self.aggregate_W     = aggregate_W
        self.input_subspace_full_data = input_subspace_full_data
        self.input_subspace_full_P = input_subspace_full_P
        # Diagnostics captured during fit (one entry per block)
        self.W_diagnostics_  = []

    # ── INPUT-SUBSPACE (Random-Forest analogue) ──────────────────────────────
    def _forward_pass(self, X):
        """Forward pass that understands both standard _Block and _InputSubspaceBlock."""
        X_curr = X
        for block in self.blocks_:
            if isinstance(block, _InputSubspaceBlock):
                proj_blocks = []
                for (cols, Omega, b, Wj) in block.sub_models:
                    Phi_j = self._feature_map(X_curr[:, cols], Omega, b,
                                              kernel=block.kernel,
                                              arc_cosine_degree=block.arc_cosine_degree)
                    proj_blocks.append(Phi_j @ Wj)
                X_curr = np.hstack(proj_blocks)
                if block.scaler is not None:
                    X_curr = block.scaler.transform(X_curr)
            else:
                Phi = self._feature_map(X_curr, block.Omega, block.b,
                                        kernel=block.kernel,
                                        arc_cosine_degree=block.arc_cosine_degree)
                if block.W is None:
                    X_curr = Phi
                else:
                    X_curr = Phi @ block.W
                    if block.scaler is not None:
                        X_curr = block.scaler.transform(X_curr)
        return X_curr

    def _is_input_subspace(self):
        return self.diversity_mode in ("input_subspace_sqrt", "input_subspace_dm")

    def _input_col_sets(self, d, m, rng):
        """Return m index arrays of INPUT columns, one per SVM."""
        if self.diversity_mode == "input_subspace_sqrt":
            k = max(1, int(round(np.sqrt(d))))
            return [rng.choice(d, k, replace=False) for _ in range(m)]
        else:  # input_subspace_dm — disjoint partition of the d input columns
            return [a for a in np.array_split(rng.permutation(d), m) if len(a) > 0]

    def _fit_block(self, X, y_enc, C_list, rng):
        """Override only for input-subspace modes; otherwise defer to the base class
        (which routes through our _train_svm_block for the Phi-space modes).

        Input-subspace block: each SVM j gets its OWN input columns S_j and its OWN
        arc-cosine projection Omega_j ~ N(0, I) in |S_j| dimensions (Cho & Saul on the
        sub-space). SVM j trains on Phi_j = featmap(X[:, S_j] @ Omega_j^T). The block
        output is the concatenation of the m per-SVM decision projections, so the next
        layer receives m genuinely different (information-decorrelated) views.
        """
        if not self._is_input_subspace():
            return super()._fit_block(X, y_enc, C_list, rng)

        n, d = X.shape
        m    = len(C_list)
        Cval = C_list[0]
        K    = len(np.unique(y_enc))
        cols_per_svm = 1 if K == 2 else K
        P_per = self.rff_features if self.input_subspace_full_P else max(1, self.rff_features // m)

        col_sets = self._input_col_sets(d, m, rng)
        # Data partition (full data unless explicitly disjoint)
        if self.input_subspace_full_data:
            data_sets = [np.arange(n)] * len(col_sets)
        else:
            data_sets = np.array_split(rng.permutation(n), len(col_sets))

        sub_models = []          # (cols, Omega, b, W_block, scaler) per SVM
        proj_blocks = []         # each SVM's contribution to X_next
        weight_cols_for_diag = []
        for j, cols in enumerate(col_sets):
            idx = data_sets[j]
            dj  = len(cols)
            gamma = self._median_heuristic(X[np.ix_(idx, cols)], rng) if self.kernel == "rbf" else 1.0
            Omega, b = self._sample_frequencies_sub(dj, P_per, gamma, rng)
            Phi_j = self._feature_map(X[np.ix_(idx, cols)], Omega, b)   # (len(idx), P_per)
            if len(idx) < 2 or len(np.unique(y_enc[idx])) < 2:
                Wj = np.zeros((P_per, cols_per_svm))
            else:
                svm = self._make_svm(Cval, rng, (len(idx), P_per), "block").fit(Phi_j, y_enc[idx])
                coef = np.atleast_2d(svm.coef_).T            # (P_per, n_cls)
                Wj = np.zeros((P_per, cols_per_svm))
                Wj[:, :coef.shape[1]] = coef[:, :cols_per_svm]
            # full-data projection so all rows get a representation for the next layer
            Phi_full = self._feature_map(X[:, cols], Omega, b)          # (n, P_per)
            proj = Phi_full @ Wj                                        # (n, cols_per_svm)
            proj_blocks.append(proj)
            for c in range(Wj.shape[1]):
                weight_cols_for_diag.append(Wj[:, c])
            sub_models.append((np.asarray(cols), Omega, b, Wj))

        X_next = np.hstack(proj_blocks)                                 # (n, m*cols_per_svm)
        # Diagnostics: treat each SVM's weight (in its own P_per space, zero-padded) —
        # we report decorrelation of the per-SVM decision projections instead, which is
        # the meaningful quantity when feature spaces differ.
        self._record_proj_diag(X_next, m, cols_per_svm)

        scaler = None
        if self.kernel == "arc_cosine" and self.normalize_inter_layer:
            scaler = StandardScaler()
            X_next = scaler.fit_transform(X_next)

        block = _InputSubspaceBlock(sub_models=sub_models, kernel=self.kernel,
                                    arc_cosine_degree=self.arc_cosine_degree,
                                    scaler=scaler, cols_per_svm=cols_per_svm)
        return block, X_next

    def _sample_frequencies_sub(self, d, P, gamma, rng):
        """Like base _sample_frequencies but with an explicit P (per-SVM budget)."""
        if self.kernel == "rbf":
            Omega = rng.standard_normal((P, d)) * np.sqrt(2 * gamma)
            b = rng.uniform(0, 2 * np.pi, P)
        else:
            Omega = rng.standard_normal((P, d))
            b = np.zeros(P)
        return Omega, b

    def _record_proj_diag(self, X_next, m, cols_per_svm):
        """Decorrelation of the per-SVM decision projections (the columns of X_next).
        For input-subspace mode the SVMs live in different feature spaces, so we measure
        how correlated their OUTPUTS (votes on the data) are — the operational notion of
        'different opinions'. Same-class columns compared across SVMs, averaged over classes.
        """
        total = X_next.shape[1]
        cps = cols_per_svm if cols_per_svm > 0 else 1
        m_actual = total // cps if (cps and total % cps == 0 and total >= cps) else total
        cps = cps if m_actual > 1 else 1
        if m_actual > 1:
            per_class = []
            for c in range(cps):
                V = X_next[:, c::cps].T            # (m_actual, n): each SVM's vote vector
                nrm = np.linalg.norm(V, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
                U = V / nrm
                G = U @ U.T
                k = G.shape[0]
                per_class.append((G.sum() - np.trace(G)) / (k * (k - 1)))
            mean_cos = float(np.mean(per_class))
        else:
            mean_cos = 1.0
        sr = float(np.sum(X_next ** 2) / (np.linalg.norm(X_next, 2) ** 2 + 1e-12))
        self.W_diagnostics_.append(dict(
            n_cols=int(total), m=int(m), hard_rank=int(np.linalg.matrix_rank(X_next)),
            stable_rank=round(sr, 3), mean_cos_sim=round(mean_cos, 4)))

    # ── resolve C values depending on mode ───────────────────────────────────
    def _resolve_C_values(self):
        if self.diversity_mode == "c_spread":
            if self.C_values is not None:
                return list(self.C_values)
            if self.svms_per_block == 1:
                return [1.0]
            return list(np.logspace(-2, 2, num=self.svms_per_block))
        else:
            # Every non-c_spread mode uses a single uniform block_C value
            return [self.block_C] * self.svms_per_block

    # ── number of features each SVM sees in feature-subset modes ──────────────
    def _n_feats(self, P):
        if self.feature_frac == "sqrt":
            return max(1, int(round(np.sqrt(P))))
        return max(1, int(round(float(self.feature_frac) * P)))

    # ── diversity-aware block training ───────────────────────────────────────
    def _train_svm_block(self, Phi, y_enc, C_list, rng):
        """Override: dispatch to the chosen diversity strategy."""
        n, P = Phi.shape
        m    = len(C_list)
        Cval = C_list[0]

        if self.diversity_mode == "feature_subset":
            W = self._train_feature_subset(Phi, y_enc, m, P, Cval, rng)
        elif self.diversity_mode == "bootstrap":
            W = self._train_bootstrap(Phi, y_enc, m, P, n, Cval, rng)
        elif self.diversity_mode == "disjoint":
            W = self._train_disjoint(Phi, y_enc, m, P, n, Cval, rng, featsub=False)
        elif self.diversity_mode == "disjoint_featsub":
            W = self._train_disjoint(Phi, y_enc, m, P, n, Cval, rng, featsub=True)
        elif self.diversity_mode == "disjoint_featpart":
            # Disjoint DATA partitions AND disjoint FEATURE partitions (P/m each),
            # so every random feature is used by exactly one SVM and the union
            # covers all P features (no information discarded, unlike sqrt-subset).
            W = self._train_disjoint(Phi, y_enc, m, P, n, Cval, rng,
                                     featsub=True, feat_partition=True)
        elif self.diversity_mode == "featpart_fulldata":
            # FEATURE partitions only (P/m disjoint feature blocks); every SVM sees
            # ALL the data. Isolates the width-from-feature-decorrelation effect from
            # the data-starvation effect of disjoint data partitioning.
            W = self._train_disjoint(Phi, y_enc, m, P, n, Cval, rng,
                                     featsub=True, feat_partition=True, full_data=True)
        else:
            W = super()._train_svm_block(Phi, y_enc, C_list, rng)

        # cols_per_svm: weight columns each SVM contributes (1 binary, K multiclass).
        K = len(np.unique(y_enc))
        cols_per_svm = 1 if K == 2 else K

        # Record vote-decorrelation diagnostics on the FULL (pre-collapse) W, so the
        # cosine similarity reflects how the m SVMs actually disagree per class.
        self._record_W_diag(W, m, cols_per_svm=cols_per_svm)

        # BAGGING vs STACKING: if aggregate_W, collapse the m per-SVM votes into a
        # single per-class averaged weight vector (true bagging = average the
        # ensemble). Otherwise keep the full W (stacking = next layer learns the
        # combination). We average ACROSS the m SVMs, never across classes, and it
        # is a no-op when there is only one SVM.
        if self.aggregate_W:
            total = W.shape[1]
            if total > cols_per_svm and total % cols_per_svm == 0:
                m_actual = total // cols_per_svm
                W3 = W.reshape(W.shape[0], m_actual, cols_per_svm)
                W = W3.mean(axis=1)             # (P, cols_per_svm): per-class mean vote
            # else: single SVM (m=1) or non-divisible → leave W unchanged (no-op)

        return W

    # ── diagnostics: how different are the SVM "votes"? ───────────────────────
    def _record_W_diag(self, W, m, cols_per_svm=1):
        """W is (P, n_cols). Each of the m SVMs contributes `cols_per_svm` columns
        (1 if binary, K if OvR multiclass). Diagnose how distinct the per-SVM votes are.

        Multiclass-correct cosine: compare SVM_i's class-k vote against SVM_j's class-k
        vote for the SAME class k, then average over classes. Pooling all m*K columns
        would mix different classes (meant to differ) and deflate the similarity.
        """
        hard_rank = int(np.linalg.matrix_rank(W, tol=1e-6 * (np.linalg.norm(W, 2) or 1.0)))
        fro2  = float(np.sum(W ** 2))
        spec2 = float(np.linalg.norm(W, 2) ** 2)
        stable_rank = fro2 / spec2 if spec2 > 0 else 0.0

        total = W.shape[1]
        cps = cols_per_svm if cols_per_svm and cols_per_svm > 0 else 1
        if total % cps == 0 and total >= cps:
            m_actual = total // cps
        else:
            m_actual, cps = total, 1

        if m_actual > 1:
            per_class = []
            for c in range(cps):
                cols = W[:, c::cps]              # (P, m_actual): class c across SVMs
                V = cols.T
                nrm = np.linalg.norm(V, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
                U = V / nrm
                G = U @ U.T
                k = G.shape[0]
                per_class.append((G.sum() - np.trace(G)) / (k * (k - 1)))
            mean_cos = float(np.mean(per_class))
        else:
            mean_cos = 1.0                       # single SVM → trivially self-identical

        self.W_diagnostics_.append(dict(
            n_cols=int(W.shape[1]), m=int(m),
            hard_rank=hard_rank, stable_rank=round(stable_rank, 3),
            mean_cos_sim=round(mean_cos, 4),
        ))

    def _train_bootstrap(self, Phi, y_enc, m, P, n, Cval, rng):
        weight_cols = []
        k = max(2, int(self.bootstrap_ratio * n))
        for _ in range(m):
            idx = rng.choice(n, k, replace=True)
            svm = self._make_svm(Cval, rng, (k, P), "block").fit(Phi[idx], y_enc[idx])
            for row in np.atleast_2d(svm.coef_):
                weight_cols.append(row)
        return np.column_stack(weight_cols)

    def _train_feature_subset(self, Phi, y_enc, m, P, Cval, rng):
        k = max(1, P // m)
        weight_cols = []
        for _ in range(m):
            feat_idx = rng.choice(P, k, replace=False)
            svm = self._make_svm(Cval, rng, (Phi.shape[0], k), "block").fit(Phi[:, feat_idx], y_enc)
            for row in np.atleast_2d(svm.coef_):
                full = np.zeros(P); full[feat_idx] = row
                weight_cols.append(full)
        return np.column_stack(weight_cols)

    def _train_disjoint(self, Phi, y_enc, m, P, n, Cval, rng, featsub,
                        feat_partition=False, full_data=False):
        """Disjoint-partition bagging trainer.

        Data axis:
          full_data=False : split n rows into m disjoint partitions (~n/m each);
                            SVM j trains only on partition j. (true data bagging)
          full_data=True  : every SVM sees ALL n rows. (isolates the feature effect)

        Feature axis:
          featsub=False                  : each SVM uses all P features.
          featsub=True, feat_partition=F : each SVM uses sqrt(P) RANDOM features
                                           (RF-style sampling; features may overlap/be unused).
          featsub=True, feat_partition=T : the P features are split into m DISJOINT
                                           blocks of ~P/m each; SVM j uses block j.
                                           Every feature used exactly once; union = all P.
        """
        K            = len(np.unique(y_enc))
        cols_per_svm = 1 if K == 2 else K

        # Data partitions
        if full_data:
            data_parts = [np.arange(n)] * m
        else:
            data_parts = np.array_split(rng.permutation(n), m)

        # Feature partitions / subsets
        if featsub and feat_partition:
            feat_parts = np.array_split(rng.permutation(P), m)   # disjoint, cover all P
        elif featsub:
            k_feat = self._n_feats(P)
            feat_parts = [rng.choice(P, k_feat, replace=False) for _ in range(m)]
        else:
            feat_parts = [np.arange(P)] * m

        weight_cols = []
        for j in range(m):
            idx, feat_idx = data_parts[j], feat_parts[j]
            if len(idx) < 2 or len(np.unique(y_enc[idx])) < 2 or len(feat_idx) < 1:
                for _ in range(cols_per_svm):
                    weight_cols.append(np.zeros(P))
                continue
            if len(feat_idx) == P:
                svm = self._make_svm(Cval, rng, (len(idx), P), "block").fit(
                    Phi[idx], y_enc[idx])
                coef = np.atleast_2d(svm.coef_)
                for r in range(cols_per_svm):
                    weight_cols.append(coef[r] if r < coef.shape[0] else np.zeros(P))
            else:
                svm = self._make_svm(Cval, rng, (len(idx), len(feat_idx)), "block").fit(
                    Phi[np.ix_(idx, feat_idx)], y_enc[idx])
                coef = np.atleast_2d(svm.coef_)
                for r in range(cols_per_svm):
                    full = np.zeros(P)
                    if r < coef.shape[0]:
                        full[feat_idx] = coef[r]
                    weight_cols.append(full)
        return np.column_stack(weight_cols)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  QMC_MLMSVMClassifier
# ══════════════════════════════════════════════════════════════════════════════

class QMC_MLMSVMClassifier(ML_MSVMClassifier):
    """
    ML-MSVM with pluggable random-feature sampling quality.

    rff_mode : str
        'standard'  — i.i.d. Gaussian rows of Ω (current baseline).
        'orf'       — Orthogonal Random Features (Yu et al., 2016).
                      Rows of Ω are structured-orthogonal; same marginal distribution
                      as Gaussian but lower inter-feature correlation → lower
                      approximation variance.
        'qmc'       — Quasi-Monte Carlo features via Sobol low-discrepancy sequence
                      (scipy.stats.qmc). Better spectral coverage per P features.
                      Falls back to 'orf' if scipy.qmc is unavailable.

    Both ORF and QMC are drop-in replacements: they generate the same Ω shape and
    the rest of the architecture is unchanged.
    """

    def __init__(
        self,
        num_layers: int = 1,
        svms_per_block: int = 1,
        C_values: Optional[Sequence[float]] = None,
        rff_features: int = 1000,
        final_C: float = 1.0,
        kernel: str = "rbf",
        arc_cosine_degree: int = 1,
        median_heuristic_subsample: Optional[int] = 1000,
        random_state: Optional[int] = None,
        normalize_inter_layer: bool = True,
        block_tol: float = 1e-2,
        block_max_iter: int = 1000,
        rff_mode: str = "standard",
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            svms_per_block=svms_per_block,
            C_values=C_values,
            rff_features=rff_features,
            final_C=final_C,
            kernel=kernel,
            arc_cosine_degree=arc_cosine_degree,
            median_heuristic_subsample=median_heuristic_subsample,
            random_state=random_state,
            normalize_inter_layer=normalize_inter_layer,
            block_tol=block_tol,
            block_max_iter=block_max_iter,
        )
        self.rff_mode = rff_mode

    def _sample_frequencies(self, d, gamma, rng):
        """Dispatch to the chosen sampling strategy."""
        P    = self.rff_features
        mode = self.rff_mode
        if mode == "orf":
            return self._sample_orf(d, gamma, P, rng)
        elif mode == "qmc":
            try:
                return self._sample_qmc(d, gamma, P, rng)
            except (ImportError, Exception):
                return self._sample_orf(d, gamma, P, rng)   # graceful fallback
        else:
            return super()._sample_frequencies(d, gamma, rng)

    # ── ORF ─────────────────────────────────────────────────────────────────
    def _sample_orf(self, d, gamma, P, rng):
        """
        Orthogonal Random Features (Yu et al. 2016).
        Rows of Ω are built from orthogonal d×d blocks, each scaled by
        chi(d) norms to preserve the N(0,I) marginal distribution.
        """
        n_blocks = int(np.ceil(P / d))
        rows = []
        for _ in range(n_blocks):
            G = rng.standard_normal((d, d))
            Q, _ = np.linalg.qr(G)                         # orthonormal rows
            # Chi(d) scaling: norms of independent d-dim Gaussian vectors
            S = np.sqrt(np.sum(rng.standard_normal((d, d)) ** 2, axis=1))
            rows.append(Q * S[:, np.newaxis])
        Omega = np.vstack(rows)[:P]                        # (P, d)
        if self.kernel == "rbf":
            Omega = Omega * np.sqrt(2 * gamma)
            b = rng.uniform(0, 2 * np.pi, P)
        else:
            b = np.zeros(P)
        return Omega, b

    # ── QMC (Sobol) ─────────────────────────────────────────────────────────
    def _sample_qmc(self, d, gamma, P, rng):
        """
        Quasi-Monte Carlo features via Sobol low-discrepancy sequences.
        Applies the inverse-normal CDF to convert uniform [0,1]^d Sobol
        points into approximately N(0,I) samples.
        """
        from scipy.stats import qmc, norm as sp_norm
        seed = int(rng.integers(0, 2**31))
        # Sobol requires power-of-2 sample count; round up
        P2 = int(2 ** np.ceil(np.log2(P)))
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        sampler.fast_forward(1)                            # skip all-zeros point
        u = sampler.random(P2)[:P]                        # (P, d) in [0,1]
        # Clip to avoid ±inf at the tails
        u = np.clip(u, 1e-6, 1 - 1e-6)
        Omega = sp_norm.ppf(u)                            # (P, d) ~ N(0,I)
        if self.kernel == "rbf":
            Omega = Omega * np.sqrt(2 * gamma)
            b = rng.uniform(0, 2 * np.pi, P)
        else:
            b = np.zeros(P)
        return Omega, b
