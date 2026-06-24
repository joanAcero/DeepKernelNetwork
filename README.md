# Deep Kernel Networks: Neural Architectures with SVM Foundations

Backpropagation-free deep kernel architectures built from stacked random-feature
maps and parallel linear SVMs. This repository accompanies the master's thesis
(TFM) of Joan Acero Pousa, supervised by Lluís Belanche (FIB, UPC Barcelona),
and extends prior work on Multilayer SVMs (Acero & Belanche, ESANN 2025).

---

## Core idea

The central architecture is **ML-MSVM** (Multilayer Multi-SVM with weight
projections), in its definitive arc-cosine variant referred to in the thesis
notes as **RDAS** (Random-feature Deep Arc-cosine SVM, Proposal 2b).

Each block performs two transformations:

1. **Nonlinear** — a random-feature map `X → Φ` (`rbf` cosine features or, in the
   definitive variant, `arc_cosine` ReLU-style features `√(2/P)·max(0, XΩᵀ)`).
2. **Linear** — `m` parallel linear SVMs whose weight columns form
   `W ∈ ℝ^{P×m}`, projecting `Φ → ΦW = X_next`, the input to the next block.

Stacking `L` such blocks and finishing with a linear SVM head yields a deep,
fully convex, backpropagation-free classifier. Passing the **weight matrix** `W`
forward (rather than the scalar SVM output) is what resolves the representation
collapse of the original ML-SVM.

Key references implemented or relied upon: Cho & Saul (2010, arc-cosine kernels),
Rahimi & Recht (2007, random features), Radhakrishnan et al. (AGOP/RFM),
Mehrkanoon & Suykens (2018, deep hybrid neural-kernel networks).

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick start

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from experiments.ml_msvm import ML_MSVMClassifier  # adjust import to your tree

clf = Pipeline([
    ("scaler", StandardScaler()),          # standardisation is assumed by the model
    ("clf", ML_MSVMClassifier(
        num_layers=2,          # L: depth (0 = flat random-feature SVM baseline)
        svms_per_block=4,      # m: parallel SVMs per block (recommended default m = d)
        rff_features=2000,     # P: number of random features per block
        kernel="arc_cosine",  # definitive variant; "rbf" for cosine RFFs
        arc_cosine_degree=1,   # 0 / 1 / 2  (Cho & Saul order)
        final_C=1.0,
        normalize_inter_layer=True,
        random_state=0,
    )),
])
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
```

### Key parameters (`ML_MSVMClassifier`)

| Parameter | Symbol | Meaning |
|---|---|---|
| `num_layers` | `L` | Block depth. `0` → single random-feature map + head (flat baseline). |
| `svms_per_block` | `m` | Parallel SVMs per block. Small `m` creates an inter-layer bottleneck; default heuristic is `m = d`. |
| `rff_features` | `P` | Random features per block. |
| `kernel` | — | `"rbf"` (cosine RFFs, needs bandwidth) or `"arc_cosine"` (ReLU features, no bandwidth). |
| `arc_cosine_degree` | `n` | Arc-cosine order 0/1/2. |
| `final_C` | — | Regularisation of the head SVM. |
| `normalize_inter_layer` | — | Standardise `X_next` between arc-cosine blocks to prevent scale drift. |

### Research variants (`mlsvm_extensions.py`)

- **`DiverseMLMSVM`** — adds `diversity_mode` to study what makes `m > 1` help:
  `c_spread`, `same_c`, `bootstrap`, `feature_subset`, `disjoint` (bagging),
  `disjoint_featsub`, `input_subspace_sqrt`, `input_subspace_dm`, etc.
- **`QMC_MLMSVMClassifier`** — adds `rff_mode ∈ {standard, orf, qmc}` (i.i.d.
  Gaussian / orthogonal random features / quasi-Monte Carlo) to study whether
  better feature sampling closes the gap to the exact SVM.

---

## Experiments

Runners write append-mode CSVs (`results/`) and mirror stdout to `logs/`, so a
crashed run resumes by skipping already-completed cells.

```bash
# Experiment 3 — width (m) and depth (L) in the large-sample regime
python experiments/3_exp_width_depth.py --csv_dir results --log_dir logs

# Restrict to a subset of datasets
python experiments/3_exp_width_depth_complete.py --only digits8x8,optdigits,satimage

# Flat-kernel baselines isolating the kernel choice from the architecture
python experiments/3_exp_width_depth_flat_baselines.py

# Benchmark vs MLP / exact RBF / RFF-RBF / published numbers
python experiments/4_benchmark_copy.py
```

---

## Citation

Building on:

> Acero, J. & Belanche, L. (2025). *A new approach to multilayer SVMs.* ESANN
> 2025 proceedings.