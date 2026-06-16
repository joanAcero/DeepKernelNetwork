"""
exp_overfit_capacity.py — Experiment 1: overfitting capacity.

Goal: demonstrate that the proposed architecture has the capacity to OVERFIT hard, non-linear
problems that LINEAR and other WEAK models cannot even fit on the training set, so that the
rest of the study (where width m and depth L are shown to REGULARISE) begins from a model that
is genuinely capacity-rich rather than intrinsically limited.

Scope (deliberately narrow): capacity is demonstrated through P, the number of random features,
with a single SVM per block (m=1, L=1). The roles of m and L are the subject of Experiments 2
and 3 and are not varied here. Capacity is shown for BOTH kernels (arc-cosine and RBF), so the
claim is about the architecture's feature-map mechanism, not one particular kernel.

Three structurally different synthetic problems are used so the result is not an artefact of one
generator. Each has n_train=1500, d=50 with only a few informative features (the rest Gaussian
noise) and d<n so the data is not trivially separable in input space; a model that fits the
training set must therefore use genuine non-linear capacity:
  - radial : radial threshold XOR two-way product (curved, multi-region boundary);
  - spiral : two interleaving spirals (classic hard non-linear boundary);
  - checker: a checkerboard parity over a 2-D informative subspace (high-frequency boundary).

Output: prints a table per dataset; optionally writes results/exp_overfit_capacity.csv.
"""
from __future__ import annotations
import argparse, importlib.util, os, sys
from pathlib import Path
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

N_SEEDS = 3
N_TRAIN = 1500
N_TEST  = 500
D_TOTAL = 50
P_GRID  = [100, 500, 1000, 2000, 5000]

# ----------------------------- synthetic datasets -----------------------------
def ds_radial(n, d=D_TOTAL, seed=0):
    rng = np.random.default_rng(seed)
    Xi = rng.standard_normal((n, 4))
    r = np.sqrt((Xi[:, :2] ** 2).sum(1))
    y = (((r > 1.0).astype(int) + (Xi[:, 2] * Xi[:, 3] > 0).astype(int)) % 2)
    Xn = rng.standard_normal((n, d - 4))
    X = np.hstack([Xi, Xn]); X = X[:, rng.permutation(d)]
    return X.astype(np.float64), y.astype(int)

def ds_spiral(n, d=D_TOTAL, seed=0, turns=2.5, noise=0.18):
    rng = np.random.default_rng(seed); nh = n // 2
    def arm(sign, m):
        t = np.sqrt(rng.random(m)) * turns * 2 * np.pi
        rr = t / (2 * np.pi * turns)
        x = sign * rr * np.cos(t) + rng.normal(0, noise, m)
        z = sign * rr * np.sin(t) + rng.normal(0, noise, m)
        return np.stack([x, z], 1)
    Xi = np.vstack([arm(1, nh), arm(-1, n - nh)])
    y = np.array([0] * nh + [1] * (n - nh))
    Xn = rng.standard_normal((n, d - 2))
    X = np.hstack([Xi, Xn]); X = X[:, rng.permutation(d)]
    order = rng.permutation(n)
    return X[order].astype(np.float64), y[order].astype(int)

def ds_checker(n, d=D_TOTAL, seed=0, freq=3):
    rng = np.random.default_rng(seed)
    Xi = rng.uniform(-1, 1, (n, 2))
    y = ((np.floor((Xi[:, 0] + 1) * freq).astype(int) +
          np.floor((Xi[:, 1] + 1) * freq).astype(int)) % 2)
    Xn = rng.standard_normal((n, d - 2))
    X = np.hstack([Xi, Xn]); X = X[:, rng.permutation(d)]
    return X.astype(np.float64), y.astype(int)

DATASETS = {"radial": ds_radial, "spiral": ds_spiral, "checker": ds_checker}

# ----------------------------- classifier import -----------------------------
def import_clf():
    here = Path(__file__).resolve().parent
    for cand in [here / "ml_msvm.py", here / ".." / "ml_msvm" / "ml_msvm.py",
                 here / "mlsvm" / "ml_msvm.py"]:
        if cand.exists():
            spec = importlib.util.spec_from_file_location("ml_msvm", str(cand.resolve()))
            mod = importlib.util.module_from_spec(spec); sys.modules["ml_msvm"] = mod
            spec.loader.exec_module(mod); return mod.ML_MSVMClassifier
    raise FileNotFoundError("ml_msvm.py not found next to script, in ../ml_msvm/, or ./mlsvm/")
ML = import_clf()

# ----------------------------- model builders --------------------------------
def linear_svm(s):
    return Pipeline([("s", StandardScaler()),
                     ("c", LinearSVC(C=1.0, dual=True, max_iter=10000))])
def dtree(s):
    return DecisionTreeClassifier(max_depth=3, random_state=0)
def arc(P):
    return lambda s: ML(num_layers=1, svms_per_block=1, rff_features=P, kernel="arc_cosine",
                        arc_cosine_degree=1, final_C=1e6, C_values=[1e6],
                        normalize_inter_layer=True, random_state=0)
def rbf(P):
    return lambda s: ML(num_layers=1, svms_per_block=1, rff_features=P, kernel="rbf",
                        final_C=1e6, C_values=[1e6], normalize_inter_layer=True, random_state=0)

def trial(make_model, gen):
    tr, te = [], []
    for s in range(N_SEEDS):
        X, y = gen(N_TRAIN + N_TEST, seed=s)
        Xtr, ytr, Xte, yte = X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]
        m = make_model(s); m.fit(Xtr, ytr)
        tr.append(m.score(Xtr, ytr)); te.append(m.score(Xte, yte))
    return float(np.mean(tr)), float(np.std(tr)), float(np.mean(te)), float(np.std(te))

def run(csv_path=None):
    rows = []
    print("Experiment 1 — overfitting capacity")
    print(f"n_train={N_TRAIN}, n_test={N_TEST}, d={D_TOTAL} (few informative), "
          f"{N_SEEDS} seeds; capacity via P only (m=1, L=1)")
    for name, gen in DATASETS.items():
        print(f"\n{'='*70}\n  DATASET: {name}\n{'='*70}")
        print(f"  {'model':30s} {'train_acc':>14s}   {'test_acc':>14s}")
        print("  " + "-" * 64)
        def line(label, res):
            a, sa, b, sb = res
            print(f"  {label:30s} {a:>6.3f} ± {sa:.3f}   {b:>6.3f} ± {sb:.3f}")
            rows.append((name, label, a, sa, b, sb))
        print("  weak models — cannot fit the training set:")
        line("Linear SVM", trial(linear_svm, gen))
        line("Decision tree (depth 3)", trial(dtree, gen))
        print("\n  arc-cosine kernel, P sweep:")
        for P in P_GRID:
            line(f"arc-cosine  P={P}", trial(arc(P), gen))
        print("\n  RBF kernel, P sweep:")
        for P in P_GRID:
            line(f"RBF  P={P}", trial(rbf(P), gen))

    print(f"\n{'='*70}")
    print("  Reading: weak models stay near chance on TRAIN; both kernels reach train=1.0")
    print("  as P grows while test stays near chance -> the architecture has the capacity to")
    print("  overfit, supplied by the feature dimension P. (m and L are studied in Exp 2-3.)")

    if csv_path:
        import csv
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset","model","train_mean","train_std","test_mean","test_std"])
            w.writerows(rows)
        print(f"\n  wrote {csv_path}")
    return rows

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None, help="optional path to write results CSV")
    a = p.parse_args()
    run(a.csv)
