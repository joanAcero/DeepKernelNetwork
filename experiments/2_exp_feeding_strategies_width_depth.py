"""
exp_width_depth_scenarios.py — Does width (m>1) help WHEN COMBINED WITH DEPTH (L>1),
and in which of the four supervisor scenarios?

THE HYPOTHESIS (untested by the bagging experiment)
---------------------------------------------------
The bagging experiment fixed L and found m>1 useless. But m>1 at L=1 is the WORST case
for width: the m hyperplanes are projected into W and fed straight to the head — no
intermediate layer can exploit the richer multi-view representation. The real claim is a
WIDTH x DEPTH INTERACTION: a wide block gives the NEXT layer m decorrelated views to keep
extracting from (like a hidden layer of width m feeding a deeper layer). This only pays
off at L>=2. We test m x L jointly, never marginally.

THE FOUR SCENARIOS (supervisor's 2x2)
-------------------------------------
            Standard n            Very large n
Simple    baseline (m1 L1)        bagging for efficiency (1.2)
Complex   depth+width (1.1)       depth+width+bagging (1.1 x 1.2)

We populate every cell by crossing:
  - data complexity : HIGGS (binary, hard, low ceiling) , CoverType (7-class, structured) ,
                      MNIST (10-class, high-d, needs sequential feature extraction)
  - n               : 1k, 5k, 10k, 50k, 100k  (+ 200k for HIGGS & CoverType)
  - m (width)       : 1, 10, 50
  - L (depth)       : 1, 2, 3

DIVERSITY MODE
--------------
featpart_fulldata : every SVM sees ALL the data, but the P random features are split into
                    m DISJOINT blocks of P/m each (your num_features/m idea). This
                    decorrelates the m hyperplanes by construction (orthogonal feature
                    subspaces, cos_sim≈0) while (a) using every feature exactly once and
                    (b) NOT starving any SVM of data. This isolates the width->depth benefit
                    from the data-starvation confound that hampered disjoint bagging.
disjoint_featpart : the large-n / efficiency variant — data ALSO split into m disjoint
                    partitions (n/m each), so each SVM is cheap. Tests scenario 1.2: can we
                    handle large n efficiently via bagging without losing accuracy?

block_C = 10 (moderate overfit per block), final_C = 1, arc-cosine, P = 1000.
3 seeds. We log train, test, W stable_rank & cos_sim, and time.

READING IT
----------
Key comparison is the (m, L) grid PER (dataset, n):
  - If test(m=10, L=2) > test(m=1, L=anything)  → width helps, but only with depth → 1.1 true.
  - If the win grows with n or appears only at large n → scenario 1.2 / combined.
  - If m=1 always wins → width adds nothing even with depth; architecture stays m=1.
  - featpart_fulldata vs disjoint_featpart at large n shows the accuracy COST of the
    efficiency trick (disjoint data) — if small, bagging buys speed nearly for free.
"""
from __future__ import annotations
import argparse, datetime, os, sys, time, itertools
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils import Tee, CSVWriter, load, hms, banner
from mlsvm_extensions import DiverseMLMSVM

EXP_ID    = "exp_width_depth_scenarios"
TEST_SIZE = 20_000
P         = 1000
BLOCK_C   = 10.0
FINAL_C   = 1.0
M_VALUES  = [1, 10, 50]
L_VALUES  = [1, 2, 3]
KERNEL    = "arc_cosine"
# Primary mechanism: input_subspace_sqrt (the Random-Forest analogue — each SVM sees
# sqrt(d) ORIGINAL input columns with its own Omega). Comparison arms:
#   input_subspace_dm  — d/m disjoint input columns per SVM (only valid for m <= d)
#   featpart_fulldata  — the Phi-feature partition (decorrelation in approximation space)
MODES     = ["input_subspace_sqrt", "input_subspace_dm", "featpart_fulldata"]
N_SEEDS   = 3
MIN_PER_SVM = 200    # skip disjoint cells whose n/m partition is too small to learn

# n grids per dataset (MNIST pool ~70k so it stops at 50k; binary/structured go to 200k)
N_GRID = {
    "HIGGS":     [1_000, 5_000, 10_000, 50_000, 100_000, 200_000],
    "CoverType": [1_000, 5_000, 10_000, 50_000, 100_000, 200_000],
    "MNIST":     [1_000, 5_000, 10_000, 50_000],
}
DATASETS = [("higgs", "HIGGS"), ("covertype", "CoverType"), ("mnist", "MNIST")]


def make_model(m, L, mode, seed):
    clf = DiverseMLMSVM(
        num_layers=L, svms_per_block=m, rff_features=P,
        kernel=KERNEL, arc_cosine_degree=1,
        diversity_mode=mode, block_C=BLOCK_C, final_C=FINAL_C,
        random_state=seed, normalize_inter_layer=True,
        input_subspace_full_data=True,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def make_baseline(L, seed):
    """Honest m=1 reference: ONE SVM that sees ALL input columns and the full P features
    (the plain architecture). This is the correct thing to beat — not a subspace-crippled
    m=1, which only sees sqrt(d) columns by construction."""
    clf = DiverseMLMSVM(
        num_layers=L, svms_per_block=1, rff_features=P,
        kernel=KERNEL, arc_cosine_degree=1,
        diversity_mode="same_c", block_C=BLOCK_C, final_C=FINAL_C,
        random_state=seed, normalize_inter_layer=True,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def run_cell(X, y, name, n_total, csv_w):
    d, K = X.shape[1], len(np.unique(y))
    pool = len(y)
    test_size = min(TEST_SIZE, max(2000, pool // 5))
    n_eff = min(n_total, pool - test_size)
    if n_eff < 500:
        return
    capped = " (capped)" if n_eff < n_total else ""

    print(f"\n  {'─'*88}")
    print(f"  {name}  n={n_eff:,}{capped}  (d={d}, K={K}, test={test_size:,})")
    print(f"  {'mode':18s} {'m':>3s} {'L':>2s} {'n/SVM':>7s} "
          f"{'train':>7s} {'test':>7s} {'Wcols':>6s} {'cos_sim':>8s} {'t/run':>7s}")
    print(f"  {'·'*88}")

    # ── Honest m=1 full-input baseline (the reference every m>1 mode must beat) ──
    for L in L_VALUES:
        tr_l, te_l, t_l = [], [], []
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(7000 + seed)
            idx = rng.permutation(pool)[: n_eff + test_size]
            Xs, ys = X[idx], y[idx]
            Xtr, Xte, ytr, yte = train_test_split(
                Xs, ys, test_size=test_size, random_state=seed, stratify=ys)
            try:
                model = make_baseline(L, seed)
                t = time.perf_counter(); model.fit(Xtr, ytr); dt = time.perf_counter()-t
                tr = model.score(Xtr, ytr); te = model.score(Xte, yte)
                tr_l.append(tr); te_l.append(te); t_l.append(dt)
                csv_w.write(dict(exp_id=EXP_ID, dataset=name, n_total=n_eff,
                    n_train=len(Xtr), n_test=test_size, d=d, n_classes=K,
                    model=f"baseline_fullinput_m1_L{L}", kernel=KERNEL,
                    mode="baseline_fullinput", m=1, L=L, P=P, block_C=BLOCK_C,
                    n_per_svm=len(Xtr), seed=seed, train_acc=round(tr,4),
                    acc=round(te,4), W_cols=1 if K==2 else K, stable_rank=None,
                    mean_cos_sim=None, time_s=round(dt,2)))
            except Exception as e:
                print(f"  baseline L={L} seed={seed} FAILED: {e}", flush=True)
        if te_l:
            print(f"  {'baseline(full,m1)':18s} {1:>3d} {L:>2d} {len(Xtr):>7d} "
                  f"{np.mean(tr_l):>7.4f} {np.mean(te_l):>7.4f} {'-':>6s} {'-':>8s} "
                  f"{np.mean(t_l):>6.1f}s", flush=True)
    print(f"  {'·'*88}")

    for mode in MODES:
        for m in M_VALUES:
            # input_subspace_dm: disjoint partition of d input columns needs m <= d
            if mode == "input_subspace_dm" and m > d:
                print(f"  {mode:18s} {m:>3d}  SKIP (m>d={d}: cannot make {m} disjoint "
                      f"input-column blocks)", flush=True)
                continue
            for L in L_VALUES:
                # disjoint data partitions (only when input_subspace_full_data=False) — n/a here
                tr_l, te_l, wc_l, cs_l, t_l = [], [], [], [], []
                for seed in range(N_SEEDS):
                    rng = np.random.RandomState(7000 + seed)
                    idx = rng.permutation(pool)[: n_eff + test_size]
                    Xs, ys = X[idx], y[idx]
                    Xtr, Xte, ytr, yte = train_test_split(
                        Xs, ys, test_size=test_size, random_state=seed, stratify=ys)
                    try:
                        model = make_model(m, L, mode, seed)
                        t = time.perf_counter(); model.fit(Xtr, ytr)
                        dt = time.perf_counter() - t
                        tr = model.score(Xtr, ytr); te = model.score(Xte, yte)
                        clf = model.named_steps["clf"]
                        diag = clf.W_diagnostics_[0] if clf.W_diagnostics_ else {}
                        tr_l.append(tr); te_l.append(te); t_l.append(dt)
                        wc_l.append(diag.get("n_cols", m*(1 if K==2 else K)))
                        cs_l.append(diag.get("mean_cos_sim", np.nan))
                        csv_w.write(dict(exp_id=EXP_ID, dataset=name, n_total=n_eff,
                            n_train=len(Xtr), n_test=test_size, d=d, n_classes=K,
                            model=f"{mode}_m{m}_L{L}", kernel=KERNEL, mode=mode,
                            m=m, L=L, P=P, block_C=BLOCK_C, n_per_svm=len(Xtr)//m,
                            seed=seed, train_acc=round(tr,4), acc=round(te,4),
                            W_cols=diag.get("n_cols"), stable_rank=diag.get("stable_rank"),
                            mean_cos_sim=diag.get("mean_cos_sim"), time_s=round(dt,2)))
                    except Exception as e:
                        print(f"  {mode} m={m} L={L} seed={seed} FAILED: {e}", flush=True)
                if te_l:
                    print(f"  {mode:18s} {m:>3d} {L:>2d} {len(Xtr)//m:>7d} "
                          f"{np.mean(tr_l):>7.4f} {np.mean(te_l):>7.4f} "
                          f"{int(np.mean(wc_l)):>6d} {np.nanmean(cs_l):>8.4f} "
                          f"{np.mean(t_l):>6.1f}s", flush=True)
            if mode == "disjoint_featpart" and 1 not in (M_VALUES):
                pass
        print()


def run(log_path, csv_path, datasets):
    tee = Tee(sys.stdout, log_path); sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Exp — Width x Depth across the four scenarios",
               f"modes={MODES}  P={P}  block_C={BLOCK_C}  seeds={N_SEEDS}",
               f"m={M_VALUES}  L={L_VALUES}  (width tested JOINTLY with depth)",
               "Q: does m>1 help when L>=2? In which (complexity, n) scenario?")
        t0 = time.perf_counter()
        loaded = {}
        for tag, name in datasets:
            try:
                X, y = load(tag); loaded[name] = (X, y)
                print(f"  loaded {name}: {len(y):,} rows, d={X.shape[1]}, K={len(np.unique(y))}")
            except Exception as e:
                print(f"  [{name}] LOAD FAILED: {e}")

        for name, (X, y) in loaded.items():
            print(f"\n\n{'='*92}")
            print(f"  DATASET: {name}")
            print(f"{'='*92}")
            for n_total in N_GRID.get(name, [10_000]):
                run_cell(X, y, name, n_total, csv_w)

        print(f"\n  exp_width_depth_scenarios complete. {hms(time.perf_counter()-t0)}")
        print("\n  READING THE RESULTS (per dataset x n, look at the m x L grid):")
        print("  - test(m=10,L>=2) > test(m=1,*)  → WIDTH HELPS WITH DEPTH (scenario 1.1).")
        print("  - effect appears/grows with n     → large-n benefit (scenario 1.2 / combined).")
        print("  - m=1 always best                 → width adds nothing; keep m=1.")
        print("  - disjoint_featpart vs featpart_fulldata at large n = accuracy COST of the")
        print("    efficiency trick (disjoint data → each SVM cheap). Small gap = cheap win.")
    finally:
        sys.stdout = tee._stream; tee.close(); csv_w.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--csv_dir", default="results")
    p.add_argument("--only", choices=["higgs","covertype","mnist"], help="run one dataset")
    a = p.parse_args()
    ds = [("higgs","HIGGS"),("covertype","CoverType"),("mnist","MNIST")]
    if a.only:
        ds = [(a.only, {"higgs":"HIGGS","covertype":"CoverType","mnist":"MNIST"}[a.only])]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run(os.path.join(a.log_dir, f"exp_width_depth_scenarios_{ts}.txt"),
        os.path.join(a.csv_dir, "exp_width_depth_scenarios.csv"), ds)
