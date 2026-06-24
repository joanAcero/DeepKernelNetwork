"""
exp_feeding_strategies.py — Assuming m>1, what is the best way to feed each SVM?

This COMPLETES the feeding-strategy comparison. Strategies reused from prior runs
(NOT recomputed here):
    A         baseline(full,m1)            — all data, all columns, m=1 (the reference)
    B_phi     featpart_fulldata            — all data + P/m random-FEATURE partition

Strategies run HERE, all with FULL P=1000 features PER SVM (so arms differ only in
WHICH data/columns each SVM sees, never in approximation budget):
    C            all d columns, DISJOINT n/m data slice per SVM      (pure data bagging)
    B_in_sqrt    all n rows, sqrt(d) random INPUT columns per SVM    (RF random subspace)
    B_in_dm      all n rows, d/m disjoint INPUT columns per SVM      (m<=d only)
    D_sqrt_fullP data split (n/m) + sqrt(d) input columns            (D, budget-matched)
    D_dm_fullP   data split (n/m) + d/m disjoint input columns       (D, budget-matched)

WHY D IS RE-RUN HERE (not reused): the earlier D runs used P/m features per SVM. At
m=50 that is only 20 features vs the full P=1000 used by B_in_*/C — so a D-vs-B
difference would conflate the data-split axis with a 50x feature-budget gap. Re-running
D at full P makes every m>1 arm budget-matched and the comparison fair.

NOTE on the three "B" kinds (for the report):
  B_in_sqrt / B_in_dm subset the RAW input variables → each SVM is blind to some inputs.
  B_phi subsets the RANDOM FEATURES (Phi) → every feature still depends on ALL inputs,
  so the SVM is NOT blind to any variable; it sees all information through fewer random
  directions. Comparing B_in_* vs B_phi isolates whether variable-blindness matters or
  only the degree of decorrelation.

Fixed settings (MATCH the existing A/D/B_phi runs so the table is comparable):
    P=1000, block_C=10, final_C=1, arc-cosine deg=1, seeds=3,
    n grid HIGGS/CoverType {1k,5k,10k,50k,100k,200k}, MNIST {1k,5k,10k,50k}.
Each SVM here builds a FULL P-feature map (input_subspace_full_P=True) so B variants
differ only in WHICH columns they see, not in approximation budget.
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils import Tee, CSVWriter, load, hms, banner
from small_datasets import load_small
from mlsvm_extensions import DiverseMLMSVM

EXP_ID    = "exp_feeding_strategies"
TEST_SIZE = 20_000
P         = 1000
BLOCK_C   = 10.0
FINAL_C   = 1.0
M_VALUES  = [1, 10, 50]
L_VALUES  = [1, 2]          # L=3 dropped: prior results show L=2 vs L=3 is within noise
KERNEL    = "arc_cosine"
N_SEEDS   = 3
MIN_PER_SVM = 200
# Cost/relevance trim: the m>1 width benefit is a SMALL-N regularization effect that
# (per earlier runs) decays by n~50k. The full-P m=50 cells are the most expensive in
# the whole grid AND least informative at large n. So m=50 is only evaluated up to
# MAX_M50_N; for n above that, only m in {1,10} run (where strategies stay distinguishable).
MAX_M50_N = 50_000

# arm = (label, diversity_mode, full_data)
#   C        : disjoint mode (data split), full columns
#   B_in_*   : input-subspace modes, full data (all rows)
#   D_*_fullP: input-subspace modes, data split (n/m) — re-run at FULL P so they are
#              budget-matched to B_in_* and C (the prior D runs used P/m per SVM, which
#              is NOT comparable at m=50: P/m=20 vs full P=1000 changes everything).
ARMS = [
    ("C_splitdata_fullcols", "disjoint",            False),
    ("B_in_sqrt",            "input_subspace_sqrt", True),
    ("B_in_dm",              "input_subspace_dm",   True),
    ("D_sqrt_fullP",         "input_subspace_sqrt", False),
    ("D_dm_fullP",           "input_subspace_dm",   False),
]
N_GRID = {
    "HIGGS":     [1_000, 5_000, 10_000, 50_000, 100_000, 200_000],
    "CoverType": [1_000, 5_000, 10_000, 50_000, 100_000, 200_000],
    "MNIST":     [1_000, 5_000, 10_000, 50_000],
    # --- small UCI datasets (<=~6k): sweep small-n, last point uses full pool ---
    "Optdigits": [500, 1_000, 2_000, 3_823],   # canonical train size 3,823
    "Satimage":  [500, 1_000, 2_000, 4_435],   # canonical train size 4,435
    "Digits8x8": [500, 1_000, 1_400],          # sklearn 8x8 digits (offline-safe)
}


def make_model(m, L, mode, full_data, seed):
    clf = DiverseMLMSVM(
        num_layers=L, svms_per_block=m, rff_features=P,
        kernel=KERNEL, arc_cosine_degree=1,
        diversity_mode=mode, block_C=BLOCK_C, final_C=FINAL_C,
        random_state=seed, normalize_inter_layer=True,
        input_subspace_full_data=full_data,
        input_subspace_full_P=True,          # each SVM builds a FULL P-feature map
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def run_cell(X, y, name, n_total, csv_w):
    d, K = X.shape[1], len(np.unique(y))
    pool = len(y)
    # test split: 20% of pool, capped at TEST_SIZE (large data) and floored at 500
    # (small data). For large pools this is exactly TEST_SIZE, unchanged from the
    # big-dataset runs, preserving commensurability; for small pools it scales down.
    test_size = min(TEST_SIZE, max(500, pool // 5))
    n_eff = min(n_total, pool - test_size)
    if n_eff < 500:
        return
    capped = " (capped)" if n_eff < n_total else ""
    print(f"\n  {'─'*90}")
    print(f"  {name}  n={n_eff:,}{capped}  (d={d}, K={K}, test={test_size:,})")
    print(f"  {'arm':22s} {'m':>3s} {'L':>2s} {'n/SVM':>7s} {'cols/SVM':>8s} "
          f"{'train':>7s} {'test':>7s} {'cos_sim':>8s} {'t/run':>7s}")
    print(f"  {'·'*90}")

    for arm_label, mode, full_data in ARMS:
        for m in M_VALUES:
            if mode == "input_subspace_dm" and m > d:
                print(f"  {arm_label:22s} {m:>3d}  SKIP (m>d={d})", flush=True); continue
            if m >= 50 and n_eff > MAX_M50_N:
                print(f"  {arm_label:22s} {m:>3d}  SKIP (m={m} only run up to n={MAX_M50_N:,}; "
                      f"large-n reserved for m<=10)", flush=True); continue
            if not full_data and m > 1 and (n_eff // m) < MIN_PER_SVM:
                print(f"  {arm_label:22s} {m:>3d}  SKIP (n/SVM={n_eff//m}<{MIN_PER_SVM})", flush=True); continue
            # columns each SVM sees (for the log)
            if mode == "input_subspace_sqrt": cps = max(1, int(round(np.sqrt(d))))
            elif mode == "input_subspace_dm": cps = max(1, d // m)
            else:                              cps = d
            for L in L_VALUES:
                tr_l, te_l, cs_l, t_l = [], [], [], []
                for seed in range(N_SEEDS):
                    rng = np.random.RandomState(7000 + seed)   # SAME seeding as prior runs
                    idx = rng.permutation(pool)[: n_eff + test_size]
                    Xs, ys = X[idx], y[idx]
                    Xtr, Xte, ytr, yte = train_test_split(
                        Xs, ys, test_size=test_size, random_state=seed, stratify=ys)
                    try:
                        model = make_model(m, L, mode, full_data, seed)
                        t = time.perf_counter(); model.fit(Xtr, ytr); dt = time.perf_counter()-t
                        tr = model.score(Xtr, ytr); te = model.score(Xte, yte)
                        clf = model.named_steps["clf"]
                        diag = clf.W_diagnostics_[0] if clf.W_diagnostics_ else {}
                        tr_l.append(tr); te_l.append(te); t_l.append(dt)
                        cs_l.append(diag.get("mean_cos_sim", np.nan))
                        csv_w.write(dict(exp_id=EXP_ID, dataset=name, n_total=n_eff,
                            n_train=len(Xtr), n_test=test_size, d=d, n_classes=K,
                            model=f"{arm_label}_m{m}_L{L}", arm=arm_label, mode=mode,
                            full_data=full_data, cols_per_svm=cps, kernel=KERNEL,
                            m=m, L=L, P=P, block_C=BLOCK_C, n_per_svm=len(Xtr)//m,
                            seed=seed, train_acc=round(tr,4), acc=round(te,4),
                            mean_cos_sim=diag.get("mean_cos_sim"), time_s=round(dt,2)))
                    except Exception as e:
                        print(f"  {arm_label} m={m} L={L} seed={seed} FAILED: {e}", flush=True)
                if te_l:
                    print(f"  {arm_label:22s} {m:>3d} {L:>2d} {len(Xtr)//m:>7d} {cps:>8d} "
                          f"{np.mean(tr_l):>7.4f} {np.mean(te_l):>7.4f} "
                          f"{np.nanmean(cs_l):>8.4f} {np.mean(t_l):>6.1f}s", flush=True)
            print()


def run(log_path, csv_path, datasets):
    tee = Tee(sys.stdout, log_path); sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Exp — Best feeding strategy for m>1 (adds C, B_in_sqrt, B_in_dm)",
               f"P={P}  block_C={BLOCK_C}  final_C={FINAL_C}  seeds={N_SEEDS}  full-P per SVM",
               f"arms={[a[0] for a in ARMS]}  m={M_VALUES}  L={L_VALUES} (m=50 only n<={MAX_M50_N:,})",
               "Reuse A (baseline) and B_phi (featpart) from prior runs for the full table.")
        t0 = time.perf_counter()
        loaded = {}
        SMALL = {"optdigits", "satimage", "digits8x8"}
        for tag, nm in datasets:
            try:
                X, y = (load_small(tag) if tag in SMALL else load(tag))
                loaded[nm] = (X, y)
                print(f"  loaded {nm}: {len(y):,} rows, d={X.shape[1]}, K={len(np.unique(y))}")
            except Exception as e:
                print(f"  [{nm}] LOAD FAILED: {e}")
        for nm, (X, y) in loaded.items():
            print(f"\n\n{'='*92}\n  DATASET: {nm}\n{'='*92}")
            for n_total in N_GRID.get(nm, [10_000]):
                run_cell(X, y, nm, n_total, csv_w)
        print(f"\n  exp_feeding_strategies complete. {hms(time.perf_counter()-t0)}")
        print("\n  READING (combine with reused A baseline + B_phi from prior runs):")
        print("  - Best m>1 strategy = highest test at each n. Compare C vs B_in_* vs (reused) D, B_phi.")
        print("  - B_in_sqrt vs B_phi: does variable-blindness beat random-direction subsetting?")
        print("  - C vs B_in_*: does splitting DATA or splitting COLUMNS feed SVMs better?")
        print("  - Track vs n: small-n regularization gain vs large-n convergence to baseline.")
    finally:
        sys.stdout = tee._stream; tee.close(); csv_w.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--csv_dir", default="results")
    p.add_argument("--only", choices=["higgs","covertype","mnist",
                                      "optdigits","satimage","digits8x8"])
    a = p.parse_args()
    ds = [("higgs","HIGGS"),("covertype","CoverType"),("mnist","MNIST"),
          ("optdigits","Optdigits"),("satimage","Satimage")]
    NAME = {"higgs":"HIGGS","covertype":"CoverType","mnist":"MNIST",
            "optdigits":"Optdigits","satimage":"Satimage","digits8x8":"Digits8x8"}
    if a.only:
        ds = [(a.only, NAME[a.only])]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run(os.path.join(a.log_dir, f"exp_feeding_strategies_{ts}.txt"),
        os.path.join(a.csv_dir, "exp_feeding_strategies.csv"), ds)
