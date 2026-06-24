"""
exp3_width_depth.py — Experiment 3: Width and Depth in the Large-Sample Complex Regime.

Implements the methodology of Section \ref{exp:widthdepth}:

  Grid:        m in {20, 50, 100}  x  L in {1, 2, 3, 4, 5}
  Datasets:    gap panel  -> poker, covertype, chess_krk, spirals, checkerboard
               controls    -> miniboone, higgs   (saturated; depth predicted ~0 gain)
  Sample size: 3 values per dataset, scaling up to 100k where the pool allows,
               otherwise the largest the dataset supports.
  Seeds:       3 per cell.
  Feeding:     regime-appropriate (Experiment 2): feature partitioning when d is large,
               instance partitioning when d is small with ample n.
  References per (dataset, n, seed):
       - flat arc-cosine SVM (architecture with no blocks; isolates kernel from machinery)
       - exact RBF SVM        (run where n <= RBF_MAX_N; marks feasibility boundary)
       - RFF-RBF SVM          (scalable RBF stand-in; runs at every n)
  Metrics (train AND test): accuracy, macro-F1, macro-AUC (OvR).
  Timing: wall-clock seconds recorded for EVERY fit (arch cells and all baselines),
          so the time-vs-(n, L, m) plots can be built from the CSV afterwards.

Falsifiable prediction recorded by construction: depth gain should track the
MLP-RBF gap of the screen — large on the gap panel, ~0 on the controls.

Output: results_exp3.csv  (one row per fit; long format for easy plotting).

Usage:
  python3 exp3_width_depth.py
  python3 exp3_width_depth.py --only covertype spirals      # subset
  python3 exp3_width_depth.py --quick                       # 1 seed, smallest n, m{20} L{1,3} smoke test
"""
from __future__ import annotations
import sys, time, csv, argparse, warnings, datetime
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.datasets import fetch_openml
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mlsvm_extensions import DiverseMLMSVM

# ----------------------------------------------------------------------
# Grid / protocol
# ----------------------------------------------------------------------
M_VALUES = [20, 50, 70]
L_VALUES = [1, 2, 3, 4, 5]
N_SEEDS  = 2
P        = 1000          # per-SVM random-feature budget (fixed; Experiment 2 control)
RBF_MAX_N = 15_000       # exact RBF only run at or below this n
RFF_P     = 1000         # RFF-RBF stand-in resolution (validated coarse<1.5pt in screen)
TEST_SIZE = 10_000
HIGH_D_THRESHOLD = 60    # d>=this -> feature partitioning; else instance partitioning

# ----------------------------------------------------------------------
# Dataset loaders.  Each returns (X float64, y int, is_temporal).
# Synthetic generators reproduce the screen's spirals/checkerboard.
# ----------------------------------------------------------------------
def _openml(name=None, data_id=None, version=1):
    d = (fetch_openml(data_id=data_id, as_frame=True, parser="auto") if data_id
         else fetch_openml(name=name, version=version, as_frame=True, parser="auto"))
    Xdf = pd.get_dummies(d.data, drop_first=False)
    X = np.nan_to_num(Xdf.to_numpy(dtype=float))
    y = LabelEncoder().fit_transform(d.target)
    return X, y

def make_spirals(n, turns=2.5, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.uniform(0, 1, (n, 1))) * turns * 2*np.pi
    a = np.hstack([-np.cos(t)*t, np.sin(t)*t]) + rng.standard_normal((n,2))*noise
    b = np.hstack([ np.cos(t)*t,-np.sin(t)*t]) + rng.standard_normal((n,2))*noise
    X = np.vstack([a, b]); y = np.hstack([np.zeros(n), np.ones(n)]).astype(int)
    i = rng.permutation(len(y)); return X[i], y[i]

def make_checkerboard(n, k=4, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, k, (n, 2))
    y = ((np.floor(X[:,0]) + np.floor(X[:,1])) % 2).astype(int)
    return X, y

DATASETS = {
    # gap panel
    "poker":       lambda: _openml(name="poker-hand", version=1),
    "chess_krk":   lambda: _openml(data_id=1481),
    "spirals":     lambda: make_spirals(60000),       # 120k total; binary
    "checkerboard":lambda: make_checkerboard(120000), # binary
    # saturated controls
    "covertype":   lambda: _openml(data_id=1596),
    "miniboone":   lambda: _openml(name="MiniBooNE", version=1),
    "higgs":       lambda: _openml(name="higgs", version=1),
}
GAP_PANEL = {"poker","covertype","chess_krk","spirals","checkerboard"}
CONTROLS  = {"miniboone","higgs"}

N_CAP   = 100_000          # hard upper bound on training size
N_FRACS = (0.3, 0.5, 0.8)  # three sample sizes as fractions of each dataset's usable pool
N_FLOOR = 1_000            # skip a fraction if it would fall below this

def n_grid_for(pool):
    """Three sample sizes per dataset, always three points where possible, never above N_CAP.

    Two regimes:
      - usable <= N_CAP : the dataset is smaller than the cap, so use fractions of its
        OWN usable pool, {0.4, 0.7, 1.0} x usable. Small datasets (e.g. chess_krk) thus
        still get three runs scaled to their size.
      - usable >  N_CAP : the dataset is larger than the cap, so fractions would all sit
        at/above 100k and collapse to one point. Instead use three FIXED sizes ending at
        the cap, {0.4, 0.7, 1.0} x N_CAP, so the n-scaling trend is still measured below
        the cap (e.g. poker/covertype -> 40k, 70k, 100k).
    Each value is floored at N_FLOOR; duplicates collapsed.
    """
    usable = max(0, pool - TEST_SIZE)
    base = usable if usable <= N_CAP else N_CAP
    grid = sorted({int(base * f) for f in N_FRACS if int(base * f) >= N_FLOOR})
    return grid or [min(usable, N_CAP)]

def feeding_for(d):
    return "disjoint_featpart" if d >= HIGH_D_THRESHOLD else "disjoint"

# ----------------------------------------------------------------------
# Metrics (train and test), robust to binary/multiclass
# ----------------------------------------------------------------------
def metrics(model, X, y, classes):
    yp = model.predict(X)
    acc = accuracy_score(y, yp)
    f1  = f1_score(y, yp, average="macro")
    try:
        if hasattr(model, "decision_function"):
            s = model.decision_function(X)
        else:
            s = model.predict_proba(X)
        if len(classes) == 2:
            s1 = s if s.ndim == 1 else s[:, 1]
            auc = roc_auc_score(y, s1)
        else:
            Yb = label_binarize(y, classes=classes)
            auc = roc_auc_score(Yb, s, average="macro", multi_class="ovr")
    except Exception:
        auc = np.nan
    return acc, f1, auc

def timed_fit(estimator, Xtr, ytr):
    t0 = time.perf_counter()
    estimator.fit(Xtr, ytr)
    return estimator, time.perf_counter() - t0

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    m_values, l_values, n_seeds = M_VALUES, L_VALUES, N_SEEDS
    names = args.only if args.only else list(DATASETS)
    if args.quick:
        m_values, l_values, n_seeds = [20], [1, 3], 1

    import os
    csv_path = "results_exp3.csv"
    HEADER = ["dataset","panel","d","n","seed","method","m","L","feeding",
              "train_acc","train_f1","train_auc","test_acc","test_f1","test_auc",
              "fit_time_s","timestamp"]
    # Resume-safe: load already-completed cells so we never recompute or clobber them.
    done = set()
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            prev = pd.read_csv(csv_path)
            for _, r in prev.iterrows():
                done.add((str(r["dataset"]), int(r["n"]), int(r["seed"]),
                          str(r["method"]), int(r["m"]), int(r["L"])))
            print(f"Resuming: {len(done)} completed cells found in {csv_path}; "
                  f"these will be skipped.")
        except Exception as e:
            print(f"(could not parse existing CSV, appending anyway: {e})")
    new_file = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    out = open(csv_path, "a", newline=""); cw = csv.writer(out)
    if new_file:
        cw.writerow(HEADER)

    def already(ds, n, seed, method, m, L):
        return (str(ds), int(n), int(seed), str(method), int(m), int(L)) in done

    def emit(ds, panel, d, n, seed, method, m, L, feeding, tr, te, t):
        cw.writerow([ds, panel, d, n, seed, method, m, L, feeding,
                     round(tr[0],4),round(tr[1],4),round(tr[2],4) if not np.isnan(tr[2]) else "",
                     round(te[0],4),round(te[1],4),round(te[2],4) if not np.isnan(te[2]) else "",
                     round(t,3), datetime.datetime.now().isoformat()]); out.flush()

    print(f"EXPERIMENT 3 — width x depth.  m={m_values} L={l_values} seeds={n_seeds}")
    print(f"{'dataset':11s} {'n':>7s} {'sd':>2s} {'m':>4s} {'L':>2s} | "
          f"{'te_acc':>7s} {'te_f1':>7s} {'te_auc':>7s} {'time_s':>8s}")
    print("-"*64)
    # track per-(dataset,n,m,seed) L=1 acc so we can print the depth gain live
    _l1 = {}

    for name in names:
        panel = "gap" if name in GAP_PANEL else "control"
        try:
            X, y = DATASETS[name]()
        except Exception as e:
            print(f"{name:12s} LOAD FAILED: {str(e)[:50]}"); continue
        d, pool = X.shape[1], len(y); classes = np.unique(y)
        feeding = feeding_for(d)
        for n in n_grid_for(pool):
            ts = min(TEST_SIZE, max(int(pool*0.2), 200))
            n_eff = min(n, pool - ts)
            for seed in range(n_seeds):
                rng = np.random.RandomState(7000+seed)
                idx = rng.permutation(pool)[:n_eff+ts]
                try:
                    Xtr,Xte,ytr,yte = train_test_split(X[idx],y[idx],test_size=ts,
                                                       random_state=seed,stratify=y[idx])
                except ValueError:
                    Xtr,Xte,ytr,yte = train_test_split(X[idx],y[idx],test_size=ts,
                                                       random_state=seed)

                if seed == 0:   # baselines once per (dataset,n); arch uses all seeds
                    # ---- baseline 1: flat arc-cosine SVM (same kernel, no blocks) ----
                    # One arc-cosine random-feature map (degree 1 = ReLU, Cho & Saul) followed
                    # by a single linear SVM. Isolates the kernel from the width/depth machinery.
                    sc0 = StandardScaler().fit(Xtr)
                    rng0 = np.random.default_rng(seed)
                    Omega0 = rng0.standard_normal((P, d))
                    def _arccos1(Z):
                        return np.sqrt(2.0 / P) * np.maximum(0.0, Z)
                    Phi_tr = _arccos1(sc0.transform(Xtr) @ Omega0.T)
                    Phi_te = _arccos1(sc0.transform(Xte) @ Omega0.T)
                    flin = LinearSVC(C=1.0, max_iter=5000, tol=1e-3, dual="auto")
                    t0 = time.perf_counter(); flin.fit(Phi_tr, ytr); tf = time.perf_counter() - t0
                    class _F:
                        def __init__(s, m): s.m = m
                        def predict(s, Z): return s.m.predict(Z)
                        def decision_function(s, Z): return s.m.decision_function(Z)
                    wf = _F(flin)
                    if not already(name,n_eff,seed,"flat_arccos",0,0):
                        emit(name,panel,d,n_eff,seed,"flat_arccos",0,0,feeding,
                             metrics(wf,Phi_tr,ytr,classes),metrics(wf,Phi_te,yte,classes),tf)

                    # ---- baseline 2: exact RBF (feasible region only) ----
                    if n_eff <= RBF_MAX_N:
                        sc=StandardScaler().fit(Xtr)
                        rbf=SVC(kernel="rbf",C=10,gamma="scale",probability=False)
                        t0=time.perf_counter(); rbf.fit(sc.transform(Xtr),ytr); tr=time.perf_counter()-t0
                        class _W:  # wrap for uniform metrics() interface
                            def __init__(s,m,sc): s.m,s.sc=m,sc
                            def predict(s,Z): return s.m.predict(s.sc.transform(Z))
                            def decision_function(s,Z): return s.m.decision_function(s.sc.transform(Z))
                        w=_W(rbf,sc)
                        if not already(name,n_eff,seed,"rbf_exact",0,0):
                            emit(name,panel,d,n_eff,seed,"rbf_exact",0,0,feeding,
                                 metrics(w,Xtr,ytr,classes),metrics(w,Xte,yte,classes),tr)

                    # ---- baseline 3: RFF-RBF (scalable stand-in; every n) ----
                    if not already(name,n_eff,seed,"rbf_rff",0,0):
                     sc=StandardScaler().fit(Xtr)
                     g=1.0/(d*sc.transform(Xtr).var()) if sc.transform(Xtr).var()>0 else 1.0
                     rffmap=RBFSampler(gamma=g,n_components=RFF_P,random_state=seed)
                     Ztr=rffmap.fit_transform(sc.transform(Xtr)); Zte=rffmap.transform(sc.transform(Xte))
                     from sklearn.linear_model import SGDClassifier
                     lin=SGDClassifier(loss="hinge",alpha=1e-4,max_iter=20,tol=1e-3,
                                       random_state=seed,n_jobs=-1)
                     t0=time.perf_counter(); lin.fit(Ztr,ytr); trr=time.perf_counter()-t0
                     class _R:
                         def __init__(s,m,mp,sc): s.m,s.mp,s.sc=m,mp,sc
                         def predict(s,Z): return s.m.predict(s.mp.transform(s.sc.transform(Z)))
                         def decision_function(s,Z): return s.m.decision_function(s.mp.transform(s.sc.transform(Z)))
                     wr=_R(lin,rffmap,sc)
                     if not already(name,n_eff,seed,"rbf_rff",0,0):
                         emit(name,panel,d,n_eff,seed,"rbf_rff",0,0,feeding,
                              metrics(wr,Xtr,ytr,classes),metrics(wr,Xte,yte,classes),trr)

                # ---- the architecture: full m x L grid ----
                for m in m_values:
                    for L in l_values:
                        if already(name,n_eff,seed,"arch",m,L):
                            continue
                        arch = Pipeline([("sc",StandardScaler()),
                            ("clf",DiverseMLMSVM(num_layers=L, svms_per_block=m, rff_features=P,
                                kernel="arc_cosine", arc_cosine_degree=1, diversity_mode=feeding,
                                block_C=10.0, final_C=1.0, random_state=seed,
                                normalize_inter_layer=True))])
                        try:
                            arch,ta = timed_fit(arch,Xtr,ytr)
                            te = metrics(arch,Xte,yte,classes)
                            emit(name,panel,d,n_eff,seed,"arch",m,L,feeding,
                                 metrics(arch,Xtr,ytr,classes),te,ta)
                            if L == min(l_values):
                                _l1[(name,n_eff,m,seed)] = te[0]
                            base = _l1.get((name,n_eff,m,seed))
                            gtxt = f"(d{te[0]-base:+.3f})" if base is not None and L!=min(l_values) else ""
                            f1auc = f"{te[1]:7.4f} {te[2]:7.4f}" if not np.isnan(te[2]) else f"{te[1]:7.4f}    -- "
                            print(f"{name:11s} {n_eff:7d} {seed:2d} {m:4d} {L:2d} | "
                                  f"{te[0]:7.4f} {f1auc} {ta:8.2f} {gtxt}")
                        except Exception as e:
                            print(f"{name:11s} {n_eff:7d} {seed:2d} {m:4d} {L:2d} FAILED: {str(e)[:35]}")
    out.close()
    print("\nWrote results_exp3.csv  (long format: one row per fit, all timed).")

    # ---- end-of-run summary: depth gain per dataset (best width, largest n) ----
    # Reads the CSV we just wrote so the summary reflects exactly what was recorded.
    try:
        df = pd.read_csv("results_exp3.csv")
        a = df[df.method == "arch"]
        print("\n" + "="*70)
        print("DEPTH-GAIN SUMMARY  (best test_acc over L, minus L=1; at each dataset's max n)")
        print("  prediction: large on the GAP panel, ~0 on the CONTROLS")
        print("="*70)
        print(f"{'dataset':12s} {'panel':>8s} {'n':>7s} {'best_m':>6s} {'acc@L1':>7s} "
              f"{'best':>7s} {'depthgain':>9s} {'best_L':>6s}")
        for name in [n for n in names if (a.dataset == n).any()]:
            sub = a[a.dataset == name]; nmax = sub.n.max(); s = sub[sub.n == nmax]
            panel = s.panel.iloc[0]
            # average over seeds, per (m,L)
            piv = s.groupby(["m","L"]).test_acc.mean().reset_index()
            best_row = piv.loc[piv.test_acc.idxmax()]
            best_m = int(best_row.m)
            mm = piv[piv.m == best_m]
            l1 = mm[mm.L == mm.L.min()].test_acc.values[0]
            best = mm.test_acc.max(); best_L = int(mm.loc[mm.test_acc.idxmax(), "L"])
            print(f"{name:12s} {panel:>8s} {int(nmax):7d} {best_m:6d} {l1:7.4f} "
                  f"{best:7.4f} {best-l1:+9.4f} {best_L:6d}")
        print("="*70)
    except Exception as e:
        print(f"(summary skipped: {e})")

if __name__ == "__main__":
    main()
