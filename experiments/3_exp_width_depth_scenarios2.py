"""
exp_width_depth_scenarios2.py
=============================
Experiment 3 — impact of WIDTH (m) and DEPTH (L), and their interaction, across regimes.

Open questions (conclusions to be drawn ONLY from the full results, not assumed here):
  Q1 (width):  does m>1 help over m=1, and does any benefit vary with n (small/medium/large)?
  Q2 (depth):  does L>1 change accuracy/F1/AUC, and does the effect differ across datasets?
  Q3 (interaction): does the effect of depth depend on m (and vice versa)?
  Q4 (compositional probe): on a synthetic target with known compositional structure, can the
       architecture recover it, and does depth change the outcome? Both outcomes are informative
       and neither is presupposed.

Design note (the only thing fixed in advance, and it concerns experimental validity, NOT the
result): the synthetic target is chosen to be learnable-in-principle and to have a non-trivial
margin, so that it can produce an interpretable signal either way. A pure high-order parity
target was rejected during piloting because it is unlearnable at any depth and would yield no
signal to interpret — that is a statement about the probe's usefulness, not about the
architecture's behaviour.

Feeding strategy is regime-dependent, taken from Experiment 1:
  high-d (d>=60) -> B_in_dm (split input columns);  low-d -> C (split data).

Metrics: accuracy, macro-F1, and AUC (binary: ROC-AUC; multiclass: macro OvR AUC).

Output: results/exp_width_depth_scenarios2.csv (per-seed), logs/..._<ts>.txt
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import Tee, CSVWriter, load, hms, banner
from mlsvm_extensions import DiverseMLMSVM

EXP_ID   = "exp_width_depth_scenarios2"
P        = 1000
BLOCK_C  = 10.0
FINAL_C  = 1.0
KERNEL   = "arc_cosine"
N_SEEDS  = 3
M_VALUES = [1, 5, 10, 20]
L_VALUES = [1, 2, 3]
MIN_PER_SVM = 200
TEST_SIZE   = 20_000
HIGH_D_THRESHOLD = 60

def feeding_for(d):
    if d >= HIGH_D_THRESHOLD:
        return "input_subspace_dm", True, "B_in_dm"
    return "disjoint", False, "C_splitdata"

# ---- synthetic compositional probe: 2-stage smooth composition -----------------------
def make_synth_compositional(n, d_noise=20, seed=0):
    """Compositional target: the label is an XOR of two smooth non-linear stage-1 features
    (a sinusoidal projection and a quadratic), embedded among noise dimensions. It has genuine
    two-stage compositional structure and a non-trivial margin, so it serves as a probe of
    whether the architecture (and depth) can recover such structure. The outcome is left open."""
    rng = np.random.default_rng(seed)
    Xi = rng.standard_normal((n, 4))
    g1 = np.sin(2.0 * Xi[:, 0]) + 0.5 * Xi[:, 1]
    g2 = Xi[:, 2] ** 2 - Xi[:, 3] ** 2
    y = ((g1 > 0) ^ (g2 > 0)).astype(int)
    Xn = rng.standard_normal((n, d_noise))
    X = np.hstack([Xi, Xn]); X = X[:, rng.permutation(X.shape[1])]
    return X.astype(np.float64), y.astype(int)

SCENARIO = {
    "SynthComp": ("probe",   "small"),   # compositional probe (outcome open)
    "Digits8x8": ("easy",    "small"),
    "Optdigits": ("easy",    "small"),
    "Satimage":  ("medium",  "small"),
    "MNIST":     ("medium",  "medium"),  # high-d
    "HIGGS":     ("hard",    "large"),
    "CoverType": ("hard",    "large"),
}
N_GRID = {
    "SynthComp": [1_000, 3_000],
    "Digits8x8": [500, 1_000, 1_297],
    "Optdigits": [500, 2_000, 3_823],
    "Satimage":  [500, 2_000, 4_435],
    "MNIST":     [2_000, 10_000, 50_000],
    "HIGGS":     [2_000, 10_000, 50_000],
    "CoverType": [2_000, 10_000, 50_000],
}

def make_model(m, L, d, seed):
    mode, full_data, _ = feeding_for(d)
    clf = DiverseMLMSVM(
        num_layers=L, svms_per_block=m, rff_features=P, kernel=KERNEL, arc_cosine_degree=1,
        diversity_mode=mode, block_C=BLOCK_C, final_C=FINAL_C, random_state=seed,
        normalize_inter_layer=True, input_subspace_full_data=full_data,
        input_subspace_full_P=True)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

def metrics(model, X, y, classes):
    pred = model.predict(X)
    acc = float(np.mean(pred == y))
    f1 = float(f1_score(y, pred, average="macro"))
    try:
        sc = np.asarray(model.decision_function(X))
        if len(classes) == 2:
            s = sc if sc.ndim == 1 else sc[:, 1]
            auc = float(roc_auc_score(y, s))
        else:
            Yb = label_binarize(y, classes=classes)
            auc = float(roc_auc_score(Yb, sc, average="macro", multi_class="ovr"))
    except Exception:
        auc = np.nan
    return acc, f1, auc

def shallow_rbf_ref(Xtr, ytr, Xte, yte, classes, n_eff, seed):
    if n_eff > 10_000:
        return None
    svc = Pipeline([("s", StandardScaler()),
                    ("c", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed))])
    svc.fit(Xtr, ytr)
    pred = svc.predict(Xte)
    return float(np.mean(pred == yte)), float(f1_score(yte, pred, average="macro"))

def get_data(tag, nm):
    if nm == "SynthComp":
        return make_synth_compositional(6000, seed=0)
    return load(tag)

def run_cell(X, y, name, n_total, csv_w):
    d, K = X.shape[1], len(np.unique(y))
    classes = np.unique(y)
    pool = len(y)
    test_size = min(TEST_SIZE, max(500, pool // 5))
    n_eff = min(n_total, pool - test_size)
    if n_eff < 500:
        return
    mode, full_data, feed_label = feeding_for(d)
    capped = " (capped)" if n_eff < n_total else ""
    cx, sc = SCENARIO.get(name, ("?", "?"))
    print(f"\n  {'─'*96}")
    print(f"  {name}  n={n_eff:,}{capped}  (d={d}, K={K}, test={test_size:,})  "
          f"scenario=[{cx}/{sc}]  feeding={feed_label}")
    print(f"  {'m':>3s} {'L':>2s} {'n/SVM':>7s} {'tr_acc':>7s} {'te_acc':>7s} "
          f"{'f1':>6s} {'auc':>6s} {'cos':>6s} {'t/run':>7s}")
    print(f"  {'·'*96}")

    ref=[]
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(7000 + seed)
        idx = rng.permutation(pool)[: n_eff + test_size]
        Xs, ys = X[idx], y[idx]
        Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=test_size,
                                              random_state=seed, stratify=ys)
        r = shallow_rbf_ref(Xtr, ytr, Xte, yte, classes, n_eff, seed)
        if r is not None:
            ref.append(r)
            csv_w.write(dict(exp_id=EXP_ID, dataset=name, n_total=n_eff, n_train=len(Xtr),
                n_test=test_size, d=d, n_classes=K, model="RBF_SVM_shallow_ref",
                feeding=feed_label, kernel="rbf", L=0, m=0, P=0, seed=seed,
                acc=round(r[0],4), macro_f1=round(r[1],4), auc=np.nan,
                train_acc=np.nan, mean_cos_sim=np.nan, time_s=np.nan))
    if ref:
        print(f"  RBF shallow ref: acc={np.mean([a for a,_ in ref]):.4f} "
              f"f1={np.mean([f for _,f in ref]):.4f}", flush=True)

    for m in M_VALUES:
        if mode == "input_subspace_dm" and m > d:
            print(f"  m={m:>2d}  SKIP (m>d={d})", flush=True); continue
        if mode == "disjoint" and m > 1 and (n_eff // m) < MIN_PER_SVM:
            print(f"  m={m:>2d}  SKIP (n/SVM={n_eff//m}<{MIN_PER_SVM})", flush=True); continue
        for L in L_VALUES:
            acc_l, f1_l, auc_l, tr_l, cs_l, t_l = [], [], [], [], [], []
            for seed in range(N_SEEDS):
                rng = np.random.RandomState(7000 + seed)
                idx = rng.permutation(pool)[: n_eff + test_size]
                Xs, ys = X[idx], y[idx]
                Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=test_size,
                                                      random_state=seed, stratify=ys)
                try:
                    model = make_model(m, L, d, seed)
                    t = time.perf_counter(); model.fit(Xtr, ytr); dt = time.perf_counter()-t
                    tr = model.score(Xtr, ytr)
                    acc, f1, auc = metrics(model, Xte, yte, classes)
                    clf = model.named_steps["clf"]
                    diag = clf.W_diagnostics_[0] if getattr(clf,"W_diagnostics_",None) else {}
                    acc_l.append(acc); f1_l.append(f1); auc_l.append(auc)
                    tr_l.append(tr); t_l.append(dt); cs_l.append(diag.get("mean_cos_sim",np.nan))
                    csv_w.write(dict(exp_id=EXP_ID, dataset=name, n_total=n_eff,
                        n_train=len(Xtr), n_test=test_size, d=d, n_classes=K,
                        model=f"{feed_label}_m{m}_L{L}", feeding=feed_label, kernel=KERNEL,
                        L=L, m=m, P=P, seed=seed, train_acc=round(tr,4), acc=round(acc,4),
                        macro_f1=round(f1,4), auc=(round(auc,4) if not np.isnan(auc) else np.nan),
                        mean_cos_sim=diag.get("mean_cos_sim"), time_s=round(dt,2)))
                except Exception as e:
                    print(f"  m={m} L={L} seed={seed} FAILED: {e}", flush=True)
            if acc_l:
                print(f"  {m:>3d} {L:>2d} {len(Xtr)//max(m,1):>7d} "
                      f"{np.mean(tr_l):>7.4f} {np.mean(acc_l):>7.4f} {np.mean(f1_l):>6.3f} "
                      f"{np.nanmean(auc_l):>6.3f} {np.nanmean(cs_l):>6.3f} "
                      f"{np.mean(t_l):>6.1f}s", flush=True)
        print()

def run(log_path, csv_path, datasets):
    tee = Tee(sys.stdout, log_path); sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Exp 3 — width (m) x depth (L) across regimes  [acc / macro-F1 / AUC]",
               f"P={P} block_C={BLOCK_C} final_C={FINAL_C} seeds={N_SEEDS}  m={M_VALUES} L={L_VALUES}",
               f"feeding=regime-dependent (B_in_dm if d>={HIGH_D_THRESHOLD} else C)",
               "SynthComp = compositional probe (outcome open). RBF exact SVM = shallow reference (n<=10k).")
        t0 = time.perf_counter()
        for tag, nm in datasets:
            try:
                X, y = get_data(tag, nm)
                print(f"\n\n{'='*98}\n  DATASET: {nm}  (d={X.shape[1]}, K={len(np.unique(y))})\n{'='*98}")
                for n_total in N_GRID.get(nm, [10_000]):
                    run_cell(X, y, nm, n_total, csv_w)
            except Exception as e:
                print(f"  [{nm}] FAILED: {e}")
        print(f"\n  {EXP_ID} complete. {hms(time.perf_counter()-t0)}")
        print("\n  READING:")
        print("  - Q1 width: best m>1 vs m=1 per (dataset,n); is the gain larger at small/medium n?")
        print("  - Q2 depth: per dataset, does acc/f1/auc rise with L? Where does it, where not?")
        print("  - Q3 interaction: pivot acc by (m,L); does the L-gain grow with m?")
        print("  - Q4 probe: on SynthComp, what accuracy is reached, and does it change with L?")
    finally:
        sys.stdout = tee._stream; tee.close(); csv_w.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--csv_dir", default="results")
    p.add_argument("--only", help="comma-separated subset of: synthcomp,digits8x8,optdigits,satimage,mnist,higgs,covertype")
    a = p.parse_args()
    tagmap = {"synthcomp":("synthcomp","SynthComp"),"digits8x8":("digits8x8","Digits8x8"),
              "optdigits":("optdigits","Optdigits"),"satimage":("satimage","Satimage"),
              "mnist":("mnist","MNIST"),"higgs":("higgs","HIGGS"),"covertype":("covertype","CoverType")}
    order = ["synthcomp","digits8x8","optdigits","satimage","mnist","higgs","covertype"]
    if a.only:
        order = [k.strip().lower() for k in a.only.split(",")]
    datasets = [tagmap[k] for k in order if k in tagmap]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(a.log_dir, exist_ok=True); os.makedirs(a.csv_dir, exist_ok=True)
    run(os.path.join(a.log_dir, f"{EXP_ID}_{ts}.txt"),
        os.path.join(a.csv_dir, f"{EXP_ID}.csv"), datasets)
