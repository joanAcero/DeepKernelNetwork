"""
exp_flat_baselines.py
=====================
Flat single-kernel SVM baselines for Experiment 3, run on the SAME datasets and the SAME
training-set sizes (n) as the width x depth study, so the deep/wide architecture can be
compared against flat references that hold the kernel fixed.

Two flat baselines per (dataset, n):
  1. ARC-COSINE flat RFF SVM  — ML_MSVMClassifier(num_layers=0): the input is mapped by the
     arc-cosine random-feature map and a single linear SVM (the head) is trained DIRECTLY on
     the P-dimensional features. No blocks, no W-projection, no stacking. This is the key
     control: it uses the SAME kernel as the proposed architecture, so any gain of the deep/
     wide model over it is attributable to the ARCHITECTURE, not to the kernel.
  2. RBF flat SVM             — exact SVC(kernel="rbf") for n<=10k; for larger n an RBF
     random-Fourier-feature linear SVM (ML_MSVMClassifier(num_layers=0, kernel="rbf")) so a
     shallow RBF reference exists at every n. This isolates the EFFECT OF THE KERNEL CHOICE
     at flat depth.

Metrics: accuracy, macro-F1, AUC (binary ROC-AUC; multiclass macro OvR). Schema and protocol
(P, final_C, seeds, n-grids, dataset loaders, regime feeding is irrelevant here since m=1)
match exp_width_depth_scenarios2 so rows merge into the same analysis.

Output APPENDS to results/exp_width_depth_scenarios2.csv by default (model strings make the
baselines unambiguous: 'flat_arccosine' and 'flat_rbf').
"""
from __future__ import annotations
import argparse, datetime, importlib.util, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from utils import Tee, CSVWriter, load, hms, banner

# reuse the exp-3 settings and the synthetic generators / loaders for commensurability
import exp_width_depth_scenarios2 as E
try:
    import exp_width_depth_complete as Cm   # for the multiclass synthetics + small loader
except Exception:
    Cm = None
try:
    from small_datasets import load_small
except Exception:
    load_small = None

P        = E.P
FINAL_C  = E.FINAL_C
N_SEEDS  = E.N_SEEDS
TEST_SIZE = E.TEST_SIZE
RBF_EXACT_MAX_N = 10_000          # exact RBF SVM only for n<=this; RFF-RBF above

def import_clf():
    # Construct the correct path relative to this script's location
    module_path = Path(__file__).parent / ".." / "ml_msvm" / "ml_msvm.py"
    
    spec = importlib.util.spec_from_file_location("ml_msvm", str(module_path.resolve()))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ml_msvm"] = mod
    spec.loader.exec_module(mod)
    return mod.ML_MSVMClassifier
ML = import_clf()

# datasets + n-grids: the union of the main run and the completion run
DATASETS = [("synthmc_blobs","SynthMC_blobs"),("synthmc_nonlinear","SynthMC_nonlinear"),
            ("digits8x8","Digits8x8"),("optdigits","Optdigits"),("satimage","Satimage"),
            ("mnist","MNIST"),("higgs","HIGGS"),("covertype","CoverType")]
N_GRID = dict(E.N_GRID)
N_GRID.setdefault("SynthMC_blobs",[1_000,3_000])
N_GRID.setdefault("SynthMC_nonlinear",[1_000,3_000])
N_GRID.setdefault("Digits8x8",[500,1_000,1_297])
N_GRID.setdefault("Optdigits",[500,2_000,3_823])
N_GRID.setdefault("Satimage",[500,2_000,4_435])

SMALL_TAGS = {"digits8x8","optdigits","satimage"}

def get_data(tag, nm):
    if nm == "SynthMC_blobs" and Cm is not None:    return Cm.make_synth_blobs(6_000, seed=0)
    if nm == "SynthMC_nonlinear" and Cm is not None: return Cm.make_synth_nonlinear_mc(6_000, seed=0)
    if tag in SMALL_TAGS:
        if load_small is None: raise RuntimeError("small_datasets loader unavailable")
        return load_small(tag)
    return load(tag)

def auc_safe(y, scores, classes):
    try:
        scores = np.asarray(scores)
        if len(classes) == 2:
            s = scores if scores.ndim == 1 else scores[:, 1]
            return float(roc_auc_score(y, s))
        Yb = label_binarize(y, classes=classes)
        return float(roc_auc_score(Yb, scores, average="macro", multi_class="ovr"))
    except Exception:
        return np.nan

def eval_model(model, Xte, yte, classes, has_decision=True):
    pred = model.predict(Xte)
    acc = float(np.mean(pred == yte))
    f1 = float(f1_score(yte, pred, average="macro"))
    auc = np.nan
    if has_decision:
        try: auc = auc_safe(yte, model.decision_function(Xte), classes)
        except Exception: auc = np.nan
    return acc, f1, auc

def run_cell(X, y, name, n_total, csv_w):
    d, K = X.shape[1], len(np.unique(y)); classes = np.unique(y); pool = len(y)
    test_size = min(TEST_SIZE, max(500, pool // 5))
    n_eff = min(n_total, pool - test_size)
    if n_eff < 500: return
    capped = " (capped)" if n_eff < n_total else ""
    print(f"\n  {'─'*82}")
    print(f"  {name}  n={n_eff:,}{capped}  (d={d}, K={K}, test={test_size:,})")
    print(f"  {'baseline':22s} {'tr_acc':>7s} {'te_acc':>7s} {'f1':>6s} {'auc':>6s} {'t/run':>7s}")
    print(f"  {'·'*82}")

    def one(make, label, kernel_tag, has_dec=True):
        tr_l, te_l, f1_l, auc_l, t_l = [], [], [], [], []
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(7000 + seed)
            idx = rng.permutation(pool)[: n_eff + test_size]
            Xs, ys = X[idx], y[idx]
            Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=test_size,
                                                  random_state=seed, stratify=ys)
            try:
                model = make(seed)
                t = time.perf_counter(); model.fit(Xtr, ytr); dt = time.perf_counter() - t
                tr = model.score(Xtr, ytr)
                acc, f1, auc = eval_model(model, Xte, yte, classes, has_dec)
                tr_l.append(tr); te_l.append(acc); f1_l.append(f1); auc_l.append(auc); t_l.append(dt)
                csv_w.write(dict(exp_id="exp_flat_baselines", dataset=name, n_total=n_eff,
                    n_train=len(Xtr), n_test=test_size, d=d, n_classes=K, model=label,
                    feeding="none", kernel=kernel_tag, L=0, m=1, P=P, seed=seed,
                    acc=round(acc,4), macro_f1=round(f1,4),
                    auc=(round(auc,4) if not np.isnan(auc) else np.nan),
                    train_acc=round(tr,4), mean_cos_sim=np.nan, time_s=round(dt,2)))
            except Exception as ex:
                print(f"  {label} seed={seed} FAILED: {ex}", flush=True)
        if te_l:
            print(f"  {label:22s} {np.mean(tr_l):>7.4f} {np.mean(te_l):>7.4f} "
                  f"{np.mean(f1_l):>6.3f} {np.nanmean(auc_l):>6.3f} {np.mean(t_l):>6.1f}s", flush=True)

    # 1) flat arc-cosine RFF SVM (num_layers=0) — the same-kernel control
    one(lambda s: Pipeline([("scaler", StandardScaler()),
            ("clf", ML(num_layers=0, svms_per_block=1, rff_features=P, kernel="arc_cosine",
                       arc_cosine_degree=1, final_C=FINAL_C, normalize_inter_layer=True,
                       random_state=s))]),
        "flat_arccosine", "arc_cosine")

    # 2) flat RBF: exact SVC for small n, RFF-RBF (num_layers=0) for large n
    if n_eff <= RBF_EXACT_MAX_N:
        one(lambda s: Pipeline([("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=s))]),
            "flat_rbf_exact", "rbf", has_dec=True)   
    else:
        one(lambda s: Pipeline([("scaler", StandardScaler()),
                ("clf", ML(num_layers=0, svms_per_block=1, rff_features=P, kernel="rbf",
                           final_C=FINAL_C, normalize_inter_layer=True, random_state=s))]),
            "flat_rbf_rff", "rbf")

def run(log_path, csv_path, datasets):
    tee = Tee(sys.stdout, log_path); sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Exp 3 — FLAT baselines (same-kernel arc-cosine control + RBF reference)",
               f"P={P} final_C={FINAL_C} seeds={N_SEEDS}",
               "flat_arccosine = ML_MSVM(num_layers=0): arc-cosine RFF + head, NO blocks/projection.",
               "flat_rbf = exact RBF SVM (n<=10k) or RBF-RFF (larger). Appends to the Exp-3 CSV.")
        t0 = time.perf_counter()
        for tag, nm in datasets:
            try:
                X, y = get_data(tag, nm)
                print(f"\n\n{'='*84}\n  DATASET: {nm}  (d={X.shape[1]}, K={len(np.unique(y))})\n{'='*84}")
                for n_total in N_GRID.get(nm, [10_000]):
                    run_cell(X, y, nm, n_total, csv_w)
            except Exception as ex:
                print(f"  [{nm}] FAILED: {ex}")
        print(f"\n  flat baselines complete. {hms(time.perf_counter()-t0)}")
        print("\n  READING: compare the deep/wide architecture (exp3 rows) against flat_arccosine")
        print("  at the SAME (dataset,n). A gain over flat_arccosine isolates the ARCHITECTURE's")
        print("  contribution (kernel held fixed); the flat_rbf row isolates the kernel choice.")
    finally:
        sys.stdout = tee._stream; tee.close(); csv_w.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--csv_dir", default="results")
    p.add_argument("--csv_name", default="exp_width_depth_scenarios2.csv")
    p.add_argument("--only", help="comma-separated subset (tags as in exp3/complete)")
    a = p.parse_args()
    ds = DATASETS
    if a.only:
        want = {k.strip().lower() for k in a.only.split(",")}
        ds = [(t, n) for (t, n) in DATASETS if t in want]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(a.log_dir, exist_ok=True); os.makedirs(a.csv_dir, exist_ok=True)
    run(os.path.join(a.log_dir, f"exp_flat_baselines_{ts}.txt"),
        os.path.join(a.csv_dir, a.csv_name), ds)
