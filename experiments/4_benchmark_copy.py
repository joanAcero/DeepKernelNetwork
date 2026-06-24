"""
exp4_benchmark.py — Experiment 4: Benchmark against Baseline Architectures.

Compares the proposed architecture against standard deep/kernel baselines:
  1. Linear SVM: Sanity check for linear separability.
  2. Gradient Boosting (HistGBM): State-of-the-art for tabular data.
  3. Shallow RFF SVM: Ablation check (flat P=1000 approximation vs deep).
  4. MLP: 3 layers, 200 neurons each.
  5. Exact RBF SVM: Run only if N <= 70,000 (otherwise skipped).
  6. Proposed Architecture: A SINGLE optimal architecture is found per dataset 
     via internal CV, then evaluated across all 3 seeds for final metrics.

Usage:
  python3 exp4_benchmark.py
"""

from __future__ import annotations
import sys, time, csv, argparse, warnings, datetime
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.datasets import fetch_openml

sys.path.insert(0, str(Path(__file__).parent))
from mlsvm_extensions import DiverseMLMSVM

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
L_VALS    = [1, 2, 3, 4, 5]              # Grid search space for L
M_VALS    = [20, 50, 70]                     # Grid search space for M
KERNELS   = ["arc_cosine", "rbf"]        # Grid search space for kernel
FEEDINGS  = ["disjoint", "disjoint_featpart"] # Grid search space for feeding strategy
DEGREES   = [0, 1, 2]                    # Arc-Cosine degree (n) search space

P_VAL     = 2000
N_SEEDS   = 5
RBF_MAX_N = 70_000                       # Exact RBF threshold
N_CAP     = 100_000                      # Max training size
TUNE_CAP  = 10_000                       # Max samples used for internal validation

# ----------------------------------------------------------------------
# Dataset loaders
# ----------------------------------------------------------------------
def _openml(data_id=None, name=None, version=None):
    if data_id is not None:
        d = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    else:
        v = version if version is not None else 1
        d = fetch_openml(name=name, version=v, as_frame=True, parser="auto")
        
    Xdf = pd.get_dummies(d.data, drop_first=False)
    X = np.nan_to_num(Xdf.to_numpy(dtype=float))
    y = LabelEncoder().fit_transform(d.target)
    return X, y

def make_spirals(n, turns=2.5, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.uniform(0, 1, (n, 1))) * turns * 2*np.pi
    a = np.hstack([-np.cos(t)*t, np.sin(t)*t]) + rng.standard_normal((n,2))*noise
    b = np.hstack([ np.cos(t)*t,-np.sin(t)*t]) + rng.standard_normal((n,2))*noise
    X = np.vstack([a, b])
    y = np.hstack([np.zeros(n), np.ones(n)]).astype(int)
    i = rng.permutation(len(y))
    return X[i], y[i]

DATASETS = { 
    "madelon":       lambda: _openml(data_id=1485),       
    "mnist":         lambda: _openml(name="mnist_784", version=1),
    "spirals":       lambda: make_spirals(60000)  
}

# ----------------------------------------------------------------------
# Metrics & Helpers
# ----------------------------------------------------------------------
def metrics(model, X, y, classes):
    yp = model.predict(X)
    acc = accuracy_score(y, yp)
    f1  = f1_score(y, yp, average="macro")
    try:
        s = model.predict_proba(X)
        if len(classes) == 2:
            s1 = s if s.ndim == 1 else s[:, 1]
            auc = roc_auc_score(y, s1)
        else:
            Yb = label_binarize(y, classes=classes)
            auc = roc_auc_score(Yb, s, average="macro", multi_class="ovr")
    except Exception:
        try:
            s = model.decision_function(X)
            if len(classes) == 2:
                auc = roc_auc_score(y, s)
            else:
                Yb = label_binarize(y, classes=classes)
                auc = roc_auc_score(Yb, s, average="macro", multi_class="ovr")
        except:
            auc = np.nan
    return acc, f1, auc

def timed_fit(estimator, Xtr, ytr):
    t0 = time.perf_counter()
    estimator.fit(Xtr, ytr)
    return estimator, time.perf_counter() - t0

# ----------------------------------------------------------------------
# Main Benchmarking Loop
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    names = args.only if args.only else list(DATASETS)

    print("=" * 80)
    print("PRE-FLIGHT CHECK: Loading datasets to ensure no overnight failures...")
    loaded_datasets = {}
    for name in names:
        try:
            print(f"  -> Fetching {name}...")
            loaded_datasets[name] = DATASETS[name]()
            d_shape = loaded_datasets[name][0].shape
            print(f"     [OK] {name} loaded successfully (Shape: {d_shape})")
        except Exception as e:
            print(f"\n[FATAL ERROR] Failed to load dataset '{name}': {e}")
            sys.exit(1)
            
    print("\nPre-flight check passed! All datasets are safely loaded in memory.")
    print("=" * 80 + "\n")

    import os
    csv_path = "results_exp4_2.csv"
    HEADER = ["dataset","d","n","seed","method",
              "train_acc","train_f1","train_auc","test_acc","test_f1","test_auc",
              "fit_time_s","timestamp"]
    
    done = set()
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            prev = pd.read_csv(csv_path)
            for _, r in prev.iterrows():
                done.add((str(r["dataset"]), int(r["n"]), int(r["seed"]), str(r["method"])))
            print(f"Resuming: {len(done)} completed cells found in {csv_path}")
        except Exception:
            pass
            
    new_file = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    out = open(csv_path, "a", newline=""); cw = csv.writer(out)
    if new_file:
        cw.writerow(HEADER)

    def already(ds, n, seed, method):
        return (str(ds), int(n), int(seed), str(method)) in done

    def emit(ds, d, n, seed, method, tr, te, t):
        cw.writerow([ds, d, n, seed, method,
                     round(tr[0],4) if not np.isnan(tr[0]) else "-",
                     round(tr[1],4) if not np.isnan(tr[1]) else "-",
                     round(tr[2],4) if not np.isnan(tr[2]) else "-",
                     round(te[0],4) if not np.isnan(te[0]) else "-",
                     round(te[1],4) if not np.isnan(te[1]) else "-",
                     round(te[2],4) if not np.isnan(te[2]) else "-",
                     round(t,3) if not np.isnan(t) else "-",
                     datetime.datetime.now().isoformat()])
        out.flush()

    print(f"EXPERIMENT 4 — Benchmark")
    print(f"{'dataset':13s} {'n':>7s} {'sd':>2s} | {'method':>11s} | "
          f"{'te_acc':>7s} {'te_f1':>7s} {'te_auc':>7s} {'time_s':>8s}")
    print("-" * 85)

    for name in names:
        X, y = loaded_datasets[name]
        d, pool = X.shape[1], len(y)
        classes = np.unique(y)
        
        ts = max(int(pool * 0.10), 200)
        n_eff = min(pool - ts, N_CAP)

        # =================================================================
        # PHASE 1: STANDARD BASELINES (Evaluated per seed)
        # =================================================================
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(7000 + seed)
            idx = rng.permutation(pool)[:n_eff + ts]
            try:
                Xtr, Xte, ytr, yte = train_test_split(X[idx], y[idx], test_size=ts,
                                                      random_state=seed, stratify=y[idx])
            except ValueError:
                Xtr, Xte, ytr, yte = train_test_split(X[idx], y[idx], test_size=ts,
                                                      random_state=seed)

            # ---- Model 1: Linear SVM ----
            if not already(name, n_eff, seed, "linear_svm"):
                lin = Pipeline([("sc", StandardScaler()),
                                ("clf", LinearSVC(dual=False, max_iter=2000, random_state=seed))])
                try:
                    lin, t_lin = timed_fit(lin, Xtr, ytr)
                    emit(name, d, n_eff, seed, "linear_svm", metrics(lin, Xtr, ytr, classes), metrics(lin, Xte, yte, classes), t_lin)
                    te = metrics(lin, Xte, yte, classes)
                    f1auc = f"{te[1]:7.4f} {te[2]:7.4f}" if not np.isnan(te[2]) else f"{te[1]:7.4f}    --  "
                    print(f"{name:13s} {n_eff:7d} {seed:2d} | {'linear_svm':>11s} | {te[0]:7.4f} {f1auc} {t_lin:8.2f}")
                except Exception as e:
                    print(f"{name:13s} linear_svm FAILED: {str(e)[:35]}")

            # ---- Model 2: Gradient Boosting ----
            if not already(name, n_eff, seed, "grad_boost"):
                gb = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
                try:
                    gb, t_gb = timed_fit(gb, Xtr, ytr)
                    emit(name, d, n_eff, seed, "grad_boost", metrics(gb, Xtr, ytr, classes), metrics(gb, Xte, yte, classes), t_gb)
                    te = metrics(gb, Xte, yte, classes)
                    f1auc = f"{te[1]:7.4f} {te[2]:7.4f}" if not np.isnan(te[2]) else f"{te[1]:7.4f}    --  "
                    print(f"{name:13s} {n_eff:7d} {seed:2d} | {'grad_boost':>11s} | {te[0]:7.4f} {f1auc} {t_gb:8.2f}")
                except Exception as e:
                    print(f"{name:13s} grad_boost FAILED: {str(e)[:35]}")

            # ---- Model 3: Shallow RFF SVM ----
            if not already(name, n_eff, seed, "shallow_rff"):
                sc = StandardScaler().fit(Xtr)
                var = sc.transform(Xtr).var()
                g = 1.0 / (d * var) if var > 0 else 1.0
                rff = Pipeline([("sc", StandardScaler()),
                                ("rff", RBFSampler(gamma=g, n_components=P_VAL, random_state=seed)),
                                ("clf", SGDClassifier(loss="hinge", alpha=1e-4, max_iter=1000, random_state=seed, n_jobs=-1))])
                try:
                    rff, t_rff = timed_fit(rff, Xtr, ytr)
                    class _RFF_Wrap:
                        def __init__(self, model): self.model = model
                        def predict(self, X): return self.model.predict(X)
                        def decision_function(self, X): return self.model.decision_function(X)
                    emit(name, d, n_eff, seed, "shallow_rff", metrics(_RFF_Wrap(rff), Xtr, ytr, classes), metrics(_RFF_Wrap(rff), Xte, yte, classes), t_rff)
                    te = metrics(_RFF_Wrap(rff), Xte, yte, classes)
                    f1auc = f"{te[1]:7.4f} {te[2]:7.4f}" if not np.isnan(te[2]) else f"{te[1]:7.4f}    --  "
                    print(f"{name:13s} {n_eff:7d} {seed:2d} | {'shallow_rff':>11s} | {te[0]:7.4f} {f1auc} {t_rff:8.2f}")
                except Exception as e:
                    print(f"{name:13s} shallow_rff FAILED: {str(e)[:35]}")

            # ---- Model 4: MLP ----
            if not already(name, n_eff, seed, "mlp"):
                mlp = Pipeline([("sc", StandardScaler()),
                                ("clf", MLPClassifier(hidden_layer_sizes=(200, 200, 200),
                                                      max_iter=500, random_state=seed, early_stopping=True))])
                try:
                    mlp, t_mlp = timed_fit(mlp, Xtr, ytr)
                    emit(name, d, n_eff, seed, "mlp", metrics(mlp, Xtr, ytr, classes), metrics(mlp, Xte, yte, classes), t_mlp)
                    te = metrics(mlp, Xte, yte, classes)
                    f1auc = f"{te[1]:7.4f} {te[2]:7.4f}" if not np.isnan(te[2]) else f"{te[1]:7.4f}    --  "
                    print(f"{name:13s} {n_eff:7d} {seed:2d} | {'mlp':>11s} | {te[0]:7.4f} {f1auc} {t_mlp:8.2f}")
                except Exception as e:
                    print(f"{name:13s} mlp FAILED: {str(e)[:35]}")

            # ---- Model 5: Exact RBF ----
            if not already(name, n_eff, seed, "rbf_exact"):
                if n_eff <= RBF_MAX_N:
                    rbf = Pipeline([("sc", StandardScaler()),
                                    ("clf", SVC(kernel="rbf", C=10, gamma="scale"))])
                    try:
                        rbf, t_rbf = timed_fit(rbf, Xtr, ytr)
                        emit(name, d, n_eff, seed, "rbf_exact", metrics(rbf, Xtr, ytr, classes), metrics(rbf, Xte, yte, classes), t_rbf)
                        te = metrics(rbf, Xte, yte, classes)
                        f1auc = f"{te[1]:7.4f} {te[2]:7.4f}" if not np.isnan(te[2]) else f"{te[1]:7.4f}    --  "
                        print(f"{name:13s} {n_eff:7d} {seed:2d} | {'rbf_exact':>11s} | {te[0]:7.4f} {f1auc} {t_rbf:8.2f}")
                    except Exception as e:
                        print(f"{name:13s} rbf_exact FAILED: {str(e)[:35]}")
                else:
                    emit(name, d, n_eff, seed, "rbf_exact", [np.nan]*3, [np.nan]*3, np.nan)
                    print(f"{name:13s} {n_eff:7d} {seed:2d} | {'rbf_exact':>11s} | - (+70k samples)")

        # =================================================================
        # PHASE 2: PROPOSED ARCHITECTURE (Tune once, eval across 3 seeds)
        # =================================================================
        # Only tune if we are missing at least one seed calculation
        if not all(already(name, n_eff, s, "arch_proposed") for s in range(N_SEEDS)):
            
            print(f"      [Finding SINGLE best architecture for {name} to use across all seeds...]")
            
            # Create a fixed tuning pool using the first seed's split logic so we don't leak test data
            rng_tune = np.random.RandomState(7000)
            idx_tune = rng_tune.permutation(pool)[:n_eff + ts]
            try:
                Xtr_pool, _, ytr_pool, _ = train_test_split(X[idx_tune], y[idx_tune], test_size=ts, 
                                                            random_state=0, stratify=y[idx_tune])
            except ValueError:
                Xtr_pool, _, ytr_pool, _ = train_test_split(X[idx_tune], y[idx_tune], test_size=ts, 
                                                            random_state=0)
            
            # Cap the tuning pool at TUNE_CAP to keep search fast
            tune_n = min(len(Xtr_pool), TUNE_CAP)
            rng_sub = np.random.RandomState(9999)
            sub_idx = rng_sub.permutation(len(Xtr_pool))[:tune_n]
            X_tune, y_tune = Xtr_pool[sub_idx], ytr_pool[sub_idx]

            best_val_acc = -1
            best_config = (L_VALS[0], M_VALS[0], KERNELS[0], FEEDINGS[0], DEGREES[1])
            
            for L_try in L_VALS:
                for m_try in M_VALS:
                    for k_try in KERNELS:
                        for feed_try in FEEDINGS:
                            degs_to_try = DEGREES if k_try == "arc_cosine" else [1]
                            for deg_try in degs_to_try:
                                
                                # Evaluate this specific config across 3 internal CV splits
                                val_accs = []
                                for cv_seed in range(3):
                                    try:
                                        Xt_sub, Xv_sub, yt_sub, yv_sub = train_test_split(
                                            X_tune, y_tune, test_size=0.2, 
                                            random_state=5000 + cv_seed, stratify=y_tune
                                        )
                                    except ValueError:
                                        Xt_sub, Xv_sub, yt_sub, yv_sub = train_test_split(
                                            X_tune, y_tune, test_size=0.2, 
                                            random_state=5000 + cv_seed
                                        )
                                    
                                    try:
                                        tune_model = Pipeline([("sc", StandardScaler()),
                                                         ("clf", DiverseMLMSVM(num_layers=L_try, svms_per_block=m_try, 
                                                                               rff_features=P_VAL, kernel=k_try, 
                                                                               arc_cosine_degree=deg_try, diversity_mode=feed_try,
                                                                               block_C=10.0, final_C=1.0, random_state=0,
                                                                               normalize_inter_layer=True))])
                                        tune_model.fit(Xt_sub, yt_sub)
                                        val_accs.append(accuracy_score(yv_sub, tune_model.predict(Xv_sub)))
                                    except Exception:
                                        pass
                                        
                                if len(val_accs) > 0:
                                    mean_val_acc = np.mean(val_accs)
                                    if mean_val_acc > best_val_acc:
                                        best_val_acc = mean_val_acc
                                        best_config = (L_try, m_try, k_try, feed_try, deg_try)

            best_L, best_m, best_k, best_feed, best_deg = best_config
            k_disp = f"arc(n={best_deg})" if best_k == "arc_cosine" else "rbf"
            print(f"      [Global Best Config Locked: L={best_L}, m={best_m}, {k_disp}, {best_feed} (Mean CV Acc: {best_val_acc:.4f})]")
            
            # ---- Evaluate the SINGLE BEST architecture across the 3 main experiment seeds ----
            for seed in range(N_SEEDS):
                if not already(name, n_eff, seed, "arch_proposed"):
                    
                    # Re-create the exact train/test split for this outer seed
                    rng = np.random.RandomState(7000 + seed)
                    idx = rng.permutation(pool)[:n_eff + ts]
                    try:
                        Xtr, Xte, ytr, yte = train_test_split(X[idx], y[idx], test_size=ts,
                                                              random_state=seed, stratify=y[idx])
                    except ValueError:
                        Xtr, Xte, ytr, yte = train_test_split(X[idx], y[idx], test_size=ts,
                                                              random_state=seed)
                        
                    arch = Pipeline([("sc", StandardScaler()),
                                     ("clf", DiverseMLMSVM(num_layers=best_L, svms_per_block=best_m, 
                                                           rff_features=P_VAL, kernel=best_k, 
                                                           arc_cosine_degree=best_deg, diversity_mode=best_feed,
                                                           block_C=10.0, final_C=1.0, random_state=seed,
                                                           normalize_inter_layer=True))])
                    try:
                        arch, t_arch = timed_fit(arch, Xtr, ytr)
                        emit(name, d, n_eff, seed, "arch_proposed", metrics(arch, Xtr, ytr, classes), metrics(arch, Xte, yte, classes), t_arch)
                        te = metrics(arch, Xte, yte, classes)
                        f1auc = f"{te[1]:7.4f} {te[2]:7.4f}" if not np.isnan(te[2]) else f"{te[1]:7.4f}    --  "
                        print(f"{name:13s} {n_eff:7d} {seed:2d} | {'arch_prop':>11s} | {te[0]:7.4f} {f1auc} {t_arch:8.2f} (L={best_L},m={best_m},{k_disp})")
                    except Exception as e:
                        print(f"{name:13s} arch FAILED: {str(e)[:35]}")

    out.close()
    
    # ---- Print Summary Averages ----
    try:
        print("\n" + "="*85)
        print("EXPERIMENT 4 SUMMARY: Mean Test Accuracy & Time across seeds")
        print("="*85)
        df = pd.read_csv("results_exp4.csv")
        df = df.replace("-", np.nan).apply(pd.to_numeric, errors='ignore')
        
        summary = df.groupby(["dataset", "method"])[["test_acc", "fit_time_s"]].mean().unstack("method")
        print(summary.to_string(na_rep="- (+70k samples)", float_format="%.4f"))
        print("="*85)
    except Exception:
        pass

if __name__ == "__main__":
    main()