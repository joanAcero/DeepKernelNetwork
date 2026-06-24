"""
probe_headroom_baselines_5L_real.py 

Tests the Headroom Hypothesis tracking BOTH exact RBF and MLP baselines.
Adapted to test depths up to L=5.
Extended with real-world UCI datasets known for deep/shallow performance gaps.

Usage:
  python3 probe_headroom_baselines_5L_real.py
  python3 probe_headroom_baselines_5L_real.py --gap-only
"""
from __future__ import annotations
import sys, time, csv, datetime, warnings, argparse
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mlsvm_extensions import DiverseMLMSVM

M = 50
L_VALUES = [1, 2, 3, 4, 5]
N_SEEDS = 3
P = 1000
TEST_SIZE = 10_000
RBF_MAX_N = 15_000

N_GRID = [1000, 5000, 10000]

# --- 1. Synthetic Generators ---
def make_clean_checkerboard(n_samples):
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 4, size=(n_samples, 2))
    coords = np.floor(X).astype(int)
    y = (np.sum(coords, axis=1) % 2).astype(int)
    return X, y

def make_spirals(n_samples, turns=2.5, noise=0.05):
    rng = np.random.default_rng(42)
    n = np.sqrt(rng.uniform(0, 1, size=(n_samples, 1))) * turns * (2 * np.pi)
    d1x = -np.cos(n) * n + rng.standard_normal((n_samples, 1)) * noise
    d1y = np.sin(n) * n + rng.standard_normal((n_samples, 1)) * noise
    X0 = np.hstack([d1x, d1y])
    X1 = np.hstack([-d1x, -d1y])
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)]).astype(int)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

def make_multiclass_spirals(n_samples, n_classes=3, turns=2.0, noise=0.05):
    rng = np.random.default_rng(42)
    samples_per_class = n_samples // n_classes
    X, y = [], []
    for i in range(n_classes):
        n = np.sqrt(rng.uniform(0, 1, size=(samples_per_class, 1))) * turns * (2 * np.pi)
        offset = i * (2 * np.pi / n_classes)
        d1x = np.cos(n + offset) * n + rng.standard_normal((samples_per_class, 1)) * noise
        d1y = np.sin(n + offset) * n + rng.standard_normal((samples_per_class, 1)) * noise
        X.append(np.hstack([d1x, d1y]))
        y.append(np.full(samples_per_class, i))
    X = np.vstack(X)
    y = np.concatenate(y)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

def make_multiclass_checkerboard(n_samples, grid_size=3):
    rng = np.random.default_rng(42)
    X = rng.uniform(0, grid_size, size=(n_samples, 2))
    coords = np.floor(X).astype(int)
    y = (np.sum(coords, axis=1) % grid_size).astype(int)
    return X, y

# --- 2. Real Dataset Fetcher ---
def fetch_real_data(name):
    # Fetch as a pandas DataFrame to handle data types properly
    data = fetch_openml(name=name, version=1, as_frame=True, parser='auto')
    X_df = data.data
    
    # Automatically one-hot encode string/categorical/boolean columns
    X_df = pd.get_dummies(X_df, drop_first=False)
    
    # Convert to pure float numpy array and catch any missing values
    X = np.nan_to_num(X_df.to_numpy(dtype=float))
    
    # Map target labels to integers
    y = LabelEncoder().fit_transform(data.target)
    
    return X, y

DATASETS = {
    # Synthetics
    "Check_2D_Bin": lambda n: make_clean_checkerboard(n),
    "Spirals_2D_B": lambda n: make_spirals(n // 2),
    "Spirals_3C":   lambda n: make_multiclass_spirals(n),
    "Check_3C":     lambda n: make_multiclass_checkerboard(n),
}

def get_baselines(Xtr, ytr, Xte, yte, n):
    t0 = time.perf_counter()
    mlp = MLPClassifier(hidden_layer_sizes=(200, 200, 200), max_iter=1000,
                        early_stopping=False, random_state=0).fit(Xtr, ytr)
    mlp_acc = accuracy_score(yte, mlp.predict(Xte))
    t_mlp = time.perf_counter() - t0
    
    rbf_acc, t_rbf = np.nan, np.nan
    if n <= RBF_MAX_N:
        try:
            t0 = time.perf_counter()
            rbf = SVC(kernel="rbf", C=10, gamma="scale").fit(Xtr, ytr)
            rbf_acc = accuracy_score(yte, rbf.predict(Xte))
            t_rbf = time.perf_counter() - t0
        except Exception:
            pass
            
    return (mlp_acc, t_mlp), (rbf_acc, t_rbf)

def fmt(acc, t):
    if np.isnan(acc): return "      NaN     "
    return f"{acc:.3f}({t:>5.1f}s)"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap-only", action="store_true", help="Only run baselines to check gap")
    args = parser.parse_args()

    out = open("results_headroom_real.csv", "w", newline="")
    cw = csv.writer(out)
    
    if args.gap_only:
        cw.writerow(["dataset","n","seed","mlp_acc","mlp_time","rbf_acc","rbf_time","gap","timestamp"])
        print(f"GAP CHECKER PROBE: Tracking RBF vs MLP (No Architecture Execution)")
        print(f"{'dataset':15s} {'n':>6s} | {'MLP':>14s} | {'RBF':>14s} | {'Gap%':>7s}")
        print("-" * 65)
    else:
        cw.writerow(["dataset","n","seed","mlp_acc","mlp_time","rbf_acc","rbf_time",
                     "accL1","tL1","accL2","tL2","accL3","tL3","accL4","tL4","accL5","tL5",
                     "gain_vs_rbf_pct","timestamp"])
        print(f"HEADROOM PROBE: Tracking RBF vs MLP with Times (m={M})")
        print(f"{'dataset':15s} {'n':>6s} | {'MLP':>14s} | {'RBF':>14s} | {'L1':>14s} | {'L2':>14s} | {'L3':>14s} | {'L4':>14s} | {'L5':>14s} | {'gain%':>7s}")
        print("-" * 135)
    
    for ds_name, data_func in DATASETS.items():
        # Pre-fetch or generate max required
        max_req = max(N_GRID) + TEST_SIZE
        try:
            X_full, y_full = data_func(max_req)
        except Exception as e:
            print(f"Failed to load {ds_name}: {e}")
            continue
            
        d = X_full.shape[1]
        pool = len(y_full)
        
        # Determine valid grid points dynamically for small datasets
        valid_ns = []
        for n in N_GRID:
            ts = min(TEST_SIZE, max(int(pool * 0.2), 100)) # Ensure at least 20% test data
            if n <= pool - ts:
                valid_ns.append((n, ts))
        if not valid_ns: # Dataset is tiny (like Zoo)
            ts = int(pool * 0.2)
            valid_ns.append((pool - ts, ts))
        
        for n, ts in valid_ns:
            gains, mlp_accs, mlp_times, rbf_accs, rbf_times = [], [], [], [], []
            l_accs = {1: [], 2: [], 3: [], 4: [], 5: []}
            l_times = {1: [], 2: [], 3: [], 4: [], 5: []}
            
            for seed in range(N_SEEDS):
                rng = np.random.RandomState(7000 + seed)
                idx = rng.permutation(pool)[:n + ts]
                X, y = X_full[idx], y_full[idx]
                
                # Use stratification if possible, fallback to standard split
                try:
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, random_state=seed, stratify=y)
                except ValueError:
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, random_state=seed)
                
                sc = StandardScaler().fit(Xtr)
                Xtr_sc, Xte_sc = sc.transform(Xtr), sc.transform(Xte)
                
                (m_acc, m_t), (r_acc, r_t) = get_baselines(Xtr_sc, ytr, Xte_sc, yte, n)
                
                mlp_accs.append(m_acc); mlp_times.append(m_t)
                rbf_accs.append(r_acc); rbf_times.append(r_t)
                
                if args.gap_only:
                    gap = (m_acc - r_acc) * 100 if not np.isnan(r_acc) else np.nan
                    gains.append(gap)
                    cw.writerow([ds_name, n, seed, round(m_acc,4), round(m_t,2), round(r_acc,4) if not np.isnan(r_acc) else "", round(r_t,2) if not np.isnan(r_t) else "", round(gap,2), datetime.datetime.now().isoformat()])
                else:
                    accs, times = {}, {}
                    for L in L_VALUES:
                        mode = "input_subspace_dm" if d > 60 else "disjoint"
                        t0 = time.perf_counter()
                        model = Pipeline([
                            ("scaler", StandardScaler()),
                            ("clf", DiverseMLMSVM(
                                num_layers=L, svms_per_block=M, rff_features=P,
                                kernel="arc_cosine", arc_cosine_degree=1, diversity_mode=mode,
                                block_C=10.0, final_C=1.0, random_state=seed,
                                normalize_inter_layer=True))
                        ]).fit(Xtr, ytr)
                        accs[L] = accuracy_score(yte, model.predict(Xte))
                        times[L] = time.perf_counter() - t0
                    
                    best_arch = max(accs.values())
                    g = (best_arch - r_acc) * 100 if not np.isnan(r_acc) else np.nan
                    gains.append(g)
                    
                    for L in L_VALUES:
                        l_accs[L].append(accs[L])
                        l_times[L].append(times[L])
                    
                    cw.writerow([ds_name, n, seed, round(m_acc,4), round(m_t,2), 
                                 round(r_acc,4) if not np.isnan(r_acc) else "", round(r_t,2) if not np.isnan(r_t) else "",
                                 round(accs[1],4), round(times[1],2), round(accs[2],4), round(times[2],2),
                                 round(accs[3],4), round(times[3],2), round(accs[4],4), round(times[4],2),
                                 round(accs[5],4), round(times[5],2), round(g,2), datetime.datetime.now().isoformat()])
                out.flush()
                
            mean_mlp = fmt(np.mean(mlp_accs), np.mean(mlp_times))
            mean_rbf = fmt(np.nanmean(rbf_accs), np.nanmean(rbf_times))
            mean_gain_str = f"{np.nanmean(gains):+6.1f}%" if not np.isnan(np.nanmean(gains)) else "   NaN%"
            
            if args.gap_only:
                print(f"{ds_name:15s} {n:6d} | {mean_mlp} | {mean_rbf} | {mean_gain_str}")
            else:
                mean_l = {L: fmt(np.mean(l_accs[L]), np.mean(l_times[L])) for L in L_VALUES}
                print(f"{ds_name:15s} {n:6d} | {mean_mlp} | {mean_rbf} | {mean_l[1]} | {mean_l[2]} | {mean_l[3]} | {mean_l[4]} | {mean_l[5]} | {mean_gain_str}")

if __name__ == "__main__":
    main()