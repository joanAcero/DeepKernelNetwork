"""
probe_headroom_baselines_images_100k.py 

Tests the Headroom Hypothesis tracking BOTH exact RBF and MLP baselines.
Uses complex raw image datasets (CIFAR, Fashion-MNIST, SVHN, EMNIST).
Automatically switches to RFF-approximated RBF for n > 15,000.
Grid: n = [5000, 10000, 50000, 100000]
"""
from __future__ import annotations
import sys, time, csv, datetime, warnings, argparse, os
import urllib.request
import tarfile
import pickle
from pathlib import Path
import numpy as np
import scipy.sparse as sp
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

sys.path.insert(0, str(Path(__file__).parent))
from mlsvm_extensions import DiverseMLMSVM

M = 50
L_VALUES = [1, 2, 3, 4, 5]
N_SEEDS = 3
P = 1000

# NEW GRID: Pushing into the 100k regime
N_GRID = [5000, 10000, 50000, 100000]
TEST_SIZE = 10_000
RBF_MAX_N = 15_000  # The threshold where Exact RBF is abandoned for RFF

# =====================================================================
# ROBUST IMAGE LOADERS
# =====================================================================

def download_and_extract_cifar(url, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    filename = url.split('/')[-1]
    filepath = os.path.join(extract_path, filename)
    if not os.path.exists(filepath):
        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
    print(f"  Extracting {filename}...")
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(path=extract_path)

def load_cifar10(n_samples):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    path = "./data/cifar10"
    download_and_extract_cifar(url, path)
    
    X, y = [], []
    for i in range(1, 6):
        with open(os.path.join(path, f"cifar-10-batches-py/data_batch_{i}"), 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            X.append(d[b'data'])
            y += d[b'labels']
    X = np.vstack(X).astype(float)
    y = np.array(y)
    return X, y

def load_cifar100(n_samples):
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    path = "./data/cifar100"
    download_and_extract_cifar(url, path)
    
    with open(os.path.join(path, "cifar-100-python/train"), 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        X = d[b'data'].astype(float)
        y = np.array(d[b'fine_labels'])
    return X, y

def fetch_image_data(name, version=1):
    print(f"  Downloading/Loading {name} from OpenML...")
    data = fetch_openml(name=name, version=version, as_frame=False, parser='auto')
    X_raw = data.data
    
    if sp.issparse(X_raw):
        X = X_raw.toarray()
    else:
        X = np.asarray(X_raw)
        
    X = np.nan_to_num(X.astype(float))
    y = LabelEncoder().fit_transform(data.target)
    return X, y

DATASETS = {
    "CIFAR-10":      lambda n: load_cifar10(n),
    "CIFAR-100":     lambda n: load_cifar100(n),
    "Fashion-MNIST": lambda n: fetch_image_data('Fashion-MNIST'),
    "SVHN":          lambda n: fetch_image_data('svhn_cropped'),     # ~99k samples
    "EMNIST":        lambda n: fetch_image_data('emnist-letters'),   # ~145k samples
}

# =====================================================================

def get_baselines(Xtr, ytr, Xte, yte, n, n_features):
    t0 = time.perf_counter()
    # Early stopping enabled because 100k images will otherwise take hours to train
    mlp = MLPClassifier(hidden_layer_sizes=(200, 200, 200), max_iter=500,
                        early_stopping=True, random_state=0).fit(Xtr, ytr)
    mlp_acc = accuracy_score(yte, mlp.predict(Xte))
    t_mlp = time.perf_counter() - t0
    
    rbf_acc, t_rbf = np.nan, np.nan
    rbf_type = "Exact"
    
    t0 = time.perf_counter()
    try:
        if n <= RBF_MAX_N:
            # Exact RBF kernel computation
            rbf = SVC(kernel="rbf", C=10, gamma="scale").fit(Xtr, ytr)
            rbf_acc = accuracy_score(yte, rbf.predict(Xte))
        else:
            # RFF Approximation for intractable N
            rbf_type = "RFF"
            gamma_scale = 1.0 / n_features 
            rff_pipe = Pipeline([
                ("rff", RBFSampler(gamma=gamma_scale, n_components=2000, random_state=0)),
                ("clf", LinearSVC(C=10.0, max_iter=1000, dual=False))
            ]).fit(Xtr, ytr)
            rbf_acc = accuracy_score(yte, rff_pipe.predict(Xte))
            
        t_rbf = time.perf_counter() - t0
    except Exception as e:
        print(f"  [RBF Failed: {e}]")
            
    return (mlp_acc, t_mlp), (rbf_acc, t_rbf), rbf_type

def fmt(acc, t):
    if np.isnan(acc): return "      NaN     "
    return f"{acc:.3f}({t:>5.1f}s)"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap-only", action="store_true", help="Only run baselines to check gap")
    args = parser.parse_args()

    out = open("results_headroom_images.csv", "w", newline="")
    cw = csv.writer(out)
    
    if args.gap_only:
        cw.writerow(["dataset","n","seed","mlp_acc","mlp_time","rbf_acc","rbf_time","rbf_type","gap","timestamp"])
        print(f"GAP CHECKER PROBE: Image Datasets up to 100k")
        print(f"{'dataset':15s} {'n':>6s} | {'MLP':>14s} | {'RBF':>14s} | {'Type':>5s} | {'Gap%':>7s}")
        print("-" * 75)
    else:
        cw.writerow(["dataset","n","seed","mlp_acc","mlp_time","rbf_acc","rbf_time","rbf_type",
                     "accL1","tL1","accL2","tL2","accL3","tL3","accL4","tL4","accL5","tL5",
                     "gain_vs_rbf_pct","timestamp"])
        print(f"HEADROOM PROBE: Tracking RBF vs MLP on Images (m={M})")
        print(f"{'dataset':15s} {'n':>6s} | {'MLP':>14s} | {'RBF':>14s} {'Type':>4s} | {'L1':>14s} | {'L2':>14s} | {'L3':>14s} | {'L4':>14s} | {'L5':>14s} | {'gain%':>7s}")
        print("-" * 145)
    
    for ds_name, data_func in DATASETS.items():
        max_req = max(N_GRID) + TEST_SIZE
        try:
            X_full, y_full = data_func(max_req)
        except Exception as e:
            print(f"Failed to load {ds_name}: {e}")
            continue
            
        d = X_full.shape[1]
        pool = len(y_full)
        
        valid_ns = []
        for n in N_GRID:
            # Dynamically limit n to the available pool size minus test set
            ts = min(TEST_SIZE, max(int(pool * 0.2), 500))
            if n <= pool - ts:
                valid_ns.append((n, ts))
            elif pool > ts and n == N_GRID[-1]:
                # If we asked for 100k but dataset is smaller, run its max possible capacity
                valid_ns.append((pool - ts, ts))
        
        # Deduplicate in case the max capacity equals a grid point
        valid_ns = sorted(list(set(valid_ns)))
        
        for n, ts in valid_ns:
            gains, mlp_accs, mlp_times, rbf_accs, rbf_times = [], [], [], [], []
            l_accs = {1: [], 2: [], 3: [], 4: [], 5: []}
            l_times = {1: [], 2: [], 3: [], 4: [], 5: []}
            last_rbf_type = ""
            
            for seed in range(N_SEEDS):
                rng = np.random.RandomState(7000 + seed)
                idx = rng.permutation(pool)[:n + ts]
                X, y = X_full[idx], y_full[idx]
                
                try:
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, random_state=seed, stratify=y)
                except ValueError:
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, random_state=seed)
                
                # Standard scaling is critical for image PCA/RFF
                sc = StandardScaler().fit(Xtr)
                Xtr_sc, Xte_sc = sc.transform(Xtr), sc.transform(Xte)
                
                (m_acc, m_t), (r_acc, r_t), rbf_type = get_baselines(Xtr_sc, ytr, Xte_sc, yte, n, d)
                last_rbf_type = rbf_type
                
                mlp_accs.append(m_acc); mlp_times.append(m_t)
                rbf_accs.append(r_acc); rbf_times.append(r_t)
                
                if args.gap_only:
                    gap = (m_acc - r_acc) * 100 if not np.isnan(r_acc) else np.nan
                    gains.append(gap)
                    cw.writerow([ds_name, n, seed, round(m_acc,4), round(m_t,2), round(r_acc,4) if not np.isnan(r_acc) else "", round(r_t,2) if not np.isnan(r_t) else "", rbf_type, round(gap,2), datetime.datetime.now().isoformat()])
                else:
                    accs, times = {}, {}
                    for L in L_VALUES:
                        # For high-dimensional images, feature partitioning is best
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
                                 round(r_acc,4) if not np.isnan(r_acc) else "", round(r_t,2) if not np.isnan(r_t) else "", rbf_type,
                                 round(accs[1],4), round(times[1],2), round(accs[2],4), round(times[2],2),
                                 round(accs[3],4), round(times[3],2), round(accs[4],4), round(times[4],2),
                                 round(accs[5],4), round(times[5],2), round(g,2), datetime.datetime.now().isoformat()])
                out.flush()
                
            mean_mlp = fmt(np.mean(mlp_accs), np.mean(mlp_times))
            mean_rbf = fmt(np.nanmean(rbf_accs), np.nanmean(rbf_times))
            mean_gap_str = f"{np.nanmean(gains):+6.1f}%" if not np.isnan(np.nanmean(gains)) else "   NaN%"
            
            if args.gap_only:
                print(f"{ds_name:15s} {n:6d} | {mean_mlp} | {mean_rbf} | {last_rbf_type[:5]:>5s} | {mean_gap_str}")
            else:
                mean_l = {L: fmt(np.mean(l_accs[L]), np.mean(l_times[L])) for L in L_VALUES}
                print(f"{ds_name:15s} {n:6d} | {mean_mlp} | {mean_rbf} {last_rbf_type[:4]:>4s} | {mean_l[1]} | {mean_l[2]} | {mean_l[3]} | {mean_l[4]} | {mean_l[5]} | {mean_gap_str}")

if __name__ == "__main__":
    main()