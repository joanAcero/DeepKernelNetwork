"""
exp_width_depth_complete.py
===========================
Completion run for Experiment 3 (width x depth). Fills the parts that failed or were
absent in the first run, WITHOUT re-running the expensive datasets already completed
(MNIST, HIGGS, CoverType, and the binary SynthComp probe).

It adds:
  1. The three SMALL real datasets that failed before because this experiment's utils.load()
     does not know their tags: Digits8x8, Optdigits, Satimage. Here they are loaded via the
     small_datasets loader (the same one the feeding experiment used), so results are
     commensurable with the rest of Experiment 3.
  2. Two LEARNABLE MULTICLASS synthetic probes (the original SynthComp was binary and, as the
     first run confirmed, unlearnable at any depth — at chance ~0.51 — so it carries no signal;
     these replacements are designed to be learnable so width/depth have room to show an effect,
     with the OUTCOME LEFT OPEN):
       - SynthMC_blobs     : K=6 Gaussian blobs in an informative subspace + noise dims (easy)
       - SynthMC_nonlinear : K=5, label = argmax of K random quadratic scoring functions (harder)

All fixed settings (P, block_C, final_C, kernel, seeds, m, L grids, feeding rule, metrics)
are imported unchanged from exp_width_depth_scenarios2 so every new row matches the schema and
protocol of the original run. Output APPENDS to the same CSV by default.
"""
from __future__ import annotations
import argparse, datetime, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import exp_width_depth_scenarios2 as E      # reuse machinery: run_cell, metrics, grids, feeding
from utils import Tee, CSVWriter, hms, banner

# small real datasets via the dedicated loader (offline-safe for digits8x8; ucimlrepo/OpenML
# for optdigits/satimage, exactly as in the feeding experiment)
try:
    from small_datasets import load_small
except Exception as e:                       # pragma: no cover
    load_small = None
    _IMPORT_ERR = e

# ---------------- learnable multiclass synthetic probes ----------------
def make_synth_blobs(n, K=6, d_inf=6, d_noise=30, sep=2.2, seed=0):
    """K Gaussian blobs in a d_inf-dim informative subspace, embedded among noise dims.
    Clearly learnable (controllable difficulty via `sep`); a baseline multiclass probe."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((K, d_inf)) * sep
    y = rng.integers(0, K, n)
    Xi = centers[y] + rng.standard_normal((n, d_inf))
    Xn = rng.standard_normal((n, d_noise))
    X = np.hstack([Xi, Xn]); X = X[:, rng.permutation(X.shape[1])]
    return X.astype(np.float64), y.astype(int)

def make_synth_nonlinear_mc(n, K=5, d_inf=8, d_noise=30, seed=0):
    """Non-linear multiclass: label = argmax over K random quadratic scoring functions, so the
    class boundaries are curved and a linear model cannot separate them. Learnable but hard,
    leaving room for width/depth effects. Outcome left open."""
    rng = np.random.default_rng(seed)
    Xi = rng.standard_normal((n, d_inf))
    quad = [rng.standard_normal((d_inf, d_inf)) for _ in range(K)]
    lin  = [rng.standard_normal((d_inf,)) for _ in range(K)]
    scores = np.stack([Xi @ lin[k] + 0.7 * ((Xi * (Xi @ quad[k])).sum(1)) for k in range(K)], axis=1)
    y = scores.argmax(1)
    Xn = rng.standard_normal((n, d_noise))
    X = np.hstack([Xi, Xn]); X = X[:, rng.permutation(X.shape[1])]
    return X.astype(np.float64), y.astype(int)

# register the new datasets into the experiment's scenario + n-grid tables
E.SCENARIO["SynthMC_blobs"]     = ("probe_mc", "small")
E.SCENARIO["SynthMC_nonlinear"] = ("probe_mc", "small")
E.SCENARIO["Digits8x8"] = ("easy", "small")
E.SCENARIO["Optdigits"] = ("easy", "small")
E.SCENARIO["Satimage"]  = ("medium", "small")
E.N_GRID["SynthMC_blobs"]     = [1_000, 3_000]
E.N_GRID["SynthMC_nonlinear"] = [1_000, 3_000]
E.N_GRID.setdefault("Digits8x8", [500, 1_000, 1_297])
E.N_GRID.setdefault("Optdigits", [500, 2_000, 3_823])
E.N_GRID.setdefault("Satimage",  [500, 2_000, 4_435])

SMALL_TAGS = {"digits8x8", "optdigits", "satimage"}

def get_data(tag, nm):
    if nm == "SynthMC_blobs":
        return make_synth_blobs(6_000, seed=0)
    if nm == "SynthMC_nonlinear":
        return make_synth_nonlinear_mc(6_000, seed=0)
    if tag in SMALL_TAGS:
        if load_small is None:
            raise RuntimeError(f"small_datasets loader unavailable: {_IMPORT_ERR}")
        return load_small(tag)
    # fall back to the project loader for anything else
    from utils import load
    return load(tag)

def run(log_path, csv_path, datasets):
    tee = Tee(sys.stdout, log_path); sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Exp 3 COMPLETION — small real datasets + learnable multiclass synthetics",
               f"P={E.P} block_C={E.BLOCK_C} final_C={E.FINAL_C} seeds={E.N_SEEDS} "
               f"m={E.M_VALUES} L={E.L_VALUES}",
               f"feeding=regime-dependent (B_in_dm if d>={E.HIGH_D_THRESHOLD} else C)",
               "Appends to the Experiment 3 CSV; schema/protocol identical to the main run.")
        import time; t0 = time.perf_counter()
        for tag, nm in datasets:
            try:
                X, y = get_data(tag, nm)
                print(f"\n\n{'='*98}\n  DATASET: {nm}  (d={X.shape[1]}, K={len(np.unique(y))})\n{'='*98}")
                for n_total in E.N_GRID.get(nm, [10_000]):
                    E.run_cell(X, y, nm, n_total, csv_w)
            except Exception as ex:
                print(f"  [{nm}] FAILED: {ex}")
        print(f"\n  completion run done. {hms(time.perf_counter()-t0)}")
    finally:
        sys.stdout = tee._stream; tee.close(); csv_w.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--csv_dir", default="results")
    p.add_argument("--csv_name", default="exp_width_depth_scenarios2.csv",
                   help="appends to the main Exp-3 CSV by default")
    p.add_argument("--only", help="comma-separated subset of: "
                   "synthmc_blobs,synthmc_nonlinear,digits8x8,optdigits,satimage")
    a = p.parse_args()
    tagmap = {"synthmc_blobs":("synthmc_blobs","SynthMC_blobs"),
              "synthmc_nonlinear":("synthmc_nonlinear","SynthMC_nonlinear"),
              "digits8x8":("digits8x8","Digits8x8"),
              "optdigits":("optdigits","Optdigits"),
              "satimage":("satimage","Satimage")}
    order = ["synthmc_blobs","synthmc_nonlinear","digits8x8","optdigits","satimage"]
    if a.only:
        order = [k.strip().lower() for k in a.only.split(",")]
    datasets = [tagmap[k] for k in order if k in tagmap]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(a.log_dir, exist_ok=True); os.makedirs(a.csv_dir, exist_ok=True)
    run(os.path.join(a.log_dir, f"exp_width_depth_complete_{ts}.txt"),
        os.path.join(a.csv_dir, a.csv_name), datasets)
