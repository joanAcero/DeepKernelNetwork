"""
evaluate.py — benchmark loop with optional per-model hyperparameter tuning.

The benchmark is driven entirely by JSON experiment configs.  Each model
entry in the JSON may have an optional "param_grid" block that triggers
inner-fold grid search (proper nested cross-validation).  Models without
a param_grid use their fixed "params" exactly as before — no behaviour
change for the untuned case.

JSON schema (see experiments/*.json for full examples):

    {
      "dataset": "ionosphere",
      "n_splits": 10,
      "seed":     42,
      "models": {
        "DKN-AGOP": {
          "class":  "dkn_agop",
          "params": {"n_layers": 2, "D": 500,
                     "rank": 50, "C": 1.0, "agop_steps": 1},
          "param_grid": {
            "C":    [0.1, 1.0, 10.0],
            "rank": [20, 50, 100],
            "D":    [200, 500, 1000]
          },
          "tuning": {
            "inner_splits": 5,
            "strategy":     "grid"
          }
        },
        "SVC-RBF": {
          "class":  "svc",
          "params": {"C": 1.0, "gamma": "scale"}
        }
      }
    }

Tuning block fields (all optional, shown with defaults):
    inner_splits : int   — inner CV folds             (default: 5)
    strategy     : str   — "grid" or "random"         (default: "grid")
    n_iter       : int   — candidates for random search (default: 20)

Usage
-----
From train.py (programmatic):
    results = benchmark(model_specs, X, y, n_splits=10)

From the command line (quick check):
    python evaluate.py --dataset iris --models mlsvm dkn_agop svc mlp
"""

import time
import json
import itertools
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing   import StandardScaler

from models.base         import BaseModel
from models.mlsvm        import MLSVM
from models.dkn_agop     import DKN_AGOP
from models.dkn_alignment import DKN_Alignment
from models.dkn_rfm_align import DKN_RFM_Align
from models.svc          import SVCBaseline
from models.mlp          import MLPBaseline
from models.xgboost      import XGBoostBaseline
from data.loaders        import load


# ------------------------------------------------------------------ #
#  Class registry — maps JSON "class" strings to Python classes       #
# ------------------------------------------------------------------ #

MODEL_CLASSES: dict[str, type] = {
    "mlsvm":        MLSVM,
    "dkn_agop":     DKN_AGOP,
    "dkn_align":    DKN_Alignment,
    "dkn_rfm_align": DKN_RFM_Align,
    "svc":          SVCBaseline,
    "mlp":          MLPBaseline,
    "xgboost":      XGBoostBaseline,
}


def _instantiate(class_key: str, params: dict) -> BaseModel:
    """Instantiate a model from its registry key and a params dict."""
    cls = MODEL_CLASSES.get(class_key)
    if cls is None:
        raise ValueError(
            f"Unknown model class '{class_key}'. "
            f"Available: {list(MODEL_CLASSES)}"
        )
    return cls(**params)


# ------------------------------------------------------------------ #
#  Default model specs (used by the CLI)                              #
# ------------------------------------------------------------------ #

def default_model_specs() -> dict[str, dict]:
    """
    Return model specs with sensible defaults and no tuning grids.
    Used by the CLI entry point; production runs should use JSON configs.
    """
    return {
        "ML-SVM": {
            "class":  "mlsvm",
            "params": {"n_layers": 2, "n_components": 1000},
        },
        "DKN-AGOP": {
            "class":  "dkn_agop",
            "params": {"n_layers": 2, "D": 500,
                       "rank": 50, "C": 1.0, "agop_steps": 1},
        },
        "DKN-Align": {
            "class":  "dkn_align",
            "params": {"n_layers": 2, "d_k": 200, "D": 500,
                       "C": 1.0, "lr": 1e-3, "epochs": 300,
                       "batch_size": 512},
        },
        "DKN-RFM-Align": {
            "class":  "dkn_rfm_align",
            "params": {"n_layers": 2, "D": 500, "rank": 50,
                       "C": 1.0, "agop_steps": 5,
                       "lr": 1e-3, "epochs": 300, "batch_size": 512},
        },
        "SVC-RBF": {
            "class":  "svc",
            "params": {"C": 1.0, "gamma": "scale"},
        },
        "MLP": {
            "class":  "mlp",
            "params": {"hidden_layer_sizes": [256, 256],
                       "alpha": 1e-4, "max_iter": 500},
        },
        "XGBoost": {
            "class":  "xgboost",
            "params": {"n_estimators": 500, "learning_rate": 0.05},
        },
    }


# ------------------------------------------------------------------ #
#  Inner-fold hyperparameter search                                   #
# ------------------------------------------------------------------ #

def _make_param_candidates(
    param_grid: dict[str, list],
    strategy: str,
    n_iter: int,
    seed: int,
) -> list[dict]:
    """
    Generate the list of hyperparameter combinations to evaluate.

    strategy="grid"   → full Cartesian product (ignores n_iter)
    strategy="random" → n_iter random samples from the grid (with replacement)
    """
    keys   = list(param_grid.keys())
    values = list(param_grid.values())

    if strategy == "grid":
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    elif strategy == "random":
        rng        = np.random.default_rng(seed)
        candidates = []
        for _ in range(n_iter):
            sample = {k: rng.choice(v).item() if isinstance(rng.choice(v), np.generic)
                      else rng.choice(v)
                      for k, v in zip(keys, values)}
            candidates.append(sample)
        return candidates

    else:
        raise ValueError(
            f"Unknown tuning strategy '{strategy}'. Use 'grid' or 'random'."
        )


def _inner_cv(
    class_key:   str,
    base_params: dict,
    param_grid:  dict[str, list],
    X_tr:        np.ndarray,
    y_tr:        np.ndarray,
    inner_splits: int = 5,
    strategy:    str  = "grid",
    n_iter:      int  = 20,
    seed:        int  = 42,
    verbose:     bool = True,
) -> tuple[dict, dict]:
    """
    Run inner cross-validation on X_tr / y_tr to select hyperparameters.

    For each candidate in the param_grid, fits the model on (inner_splits-1)
    folds and evaluates on the held-out inner fold.  The candidate with the
    highest mean inner accuracy is returned.

    Parameters
    ----------
    class_key    : model registry key (e.g. "dkn_agop")
    base_params  : fixed params not in the grid (always passed to __init__)
    param_grid   : dict of param_name → list of values to try
    X_tr, y_tr   : outer training fold (inner CV runs entirely inside this)
    inner_splits : number of inner CV folds
    strategy     : "grid" or "random"
    n_iter       : number of candidates for random search
    seed         : RNG seed for random search and inner fold splitting
    verbose      : print inner CV progress

    Returns
    -------
    best_params : dict — base_params merged with the best grid combination
    cv_scores   : dict — {str(candidate): mean_inner_acc} for all candidates
    """
    candidates = _make_param_candidates(param_grid, strategy, n_iter, seed)
    inner_skf  = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=seed
    )

    cv_scores: dict[str, float] = {}

    for candidate in candidates:
        params   = {**base_params, **candidate}
        fold_accs = []

        for inner_tr_idx, inner_val_idx in inner_skf.split(X_tr, y_tr):
            X_itr, X_ival = X_tr[inner_tr_idx], X_tr[inner_val_idx]
            y_itr, y_ival = y_tr[inner_tr_idx], y_tr[inner_val_idx]

            # Scale inside the inner fold to prevent leakage
            import scipy.sparse as sp
            _with_mean_inner = not sp.issparse(X_itr)
            inner_scaler = StandardScaler(with_mean=_with_mean_inner).fit(X_itr)
            X_itr  = inner_scaler.transform(X_itr)
            X_ival = inner_scaler.transform(X_ival)
            if sp.issparse(X_itr):
                X_itr  = X_itr.toarray()
                X_ival = X_ival.toarray()

            model = _instantiate(class_key, params)
            model.fit(X_itr, y_itr)
            fold_accs.append(model.score(X_ival, y_ival))

        mean_acc = float(np.mean(fold_accs))
        key      = json.dumps(candidate, sort_keys=True)
        cv_scores[key] = mean_acc

        if verbose:
            print(
                f"    [inner CV] {candidate} → "
                f"acc={mean_acc:.4f} ± {np.std(fold_accs):.4f}"
            )

    # Select best candidate
    best_key       = max(cv_scores, key=cv_scores.__getitem__)
    best_candidate = json.loads(best_key)
    best_params    = {**base_params, **best_candidate}

    if verbose:
        print(f"    [inner CV] best: {best_candidate} (acc={cv_scores[best_key]:.4f})")

    return best_params, cv_scores


# ------------------------------------------------------------------ #
#  Core benchmark function                                            #
# ------------------------------------------------------------------ #

def benchmark(
    model_specs:  dict[str, dict],
    X:            np.ndarray,
    y:            np.ndarray,
    n_splits:     int   = 10,
    seed:         int   = 42,
    verbose:      bool  = False,
    eta_interval: float = 60.0,
) -> dict[str, dict]:
    """
    Stratified K-fold benchmark with optional per-model inner-fold tuning.

    For each outer fold:
        1.  Split into outer train / test.
        2.  Fit StandardScaler on the outer training fold only.
        3.  Transform both outer train and test.
        4.  For each model:
              a. If "param_grid" is present: run inner CV on the outer
                 training fold to select the best hyperparameters
                 (proper nested CV — no leakage into the test fold).
              b. Instantiate (or re-instantiate) the model with the
                 selected (or fixed) parameters.
              c. Fit on the outer scaled training fold.
              d. Evaluate accuracy on the outer scaled test fold.

    Parameters
    ----------
    model_specs : dict  name → spec dict with keys:
                    "class"      : str  (required)
                    "params"     : dict (required)
                    "param_grid" : dict (optional) — triggers tuning
                    "tuning"     : dict (optional) — inner CV settings:
                        "inner_splits" : int  (default 5)
                        "strategy"     : str  "grid"|"random" (default "grid")
                        "n_iter"       : int  (default 20, random only)
    X           : (n, d) feature matrix (unscaled)
    y           : (n,)   integer labels
    n_splits    : K in outer K-fold
    seed        : random seed for outer fold splitting and inner CV
    verbose     : print per-fold progress

    Returns
    -------
    results : dict with structure:
        {
          model_name: {
              "fold_accuracies":        [float, ...],
              "mean_accuracy":          float,
              "std_accuracy":           float,
              "fold_times":             [float, ...],
              "mean_time":              float,
              "params":                 dict,   # fixed params
              "param_grid":             dict,   # grid used (or null)
              "best_params_per_fold":   [dict, ...],  # null if no tuning
              "inner_cv_scores":        [dict, ...],  # null if no tuning
          }
        }
    """
    outer_skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )

    results = {}
    for name, spec in model_specs.items():
        results[name] = {
            "fold_accuracies":      [],
            "fold_times":           [],
            "params":               spec["params"],
            "param_grid":           spec.get("param_grid", None),
            "best_params_per_fold": [],
            "inner_cv_scores":      [],
        }

    # ------------------------------------------------------------------ #
    #  ETA tracking — time-gated, quiet by default                        #
    #  Prints one progress line every `eta_interval` seconds so the       #
    #  terminal is not flooded, but you can always see where things are.  #
    #  verbose=True adds per-fold per-model lines (useful for debugging). #
    # ------------------------------------------------------------------ #
    experiment_start  = time.perf_counter()
    total_units       = n_splits * len(model_specs)
    completed_units   = 0
    last_eta_print    = experiment_start

    def _fmt_seconds(s: float) -> str:
        s = int(s)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        if h:   return f"{h}h {m}m {sec}s"
        if m:   return f"{m}m {sec}s"
        return f"{sec}s"

    def _maybe_print_eta(completed: int, total: int, now: float) -> None:
        nonlocal last_eta_print
        if completed == 0:
            return
        elapsed   = now - experiment_start
        remaining = (elapsed / completed) * (total - completed)
        pct       = 100.0 * completed / total
        print(
            f"[{_fmt_seconds(elapsed)} elapsed] "
            f"{completed}/{total} units ({pct:.0f}%) | "
            f"ETA ~{_fmt_seconds(remaining)}",
            flush=True,
        )
        last_eta_print = now

    # Print start banner once
    model_list = ', '.join(model_specs.keys())
    print(
        f'Starting benchmark — {len(model_specs)} model(s): {model_list} | '
        f'{n_splits} folds | ETA printed every {int(eta_interval)}s',
        flush=True,
    )

    for fold, (train_idx, test_idx) in enumerate(outer_skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Scale on the outer training fold only
        # with_mean=False supports sparse matrices (e.g. GISETTE);
        # has no effect on dense arrays.
        import scipy.sparse as sp
        _with_mean = not sp.issparse(X_tr)
        scaler = StandardScaler(with_mean=_with_mean).fit(X_tr)
        X_tr   = scaler.transform(X_tr)
        X_te   = scaler.transform(X_te)
        # Convert sparse to dense after scaling — all models expect dense input
        if sp.issparse(X_tr):
            X_tr = X_tr.toarray()
            X_te = X_te.toarray()

        for name, spec in model_specs.items():
            class_key   = spec["class"]
            base_params = spec["params"].copy()
            param_grid  = spec.get("param_grid", None)
            tuning_cfg  = spec.get("tuning", {})

            if param_grid:
                inner_splits = tuning_cfg.get("inner_splits", 5)
                strategy     = tuning_cfg.get("strategy", "grid")
                n_iter       = tuning_cfg.get("n_iter", 20)
                inner_seed   = seed + fold * 1000

                best_params, cv_scores = _inner_cv(
                    class_key    = class_key,
                    base_params  = base_params,
                    param_grid   = param_grid,
                    X_tr         = X_tr,
                    y_tr         = y_tr,
                    inner_splits = inner_splits,
                    strategy     = strategy,
                    n_iter       = n_iter,
                    seed         = inner_seed,
                    verbose      = False,     # inner CV never prints
                )
                results[name]["best_params_per_fold"].append(best_params)
                results[name]["inner_cv_scores"].append(cv_scores)

            else:
                best_params = base_params
                results[name]["best_params_per_fold"].append(None)
                results[name]["inner_cv_scores"].append(None)

            # -------------------------------------------------------- #
            # Outer fit and evaluation                                  #
            # -------------------------------------------------------- #
            model = _instantiate(class_key, best_params)

            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            t1 = time.perf_counter()

            acc = model.score(X_te, y_te)
            results[name]["fold_accuracies"].append(acc)
            results[name]["fold_times"].append(t1 - t0)

            completed_units += 1
            now              = time.perf_counter()

            if verbose:
                # Detailed per-unit line (debug / development mode)
                tuned_tag = " [tuned]" if param_grid else ""
                print(
                    f"Fold {fold+1:2d}/{n_splits} | {name:15s} | "
                    f"acc={acc:.4f} | {t1-t0:.1f}s{tuned_tag}",
                    flush=True,
                )

            # Always print ETA every eta_interval seconds
            if now - last_eta_print >= eta_interval:
                _maybe_print_eta(completed_units, total_units, now)

    # Final ETA flush (100% complete)
    total_elapsed = time.perf_counter() - experiment_start
    print(f'Benchmark complete — total time {_fmt_seconds(total_elapsed)}', flush=True)

    # Aggregate
    for name in results:
        accs  = results[name]["fold_accuracies"]
        times = results[name]["fold_times"]
        results[name]["mean_accuracy"] = float(np.mean(accs))
        results[name]["std_accuracy"]  = float(np.std(accs))
        results[name]["mean_time"]     = float(np.mean(times))

    return results


# ------------------------------------------------------------------ #
#  Pretty-print summary                                               #
# ------------------------------------------------------------------ #

def print_summary(results: dict[str, dict]) -> None:
    """Print a ranked table of results."""
    print("\n" + "=" * 62)
    print(f"{'Model':<18}  {'Acc (mean±std)':<20}  {'Time (s)'}")
    print("=" * 62)

    ranked = sorted(results.items(), key=lambda kv: -kv[1]["mean_accuracy"])
    for name, r in ranked:
        mean   = r["mean_accuracy"]
        std    = r["std_accuracy"]
        t      = r["mean_time"]
        tuned  = " *" if r.get("param_grid") else ""
        print(f"{name:<18}  {mean:.4f} ± {std:.4f}         {t:6.1f}{tuned}")

    if any(r.get("param_grid") for r in results.values()):
        print("  * tuned via inner CV")
    print("=" * 62)


# ------------------------------------------------------------------ #
#  Save results to JSON                                               #
# ------------------------------------------------------------------ #

def save_results(
    results:    dict[str, dict],
    dataset:    str,
    output_dir: str = "results",
) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"{dataset}.json"
    with open(path, "w") as f:
        json.dump({"dataset": dataset, "models": results}, f, indent=2)
    print(f"\nResults saved to {path}")
    return path


# ------------------------------------------------------------------ #
#  CLI entry point                                                    #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Run DKN benchmark")
    parser.add_argument("--dataset",  default="iris",
                        help="Dataset name (see data/loaders.py)")
    parser.add_argument("--models",   nargs="+",
                        default=list(MODEL_CLASSES),
                        choices=list(MODEL_CLASSES),
                        help="Which models to evaluate")
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--output",   default="results")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    X, y = load(args.dataset)
    print(f"  X.shape={X.shape}, classes={np.unique(y)}")

    all_specs   = default_model_specs()
    model_specs = {
        name: spec for name, spec in all_specs.items()
        if spec["class"] in args.models
    }

    results = benchmark(model_specs, X, y,
                        n_splits=args.n_splits, seed=args.seed, verbose=True)
    print_summary(results)
    save_results(results, dataset=args.dataset, output_dir=args.output)


if __name__ == "__main__":
    main()
