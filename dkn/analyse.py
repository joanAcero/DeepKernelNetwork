"""
analyse.py — load results from results/*.json and produce:

    1. A summary table (mean accuracy ± std, mean train time) per model
       across all datasets.
    2. A Friedman rank plot (lower = better) for model comparison.
    3. Per-dataset accuracy bar charts with error bars.
    4. A training time comparison chart.

Usage
-----
    python analyse.py                    # all results in results/
    python analyse.py --results results/ # explicit directory
    python analyse.py --no-plots         # print table only, no matplotlib
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import rankdata, wilcoxon


# ------------------------------------------------------------------ #
#  Load results                                                       #
# ------------------------------------------------------------------ #

def load_results(results_dir: str = "results") -> dict[str, dict]:
    """
    Load all result JSONs.

    Returns
    -------
    data : {dataset_name: {model_name: {mean_accuracy, std_accuracy,
                                         fold_accuracies, mean_time}}}
    """
    data = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        with open(path) as f:
            raw = json.load(f)
        dataset = raw.get("dataset", path.stem)
        data[dataset] = raw["models"]
    return data


# ------------------------------------------------------------------ #
#  Summary table                                                      #
# ------------------------------------------------------------------ #

def print_summary_table(data: dict[str, dict]) -> None:
    """Print a per-dataset × per-model accuracy table."""
    all_models  = sorted({m for d in data.values() for m in d})
    all_datasets = sorted(data.keys())

    col_w = 18
    header = f"{'Dataset':<20}" + "".join(f"{m[:col_w]:<{col_w}}" for m in all_models)
    print("\n" + header)
    print("-" * len(header))

    for dataset in all_datasets:
        row = f"{dataset:<20}"
        for model in all_models:
            if model in data[dataset]:
                r   = data[dataset][model]
                row += f"{r['mean_accuracy']:.3f}±{r['std_accuracy']:.3f}  "
            else:
                row += f"{'N/A':<{col_w}}"
        print(row)


# ------------------------------------------------------------------ #
#  Friedman ranking                                                   #
# ------------------------------------------------------------------ #

def friedman_ranks(data: dict[str, dict]) -> dict[str, float]:
    """
    Compute average Friedman rank per model across datasets.
    Lower rank = better. Only datasets where all models have results
    are included.
    """
    all_models   = sorted({m for d in data.values() for m in d})
    all_datasets = sorted(data.keys())

    # Only keep datasets where every model has a result
    complete = [
        ds for ds in all_datasets
        if all(m in data[ds] for m in all_models)
    ]

    if not complete:
        print("No dataset has results for all models — skipping Friedman ranks.")
        return {}

    # Accuracy matrix: (n_datasets, n_models)
    acc_matrix = np.array([
        [data[ds][m]["mean_accuracy"] for m in all_models]
        for ds in complete
    ])

    # Rank within each dataset row (higher acc = lower rank = better)
    rank_matrix = np.apply_along_axis(
        lambda row: rankdata(-row),   # negate so rank 1 = highest acc
        axis=1,
        arr=acc_matrix,
    )

    avg_ranks = rank_matrix.mean(axis=0)
    return dict(zip(all_models, avg_ranks))


def print_friedman_table(ranks: dict[str, float]) -> None:
    if not ranks:
        return
    print("\nFriedman average ranks (lower = better):")
    for model, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
        bar = "█" * int(rank * 4)
        print(f"  {model:<18} {rank:.3f}  {bar}")


# ------------------------------------------------------------------ #
#  Wilcoxon pairwise tests                                            #
# ------------------------------------------------------------------ #

def wilcoxon_vs_baseline(
    data: dict[str, dict],
    baseline: str = "ML-SVM",
    alpha: float  = 0.05,
) -> None:
    """
    For each model, run a Wilcoxon signed-rank test against the baseline
    over the per-fold accuracies pooled across datasets.
    """
    all_models = sorted({m for d in data.values() for m in d})
    if baseline not in all_models:
        print(f"Baseline '{baseline}' not found in results.")
        return

    print(f"\nWilcoxon signed-rank tests vs {baseline} (alpha={alpha}):")
    print(f"  {'Model':<20} {'p-value':>10}  {'significant':>12}  {'direction'}")
    print("  " + "-" * 60)

    for model in all_models:
        if model == baseline:
            continue

        # Pool fold-level accuracies across datasets
        base_scores, model_scores = [], []
        for ds_results in data.values():
            if baseline in ds_results and model in ds_results:
                base_scores.extend(ds_results[baseline]["fold_accuracies"])
                model_scores.extend(ds_results[model]["fold_accuracies"])

        if len(base_scores) < 5:
            print(f"  {model:<20} {'insufficient data':>10}")
            continue

        b  = np.array(base_scores)
        m  = np.array(model_scores)

        if np.all(b == m):
            print(f"  {model:<20} {'identical':>10}")
            continue

        stat, p = wilcoxon(m, b)
        sig     = "✓" if p < alpha else ""
        direction = "better" if m.mean() > b.mean() else "worse"
        print(f"  {model:<20} {p:>10.4f}  {sig:>12}  {direction}")


# ------------------------------------------------------------------ #
#  Plots (optional — requires matplotlib)                             #
# ------------------------------------------------------------------ #

def plot_accuracy_bars(data: dict[str, dict], output_dir: str = "results") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        return

    all_models   = sorted({m for d in data.values() for m in d})
    all_datasets = sorted(data.keys())
    n_models     = len(all_models)
    colours      = plt.cm.tab10(np.linspace(0, 1, n_models))

    fig, axes = plt.subplots(
        1, len(all_datasets),
        figsize=(5 * len(all_datasets), 4),
        sharey=True,
    )
    if len(all_datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, all_datasets):
        xs     = np.arange(n_models)
        means  = []
        stds   = []
        labels = []
        for model in all_models:
            if model in data[dataset]:
                means.append(data[dataset][model]["mean_accuracy"])
                stds.append(data[dataset][model]["std_accuracy"])
            else:
                means.append(0)
                stds.append(0)
            labels.append(model)

        bars = ax.bar(xs, means, yerr=stds, capsize=4,
                      color=colours, alpha=0.85, edgecolor="white")
        ax.set_title(dataset, fontsize=11)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0.5, 1.02)
        ax.set_ylabel("Accuracy" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Mean accuracy ± std (10-fold CV)", fontsize=13)
    plt.tight_layout()
    out = Path(output_dir) / "accuracy_bars.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_training_times(data: dict[str, dict], output_dir: str = "results") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    all_models   = sorted({m for d in data.values() for m in d})
    all_datasets = sorted(data.keys())

    # Average train time per model across datasets
    avg_times = {}
    for model in all_models:
        times = [
            data[ds][model]["mean_time"]
            for ds in all_datasets
            if model in data[ds]
        ]
        if times:
            avg_times[model] = np.mean(times)

    models_sorted = sorted(avg_times, key=avg_times.get)
    times_sorted  = [avg_times[m] for m in models_sorted]

    fig, ax = plt.subplots(figsize=(7, 3))
    colours = plt.cm.tab10(np.linspace(0, 1, len(models_sorted)))
    ax.barh(models_sorted, times_sorted, color=colours, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Mean training time per fold (s)")
    ax.set_title("Training time comparison (avg across datasets)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = Path(output_dir) / "training_times.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_friedman_ranks(ranks: dict[str, float], output_dir: str = "results") -> None:
    if not ranks:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    models  = sorted(ranks, key=ranks.get)
    values  = [ranks[m] for m in models]
    colours = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(models)))

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(models, values, color=colours, edgecolor="white", alpha=0.9)
    ax.set_xlabel("Average Friedman rank (lower = better)")
    ax.set_title("Friedman ranking across datasets")
    ax.axvline(1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = Path(output_dir) / "friedman_ranks.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Analyse DKN benchmark results")
    parser.add_argument("--results",   default="results")
    parser.add_argument("--baseline",  default="ML-SVM",
                        help="Model to use as Wilcoxon baseline")
    parser.add_argument("--no-plots",  action="store_true")
    args = parser.parse_args()

    data = load_results(args.results)
    if not data:
        print(f"No result files found in '{args.results}/'.")
        return

    print(f"Loaded results for {len(data)} dataset(s): {sorted(data)}")

    print_summary_table(data)

    ranks = friedman_ranks(data)
    print_friedman_table(ranks)

    wilcoxon_vs_baseline(data, baseline=args.baseline)

    if not args.no_plots:
        plot_accuracy_bars(data, output_dir=args.results)
        plot_training_times(data, output_dir=args.results)
        plot_friedman_ranks(ranks, output_dir=args.results)


if __name__ == "__main__":
    main()
