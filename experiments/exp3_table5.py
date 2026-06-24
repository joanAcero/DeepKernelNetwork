#!/usr/bin/env python3
"""
Compute the Table 5 (Experiment 3) summary from the raw results CSV.

Convention (matching the thesis caption):
  - For each dataset, restrict to the LARGEST sample size n present.
  - Headline metric: macro-F1 on imbalanced datasets, accuracy on balanced ones.
  - "Best"   = max headline test metric over the (m, L) grid (arch rows only),
               averaged across seeds.
  - "Best (m,L)" = the (m, L) achieving that Best.
  - "L1"     = arch headline test metric at L=1 and the SAME width m as Best,
               averaged across seeds.   <-- this is the missing anchor column
  - "Depth gain" = (Best - L1) / L1 * 100   (relative percentage)
  - "Flat"   = flat_arccos test headline metric (seed-averaged) at largest n.
  - "RBF"    = rbf_rff (or exact rbf if present) test headline metric at largest n.

Usage:
  python3 exp3_table5.py results.csv
"""

import sys
import csv
from collections import defaultdict

# Which datasets are scored on macro-F1 (imbalanced) vs accuracy (balanced).
# Edit this mapping to match your reporting decisions.
HEADLINE_F1 = {"poker"}                      # imbalanced -> macro-F1
# everything else defaults to accuracy
def headline_field(dataset):
    return "test_f1" if dataset.lower() in HEADLINE_F1 else "test_acc"

def fnum(x):
    x = (x or "").strip()
    return float(x) if x else None

def main(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    # group rows by dataset
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)

    print(f"{'dataset':14s} {'metric':8s} {'n':>8s} {'Flat':>7s} {'RBF':>7s} "
          f"{'L1':>7s} {'Best':>7s} {'(m,L)':>8s} {'gain%':>8s}")
    print("-" * 86)

    for ds, rs in sorted(by_ds.items()):
        metric = headline_field(ds)
        # largest n for this dataset
        ns = sorted({int(r["n"]) for r in rs if r["n"]})
        n_big = ns[-1]
        sub = [r for r in rs if int(r["n"]) == n_big]

        # ---- arch grid: average headline test metric over seeds, per (m,L) ----
        cell = defaultdict(list)   # (m,L) -> [metric values across seeds]
        for r in sub:
            if r["method"] != "arch":
                continue
            v = fnum(r[metric])
            if v is None:
                continue
            cell[(int(r["m"]), int(r["L"]))].append(v)
        if not cell:
            print(f"{ds:14s}  (no arch rows at n={n_big})")
            continue
        cell_mean = {k: sum(v) / len(v) for k, v in cell.items()}

        # Best over the grid
        best_key = max(cell_mean, key=cell_mean.get)
        best_m, best_L = best_key
        best_val = cell_mean[best_key]

        # L1 anchor at the SAME width as Best
        l1_key = (best_m, 1)
        l1_val = cell_mean.get(l1_key)

        gain = ((best_val - l1_val) / l1_val * 100) if l1_val else float("nan")

        # ---- baselines, seed-averaged at largest n ----
        def base_mean(method_names):
            vals = []
            for r in sub:
                if r["method"] in method_names:
                    v = fnum(r[metric])
                    if v is not None:
                        vals.append(v)
            return sum(vals) / len(vals) if vals else None

        flat = base_mean({"flat_arccos"})
        rbf = base_mean({"rbf_rff", "rbf", "rbf_exact"})

        def fmt(x):
            return f"{x:.3f}" if x is not None else "  -  "

        print(f"{ds:14s} {('F1' if metric=='test_f1' else 'acc'):8s} "
              f"{n_big:8d} {fmt(flat):>7s} {fmt(rbf):>7s} "
              f"{fmt(l1_val):>7s} {fmt(best_val):>7s} "
              f"{f'({best_m},{best_L})':>8s} {gain:>+7.1f}%")

    print()
    print("Notes:")
    print(" - 'L1' is the depth-1 architecture at the SAME width m as Best.")
    print(" - gain% = (Best - L1)/L1 * 100  (relative percentage).")
    print(" - If a dataset's largest n in this CSV differs from the value you")
    print("   report in the thesis, pass a filtered CSV or adjust n selection.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 exp3_table5.py results.csv")
        sys.exit(1)
    main(sys.argv[1])
