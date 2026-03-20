"""
train.py — load an experiment config and run the benchmark.

This file is the bridge between the JSON experiment configs and
evaluate.py.  It reads the config, passes model specs (with optional
param_grid) directly to benchmark(), and saves results.

Usage
-----
    python train.py experiments/ionosphere.json
    python train.py experiments/magic.json
"""

import json
import argparse
import numpy as np
from pathlib import Path

from evaluate import benchmark, print_summary, save_results
from data.loaders import load


def run_from_config(config_path: str) -> dict:
    """
    Load a JSON experiment config and run the full benchmark.

    The config is passed verbatim to benchmark() — model specs (class,
    params, param_grid, tuning) are read directly from the JSON so all
    experimental decisions live in one place.

    Parameters
    ----------
    config_path : path to a JSON experiment config

    Returns
    -------
    results : benchmark output dict (also saved to results/<dataset>.json)
    """
    with open(config_path) as f:
        cfg = json.load(f)

    dataset  = cfg["dataset"]
    n_splits = cfg.get("n_splits", 10)
    seed     = cfg.get("seed",     42)
    specs    = cfg["models"]

    print(f"Loading dataset: {dataset}")
    X, y = load(dataset)
    print(f"  X.shape={X.shape}, n_classes={len(np.unique(y))}")

    results = benchmark(
        model_specs = specs,
        X           = X,
        y           = y,
        n_splits    = n_splits,
        seed        = seed,
        verbose     = True,
    )

    print_summary(results)
    save_results(results, dataset=dataset)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run a single DKN experiment from a JSON config."
    )
    parser.add_argument(
        "config",
        help="Path to experiment JSON (e.g. experiments/ionosphere.json)"
    )
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
