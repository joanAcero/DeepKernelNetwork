#!/usr/bin/env python3
"""
run_all.py — run every experiment config in experiments/ sequentially.

Usage
-----
    python run_all.py                          # all configs, all models
    python run_all.py --skip dkn_align         # skip slow alignment model
    python run_all.py --only ionosphere magic  # only these datasets

Results are written to results/<dataset>.json after each experiment so
progress is never lost if the run is interrupted.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="+", default=[],
                        help="Model class keys to exclude (e.g. dkn_align)")
    parser.add_argument("--only", nargs="+", default=[],
                        help="Run only these dataset names")
    args = parser.parse_args()

    configs = sorted(Path("experiments").glob("*.json"))
    if not configs:
        print("No configs found in experiments/. Run from the dkn/ root.")
        return

    # Filter by --only
    if args.only:
        configs = [c for c in configs if c.stem in args.only]

    print(f"Running {len(configs)} experiment(s):")
    for c in configs:
        print(f"  {c}")

    for config_path in configs:
        with open(config_path) as f:
            cfg = json.load(f)

        # Remove skipped models from config in-memory (don't touch the file)
        if args.skip:
            cfg["models"] = {
                name: spec for name, spec in cfg["models"].items()
                if spec["class"] not in args.skip
            }
            if not cfg["models"]:
                print(f"\nSkipping {config_path.stem} — all models excluded.")
                continue

        # Write a temporary patched config
        tmp_path = config_path.parent / f"_tmp_{config_path.name}"
        with open(tmp_path, "w") as f:
            json.dump(cfg, f)

        print(f"\n{'='*60}")
        print(f"Dataset: {config_path.stem}")
        print(f"{'='*60}")

        try:
            from train import run_from_config
            run_from_config(str(tmp_path))
        except Exception as e:
            print(f"ERROR on {config_path.stem}: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
