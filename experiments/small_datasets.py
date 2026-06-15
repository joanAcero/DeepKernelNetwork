"""
small_datasets.py — small (<=~6k) UCI classification datasets for the feeding-strategy
comparison, returned in the SAME (X: float ndarray, y: int ndarray) form as utils.load(),
so exp_feeding_strategies.py treats them identically to HIGGS/CoverType/MNIST.

Provides load_small(tag) for tags:
    'optdigits'  Optical Recognition of Handwritten Digits  (~5620, d=64, K=10)
    'satimage'   Statlog Landsat Satellite                  (~6435, d=36, K=6)
    'digits8x8'  sklearn bundled 8x8 digits (offline fallback) (1797, d=64, K=10)

Each loader tries, in order: (1) ucimlrepo, (2) sklearn fetch_openml, (3) for
'digits8x8', the sklearn-bundled dataset (always available, no network).
X is float64, y is contiguous int labels 0..K-1 (LabelEncoder), matching the
downstream StandardScaler + DiverseMLMSVM pipeline.
"""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import LabelEncoder

# UCI ml-repo numeric ids for ucimlrepo.fetch_ucirepo(id=...)
_UCI_ID = {"optdigits": 80, "satimage": 146}
# OpenML names for sklearn.datasets.fetch_openml(name, version=1)
_OPENML = {"optdigits": "optdigits", "satimage": "satimage"}


def _finalize(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = LabelEncoder().fit_transform(np.asarray(y).ravel())
    return X, y.astype(int)


def _via_ucimlrepo(tag):
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=_UCI_ID[tag])
    X = ds.data.features.to_numpy()
    y = ds.data.targets.to_numpy()
    return _finalize(X, y)


def _via_openml(tag):
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(_OPENML[tag], version=1, as_frame=False)
    return _finalize(ds.data, ds.target)


def load_small(tag: str):
    tag = tag.lower()
    if tag == "digits8x8":
        from sklearn.datasets import load_digits
        d = load_digits()
        return _finalize(d.data, d.target)

    if tag not in _UCI_ID:
        raise ValueError(f"unknown small-dataset tag: {tag!r} "
                         f"(known: {sorted(_UCI_ID)} + 'digits8x8')")

    errors = []
    for loader in (_via_ucimlrepo, _via_openml):
        try:
            return loader(tag)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{loader.__name__}: {e!r}")
    raise RuntimeError(f"could not load {tag!r}; tried:\n  " + "\n  ".join(errors))


if __name__ == "__main__":
    # Self-test whatever is reachable in this environment.
    for tag in ["digits8x8", "optdigits", "satimage"]:
        try:
            X, y = load_small(tag)
            assert X.dtype == np.float64 and y.dtype == int
            assert X.ndim == 2 and y.ndim == 1 and len(X) == len(y)
            print(f"  {tag:12s} OK  n={len(y):,}  d={X.shape[1]}  K={len(np.unique(y))}  "
                  f"y in [{y.min()},{y.max()}]")
        except Exception as e:
            print(f"  {tag:12s} unavailable here: {repr(e)[:80]}")
