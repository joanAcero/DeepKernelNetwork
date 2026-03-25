"""
Dataset loaders.

Every loader returns a tuple (X, y) where:
    X : (n, d) float64 numpy array — NOT standardised (scaling is done
        per fold inside evaluate.py to avoid data leakage)
    y : (n,)   int numpy array     — class labels starting at 0

Datasets
--------
Small / sklearn builtins:
    load_iris, load_wine, load_breast_cancer

UCI core benchmarks (from Acero-Pousa & Belanche 2025):
    load_magic, load_covertype, load_spambase, load_ionosphere

Large-scale:
    load_susy (subsample available)

Grinsztajn tabular benchmark:
    load_grinsztajn_dataset (by name, requires openml)

High-dimensional benchmarks (DKN evaluation):
    load_madelon, load_gisette, load_musk2
"""

import numpy as np
from pathlib import Path

# ------------------------------------------------------------------ #
#  Small / sklearn builtins                                           #
# ------------------------------------------------------------------ #

def load_iris() -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_iris as _load
    data = _load()
    return data.data.astype(np.float64), data.target.astype(int)


def load_wine() -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_wine as _load
    data = _load()
    return data.data.astype(np.float64), data.target.astype(int)


def load_breast_cancer() -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_breast_cancer as _load
    data = _load()
    return data.data.astype(np.float64), data.target.astype(int)


# ------------------------------------------------------------------ #
#  UCI core benchmarks                                                #
# ------------------------------------------------------------------ #

def load_magic() -> tuple[np.ndarray, np.ndarray]:
    """
    MAGIC Gamma Telescope (19 020 samples, 10 features, 2 classes).
    https://archive.ics.uci.edu/dataset/159
    """
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(name="MagicTelescope", version=1, as_frame=False, parser="auto")
    X  = ds.data.astype(np.float64)
    # Target is 'g' / 'h' — encode as 0/1
    y  = (ds.target == "g").astype(int)
    return X, y


def load_covertype() -> tuple[np.ndarray, np.ndarray]:
    """
    Forest Cover Type (581 012 samples, 54 features, 7 classes).
    Uses sklearn's built-in loader.
    """
    from sklearn.datasets import fetch_covtype
    ds = fetch_covtype(as_frame=False)
    X  = ds.data.astype(np.float64)
    y  = (ds.target - 1).astype(int)   # labels 1-7 → 0-6
    return X, y


def load_spambase() -> tuple[np.ndarray, np.ndarray]:
    """
    Spambase (4 601 samples, 57 features, 2 classes).
    https://archive.ics.uci.edu/dataset/94
    """
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(name="spambase", version=1, as_frame=False, parser="auto")
    X  = ds.data.astype(np.float64)
    y  = ds.target.astype(int)
    return X, y


def load_ionosphere() -> tuple[np.ndarray, np.ndarray]:
    """
    Ionosphere (351 samples, 34 features, 2 classes).
    https://archive.ics.uci.edu/dataset/52
    """
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(name="ionosphere", version=1, as_frame=False, parser="auto")
    X  = ds.data.astype(np.float64)
    y  = (ds.target == "g").astype(int)
    return X, y


# ------------------------------------------------------------------ #
#  Large-scale                                                        #
# ------------------------------------------------------------------ #

def load_susy(
    n_samples: int | None = 50_000,
    data_dir: str = "data/raw",
) -> tuple[np.ndarray, np.ndarray]:
    """
    SUSY dataset (5M samples, 18 features, 2 classes).
    Download from: https://archive.ics.uci.edu/dataset/279
    and place SUSY.csv.gz in data/raw/.

    Parameters
    ----------
    n_samples : subsample size (None = load all 5M rows)
    data_dir  : directory containing SUSY.csv.gz
    """
    import pandas as pd

    path = Path(data_dir) / "SUSY.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"SUSY dataset not found at {path}.\n"
            "Download from https://archive.ics.uci.edu/dataset/279 "
            "and place SUSY.csv.gz in data/raw/."
        )

    df = pd.read_csv(path, header=None, nrows=n_samples)
    y  = df.iloc[:, 0].astype(int).values
    X  = df.iloc[:, 1:].astype(np.float64).values
    return X, y


# ------------------------------------------------------------------ #
#  Grinsztajn tabular benchmark via OpenML                           #
# ------------------------------------------------------------------ #

# A selection of dataset IDs from the Grinsztajn et al. (2022) benchmark.
# Full list: https://github.com/LeoGrin/tabular-benchmark
GRINSZTAJN_IDS: dict[str, int] = {
    "california_housing":   41540,
    "adult":                1590,
    "higgs_small":          23512,
    "jannis":               41168,
    "helena":               41169,
    "volkert":              41166,
    "MiniBooNE":            41150,
}


def load_grinsztajn_dataset(
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset from the Grinsztajn tabular benchmark via OpenML.

    Parameters
    ----------
    name : one of the keys in GRINSZTAJN_IDS

    Returns
    -------
    (X, y)
    """
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder

    if name not in GRINSZTAJN_IDS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(GRINSZTAJN_IDS)}"
        )
    ds_id = GRINSZTAJN_IDS[name]
    ds    = fetch_openml(data_id=ds_id, as_frame=False, parser="auto")
    X     = ds.data.astype(np.float64)
    le    = LabelEncoder()
    y     = le.fit_transform(ds.target)
    return X, y


# ------------------------------------------------------------------ #
#  High-dimensional benchmarks (DKN evaluation)                      #
# ------------------------------------------------------------------ #

def load_madelon() -> tuple[np.ndarray, np.ndarray]:
    """
    MADELON (2 600 samples, 500 features, 2 classes).
    https://archive.ics.uci.edu/dataset/171

    A synthetic dataset in which only 20 of 500 features are informative;
    the remaining 480 are noise or redundant linear combinations of the
    informative ones.  Designed to challenge feature selection and
    high-dimensional classifiers — ideal for evaluating AGOP rank truncation.

    Labels are originally {1, 2} — remapped to {0, 1}.
    The UCI version comes pre-split into train (2 000) and test (600);
    fetch_openml returns the merged set (n=2 600) which we use for
    our own 10-fold CV.
    """
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(name="madelon", version=1, as_frame=False, parser="auto")
    X  = ds.data.astype(np.float64)
    y  = ds.target.astype(int)
    unique = np.unique(y)
    label_map = {v: i for i, v in enumerate(unique)}
    y = np.array([label_map[yi] for yi in y])
    return X, y


def load_gisette() -> tuple[np.ndarray, np.ndarray]:
    """
    GISETTE (7 000 samples, 5 000 features, 2 classes).
    https://archive.ics.uci.edu/dataset/170

    Digit recognition task (4 vs 9 from MNIST) with very high-dimensional
    features, many of which are redundant or pure noise.  Created for the
    NIPS 2003 Feature Selection Challenge.

    Labels are originally {-1, 1} — remapped to {0, 1}.

    Note: first call downloads ~170 MB; cached in ~/scikit_learn_data/
    for subsequent runs.  Version is not pinned — OpenML's active version
    is used automatically to avoid the inactive-version warning.
    """
    from sklearn.datasets import fetch_openml
    # Do not pin version — OpenML version 1 is marked inactive;
    # omitting version lets fetch_openml select the current active version.
    ds = fetch_openml(name="gisette", as_frame=False, parser="auto")
    X  = ds.data.astype(np.float64)
    # Target may be strings ("-1", "1") or ints depending on OpenML version
    try:
        y_raw = ds.target.astype(int)
    except (ValueError, TypeError):
        y_raw = np.array([int(v) for v in ds.target])
    unique = np.unique(y_raw)
    label_map = {v: i for i, v in enumerate(unique)}
    y = np.array([label_map[yi] for yi in y_raw])
    return X, y


def load_musk2() -> tuple[np.ndarray, np.ndarray]:
    """
    MUSK Version 2 (6 598 samples, 166 features, 2 classes).
    https://archive.ics.uci.edu/dataset/75

    Molecular activity prediction (musk / non-musk).  Each molecule is
    described by 166 geometric conformation features.  Known to have
    complex non-linear structure that challenges fixed-bandwidth kernel
    methods — a useful intermediate between ionosphere (d=34) and
    GISETTE (d=5 000).

    Fetched by OpenML data_id=1116 (unambiguous; avoids version mismatch).
    The raw dataset contains molecule/conformation ID columns (strings)
    which are dropped — only numeric feature columns are retained.
    Labels are originally {0, 1} — already 0-based.
    """
    import pandas as pd
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(data_id=1116, as_frame=True, parser="auto")
    df = ds.data
    # Drop any columns that cannot be cast to float (e.g. molecule ID strings)
    numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().all()]
    X = df[numeric_cols].to_numpy(dtype=np.float64)
    try:
        y_raw = ds.target.astype(int).to_numpy()
    except (ValueError, TypeError):
        y_raw = np.array([int(v) for v in ds.target])
    unique = np.unique(y_raw)
    label_map = {v: i for i, v in enumerate(unique)}
    y = np.array([label_map[yi] for yi in y_raw])
    return X, y


def load_arcene() -> tuple[np.ndarray, np.ndarray]:
    """
    ARCENE — NIPS 2003 Feature Selection Challenge (OpenML data_id=1458).

    n=900 (combined train + validation sets), d=10 000, 2 classes.
    Mass spectrometry data for cancer detection.  Only ~7% of features
    are relevant; the remainder are noise probes inserted deliberately.
    This extreme sparsity of the signal makes it a canonical benchmark
    for high-dimensional feature selection and representation learning.

    Labels are originally {-1, 1} — remapped to {0, 1}.

    Note: n=900 is small relative to d=10 000 (p >> n regime).
    Use inner_splits=3 in tuning configs to preserve adequate inner
    training set sizes.
    """
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(data_id=1458, as_frame=False, parser="auto")
    X  = ds.data.astype(np.float64)
    try:
        y_raw = ds.target.astype(int)
    except (ValueError, TypeError):
        y_raw = np.array([int(float(v)) for v in ds.target])
    unique = np.unique(y_raw)
    label_map = {v: i for i, v in enumerate(unique)}
    y = np.array([label_map[yi] for yi in y_raw])
    return X, y


# ------------------------------------------------------------------ #
#  Registry — used by train.py to load by name                       #
# ------------------------------------------------------------------ #

LOADERS: dict[str, callable] = {
    "iris":           load_iris,
    "wine":           load_wine,
    "breast_cancer":  load_breast_cancer,
    "magic":          load_magic,
    "covertype":      load_covertype,
    "spambase":       load_spambase,
    "ionosphere":     load_ionosphere,
    "susy":           load_susy,
    **{k: (lambda k=k: load_grinsztajn_dataset(k)) for k in GRINSZTAJN_IDS},
    "madelon":        load_madelon,
    "gisette":        load_gisette,
    "musk2":          load_musk2,
    "arcene":         load_arcene,
}


def load(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Load dataset by name.

    Parameters
    ----------
    name   : key in LOADERS
    kwargs : passed to the loader (e.g. n_samples for SUSY)
    """
    if name not in LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(LOADERS)}"
        )
    loader = LOADERS[name]
    return loader(**kwargs) if kwargs else loader()
