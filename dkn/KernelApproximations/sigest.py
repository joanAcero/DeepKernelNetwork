"""
sigest.py — Python port of the R kernlab sigest function.

Original R implementation:
    kernlab package, https://cran.r-project.org/web/packages/kernlab/index.html

Author of original Python port: Joan Acero Pousa (May 2024)
Bugs fixed (this version):
    FIX 1 — Masked-array slicing failed when no NaN/Inf values present.
             Replaced np.ma with explicit row-wise boolean mask.
    FIX 2 — process_input mutated the caller's array in-place.
             Added self.x = self.x.copy() at entry.
    FIX 3 — Hilbert-space distance used outer-product indexing (n×n matrix)
             then summed over axis=1, giving wrong paired distances.
             Fixed to vectorised paired indexing: K[i,i] + K[j,j] - 2K[i,j].
    FIX 4 — On detecting a constant column, scaling was disabled for ALL
             columns instead of only the offending ones.

Public API
----------
sigest_gamma(X, frac=1, scaled=True, quantile=0.5) -> float
    Convenience wrapper: runs sigest and returns a single gamma value
    (default: median of the estimated range, recommended by kernlab).

MySigest
    Full class interface, returns the [90th, 50th, 10th] percentile range.
"""

import numpy as np
from numpy.random import default_rng
from scipy.stats import zscore


# ------------------------------------------------------------------ #
#  Convenience wrapper (recommended entry point)                      #
# ------------------------------------------------------------------ #

def sigest_gamma(
    X: np.ndarray,
    frac: float = 1.0,
    scaled: bool = True,
    quantile: float = 0.5,
    seed: int = 42,
) -> float:
    """
    Estimate the RBF kernel gamma from the data distribution using sigest.

    Runs the Euclidean-distance version of sigest and returns a single
    gamma value.  The default quantile=0.5 (median) is the value recommended
    by the original kernlab implementation for general use.

    Parameters
    ----------
    X        : (n, d) input array (will not be modified)
    frac     : fraction of n to sample for distance estimation.
               frac=1 draws n pairs with replacement; for large n (>5000)
               consider frac=0.5 to reduce computation.
    scaled   : whether to z-score each feature before computing distances.
               Set to False if X is already standardised.
    quantile : which quantile of the estimated sigma range to return.
               0.5  → median (default, recommended)
               0.1  → smaller sigma / sharper kernel
               0.9  → larger sigma / smoother kernel
    seed     : random seed for reproducible sampling.

    Returns
    -------
    gamma : float  (= 1 / sigma at the requested quantile)
    """
    est = MySigest(X, distance="Euclidean", frac=frac, scaled=scaled, seed=seed)
    srange = est.sigest()   # [gamma_90, gamma_50, gamma_10]

    # Map quantile to the correct index in the returned array.
    # srange[0] corresponds to the 90th percentile of distances (smallest sigma → largest gamma).
    # srange[2] corresponds to the 10th percentile of distances (largest  sigma → smallest gamma).
    q_map = {0.9: 0, 0.5: 1, 0.1: 2}
    if quantile in q_map:
        return float(srange[q_map[quantile]])
    else:
        # Interpolate for arbitrary quantiles
        gammas = np.array(srange)  # [gamma_90, gamma_50, gamma_10] — decreasing
        qs     = np.array([0.9, 0.5, 0.1])
        return float(np.interp(quantile, qs[::-1], gammas[::-1]))


# ------------------------------------------------------------------ #
#  Full class interface                                               #
# ------------------------------------------------------------------ #

class MySigest:
    """
    Python port of the R kernlab sigest function.

    Estimates the sigma (bandwidth) parameter for an RBF kernel by
    computing the distribution of pairwise distances on a random sample
    of the data, then returning the 1/sigma values at the 90th, 50th,
    and 10th percentiles of that distribution.

    The 50th percentile (median) is the recommended default for SVM
    bandwidth selection (see kernlab documentation).

    Parameters
    ----------
    x              : (n, d) input array.  Will not be modified.
    distance       : 'Euclidean' or 'HilbertSpace'.
    frac           : fraction of n to sample (default 1).
    na_action      : 'omit' removes rows containing NaN or Inf.
    scaled         : bool or array of bool per feature.
    previousKernel : (n, n) precomputed kernel matrix, required when
                     distance='HilbertSpace'.
    seed           : random seed for reproducible sampling.
    """

    def __init__(
        self,
        x: np.ndarray,
        distance: str = "Euclidean",
        frac: float = 1.0,
        na_action: str = "omit",
        scaled: bool = True,
        previousKernel: np.ndarray = None,
        seed: int = 42,
    ):
        self.x              = x
        self.distance       = distance
        self.frac           = frac
        self.na_action      = na_action
        self.scaled         = scaled
        self.previousKernel = previousKernel
        self.seed           = seed

        # Validate HilbertSpace mode eagerly
        if distance == "HilbertSpace" and previousKernel is None:
            raise ValueError(
                "distance='HilbertSpace' requires previousKernel to be provided."
            )

    def process_input(self):
        """
        Copy the input, remove invalid rows, and scale features if requested.

        FIX 2: self.x is copied before any mutation so the caller's array
                is never modified.
        FIX 1: NaN/Inf detection uses explicit boolean masking rather than
                np.ma, which silently failed when no invalid values existed.
        FIX 4: Constant-column detection disables scaling only for the
                offending columns, not for the entire array.
        """
        # FIX 2: always work on a copy
        self.x = self.x.copy()

        # FIX 1: remove rows that contain any NaN or Inf
        if self.na_action == "omit":
            invalid_rows = ~(np.isfinite(self.x).all(axis=1))
            if invalid_rows.any():
                n_removed = invalid_rows.sum()
                print(f"sigest: removed {n_removed} row(s) containing NaN or Inf.")
                self.x = self.x[~invalid_rows]

        n, d = self.x.shape

        # Normalise scaled to a boolean array of length d
        if isinstance(self.scaled, bool):
            self.scaled = np.full(d, self.scaled, dtype=bool)
        else:
            self.scaled = np.asarray(self.scaled, dtype=bool)

        if self.scaled.any():
            # FIX 4: compute variance only for columns marked for scaling
            col_vars = self.x[:, self.scaled].var(axis=0)
            constant_mask = (col_vars == 0)

            if constant_mask.any():
                # Identify which *original* column indices are constant
                scaled_indices = np.where(self.scaled)[0]
                constant_indices = scaled_indices[constant_mask]
                # Disable scaling only for those columns
                self.scaled[constant_indices] = False
                print(
                    "sigest: column(s) "
                    + ", ".join(str(i) for i in constant_indices)
                    + " are constant — scaling disabled for those column(s) only."
                )

            # Apply z-score scaling to the remaining columns to be scaled
            if self.scaled.any():
                self.x[:, self.scaled] = zscore(self.x[:, self.scaled], axis=0)

    def compute_distances(self) -> np.ndarray:
        """
        Compute paired distances between randomly drawn sample pairs.

        FIX 3: the Hilbert-space branch previously produced an (n, n)
                outer-product matrix and summed over axis=1, giving an
                incorrect aggregation.  The correct formula for a single
                pair (i, j) is K(i,i) + K(j,j) - 2K(i,j), applied
                element-wise to the index vectors.

        Returns
        -------
        dist : (n_samples,) array of non-negative pairwise distances.
        """
        rng = default_rng(self.seed)
        m   = self.x.shape[0]
        n   = max(1, int(np.floor(self.frac * m)))

        index  = rng.choice(m, size=n, replace=True)
        index2 = rng.choice(m, size=n, replace=True)

        if self.distance == "Euclidean":
            diff = self.x[index] - self.x[index2]          # (n, d)
            dist = np.sum(diff ** 2, axis=1)                # (n,)

        elif self.distance == "HilbertSpace":
            # FIX 3: paired indexing — one scalar distance per pair (i, j)
            K    = self.previousKernel
            diag = np.diag(K)
            dist = diag[index] + diag[index2] - 2.0 * K[index, index2]  # (n,)

        else:
            raise ValueError(
                f"Unknown distance metric '{self.distance}'. "
                "Use 'Euclidean' or 'HilbertSpace'."
            )

        return dist

    def sigest(self) -> np.ndarray:
        """
        Estimate the RBF sigma range from the data.

        Returns
        -------
        srange : ndarray of shape (3,)
            gamma = 1/sigma values at the [90th, 50th, 10th] percentiles
            of the pairwise distance distribution.
            srange[1] (the median) is the recommended default for SVM use.
        """
        if not isinstance(self.x, np.ndarray):
            raise TypeError("Input x must be a numpy ndarray.")

        self.process_input()
        dist   = self.compute_distances()

        nonzero = dist[dist > 0]
        if len(nonzero) == 0:
            raise ValueError(
                "All sampled pairwise distances are zero.  "
                "Check that the data has variance and frac > 0."
            )

        # 1/sigma at the [90th, 50th, 10th] distance percentiles
        srange = 1.0 / np.percentile(nonzero, q=[90, 50, 10])
        return srange
