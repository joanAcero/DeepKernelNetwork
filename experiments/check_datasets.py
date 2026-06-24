"""
check_datasets.py — extreme-quick sanity check: do all Experiment 3 datasets LOAD?

No architecture, no baselines, no training. For each dataset it only:
  - loads it (or generates, for synthetics),
  - reports shape (n, d), number of classes, class balance, n-grid that the runner
    would use, and whether exact RBF will be feasible at the largest n.
Anything that fails to load is reported, not crashed on, so you see all problems at once.

Run BEFORE the full experiment to catch a broken OpenML id / sparse-ARFF / rename.

Usage:  python3 check_datasets.py
"""
from __future__ import annotations
import warnings, time
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
import pandas as pd

# --- mirror the runner's protocol constants so the reported grid matches exactly ---
N_CAP, N_FRACS, N_FLOOR, TEST_SIZE, RBF_MAX_N = 100_000, (0.4,0.7,1.0), 1_000, 10_000, 15_000

def n_grid_for(pool):
    usable=max(0,pool-TEST_SIZE); base=usable if usable<=N_CAP else N_CAP
    grid=sorted({int(base*f) for f in N_FRACS if int(base*f)>=N_FLOOR})
    return grid or [min(usable,N_CAP)]

def _openml(name=None, data_id=None, version=1):
    d=(fetch_openml(data_id=data_id,as_frame=True,parser="auto") if data_id
       else fetch_openml(name=name,version=version,as_frame=True,parser="auto"))
    Xdf=pd.get_dummies(d.data,drop_first=False)
    X=np.nan_to_num(Xdf.to_numpy(dtype=float)); y=LabelEncoder().fit_transform(d.target)
    return X,y

def make_spirals(n,turns=2.5,noise=0.05,seed=42):
    rng=np.random.default_rng(seed)
    t=np.sqrt(rng.uniform(0,1,(n,1)))*turns*2*np.pi
    a=np.hstack([-np.cos(t)*t,np.sin(t)*t])+rng.standard_normal((n,2))*noise
    b=np.hstack([np.cos(t)*t,-np.sin(t)*t])+rng.standard_normal((n,2))*noise
    X=np.vstack([a,b]); y=np.hstack([np.zeros(n),np.ones(n)]).astype(int)
    i=rng.permutation(len(y)); return X[i],y[i]

def make_checkerboard(n,k=4,seed=42):
    rng=np.random.default_rng(seed); X=rng.uniform(0,k,(n,2))
    y=((np.floor(X[:,0])+np.floor(X[:,1]))%2).astype(int); return X,y

DATASETS={
    "poker":       lambda: _openml(name="poker-hand", version=1),
    "covertype":   lambda: _openml(data_id=1596),
    "chess_krk":   lambda: _openml(data_id=1481),
    "spirals":     lambda: make_spirals(60000),
    "checkerboard":lambda: make_checkerboard(120000),
    "miniboone":   lambda: _openml(name="MiniBooNE", version=1),
    "higgs":       lambda: _openml(name="higgs", version=1),
}

def main():
    print(f"{'dataset':12s} {'status':>7s} {'n':>8s} {'d':>4s} {'K':>3s} "
          f"{'minclass%':>9s} {'n-grid':>22s} {'RBF@maxn':>9s}  {'t(s)':>5s}")
    print("-"*92)
    ok=0
    for name,loader in DATASETS.items():
        t0=time.perf_counter()
        try:
            X,y=loader(); t=time.perf_counter()-t0
            n,d=X.shape; K=len(np.unique(y))
            _,cnts=np.unique(y,return_counts=True); minpct=100*cnts.min()/n
            grid=n_grid_for(n); maxn=max(grid)
            rbf="exact" if maxn<=RBF_MAX_N else "RFF-only"
            print(f"{name:12s} {'OK':>7s} {n:8d} {d:4d} {K:3d} {minpct:8.1f}% "
                  f"{str(grid):>22s} {rbf:>9s}  {t:5.1f}")
            ok+=1
        except Exception as e:
            print(f"{name:12s} {'FAIL':>7s}  -> {str(e)[:55]}")
    print("-"*92)
    print(f"{ok}/{len(DATASETS)} datasets loaded OK.")
    print("Check: K matches expectation (binary=2), no class is ~0% (degenerate split risk),")
    print("and n-grid has 3 points. RFF-only at max n means exact RBF won't run there (expected).")

if __name__=="__main__":
    main()
