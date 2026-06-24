"""
diagnose_poker.py — why does depth not help poker? Two competing mechanisms, one quick fit.

Distinguishes:
  (A) COLLAPSE: the m W-columns point the same way (rank-1 projection) -> depth idle.
                signature: low stable-rank of X_next, high mean|cos| of W columns,
                train AND test both flat with depth.
  (B) OVERFIT:  depth adds capacity that fits train but not test.
                signature: train acc/F1 RISE with L while test stays flat/falls;
                widening train-test gap. (This is what the CSV already suggests.)

Fits ONE configuration (m, L sweep at a single n, single seed) and prints, per layer:
  stable rank of the inter-layer representation, mean|cos| of the W columns,
  and TRAIN vs TEST accuracy and macro-F1. ~1-2 min, no grid.

Usage:  python3 diagnose_poker.py
"""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import fetch_openml
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mlsvm_extensions import DiverseMLMSVM, _InputSubspaceBlock

N      = 40000
M      = 50
L_MAX  = 5
SEED   = 0
P      = 1000

def load_poker():
    d = fetch_openml(name="covertype", version=1, as_frame=True, parser="auto")
    X = np.nan_to_num(pd.get_dummies(d.data, drop_first=False).to_numpy(dtype=float))
    y = LabelEncoder().fit_transform(d.target)
    return X, y

def stable_rank(A):
    A = A - A.mean(0); s = np.linalg.svd(A, compute_uv=False)
    return float((s**2).sum()/(s[0]**2)) if s[0] > 0 else 0.0

def mean_abs_cos(W):
    V = W.T; nrm = np.linalg.norm(V, axis=1, keepdims=True); nrm[nrm==0]=1
    G = (V/nrm) @ (V/nrm).T; k = G.shape[0]
    return (np.abs(G).sum()-k)/(k*(k-1)) if k > 1 else 1.0

def layer_reps(model, X):
    clf = model.named_steps["clf"]; sc = model.named_steps["scaler"]
    Xc = sc.transform(X); reps=[Xc.copy()]; Ws=[]
    for b in clf.blocks_:
        if isinstance(b,_InputSubspaceBlock):
            pj=[]
            for (cols,Om,bb,Wj) in b.sub_models:
                Phi=clf._feature_map(Xc[:,cols],Om,bb,kernel=b.kernel,arc_cosine_degree=b.arc_cosine_degree)
                pj.append(Phi@Wj)
            Xn=np.hstack(pj); Ws.append(np.hstack([w for (_,_,_,w) in b.sub_models]))
        else:
            Phi=clf._feature_map(Xc,b.Omega,b.b,kernel=b.kernel,arc_cosine_degree=b.arc_cosine_degree)
            Xn=Phi@b.W; Ws.append(b.W)
        if getattr(b,"scaler",None) is not None: Xn=b.scaler.transform(Xn)
        reps.append(Xn); Xc=Xn
    return reps, Ws

def main():
    print("Loading poker..."); X,y=load_poker()
    ts=10000
    rng=np.random.RandomState(7000+SEED); idx=rng.permutation(len(y))[:N+ts]
    Xtr,Xte,ytr,yte=train_test_split(X[idx],y[idx],test_size=ts,random_state=SEED,stratify=y[idx])
    feeding="disjoint"   # d=10 -> instance partitioning, matches the runner
    print(f"poker n={N} m={M} feeding={feeding}\n")
    print(f"{'L':>2s} | {'tr_acc':>6s} {'te_acc':>6s} {'gap':>6s} | {'tr_f1':>6s} {'te_f1':>6s} "
          f"| {'meanW|cos|':>10s} {'stab_rank':>9s}  verdict")
    print("-"*82)
    for L in range(1,L_MAX+1):
        m=Pipeline([("scaler",StandardScaler()),
            ("clf",DiverseMLMSVM(num_layers=L,svms_per_block=M,rff_features=P,
                kernel="arc_cosine",arc_cosine_degree=1,diversity_mode=feeding,
                block_C=10.0,final_C=1.0,random_state=SEED,normalize_inter_layer=True))]).fit(Xtr,ytr)
        tr=accuracy_score(ytr,m.predict(Xtr)); te=accuracy_score(yte,m.predict(Xte))
        trf=f1_score(ytr,m.predict(Xtr),average="macro"); tef=f1_score(yte,m.predict(Xte),average="macro")
        reps,Ws=layer_reps(m,Xtr)
        last_cos=mean_abs_cos(Ws[-1]); last_rank=stable_rank(reps[-1])
        gap=tr-te
        verdict = "OVERFIT" if (gap>0.03 and L>1) else ("COLLAPSE" if last_cos>0.9 else "")
        print(f"{L:2d} | {tr:6.3f} {te:6.3f} {gap:+6.3f} | {trf:6.3f} {tef:6.3f} "
              f"| {last_cos:10.3f} {last_rank:9.2f}  {verdict}")
    print("\nRead: if train-test GAP widens with L and meanW|cos| is LOW -> OVERFIT (capacity, not collapse).")
    print("      if meanW|cos|~1 and stable rank~1 with flat train too -> COLLAPSE (depth idle).")

if __name__=="__main__":
    main()
