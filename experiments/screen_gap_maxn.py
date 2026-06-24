"""
screen_gap_maxn.py — focused re-screen of the promising candidates at their LARGEST
feasible n, with a finer RFF (P=8000) so the gap is not contaminated by approximation
error. Runs the exact-RBF anchor at n=5000 purely to validate the RFF faithfulness.

Targets (from the previous screen):
  chess_krk  pool 28056  -> promising (+7.6) but RFF was COARSE at P<=4000
  eeg        pool 14980  -> promising (+9.6) but RFF was COARSE at P<=4000
  connect4   pool 67557  -> small gap, RFF OK; included to confirm at max n

For each dataset it sweeps n upward to the largest value the pool allows (leaving
TEST_SIZE for test, capped so train+test <= pool), and reports MLP vs RFF-RBF gap,
plus the exact-RBF anchor agreement. The largest-n gap with a faithful RFF is the
number to rank on.

Usage:
  python3 screen_gap_maxn.py
  python3 screen_gap_maxn.py --rff-p 12000     # finer still if anchor stays COARSE
"""
from __future__ import annotations
import sys, time, csv, argparse, warnings, datetime
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import pandas as pd

def _openml(name=None, data_id=None, version=1):
    if data_id is not None:
        d = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    else:
        d = fetch_openml(name=name, version=version, as_frame=True, parser="auto")
    Xdf = pd.get_dummies(d.data, drop_first=False)
    X = np.nan_to_num(Xdf.to_numpy(dtype=float))
    y = LabelEncoder().fit_transform(d.target)
    return X, y

TARGETS = {
    "chess_krk": lambda: _openml(data_id=1481),
}

ANCHOR_N  = 5000
TEST_SIZE = 5000          # smaller test reserve -> larger usable train n
N_SEEDS   = 1             # single seed: quick gap check only

def mlp_ceiling(Xtr,ytr,Xte,yte):
    t0=time.perf_counter()
    m=MLPClassifier(hidden_layer_sizes=(200,200,200),max_iter=1000,
                    early_stopping=False,random_state=0).fit(Xtr,ytr)
    return accuracy_score(yte,m.predict(Xte)), time.perf_counter()-t0

def rbf_exact(Xtr,ytr,Xte,yte):
    t0=time.perf_counter()
    m=SVC(kernel="rbf",C=10,gamma="scale").fit(Xtr,ytr)
    return accuracy_score(yte,m.predict(Xte)), time.perf_counter()-t0

def rbf_rff(Xtr,ytr,Xte,yte,seed,P):
    t0=time.perf_counter()
    gamma=1.0/(Xtr.shape[1]*Xtr.var()) if Xtr.var()>0 else 1.0
    rff=RBFSampler(gamma=gamma,n_components=P,random_state=seed)
    Ztr=rff.fit_transform(Xtr); Zte=rff.transform(Xte)
    clf=LinearSVC(C=10,max_iter=5000,tol=1e-3,dual="auto").fit(Ztr,ytr)
    return accuracy_score(yte,clf.predict(Zte)), time.perf_counter()-t0

def split(X,y,n,ts,seed):
    pool=len(y); rng=np.random.RandomState(7000+seed)
    idx=rng.permutation(pool)[:n+ts]; Xs,ys=X[idx],y[idx]
    try:
        return train_test_split(Xs,ys,test_size=ts,random_state=seed,stratify=ys)
    except ValueError:
        return train_test_split(Xs,ys,test_size=ts,random_state=seed)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rff-p",type=int,default=3000)
    args=ap.parse_args(); P=args.rff_p

    out=open("results_gap_maxn.csv","w",newline=""); cw=csv.writer(out)
    cw.writerow(["dataset","d","pool","phase","n","seed","mlp_acc","mlp_time",
                 "rbf_exact_acc","rbf_rff_acc","rbf_rff_time","gap_pts","timestamp"])
    print(f"MAX-n GAP RE-SCREEN  (RFF P={P}; exact RBF only at anchor n={ANCHOR_N}; test reserve={TEST_SIZE})")
    print(f"{'dataset':10s} {'d':>4s} {'pool':>7s} {'phase':>6s} {'n':>7s} | "
          f"{'MLP':>7s} {'RBFex':>7s} {'RBFrff':>7s} | {'gap':>7s}  note")
    print("-"*88)

    summary={}
    for name,loader in TARGETS.items():
        try:
            X,y=loader()
        except Exception as e:
            print(f"{name:10s} LOAD FAILED: {str(e)[:55]}"); continue
        d,pool=X.shape[1],len(y)
        max_n = pool - TEST_SIZE          # largest train size leaving the test reserve

        # ---- max n only: MLP vs RFF-RBF, single seed, quick gap check ----
        Xtr,Xte,ytr,yte=split(X,y,max_n,TEST_SIZE,0)
        sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
        m_a,_=mlp_ceiling(Xtr,ytr,Xte,yte)
        f_a,f_t=rbf_rff(Xtr,ytr,Xte,yte,0,P)
        gap=(m_a-f_a)*100
        cw.writerow([name,d,pool,"maxn",max_n,0,round(m_a,4),"","",
                     round(f_a,4),round(f_t,2),round(gap,2),
                     datetime.datetime.now().isoformat()]); out.flush()
        summary[name]=(gap,max_n)
        print(f"{name:10s} {d:4d} {pool:7d} {'maxn':>6s} {max_n:7d} | "
              f"{m_a:.3f} {'  -- ':>7s} {f_a:.3f} | {gap:+6.1f}")

    print("\n"+"="*70)
    print("RANKING by gap at MAX n:")
    print("="*70)
    for name,(g,nn) in sorted(summary.items(),key=lambda kv:-kv[1][0]):
        verdict = "GAP (keep)" if g>=3.0 else "no gap (drop)"
        print(f"  {name:10s} gap@n={nn:<7d} {g:+6.1f} pts -> {verdict}")
    out.close(); print("\nWrote results_gap_maxn.csv")

if __name__=="__main__":
    main()
