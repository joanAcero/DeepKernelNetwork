"""
screen_gap_datasets.py — find REAL datasets where MLP >> RBF AT LARGE n.

Rationale (revised): the MLP-vs-RBF gap is an ASYMPTOTIC property. At small n
both underfit and look similar; the gap EMERGES at large n, where the MLP has
enough data to exploit interaction structure a single RBF kernel cannot — and
that is exactly the regime where exact RBF is intractable. So this screen is
weighted toward large n, with one small-n exact-RBF point used only to validate
the RFF approximation.

Per dataset:
  - ANCHOR (small n, default 5k): exact RBF SVM  +  RFF-RBF  +  MLP.
      Purpose: confirm RFF-RBF ~ exact RBF here, so the RFF numbers at large n
      are trustworthy. Prints |exact - rff| as the approximation error.
  - SCALE  (large n: 30k, 60k, 100k, capped by pool): MLP + RFF-RBF only.
      Exact RBF is intractable here by design; RFF-RBF (Rahimi-Recht, O(n)) is
      the honest stand-in. The gap MLP - RFF-RBF at the LARGEST feasible n is
      the headroom number to rank on.

Selection rule (committed in advance):
  GOOD Experiment-3 dataset  <=>  gap at large n is clearly positive
                                  AND exact RBF is intractable at full n
                                  AND the small-n anchor showed RFF ~ exact
                                      (so the large-n gap is real, not approx error).

Usage:
  python3 screen_gap_datasets.py
  python3 screen_gap_datasets.py --only poker covertype higgs
  python3 screen_gap_datasets.py --rff-p 5000     # finer RFF if anchor disagrees
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

CANDIDATES = {


    # --- NEW: poker-like = deterministic/rule-based labels, low Bayes error, non-smooth boundary ---
    "chess_krk":  lambda: _openml(data_id=1481),                        # krkopt: King-Rook-King optimal depth, 28k, deterministic, multiclass
    "eeg":        lambda: _openml(name="EEG-eye-state", version=1),     # 15k, temporal non-smooth boundary
    "connect4":   lambda: _openml(data_id=40668),                       # dense ARFF version (v1 sparse-load failed); 67k, deterministic game outcome

    # --- confirmed real hits (poker's property: low-noise, non-smooth combinatorial boundary) ---
    "poker":      lambda: _openml(name="poker-hand", version=1),
    "covertype":  lambda: _openml(data_id=1596),
}

ANCHOR_N   = 10                    # small-n exact-RBF validation point
SCALE_NS   = [3000, 10000, 15000, 20000, 50000]  # large-n where the gap should emerge
TEST_SIZE  = 10000
N_SEEDS    = 2

def mlp_ceiling(Xtr, ytr, Xte, yte):
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
    gamma=1.0/(Xtr.shape[1]*Xtr.var()) if Xtr.var()>0 else 1.0   # sklearn 'scale'
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
    ap.add_argument("--only",nargs="*",default=None)
    ap.add_argument("--rff-p",type=int,default=4000)  # finer default; coarse run showed 1000 too low
    args=ap.parse_args(); P=args.rff_p
    names=args.only if args.only else list(CANDIDATES)

    out=open("results_gap_screen.csv","w",newline=""); cw=csv.writer(out)
    cw.writerow(["dataset","d","pool","phase","n","seed","mlp_acc","mlp_time",
                 "rbf_exact_acc","rbf_exact_time","rbf_rff_acc","rbf_rff_time",
                 "gap_pts","timestamp"])
    print(f"GAP SCREEN — does MLP>>RBF emerge at LARGE n?  (RFF P={P}; exact RBF only at anchor n={ANCHOR_N})")
    print(f"{'dataset':12s} {'d':>4s} {'pool':>8s} {'phase':>7s} {'n':>7s} | "
          f"{'MLP':>7s} {'RBFex':>7s} {'RBFrff':>7s} | {'gap':>7s}  note")
    print("-"*92)

    summary={}
    for name in names:
        try:
            X,y=CANDIDATES[name]()
        except Exception as e:
            print(f"{name:12s}  LOAD FAILED: {str(e)[:55]}"); continue
        d,pool=X.shape[1],len(y)
        anchor_err=np.nan; large_gap=-99.0; largest_n=0

        # ---- ANCHOR: small n, exact vs rff vs mlp ----
        ts=min(TEST_SIZE,max(int(pool*0.2),200))
        if ANCHOR_N<=pool-ts:
            ma,rea,rfa=[],[],[]
            for seed in range(N_SEEDS):
                Xtr,Xte,ytr,yte=split(X,y,ANCHOR_N,ts,seed)
                sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
                m_a,m_t=mlp_ceiling(Xtr,ytr,Xte,yte)
                e_a,e_t=rbf_exact(Xtr,ytr,Xte,yte)
                f_a,f_t=rbf_rff(Xtr,ytr,Xte,yte,seed,P)
                ma.append(m_a);rea.append(e_a);rfa.append(f_a)
                cw.writerow([name,d,pool,"anchor",ANCHOR_N,seed,round(m_a,4),round(m_t,2),
                             round(e_a,4),round(e_t,2),round(f_a,4),round(f_t,2),
                             round((m_a-e_a)*100,2),datetime.datetime.now().isoformat()])
                out.flush()
            anchor_err=abs(np.mean(rea)-np.mean(rfa))*100
            note=f"RFF~exact err={anchor_err:.1f}pt" + (" [RFF OK]" if anchor_err<1.5 else " [RFF COARSE!]")
            print(f"{name:12s} {d:4d} {pool:8d} {'anchor':>7s} {ANCHOR_N:7d} | "
                  f"{np.mean(ma):.3f} {np.mean(rea):.3f} {np.mean(rfa):.3f} | "
                  f"{(np.mean(ma)-np.mean(rea))*100:+6.1f}  {note}")

        # ---- SCALE: large n, mlp vs rff only ----
        for n in SCALE_NS:
            if n>pool-ts:
                continue
            ma,rfa=[],[]
            for seed in range(N_SEEDS):
                Xtr,Xte,ytr,yte=split(X,y,n,ts,seed)
                sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
                m_a,m_t=mlp_ceiling(Xtr,ytr,Xte,yte)
                f_a,f_t=rbf_rff(Xtr,ytr,Xte,yte,seed,P)
                ma.append(m_a);rfa.append(f_a)
                cw.writerow([name,d,pool,"scale",n,seed,round(m_a,4),round(m_t,2),
                             "","",round(f_a,4),round(f_t,2),
                             round((m_a-f_a)*100,2),datetime.datetime.now().isoformat()])
                out.flush()
            g=(np.mean(ma)-np.mean(rfa))*100
            if g>large_gap: large_gap=g; largest_n=n
            print(f"{name:12s} {d:4d} {pool:8d} {'scale':>7s} {n:7d} | "
                  f"{np.mean(ma):.3f} {'  -- ':>7s} {np.mean(rfa):.3f} | {g:+6.1f}")
        summary[name]=(large_gap,largest_n,anchor_err,pool)

    print("\n"+"="*78)
    print("RANKING by MLP-RBF gap at LARGE n (the regime the architecture targets):")
    print("="*78)
    for name,(g,nn,err,pool) in sorted(summary.items(),key=lambda kv:-kv[1][0]):
        rff_ok = (not np.isnan(err)) and err<1.5
        if g>=3.0 and rff_ok:
            verdict="GOOD candidate"
        elif g>=3.0 and not rff_ok:
            verdict="gap BUT RFF coarse — re-run --rff-p 5000 to confirm"
        else:
            verdict="no large-n gap (drop)"
        errtxt = f"{err:.1f}pt" if not np.isnan(err) else "n/a"
        print(f"  {name:12s} gap@n={nn:<6d} {g:+6.1f} pts | RFF-anchor-err {errtxt:>6s} -> {verdict}")
    print("\nRead order: (1) is anchor RFF~exact? (2) does the gap GROW with n? "
          "(3) is the large-n gap >~3pt? Keep datasets that pass all three.")
    print("Note: 'fashion' = flattened pixels, sanity only (conv bias owns its headroom).")
    out.close(); print("\nWrote results_gap_screen.csv")

if __name__=="__main__":
    main()
