# Sigma4 - Algothon 2026
# Consensus-sign full deployment over the Sigma3 signal book.
#
# Diagnosis behind it: the leaderboard's StdDev column is pinned at ~1610
# across teams whose means differ by 45% - that is the fingerprint of books
# running at the position caps on every name, every day. At that point the
# competition is a sign-accuracy contest at fixed size (score ~ mu once
# Sharpe > 4; mu of a sign book = (2p-1) x ~$9.7k of daily absolute moves).
#
# Construction: discover units exactly as Sigma3 (seed pairs + singles with
# recent-window kill switches; gray sleeve = all non-core pairs with
# t_full < -2.6 and recent-250 t < -2.0, overlap allowed). Each unit casts a
# bang-bang vote on its legs; each name deploys FULL +/-$10k in the sign of
# its net dollar vote; ALGO carries the book's return-beta hedge.
#
# Refinements tested and rejected on folds: EW/adaptive hedge ratios (the
# planted ratios are constant - adaptivity is pure noise), proportional
# votes and proportional deployment (sign is the information under linear
# OU), abstention thresholds (weak-vote names still carry >50% signs).
#
# Evidence: beats Sigma3 on ALL 7 walk-forward folds (min +9, mean +50)
# and all 4 holdout windows. Official-window score ~498 (Sigma3 488.7,
# Sigma2 419.8, Sigma1 374.6, Sigma0 326.0).
#
# numpy only.

import numpy as np

LEG=10_000
BAND=0.25; SBAND=0.25; GBANDV=0.20
RESCAN=20; WARMUP=120
T_SINGLE=-3.1; T_KEEP=-2.5; KEEPR=-1.0; MAX_SNG=6
WIN=350; EWMA_HL=20.0
GLO=-2.6; GREC=-2.0
MODE=0      # 0 bang-bang votes, 1 proportional-z votes
TH=0.0        # abstention threshold (x LEG) on net name vote
EWH=0        # gray hedge-ratio EW halflife (0=flat full-window OLS)
WSEED=1.0  # vote multiplier for seed pairs + singles
PART=0      # 0 full-or-nothing deployment, 1 proportional-capped

SEED_PAIRS=[(10,46),(1,20),(13,45),(8,27),(49,50),(25,37),(7,40)]
SEED_SINGLES=[35,33,18,44]
state={"scanned":-1_000_000,"units":[],"gunits":[],"rbeta":None}

def ar1_t(s):
    s=s-s.mean(axis=1,keepdims=True)
    num=np.einsum("ij,ij->i",s[:,1:],s[:,:-1])
    den=np.einsum("ij,ij->i",s[:,:-1],s[:,:-1])
    phi=num/np.maximum(den,1e-30)
    se=np.sqrt(np.maximum(1.0-phi*phi,1e-12)/(s.shape[1]-1))
    return phi,(phi-1.0)/se

def rec_t(x,w=250):
    x=x[-min(w,len(x)):]
    return ar1_t(x[None,:])[1][0]

def residualize(logp):
    Lc=logp-logp.mean(axis=1,keepdims=True)
    x0=Lc[0]; bl=(Lc@x0)/(x0@x0)
    Z=Lc-np.outer(bl,x0); Z[0]=0.0
    return Z,bl

def scan_gray(logp,n,claimed):
    L=logp-logp.mean(axis=1,keepdims=True); out=[]
    for i in range(1,n-1):
        if i in claimed: continue
        x=L[i]; xx=max(x@x,1e-30)
        for j in range(i+1,n):
            if j in claimed: continue
            b=(L[j]@x)/xx; R=L[j]-b*x
            _,t=ar1_t(R[None,:])
            if t[0]<GLO and abs(b)<4 and rec_t(R)<GREC:
                out.append((t[0],j,i))
    out.sort(); return out

def zstd(s):
    ds=np.diff(s)
    if len(ds)==0: return max(s.std(),1e-12)
    a=1.0-0.5**(1.0/EWMA_HL); v=ds[0]*ds[0]
    for d in ds[1:]: v=(1.0-a)*v+a*d*d
    return max(np.sqrt(v/(2.0*0.15)),1e-12)

def wfit(y,x,H):
    # weighted (EW halflife H) or flat OLS of y on x with intercept
    T=len(y)
    if H<=0:
        b,c=np.polyfit(x,y,1); return b,c
    w=0.5**((T-1-np.arange(T))/H); sw=np.sqrt(w)
    X=np.vstack([x*sw, sw]).T
    beta,*_=np.linalg.lstsq(X, y*sw, rcond=None)
    return beta[0], beta[1]

def vote(z, band):
    if MODE==0:
        return 0.0 if abs(z)<=band else (-1.0 if z>0 else 1.0)
    zz = 0.0 if abs(z)<=band else z
    return -np.clip(zz,-2.0,2.0)/2.0

def getMyPosition(prcSoFar):
    n,nt=prcSoFar.shape
    if nt<WARMUP: return np.zeros(n,dtype=int)
    logp=np.log(prcSoFar)
    Z,bl=residualize(logp)
    if nt-state["scanned"]>=RESCAN:
        units,used=[],set()
        for i,j in SEED_PAIRS:
            if j<n:
                sp=(logp[i]-logp[j])[None,:]
                _,t=ar1_t(sp)
                if t[0]<T_KEEP and rec_t(sp[0])<KEEPR:
                    units.append(("P",i,j)); used.update((i,j))
        _,ts=ar1_t(Z[1:])
        cand=[(ts[i-1],i) for i in SEED_SINGLES
              if i<n and i not in used and ts[i-1]<T_KEEP and rec_t(Z[i])<KEEPR]
        cand+=[(ts[k],k+1) for k in np.where(ts<T_SINGLE)[0] if (k+1) not in used]
        cand.sort(); ns=0
        for t,i in cand:
            if i in used or ns>=MAX_SNG: continue
            units.append(("S",i)); used.add(i); ns+=1
        state["gunits"]=[(i,j) for t,i,j in scan_gray(logp,n,used)]
        state["scanned"]=nt; state["units"]=units
        r=np.diff(logp,axis=1); r=r-r.mean(axis=1,keepdims=True)
        state["rbeta"]=(r@r[0])/max(r[0]@r[0],1e-30)

    px=prcSoFar[:,-1]
    Wsl=slice(-WIN,None)
    v=np.zeros(n); hedge_dlr=0.0
    for u in state["units"]:
        if u[0]=="P":
            _,i,j=u
            b,c=np.polyfit(logp[j][Wsl],logp[i][Wsl],1)
            s=logp[i][Wsl]-b*logp[j][Wsl]-c
            f=vote((s[-1]-s.mean())/zstd(s), BAND)*WSEED
            v[i]+=f*LEG; v[j]-=f*b*LEG
        else:
            _,i=u
            zi=Z[i][Wsl]
            f=vote((zi[-1]-zi.mean())/zstd(zi), SBAND)*WSEED
            v[i]+=f*LEG
    for i,j in state["gunits"]:
        b,c=wfit(logp[i],logp[j],EWH)
        s=logp[i]-b*logp[j]-c
        f=vote((s[-1]-s.mean())/max(s.std(),1e-12), GBANDV)
        v[i]+=f*LEG; v[j]-=f*b*LEG

    tgt=np.zeros(n)
    for i in range(1,n):
        if abs(v[i])>=TH*LEG and v[i]!=0.0:
            frac=1.0 if PART==0 else min(abs(v[i])/LEG,1.0)
            tgt[i]=np.sign(v[i])*frac*10_000/px[i]
    rb=state["rbeta"]
    hedge_dlr=-float(np.sum(tgt[1:]*px[1:]*rb[1:]))
    tgt[0]=hedge_dlr/px[0]
    if not state["units"] and not state["gunits"]:
        sh=max(1,int(300/px[0])); tgt[0]=sh if nt%2==0 else -sh
    lim=np.full(n,10_000.0); lim[0]=100_000.0
    lim=(lim/px).astype(int)
    return np.clip(tgt,-lim,lim).astype(int)
