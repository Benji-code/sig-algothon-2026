# Sigma5 - Algothon 2026
# Two orthogonal alpha families blended in one consensus-sign book:
#
#   1. REVERSION (Sigma4 core): planted-pair seeds + ALGO-residual singles +
#      gray sleeve (all plausibly-cointegrated pairs, overlap allowed), each
#      unit casting bang-bang votes on its legs.
#   2. LEAD-LAG: L[i,j] = corr(resid_i[t], resid_j[t+1]) on beta-hedged
#      normalised residual returns, refit each rescan on all history.
#      Per-entry correlations are individually insignificant (whole-matrix
#      split-half stability +0.11) but aggregate through breadth:
#      signal_j = sum_i L[i,j] z_i(t), walk-forward +9.2bp/name/day (t=5.9).
#      Sign agreement with the reversion book: 0.507 - orthogonal.
#
# Votes sum in dollars (lead-lag weight 1.5x); each name deploys full
# +/-$10k in the sign of its net vote; ALGO carries the return-beta hedge.
#
# Tested and rejected: hysteresis bands (live-validated gain for a pure
# lead-lag book elsewhere; measurably negative on this blend - the
# reversion votes already carry band-level stickiness, and position-level
# holds delay reversion flips). Zero-vote policy immaterial.
#
# Evidence vs Sigma4: all 7 walk-forward folds positive (min +49, mean
# +74); all 4 holdout windows positive (min +57, mean +92; holdout scores
# 573-630). Official window ~557 (Sigma4 497.9, Sigma3 488.7, Sigma2
# 419.8, Sigma1 374.6, Sigma0 326.0).
#
# numpy only.

import numpy as np

LEG=10_000
BAND=0.25; SBAND=0.25; GBANDV=0.20
RESCAN=20; WARMUP=120
T_SINGLE=-3.1; T_KEEP=-2.5; KEEPR=-1.0; MAX_SNG=6
WIN=350; EWMA_HL=20.0; GLO=-2.6; GREC=-2.0
LLW=1.5      # lead-lag vote weight (x LEG)
HYST=0.0    # hysteresis: opposing vote must exceed HYST x LEG to flip sign
REV=1
ZFLAT=0   # zero net vote -> flat (1) or hold prev sign (0)        # reversion votes on/off (ablation)

SEED_PAIRS=[(10,46),(1,20),(13,45),(8,27),(49,50),(25,37),(7,40)]
SEED_SINGLES=[35,33,18,44]
state={"scanned":-1_000_000,"units":[],"gunits":[],"rbeta":None,"L":None,"prev":None}

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

def resid_z(logp):
    r=np.diff(logp,axis=1); rc=r-r.mean(1,keepdims=True)
    beta=(rc@rc[0])/max(rc[0]@rc[0],1e-30)
    res=rc-np.outer(beta,rc[0]); res[0]=0.0
    sd=res.std(1,keepdims=True); sd[0]=1.0
    return res/np.maximum(sd,1e-12)

def getMyPosition(prcSoFar):
    n,nt=prcSoFar.shape
    if state["prev"] is None: state["prev"]=np.zeros(n)
    if nt<WARMUP: return np.zeros(n,dtype=int)
    logp=np.log(prcSoFar)
    Z,bl=residualize(logp)
    zres=resid_z(logp)

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
        a=zres[:,:-1]; b=zres[:,1:]
        Lm=(a@b.T)/a.shape[1]; np.fill_diagonal(Lm,0.0); Lm[0,:]=0.0; Lm[:,0]=0.0
        state["L"]=Lm
        state["scanned"]=nt; state["units"]=units
        r=np.diff(logp,axis=1); r=r-r.mean(axis=1,keepdims=True)
        state["rbeta"]=(r@r[0])/max(r[0]@r[0],1e-30)

    px=prcSoFar[:,-1]
    Wsl=slice(-WIN,None)
    v=np.zeros(n)
    if REV:
        for u in state["units"]:
            if u[0]=="P":
                _,i,j=u
                b,c=np.polyfit(logp[j][Wsl],logp[i][Wsl],1)
                s=logp[i][Wsl]-b*logp[j][Wsl]-c
                z=(s[-1]-s.mean())/zstd(s)
                f=0.0 if abs(z)<=BAND else (-1.0 if z>0 else 1.0)
                v[i]+=f*LEG; v[j]-=f*b*LEG
            else:
                _,i=u
                zi=Z[i][Wsl]
                z=(zi[-1]-zi.mean())/zstd(zi)
                f=0.0 if abs(z)<=SBAND else (-1.0 if z>0 else 1.0)
                v[i]+=f*LEG
        for i,j in state["gunits"]:
            b,c=np.polyfit(logp[j],logp[i],1)
            s=logp[i]-b*logp[j]-c
            z=(s[-1]-s.mean())/max(s.std(),1e-12)
            f=0.0 if abs(z)<=GBANDV else (-1.0 if z>0 else 1.0)
            v[i]+=f*LEG; v[j]-=f*b*LEG
    if LLW>0:
        sig=zres[:,-1] @ state["L"]
        ssd=sig[1:].std()
        if ssd>1e-12:
            sn=np.clip(sig/ssd,-2,2)/2.0
            v[1:]+=LLW*LEG*sn[1:]

    prev=state["prev"]
    tgt=np.zeros(n)
    for i in range(1,n):
        s_new=np.sign(v[i])
        s_prev=np.sign(prev[i])
        if s_new==0:
            s=0.0 if ZFLAT else s_prev
        elif s_prev!=0 and s_new!=s_prev and abs(v[i])<HYST*LEG:
            s=s_prev
        else:
            s=s_new
        if s!=0: tgt[i]=s*10_000/px[i]
    rb=state["rbeta"]
    tgt[0]=-float(np.sum(tgt[1:]*px[1:]*rb[1:]))/px[0]
    if not state["units"] and not state["gunits"] and LLW==0:
        sh=max(1,int(300/px[0])); tgt[0]=sh if nt%2==0 else -sh
    lim=np.full(n,10_000.0); lim[0]=100_000.0
    lim=(lim/px).astype(int)
    out=np.clip(tgt,-lim,lim).astype(int)
    state["prev"]=out*px
    return out
