# Sigma6.1 - Algothon 2026
# Sigma6 (two-factor lead-lag + reversion consensus book) with two inherited
# constants re-tuned for the current chassis: core signal window WIN 350->450
# and gray vote band 0.20->0.25. Both were optimal on the Sigma2/Sigma3
# architecture and went stale through the consensus-deployment and lead-lag
# blend changes; the re-tune dominates Sigma6 on late folds (+16), holdout
# (+36, scores 611/653/681/689) AND the official window (645.9 vs 596.7)
# simultaneously - a plateau shift, not an in-sample cherry-pick. Broad
# plateau: WIN 400-500 x GBAND .25-.30 all dominate the old constants.
#
# Lineage: Sigma0 326 -> Sigma1 374.6 -> Sigma2 419.8 -> Sigma3 488.7 ->
# Sigma4 497.9 -> Sigma5 557.4 -> Sigma6 596.7 -> Sigma6.1 645.9.
#
# numpy only.

import numpy as np

LEG=10_000
BAND=0.25; SBAND=0.25; GBANDV=0.25
RESCAN=20; WARMUP=120
T_SINGLE=-3.1; T_KEEP=-2.5; KEEPR=-1.0; MAX_SNG=6
WIN=450; EWMA_HL=20.0; GLO=-2.6; GREC=-2.0
LLW=1.5
NFAC=2        # residualization factors for LL (1 = ALGO only, 2 = +PC1)
RVW=0.0        # reversal stream weight (raw 20d)
RVK=20
PNC=1.0        # LL+reversal vote multiplier on reversion-core names
BANDIT=0    # trailing window for arm weights (0 = off)
GTHR=0
BIRTHW=150
BGATE=-3.3
GW=0.0
UGATE=0

SEED_PAIRS=[(10,46),(1,20),(13,45),(8,27),(49,50),(25,37),(7,40)]
SEED_SINGLES=[35,33,18,44]
state={"scanned":-1_000_000,"units":[],"gunits":[],"rbeta":None,"L":None,
       "core":set(),"prev_signs":None,"prev_px":None,"apnl":{"R":[],"L":[],"V":[]},
       "last_nt":0,"n":None,"gprev":None,"gpx":None,"gpnl":[],"bunits":[],"uh":{},"uprev":{}}

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
    x0=Lc[0]
    den=max(float(x0@x0),1e-30)
    bl=(Lc@x0)/den
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
    if NFAC>=2:
        C=res[1:]@res[1:].T
        w=np.linalg.eigh(C)[1][:,-1]
        f=w@res[1:]
        b=(res@f)/max(f@f,1e-30)
        res=res-np.outer(b,f); res[0]=0.0
    sd=res.std(1,keepdims=True); sd[0]=1.0
    return res/np.maximum(sd,1e-12), rc

def getMyPosition(prcSoFar):
    n,nt=prcSoFar.shape
    if state.get("n") != n or nt <= state.get("last_nt",0):
        state["scanned"]=-1_000_000
        state["units"]=[]
        state["gunits"]=[]
        state["rbeta"]=None
        state["L"]=None
        state["core"]=set()
        state["prev_signs"]=None
        state["prev_px"]=None
        state["apnl"]={"R":[],"L":[],"V":[]}
        state["gprev"]=None
        state["gpx"]=None
        state["gpnl"]=[]
        state["bunits"]=[]
        state["uh"]={}
        state["uprev"]={}
        state["n"]=n
    state["last_nt"]=nt
    if nt<WARMUP: return np.zeros(n,dtype=int)
    logp=np.log(np.maximum(prcSoFar,1e-12))
    Z,bl=residualize(logp)
    zres,rc=resid_z(logp)

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
        state["bunits"]=[]
        if BIRTHW>0 and nt>BIRTHW+60:
            Lw=logp[:,-BIRTHW:]; Lc2=Lw-Lw.mean(1,keepdims=True)
            Lf=logp-logp.mean(1,keepdims=True)
            for i in range(1,n-1):
                if i in used: continue
                x=Lc2[i]; xx=max(float(x@x),1e-30)
                for j in range(i+1,n):
                    if j in used: continue
                    b0=(Lc2[j]@x)/xx; R=Lc2[j]-b0*x
                    _,tt=ar1_t(R[None,:])
                    if tt[0]<BGATE and abs(b0)<4:
                        xf=Lf[i]; bf=(Lf[j]@xf)/max(float(xf@xf),1e-30)
                        _,tf=ar1_t((Lf[j]-bf*xf)[None,:])
                        if tf[0]>-2.6:
                            state["bunits"].append((j,i))
        a=zres[:,:-1]; b=zres[:,1:]
        Lm=(a@b.T)/a.shape[1]; np.fill_diagonal(Lm,0.0); Lm[0,:]=0.0; Lm[:,0]=0.0
        state["L"]=Lm
        core=set(used)
        for i,j in state["gunits"]: core.update((i,j))
        state["core"]=core
        state["scanned"]=nt; state["units"]=units
        r=np.diff(logp,axis=1); r=r-r.mean(axis=1,keepdims=True)
        state["rbeta"]=(r@r[0])/max(r[0]@r[0],1e-30)

    px=prcSoFar[:,-1]
    Wsl=slice(-WIN,None)
    # --- arm votes ---
    vR=np.zeros(n); vL=np.zeros(n); vV=np.zeros(n)
    for u in state["units"]:
        if u[0]=="P":
            _,i,j=u
            b,c=np.polyfit(logp[j][Wsl],logp[i][Wsl],1)
            s=logp[i][Wsl]-b*logp[j][Wsl]-c
            z=(s[-1]-s.mean())/zstd(s)
            f=0.0 if abs(z)<=BAND else (-1.0 if z>0 else 1.0)
            vR[i]+=f*LEG; vR[j]-=f*b*LEG
        else:
            _,i=u
            zi=Z[i][Wsl]
            z=(zi[-1]-zi.mean())/zstd(zi)
            f=0.0 if abs(z)<=SBAND else (-1.0 if z>0 else 1.0)
            vR[i]+=f*LEG
    for i,j in state.get("bunits",[]):
        yw=logp[i,-BIRTHW:]; xw=logp[j,-BIRTHW:]
        b,c=np.polyfit(xw,yw,1)
        sp=yw-b*xw-c
        z=(sp[-1]-sp.mean())/max(float(sp.std()),1e-12)
        f=0.0 if abs(z)<=GBANDV else (-1.0 if z>0 else 1.0)
        vR[i]+=GW*f*LEG; vR[j]-=GW*f*b*LEG

    vG=np.zeros(n)
    if UGATE>0:
        r1=np.log(px/prcSoFar[:,-2]) if nt>1 else np.zeros(n)
        for key,(fp,bp) in list(state["uprev"].items()):
            i,j=key
            if i<n and j<n:
                state["uh"].setdefault(key,[]).append(float(fp*LEG*(r1[i]-bp*r1[j])))
        state["uprev"]={}
    for i,j in state["gunits"]:
        b,c=np.polyfit(logp[j],logp[i],1)
        sp=logp[i]-b*logp[j]-c
        z=(sp[-1]-sp.mean())/max(float(sp.std()),1e-12)
        f=0.0 if abs(z)<=GBANDV else (-1.0 if z>0 else 1.0)
        if UGATE>0:
            h=state["uh"].get((i,j),[])[-UGATE:]
            gate=1.0 if len(h)<30 else (1.0 if np.mean(h)>0 else 0.0)
            state["uprev"][(i,j)]=(f,b)
            f*=gate
        vG[i]+=f*LEG; vG[j]-=f*b*LEG
    gm=1.0
    if GTHR>0:
        if state["gprev"] is not None:
            ret=px/state["gpx"]-1.0
            state["gpnl"].append(float(10_000*np.sum(state["gprev"][1:]*ret[1:])))
        state["gprev"]=np.sign(vG); state["gpx"]=px.copy()
        h=state["gpnl"][-GTHR:]
        if len(h)>=40:
            m=np.mean(h); se=np.std(h)/np.sqrt(len(h))+1e-9
            gm=float(np.clip(m/(2*se),0.0,1.0))
    vR+=GW*gm*vG
    sig=zres[:,-1] @ state["L"]
    ssd=sig[1:].std()
    if ssd>1e-12:
        vL[1:]=LLW*LEG*(np.clip(sig/ssd,-2,2)/2.0)[1:]
    if RVW>0 and rc.shape[1]>=RVK:
        vV[1:]=RVW*LEG*np.sign(-rc[1:,-RVK:].mean(1))
    if PNC!=1.0:
        ci=np.array([i for i in state["core"] if i>0],dtype=int)
        if len(ci): vL[ci]*=PNC; vV[ci]*=PNC

    # --- bandit meta-layer: weight arms by trailing virtual sign-book PnL ---
    wts={"R":1.0,"L":1.0,"V":1.0}
    if BANDIT>0:
        if state["prev_signs"] is not None:
            ret=px/state["prev_px"]-1.0
            for k in ("R","L","V"):
                s_prev=state["prev_signs"][k]
                state["apnl"][k].append(float(10_000*np.sum(s_prev[1:]*ret[1:])))
        m={}
        for k in ("R","L","V"):
            h=state["apnl"][k][-BANDIT:]
            m[k]=np.mean(h) if len(h)>=20 else 0.0
        if any(v>0 for v in m.values()):
            tot=sum(max(v,0.0) for v in m.values())
            for k in m: wts[k]=3.0*max(m[k],0.0)/max(tot,1e-9)
        state["prev_signs"]={"R":np.sign(vR),"L":np.sign(vL),"V":np.sign(vV)}
        state["prev_px"]=px.copy()

    v=wts["R"]*vR+wts["L"]*vL+wts["V"]*vV
    prev=np.zeros(n)
    tgt=np.zeros(n)
    for i in range(1,n):
        s=np.sign(v[i])
        if s!=0: tgt[i]=s*10_000/px[i]
    rb=state["rbeta"]
    if rb is None:
        rb=np.zeros(n)

    # ALGO has the 10x limit and lowest commission. On the public generator it
    # mean-reverts strongly at very short horizons; using it as a direct 5-day
    # reversal profit centre beats treating it only as a beta hedge.
    if nt > 5:
        r5 = logp[0, -1] - logp[0, -6]
        s0 = -1.0 if r5 > 0 else (1.0 if r5 < 0 else 0.0)
        tgt[0] = s0 * 100_000.0 / px[0]
    else:
        tgt[0] = -float(np.sum(tgt[1:]*px[1:]*rb[1:]))/px[0]
    if not state["units"] and not state["gunits"]:
        sh=max(1,int(300/px[0])); tgt[0]=sh if nt%2==0 else -sh
    tgt=np.nan_to_num(tgt,nan=0.0,posinf=0.0,neginf=0.0)
    lim=np.full(n,10_000.0); lim[0]=100_000.0
    lim=(lim/px).astype(int)
    return np.clip(tgt,-lim,lim).astype(int)