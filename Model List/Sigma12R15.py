# Sigma12R15 - Algothon 2026 (pre-registered recency-blend, built on 1-500 only)
# Sigma12A + recency lead-lag blend lambda=0.15 rec=150 (tightest spread)
# Config: S6_GLO=-3.4 S6_GREC=-2.8 S6_GW=0.5 S6_R2=0.20 S6_MAXS=8 S6_LLREC=150 S6_LLLAM=0.15
# numpy only.

import numpy as np

LEG=10_000
BAND=0.25; SBAND=0.25; GBANDV=0.25
RESCAN=20; WARMUP=120
T_SINGLE=-3.1; T_KEEP=-2.5; KEEPR=-1.0; MAX_SNG=8
SMAG=0  # 1=size singles by |t|/norm instead of flat
DYNSW=1.0  # weight for dynamically-discovered singles (fixed stay 1.0)
LLREC=150   # recent-window length for lead-lag blend (0=off, expanding only)
LLLAM=0.15 # blend weight on recent LL matrix
WIN=450; EWMA_HL=20.0; GLO=-3.4; GREC=-2.8
LLW=1.5
NFAC=2        # residualization factors for LL (1 = ALGO only, 2 = +PC1)
RVW=0.0        # reversal stream weight (raw 20d)
RVK=20
PNC=1.0        # LL+reversal vote multiplier on reversion-core names
BANDIT=0
GTHR=0      # gray throttle: trailing causal-PnL window (0=off)
BIRTHW=150   # birth-catcher recent window (0=off; e.g. 150)
BGATE=-4.0 # birth gate on recent-window t
GW=0.5        # gray/survivor tier vote weight
UGATE=0    # per-unit causal gate: trailing own-PnL window (0=off)
UMAG=0      # unit gate: 0=binary(mean>0), 1=magnitude weight clip(.5+t/2,0,1.5)
LLG=0        # per-name LL gate: trailing LL sign-book PnL window (0=off)
RW=0.0        # learned-ridge vote weight (0=off)
SEEDW=1.0  # seed vote weight (or -1 = depth-weighted by |t|/5)
SEEDK=7    # keep only the K deepest seeds
HEDGE=1    # 1=ALGO factor hedge on, 0=off (free the inst-0 slot)
EXP=1.0     # flat exposure multiplier on non-ALGO book
R2MIN=0.20    # min regression R^2 for a gray pair to count (0=off)
HUB=0       # 1=allow instruments in multiple gray pairs (hub-aware)
HUBN=1     # 1=normalize a hub's votes by sqrt(pair count)
VT=0.0        # vol-target: trailing-40d book sd target in $ (0=off)
VTLO=0.6    # floor on the vol-target scale factor    # trailing window for arm weights (0 = off)

SEED_PAIRS=[(10,46),(1,20),(13,45),(8,27),(49,50),(25,37),(7,40)]
SEED_SINGLES=[35,33,18,44]
state={"scanned":-1_000_000,"units":[],"gunits":[],"rbeta":None,"L":None,
       "core":set(),"prev_signs":None,"prev_px":None,"apnl":{"R":[],"L":[],"V":[]},
       "gprev":None,"gpx":None,"gpnl":[],"bunits":[],"uh":{},"uprev":{},"llh":{},"llprev":None,"llpx":None,"FX":[],"FY":[],"Fprev":None,"W":None,"bkprev":None,"bkpx":None,"bkpnl":[]}

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

def scan_gray(logp,n,claimed,hub=0):
    L=logp-logp.mean(axis=1,keepdims=True); out=[]
    for i in range(1,n-1):
        if not hub and i in claimed: continue
        x=L[i]; xx=max(x@x,1e-30)
        for j in range(i+1,n):
            if not hub and j in claimed: continue
            if hub and (i in claimed and j in claimed): continue  # don't re-pair two seed legs
            b=(L[j]@x)/xx; R=L[j]-b*x
            if R2MIN>0:
                yv=L[j]@L[j]
                r2=1.0-(R@R)/max(yv,1e-30)
                if r2<R2MIN: continue
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
    if nt<WARMUP: return np.zeros(n,dtype=int)
    logp=np.log(prcSoFar)
    Z,bl=residualize(logp)
    zres,rc=resid_z(logp)

    if nt-state["scanned"]>=RESCAN:
        units,used=[],set()
        scand=[]
        for i,j in SEED_PAIRS:
            if j<n:
                sp=(logp[i]-logp[j])[None,:]
                _,t=ar1_t(sp)
                if t[0]<T_KEEP and rec_t(sp[0])<KEEPR:
                    scand.append((t[0],i,j))
        scand.sort()
        for t0s,i,j in scand[:SEEDK]:
            wgt = SEEDW if SEEDW>0 else min(abs(t0s)/5.0,2.0)
            units.append(("P",i,j,wgt)); used.update((i,j))
        _,ts=ar1_t(Z[1:])
        fixset=set(SEED_SINGLES)
        cand=[(ts[i-1],i) for i in SEED_SINGLES
              if i<n and i not in used and ts[i-1]<T_KEEP and rec_t(Z[i])<KEEPR]
        cand+=[(ts[k],k+1) for k in np.where(ts<T_SINGLE)[0] if (k+1) not in used]
        cand.sort(); ns=0
        for t,i in cand:
            if i in used or ns>=MAX_SNG: continue
            src_w=1.0 if i in fixset else DYNSW
            units.append(("S",i,abs(t),src_w)); used.add(i); ns+=1
        state["gunits"]=[(i,j) for t,i,j in scan_gray(logp,n,used,hub=HUB)]
        state["bunits"]=[]
        if BIRTHW>0 and nt>BIRTHW+60:
            Lw=logp[:,-BIRTHW:]; Lc2=Lw-Lw.mean(1,keepdims=True)
            Lf=logp-logp.mean(1,keepdims=True)
            for i in range(1,n-1):
                if i in used: continue
                x=Lc2[i]; xx=max(x@x,1e-30)
                for j in range(i+1,n):
                    if j in used: continue
                    b=(Lc2[j]@x)/xx; R=Lc2[j]-b*x
                    _,tt=ar1_t(R[None,:])
                    if tt[0]<BGATE and abs(b)<4:
                        xf=Lf[i]; bf=(Lf[j]@xf)/max(xf@xf,1e-30)
                        _,tf=ar1_t((Lf[j]-bf*xf)[None,:])
                        if tf[0]>-2.6:
                            state["bunits"].append((j,i))
        a=zres[:,:-1]; b=zres[:,1:]
        Lm=(a@b.T)/a.shape[1]; np.fill_diagonal(Lm,0.0); Lm[0,:]=0.0; Lm[:,0]=0.0
        if LLREC>0 and LLLAM>0 and zres.shape[1]>LLREC+5:
            zr=zres[:,-(LLREC+1):]
            ar=zr[:,:-1]; br=zr[:,1:]
            Lrec=(ar@br.T)/ar.shape[1]; np.fill_diagonal(Lrec,0.0); Lrec[0,:]=0.0; Lrec[:,0]=0.0
            Lm=(1.0-LLLAM)*Lm+LLLAM*Lrec
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
            _,i,j=u[1],u[2],u[3] if False else u[1]
            i,j=u[1],u[2]; swgt=u[3] if len(u)>3 else 1.0
            b,c=np.polyfit(logp[j][Wsl],logp[i][Wsl],1)
            s=logp[i][Wsl]-b*logp[j][Wsl]-c
            z=(s[-1]-s.mean())/zstd(s)
            f=0.0 if abs(z)<=BAND else (-1.0 if z>0 else 1.0)
            vR[i]+=f*LEG; vR[j]-=f*b*LEG
        else:
            i=u[1]; tmag=u[2] if len(u)>2 else 3.1
            src_w=u[3] if len(u)>3 else 1.0
            zi=Z[i][Wsl]
            z=(zi[-1]-zi.mean())/zstd(zi)
            f=0.0 if abs(z)<=SBAND else (-1.0 if z>0 else 1.0)
            sw=(min(tmag/3.1,1.6) if SMAG else 1.0)*src_w
            vR[i]+=f*LEG*sw
    for i,j in state.get("bunits",[]):
        yw=logp[i,-BIRTHW:]; xw=logp[j,-BIRTHW:]
        b,c=np.polyfit(xw,yw,1)
        s=yw-b*xw-c
        z=(s[-1]-s.mean())/max(s.std(),1e-12)
        f=0.0 if abs(z)<=GBANDV else (-1.0 if z>0 else 1.0)
        vR[i]+=GW*f*LEG; vR[j]-=GW*f*b*LEG
    vG=np.zeros(n)
    if UGATE>0:
        r1=np.log(px/prcSoFar[:,-2]) if nt>1 else np.zeros(n)
        for key,(fp,bp) in list(state["uprev"].items()):
            i,j=key
            state["uh"].setdefault(key,[]).append(float(fp*LEG*(r1[i]-bp*r1[j])))
        state["uprev"]={}
    uzn={k:[] for k in range(n)}
    for i,j in state["gunits"]:
        b,c=np.polyfit(logp[j],logp[i],1)
        s=logp[i]-b*logp[j]-c
        z=(s[-1]-s.mean())/max(s.std(),1e-12)
        uzn[i].append(z); uzn[j].append(-z)
        f=0.0 if abs(z)<=GBANDV else (-1.0 if z>0 else 1.0)
        if UGATE>0:
            h=state["uh"].get((i,j),[])[-UGATE:]
            if len(h)<30: gate=1.0
            elif UMAG:
                tt=np.mean(h)/(np.std(h)/np.sqrt(len(h))+1e-9)
                gate=float(np.clip(0.5+tt/2.0,0.0,1.5))
            else:
                gate = 1.0 if np.mean(h)>0 else 0.0
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
    if RW>0:
        sigv = zres[:,-1] @ state["L"]
        F=np.zeros((n,7))
        F[:,0]=sigv
        for k in range(1,n):
            if uzn[k]:
                F[k,1]=uzn[k][int(np.argmax(np.abs(uzn[k])))]
                F[k,2]=np.mean(uzn[k])
        F[:,3]=zres[:,-1]
        F[:,4]=zres[:,-2] if zres.shape[1]>1 else 0
        F[:,5]=zres[:,-5:].sum(1); F[:,6]=zres[:,-20:].sum(1)
        if not state["FX"] and zres.shape[1]>80:
            L0=state["L"]; Q=zres.shape[1]
            gu=state["gunits"]
            UZH=np.zeros((len(gu), nt))
            for uix,(gi,gj) in enumerate(gu):
                b0,c0=np.polyfit(logp[gj],logp[gi],1)
                sser=logp[gi]-b0*logp[gj]-c0
                cm=np.cumsum(sser)/np.arange(1,nt+1)
                cv=np.sqrt(np.maximum(np.cumsum(sser*sser)/np.arange(1,nt+1)-cm*cm,1e-12))
                UZH[uix]=(sser-cm)/cv
            mb={k:[] for k in range(n)}
            for uix,(gi,gj) in enumerate(gu):
                mb[gi].append((uix,1.0)); mb[gj].append((uix,-1.0))
            for q in range(25, Q-1):
                Fh=np.zeros((n,7))
                Fh[:,0]=zres[:,q]@L0
                for k in range(1,n):
                    if mb[k]:
                        vals=[sg*UZH[u,q+1] for u,sg in mb[k]]
                        Fh[k,1]=vals[int(np.argmax(np.abs(vals)))]; Fh[k,2]=np.mean(vals)
                Fh[:,3]=zres[:,q]; Fh[:,4]=zres[:,q-1]
                Fh[:,5]=zres[:,q-4:q+1].sum(1); Fh[:,6]=zres[:,q-19:q+1].sum(1)
                state["FX"].append(Fh[1:]); state["FY"].append(zres[1:,q+1].copy())
        if state["Fprev"] is not None:
            state["FX"].append(state["Fprev"]); state["FY"].append(zres[1:,-1].copy())
        state["Fprev"]=F[1:].copy()
        if (state["W"] is None or nt==state["scanned"]) and len(state["FX"])>=150:
            X=np.vstack(state["FX"]); Y=np.concatenate(state["FY"])
            A=X.T@X+100.0*np.eye(7)
            state["W"]=np.linalg.solve(A, X.T@Y)
        if state["W"] is not None:
            pr=F@state["W"]
            psd=pr[1:].std()
            if psd>1e-12:
                v[1:]+=RW*LEG*(np.clip(pr/psd,-2,2)/2.0)[1:]
    prev=np.zeros(n)
    tgt=np.zeros(n)
    for i in range(1,n):
        s=np.sign(v[i])
        if s!=0: tgt[i]=s*10_000/px[i]
    if EXP!=1.0:
        tgt[1:]*=EXP
    rb=state["rbeta"]
    if HEDGE:
        tgt[0]=-float(np.sum(tgt[1:]*px[1:]*rb[1:]))/px[0]
    else:
        tgt[0]=0.0
    if not state["units"] and not state["gunits"]:
        sh=max(1,int(300/px[0])); tgt[0]=sh if nt%2==0 else -sh
    scale=1.0
    if VT>0:
        if state["bkprev"] is not None:
            ret=px/state["bkpx"]-1.0
            state["bkpnl"].append(float(np.sum(state["bkprev"]*state["bkpx"]*ret)))
        h=state["bkpnl"][-40:]
        if len(h)>=30:
            rsd=np.std(h)
            if rsd>VT: scale=max(VT/rsd, VTLO)
    lim=np.full(n,10_000.0); lim[0]=100_000.0
    lim=(lim/px).astype(int)
    out=np.clip(tgt*scale,-lim,lim).astype(int)
    if VT>0:
        state["bkprev"]=out.astype(float); state["bkpx"]=px.copy()
        return out
    return np.clip(tgt,-lim,lim).astype(int)
