"""Zigma1 robust fallback — exact 75% factor-sleeve specification.

This file reproduces the model whose recorded days 501-750 score was 773.92.
It is deliberately less aggressive than Zigma1.py: when the independently
validated factor sleeve activates, it uses 75% directional ALGO exposure and
retains 25% of the relative-value beta hedge.
"""

import numpy as np

LEG=10_000
BAND=0.25; SBAND=0.25; GBANDV=0.25
RESCAN=20; WARMUP=120
T_SINGLE=-3.1; T_KEEP=-2.5; KEEPR=-1.0; MAX_SNG=0
WIN=450; EWMA_HL=20.0; GLO=-3.4; GREC=-2.8
LLW=1.5
NFAC=2
RVW=0.0
RVK=20
PNC=1.0
BANDIT=0
GTHR=0
BIRTHW=150
BGATE=-4.0
GW=0.5

FAC_LAGS=10
FAC_NCOMP=8
FAC_RESCAN=50
FAC_VALID=40
FAC_GATE_SR=1.0
FAC_BLEND=0.75

SEED_PAIRS=[(10,46),(1,20),(13,45),(8,27),(49,50),(25,37),(7,40)]
SEED_SINGLES=[]

def _new_state():
    return {"scanned":-1_000_000,"units":[],"gunits":[],"rbeta":None,"L":None,
       "core":set(),"prev_signs":None,"prev_px":None,"apnl":{"R":[],"L":[],"V":[]},
       "gprev":None,"gpx":None,"gpnl":[],"bunits":[],
       "fac_scanned":-1_000_000,"fac_model":None,"fac_active":False,
       "fac_val_sr":0.0,"last_nt":-1}
state=_new_state()

def reset_state():
    global state
    state=_new_state()

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
    if NFAC>=2:
        C=res[1:]@res[1:].T
        w=np.linalg.eigh(C)[1][:,-1]
        f=w@res[1:]
        b=(res@f)/max(f@f,1e-30)
        res=res-np.outer(b,f); res[0]=0.0
    sd=res.std(1,keepdims=True); sd[0]=1.0
    return res/np.maximum(sd,1e-12), rc

class _PLSModel:
    def __init__(self,n_components=8,max_iter=500,tol=1e-6):
        self.n_components=n_components; self.max_iter=max_iter; self.tol=tol

    def fit(self,X,Y):
        X=np.asarray(X,dtype=float); Y=np.asarray(Y,dtype=float)
        if Y.ndim==1: Y=Y[:,None]
        self.xmean=X.mean(0); self.ymean=Y.mean(0)
        self.xstd=X.std(0,ddof=1); self.ystd=Y.std(0,ddof=1)
        self.xstd[self.xstd<1e-12]=1.0; self.ystd[self.ystd<1e-12]=1.0
        Xk=(X-self.xmean)/self.xstd; Yk=(Y-self.ymean)/self.ystd
        n,p=Xk.shape; q=Yk.shape[1]
        nc=max(1,min(self.n_components,n-1,p))
        W=np.zeros((p,nc)); P=np.zeros((p,nc)); Q=np.zeros((q,nc))
        actual=0; eps=np.finfo(float).eps
        for a in range(nc):
            norms=np.sum(Yk*Yk,axis=0)
            u=Yk[:,int(np.argmax(norms))].copy()
            wold=np.zeros(p)
            for _ in range(self.max_iter):
                uu=float(u@u)
                if uu<eps: break
                w=(Xk.T@u)/uu
                nw=float(np.linalg.norm(w))
                if nw<eps: break
                w/=nw
                t=Xk@w; tt=float(t@t)
                if tt<eps: break
                c=(Yk.T@t)/tt; cc=float(c@c)
                if cc<eps: break
                unew=(Yk@c)/cc
                if float(np.sum((w-wold)**2))<self.tol:
                    u=unew; break
                wold=w.copy(); u=unew
            t=Xk@w; tt=float(t@t)
            if tt<eps: break
            pv=(Xk.T@t)/tt; qv=(Yk.T@t)/tt
            Xk-=np.outer(t,pv); Yk-=np.outer(t,qv)
            W[:,a]=w; P[:,a]=pv; Q[:,a]=qv; actual+=1
        W=W[:,:actual]; P=P[:,:actual]; Q=Q[:,:actual]
        self.coef=W@np.linalg.pinv(P.T@W)@Q.T
        return self

    def predict(self,X):
        X=np.asarray(X,dtype=float)
        return (((X-self.xmean)/self.xstd)@self.coef)*self.ystd+self.ymean

def _factor_xy(logp):
    r=np.diff(logp,axis=1).T
    L=FAC_LAGS
    if len(r)<L+2: return None,None,None
    X=np.stack([r[e-L+1:e+1].reshape(-1) for e in range(L-1,len(r)-1)])
    Y=np.stack([r[e+1] for e in range(L-1,len(r)-1)])
    latest=r[-L:].reshape(1,-1)
    return X,Y,latest

def _validation_sharpe(X,Y):
    v=FAC_VALID
    if len(X)<v+40: return -1e9
    model=_PLSModel(FAC_NCOMP).fit(X[:-v],Y[:-v])
    pred=model.predict(X[-v:])[:,0]
    signs=np.sign(pred); actual=Y[-v:,0]
    pnl=100_000.0*signs*actual
    pnl-=100_000.0*0.00002*np.abs(np.diff(np.r_[0.0,signs]))
    sd=float(pnl.std())
    return 0.0 if sd<1e-12 else float(np.sqrt(250.0)*pnl.mean()/sd)

def _update_factor(logp,nt):
    if nt-state["fac_scanned"]<FAC_RESCAN and state["fac_model"] is not None:
        return
    X,Y,_=_factor_xy(logp)
    if X is None or len(X)<FAC_VALID+40:
        state["fac_active"]=False; return
    sr=_validation_sharpe(X,Y)
    state["fac_val_sr"]=sr
    state["fac_active"]=bool(sr>FAC_GATE_SR)
    state["fac_model"]=_PLSModel(FAC_NCOMP).fit(X,Y)
    state["fac_scanned"]=nt

def _factor_sign(logp):
    model=state.get("fac_model")
    if model is None: return 0.0
    _,_,latest=_factor_xy(logp)
    if latest is None: return 0.0
    return float(np.sign(model.predict(latest)[0,0]))

def getMyPosition(prcSoFar):
    global state
    n,nt=prcSoFar.shape
    if nt<state.get("last_nt",-1): state=_new_state()
    state["last_nt"]=nt
    if nt<WARMUP: return np.zeros(n,dtype=int)
    logp=np.log(prcSoFar)
    _update_factor(logp,nt)
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
        state["L"]=Lm
        core=set(used)
        for i,j in state["gunits"]: core.update((i,j))
        state["core"]=core
        state["scanned"]=nt; state["units"]=units
        r=np.diff(logp,axis=1); r=r-r.mean(axis=1,keepdims=True)
        state["rbeta"]=(r@r[0])/max(r[0]@r[0],1e-30)

    px=prcSoFar[:,-1]
    Wsl=slice(-WIN,None)
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
        s=yw-b*xw-c
        z=(s[-1]-s.mean())/max(s.std(),1e-12)
        f=0.0 if abs(z)<=GBANDV else (-1.0 if z>0 else 1.0)
        vR[i]+=GW*f*LEG; vR[j]-=GW*f*b*LEG
    vG=np.zeros(n)
    for i,j in state["gunits"]:
        b,c=np.polyfit(logp[j],logp[i],1)
        s=logp[i]-b*logp[j]-c
        z=(s[-1]-s.mean())/max(s.std(),1e-12)
        f=0.0 if abs(z)<=GBANDV else (-1.0 if z>0 else 1.0)
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
    tgt=np.zeros(n)
    for i in range(1,n):
        s=np.sign(v[i])
        if s!=0: tgt[i]=s*10_000/px[i]
    rb=state["rbeta"]
    tgt[0]=-float(np.sum(tgt[1:]*px[1:]*rb[1:]))/px[0]
    if state.get("fac_active",False):
        fs=_factor_sign(logp)
        if fs!=0:
            hedge_dollars=tgt[0]*px[0]
            alpha_dollars=fs*100_000.0
            tgt[0]=((1.0-FAC_BLEND)*hedge_dollars+FAC_BLEND*alpha_dollars)/px[0]
    if not state["units"] and not state["gunits"]:
        sh=max(1,int(300/px[0])); tgt[0]=sh if nt%2==0 else -sh
    lim=np.full(n,10_000.0); lim[0]=100_000.0
    lim=(lim/px).astype(int)
    return np.clip(tgt,-lim,lim).astype(int)
