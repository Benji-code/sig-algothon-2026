import numpy as np
LEG = 10_000
BAND = 0.25
SBAND = 0.25
GBANDV = 0.25
RESCAN = 20
WARMUP = 120
T_SINGLE = -3.1
T_KEEP = -2.5
KEEPR = -1.0
MAX_SNG = 0
WIN = 450
EWMA_HL = 20.0
GLO = -3.4
GREC = -2.8
LLW = 1.3
LL_ROLL = 450
LL_BLEND = 0.1
LL_THRESH = 0.04
NFAC = 2
BIRTHW = 150
BGATE = -4.0
GW = 0.5
ALGO_REV_HORIZON = 5
ALGO_REV_DOLLARS = 100_000.0
# --- rebuilt version of B14's core idea (SigmaB10 chassis unchanged below).
# Built in-house, in-sample only on days 1-500, after the received SigmaB14
# disclosed its own design had seen days 251-750 during selection (see Alpha
# Exploration Log). B14's actual mechanism -- a causal shadow-PnL selector
# switching between a raw and a relative ALGO-reversal arm -- was rebuilt and
# tested the same way, and REJECTED: every window/deadband combination
# scored worse than just using the relative arm alone (Sharpe 0.45-1.37 vs
# 2.13 for the fixed relative arm, days 120-499). What DID hold up, robustly,
# across all 6 beta_window x horizon combinations tested (no exceptions):
# trading ONLY when the raw and relative arms agree in direction, and going
# flat otherwise -- no selector, no switching, no shadow PnL bookkeeping.
ALGO_REL_HORIZON = 3
ALGO_REL_BETA_WINDOW = 60
ALGO_REL_ZLOOKBACK = 120
ALGO_REL_ZTHRESH = 0.5
SEED_PAIRS = [(10,46),(1,20),(13,45),(8,27),(49,50),(25,37),(7,40)]
SEED_SINGLES = []
def _new_state():
    return {"scanned":-1_000_000,"units":[],"gunits":[],"bunits":[],"rbeta":None,"L":None,"core":set(),"last_nt":-1}
state=_new_state()
def reset_state():
    global state; state=_new_state()
def ar1_t(s):
    s=s-s.mean(axis=1,keepdims=True)
    num=np.einsum("ij,ij->i",s[:,1:],s[:,:-1]);den=np.einsum("ij,ij->i",s[:,:-1],s[:,:-1])
    phi=num/np.maximum(den,1e-30);se=np.sqrt(np.maximum(1.0-phi*phi,1e-12)/max(s.shape[1]-1,1))
    return phi,(phi-1.0)/se
def rec_t(x,w=250):
    x=x[-min(w,len(x)):];return ar1_t(x[None,:])[1][0]
def residualize(logp):
    centred=logp-logp.mean(axis=1,keepdims=True);x0=centred[0];den=max(float(x0@x0),1e-30)
    beta=(centred@x0)/den;residual=centred-np.outer(beta,x0);residual[0]=0.0;return residual,beta
def scan_gray(logp,n,claimed):
    centred=logp-logp.mean(axis=1,keepdims=True);out=[]
    for i in range(1,n-1):
        if i in claimed: continue
        x=centred[i];xx=max(float(x@x),1e-30)
        for j in range(i+1,n):
            if j in claimed: continue
            beta=float(centred[j]@x)/xx;residual=centred[j]-beta*x;_,tstat=ar1_t(residual[None,:])
            if tstat[0]<GLO and abs(beta)<4 and rec_t(residual)<GREC: out.append((tstat[0],j,i))
    out.sort();return out
def zstd(s):
    ds=np.diff(s)
    if len(ds)==0: return max(float(s.std()),1e-12)
    alpha=1.0-0.5**(1.0/EWMA_HL);variance=ds[0]*ds[0]
    for d in ds[1:]: variance=(1.0-alpha)*variance+alpha*d*d
    return max(float(np.sqrt(variance/(2.0*0.15))),1e-12)
def resid_z(logp):
    returns=np.diff(logp,axis=1);centred_returns=returns-returns.mean(axis=1,keepdims=True)
    beta=centred_returns@centred_returns[0]/max(float(centred_returns[0]@centred_returns[0]),1e-30)
    residual=centred_returns-np.outer(beta,centred_returns[0]);residual[0]=0.0
    if NFAC>=2:
        covariance=residual[1:]@residual[1:].T;weights=np.linalg.eigh(covariance)[1][:,-1]
        factor=weights@residual[1:];loading=residual@factor/max(float(factor@factor),1e-30)
        residual=residual-np.outer(loading,factor);residual[0]=0.0
    sd=residual.std(axis=1,keepdims=True);sd[0]=1.0;return residual/np.maximum(sd,1e-12),centred_returns
def _algo_relative_signal(logp):
    """Causal: beta-residualize ALGO's return against the equal-weight market,
    fade the H-day cumulative residual, z-scored against trailing history."""
    all_ret=np.diff(logp,axis=1)
    if all_ret.shape[1] < ALGO_REL_BETA_WINDOW+ALGO_REL_ZLOOKBACK+ALGO_REL_HORIZON+5:
        return 0.0
    algo_ret=all_ret[0]; mkt_ret=all_ret[1:].mean(axis=0)
    w=min(ALGO_REL_BETA_WINDOW,len(algo_ret))
    yw=algo_ret[-w:];mw=mkt_ret[-w:]
    yc=yw-yw.mean();mc=mw-mw.mean()
    beta=float(yc@mc)/max(float(mc@mc),1e-30)
    residual=algo_ret-beta*mkt_ret
    csum=np.cumsum(np.r_[0.0,residual])
    rolling=csum[ALGO_REL_HORIZON:]-csum[:-ALGO_REL_HORIZON]
    if len(rolling)<2: return 0.0
    current=float(rolling[-1])
    history=rolling[:-1][-ALGO_REL_ZLOOKBACK:]
    if len(history)<20: return 0.0
    sd=max(float(history.std()),1e-12)
    z=(current-float(history.mean()))/sd
    return 0.0 if abs(z)<ALGO_REL_ZTHRESH else (-1.0 if z>0 else 1.0)
def getMyPosition(prcSoFar):
    global state
    prices=np.asarray(prcSoFar,dtype=float);n,nt=prices.shape
    if nt<=state["last_nt"]: state=_new_state()
    state["last_nt"]=nt
    if nt<WARMUP: return np.zeros(n,dtype=int)
    logp=np.log(np.maximum(prices,1e-12));level_residual,_=residualize(logp);zres,_=resid_z(logp)
    if nt-state["scanned"]>=RESCAN:
        units=[];used=set()
        for i,j in SEED_PAIRS:
            if i<n and j<n:
                spread=(logp[i]-logp[j])[None,:];_,tstat=ar1_t(spread)
                if tstat[0]<T_KEEP and rec_t(spread[0])<KEEPR: units.append(("P",i,j));used.update((i,j))
        _,single_t=ar1_t(level_residual[1:])
        candidates=[(single_t[k],k+1) for k in np.where(single_t<T_SINGLE)[0] if(k+1) not in used]
        candidates.sort();singles_added=0
        for _,i in candidates:
            if i in used or singles_added>=MAX_SNG: continue
            units.append(("S",i));used.add(i);singles_added+=1
        state["gunits"]=[(i,j) for _,i,j in scan_gray(logp,n,used)]
        state["bunits"]=[]
        if BIRTHW>0 and nt>BIRTHW+60:
            recent=logp[:,-BIRTHW:];recent_c=recent-recent.mean(axis=1,keepdims=True);full_c=logp-logp.mean(axis=1,keepdims=True)
            for i in range(1,n-1):
                if i in used: continue
                x=recent_c[i];xx=max(float(x@x),1e-30)
                for j in range(i+1,n):
                    if j in used: continue
                    beta_recent=float(recent_c[j]@x)/xx;residual_recent=recent_c[j]-beta_recent*x;_,recent_t=ar1_t(residual_recent[None,:])
                    if recent_t[0]<BGATE and abs(beta_recent)<4:
                        xf=full_c[i];beta_full=float(full_c[j]@xf)/max(float(xf@xf),1e-30)
                        residual_full=full_c[j]-beta_full*xf;_,full_t=ar1_t(residual_full[None,:])
                        if full_t[0]>-2.6: state["bunits"].append((j,i))
        a=zres[:,:-1];b=zres[:,1:]
        linear_full=(a@b.T)/max(a.shape[1],1);np.fill_diagonal(linear_full,0.0);linear_full[0,:]=0.0;linear_full[:,0]=0.0
        recent_width=min(LL_ROLL,zres.shape[1]-1);ar=zres[:,-(recent_width+1):-1];br=zres[:,-recent_width:]
        linear_recent=(ar@br.T/max(recent_width,1));np.fill_diagonal(linear_recent,0.0);linear_recent[0,:]=0.0;linear_recent[:,0]=0.0
        if LL_THRESH>0: linear_recent[np.abs(linear_recent)<LL_THRESH]=0.0
        state["L"]=(1.0-LL_BLEND)*linear_full+LL_BLEND*linear_recent
        state["units"]=units;state["scanned"]=nt
    px=prices[:,-1];window_slice=slice(-WIN,None)
    vR=np.zeros(n);vL=np.zeros(n)
    for unit in state["units"]:
        if unit[0]=="P":
            _,i,j=unit
            beta,intercept=np.polyfit(logp[j][window_slice],logp[i][window_slice],1)
            spread=logp[i][window_slice]-beta*logp[j][window_slice]-intercept
            spread_sd=zstd(spread);z=(float(spread[-1])-float(spread.mean()))/spread_sd
            vote=0.0 if abs(z)<=BAND else(-1.0 if z>0 else 1.0);vR[i]+=vote*LEG;vR[j]-=vote*beta*LEG
        else:
            _,i=unit;residual=level_residual[i][window_slice];residual_sd=zstd(residual)
            z=(float(residual[-1])-float(residual.mean()))/residual_sd
            vote=0.0 if abs(z)<=SBAND else(-1.0 if z>0 else 1.0);vR[i]+=vote*LEG
    for i,j in state.get("bunits",[]):
        y=logp[i,-BIRTHW:];x=logp[j,-BIRTHW:];beta,intercept=np.polyfit(x,y,1)
        spread=y-beta*x-intercept;spread_sd=max(float(spread.std()),1e-12);z=(float(spread[-1])-float(spread.mean()))/spread_sd
        vote=0.0 if abs(z)<=GBANDV else(-1.0 if z>0 else 1.0);vR[i]+=GW*vote*LEG;vR[j]-=GW*vote*beta*LEG
    vG=np.zeros(n)
    for i,j in state["gunits"]:
        beta,intercept=np.polyfit(logp[j],logp[i],1);spread=logp[i]-beta*logp[j]-intercept
        spread_sd=max(float(spread.std()),1e-12);z=(float(spread[-1])-float(spread.mean()))/spread_sd
        vote=0.0 if abs(z)<=GBANDV else(-1.0 if z>0 else 1.0);vG[i]+=vote*LEG;vG[j]-=vote*beta*LEG
    vR+=GW*vG
    raw_ll_signal=zres[:,-1]@state["L"];signal_sd=max(float(raw_ll_signal[1:].std()),1e-12);normalised_ll=raw_ll_signal/signal_sd
    vL[1:]=LLW*LEG*(np.clip(normalised_ll,-2.0,2.0)/2.0)[1:]
    combined=vR+vL;target=np.zeros(n)
    for i in range(1,n):
        direction=np.sign(combined[i])
        if direction!=0: target[i]=direction*LEG/px[i]
    # ALGO slot: agree-only construction (rebuilt, see header comment).
    if logp.shape[1]>ALGO_REV_HORIZON:
        algo_move=float(logp[0,-1])-float(logp[0,-1-ALGO_REV_HORIZON]);raw_sig=-float(np.sign(algo_move))
    else: raw_sig=0.0
    rel_sig=_algo_relative_signal(logp)
    if raw_sig!=0.0 and rel_sig!=0.0 and np.sign(raw_sig)==np.sign(rel_sig):
        algo_dollars=ALGO_REV_DOLLARS*raw_sig
    else:
        algo_dollars=0.0
    algo_dollars=float(np.clip(algo_dollars,-100_000.0,100_000.0))
    target[0]=algo_dollars/px[0]
    target=np.nan_to_num(target,nan=0.0,posinf=0.0,neginf=0.0)
    limits=np.full(n,LEG);limits[0]=100_000.0;share_limits=(limits/np.maximum(px,1e-12)).astype(int)
    return np.clip(target,-share_limits,share_limits).astype(int)
