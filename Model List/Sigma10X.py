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
LL_TOPK = 0
NFAC = 2
RVW = 0.0
RVK = 20
PNC = 1.0
BANDIT = 0
GTHR = 0
BIRTHW = 150
BGATE = -4.0
GW = 0.5
ALGO_REV_HORIZON = 5
ALGO_REV_DOLLARS = 0.0
ALGO_HEDGE_WEIGHT = -1.5
DIAG_MAX_OBS = 12_000
DIAG_MIN_CENTRE = 120
DIAG_MIN_TAIL = 60
PAIR_CENTRE_MAX = 1.5
PAIR_TAIL_MIN = 2.0
PAIR_EXIT_Z = 0.15
PAIR_STOP_Z = 3.0
LL_CENTRE_MAX = 1.5
LL_TAIL_MIN = 2.0
LL_DEADBAND = 1.0
LL_TANH_K = 1.0
SEED_PAIRS = [(10,46),(1,20),(13,45),(8,27),(49,50),(25,37),(7,40)]
SEED_SINGLES = []
def _new_state():
    return {"scanned":-1_000_000,"units":[],"gunits":[],"bunits":[],"rbeta":None,"L":None,"core":set(),
        "prev_signs":None,"prev_px":None,"apnl":{"R":[],"L":[],"V":[]},"gprev":None,"gpx":None,"gpnl":[],
        "last_nt":-1,"pair_x":[],"pair_y":[],"ll_x":[],"ll_y":[],"pair_prev":[],"ll_prev":None,
        "pair_tail_weak":False,"ll_tail_mode":False,"pair_state":{},"diag":{}}
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
def _slope(x,y,mask):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);mask=np.asarray(mask,dtype=bool)
    x=x[mask];y=y[mask];n=len(x)
    if n<3: return np.nan,np.nan,n
    xc=x-x.mean();den=max(float(xc@xc),1e-30);beta=float(xc@(y-y.mean()))/den
    residual=y-y.mean()-beta*xc;se=float(np.sqrt(max(float(residual@residual),0.0)/max(n-2,1)/den))
    return beta,se,n
def _trim_history(key):
    values=state[key]
    if len(values)>DIAG_MAX_OBS: del values[:-DIAG_MAX_OBS]
def _update_regimes():
    px=np.asarray(state["pair_x"],dtype=float);py=np.asarray(state["pair_y"],dtype=float)
    if len(px):
        pc,pce,pcn=_slope(px,py,np.abs(px)<PAIR_CENTRE_MAX);pt,pte,ptn=_slope(px,py,np.abs(px)>=PAIR_TAIL_MIN)
        pair_tail_weak=(pcn>=DIAG_MIN_CENTRE and ptn>=DIAG_MIN_TAIL and np.isfinite(pc) and np.isfinite(pt)
            and np.isfinite(pce) and np.isfinite(pte) and pt>pc+2.0*(pce+pte))
        state["pair_tail_weak"]=bool(pair_tail_weak)
    lx=np.asarray(state["ll_x"],dtype=float);ly=np.asarray(state["ll_y"],dtype=float)
    if len(lx):
        lc,lce,lcn=_slope(lx,ly,np.abs(lx)<LL_CENTRE_MAX);lt,lte,ltn=_slope(lx,ly,np.abs(lx)>=LL_TAIL_MIN)
        ll_tail_mode=(lcn>=DIAG_MIN_CENTRE and ltn>=DIAG_MIN_TAIL and np.isfinite(lc) and np.isfinite(lt)
            and np.isfinite(lte) and lt>2.0*max(lc,0.0) and lt>2.0*lte)
        state["ll_tail_mode"]=bool(ll_tail_mode)
def _update_realized_diagnostics(logp,zres):
    for item in state["pair_prev"]:
        i,j,beta,intercept,spread_last,spread_sd,z_prev=item
        if i>=logp.shape[0] or j>=logp.shape[0]: continue
        spread_now=float(logp[i,-1])-beta*float(logp[j,-1])-intercept
        response=(spread_now-spread_last)/max(spread_sd,1e-12)
        if np.isfinite(z_prev) and np.isfinite(response):
            state["pair_x"].append(float(z_prev));state["pair_y"].append(float(response))
    state["pair_prev"]=[]
    if state["ll_prev"] is not None and zres.shape[1]>0:
        previous_signal=np.asarray(state["ll_prev"],dtype=float);realised=np.asarray(zres[:,-1],dtype=float)
        if len(previous_signal)==len(realised):
            valid=np.isfinite(previous_signal[1:])&np.isfinite(realised[1:])
            state["ll_x"].extend(previous_signal[1:][valid].tolist());state["ll_y"].extend(realised[1:][valid].tolist())
    state["ll_prev"]=None
    for key in ("pair_x","pair_y","ll_x","ll_y"): _trim_history(key)
def _pair_vote(key,z,band):
    z=float(z)
    if not state["pair_tail_weak"]:
        vote=0.0 if abs(z)<=band else(-1.0 if z>0 else 1.0);state["pair_state"][key]=vote;return vote
    previous=float(state["pair_state"].get(key,0.0));magnitude=abs(z)
    if magnitude>=PAIR_STOP_Z: current=0.0
    elif previous==0.0: current=0.0 if magnitude<=band else(-1.0 if z>0 else 1.0)
    elif magnitude<=PAIR_EXIT_Z: current=0.0
    else: current=previous
    state["pair_state"][key]=current;return current
def _ll_response(normalised_signal):
    normalised_signal=np.asarray(normalised_signal,dtype=float)
    if not state["ll_tail_mode"]: return np.clip(normalised_signal,-2.0,2.0)/2.0
    magnitude=np.abs(normalised_signal);excess=np.maximum(magnitude-LL_DEADBAND,0.0)
    response=np.sign(normalised_signal)*np.tanh(LL_TANH_K*excess);response[magnitude<=LL_DEADBAND]=0.0;return response
def getMyPosition(prcSoFar):
    global state
    prices=np.asarray(prcSoFar,dtype=float);n,nt=prices.shape
    if nt<=state["last_nt"]: state=_new_state()
    state["last_nt"]=nt
    if nt<WARMUP: return np.zeros(n,dtype=int)
    logp=np.log(np.maximum(prices,1e-12));level_residual,_=residualize(logp);zres,centred_returns=resid_z(logp)
    _update_realized_diagnostics(logp,zres)
    if nt-state["scanned"]>=RESCAN:
        units=[];used=set()
        for i,j in SEED_PAIRS:
            if i<n and j<n:
                spread=(logp[i]-logp[j])[None,:];_,tstat=ar1_t(spread)
                if tstat[0]<T_KEEP and rec_t(spread[0])<KEEPR: units.append(("P",i,j));used.update((i,j))
        _,single_t=ar1_t(level_residual[1:])
        candidates=[(single_t[i-1],i) for i in SEED_SINGLES if(i<n and i not in used and single_t[i-1]<T_KEEP and rec_t(level_residual[i])<KEEPR)]
        candidates+=[(single_t[k],k+1) for k in np.where(single_t<T_SINGLE)[0] if(k+1) not in used]
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
        core=set(used)
        for i,j in state["gunits"]: core.update((i,j))
        state["core"]=core;state["scanned"]=nt;state["units"]=units
        returns=np.diff(logp,axis=1);returns=returns-returns.mean(axis=1,keepdims=True)
        state["rbeta"]=returns@returns[0]/max(float(returns[0]@returns[0]),1e-30)
        _update_regimes()
    px=prices[:,-1];window_slice=slice(-WIN,None)
    vR=np.zeros(n);vL=np.zeros(n);vV=np.zeros(n);pair_prev=[]
    for unit in state["units"]:
        if unit[0]=="P":
            _,i,j=unit
            beta,intercept=np.polyfit(logp[j][window_slice],logp[i][window_slice],1)
            spread=logp[i][window_slice]-beta*logp[j][window_slice]-intercept
            spread_sd=zstd(spread);z=(float(spread[-1])-float(spread.mean()))/spread_sd
            vote=_pair_vote(("seed",i,j),z,BAND);vR[i]+=vote*LEG;vR[j]-=vote*beta*LEG
            pair_prev.append((i,j,float(beta),float(intercept),float(spread[-1]),float(spread_sd),float(z)))
        else:
            _,i=unit;residual=level_residual[i][window_slice];residual_sd=zstd(residual)
            z=(float(residual[-1])-float(residual.mean()))/residual_sd
            vote=0.0 if abs(z)<=SBAND else(-1.0 if z>0 else 1.0);vR[i]+=vote*LEG
    for i,j in state.get("bunits",[]):
        y=logp[i,-BIRTHW:];x=logp[j,-BIRTHW:];beta,intercept=np.polyfit(x,y,1)
        spread=y-beta*x-intercept;spread_sd=max(float(spread.std()),1e-12);z=(float(spread[-1])-float(spread.mean()))/spread_sd
        vote=_pair_vote(("birth",i,j),z,GBANDV);vR[i]+=GW*vote*LEG;vR[j]-=GW*vote*beta*LEG
        pair_prev.append((i,j,float(beta),float(intercept),float(spread[-1]),float(spread_sd),float(z)))
    vG=np.zeros(n)
    for i,j in state["gunits"]:
        beta,intercept=np.polyfit(logp[j],logp[i],1);spread=logp[i]-beta*logp[j]-intercept
        spread_sd=max(float(spread.std()),1e-12);z=(float(spread[-1])-float(spread.mean()))/spread_sd
        vote=_pair_vote(("gray",i,j),z,GBANDV);vG[i]+=vote*LEG;vG[j]-=vote*beta*LEG
        pair_prev.append((i,j,float(beta),float(intercept),float(spread[-1]),float(spread_sd),float(z)))
    state["pair_prev"]=pair_prev
    vR+=GW*vG
    raw_ll_signal=zres[:,-1]@state["L"];signal_sd=max(float(raw_ll_signal[1:].std()),1e-12);normalised_ll=raw_ll_signal/signal_sd
    state["ll_prev"]=normalised_ll.copy();vL[1:]=LLW*LEG*_ll_response(normalised_ll)[1:]
    combined=vR+vL+vV;target=np.zeros(n)
    for i in range(1,n):
        direction=np.sign(combined[i])
        if direction!=0: target[i]=direction*LEG/px[i]
    beta=state["rbeta"]
    if beta is None: beta=np.zeros(n)
    hedge_dollars=-float(np.sum(target[1:]*px[1:]*beta[1:]))
    if logp.shape[1]>ALGO_REV_HORIZON:
        algo_move=float(logp[0,-1])-float(logp[0,-1-ALGO_REV_HORIZON]);reversal_sign=-float(np.sign(algo_move))
        reversal_dollars=ALGO_REV_DOLLARS*reversal_sign
    else: reversal_dollars=0.0
    algo_dollars=ALGO_HEDGE_WEIGHT*hedge_dollars+reversal_dollars;algo_dollars=float(np.clip(algo_dollars,-100_000.0,100_000.0))
    target[0]=algo_dollars/px[0]
    target=np.nan_to_num(target,nan=0.0,posinf=0.0,neginf=0.0)
    limits=np.full(n,LEG);limits[0]=100_000.0;share_limits=(limits/np.maximum(px,1e-12)).astype(int)
    return np.clip(target,-share_limits,share_limits).astype(int)
