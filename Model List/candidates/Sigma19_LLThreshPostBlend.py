# Sigma19_LLThreshPostBlend - Round 4 Track A candidate (Agent 1)
# Baseline: Sigma19.py, UNCHANGED except for exactly one isolated modification
# to the lead-lag matrix estimation/weighting (this round's directed target).
#
# THE ONE CHANGE:
#   Baseline applies the small-signal magnitude threshold LL_THRESH (=0.04)
#   only to linear_recent (the LL_BLEND=0.1-weighted, 450-day rolling
#   component) BEFORE blending:
#       if LL_THRESH>0: linear_recent[np.abs(linear_recent)<LL_THRESH]=0.0
#       state["L"]=(1.0-LL_BLEND)*linear_full+LL_BLEND*linear_recent
#   linear_full -- the dominant 90%-weight expanding-history component -- is
#   never denoised, so every one of its ~50x50 pairwise lead-lag
#   coefficients (many plausibly statistically insignificant, since it's a
#   raw uniform average over the full expanding sample with no significance
#   filter) feeds into raw_ll_signal = zres[:,-1] @ state["L"] every day.
#   That is a plausible source of pure cross-sectional noise in the dominant
#   mechanism of the book.
#
#   This candidate moves the SAME threshold to apply once, after blending,
#   to the combined matrix instead of only to the small recent-window slice:
#       state["L"]=(1.0-LL_BLEND)*linear_full+LL_BLEND*linear_recent
#       if LL_THRESH>0: state["L"][np.abs(state["L"])<LL_THRESH]=0.0
#   Same LL_THRESH constant, same magnitude semantics -- just widened to
#   cover the component that actually drives the signal. No new parameter,
#   no new mechanism, no change to LL_BLEND, LL_ROLL, or lag (still 1 day).
#
# Internal-fold and 251-500-window results for this change vs. baseline are
# reported separately by Agent 1 in its round report -- NOT recorded in this
# file, and nothing about days 501-750 is claimed or implied here.
#
# Everything below this point is byte-for-byte identical to Sigma19.py
# except the two lines noted above.

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
BIRTHW = 0
BGATE = -4.0
GW = 0.5
ALGO_REV_HORIZON = 5
ALGO_REV_DOLLARS = 100_000.0
ALGO_HEDGE_WEIGHT = 0.0
# --- grafted from the received Sigma7p_robust_fit500.py, validated standalone
# before being added here (see Alpha Exploration Log): an ALGO-shock-conditional
# lead vector. When ALGO's own contemporaneous return is an unusually large
# shock (|z| > AG_THRESH), it directionally nudges the lead-lag forecast for
# every other name. Ablated out of that file's 7 bundled changes, this ONE
# piece was responsible for ~69% of its entire out-of-sample improvement;
# everything else in that file (RESCAN/BAND/LLW retuning, matrix
# double-centering, antisymmetric emphasis, snapshot blending) was weak,
# inconsistent, or in BAND's case negative on its own. Neighbourhood-tested
# clean: every threshold in [0.5,2.0] and every weight in [0.5,4.0] improved
# both 251-500 and 501-750 vs. a no-ag-vector baseline, no exceptions.
# Unlike that file, this version rescans (and re-freezes nothing) on the
# same RESCAN=20 cadence as the rest of the book -- the day-500 freeze in
# the original was an artifact of its own fit-500 experiment design, not
# something we want in a live, always-adapting model.
AG_THRESH = 1.0
AG_WEIGHT = 2.0
AG_BLEND = 0.10
SEED_PAIRS = [(10,46),(1,20),(13,45),(8,27),(49,50),(25,37),(7,40)]
SEED_SINGLES = []
def _new_state():
    return {"scanned":-1_000_000,"units":[],"gunits":[],"bunits":[],"rbeta":None,"L":None,"core":set(),
        "last_nt":-1,"ag":None,"agprev":None}
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
def getMyPosition(prcSoFar):
    global state
    prices=np.asarray(prcSoFar,dtype=float);n,nt=prices.shape
    if nt<=state["last_nt"]: state=_new_state()
    state["last_nt"]=nt
    if nt<WARMUP: return np.zeros(n,dtype=int)
    logp=np.log(np.maximum(prices,1e-12));level_residual,_=residualize(logp);zres,centred_returns=resid_z(logp)
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
        # --- ONE ISOLATED CHANGE vs. Sigma19.py: the LL_THRESH small-signal
        # threshold is now applied ONCE, after blending, to the combined
        # matrix -- covering the dominant 90%-weight linear_full component
        # too -- instead of only to linear_recent before the blend.
        state["L"]=(1.0-LL_BLEND)*linear_full+LL_BLEND*linear_recent
        if LL_THRESH>0: state["L"][np.abs(state["L"])<LL_THRESH]=0.0
        core=set(used)
        for i,j in state["gunits"]: core.update((i,j))
        state["core"]=core;state["scanned"]=nt;state["units"]=units
        returns=np.diff(logp,axis=1);returns=returns-returns.mean(axis=1,keepdims=True)
        state["rbeta"]=returns@returns[0]/max(float(returns[0]@returns[0]),1e-30)
        # ALGO lead vector: does today's ALGO return predict tomorrow's
        # residual return for each other name? Always re-fit on the full
        # expanding history at every rescan, exactly like the rest of the book.
        ax=centred_returns[0]/max(float(centred_returns[0].std()),1e-12)
        ag=(zres[1:,1:] @ ax[:-1])/max(len(ax)-1,1)
        state["agprev"]=state.get("ag");state["ag"]=ag
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
    raw_ll_signal=zres[:,-1]@state["L"]
    if state.get("ag") is not None:
        az=float(centred_returns[0,-1])/max(float(centred_returns[0].std()),1e-12)
        if abs(az)>AG_THRESH:
            ag_vec=state["ag"]
            if AG_BLEND>0 and state.get("agprev") is not None:
                ag_vec=(1.0-AG_BLEND)*ag_vec+AG_BLEND*state["agprev"]
            raw_ll_signal[1:]+=AG_WEIGHT*az*ag_vec
    signal_sd=max(float(raw_ll_signal[1:].std()),1e-12);normalised_ll=raw_ll_signal/signal_sd
    vL[1:]=LLW*LEG*(np.clip(normalised_ll,-2.0,2.0)/2.0)[1:]
    combined=vR+vL;target=np.zeros(n)
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
