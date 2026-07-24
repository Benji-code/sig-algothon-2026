# Sigma19_IdioResid - Track A, round 2 candidate
# Base: Sigma19.py (birth-catcher removed), completely untouched except for
# ONE addition described below. Everything else -- seeds, gray tier, dense
# 2-factor lead-lag with the ALGO-shock-conditional lead vector, plain 5-day
# ALGO reversal -- is byte-for-byte identical to Sigma19.py.
#
# THE ONE CHANGE: a new "idiosyncratic residual-after-hedge" mean-reversion
# book (IDIO units).
#
# Rationale / why this is structurally different from everything already in
# the chassis:
#   - The existing single-name mean-reversion path (S units, level_residual
#     from residualize()) only hedges price LEVELS against ALGO (a single
#     factor) and is currently disabled anyway (MAX_SNG=0 in the baseline).
#   - The existing lead-lag book (zres from resid_z) DOES hedge out both ALGO
#     beta and a second cross-sectional eigenvector factor, but only uses that
#     residual in RETURN space to predict OTHER instruments' next-day moves
#     (cross-sectional lead-lag). It never asks whether a single instrument's
#     own purified (2-factor-hedged) residual mean-reverts against itself.
#   - This candidate closes that gap: it cumulates the same 2-factor-hedged
#     residual RETURNS already computed for the lead-lag book into a level
#     series per instrument, tests each instrument's own idiosyncratic level
#     for AR(1) mean reversion (same test used for pairs/singles/gray), and
#     trades it as its own independent full-limit sign book, keyed off names
#     NOT already claimed by pairs/singles/gray (to avoid double-signaling
#     the same name from two different mechanisms).
#   - This is a genuinely different mechanism from round 1's rejected
#     cross-sectional trend/momentum overlay: it is mean-reverting, not
#     trend-following, and it trades a name against its OWN hedged history,
#     not against a cross-sectional rank of other names' trailing returns.
#
# IDIO units run AFTER gray tier and only claim names none of pairs/singles/
# gray already wanted (mirrors the existing precedence where singles defer to
# pairs, and gray defers to both). An earlier build of this candidate let
# IDIO claim names BEFORE gray tier ran; that version degraded every fold and
# the official window because it was cannibalizing gray tier's own (already-
# validated) picks on the exact same names, not because idiosyncratic mean
# reversion itself was harmful. This precedence-corrected version prevents
# that by construction.
#
# Weight IDIO_W=0.5 mirrors the existing GW=0.5 gray-tier weight (a
# conservative default for a brand-new, previously untested book layered on
# top of an already-full sign-vote system) -- full-limit sign deployment is
# preserved per-name; IDIO_W only scales this book's dollar contribution into
# the shared vR ledger before the final sign() is taken, exactly like GW does
# for gray tier already.
#
# Not yet tested live. Not yet evaluated on held-out data (Agent 1 does not
# have access to days 501+).
# numpy only.

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
AG_THRESH = 1.0
AG_WEIGHT = 2.0
AG_BLEND = 0.10
SEED_PAIRS = [(10,46),(1,20),(13,45),(8,27),(49,50),(25,37),(7,40)]
SEED_SINGLES = []
# --- new: idiosyncratic residual-after-2-factor-hedge mean-reversion book ---
IDIO_T = -3.0
IDIO_BAND = 0.25
IDIO_W = 0.3
MAX_IDIO = 8
def _new_state():
    return {"scanned":-1_000_000,"units":[],"gunits":[],"bunits":[],"iunits":[],"rbeta":None,"L":None,"core":set(),
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
def idio_level_series(logp):
    # Same 2-factor hedge as resid_z (ALGO beta + top cross-sectional
    # eigenvector of the residual covariance), but returns the RAW
    # (unstandardized) residual returns cumulated into a level series per
    # instrument, i.e. each name's own idiosyncratic price path after
    # removing both factors. This is a new object -- resid_z only ever
    # exposes the standardized, per-day version used for cross-sectional
    # lead-lag; nothing in the baseline mean-reverts a name against this
    # purified version of its own history.
    returns=np.diff(logp,axis=1);centred_returns=returns-returns.mean(axis=1,keepdims=True)
    beta=centred_returns@centred_returns[0]/max(float(centred_returns[0]@centred_returns[0]),1e-30)
    residual=centred_returns-np.outer(beta,centred_returns[0]);residual[0]=0.0
    if NFAC>=2:
        covariance=residual[1:]@residual[1:].T;weights=np.linalg.eigh(covariance)[1][:,-1]
        factor=weights@residual[1:];loading=residual@factor/max(float(factor@factor),1e-30)
        residual=residual-np.outer(loading,factor);residual[0]=0.0
    n,T=residual.shape;level=np.zeros((n,T+1));level[:,1:]=np.cumsum(residual,axis=1);return level
def getMyPosition(prcSoFar):
    global state
    prices=np.asarray(prcSoFar,dtype=float);n,nt=prices.shape
    if nt<=state["last_nt"]: state=_new_state()
    state["last_nt"]=nt
    if nt<WARMUP: return np.zeros(n,dtype=int)
    logp=np.log(np.maximum(prices,1e-12));level_residual,_=residualize(logp);zres,centred_returns=resid_z(logp)
    idio_level=idio_level_series(logp)
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
        for i,j in state["gunits"]: used.add(i);used.add(j)
        # --- new: select idiosyncratic mean-reversion units, leftover names
        # only (gray tier already ran and claimed the strongest pairwise
        # residual-mean-reversion names above; idio only trades names none of
        # pairs/singles/gray wanted, so it cannot cannibalize an
        # already-validated tier -- see report for why this ordering matters) ---
        _,idio_t=ar1_t(idio_level[1:])
        idio_candidates=[(idio_t[k],k+1) for k in np.where(idio_t<IDIO_T)[0] if(k+1) not in used]
        idio_candidates.sort();iunits=[]
        for _,i in idio_candidates:
            if i in used or len(iunits)>=MAX_IDIO: continue
            iunits.append(i);used.add(i)
        state["iunits"]=iunits
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
    # --- new: trade idiosyncratic units (same pattern as S units, windowed) ---
    for i in state.get("iunits",[]):
        idio_slice=idio_level[i][window_slice];idio_sd=zstd(idio_slice)
        z=(float(idio_slice[-1])-float(idio_slice.mean()))/idio_sd
        vote=0.0 if abs(z)<=IDIO_BAND else(-1.0 if z>0 else 1.0);vR[i]+=IDIO_W*vote*LEG
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
