# Sigma19_LLFullSoftThresh - Agent 1, Round 5 (last round of this Track A push)
#
# Baseline: Sigma19.py, unmodified except for the ONE change below.
#
# THE ONE CHANGE: soft-threshold (shrinkage) denoising of linear_full, the
# expanding-history lead-lag component, which carries (1-LL_BLEND)=0.9 of the
# blend weight in state["L"] and was, in the baseline, completely unfiltered
# (only linear_recent, the 0.1-weight recent-window component, had any
# threshold applied, via LL_THRESH=0.04, pre-blend).
#
#   linear_full = sign(linear_full) * max(|linear_full| - LL_THRESH_FULL, 0)
#
# applied to linear_full BEFORE the LL_BLEND mix, immediately after it is
# computed and zeroed on the diagonal/instrument-0 rows/cols. linear_recent's
# own existing hard threshold (LL_THRESH=0.04) is untouched.
#
# Why this is not a repeat of the round-4 idea that was rejected: round 4
# took LL_THRESH=0.04 -- a threshold tuned for the recent, 0.1-weight
# component -- and reused it wholesale, as a HARD cutoff, on the already-
# blended L (i.e. effectively on both components at once, post-blend). The
# track coordinator's verdict was that this showed genuine sub-window
# instability: one contiguous ~62-day held-out block drove the entire gain,
# a different block erased it and more, and one internal fold moved sharply
# against the others. The coordinator's recommendation was explicit: if
# revisiting "the dominant unfiltered component" angle, avoid an all-or-
# nothing hard threshold reused across structurally different components --
# use a softer/graduated version, or a separately-tuned criterion for that
# component specifically -- and check sub-window stability directly, not
# just the fold mean.
#
# This candidate addresses both parts:
#  1. Soft-threshold (shrinkage), not hard threshold: entries near the
#     cutoff are shrunk continuously toward zero rather than snapping from
#     "kept exactly as-is" to "zeroed" at a hard boundary. Every surviving
#     entry (not just the small ones) is pulled toward zero by
#     LL_THRESH_FULL, which mutes the previously-unfiltered component's
#     largest, noisiest realisations too, not just the near-zero ones.
#  2. Separately-tuned, deliberately conservative parameter, applied ONLY to
#     linear_full and left as its own constant (LL_THRESH_FULL), not reused
#     from LL_THRESH. Diagnostic on prices.txt alone (days 200/300/400/500
#     expanding snapshots) showed linear_full's off-diagonal |value|
#     distribution has median ~0.035-0.05, so naively reusing LL_THRESH=0.04
#     as a hard cutoff here would zero 44-56% of entries -- comparable in
#     scale to what round 4 did and plausibly why it was unstable.
#
# LL_THRESH_FULL=0.008 was chosen after a fine sweep (0.004-0.030) of the
# internal-fold behaviour, not a single point estimate: T in [0.006,0.009]
# is a genuine plateau where all four internal folds move together
# (fold1/fold2 roughly flat, fold3/fold4 solidly positive, official 251-500
# flat-to-mildly-positive) -- the same "neighbourhood-tested clean" bar this
# file's own AG-vector was held to. T>=0.010 was explicitly rejected despite
# a BETTER official-window number (+3.8% vs +1.0% here) because fold1 breaks
# sharply and discontinuously at that point (-3% at T=0.010 falling to -15%
# at T=0.011) -- exactly the kind of cliff/regime-fragility the round-4
# critique warned against, so the more attractive-looking official score at
# T=0.010+ was deliberately not chosen.
#
# Internal-fold sub-window (day-by-day, not just fold-mean) stability was
# also checked explicitly at this final T=0.008 before recommending this
# candidate -- see Agent 1's round-5 report for the full breakdown.
#
# Not yet tested live. numpy only.

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
LL_THRESH_FULL = 0.008  # NEW: separately-tuned soft-threshold for linear_full only
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
        if LL_THRESH_FULL>0:
            linear_full=np.sign(linear_full)*np.maximum(np.abs(linear_full)-LL_THRESH_FULL,0.0)
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
