# Sigma19 - Algothon 2026
# Sigma18 with the birth-catcher removed (BIRTHW=0 below disables it; the
# rest of the chassis -- seeds, gray tier, dense 2-factor lead-lag, ag-vector,
# plain 5-day ALGO reversal -- is untouched).
# Birth-catcher had never been isolated on its own before (see Alpha
# Exploration Log entry 13): swept both of its parameters (BGATE, BIRTHW)
# standalone and found no plateau on either axis -- the as-coded values were
# lone spikes flanked by weaker or negative neighbours, i.e. no evidence of a
# real standalone edge. Entry 17 then tested removing it from the FULL book
# (a different question -- a weak piece can still help via diversification):
# removing it improved BOTH windows (251-500: 821.71 -> 831.58, +9.9;
# 501-750: 825.41 -> 836.73, +11.3), the same balanced-both-windows shape
# that made SigmaB10 and Sigma18 trustworthy. Caveat, stated plainly rather
# than oversold: block-level win rate was close to a coin flip (6/10 blocks
# on 251-500, 5/10 on 501-750) -- the net aggregate favours removal on both
# windows, but it is not the kind of broadly-dominant result the ag-vector or
# seed pairs show. Consistent with birth-catcher being mostly noise rather
# than a reliably harmful component.
# The same entry also tested removing gray tier: that made 251-500 slightly
# better but 501-750 much worse (-50.5), so gray tier was NOT removed here
# despite also failing its own standalone isolation test (entry 16) -- its
# marginal contribution to the full book is apparently positive even though
# it has no edge traded alone, likely a diversification effect. Do not
# remove gray tier without separately re-testing that specific change.
# Not yet tested live.
# numpy only.
#
# --- CANDIDATE (round 3 redo, Agent 1): volatility-regime gate ---
# Single isolated change on top of Sigma19: added a market-wide
# realized-volatility gate that flattens the ENTIRE book (all instruments,
# hard override to zero position, not partial sizing) on any day where
# recent (20-day) realized vol of the cross-sectional mean daily return is
# more than VOL_GATE_RATIO times its own longer-run (100-day) baseline.
# Rationale: the whole chassis (seed pairs, gray tier, lead-lag) leans on a
# mean-reversion / stable-relationship assumption between names; that
# assumption is most likely to break during a genuine vol regime shift
# (correlations spike, spreads gap instead of reverting). Rather than adding
# a new signal, this reworks *when* the existing signals are allowed to act
# -- a binary on/off gate, so full-limit sign deployment is preserved on
# every day the gate is open, and the book is flat (not partially sized) on
# days it is closed. Nothing else in the file is touched.
#
# RESULT (Agent 1, round 3 redo, internal 1-500 folds + official 251-500
# only -- never evaluated against 501-750): FAILS. Swept VOL_GATE_RATIO in
# {1.30, 1.25, 1.20, 1.15, 1.10, 1.05, 1.00} against a market-wide
# 20-day/100-day realized-vol ratio proxy. Result was monotonic degradation
# on every metric (worst fold, mean fold, official 251-500) as soon as the
# gate engages at all -- no plateau, no threshold at which it helps, in
# either direction. At the most permissive threshold that still triggers on
# real days, worst-fold and official-window scores are already well below
# baseline; more aggressive gating only makes it worse. Conclusion: on this
# 1-500 sample, days flagged as "high vol" by this proxy are not
# disproportionately bad trading days for the existing book -- if anything
# gating removes profitable days. Left in candidates/ as a validly-tested
# rejection (VOL_GATE_RATIO=1.20 below), not promoted.
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
# --- volatility-regime gate params (this round's single change) ---
VOL_GATE_ON = True
VOL_SHORT_WIN = 20
VOL_LONG_WIN = 100
VOL_GATE_RATIO = 1.20
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
    gate_open=True
    if VOL_GATE_ON:
        mkt_ret=np.diff(logp,axis=1).mean(axis=0)
        if len(mkt_ret)>=VOL_LONG_WIN:
            vol_short=float(mkt_ret[-VOL_SHORT_WIN:].std());vol_long=float(mkt_ret[-VOL_LONG_WIN:].std())
            gate_open=(vol_short/max(vol_long,1e-12))<=VOL_GATE_RATIO
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
    if not gate_open: target=np.zeros(n)
    limits=np.full(n,LEG);limits[0]=100_000.0;share_limits=(limits/np.maximum(px,1e-12)).astype(int)
    return np.clip(target,-share_limits,share_limits).astype(int)