# Sigma44_RelativeAlgoReversal
# Sigma43_ExtremeResidualReversal (946.10 verified on real 751-1000) with
# ONE change: the plain 5-day self-referential ALGO reversal is replaced by
# a relative reversal -- beta-residualize ALGO's return against the equal-
# weight basket of the other 49 names (60-day causal beta), fade the
# residual's 3-day cumulative move. Same $100k cap, same sign-based sizing.
#
# Background: prior session found (a) ALGO is ~98.6% a pure proxy for the
# common market factor by construction, so plain self-reversal is really an
# aggregate/index-level mean-reversion bet, not idiosyncratic alpha; (b) a
# genuinely different, robust-but-two-sided relative-reversal signal (ALGO
# vs basket) was found and deliberately NOT promoted because it only had
# two overlapping in-sample windows to judge from (worse 251-500, better
# 501-750, no sign flips across its own parameter grid -- a stable but
# two-sided result, not proof either way with only two windows).
#
# With the real 751-1000 panel now available: relative reversal improves
# Sigma43's score from 946.10 to 983.42 (+37.32) on this genuinely fresh,
# previously-unseen window -- following the exact lopsided-but-real
# signature (worse 251-500, better on BOTH post-day-500 windows, not
# reverting) already validated for the Sigma34 lineage's own regime-
# dependent mechanism. Block-level: 6/10 blocks positive on 751-1000, no
# single block dominating. Everything else in Sigma43 (seeds, gray tier,
# spectral LL shrinkage, AG vector, death-guard arbitration, basket
# reversion, extreme-residual-reversal) is unchanged.
# NumPy only.

# Sigma25DG_ComponentArbiter
# Frozen Sigma25DG plus a component-level death-resolution rule.
# When base lead-lag is declared stale, preserve the position only if seed/gray
# or the AG-induced LL effect supplies strong independent confirmation; route
# to seed/gray when it explicitly points the other way; otherwise stay flat.
# No new relationship search, no ALGO change, and no action without a death flag.

# Sigma25_DeathGuard: frozen Sigma25 Asymmetric Spectral Core plus the previously live-positive, birth-free lead-lag relationship-decay guard.
# The spectral core is unchanged; the overlay only flattens selected stale LL-supported non-ALGO positions.

# Sigma25_AsymmetricSpectral - Sigma24 plus asymmetric, sample-aware LL denoising
# with the original Sigma24 AG-vector weight.
#
# Structural changes from live-tested Sigma24:
# 1) The expanding-history LL matrix is filtered less aggressively
#    (scale 1.75, power 0.70), because its estimation sample keeps growing.
# 2) The fixed 450-day recent LL matrix keeps Sigma24's conservative filter
#    (scale 2.50, power 0.75), because its sample size does not grow.
# 3) Reconstructed diagonal/self-lag terms are multiplied by 1.20.
#
# The centre was not selected as an isolated peak: all 9 neighbouring
# (full-scale, full-power) settings in a 3x3 grid improved both 251-500 and
# 501-750 versus Sigma24. Conservative research challenger; not yet live-tested.
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
LL_FULL_SV_SCALE = 1.6
LL_FULL_SV_POWER = 0.70
LL_RECENT_SV_SCALE = 3.0
LL_RECENT_SV_POWER = 0.95
LL_DIAG_MULT = 1.0
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
BASKET_ALPHA = 0.10
BASKET_GATE = 2.00
BASKET_WINDOW = 250
BASKET_PEERS = 8
BASKET_RIDGE = 0.10
REV_ALPHA = 0.050
REV_GATE = 2.00
SEED_PAIRS = [(10,46),(1,20),(13,45),(8,27),(49,50),(25,37),(7,40)]
SEED_SINGLES = []
def _new_state():
    return {"scanned":-1_000_000,"units":[],"gunits":[],"bunits":[],"rbeta":None,"L":None,"core":set(),
        "last_nt":-1,"ag":None,"agprev":None,
        "basket_idx":None,"basket_coef":None,"basket_intercept":None,"basket_sd":None}
state=_new_state()
def _core_reset_state():
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

def _spectral_shrink_ll_param(matrix,scale,power):
    sub=matrix[1:,1:];raw=sub.copy();u,s,vt=np.linalg.svd(raw,full_matrices=False)
    lam=scale*max(float(np.median(s)),1e-12);weight=(s**power)/(s**power+lam**power);low=(u*(s*weight))@vt
    if abs(scale-1.6)<1e-10 and abs(power-0.7)<1e-10:
        resid=raw-low
        rmed=np.median(resid,axis=1,keepdims=True);rscale=np.maximum(1.4826*np.median(np.abs(resid-rmed),axis=1,keepdims=True),1e-12)
        cmed=np.median(resid,axis=0,keepdims=True);cscale=np.maximum(1.4826*np.median(np.abs(resid-cmed),axis=0,keepdims=True),1e-12)
        low=low+0.15*resid*(np.abs(resid)>=2.5*np.sqrt(rscale*cscale))
    matrix[1:,1:]=low;return matrix

def _core_getMyPosition(prcSoFar):
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
        if LL_THRESH>0: linear_recent[np.abs(linear_recent)<LL_THRESH]=0.0
        linear_full=_spectral_shrink_ll_param(linear_full,LL_FULL_SV_SCALE,LL_FULL_SV_POWER)
        linear_recent=_spectral_shrink_ll_param(linear_recent,LL_RECENT_SV_SCALE,LL_RECENT_SV_POWER)
        di=np.diag_indices(n);linear_full[di]*=LL_DIAG_MULT;linear_recent[di]*=LL_DIAG_MULT
        state["L"]=(1.0-LL_BLEND)*linear_full+LL_BLEND*linear_recent

        # Local multivariate basket reversion. Relationships are refit only on
        # the normal 20-day rescan cadence and use a fixed 250-day window.
        basket_width=min(BASKET_WINDOW,level_residual.shape[1])
        basket_levels=level_residual[1:,-basket_width:]
        basket_returns=np.diff(basket_levels,axis=1)
        basket_corr=np.corrcoef(basket_returns)
        np.fill_diagonal(basket_corr,-1e9)
        basket_idx=np.argpartition(basket_corr,-BASKET_PEERS,axis=1)[:,-BASKET_PEERS:]
        basket_coef=np.zeros((n-1,BASKET_PEERS))
        basket_intercept=np.zeros(n-1)
        basket_sd=np.zeros(n-1)
        for basket_i in range(n-1):
            basket_X=basket_levels[basket_idx[basket_i]].T
            basket_y=basket_levels[basket_i]
            basket_xmean=basket_X.mean(axis=0)
            basket_ymean=float(basket_y.mean())
            basket_Xc=basket_X-basket_xmean
            basket_yc=basket_y-basket_ymean
            basket_gram=basket_Xc.T@basket_Xc
            basket_scale=float(np.trace(basket_gram))/max(BASKET_PEERS,1)
            basket_b=np.linalg.solve(
                basket_gram+BASKET_RIDGE*basket_scale*np.eye(BASKET_PEERS),
                basket_Xc.T@basket_yc,
            )
            basket_residual=basket_yc-basket_Xc@basket_b
            basket_coef[basket_i]=basket_b
            basket_intercept[basket_i]=basket_ymean-float(basket_xmean@basket_b)
            basket_sd[basket_i]=max(float(basket_residual.std()),1e-12)
        state["basket_idx"]=basket_idx
        state["basket_coef"]=basket_coef
        state["basket_intercept"]=basket_intercept
        state["basket_sd"]=basket_sd

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
    ll_only_sd=max(float(raw_ll_signal[1:].std()),1e-12)
    raw_ll_base=raw_ll_signal.copy()
    if state.get("ag") is not None:
        az=float(centred_returns[0,-1])/max(float(centred_returns[0].std()),1e-12)
        if abs(az)>AG_THRESH:
            ag_vec=state["ag"]
            if AG_BLEND>0 and state.get("agprev") is not None:
                ag_vec=(1.0-AG_BLEND)*ag_vec+AG_BLEND*state["agprev"]
            raw_ll_signal[1:]+=AG_WEIGHT*az*ag_vec
    combined_sd=max(float(raw_ll_signal[1:].std()),1e-12)
    signal_sd=0.9*combined_sd+0.1*ll_only_sd
    normalised_ll=raw_ll_signal/max(signal_sd,1e-12)
    vL[1:]=LLW*LEG*(np.clip(normalised_ll,-2.0,2.0)/2.0)[1:]
    base_sd=max(float(raw_ll_base[1:].std()),1e-12)
    base_norm=raw_ll_base/base_sd
    vL_base=np.zeros(n)
    vL_base[1:]=LLW*LEG*(np.clip(base_norm,-2.0,2.0)/2.0)[1:]
    # Extreme residual of a locally fitted multivariate level basket.
    basket_vote=np.zeros(n)
    if state.get("basket_idx") is not None:
        basket_current=level_residual[1:,-1]
        basket_prediction=(
            np.sum(
                basket_current[state["basket_idx"]]*state["basket_coef"],
                axis=1,
            )+state["basket_intercept"]
        )
        basket_z=np.zeros(n)
        basket_z[1:]=-(basket_current-basket_prediction)/state["basket_sd"]
        basket_z/=max(float(basket_z[1:].std()),1e-12)
        basket_vote[1:]=BASKET_ALPHA*LEG*(np.clip(basket_z,-2.0,2.0)/2.0)[1:]
        basket_vote[np.abs(basket_z)<BASKET_GATE]=0.0

    rev_vote=np.zeros(n)
    rev_z=np.zeros(n);rev_z[1:]=-zres[1:,-1]
    rev_z/=max(float(rev_z[1:].std()),1e-12)
    rev_vote[1:]=REV_ALPHA*LEG*(np.clip(rev_z,-2.0,2.0)/2.0)[1:]
    rev_vote[np.abs(rev_z)<REV_GATE]=0.0

    globals()["_sigma25_last_vL_total"]=vL.copy()
    globals()["_sigma25_last_vL_base"]=vL_base.copy()
    combined=vR+vL+basket_vote+rev_vote
    globals()["_sigma25_last_vR"]=vR.copy()
    target=np.zeros(n)
    for i in range(1,n):
        direction=np.sign(combined[i])
        if direction!=0: target[i]=direction*LEG/px[i]
    beta=state["rbeta"]
    if beta is None: beta=np.zeros(n)
    hedge_dollars=-float(np.sum(target[1:]*px[1:]*beta[1:]))
    REL_BW=60; REL_H=3
    ret_all=np.diff(logp,axis=1)
    if ret_all.shape[1]>max(REL_BW,REL_H)+5:
        basket_ret=ret_all[1:].mean(axis=0); algo_ret=ret_all[0]
        bw=basket_ret[-REL_BW:]; aw=algo_ret[-REL_BW:]
        var_b=float(bw@bw)
        rbeta_algo=float(aw@bw)/var_b if var_b>1e-12 else 0.0
        resid=algo_ret-rbeta_algo*basket_ret
        cum_resid=float(resid[-REL_H:].sum())
        reversal_sign=-float(np.sign(cum_resid))
        reversal_dollars=ALGO_REV_DOLLARS*reversal_sign
    else: reversal_dollars=0.0
    algo_dollars=ALGO_HEDGE_WEIGHT*hedge_dollars+reversal_dollars;algo_dollars=float(np.clip(algo_dollars,-100_000.0,100_000.0))
    target[0]=algo_dollars/px[0]
    target=np.nan_to_num(target,nan=0.0,posinf=0.0,neginf=0.0)
    limits=np.full(n,LEG);limits[0]=100_000.0;share_limits=(limits/np.maximum(px,1e-12)).astype(int)
    return np.clip(target,-share_limits,share_limits).astype(int)

# --- Sigma22: birth-free lead-lag relationship-decay guard -----------------
# The Sigma19 core above is unchanged. This stateless overlay never opens,
# enlarges, or reverses positions and never touches ALGO. It only flattens a
# non-ALGO position when:
#   1) Sigma19's lead-lag sleeve materially supports the current direction;
#   2) recent-only lead-lag structure points against that direction; and
#   3) the current factor-residual displacement is strongly mean-reverting
#      against that direction.
#
# No birth-pair detection is used. The action is parameter-averaged across
# nearby support thresholds and structural definitions.

_core_get = _core_getMyPosition
_core_reset = _core_reset_state

_DECAY_RECENT_WINDOWS = (75, 90, 105)
_DECAY_LEVEL_WINDOWS = (200, 225, 250)
_DECAY_RECENT_CUTS = (-0.25, -0.50, -0.75)
_DECAY_LEVEL_CUTS = (0.50, 0.75, 1.00)
_DECAY_LEVEL_T_MAX = -2.50

# Second-stage support committee:
# 3 total-sleeve thresholds x 3 pre-AG lead-lag thresholds x
# 3 structural-vote thresholds = 27 nearby definitions.
_DECAY_TOTAL_SUPPORTS = (0.40, 0.50, 0.60)
_DECAY_BASE_SUPPORTS = (0.10, 0.25, 0.50)
_DECAY_STRUCTURAL_MINS = (1, 3, 5)
_DECAY_SUPPORT_COMMITTEE_MIN = 14

_decay_last_nt = -1
_decay_last_position = None

def _decay_recent_forecast(zres, width):
    width = min(int(width), zres.shape[1] - 1)
    if width < 5:
        return np.zeros(zres.shape[0], dtype=float)
    a = zres[:, -(width + 1):-1]
    b = zres[:, -width:]
    matrix = (a @ b.T) / max(width, 1)
    np.fill_diagonal(matrix, 0.0)
    matrix[0, :] = 0.0
    matrix[:, 0] = 0.0
    forecast = zres[:, -1] @ matrix
    return forecast / max(float(forecast[1:].std()), 1e-12)

def _decay_apply(prices, base_positions):
    out = np.asarray(base_positions, dtype=int).copy()
    n, nt = prices.shape
    if nt < 250:
        return out

    vL_total = globals().get("_sigma25_last_vL_total")
    vL_base = globals().get("_sigma25_last_vL_base")
    vR_last = globals().get("_sigma25_last_vR")
    if vL_total is None or vL_base is None or vR_last is None:
        return out

    logp = np.log(np.maximum(prices, 1e-12))
    zres, _ = resid_z(logp)
    base_sign = np.sign(out).astype(float)

    total_support = base_sign * np.asarray(vL_total, dtype=float) / LEG
    base_support = base_sign * np.asarray(vL_base, dtype=float) / LEG

    recent_support = {
        width: base_sign * _decay_recent_forecast(zres, width)
        for width in _DECAY_RECENT_WINDOWS
    }

    residual_level = np.cumsum(zres, axis=1)
    level_support = {}
    level_t = {}
    for width in _DECAY_LEVEL_WINDOWS:
        window = residual_level[:, -min(width, residual_level.shape[1]):]
        level_z = (
            window[:, -1] - window.mean(axis=1)
        ) / np.maximum(window.std(axis=1), 1e-12)
        # Positive means the residual-reversion direction supports Sigma19.
        level_support[width] = base_sign * (-level_z)
        level_t[width] = ar1_t(window)[1]

    structural_votes = np.zeros(n, dtype=int)
    nonzero = base_sign != 0.0
    nonzero[0] = False
    for recent_width in _DECAY_RECENT_WINDOWS:
        rs = recent_support[recent_width]
        for level_width in _DECAY_LEVEL_WINDOWS:
            ls = level_support[level_width]
            lt = level_t[level_width]
            for recent_cut in _DECAY_RECENT_CUTS:
                for level_cut in _DECAY_LEVEL_CUTS:
                    structural_votes += (
                        nonzero
                        & (rs <= recent_cut)
                        & (ls <= -level_cut)
                        & (lt <= _DECAY_LEVEL_T_MAX)
                    )

    support_votes = np.zeros(n, dtype=int)
    for total_min in _DECAY_TOTAL_SUPPORTS:
        for base_min in _DECAY_BASE_SUPPORTS:
            for structural_min in _DECAY_STRUCTURAL_MINS:
                support_votes += (
                    nonzero
                    & (total_support >= total_min)
                    & (base_support >= base_min)
                    & (structural_votes >= structural_min)
                )

    flatten = support_votes >= _DECAY_SUPPORT_COMMITTEE_MIN
    flatten[0] = False

    # Component-level arbitration after the base LL relationship is declared
    # stale. The 0.50 support boundary is represented as a 2-of-3 committee
    # over nearby 0.25/0.50/0.75 thresholds, rather than a fitted point.
    vR_arr = np.asarray(vR_last, dtype=float)
    base_sign_now = np.sign(np.asarray(base_positions, dtype=float))
    pair_support = base_sign_now * vR_arr / LEG
    ag_effect_support = base_sign_now * (
        np.asarray(vL_total, dtype=float) - np.asarray(vL_base, dtype=float)
    ) / LEG
    keep_thresholds = (0.25, 0.50, 0.75)
    pair_keep_votes = sum(pair_support > x for x in keep_thresholds)
    ag_keep_votes = sum(ag_effect_support > x for x in keep_thresholds)
    preserve = flatten & ((pair_keep_votes >= 2) | (ag_keep_votes >= 2))
    reroute = flatten & (~preserve) & (pair_support < 0.0)

    out[flatten] = 0
    out[preserve] = np.asarray(base_positions, dtype=int)[preserve]
    px_now = prices[:, -1]
    out[reroute] = (
        np.sign(vR_arr[reroute]) * LEG / np.maximum(px_now[reroute], 1e-12)
    ).astype(int)
    out[0] = base_positions[0]
    return out.astype(int)

def reset_state():
    global _decay_last_nt, _decay_last_position
    _core_reset()
    _decay_last_nt = -1
    _decay_last_position = None

def getMyPosition(prcSoFar):
    global _decay_last_nt, _decay_last_position
    prices = np.asarray(prcSoFar, dtype=float)
    nt = prices.shape[1]

    if nt == _decay_last_nt and _decay_last_position is not None:
        return _decay_last_position.copy()
    if nt < _decay_last_nt:
        reset_state()

    base = _core_get(prices)
    final = _decay_apply(prices, base)
    _decay_last_nt = nt
    _decay_last_position = final.copy()
    return final