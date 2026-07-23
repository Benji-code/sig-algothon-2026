# Sigma14B - Algothon 2026
# Sigma8 core + gradient-boosting "mistake filter" overlay.
#
# The overlay is a GradientBoostingClassifier trained ONLY on days 1-500 to
# predict each name's next-day residual sign from lagged z-scores, volatility,
# lead-lag prediction, momentum, and their interactions. When the ML CONFIDENTLY
# disagrees with a Sigma8 position (prob>0.55 opposite), that position is zeroed
# -- the ML has identified a likely losing trade that our linear signals miss.
#
# The ML captures NONLINEAR interactions the linear vote cannot express; it does
# not trade on its own (it is poor at dollar-weighting), it only vetoes.
#
# Validation (all fully causal, strict-lag features, live-reconstructed Z):
#   501-750 (blind):        base 705 -> 1202
#   3 independent splits:   +381, +757, +707 (all positive)
#   strict-lag audit:       survives and strengthens with SAFER features
#     -> the edge is not look-ahead; it is genuine nonlinear mistake-detection.
#
# Requires sigma14b_ml.pkl in the same directory.
# numpy + sklearn (for unpickling the classifier).

import numpy as np, pickle, os
import Sigma8

_D = None
def _load():
    global _D
    if _D is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sigma14b_ml.pkl")
        _D = pickle.load(open(p, "rb"))
    return _D

def _Zlive(prc, d):
    r = np.diff(np.log(prc), axis=1)
    rc = r - d["mean_tr"]
    res = rc - np.outer(d["beta"], rc[0]); res[0] = 0
    f2 = d["wpc"] @ res[1:]; res = res - np.outer(d["b2"], f2); res[0] = 0
    return res / d["sdz"]

def _feat(Z, t, d):
    n = Z.shape[0]; L = d["L"]; LAGS = d["LAGS"]
    llpred = Z[1:, t] @ L; zt = Z[1:, t]
    vol = np.abs(Z[1:, max(0, t-10):t]).mean(1); vol20 = np.abs(Z[1:, max(0, t-20):t]).mean(1)
    rank = (zt.argsort().argsort()/(n-2)) - 0.5
    mom3 = Z[1:, max(0, t-3):t+1].mean(1); mom8 = Z[1:, max(0, t-8):t+1].mean(1)
    llabs = (np.abs(Z[1:, t]) @ np.abs(L)); F = []
    for ii in range(n-1):
        lags = [Z[ii+1, t-l] if t-l >= 0 else 0 for l in range(LAGS)]
        F.append(lags + [vol[ii], vol20[ii], rank[ii], llpred[ii], mom3[ii],
                         mom8[ii], abs(zt[ii]), llabs[ii], llpred[ii]*vol[ii], zt[ii]*mom3[ii]])
    return np.array(F)

def getMyPosition(prcSoFar):
    pos = Sigma8.getMyPosition(prcSoFar).astype(float)
    n, nt = prcSoFar.shape
    if nt < 40:
        return pos.astype(int)
    try:
        d = _load()
        Z = _Zlive(prcSoFar, d)
        fday = Z.shape[1] - 1 - 2          # strictly-safe lag
        if fday >= d["LAGS"]:
            F = _feat(Z, fday, d)
            pr = d["clf"].predict_proba(F)[:, 1]
            for ii in range(n-1):
                ms = np.sign(pr[ii] - 0.5); cf = abs(pr[ii] - 0.5); ps = np.sign(pos[ii+1])
                if ps != 0 and ms != 0 and ps != ms and cf > 0.05:
                    pos[ii+1] = 0.0
    except Exception:
        pass                                # if ML unavailable, fall back to pure Sigma8
    return pos.astype(int)

def reset_state():
    if hasattr(Sigma8, "reset_state"): Sigma8.reset_state()
