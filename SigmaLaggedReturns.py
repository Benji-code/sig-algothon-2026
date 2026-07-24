"""
SigmaLaggedReturns.py -- explicit lagged-log-return reversal strategy.

This is intentionally simple and validation-friendly: features at day t use only
prices up to day t, and positions earn PnL from t -> t+1.  No fitted model is
trained on the test window; the rolling-window evaluator is used to choose/tune
these constants out-of-sample.
"""

import numpy as np

# ---------------- tunables ----------------
LAGS        = (1, 2, 5, 10, 20, 40, 60)   # lagged-return features
WEIGHTS     = (0.5, 0.8, 1.2, 1.8, 2.2, 1.2, 0.6)
STD_WINDOW  = 250                         # history used to z-score lagged returns
K           = 1.35                        # tanh aggressiveness
CLIP_Z      = 4.0                         # cap per-lag z-score
XSEC_DEMEAN = True                        # remove same-horizon market move
RESID_ALGO  = True                        # residualise non-ALGO names vs ALGO lagged return
ALGO_IDX    = 0
ALGO_LIMIT  = 100_000.0
DFLT_LIMIT  = 10_000.0
# ------------------------------------------


def _safe_std(x, axis=None):
    return np.maximum(np.std(x, axis=axis), 1e-9)


def getMyPosition(prcSoFar):
    prc = np.asarray(prcSoFar, dtype=float)
    nInst, nDays = prc.shape

    dlrLimit = np.full(nInst, DFLT_LIMIT, dtype=float)
    if nInst > ALGO_IDX:
        dlrLimit[ALGO_IDX] = ALGO_LIMIT

    max_lag = max(LAGS)
    if nDays < max_lag + 3:
        return np.zeros(nInst, dtype=int)

    logp = np.log(np.maximum(prc, 1e-12))
    signal = np.zeros(nInst, dtype=float)
    total_w = 0.0

    for lag, wt in zip(LAGS, WEIGHTS):
        if nDays <= lag + 2:
            continue

        # Current lagged return feature: r_t(lag) = logP_t - logP_{t-lag}
        r_now = logp[:, -1] - logp[:, -1 - lag]

        # Historical same-lag returns for z-scoring; all end at or before today.
        hist = logp[:, lag:] - logp[:, :-lag]
        hist = hist[:, -min(STD_WINDOW, hist.shape[1]):]

        h_mean = hist.mean(axis=1)
        h_std = _safe_std(hist, axis=1)
        z = (r_now - h_mean) / h_std

        if XSEC_DEMEAN and nInst > 1:
            z = z - z[1:].mean() if nInst > 1 else z

        if RESID_ALGO and nInst > 2:
            # Remove contemporaneous ALGO-lag feature exposure from names 1..n.
            x = z[ALGO_IDX]
            if abs(x) > 1e-9:
                y = z.copy()
                # shrink rather than fully neutralise; preserves ALGO profit centre.
                beta = np.zeros(nInst)
                denom = max(float(x * x), 1e-9)
                beta[1:] = (y[1:] * x) / denom
                z[1:] = y[1:] - 0.35 * beta[1:] * x

        z = np.clip(z, -CLIP_Z, CLIP_Z)

        # Reversal: positive lagged return => short; negative lagged return => long.
        signal += wt * (-z)
        total_w += abs(wt)

    if total_w > 0:
        signal /= total_w

    dollar = dlrLimit * np.tanh(K * signal)
    pos = dollar / np.maximum(prc[:, -1], 1e-12)
    pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    return pos.astype(int)
