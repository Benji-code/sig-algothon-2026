"""
Sigma0.py  --  Algothon 2026 submission
=======================================

Strategy:  MULTI-HORIZON STATISTICAL MEAN-REVERSION  (limit-saturated)

Why this works on this universe
-------------------------------
1. The universe is strongly and *consistently* mean-reverting. Log-return
   reversal has a POSITIVE information coefficient at every horizon we tested
   (1d..60d), peaking around 20 days; momentum has NEGATIVE IC everywhere.
   The reversal edge is stable across independent sub-periods (Sharpe ~2 in
   each half of the data), so it is a structural property of the data, not a
   fit to one price path.

2. Instrument 0 (ALGO) is the market factor -- every other instrument's single
   highest correlation is with ALGO (0.5-0.6). ALGO also carries the special
   $100k position limit (10x) and 0.2bp commission (1/5). It mean-reverts too,
   so we deliberately run it at its full limit: it is the intended profit
   centre and sizing it up raises BOTH mean-PnL and Sharpe.

Scoring insight
---------------
Score = mu * SR^2/(SR^2+1),  SR = sqrt(250)*mu/sigma.
Once the Sharpe is comfortably above ~1.5 the penalty term is already ~0.8 and
nearly flat, so the score becomes dominated by MEAN daily PnL. We therefore run
the directional reversal book at size (rather than dollar/beta-neutralising it):
neutralising lifts Sharpe a touch but throws away a third of the mean-PnL and
nets a LOWER score. We maximise mu while keeping Sharpe high.

Signal
------
For each instrument, z = (logP_today - mean(logP, lookback)) / std(logP, lookback),
averaged over lookbacks (10, 20). Position leans against the z-score
(rich -> short, cheap -> long). We map the signal through tanh so strong
dislocations size up to the position limit while weak ones stay small, then
scale by each instrument's dollar limit (ALGO = $100k, others = $10k).

Tunable knobs are at the top. TREND_HEDGE is optional regime insurance:
0.0 is the highest-expected-score setting on all evidence here; raising it to
~0.15 tilts a slice of the book with long-run trend, cheap insurance if you
fear the hidden evaluation segment trends instead of reverting.
"""

import numpy as np

# ---------------- tunables ----------------
LOOKBACKS   = (10, 20)   # mean-reversion horizons (days) to ensemble
K           = 2.0        # aggressiveness of the tanh limit-saturation
CLIP_Z      = 4.0        # cap per-horizon z-score (robustness to outliers)
TREND_HEDGE = 0.0        # 0.0 = pure reversal (best here); 0.10-0.20 = regime insurance
TREND_LB    = 60         # horizon used for the optional trend sleeve
ALGO_IDX    = 0          # the special instrument (ALGO)
ALGO_LIMIT  = 100_000.0  # $ position limit for ALGO
DFLT_LIMIT  = 10_000.0   # $ position limit for every other instrument
# ------------------------------------------


def getMyPosition(prcSoFar):
    """
    prcSoFar : np.ndarray, shape (nInst, nDays), prices up to and including today.
    returns  : np.ndarray of integer share positions (one per instrument).

    The evaluator clips dollar positions to the per-instrument limits and rounds
    to integer shares; we already target within-limit dollar sizes so clipping
    is essentially a no-op safety net.
    """
    prc = np.asarray(prcSoFar, dtype=float)
    nInst, nDays = prc.shape

    # dollar position limits (generalises to any nInst; ALGO gets the big limit)
    dlrLimit = np.full(nInst, DFLT_LIMIT)
    if nInst > ALGO_IDX:
        dlrLimit[ALGO_IDX] = ALGO_LIMIT

    maxLb = max(max(LOOKBACKS), TREND_LB if TREND_HEDGE > 0 else 0)
    # need enough history to form the longest window; otherwise stay flat
    if nDays < maxLb + 2:
        return np.zeros(nInst)

    logP = np.log(prc)

    # ---- ensemble z-score reversal signal ----
    z = np.zeros(nInst)
    for lb in LOOKBACKS:
        window = logP[:, -lb:]
        mean = window.mean(axis=1)
        std = window.std(axis=1) + 1e-9
        zi = (logP[:, -1] - mean) / std
        z += np.clip(zi, -CLIP_Z, CLIP_Z)
    z /= len(LOOKBACKS)
    signal = -z  # lean AGAINST the dislocation (mean-revert)

    # ---- optional long-horizon trend sleeve (regime insurance) ----
    if TREND_HEDGE > 0.0:
        tw = logP[:, -TREND_LB:]
        ztr = np.clip((logP[:, -1] - tw.mean(1)) / (tw.std(1) + 1e-9),
                      -CLIP_Z, CLIP_Z)
        signal = (1.0 - TREND_HEDGE) * signal + TREND_HEDGE * ztr

    # ---- size to limits with a saturating map ----
    dollar = dlrLimit * np.tanh(K * signal)
    pos = dollar / prc[:, -1]

    # guard against any non-finite values, then integerise
    pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    return pos.astype(int)