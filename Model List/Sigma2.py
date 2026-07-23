# Sigma2 - Algothon 2026
# Same book structure as Sigma1 (planted pairs + ALGO-residual singles +
# dragnet, ALGO as the hedge leg), with three changes selected by 7-fold
# walk-forward cross-validation and confirmed on 4 held-out window shapes:
#
#   1. stability gate - the dragnet admits a pair on a looser full-history
#      t-stat (-3.2) but additionally requires the *traded spread* to test
#      stationary on the most recent 250 days (t < -2.8). Recency is the
#      condition that predicts the next 250 days; full-history-only gates
#      admit pairs whose plant has already died.
#   2. rolling estimation - hedge ratio, mean and z-denominator are fit on
#      the last 350 days rather than all history, tracking the plants
#      without dragging in stale regimes (WIN 250 was too noisy, 350-400
#      is a plateau).
#   3. EWMA z-denominator - spread z uses an EWMA (halflife 20d) of
#      innovation variance scaled to OU stationary vol, shrinking positions
#      automatically when a spread's recent noise regime widens (soft
#      breakage detector, complements the stability gate).
#
# Plus a free insurance policy: seeded units are killed if their recent-250d
# t-stat rises above -1.0 (never fires on the released file; protects the
# next stage against a plant dying mid-file).
#
# CV evidence vs Sigma1 (score deltas, 7 selection folds + 4 holdout
# windows): min +6.5, mean +35.4 (selection); min +6.9, mean +38.4
# (holdout); positive on all 11. Released-file score: 419.8 (Sigma1 374.6,
# Sigma0 326.0).
#
# Explored and rejected with evidence: lead-lag lags 1-5 (inside null band),
# ARCH/vol-clustering (absent), ALGO drift (sign-flips across windows),
# 3-name baskets (62k-test null generates the same candidates; causal PnL
# worse than the pairs they displace), RESCAN 10 and TS -3.3 (fold-volatile,
# fail worst-fold criterion).
#
# numpy only.

import numpy as np

LEG      = 10_000
BAND     = 0.25
SBAND    = 0.25
RESCAN   = 20
WARMUP   = 120
T_GATE   = -3.2      # dragnet full-history gate
T_REC    = -2.8      # dragnet recent-250d confirmation gate
T_SINGLE = -3.1
T_KEEP   = -2.5      # full-history keep gate for seeded units
KEEPR    = -1.0      # recent-250d kill switch for seeded units
MAX_PAIRS = 24
MAX_SNG   = 6
WIN      = 350       # signal estimation window
EWMA_HL  = 20.0      # halflife for z-denominator innovation vol

SEED_PAIRS   = [(10, 46), (1, 20), (13, 45), (8, 27), (49, 50), (25, 37), (7, 40)]
SEED_SINGLES = [35, 33, 18, 44]     # NAYO, MTNS, RTTH, HTRK

state = {"scanned": -1_000_000, "units": [], "rbeta": None}


def ar1_t(s):
    s = s - s.mean(axis=1, keepdims=True)
    num = np.einsum("ij,ij->i", s[:, 1:], s[:, :-1])
    den = np.einsum("ij,ij->i", s[:, :-1], s[:, :-1])
    phi = num / np.maximum(den, 1e-30)
    se = np.sqrt(np.maximum(1.0 - phi * phi, 1e-12) / (s.shape[1] - 1))
    return phi, (phi - 1.0) / se


def rec_t(x, w=250):
    x = x[-min(w, len(x)):]
    return ar1_t(x[None, :])[1][0]


def residualize(logp):
    Lc = logp - logp.mean(axis=1, keepdims=True)
    x0 = Lc[0]
    bl = (Lc @ x0) / (x0 @ x0)
    Z = Lc - np.outer(bl, x0)
    Z[0] = 0.0
    return Z, bl


def scan_raw(logp, n):
    L = logp - logp.mean(axis=1, keepdims=True)
    hits = []
    for i in range(1, n - 1):
        x = L[i]; Y = L[i + 1:]
        b = (Y @ x) / max(x @ x, 1e-30)
        R = Y - np.outer(b, x)
        _, t = ar1_t(R)
        for k in np.where(t < T_GATE)[0]:
            if rec_t(R[k]) < T_REC:
                hits.append((t[k], i, i + 1 + k))
    hits.sort()
    return hits


def zstd(s):
    # EWMA of innovation variance, scaled to OU stationary vol (phi ~ 0.85
    # daily, matching the planted pairs' ~5d half-life)
    ds = np.diff(s)
    if len(ds) == 0:
        return max(s.std(), 1e-12)
    a = 1.0 - 0.5 ** (1.0 / EWMA_HL)
    v = ds[0] * ds[0]
    for d in ds[1:]:
        v = (1.0 - a) * v + a * d * d
    return max(np.sqrt(v / (2.0 * (1.0 - 0.85))), 1e-12)


def getMyPosition(prcSoFar):
    n, nt = prcSoFar.shape
    if nt < WARMUP:
        return np.zeros(n, dtype=int)
    logp = np.log(prcSoFar)
    Z, bl = residualize(logp)

    if nt - state["scanned"] >= RESCAN:
        units, used = [], set()
        _, ts = ar1_t(Z[1:])
        for i, j in SEED_PAIRS:                       # 1. seed pairs
            if j < n:
                sp = (logp[i] - logp[j])[None, :]
                _, t = ar1_t(sp)
                if t[0] < T_KEEP and rec_t(sp[0]) < KEEPR:
                    units.append(("P", i, j)); used.update((i, j))
        cand = [(ts[i - 1], i) for i in SEED_SINGLES   # 2. singles
                if i < n and i not in used and ts[i - 1] < T_KEEP
                and rec_t(Z[i]) < KEEPR]
        cand += [(ts[k], k + 1) for k in np.where(ts < T_SINGLE)[0]
                 if (k + 1) not in used]
        cand.sort()
        ns = 0
        for t, i in cand:
            if i in used or ns >= MAX_SNG:
                continue
            units.append(("S", i)); used.add(i); ns += 1
        for t, i, j in scan_raw(logp, n):             # 3. dragnet
            if i not in used and j not in used and \
               sum(u[0] == "P" for u in units) < MAX_PAIRS:
                units.append(("P", i, j)); used.update((i, j))
        state["scanned"] = nt
        state["units"] = units
        r = np.diff(logp, axis=1)
        r = r - r.mean(axis=1, keepdims=True)
        state["rbeta"] = (r @ r[0]) / max(r[0] @ r[0], 1e-30)

    px = prcSoFar[:, -1]
    W = slice(-WIN, None)
    tgt = np.zeros(n)
    hedge_dlr = 0.0
    pair_idx = []
    for u in state["units"]:
        if u[0] == "P":
            _, i, j = u
            b, c = np.polyfit(logp[j][W], logp[i][W], 1)
            s = logp[i][W] - b * logp[j][W] - c
            z = (s[-1] - s.mean()) / zstd(s)
            if abs(z) > BAND:
                w = -1.0 if z > 0 else 1.0
                tgt[i] += w * LEG / px[i]
                tgt[j] -= w * b * LEG / px[j]
                pair_idx += [i, j]
        else:
            _, i = u
            zi = Z[i][W]
            z = (zi[-1] - zi.mean()) / zstd(zi)
            if abs(z) > SBAND:
                w = -1.0 if z > 0 else 1.0
                tgt[i] = w * LEG / px[i]
                hedge_dlr -= w * bl[i] * LEG

    if pair_idx:
        rb = state["rbeta"]
        pi = np.array(sorted(set(pair_idx)))
        hedge_dlr -= float(np.sum(tgt[pi] * px[pi] * rb[pi]))

    tgt[0] += hedge_dlr / px[0]

    if not state["units"]:
        sh = max(1, int(300 / px[0]))
        tgt[0] = sh if nt % 2 == 0 else -sh

    lim = np.full(n, 10_000.0)
    lim[0] = 100_000.0
    lim = (lim / px).astype(int)
    return np.clip(tgt, -lim, lim).astype(int)
