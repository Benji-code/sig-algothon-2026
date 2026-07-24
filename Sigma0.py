# Sigma LL-Core + Seed Modifier + ALGO5 - Algothon 2026
#
# Simple, deliberately low-complexity strategy:
#
#   1. Raw expanding-history lag-one residual lead-lag is the primary signal.
#   2. Verified seed-pair mean reversion is a bounded secondary modifier.
#      It can overturn only a weak LL prediction.
#   3. ALGO uses the independently successful full-capacity five-day reversal.
#
# Removed from the poor V8 version:
#   - ridge regression
#   - target calibration
#   - coefficient stability filtering
#   - dual residualisation
#   - seed-spread ML
#   - gray/birth pairs
#
# NumPy only. Causal. Lag one only.

import numpy as np


WARMUP = 120
RESCAN = 20
DOLLARS = 10_000.0

# Primary lag-one LL.
LL_WEIGHT = 1.0

# Seed pairs can flip LL only when |standardized LL| < SEED_WEIGHT.
SEED_WEIGHT = 0.60
SEED_WIN = 450
SEED_BAND = 0.25
T_KEEP = -2.5
KEEP_RECENT = -1.0
RECENT_WINDOW = 250

# Independent ALGO sleeve.
ALGO5_DOLLARS = 100_000.0

SEED_PAIRS = [
    (10, 46),
    (1, 20),
    (13, 45),
    (8, 27),
    (49, 50),
    (25, 37),
    (7, 40),
]


def _new_state():
    return {
        "scanned": -10**9,
        "L": None,
        "seed_units": [],
        "last_nt": -1,
    }


_state = _new_state()


def reset_state():
    global _state
    _state = _new_state()


def _ar1_t(series):
    centred = series - series.mean(axis=1, keepdims=True)

    numerator = np.einsum(
        "ij,ij->i",
        centred[:, 1:],
        centred[:, :-1],
    )
    denominator = np.einsum(
        "ij,ij->i",
        centred[:, :-1],
        centred[:, :-1],
    )

    phi = numerator / np.maximum(denominator, 1e-30)

    standard_error = np.sqrt(
        np.maximum(1.0 - phi * phi, 1e-12)
        / max(series.shape[1] - 1, 1)
    )

    return (phi - 1.0) / standard_error


def _recent_ar1_t(series, window=RECENT_WINDOW):
    recent = series[-min(window, len(series)):]
    return _ar1_t(recent[None, :])[0]


def _zres(logp):
    """
    Standardized residual returns.

    Remove:
      1. ALGO return exposure.
      2. The dominant residual principal component.
    """
    returns = np.diff(logp, axis=1)
    centred = returns - returns.mean(axis=1, keepdims=True)

    algo_denominator = max(float(centred[0] @ centred[0]), 1e-30)
    algo_beta = (centred @ centred[0]) / algo_denominator

    residual = centred - np.outer(algo_beta, centred[0])
    residual[0] = 0.0

    covariance = residual[1:] @ residual[1:].T
    pc1 = np.linalg.eigh(covariance)[1][:, -1]

    factor = pc1 @ residual[1:]
    factor_denominator = max(float(factor @ factor), 1e-30)
    loading = (residual @ factor) / factor_denominator

    residual = residual - np.outer(loading, factor)
    residual[0] = 0.0

    sd = residual.std(axis=1, keepdims=True)
    sd[0] = 1.0

    return residual / np.maximum(sd, 1e-12)


def _estimate_raw_lag_one(zres):
    """
    Exact raw lag-one cross-covariance estimator used by the strong baseline.
    """
    lagged = zres[:, :-1]
    next_day = zres[:, 1:]

    L = (lagged @ next_day.T) / max(lagged.shape[1], 1)

    np.fill_diagonal(L, 0.0)
    L[0, :] = 0.0
    L[:, 0] = 0.0

    return L


def _scan_seed_pairs(logp, n):
    active = []

    for i, j in SEED_PAIRS:
        if i >= n or j >= n:
            continue

        raw_spread = logp[i] - logp[j]

        if (
            _ar1_t(raw_spread[None, :])[0] < T_KEEP
            and _recent_ar1_t(raw_spread) < KEEP_RECENT
        ):
            active.append((i, j))

    return active


def _seed_votes(logp, units):
    """
    Return one bounded directional vote per instrument.

    Multiple pair contributions are collapsed to sign so the seed sleeve
    remains a modifier and cannot swamp the primary LL signal.
    """
    n, nt = logp.shape
    raw_votes = np.zeros(n)

    window = slice(
        -min(SEED_WIN, nt - 1),
        None,
    )

    for i, j in units:
        x = logp[j][window]
        y = logp[i][window]

        beta, intercept = np.polyfit(x, y, 1)
        spread = y - beta * x - intercept

        spread_sd = max(float(spread.std()), 1e-12)
        z = (
            float(spread[-1]) - float(spread.mean())
        ) / spread_sd

        if abs(z) <= SEED_BAND:
            continue

        direction = -1.0 if z > 0.0 else 1.0

        raw_votes[i] += direction
        raw_votes[j] -= direction * beta

    return np.sign(raw_votes)


def getMyPosition(prcSoFar):
    global _state

    prices = np.asarray(prcSoFar, dtype=float)
    n, nt = prices.shape

    if nt < _state["last_nt"]:
        _state = _new_state()

    _state["last_nt"] = nt

    if nt < WARMUP:
        return np.zeros(n, dtype=int)

    logp = np.log(np.maximum(prices, 1e-12))
    zres = _zres(logp)

    if (
        nt - _state["scanned"] >= RESCAN
        or _state["L"] is None
    ):
        _state["L"] = _estimate_raw_lag_one(zres)
        _state["seed_units"] = _scan_seed_pairs(logp, n)
        _state["scanned"] = nt

    # Primary LL forecast.
    ll_raw = zres[:, -1] @ _state["L"]
    ll_sd = max(float(ll_raw[1:].std()), 1e-12)
    ll_score = ll_raw / ll_sd
    ll_score[0] = 0.0

    # Secondary verified-pair modifier.
    seed_vote = _seed_votes(
        logp,
        _state["seed_units"],
    )
    seed_vote[0] = 0.0

    # Because SEED_WEIGHT < 1, a seed vote cannot reverse an LL score whose
    # absolute magnitude exceeds SEED_WEIGHT.
    combined = (
        LL_WEIGHT * ll_score
        + SEED_WEIGHT * seed_vote
    )
    combined[0] = 0.0

    current_price = prices[:, -1]
    target = np.zeros(n)

    target[1:] = (
        np.sign(combined[1:])
        * DOLLARS
        / np.maximum(current_price[1:], 1e-12)
    )

    # Independent five-day ALGO reversal.
    if ALGO5_DOLLARS > 0.0 and nt >= 6:
        algo_move = (
            float(logp[0, -1])
            - float(logp[0, -6])
        )
        algo_direction = -float(np.sign(algo_move))

        target[0] = (
            algo_direction
            * ALGO5_DOLLARS
            / max(float(current_price[0]), 1e-12)
        )

    target = np.nan_to_num(
        target,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    dollar_limits = np.full(n, DOLLARS)
    dollar_limits[0] = 100_000.0

    share_limits = (
        dollar_limits
        / np.maximum(current_price, 1e-12)
    ).astype(int)

    return np.clip(
        target,
        -share_limits,
        share_limits,
    ).astype(int)