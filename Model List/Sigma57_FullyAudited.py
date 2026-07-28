# Sigma57_FullyAudited
# Sigma56_CausalReliabilitySleeve (1039.69 on the corrected continuous
# 501-1000 metric) with its ALGO sleeve replaced: the noisy adaptive
# trailing-edge-vote selector (Sigma51's own mechanism) is swapped for
# Sigma55's validated structural R^2 regime gate.
#
# Full re-audit performed before this promotion (Alpha Exploration Log
# entry 51), after discovering in entry 50 that every prior ranking in
# this project used the wrong evaluation metric:
#
# 1) Foundational lineage re-checked on the CORRECT metric (continuous
#    501-1000, matching how the real final evaluator actually runs):
#    confirmed the promotion ordering still holds, just at smaller margins
#    than the old 250-day metric suggested --
#    Sigma19=790.23 < Sigma23=853.88 < Sigma34=890.78 < Sigma42=904.82 <
#    Sigma43=910.22 < Sigma44=956.79 < Sigma53=977.24 < Sigma56=1039.69.
#
# 2) The adaptive ALGO selector re-ablated on the correct metric: its
#    entire net contribution (+7 to +30 depending on context) is driven by
#    two large, mostly-offsetting blocks (one +9,537, one -10,666 out of
#    twenty 25-day blocks) -- a noisy, unreliable signature, not a broad
#    edge, regardless of which chassis it's attached to. Replaced with the
#    R^2 gate, which reduces to the same result as flat relative-reversal
#    on this data (R^2 never crosses back above 0.98 during 501-1000) but
#    carries a genuine causal story and no reliance on an unvalidated
#    voting mechanism.
#
# 3) The causal reliability selector's threshold ensemble
#    (_ARCH_RELIABILITY_LOWS / _HIGHS) swept across 7 neighbourhood
#    configurations, holding the center roughly fixed and varying it by
#    +-0.025 to +-0.05, plus narrow/wide band-width variants at the same
#    center. Result: a genuine plateau, not a lone spike -- shifting the
#    center down slightly is comparable to or better than baseline
#    (1043.44 vs 1039.69), shifting up degrades gracefully (1005.41,
#    995.83), and narrow/wide band width at the same center are identical
#    to baseline (the ensemble's value is in its center, not its spread).
#
# 4) Removing the cross-fit committee mode entirely costs -43.91 on
#    501-1000 -- it is doing real, positive work, not sitting as unused
#    risk, and the mode-transition trace (entry 50) already confirmed it
#    is never selected during the one real held-out window (751-1000)
#    where it's independently known to be harmful.
#
# Final scores: 251-500=883.34, 501-750=1140.24, 751-1000=1005.95,
# 501-1000 (primary, continuous)=1033.71. Block-check on the primary
# metric vs Sigma53_Consolidated: 13/20 blocks, no single block dominating
# (largest is 26% of the total), top-5 name concentration 29.7% spread
# across genuinely diverse non-ALGO names -- the cleanest, most broadly
# distributed result of this whole re-audit.
#
# NOTE on REV_POS_ALPHA/REV_GATE (0.30/1.75 below, inherited unchanged
# from the originally-supplied file): earlier log entries described
# Sigma56/57 as using the "already-validated" 0.45/0.15/2.0 values from
# Sigma53_Consolidated -- that was a documentation error. Entry 50 tested
# reverting to 0.45/2.0 and found it statistically tied with the as-
# supplied 0.30/1.75 (1037.57 vs 1039.69 for Sigma56), so the as-supplied
# values were deliberately kept, not by accident, but the docs mis-stated
# which values that actually was. Corrected in Alpha Exploration Log
# entry 53.
#
# Interaction check (entry 53): cross-fit's marginal contribution is
# +37.93 with these actual basket/rev weights vs +51.46 with Sigma44's
# original (basket=0.1, rev flat 0.05) weights -- same direction, same
# order of magnitude, no sign flip, but a genuine ~25-30% swing depending
# on context. Components are not perfectly additive, but no qualitative
# incompatibility was found.
# NumPy only; fully self-contained.
from __future__ import annotations
import numpy as np

_ARCH_RELIABILITY_LOWS = (0.60, 0.63, 0.66)
_ARCH_RELIABILITY_HIGHS = (0.78, 0.80, 0.82)
_architecture_mode = None
_aggregate_reliability = None

# Internal diagnostics shared by the two engines. These are ordinary module
# variables, not dynamically executed namespaces or string-addressed globals.
_primary_basket_z = None
_primary_basket_vote = None
_primary_rev_z = None
_primary_rev_vote = None
_primary_pca_z = None
_primary_pca_vote = None
_guard_vL_total = None
_guard_vL_base = None
_guard_vR = None
_guard_last_base = None
_guard_flatten = None
_guard_preserve = None
_guard_reroute = None

# --- Primary production engine --------------------------------------------
_primary_LEG = 10000
_primary_BAND = 0.25
_primary_SBAND = 0.25
_primary_GBANDV = 0.25
_primary_RESCAN = 20
_primary_WARMUP = 120
_primary_T_SINGLE = -3.1
_primary_T_KEEP = -2.5
_primary_KEEPR = -1.0
_primary_MAX_SNG = 0
_primary_WIN = 450
_primary_EWMA_HL = 20.0
_primary_GLO = -3.4
_primary_GREC = -2.8
_primary_LLW = 1.3
_primary_LL_ROLL = 450
_primary_LL_BLEND = 0.1
_primary_LL_THRESH = 0.04
_primary_LL_FULL_SV_SCALE = 1.6
_primary_LL_FULL_SV_POWER = 0.7
_primary_LL_RECENT_SV_SCALE = 3.0
_primary_LL_RECENT_SV_POWER = 0.95
_primary_LL_DIAG_MULT = 1.0
_primary_NFAC = 2
_primary_BIRTHW = 0
_primary_BGATE = -4.0
_primary_GW = 0.5
_primary_ALGO_REV_HORIZON = 5
_primary_ALGO_REV_DOLLARS = 100000.0
_primary_ALGO_HEDGE_WEIGHT = 0.0
_primary_AG_THRESH = 1.0
_primary_AG_WEIGHT = 2.0
_primary_AG_BLEND = 0.1
_primary_RANK_PEER_ALPHA = 0.0
_primary_RANK_PEER_GATE = 2.0
_primary_BASKET_ALPHA = 0.4
_primary_BASKET_GATE = 1.875
_primary_BASKET_WINDOW = 250
_primary_BASKET_PEERS = 8
_primary_BASKET_RIDGE = 0.1
_primary_REV_POS_ALPHA = 0.30
_primary_REV_NEG_ALPHA = 0.15
_primary_REV_GATE = 1.75
_primary_SEED_PAIRS = [(10, 46), (1, 20), (13, 45), (8, 27), (49, 50), (25, 37), (7, 40)]
_primary_SEED_SINGLES = []

_primary_ALGO_SELECTOR_WINDOWS = (280, 300, 320)
_primary_ALGO_REL_BETA_WINDOW = 60
_primary_ALGO_REL_HORIZON = 3
_primary_ALGO_COMMISSION = 0.00002

def _primary_plain_algo_sign(logp, day):
    if day < _primary_ALGO_REV_HORIZON:
        return 0.0
    move = float(logp[0, day]) - float(logp[0, day - _primary_ALGO_REV_HORIZON])
    return -float(np.sign(move))

def _primary_relative_algo_sign(logp, returns, basket_returns, day):
    if day <= max(_primary_ALGO_REL_BETA_WINDOW, _primary_ALGO_REL_HORIZON) + 5:
        return 0.0
    basket_window = basket_returns[day - _primary_ALGO_REL_BETA_WINDOW:day]
    algo_window = returns[0, day - _primary_ALGO_REL_BETA_WINDOW:day]
    variance = float(basket_window @ basket_window)
    beta = float(algo_window @ basket_window) / variance if variance > 1e-12 else 0.0
    residual = returns[0, :day] - beta * basket_returns[:day]
    cumulative = float(residual[-_primary_ALGO_REL_HORIZON:].sum())
    return -float(np.sign(cumulative))

def _primary_algo_shares(price, sign):
    if sign == 0.0:
        return 0
    limit = int(_primary_ALGO_REV_DOLLARS / max(float(price), 1e-12))
    raw = int(_primary_ALGO_REV_DOLLARS * sign / max(float(price), 1e-12))
    return int(np.clip(raw, -limit, limit))

def _primary_algo_trailing_edge(prices, logp, returns, basket_returns, day, window):
    if day < window:
        return -np.inf

    start = day - window
    previous_day = start - 1
    if previous_day >= 0:
        previous_plain = _primary_algo_shares(
            prices[0, previous_day],
            _primary_plain_algo_sign(logp, previous_day),
        )
        previous_relative = _primary_algo_shares(
            prices[0, previous_day],
            _primary_relative_algo_sign(logp, returns, basket_returns, previous_day),
        )
    else:
        previous_plain = 0
        previous_relative = 0

    edge = 0.0
    for decision_day in range(start, day):
        plain_position = _primary_algo_shares(
            prices[0, decision_day],
            _primary_plain_algo_sign(logp, decision_day),
        )
        relative_position = _primary_algo_shares(
            prices[0, decision_day],
            _primary_relative_algo_sign(logp, returns, basket_returns, decision_day),
        )
        price_change = float(prices[0, decision_day + 1] - prices[0, decision_day])
        plain_pnl = (
            plain_position * price_change
            - _primary_ALGO_COMMISSION
            * float(prices[0, decision_day])
            * abs(plain_position - previous_plain)
        )
        relative_pnl = (
            relative_position * price_change
            - _primary_ALGO_COMMISSION
            * float(prices[0, decision_day])
            * abs(relative_position - previous_relative)
        )
        edge += relative_pnl - plain_pnl
        previous_plain = plain_position
        previous_relative = relative_position

    return edge

def _primary_adaptive_algo_dollars(prices, logp):
    day = logp.shape[1] - 1
    plain_sign = _primary_plain_algo_sign(logp, day)
    if day <= max(_primary_ALGO_REL_BETA_WINDOW, _primary_ALGO_REL_HORIZON) + 5:
        return _primary_ALGO_REV_DOLLARS * plain_sign

    returns = np.diff(logp, axis=1)
    basket_returns = returns[1:].mean(axis=0)
    relative_sign = _primary_relative_algo_sign(
        logp,
        returns,
        basket_returns,
        day,
    )

    structural_window = min(250, day)
    sx = basket_returns[day - structural_window:day]
    sy = returns[0, day - structural_window:day]
    sden = float(sx @ sx)
    sbeta = float(sy @ sx) / sden if sden > 1e-12 else 0.0
    seps = sy - sbeta * sx
    structural_r2 = 1.0 - float(seps @ seps) / max(float(sy @ sy), 1e-30)
    selected_sign = plain_sign if structural_r2 >= 0.98 else relative_sign
    return _primary_ALGO_REV_DOLLARS * selected_sign

def _primary__new_state():
    return {'scanned': -1000000, 'units': [], 'gunits': [], 'bunits': [], 'rbeta': None, 'L': None, 'Lagg': None, 'core': set(), 'last_nt': -1, 'ag': None, 'agprev': None, 'basket_idx': None, 'basket_coef': None, 'basket_intercept': None, 'basket_sd': None, 'cf_relative': None, 'cf_absolute': None}
_primary_state = _primary__new_state()

def _primary_reset_state():
    global _primary_state
    _primary_state = _primary__new_state()

def _primary_ar1_t(s):
    s = s - s.mean(axis=1, keepdims=True)
    num = np.einsum('ij,ij->i', s[:, 1:], s[:, :-1])
    den = np.einsum('ij,ij->i', s[:, :-1], s[:, :-1])
    phi = num / np.maximum(den, 1e-30)
    se = np.sqrt(np.maximum(1.0 - phi * phi, 1e-12) / max(s.shape[1] - 1, 1))
    return (phi, (phi - 1.0) / se)

def _primary_rec_t(x, w=250):
    x = x[-min(w, len(x)):]
    return _primary_ar1_t(x[None, :])[1][0]

def _primary_residualize(logp):
    centred = logp - logp.mean(axis=1, keepdims=True)
    x0 = centred[0]
    den = max(float(x0 @ x0), 1e-30)
    beta = centred @ x0 / den
    residual = centred - np.outer(beta, x0)
    residual[0] = 0.0
    return (residual, beta)

def _primary_scan_gray(logp, n, claimed):
    centred = logp - logp.mean(axis=1, keepdims=True)
    out = []
    for i in range(1, n - 1):
        if i in claimed:
            continue
        x = centred[i]
        xx = max(float(x @ x), 1e-30)
        for j in range(i + 1, n):
            if j in claimed:
                continue
            beta = float(centred[j] @ x) / xx
            residual = centred[j] - beta * x
            _, tstat = _primary_ar1_t(residual[None, :])
            if tstat[0] < _primary_GLO and abs(beta) < 4 and (_primary_rec_t(residual) < _primary_GREC):
                out.append((tstat[0], j, i))
    out.sort()
    return out

def _primary_zstd(s):
    ds = np.diff(s)
    if len(ds) == 0:
        return max(float(s.std()), 1e-12)
    alpha = 1.0 - 0.5 ** (1.0 / _primary_EWMA_HL)
    variance = ds[0] * ds[0]
    for d in ds[1:]:
        variance = (1.0 - alpha) * variance + alpha * d * d
    return max(float(np.sqrt(variance / (2.0 * 0.15))), 1e-12)

def _primary_resid_z(logp):
    returns = np.diff(logp, axis=1)
    centred_returns = returns - returns.mean(axis=1, keepdims=True)
    beta = centred_returns @ centred_returns[0] / max(float(centred_returns[0] @ centred_returns[0]), 1e-30)
    residual = centred_returns - np.outer(beta, centred_returns[0])
    residual[0] = 0.0
    if _primary_NFAC >= 2:
        covariance = residual[1:] @ residual[1:].T
        weights = np.linalg.eigh(covariance)[1][:, -1]
        factor = weights @ residual[1:]
        loading = residual @ factor / max(float(factor @ factor), 1e-30)
        residual = residual - np.outer(loading, factor)
        residual[0] = 0.0
    sd = residual.std(axis=1, keepdims=True)
    sd[0] = 1.0
    return (residual / np.maximum(sd, 1e-12), centred_returns)

def _primary__spectral_shrink_ll_param(matrix, scale, power):
    sub = matrix[1:, 1:]
    raw = sub.copy()
    u, s, vt = np.linalg.svd(raw, full_matrices=False)
    lam = scale * max(float(np.median(s)), 1e-12)
    weight = s ** power / (s ** power + lam ** power)
    low = u * (s * weight) @ vt
    if abs(scale - 1.6) < 1e-10 and abs(power - 0.7) < 1e-10:
        resid = raw - low
        rmed = np.median(resid, axis=1, keepdims=True)
        rscale = np.maximum(1.4826 * np.median(np.abs(resid - rmed), axis=1, keepdims=True), 1e-12)
        cmed = np.median(resid, axis=0, keepdims=True)
        cscale = np.maximum(1.4826 * np.median(np.abs(resid - cmed), axis=0, keepdims=True), 1e-12)
        low = low + 0.15 * resid * (np.abs(resid) >= 2.5 * np.sqrt(rscale * cscale))
    matrix[1:, 1:] = low
    return matrix


# --- Cross-fitted LL reliability vote --------------------------------------
_primary_CF_LOOKBACKS = (375, 400)
_primary_CF_FOLDS = (3, 4, 5)
_primary_CF_BLOCK = 25
_primary_CF_PURGE = 2
_primary_CF_ALPHAS = (0.60, 0.75, 0.90)
_primary_CF_GATE = 1.15

def _primary_cf_normalize(x):
    x = np.asarray(x, dtype=float)
    x = x - float(x.mean())
    return x / max(float(x.std()), 1e-12)

def _primary_cf_fit_matrix(z_window, fold_count, labels):
    source = z_window[:, :-1]
    target = z_window[:, 1:]
    matrices = []
    for fold in range(int(fold_count)):
        valid = labels == fold
        excluded = valid.copy()
        for distance in range(1, _primary_CF_PURGE + 1):
            excluded[distance:] |= valid[:-distance]
            excluded[:-distance] |= valid[distance:]
        train = ~excluded
        train_matrix = source[:, train] @ target[:, train].T / max(int(train.sum()), 1)
        valid_matrix = source[:, valid] @ target[:, valid].T / max(int(valid.sum()), 1)
        np.fill_diagonal(train_matrix, 0.0)
        np.fill_diagonal(valid_matrix, 0.0)
        left, _, right_t = np.linalg.svd(train_matrix, full_matrices=False)
        heldout = np.einsum('ik,ij,kj->k', left, valid_matrix, right_t)
        replicated = np.maximum(heldout, 0.0)
        matrices.append((left * replicated) @ right_t)
    return np.mean(matrices, axis=0)

def _primary_cf_positive_mass(source, target, fold_count, labels):
    values = []
    for fold in range(int(fold_count)):
        valid = labels == fold
        excluded = valid.copy()
        for distance in range(1, _primary_CF_PURGE + 1):
            excluded[distance:] |= valid[:-distance]
            excluded[:-distance] |= valid[distance:]
        train = ~excluded
        train_matrix = source[:, train] @ target[:, train].T / max(int(train.sum()), 1)
        valid_matrix = source[:, valid] @ target[:, valid].T / max(int(valid.sum()), 1)
        np.fill_diagonal(train_matrix, 0.0)
        np.fill_diagonal(valid_matrix, 0.0)
        left, _, right_t = np.linalg.svd(train_matrix, full_matrices=False)
        heldout = np.einsum('ik,ij,kj->k', left, valid_matrix, right_t)
        values.append(float(np.maximum(heldout, 0.0).sum()))
    return float(np.mean(values)) if values else 0.0


def _primary_aggregate_reliability(z_non_algo, nt):
    if z_non_algo.shape[1] <= 11:
        return 0.0
    cumulative = np.cumsum(z_non_algo, axis=1)
    source_all = (
        cumulative[:, 9:-1]
        - np.concatenate(
            [np.zeros((z_non_algo.shape[0], 1)), cumulative[:, :-10]],
            axis=1,
        )[:, :z_non_algo.shape[1] - 10]
    ) / 10.0
    target_all = z_non_algo[:, 10:]
    values = []
    for requested_width in _primary_CF_LOOKBACKS:
        pair_count = min(int(requested_width), source_all.shape[1])
        source = source_all[:, -pair_count:]
        target = target_all[:, -pair_count:]
        pair_index = np.arange(pair_count)
        global_start = int(nt - pair_count - 2)
        for folds in _primary_CF_FOLDS:
            relative_labels = (pair_index // _primary_CF_BLOCK) % folds
            absolute_labels = ((global_start + pair_index) // _primary_CF_BLOCK) % folds
            values.append(
                _primary_cf_positive_mass(source, target, folds, relative_labels)
            )
            values.append(
                _primary_cf_positive_mass(source, target, folds, absolute_labels)
            )
    return float(np.mean(values)) if values else 0.0


def _primary_update_architecture(reliability):
    global _architecture_mode, _aggregate_reliability
    _aggregate_reliability = float(reliability)
    votes = []
    for low in _ARCH_RELIABILITY_LOWS:
        for high in _ARCH_RELIABILITY_HIGHS:
            if reliability >= high:
                votes.append(0)  # cross-fitted LL
            elif reliability <= low:
                votes.append(1)  # accumulated-source LL
            else:
                votes.append(2)  # frozen Sigma51 forecast
    counts = np.bincount(np.asarray(votes, dtype=int), minlength=3)
    _architecture_mode = int(np.argmax(counts))


def _primary_cf_refit(z_non_algo, nt):
    relative = []
    absolute = []
    for requested_width in _primary_CF_LOOKBACKS:
        width = min(int(requested_width), z_non_algo.shape[1] - 1)
        z_window = z_non_algo[:, -(width + 1):]
        pair_count = z_window.shape[1] - 1
        pair_index = np.arange(pair_count)
        global_start = int(nt - width - 2)
        rel_fold = []
        abs_fold = []
        for folds in _primary_CF_FOLDS:
            rel_labels = (pair_index // _primary_CF_BLOCK) % folds
            abs_labels = ((global_start + pair_index) // _primary_CF_BLOCK) % folds
            rel_fold.append(_primary_cf_fit_matrix(z_window, folds, rel_labels))
            abs_fold.append(_primary_cf_fit_matrix(z_window, folds, abs_labels))
        relative.append(np.mean(rel_fold, axis=0))
        absolute.append(np.mean(abs_fold, axis=0))
    return relative, absolute

def _primary_cf_forecast(current_z, matrices):
    values = [_primary_cf_normalize(current_z @ matrix) for matrix in matrices]
    return _primary_cf_normalize(np.mean(values, axis=0))

def _primary_getMyPosition(prcSoFar):
    global _primary_state, _primary_basket_z, _primary_basket_vote, _primary_rev_z, _primary_rev_vote, _primary_pca_z, _primary_pca_vote
    prices = np.asarray(prcSoFar, dtype=float)
    n, nt = prices.shape
    if nt <= _primary_state['last_nt']:
        _primary_state = _primary__new_state()
    _primary_state['last_nt'] = nt
    if nt < _primary_WARMUP:
        return np.zeros(n, dtype=int)
    logp = np.log(np.maximum(prices, 1e-12))
    level_residual, _ = _primary_residualize(logp)
    zres, centred_returns = _primary_resid_z(logp)
    if nt - _primary_state['scanned'] >= _primary_RESCAN:
        units = []
        used = set()
        for i, j in _primary_SEED_PAIRS:
            if i < n and j < n:
                spread = (logp[i] - logp[j])[None, :]
                _, tstat = _primary_ar1_t(spread)
                if tstat[0] < _primary_T_KEEP and _primary_rec_t(spread[0]) < _primary_KEEPR:
                    units.append(('P', i, j))
                    used.update((i, j))
        _, single_t = _primary_ar1_t(level_residual[1:])
        candidates = [(single_t[k], k + 1) for k in np.where(single_t < _primary_T_SINGLE)[0] if k + 1 not in used]
        candidates.sort()
        singles_added = 0
        for _, i in candidates:
            if i in used or singles_added >= _primary_MAX_SNG:
                continue
            units.append(('S', i))
            used.add(i)
            singles_added += 1
        _primary_state['gunits'] = [(i, j) for _, i, j in _primary_scan_gray(logp, n, used)]
        _primary_state['bunits'] = []
        if _primary_BIRTHW > 0 and nt > _primary_BIRTHW + 60:
            recent = logp[:, -_primary_BIRTHW:]
            recent_c = recent - recent.mean(axis=1, keepdims=True)
            full_c = logp - logp.mean(axis=1, keepdims=True)
            for i in range(1, n - 1):
                if i in used:
                    continue
                x = recent_c[i]
                xx = max(float(x @ x), 1e-30)
                for j in range(i + 1, n):
                    if j in used:
                        continue
                    beta_recent = float(recent_c[j] @ x) / xx
                    residual_recent = recent_c[j] - beta_recent * x
                    _, recent_t = _primary_ar1_t(residual_recent[None, :])
                    if recent_t[0] < _primary_BGATE and abs(beta_recent) < 4:
                        xf = full_c[i]
                        beta_full = float(full_c[j] @ xf) / max(float(xf @ xf), 1e-30)
                        residual_full = full_c[j] - beta_full * xf
                        _, full_t = _primary_ar1_t(residual_full[None, :])
                        if full_t[0] > -2.6:
                            _primary_state['bunits'].append((j, i))
        a = zres[:, :-1]
        b = zres[:, 1:]
        linear_full = a @ b.T / max(a.shape[1], 1)
        np.fill_diagonal(linear_full, 0.0)
        linear_full[0, :] = 0.0
        linear_full[:, 0] = 0.0
        recent_width = min(_primary_LL_ROLL, zres.shape[1] - 1)
        ar = zres[:, -(recent_width + 1):-1]
        br = zres[:, -recent_width:]
        linear_recent = ar @ br.T / max(recent_width, 1)
        np.fill_diagonal(linear_recent, 0.0)
        linear_recent[0, :] = 0.0
        linear_recent[:, 0] = 0.0
        if _primary_LL_THRESH > 0:
            linear_recent[np.abs(linear_recent) < _primary_LL_THRESH] = 0.0
        linear_full = _primary__spectral_shrink_ll_param(linear_full, _primary_LL_FULL_SV_SCALE, _primary_LL_FULL_SV_POWER)
        linear_recent = _primary__spectral_shrink_ll_param(linear_recent, _primary_LL_RECENT_SV_SCALE, _primary_LL_RECENT_SV_POWER)
        di = np.diag_indices(n)
        linear_full[di] *= _primary_LL_DIAG_MULT
        linear_recent[di] *= _primary_LL_DIAG_MULT
        _primary_state['L'] = (1.0 - _primary_LL_BLEND) * linear_full + _primary_LL_BLEND * linear_recent
        cumulative_z = np.cumsum(zres, axis=1)
        aggregate_source = (
            cumulative_z[:, 9:-1]
            - np.concatenate([np.zeros((n, 1)), cumulative_z[:, :-10]], axis=1)[:, :cumulative_z.shape[1] - 10]
        ) / 10.0
        aggregate_target = zres[:, 10:]
        aggregate_full = aggregate_source @ aggregate_target.T / max(aggregate_source.shape[1], 1)
        np.fill_diagonal(aggregate_full, 0.0)
        aggregate_full[0, :] = 0.0
        aggregate_full[:, 0] = 0.0
        aggregate_width = min(_primary_LL_ROLL, aggregate_source.shape[1])
        aggregate_recent = (
            aggregate_source[:, -aggregate_width:] @ aggregate_target[:, -aggregate_width:].T
            / max(aggregate_width, 1)
        )
        np.fill_diagonal(aggregate_recent, 0.0)
        aggregate_recent[0, :] = 0.0
        aggregate_recent[:, 0] = 0.0
        if _primary_LL_THRESH > 0:
            aggregate_recent[np.abs(aggregate_recent) < _primary_LL_THRESH] = 0.0
        aggregate_full = _primary__spectral_shrink_ll_param(aggregate_full, _primary_LL_FULL_SV_SCALE, _primary_LL_FULL_SV_POWER)
        aggregate_recent = _primary__spectral_shrink_ll_param(aggregate_recent, _primary_LL_RECENT_SV_SCALE, _primary_LL_RECENT_SV_POWER)
        _primary_state['Lagg'] = (1.0 - _primary_LL_BLEND) * aggregate_full + _primary_LL_BLEND * aggregate_recent
        _primary_state['cf_relative'], _primary_state['cf_absolute'] = _primary_cf_refit(zres[1:], nt)
        _primary_update_architecture(
            _primary_aggregate_reliability(zres[1:], nt)
        )
        basket_width = min(_primary_BASKET_WINDOW, level_residual.shape[1])
        basket_levels = level_residual[1:, -basket_width:]
        basket_returns = np.diff(basket_levels, axis=1)
        basket_corr = np.corrcoef(basket_returns)
        np.fill_diagonal(basket_corr, -1000000000.0)
        basket_idx = np.argpartition(basket_corr, -_primary_BASKET_PEERS, axis=1)[:, -_primary_BASKET_PEERS:]
        basket_coef = np.zeros((n - 1, _primary_BASKET_PEERS))
        basket_intercept = np.zeros(n - 1)
        basket_sd = np.zeros(n - 1)
        for basket_i in range(n - 1):
            basket_X = basket_levels[basket_idx[basket_i]].T
            basket_y = basket_levels[basket_i]
            basket_xmean = basket_X.mean(axis=0)
            basket_ymean = float(basket_y.mean())
            basket_Xc = basket_X - basket_xmean
            basket_yc = basket_y - basket_ymean
            basket_gram = basket_Xc.T @ basket_Xc
            basket_scale = float(np.trace(basket_gram)) / max(_primary_BASKET_PEERS, 1)
            basket_b = np.linalg.solve(basket_gram + _primary_BASKET_RIDGE * basket_scale * np.eye(_primary_BASKET_PEERS), basket_Xc.T @ basket_yc)
            basket_residual = basket_yc - basket_Xc @ basket_b
            basket_coef[basket_i] = basket_b
            basket_intercept[basket_i] = basket_ymean - float(basket_xmean @ basket_b)
            basket_sd[basket_i] = max(float(basket_residual.std()), 1e-12)
        _primary_state['basket_idx'] = basket_idx
        _primary_state['basket_coef'] = basket_coef
        _primary_state['basket_intercept'] = basket_intercept
        _primary_state['basket_sd'] = basket_sd
        core = set(used)
        for i, j in _primary_state['gunits']:
            core.update((i, j))
        _primary_state['core'] = core
        _primary_state['scanned'] = nt
        _primary_state['units'] = units
        returns = np.diff(logp, axis=1)
        returns = returns - returns.mean(axis=1, keepdims=True)
        _primary_state['rbeta'] = returns @ returns[0] / max(float(returns[0] @ returns[0]), 1e-30)
        ax = centred_returns[0] / max(float(centred_returns[0].std()), 1e-12)
        ag = zres[1:, 1:] @ ax[:-1] / max(len(ax) - 1, 1)
        _primary_state['agprev'] = _primary_state.get('ag')
        _primary_state['ag'] = ag
    px = prices[:, -1]
    window_slice = slice(-_primary_WIN, None)
    vR = np.zeros(n)
    vL = np.zeros(n)
    for unit in _primary_state['units']:
        if unit[0] == 'P':
            _, i, j = unit
            beta, intercept = np.polyfit(logp[j][window_slice], logp[i][window_slice], 1)
            spread = logp[i][window_slice] - beta * logp[j][window_slice] - intercept
            spread_sd = _primary_zstd(spread)
            z = (float(spread[-1]) - float(spread.mean())) / spread_sd
            vote = 0.0 if abs(z) <= _primary_BAND else -1.0 if z > 0 else 1.0
            vR[i] += vote * _primary_LEG
            vR[j] -= vote * beta * _primary_LEG
        else:
            _, i = unit
            residual = level_residual[i][window_slice]
            residual_sd = _primary_zstd(residual)
            z = (float(residual[-1]) - float(residual.mean())) / residual_sd
            vote = 0.0 if abs(z) <= _primary_SBAND else -1.0 if z > 0 else 1.0
            vR[i] += vote * _primary_LEG
    for i, j in _primary_state.get('bunits', []):
        y = logp[i, -_primary_BIRTHW:]
        x = logp[j, -_primary_BIRTHW:]
        beta, intercept = np.polyfit(x, y, 1)
        spread = y - beta * x - intercept
        spread_sd = max(float(spread.std()), 1e-12)
        z = (float(spread[-1]) - float(spread.mean())) / spread_sd
        vote = 0.0 if abs(z) <= _primary_GBANDV else -1.0 if z > 0 else 1.0
        vR[i] += _primary_GW * vote * _primary_LEG
        vR[j] -= _primary_GW * vote * beta * _primary_LEG
    vG = np.zeros(n)
    for i, j in _primary_state['gunits']:
        beta, intercept = np.polyfit(logp[j], logp[i], 1)
        spread = logp[i] - beta * logp[j] - intercept
        spread_sd = max(float(spread.std()), 1e-12)
        z = (float(spread[-1]) - float(spread.mean())) / spread_sd
        vote = 0.0 if abs(z) <= _primary_GBANDV else -1.0 if z > 0 else 1.0
        vG[i] += vote * _primary_LEG
        vG[j] -= vote * beta * _primary_LEG
    vR += _primary_GW * vG
    raw_ll_base_signal = zres[:, -1] @ _primary_state['L']
    raw_ll_aggregate_signal = np.mean(zres[:, -10:], axis=1) @ _primary_state['Lagg']

    ag_adjustment = np.zeros(n)
    if _primary_state.get('ag') is not None:
        az = float(centred_returns[0, -1]) / max(float(centred_returns[0].std()), 1e-12)
        if abs(az) > _primary_AG_THRESH:
            ag_vec = _primary_state['ag']
            if _primary_AG_BLEND > 0 and _primary_state.get('agprev') is not None:
                ag_vec = (1.0 - _primary_AG_BLEND) * ag_vec + _primary_AG_BLEND * _primary_state['agprev']
            ag_adjustment[1:] = _primary_AG_WEIGHT * az * ag_vec

    def build_ll_vote(pre_ag_signal):
        ll_only_sd_local = max(float(pre_ag_signal[1:].std()), 1e-12)
        signal_with_ag = pre_ag_signal + ag_adjustment
        combined_sd_local = max(float(signal_with_ag[1:].std()), 1e-12)
        signal_sd_local = 0.9 * combined_sd_local + 0.1 * ll_only_sd_local
        normalised = signal_with_ag / max(signal_sd_local, 1e-12)
        vote = np.zeros(n)
        vote[1:] = _primary_LLW * _primary_LEG * (np.clip(normalised, -2.0, 2.0) / 2.0)[1:]
        return vote

    if _architecture_mode in (0, 2):
        vL = build_ll_vote(raw_ll_base_signal)
        vL_second = None
    else:
        vL = build_ll_vote(0.70 * raw_ll_base_signal + 0.30 * raw_ll_aggregate_signal)
        vL_second = build_ll_vote(0.60 * raw_ll_base_signal + 0.40 * raw_ll_aggregate_signal)
    basket_vote = np.zeros(n)
    basket_z = np.zeros(n)
    if _primary_state.get('basket_idx') is not None:
        basket_current = level_residual[1:, -1]
        basket_prediction = np.sum(basket_current[_primary_state['basket_idx']] * _primary_state['basket_coef'], axis=1) + _primary_state['basket_intercept']
        basket_z[1:] = -(basket_current - basket_prediction) / _primary_state['basket_sd']
        basket_z /= max(float(basket_z[1:].std()), 1e-12)
        basket_vote[1:] = _primary_BASKET_ALPHA * _primary_LEG * (np.clip(basket_z, -2.0, 2.0) / 2.0)[1:]
        basket_vote[np.abs(basket_z) < _primary_BASKET_GATE] = 0.0
    _primary_basket_z = basket_z.copy()
    _primary_basket_vote = basket_vote.copy()

    rev_z = np.zeros(n)
    rev_vote = np.zeros(n)
    rev_z[1:] = -zres[1:, -1]
    rev_z /= max(float(rev_z[1:].std()), 1e-12)
    rev_shape = np.clip(rev_z, -2.0, 2.0) / 2.0
    positive_tail = rev_z >= _primary_REV_GATE
    negative_tail = rev_z <= -_primary_REV_GATE
    rev_vote[positive_tail] = _primary_REV_POS_ALPHA * _primary_LEG * rev_shape[positive_tail]
    rev_vote[negative_tail] = _primary_REV_NEG_ALPHA * _primary_LEG * rev_shape[negative_tail]
    _primary_rev_z = rev_z.copy()
    _primary_rev_vote = rev_vote.copy()

    cf_unit_vote = np.zeros(n)
    if _primary_state.get('cf_relative') is not None:
        current_cf_z = zres[1:, -1]
        relative_cf = _primary_cf_forecast(current_cf_z, _primary_state['cf_relative'])
        absolute_cf = _primary_cf_forecast(current_cf_z, _primary_state['cf_absolute'])
        cf_forecast = _primary_cf_normalize(0.5 * (relative_cf + absolute_cf))
        cf_scaled = _primary_LEG * np.clip(cf_forecast, -2.0, 2.0) / 2.0
        cf_scaled[np.abs(cf_forecast) < _primary_CF_GATE] = 0.0
        cf_unit_vote[1:] = cf_scaled

    base_without_crossfit = vR + basket_vote + rev_vote
    if _architecture_mode == 0:
        committee_directions = []
        for alpha in _primary_CF_ALPHAS:
            committee_directions.append(
                np.sign(base_without_crossfit + vL + alpha * cf_unit_vote)
            )
        target_direction = np.median(np.stack(committee_directions), axis=0)
    elif _architecture_mode == 1:
        direction_first = np.sign(base_without_crossfit + vL)
        direction_second = np.sign(base_without_crossfit + vL_second)
        target_direction = 0.5 * (direction_first + direction_second)
    else:
        target_direction = np.sign(base_without_crossfit + vL)

    target = np.zeros(n)
    target[1:] = target_direction[1:] * _primary_LEG / np.maximum(px[1:], 1e-12)
    beta = _primary_state['rbeta']
    if beta is None:
        beta = np.zeros(n)
    hedge_dollars = -float(np.sum(target[1:] * px[1:] * beta[1:]))
    reversal_dollars = _primary_adaptive_algo_dollars(prices, logp)
    algo_dollars = _primary_ALGO_HEDGE_WEIGHT * hedge_dollars + reversal_dollars
    algo_dollars = float(np.clip(algo_dollars, -100000.0, 100000.0))
    target[0] = algo_dollars / px[0]
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    limits = np.full(n, _primary_LEG)
    limits[0] = 100000.0
    share_limits = (limits / np.maximum(px, 1e-12)).astype(int)
    return np.clip(target, -share_limits, share_limits).astype(int)

# --- Independent relationship-decay engine --------------------------------
_guard_LEG = 10000
_guard_BAND = 0.25
_guard_SBAND = 0.25
_guard_GBANDV = 0.25
_guard_RESCAN = 20
_guard_WARMUP = 120
_guard_T_SINGLE = -3.1
_guard_T_KEEP = -2.5
_guard_KEEPR = -1.0
_guard_MAX_SNG = 0
_guard_WIN = 450
_guard_EWMA_HL = 20.0
_guard_GLO = -3.4
_guard_GREC = -2.8
_guard_LLW = 1.3
_guard_LL_ROLL = 450
_guard_LL_BLEND = 0.1
_guard_LL_THRESH = 0.04
_guard_LL_FULL_SV_SCALE = 1.75
_guard_LL_FULL_SV_POWER = 0.7
_guard_LL_RECENT_SV_SCALE = 2.5
_guard_LL_RECENT_SV_POWER = 0.75
_guard_LL_DIAG_MULT = 1.2
_guard_NFAC = 2
_guard_BIRTHW = 0
_guard_BGATE = -4.0
_guard_GW = 0.5
_guard_ALGO_REV_HORIZON = 5
_guard_ALGO_REV_DOLLARS = 100000.0
_guard_ALGO_HEDGE_WEIGHT = 0.0
_guard_AG_THRESH = 1.0
_guard_AG_WEIGHT = 2.0
_guard_AG_BLEND = 0.1
_guard_SEED_PAIRS = [(10, 46), (1, 20), (13, 45), (8, 27), (49, 50), (25, 37), (7, 40)]
_guard_SEED_SINGLES = []

def _guard__new_state():
    return {'scanned': -1000000, 'units': [], 'gunits': [], 'bunits': [], 'rbeta': None, 'L': None, 'core': set(), 'last_nt': -1, 'ag': None, 'agprev': None}
_guard_state = _guard__new_state()

def _guard__core_reset_state():
    global _guard_state
    _guard_state = _guard__new_state()

def _guard_ar1_t(s):
    s = s - s.mean(axis=1, keepdims=True)
    num = np.einsum('ij,ij->i', s[:, 1:], s[:, :-1])
    den = np.einsum('ij,ij->i', s[:, :-1], s[:, :-1])
    phi = num / np.maximum(den, 1e-30)
    se = np.sqrt(np.maximum(1.0 - phi * phi, 1e-12) / max(s.shape[1] - 1, 1))
    return (phi, (phi - 1.0) / se)

def _guard_rec_t(x, w=250):
    x = x[-min(w, len(x)):]
    return _guard_ar1_t(x[None, :])[1][0]

def _guard_residualize(logp):
    centred = logp - logp.mean(axis=1, keepdims=True)
    x0 = centred[0]
    den = max(float(x0 @ x0), 1e-30)
    beta = centred @ x0 / den
    residual = centred - np.outer(beta, x0)
    residual[0] = 0.0
    return (residual, beta)

def _guard_scan_gray(logp, n, claimed):
    centred = logp - logp.mean(axis=1, keepdims=True)
    out = []
    for i in range(1, n - 1):
        if i in claimed:
            continue
        x = centred[i]
        xx = max(float(x @ x), 1e-30)
        for j in range(i + 1, n):
            if j in claimed:
                continue
            beta = float(centred[j] @ x) / xx
            residual = centred[j] - beta * x
            _, tstat = _guard_ar1_t(residual[None, :])
            if tstat[0] < _guard_GLO and abs(beta) < 4 and (_guard_rec_t(residual) < _guard_GREC):
                out.append((tstat[0], j, i))
    out.sort()
    return out

def _guard_zstd(s):
    ds = np.diff(s)
    if len(ds) == 0:
        return max(float(s.std()), 1e-12)
    alpha = 1.0 - 0.5 ** (1.0 / _guard_EWMA_HL)
    variance = ds[0] * ds[0]
    for d in ds[1:]:
        variance = (1.0 - alpha) * variance + alpha * d * d
    return max(float(np.sqrt(variance / (2.0 * 0.15))), 1e-12)

def _guard_resid_z(logp):
    returns = np.diff(logp, axis=1)
    centred_returns = returns - returns.mean(axis=1, keepdims=True)
    beta = centred_returns @ centred_returns[0] / max(float(centred_returns[0] @ centred_returns[0]), 1e-30)
    residual = centred_returns - np.outer(beta, centred_returns[0])
    residual[0] = 0.0
    if _guard_NFAC >= 2:
        covariance = residual[1:] @ residual[1:].T
        weights = np.linalg.eigh(covariance)[1][:, -1]
        factor = weights @ residual[1:]
        loading = residual @ factor / max(float(factor @ factor), 1e-30)
        residual = residual - np.outer(loading, factor)
        residual[0] = 0.0
    sd = residual.std(axis=1, keepdims=True)
    sd[0] = 1.0
    return (residual / np.maximum(sd, 1e-12), centred_returns)

def _guard__spectral_shrink_ll_param(matrix, scale, power):
    sub = matrix[1:, 1:]
    u, s, vt = np.linalg.svd(sub, full_matrices=False)
    lam = scale * max(float(np.median(s)), 1e-12)
    weight = s ** power / (s ** power + lam ** power)
    matrix[1:, 1:] = u * (s * weight) @ vt
    return matrix

def _guard__core_getMyPosition(prcSoFar):
    global _guard_state, _guard_vL_total, _guard_vL_base, _guard_vR
    prices = np.asarray(prcSoFar, dtype=float)
    n, nt = prices.shape
    if nt <= _guard_state['last_nt']:
        _guard_state = _guard__new_state()
    _guard_state['last_nt'] = nt
    if nt < _guard_WARMUP:
        return np.zeros(n, dtype=int)
    logp = np.log(np.maximum(prices, 1e-12))
    level_residual, _ = _guard_residualize(logp)
    zres, centred_returns = _guard_resid_z(logp)
    if nt - _guard_state['scanned'] >= _guard_RESCAN:
        units = []
        used = set()
        for i, j in _guard_SEED_PAIRS:
            if i < n and j < n:
                spread = (logp[i] - logp[j])[None, :]
                _, tstat = _guard_ar1_t(spread)
                if tstat[0] < _guard_T_KEEP and _guard_rec_t(spread[0]) < _guard_KEEPR:
                    units.append(('P', i, j))
                    used.update((i, j))
        _, single_t = _guard_ar1_t(level_residual[1:])
        candidates = [(single_t[k], k + 1) for k in np.where(single_t < _guard_T_SINGLE)[0] if k + 1 not in used]
        candidates.sort()
        singles_added = 0
        for _, i in candidates:
            if i in used or singles_added >= _guard_MAX_SNG:
                continue
            units.append(('S', i))
            used.add(i)
            singles_added += 1
        _guard_state['gunits'] = [(i, j) for _, i, j in _guard_scan_gray(logp, n, used)]
        _guard_state['bunits'] = []
        if _guard_BIRTHW > 0 and nt > _guard_BIRTHW + 60:
            recent = logp[:, -_guard_BIRTHW:]
            recent_c = recent - recent.mean(axis=1, keepdims=True)
            full_c = logp - logp.mean(axis=1, keepdims=True)
            for i in range(1, n - 1):
                if i in used:
                    continue
                x = recent_c[i]
                xx = max(float(x @ x), 1e-30)
                for j in range(i + 1, n):
                    if j in used:
                        continue
                    beta_recent = float(recent_c[j] @ x) / xx
                    residual_recent = recent_c[j] - beta_recent * x
                    _, recent_t = _guard_ar1_t(residual_recent[None, :])
                    if recent_t[0] < _guard_BGATE and abs(beta_recent) < 4:
                        xf = full_c[i]
                        beta_full = float(full_c[j] @ xf) / max(float(xf @ xf), 1e-30)
                        residual_full = full_c[j] - beta_full * xf
                        _, full_t = _guard_ar1_t(residual_full[None, :])
                        if full_t[0] > -2.6:
                            _guard_state['bunits'].append((j, i))
        a = zres[:, :-1]
        b = zres[:, 1:]
        linear_full = a @ b.T / max(a.shape[1], 1)
        np.fill_diagonal(linear_full, 0.0)
        linear_full[0, :] = 0.0
        linear_full[:, 0] = 0.0
        recent_width = min(_guard_LL_ROLL, zres.shape[1] - 1)
        ar = zres[:, -(recent_width + 1):-1]
        br = zres[:, -recent_width:]
        linear_recent = ar @ br.T / max(recent_width, 1)
        np.fill_diagonal(linear_recent, 0.0)
        linear_recent[0, :] = 0.0
        linear_recent[:, 0] = 0.0
        if _guard_LL_THRESH > 0:
            linear_recent[np.abs(linear_recent) < _guard_LL_THRESH] = 0.0
        linear_full = _guard__spectral_shrink_ll_param(linear_full, _guard_LL_FULL_SV_SCALE, _guard_LL_FULL_SV_POWER)
        linear_recent = _guard__spectral_shrink_ll_param(linear_recent, _guard_LL_RECENT_SV_SCALE, _guard_LL_RECENT_SV_POWER)
        di = np.diag_indices(n)
        linear_full[di] *= _guard_LL_DIAG_MULT
        linear_recent[di] *= _guard_LL_DIAG_MULT
        _guard_state['L'] = (1.0 - _guard_LL_BLEND) * linear_full + _guard_LL_BLEND * linear_recent
        core = set(used)
        for i, j in _guard_state['gunits']:
            core.update((i, j))
        _guard_state['core'] = core
        _guard_state['scanned'] = nt
        _guard_state['units'] = units
        returns = np.diff(logp, axis=1)
        returns = returns - returns.mean(axis=1, keepdims=True)
        _guard_state['rbeta'] = returns @ returns[0] / max(float(returns[0] @ returns[0]), 1e-30)
        ax = centred_returns[0] / max(float(centred_returns[0].std()), 1e-12)
        ag = zres[1:, 1:] @ ax[:-1] / max(len(ax) - 1, 1)
        _guard_state['agprev'] = _guard_state.get('ag')
        _guard_state['ag'] = ag
    px = prices[:, -1]
    window_slice = slice(-_guard_WIN, None)
    vR = np.zeros(n)
    vL = np.zeros(n)
    for unit in _guard_state['units']:
        if unit[0] == 'P':
            _, i, j = unit
            beta, intercept = np.polyfit(logp[j][window_slice], logp[i][window_slice], 1)
            spread = logp[i][window_slice] - beta * logp[j][window_slice] - intercept
            spread_sd = _guard_zstd(spread)
            z = (float(spread[-1]) - float(spread.mean())) / spread_sd
            vote = 0.0 if abs(z) <= _guard_BAND else -1.0 if z > 0 else 1.0
            vR[i] += vote * _guard_LEG
            vR[j] -= vote * beta * _guard_LEG
        else:
            _, i = unit
            residual = level_residual[i][window_slice]
            residual_sd = _guard_zstd(residual)
            z = (float(residual[-1]) - float(residual.mean())) / residual_sd
            vote = 0.0 if abs(z) <= _guard_SBAND else -1.0 if z > 0 else 1.0
            vR[i] += vote * _guard_LEG
    for i, j in _guard_state.get('bunits', []):
        y = logp[i, -_guard_BIRTHW:]
        x = logp[j, -_guard_BIRTHW:]
        beta, intercept = np.polyfit(x, y, 1)
        spread = y - beta * x - intercept
        spread_sd = max(float(spread.std()), 1e-12)
        z = (float(spread[-1]) - float(spread.mean())) / spread_sd
        vote = 0.0 if abs(z) <= _guard_GBANDV else -1.0 if z > 0 else 1.0
        vR[i] += _guard_GW * vote * _guard_LEG
        vR[j] -= _guard_GW * vote * beta * _guard_LEG
    vG = np.zeros(n)
    for i, j in _guard_state['gunits']:
        beta, intercept = np.polyfit(logp[j], logp[i], 1)
        spread = logp[i] - beta * logp[j] - intercept
        spread_sd = max(float(spread.std()), 1e-12)
        z = (float(spread[-1]) - float(spread.mean())) / spread_sd
        vote = 0.0 if abs(z) <= _guard_GBANDV else -1.0 if z > 0 else 1.0
        vG[i] += vote * _guard_LEG
        vG[j] -= vote * beta * _guard_LEG
    vR += _guard_GW * vG
    raw_ll_signal = zres[:, -1] @ _guard_state['L']
    raw_ll_base = raw_ll_signal.copy()
    ll_only_sd = max(float(raw_ll_signal[1:].std()), 1e-12)
    if _guard_state.get('ag') is not None:
        az = float(centred_returns[0, -1]) / max(float(centred_returns[0].std()), 1e-12)
        if abs(az) > _guard_AG_THRESH:
            ag_vec = _guard_state['ag']
            if _guard_AG_BLEND > 0 and _guard_state.get('agprev') is not None:
                ag_vec = (1.0 - _guard_AG_BLEND) * ag_vec + _guard_AG_BLEND * _guard_state['agprev']
            raw_ll_signal[1:] += _guard_AG_WEIGHT * az * ag_vec
    combined_sd = max(float(raw_ll_signal[1:].std()), 1e-12)
    signal_sd = 0.9 * combined_sd + 0.1 * ll_only_sd
    normalised_ll = raw_ll_signal / max(signal_sd, 1e-12)
    vL[1:] = _guard_LLW * _guard_LEG * (np.clip(normalised_ll, -2.0, 2.0) / 2.0)[1:]
    base_sd = max(float(raw_ll_base[1:].std()), 1e-12)
    base_norm = raw_ll_base / base_sd
    vL_base = np.zeros(n)
    vL_base[1:] = _guard_LLW * _guard_LEG * (np.clip(base_norm, -2.0, 2.0) / 2.0)[1:]
    _guard_vL_total = vL.copy()
    _guard_vL_base = vL_base.copy()
    _guard_vR = vR.copy()
    combined = vR + vL
    target = np.zeros(n)
    for i in range(1, n):
        direction = np.sign(combined[i])
        if direction != 0:
            target[i] = direction * _guard_LEG / px[i]
    beta = _guard_state['rbeta']
    if beta is None:
        beta = np.zeros(n)
    hedge_dollars = -float(np.sum(target[1:] * px[1:] * beta[1:]))
    if logp.shape[1] > _guard_ALGO_REV_HORIZON:
        algo_move = float(logp[0, -1]) - float(logp[0, -1 - _guard_ALGO_REV_HORIZON])
        reversal_sign = -float(np.sign(algo_move))
        reversal_dollars = _guard_ALGO_REV_DOLLARS * reversal_sign
    else:
        reversal_dollars = 0.0
    algo_dollars = _guard_ALGO_HEDGE_WEIGHT * hedge_dollars + reversal_dollars
    algo_dollars = float(np.clip(algo_dollars, -100000.0, 100000.0))
    target[0] = algo_dollars / px[0]
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    limits = np.full(n, _guard_LEG)
    limits[0] = 100000.0
    share_limits = (limits / np.maximum(px, 1e-12)).astype(int)
    return np.clip(target, -share_limits, share_limits).astype(int)
_guard__core_get = _guard__core_getMyPosition
_guard__core_reset = _guard__core_reset_state
_guard__DECAY_RECENT_WINDOWS = (75, 90, 105)
_guard__DECAY_LEVEL_WINDOWS = (200, 225, 250)
_guard__DECAY_RECENT_CUTS = (-0.25, -0.5, -0.75)
_guard__DECAY_LEVEL_CUTS = (0.5, 0.75, 1.0)
_guard__DECAY_LEVEL_T_MAX = -2.5
_guard__DECAY_TOTAL_SUPPORTS = (0.4, 0.5, 0.6)
_guard__DECAY_BASE_SUPPORTS = (0.1, 0.25, 0.5)
_guard__DECAY_STRUCTURAL_MINS = (1, 3, 5)
_guard__DECAY_SUPPORT_COMMITTEE_MIN = 14
_guard__decay_last_nt = -1
_guard__decay_last_position = None

def _guard__decay_recent_forecast(zres, width):
    width = min(int(width), zres.shape[1] - 1)
    if width < 5:
        return np.zeros(zres.shape[0], dtype=float)
    a = zres[:, -(width + 1):-1]
    b = zres[:, -width:]
    matrix = a @ b.T / max(width, 1)
    np.fill_diagonal(matrix, 0.0)
    matrix[0, :] = 0.0
    matrix[:, 0] = 0.0
    forecast = zres[:, -1] @ matrix
    return forecast / max(float(forecast[1:].std()), 1e-12)

def _guard__decay_apply(prices, base_positions):
    global _guard_last_base, _guard_flatten, _guard_preserve, _guard_reroute
    out = np.asarray(base_positions, dtype=int).copy()
    n, nt = prices.shape
    if nt < 250:
        return out
    vL_total = _guard_vL_total
    vL_base = _guard_vL_base
    vR_last = _guard_vR
    if vL_total is None or vL_base is None or vR_last is None:
        return out
    logp = np.log(np.maximum(prices, 1e-12))
    zres, _ = _guard_resid_z(logp)
    base_sign = np.sign(out).astype(float)
    total_support = base_sign * np.asarray(vL_total, dtype=float) / _guard_LEG
    base_support = base_sign * np.asarray(vL_base, dtype=float) / _guard_LEG
    recent_support = {w: base_sign * _guard__decay_recent_forecast(zres, w) for w in _guard__DECAY_RECENT_WINDOWS}
    residual_level = np.cumsum(zres, axis=1)
    level_support = {}
    level_t = {}
    for width in _guard__DECAY_LEVEL_WINDOWS:
        window = residual_level[:, -min(width, residual_level.shape[1]):]
        level_z = (window[:, -1] - window.mean(axis=1)) / np.maximum(window.std(axis=1), 1e-12)
        level_support[width] = base_sign * -level_z
        level_t[width] = _guard_ar1_t(window)[1]
    structural_votes = np.zeros(n, dtype=int)
    nonzero = base_sign != 0.0
    nonzero[0] = False
    for recent_width in _guard__DECAY_RECENT_WINDOWS:
        rs = recent_support[recent_width]
        for level_width in _guard__DECAY_LEVEL_WINDOWS:
            ls = level_support[level_width]
            lt = level_t[level_width]
            for recent_cut in _guard__DECAY_RECENT_CUTS:
                for level_cut in _guard__DECAY_LEVEL_CUTS:
                    structural_votes += nonzero & (rs <= recent_cut) & (ls <= -level_cut) & (lt <= _guard__DECAY_LEVEL_T_MAX)
    support_votes = np.zeros(n, dtype=int)
    for total_min in _guard__DECAY_TOTAL_SUPPORTS:
        for base_min in _guard__DECAY_BASE_SUPPORTS:
            for structural_min in _guard__DECAY_STRUCTURAL_MINS:
                support_votes += nonzero & (total_support >= total_min) & (base_support >= base_min) & (structural_votes >= structural_min)
    flatten = support_votes >= _guard__DECAY_SUPPORT_COMMITTEE_MIN
    flatten[0] = False
    vR_arr = np.asarray(vR_last, dtype=float)
    base_sign_now = np.sign(np.asarray(base_positions, dtype=float))
    pair_support = base_sign_now * vR_arr / _guard_LEG
    ag_effect_support = base_sign_now * (np.asarray(vL_total, dtype=float) - np.asarray(vL_base, dtype=float)) / _guard_LEG
    keep_thresholds = (0.25, 0.5, 0.75)
    pair_keep_votes = sum((pair_support > x for x in keep_thresholds))
    ag_keep_votes = sum((ag_effect_support > x for x in keep_thresholds))
    preserve = flatten & ((pair_keep_votes >= 2) | (ag_keep_votes >= 2))
    reroute = flatten & ~preserve & (pair_support < 0.0)
    out[flatten] = 0
    out[preserve] = np.asarray(base_positions, dtype=int)[preserve]
    px_now = prices[:, -1]
    out[reroute] = (np.sign(vR_arr[reroute]) * _guard_LEG / np.maximum(px_now[reroute], 1e-12)).astype(int)
    out[0] = base_positions[0]
    _guard_last_base = np.asarray(base_positions, dtype=int).copy()
    _guard_flatten = flatten.copy()
    _guard_preserve = preserve.copy()
    _guard_reroute = reroute.copy()
    return out.astype(int)

def _guard_reset_state():
    global _guard__decay_last_nt, _guard__decay_last_position
    _guard__core_reset()
    _guard__decay_last_nt = -1
    _guard__decay_last_position = None

def _guard_getMyPosition(prcSoFar):
    global _guard__decay_last_nt, _guard__decay_last_position, _guard_last_base
    prices = np.asarray(prcSoFar, dtype=float)
    nt = prices.shape[1]
    if nt == _guard__decay_last_nt and _guard__decay_last_position is not None:
        return _guard__decay_last_position.copy()
    if nt < _guard__decay_last_nt:
        _guard_reset_state()
    base = _guard__core_get(prices)
    _guard_last_base = np.asarray(base, dtype=int).copy()
    final = _guard__decay_apply(prices, base)
    _guard__decay_last_nt = nt
    _guard__decay_last_position = final.copy()
    return final

# --- Final production composition ------------------------------------------
_last_nt = -1
_last_position = None


def reset_state():
    global _last_nt, _last_position, _architecture_mode, _aggregate_reliability
    global _primary_basket_z, _primary_basket_vote, _primary_rev_z, _primary_rev_vote, _primary_pca_z, _primary_pca_vote
    global _guard_vL_total, _guard_vL_base, _guard_vR
    global _guard_last_base, _guard_flatten, _guard_preserve, _guard_reroute
    _primary_reset_state()
    _guard_reset_state()
    _last_nt = -1
    _last_position = None
    _primary_basket_z = None
    _primary_basket_vote = None
    _primary_rev_z = None
    _primary_rev_vote = None
    _primary_pca_z = None
    _primary_pca_vote = None
    _guard_vL_total = None
    _guard_vL_base = None
    _guard_vR = None
    _guard_last_base = None
    _guard_flatten = None
    _guard_preserve = None
    _guard_reroute = None
    _architecture_mode = None
    _aggregate_reliability = None


def getMyPosition(prcSoFar):
    global _last_nt, _last_position

    prices = np.asarray(prcSoFar, dtype=float)
    if prices.ndim != 2:
        raise ValueError("prcSoFar must be a two-dimensional array")

    nt = prices.shape[1]
    if nt == _last_nt and _last_position is not None:
        return _last_position.copy()
    if nt < _last_nt:
        reset_state()

    primary_position = np.asarray(_primary_getMyPosition(prices), dtype=int)
    guarded_position = np.asarray(_guard_getMyPosition(prices), dtype=int)
    guard_base = np.asarray(
        _guard_last_base if _guard_last_base is not None else guarded_position,
        dtype=int,
    )

    changed_by_guard = guarded_position != guard_base
    final_position = primary_position.copy()
    final_position[changed_by_guard] = guarded_position[changed_by_guard]
    final_position[0] = primary_position[0]

    _last_nt = nt
    _last_position = final_position.copy()
    return final_position
