#!/usr/bin/env python3
"""
eval.py - Algothon 2026 honest evaluator + selection protocol.

    python3 eval.py            # full protocol (writes results.csv, SUMMARY.txt)
    python3 eval.py --quick    # official window on shipped Sigma0.py defaults

What "honest" means here
------------------------
A sliding-window eval with step 50 controls look-ahead *within* a window but
its windows overlap ~80%, so its "n" is inflated ~5x and its worst-window is
an order statistic of correlated draws. And no in-file eval can control for
the selection bias of eight strategy generations tuned on this same path.
This protocol:

  OFFICIAL   last 250 days (train 250 -> test 251..500). The structural
             analog of the hidden window (strategy will have >=500d history).
  FOLDS      non-overlapping 80d test windows, 5d embargo, from day 150.
             Independent looks. Early folds are history-starved (pessimistic
             but diagnostic for mechanism fragility).
  SELECTION  pre-registered before any number was seen:
             winner = argmax over configs of MIN(official, all folds),
             tie-break by mean. Official alone is NOT the criterion.
  PBO        CSCV (Bailey/Lopez de Prado) over the config grid on official-
             window daily PnL: P(the in-sample-best config is below-median
             out-of-sample). >0.5 => selecting on IS score is anti-predictive.
  BOOTSTRAP  stationary block bootstrap of the winner's official PnL: the
             noise band around the headline. Captures path noise only, NOT
             lineage selection bias.
  DSR        deflated Sharpe at K=12 (this grid) and K=500 (honest guess at
             total trials across the Sigma lineage's sweeps).

Caveat that no eval can remove: seeds and every constant were chosen with
knowledge of this whole 500-day path, so even fold scores are optimistic
relative to the hidden window. These numbers bound overfitting; they cannot
prove alpha.
"""
import argparse, importlib.util, itertools, math, os, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

SCORE_PARAM = 1.0
DEFAULT_COMM = 0.0001
INST0_COMM = 0.00002
DEFAULT_LIMIT = 10_000.0
INST0_LIMIT = 100_000.0
GAMMA = 0.5772156649015329
STRAT = "Sigma0.py"


# --------------------------------------------------------------- plumbing
def load_prices(path):
    with open(path) as f:
        first = f.readline().split()
    def num(tok):
        try:
            float(tok); return True
        except ValueError:
            return False
    if first and all(num(t) for t in first):
        arr = np.loadtxt(path)
    else:
        import pandas as pd
        arr = pd.read_csv(path, sep=r"\s+", header=0).values
    return arr.T


def score(mu, sigma, param=SCORE_PARAM):
    if mu <= 0 or sigma < 1e-10:
        return float(mu)
    sr = math.sqrt(250.0) * mu / sigma
    return float(mu * (sr * sr) / (sr * sr + param * param))


def load_strategy(path, overrides=None):
    name = "strat_" + str(abs(hash((path, os.urandom(8)))))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    if overrides:
        # Case 1: strategy is written with a CFG dict
        if hasattr(mod, "CFG") and isinstance(mod.CFG, dict):
            mod.CFG.update(overrides)

        # Case 2: strategy is written with module-level constants
        # e.g. GLO=-2.6, MAX_SNG=6, BIRTHW=150
        else:
            for k, v in overrides.items():
                setattr(mod, k, v)

    return mod.getMyPosition


def calc_pl(prc, path, train_end, test_len, embargo=0, overrides=None):
    n_inst, n_days = prc.shape
    get_position = load_strategy(path, overrides)
    comm = np.full(n_inst, DEFAULT_COMM); comm[0] = INST0_COMM
    dlim = np.full(n_inst, DEFAULT_LIMIT); dlim[0] = INST0_LIMIT
    start_day = train_end + embargo
    end_day = start_day + test_len
    if start_day < 2 or end_day > n_days:
        raise ValueError(f"bad window te={train_end} len={test_len}")
    cash = 0.0; cur_pos = np.zeros(n_inst)
    total_dvol = 0.0; value = 0.0; pending = 0.0
    daily = []
    for t in range(start_day, end_day + 1):
        hist = prc[:, :t]
        cur_prices = hist[:, -1]
        if t < end_day:
            raw = np.asarray(get_position(hist)).astype(float)
            plim = (dlim / cur_prices).astype(int)
            new_pos = np.clip(raw, -plim, plim).astype(int)
        else:
            new_pos = np.array(cur_pos)
        delta = new_pos - cur_pos
        cash -= cur_prices.dot(delta) + pending
        dv = cur_prices * np.abs(delta)
        total_dvol += float(np.sum(dv))
        pending = float(np.sum(dv * comm))
        cur_pos = np.array(new_pos, dtype=float)
        value_new = cash + float(cur_pos.dot(cur_prices))
        today = value_new - value
        value = value_new
        if t > start_day:
            daily.append(today)
    pl = np.asarray(daily, float)
    mu = float(pl.mean()); sd = float(pl.std())
    shrp = float(math.sqrt(250.0) * mu / sd) if sd > 0 else 0.0
    return dict(mean=mu, std=sd, sharpe=shrp, score=score(mu, sd),
                dvol=total_dvol, pl=pl)


# ------------------------------------------------------------ diagnostics
def stationary_bootstrap(pl, n_boot=2000, mean_block=20, seed=0):
    rng = np.random.default_rng(seed)
    T = len(pl); p = 1.0 / mean_block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(T, dtype=int)
        i = rng.integers(0, T)
        for k in range(T):
            idx[k] = i
            i = rng.integers(0, T) if rng.random() < p else (i + 1) % T
        s = pl[idx]
        out[b] = score(float(s.mean()), float(s.std()))
    return out


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p):
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plo, phi = 0.02425, 1 - 0.02425
    if p < plo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phi:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def deflated_sharpe(pl, n_trials):
    T = len(pl)
    mu, sd = pl.mean(), pl.std(ddof=1)
    if sd <= 0:
        return None
    sr = mu / sd
    m3 = ((pl - mu) ** 3).mean() / sd ** 3
    m4 = ((pl - mu) ** 4).mean() / sd ** 4
    var_sr = (1.0 - m3 * sr + (m4 - 1.0) / 4.0 * sr * sr) / (T - 1)
    v = math.sqrt(max(var_sr, 1e-18))
    K = max(int(n_trials), 2)
    sr0 = v * ((1 - GAMMA) * _norm_ppf(1 - 1.0 / K) +
               GAMMA * _norm_ppf(1 - 1.0 / (K * math.e)))
    den = math.sqrt(max(1.0 - m3 * sr + (m4 - 1.0) / 4.0 * sr * sr, 1e-12))
    z = (sr - sr0) * math.sqrt(T - 1) / den
    return dict(sr_ann=sr * math.sqrt(250), sr0_ann=sr0 * math.sqrt(250),
                dsr=_norm_cdf(z), K=K)


def pbo_cscv(mat, n_splits=8):
    T, C = mat.shape
    n_splits -= n_splits % 2
    if C < 2 or n_splits < 4:
        return None
    bounds = np.linspace(0, T, n_splits + 1).astype(int)
    blocks = [mat[bounds[i]:bounds[i + 1]] for i in range(n_splits)]
    below = 0
    combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    for tr in combos:
        te = tuple(i for i in range(n_splits) if i not in tr)
        A = np.vstack([blocks[i] for i in tr])
        B = np.vstack([blocks[i] for i in te])
        s_is = [score(A[:, c].mean(), A[:, c].std()) for c in range(C)]
        s_oos = np.array([score(B[:, c].mean(), B[:, c].std())
                          for c in range(C)])
        best = int(np.argmax(s_is))
        rank = (float((s_oos < s_oos[best]).sum()) + 1) / (C + 1)
        if rank <= 0.5:
            below += 1
    return dict(pbo=below / len(combos), n_combos=len(combos))


# ------------------------------------------------------------------ grid
S61 = dict(GLO=-2.6, GRAY_UNIFY=0, ALGO_MODE="rev5", MAX_SNG=6,
           SEED_SINGLES=[35, 33, 18, 44], PAIR_THR=0, BIRTHW=0)
S7P = dict(GLO=-9.0, MAX_SNG=0, SEED_SINGLES=[], ALGO_MODE="hedge",
           PAIR_THR=0, BIRTHW=0)
S9 = dict(GLO=-2.6, GRAY_UNIFY=0, ALGO_MODE="hedge", MAX_SNG=0,
          SEED_SINGLES=[], PAIR_MODE="binary", PAIR_THR=60, PAIR_MIN=30,
          BIRTHW=150, BGATE=-4.0)

CONFIGS = {
    # ancestors, faithful
    "s61":         S61,                                   # Ben's Sigma6.1
    "s7p":         S7P,                                   # probe
    "s9":          S9,                                    # friend's Sigma9
    # my Sigma8 stance: tight net + t-gates + gated ALGO
    "s8":          dict(GLO=-3.0, GRAY_UNIFY=1, ALGO_MODE="gated",
                        PAIR_MODE="t", PAIR_THR=250, PAIR_MIN=40,
                        GTHR=250, GTHR_MIN=30, BIRTHW=0, MAX_SNG=6,
                        SEED_SINGLES=[35, 33, 18, 44]),
    # single-mechanism ablations of s9
    "s9_nobirth":  {**S9, "BIRTHW": 0},
    "s9_nougate":  {**S9, "PAIR_THR": 0},
    "s9_unify":    {**S9, "GRAY_UNIFY": 1},
    "s9_tgate":    {**S9, "PAIR_MODE": "t", "PAIR_THR": 250, "PAIR_MIN": 40},
    "s9_algogate": {**S9, "ALGO_MODE": "gated"},
    # merges
    "s10":         {**S9, "GRAY_UNIFY": 1, "ALGO_MODE": "gated"},
    "s10_glo30":   {**S9, "GRAY_UNIFY": 1, "ALGO_MODE": "gated",
                    "GLO": -3.0},
    "s10_sngl":    {**S9, "GRAY_UNIFY": 1, "ALGO_MODE": "gated",
                    "MAX_SNG": 6, "SEED_SINGLES": [35, 33, 18, 44]},
}

FOLD_LEN, FOLD_EMB, FOLD_MIN = 80, 5, 150


def make_folds(n_days):
    out, te = [], FOLD_MIN
    while te + FOLD_EMB + FOLD_LEN <= n_days:
        out.append(te)
        te += FOLD_LEN + FOLD_EMB
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=STRAT)
    ap.add_argument("--prices", default="prices.txt")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    prc = load_prices(a.prices)
    n_inst, n_days = prc.shape
    off_te = n_days - 250
    print(f"instruments {n_inst}  days {n_days}  official window: "
          f"train {off_te} -> test {off_te+1}..{n_days}", flush=True)

    if a.quick:
        r = calc_pl(prc, a.strategy, off_te, 250)
        print(f"OFFICIAL  mean {r['mean']:.2f}  std {r['std']:.2f}  "
              f"sharpe {r['sharpe']:.2f}  Score {r['score']:.2f}")
        return

    folds = make_folds(n_days)
    print(f"folds (len {FOLD_LEN}, embargo {FOLD_EMB}): train_ends {folds}",
          flush=True)
    print("selection rule (pre-registered): argmax of MIN(official, folds); "
          "tie-break mean\n", flush=True)

    csv = open("results.csv", "w")
    csv.write("config,window,train_end,test_len,mean,std,sharpe,score\n")
    rows = {}          # config -> dict window->score
    off_pl = {}        # config -> official daily pl

    for name, ov in CONFIGS.items():
        t0 = time.time()
        r = calc_pl(prc, a.strategy, off_te, 250, 0, dict(ov))
        rows[name] = {"off": r["score"]}
        off_pl[name] = r["pl"]
        csv.write(f"{name},off,{off_te},250,{r['mean']:.2f},{r['std']:.2f},"
                  f"{r['sharpe']:.3f},{r['score']:.2f}\n"); csv.flush()
        print(f"[off ] {name:12s} mean {r['mean']:8.2f} sharpe "
              f"{r['sharpe']:6.2f} score {r['score']:8.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    for name, ov in CONFIGS.items():
        for fi, te in enumerate(folds, 1):
            r = calc_pl(prc, a.strategy, te, FOLD_LEN, FOLD_EMB, dict(ov))
            rows[name][f"f{fi}"] = r["score"]
            csv.write(f"{name},f{fi},{te},{FOLD_LEN},{r['mean']:.2f},"
                      f"{r['std']:.2f},{r['sharpe']:.3f},{r['score']:.2f}\n")
            csv.flush()
        fs = " ".join(f"{rows[name][f'f{k+1}']:7.1f}" for k in range(len(folds)))
        print(f"[fold] {name:12s} {fs}", flush=True)
    csv.close()

    # ---------------- selection ----------------
    names = list(CONFIGS)
    def worst(nm):
        return min(rows[nm].values())
    def meanv(nm):
        return float(np.mean(list(rows[nm].values())))
    order = sorted(names, key=lambda nm: (worst(nm), meanv(nm)), reverse=True)
    win = order[0]

    lines = []
    lines.append(f"{'config':12s} {'official':>9s} " +
                 " ".join(f"{'f'+str(k+1):>7s}" for k in range(len(folds))) +
                 f" {'worst':>8s} {'mean':>8s}")
    for nm in order:
        f = " ".join(f"{rows[nm][f'f{k+1}']:7.1f}" for k in range(len(folds)))
        lines.append(f"{nm:12s} {rows[nm]['off']:9.1f} {f} "
                     f"{worst(nm):8.1f} {meanv(nm):8.1f}")
    lines.append("")
    lines.append(f"WINNER (pre-registered rule): {win}")
    lines.append(f"  overrides: {CONFIGS[win]}")
    lines.append(f"  official {rows[win]['off']:.1f}   worst {worst(win):.1f}")

    argmax_off = max(names, key=lambda nm: rows[nm]["off"])
    lines.append(f"argmax-official (NOT the rule): {argmax_off} "
                 f"({rows[argmax_off]['off']:.1f}, worst "
                 f"{worst(argmax_off):.1f})")

    # ---------------- diagnostics ----------------
    T = min(len(v) for v in off_pl.values())
    mat = np.column_stack([off_pl[nm][:T] for nm in names])
    p = pbo_cscv(mat)
    if p:
        lines.append(f"\nPBO over grid (official window): {p['pbo']:.3f} "
                     f"({p['n_combos']} CSCV combos)")
        lines.append("  >0.5 = picking the IS-best config is anti-predictive; "
                     "<0.25 = selection carries signal")

    bs = stationary_bootstrap(off_pl[win])
    q = np.percentile(bs, [5, 25, 50, 75, 95])
    lines.append(f"\nbootstrap of {win} official score: "
                 f"p5 {q[0]:.0f}  p25 {q[1]:.0f}  p50 {q[2]:.0f}  "
                 f"p75 {q[3]:.0f}  p95 {q[4]:.0f}   "
                 f"P(<=0) {float((bs<=0).mean()):.3f}")
    lines.append("  (path noise only; lineage selection bias not included)")

    for K in (12, 500):
        d = deflated_sharpe(off_pl[win], K)
        if d:
            lines.append(f"DSR K={K:3d}: observed annSR {d['sr_ann']:.2f}  "
                         f"null-max {d['sr0_ann']:.2f}  "
                         f"P(skill) {d['dsr']:.4f}")

    lines.append("\nHONESTY NOTE: seeds/constants were chosen with knowledge "
                 "of this full path; folds bound overfitting, they cannot "
                 "prove alpha. The hidden window is the only clean test.")

    summary = "\n".join(lines)
    print("\n" + summary, flush=True)
    with open("SUMMARY.txt", "w") as f:
        f.write(summary + "\n")
    open("DONE", "w").write("done\n")


if __name__ == "__main__":
    main()