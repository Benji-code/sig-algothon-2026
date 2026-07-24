# protocol.py - frozen-strategy comparison & diagnosis (days 501-750)
# Implements the agreed protocol. Window (501,750) is DIAGNOSIS ONLY: this
# script never tunes anything, and frozen files are opened read-only.
#
# Commands:
#   python protocol.py freeze <file> <label> <recorded_score>
#       -> copies to frozen/, sha256, auto-extracts CONSTANTS, manifest.jsonl
#   python protocol.py run <prices.txt>
#       -> sections 3,4,5,6 for CONFIG below (one sim per file, rest derived)
#   python protocol.py log "<hypothesis>" <file> <minF> <notes...>
#       -> appends one trial to trials.jsonl (s7 registry)
#   python protocol.py deflate <K> [score_SE]
#       -> expected max-of-K inflation, default SE=90
#
# Strategy files are treated as BLACK BOXES exposing getMyPosition(prcSoFar).
# Sleeve removal (s4/s5) needs labeled variant copies (originals untouched).
# For the Sigma8/9 family the one-line zeroings are:
#   gray off: GW=0   lead-lag off: LLW=0   birth off: BIRTHW=0
#   seeds off: SEED_PAIRS=[]
# Hedge and per-instrument attribution need no variant: they come exactly
# from the stored per-instrument PnL matrix.
#
import sys, os, json, hashlib, shutil, re, importlib.util
import numpy as np

CONFIG = dict(
    A=("frozen/sigma61_887.py", "Sigma6.1_887"),
    B=("frozen/sigma8_705.py",  "Sigma8_705"),
    window=(501, 750),              # 1-indexed PnL days
    variants=[                       # (label, path) copies of B, sleeve off
        # ("B-noGray",  "frozen/sigma8_GW0.py"),
        ("B-noLL",    "frozen/sigma8_LLW0.py"),
        # ("B-noBirth", "frozen/sigma8_BW0.py"),
        # ("B-noSeeds", "frozen/sigma8_NS.py"),
    ],
    comm=1e-4, comm_algo=2e-5,       # VERIFY against eval.py
    dlr=10_000.0, algo_lim=100_000.0, idx=0, start_col=60,
)


# ------------------------------------------------------------------ infra
def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def load_fn(path, tag):
    spec = importlib.util.spec_from_file_location(f"strat_{tag}_{sha(path)}",
                                                  path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def official(pl):
    mu, sd = pl.mean(), pl.std()
    if mu <= 0 or sd < 1e-12:
        return float(mu)
    sr2 = 250.0 * (mu / sd) ** 2
    return float(mu * sr2 / (sr2 + 1.0))


# ------------------------------------------------------------- simulation
def run_positions(path, tag, prc):
    """One pass. Returns POS[k,n] for k=0..K-1 where POS[k] is held over
    transition col t=start+k -> t+1 (PnL day t+2, 1-indexed). Also polls
    module-level `state` (Sigma8/9 family) for active-unit counts."""
    m = load_fn(path, tag)
    n, T = prc.shape
    s0 = CONFIG["start_col"]
    POS, counts = [], []
    last_scan = None
    for t in range(s0, T - 1):
        POS.append(np.asarray(m.getMyPosition(prc[:, :t + 1]), float))
        st = getattr(m, "state", None)
        if isinstance(st, dict) and st.get("scanned") != last_scan:
            last_scan = st.get("scanned")
            counts.append((t + 2, len(st.get("units", [])),
                           len(st.get("gunits", [])),
                           len(st.get("bunits", []))))
    return np.array(POS), counts


def derive(POS, prc, cost_mult=1.0, delay=0, cap=1.0, drop_days=None):
    """Per-instrument daily net PnL from stored positions. All stresses are
    post-processing; nothing is refit."""
    n, T = prc.shape
    s0 = CONFIG["start_col"]
    cm = np.full(n, CONFIG["comm"]); cm[CONFIG["idx"]] = CONFIG["comm_algo"]
    P = POS.copy()
    if delay:
        P = np.vstack([np.zeros((delay, n)), P[:-delay]])
    K = P.shape[0]
    cols = np.arange(s0, s0 + K)
    if cap < 1.0:
        px = prc[:, cols].T
        lim = np.full(n, CONFIG["dlr"]); lim[CONFIG["idx"]] = CONFIG["algo_lim"]
        sh = cap * lim[None, :] / px
        P = np.clip(P, -sh, sh)
    px_t = prc[:, cols].T
    dpx = prc[:, cols + 1].T - px_t
    dP = np.vstack([P[0:1], np.diff(P, axis=0)])
    M = P * dpx - cost_mult * (np.abs(dP) * px_t) * cm[None, :]
    days = cols + 2
    if drop_days is not None:
        keep = ~np.isin(days, drop_days)
        return M[keep], days[keep]
    return M, days


def wsel(days, a, b):
    return (days >= a) & (days <= b)


def stats(pl):
    mu, sd = pl.mean(), pl.std()
    sr = np.sqrt(250) * mu / max(sd, 1e-12)
    cum = np.cumsum(pl)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    return official(pl), mu, sd, sr, dd


def srow(name, pl, vol=None):
    o, mu, sd, sr, dd = stats(pl)
    v = f" vol={vol/1e6:6.1f}M" if vol is not None else ""
    print(f"  {name:22s} score={o:8.2f} mu={mu:7.1f} sd={sd:7.0f} "
          f"SR={sr:5.2f} maxDD={dd:8.0f}{v}")
    return o


# ------------------------------------------------------------------ main
def run(prices_path):
    raw = np.loadtxt(prices_path)
    prc = raw.T if raw.shape[0] > raw.shape[1] else raw
    n, T = prc.shape
    a, b = CONFIG["window"]
    pA, lA = CONFIG["A"]; pB, lB = CONFIG["B"]
    print(f"prices {n} x {T}; window days {a}-{b}; "
          f"A={lA} [{sha(pA)}]  B={lB} [{sha(pB)}]\n")

    POSA, cntA = run_positions(pA, "A", prc)
    POSB, cntB = run_positions(pB, "B", prc)
    MA, days = derive(POSA, prc)
    MB, _ = derive(POSB, prc)
    w = wsel(days, a, b)
    dA, dB = MA[w].sum(1), MB[w].sum(1)
    cmv = np.full(n, CONFIG["comm"]); cmv[CONFIG["idx"]] = CONFIG["comm_algo"]

    def vol_of(POS):
        cols = np.arange(CONFIG["start_col"], CONFIG["start_col"] + POS.shape[0])
        dP = np.vstack([POS[0:1], np.diff(POS, axis=0)])
        return float((np.abs(dP) * prc[:, cols].T).sum(1)[w[:POS.shape[0]]].sum())

    print(f"== s3: head-to-head on days {a}-{b} ==")
    srow(lA, dA, vol_of(POSA))
    srow(lB, dB, vol_of(POSB))
    rho = float(np.corrcoef(dA, dB)[0, 1])
    d = dB - dA
    se = d.std() / np.sqrt(len(d)) + 1e-12
    print(f"  corr(A,B)={rho:5.2f}   B-A: d_mu={d.mean():7.2f}/day "
          f"paired_t={d.mean()/se:5.2f}  score_gap={official(dB)-official(dA):7.1f}")
    print("  5x50d block scores:")
    for nm, dd_ in ((lA, dA), (lB, dB)):
        bl = [official(x) for x in np.array_split(dd_, 5)]
        print(f"    {nm:18s} " + " ".join(f"{x:8.1f}" for x in bl)
              + f"   pos_blocks={sum(x>0 for x in bl)}/5")
    r80 = [official(dB[i:i + 80]) for i in range(0, len(dB) - 80 + 1, 10)]
    print(f"  B rolling80: min={min(r80):7.1f} med={np.median(r80):7.1f} "
          f"max={max(r80):7.1f}")

    print(f"\n== s4: attribution of {lB} ==")
    inst = MB[w].sum(0); tot = inst.sum()
    order = np.argsort(-np.abs(inst))
    print(f"  ALGO/hedge leg PnL = {inst[CONFIG['idx']]:+.0f} "
          f"({inst[CONFIG['idx']]/max(tot,1e-9):+.1%} of total)")
    print("  top instruments:   " +
          " ".join(f"[{i}]{inst[i]:+.0f}" for i in order[:8]))
    print(f"  best-instrument share = {inst[order[0]]/max(tot,1e-9):5.1%} "
          "(pair share = sum of its two legs; needs unit list)")
    dso = np.argsort(dB)
    print("  top5 days:  " + " ".join(f"d{days[w][i]}:{dB[i]:+.0f}"
                                      for i in dso[-5:][::-1]))
    print("  worst5 days:" + " ".join(f"d{days[w][i]}:{dB[i]:+.0f}"
                                      for i in dso[:5]))
    t5 = dB[dso[-5:]].sum()
    mu_, sd_ = dB.mean(), dB.std()
    sim = np.random.default_rng(0).normal(mu_, sd_, (4000, len(dB)))
    exp5 = np.sort(sim, 1)[:, -5:].sum(1).mean()
    print(f"  top5-day PnL={t5:.0f} ({t5/max(tot,1e-9):.1%}); Gaussian-null "
          f"expects {exp5:.0f} ({exp5/max(tot,1e-9):.1%}) - excess is the signal")
    if cntB:
        u = np.array([c[1:] for c in cntB])
        print(f"  active units/rescan (seed,gray,birth): "
              f"mean {u.mean(0).round(1)}  last {u[-1]}")
    for lab, vp in CONFIG["variants"]:
        PV, _ = run_positions(vp, lab, prc)
        MV, dv = derive(PV, prc)
        loo = official(dB) - official(MV[wsel(dv, a, b)].sum(1))
        print(f"  LOO {lab:12s} sleeve worth {loo:+8.2f} score")

    print(f"\n== s5: stress (post-hoc on frozen positions; no refits) ==")
    for nm, POS, base in ((lA, POSA, dA), (lB, POSB, dB)):
        print(f"  {nm}: base score {official(base):.2f}")
        for tag, kw in (("cost x1.25", dict(cost_mult=1.25)),
                        ("cost x1.5",  dict(cost_mult=1.5)),
                        ("delay 1d",   dict(delay=1)),
                        ("caps 75%",   dict(cap=0.75))):
            Mx, dx = derive(POS, prc, **kw)
            srow("  " + tag, Mx[wsel(dx, a, b)].sum(1))
        top5d = days[w][np.argsort(base)[-5:]]
        Mx, dx = derive(POS, prc, drop_days=top5d)
        srow("  drop best5", Mx[wsel(dx, a, b)].sum(1))

    print(f"\n== s6: confidence checklist for {lB} ==")
    blB = [official(x) for x in np.array_split(dB, 5)]
    Mx, dx = derive(POSB, prc, drop_days=days[w][np.argsort(dB)[-5:]])
    d5 = official(Mx[wsel(dx, a, b)].sum(1))
    Mc, dc = derive(POSB, prc, cost_mult=1.5)
    c15 = official(Mc[wsel(dc, a, b)].sum(1))
    checks = [
        ("beats A, paired_t > 2",           d.mean() / se > 2),
        ("profitable >= 4/5 blocks",        sum(x > 0 for x in blB) >= 4),
        ("best instrument < 30% of PnL",    inst[order[0]] / max(tot, 1e-9) < .30),
        ("drop-best5 still > A",            d5 > official(dA)),
        ("cost x1.5 still > A",             c15 > official(dA)),
        ("hedge not the profit centre",     abs(inst[CONFIG['idx']]) < .5 * abs(tot)),
    ]
    for txt, ok in checks:
        print(f"  [{'x' if ok else ' '}] {txt}")
    print("  (sleeve-contribution box needs the LOO variants configured)")
    print("\nreminder: window", CONFIG["window"], "is now SPENT for selection."
          "\nnext unseen segment is the only clean confirmation (s7).")


# ------------------------------------------------------- freeze & registry
def freeze(path, label, score):
    os.makedirs("frozen", exist_ok=True)
    dst = os.path.join("frozen", f"{label}.py")
    shutil.copyfile(path, dst)
    consts = dict(re.findall(r"^([A-Z][A-Z0-9_]*)\s*=\s*(.+?)\s*$",
                             open(path).read(), re.M))
    rec = dict(label=label, src=path, frozen=dst, sha256=sha(dst),
               recorded_score=score, constants=consts)
    with open("frozen/manifest.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"frozen {label}: {dst} [{rec['sha256']}] score={score} "
          f"({len(consts)} constants recorded)")


def log_trial(argv):
    rec = dict(hypothesis=argv[0], file=argv[1], sha256=sha(argv[1]),
               minF=float(argv[2]), notes=" ".join(argv[3:]))
    with open("trials.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    k = sum(1 for _ in open("trials.jsonl"))
    print(f"trial {k} logged. current deflation at K={k}:")
    deflate(k)


def deflate(K, se=90.0):
    K = int(K)
    mx = np.random.default_rng(0).normal(0, se, (20000, max(K, 1))).max(1)
    print(f"  expected max-of-{K} inflation = {mx.mean():.0f} "
          f"(90th pct {np.percentile(mx,90):.0f}) at score SE={se:.0f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "freeze":
        freeze(sys.argv[2], sys.argv[3], float(sys.argv[4]))
    elif cmd == "log":
        log_trial(sys.argv[2:])
    elif cmd == "deflate":
        deflate(sys.argv[2], *(float(x) for x in sys.argv[3:4]))
    else:
        run(sys.argv[2] if len(sys.argv) > 2 else "prices.txt")