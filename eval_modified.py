#!/usr/bin/env python3
"""
Rolling / sliding-window evaluator for Algothon strategies.

Key convention: a window (train_end, test_len) starts by calling getMyPosition
with prices up to train_end. The position then earns PnL on subsequent days.
The strategy never sees prices after the current day t. This mimics live trading
and avoids using future test information when constructing lagged-return features.

Example:
    python3 eval_windows.py --strategy SigmaLaggedReturns.py --prices prices.txt \
        --test-windows 100,250 --min-train 250 --step 50
"""

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

SCORE_PARAM = 1.0
DEFAULT_COMM = 0.0001
INST0_COMM = 0.00002
DEFAULT_LIMIT = 10_000.0
INST0_LIMIT = 100_000.0


@dataclass
class WindowResult:
    train_end: int
    test_start: int
    test_end: int
    test_len: int
    mean_pl: float
    std_pl: float
    sharpe: float
    score: float
    ret: float
    dvol: float
    final_value: float


def load_prices(path):
    df = pd.read_csv(path, sep=r"\s+", header=0, index_col=None)
    return df.values.T, list(df.columns)


def score(mu, sigma, param=SCORE_PARAM):
    if mu <= 0 or sigma < 1e-10:
        return float(mu)
    sr = np.sqrt(250.0) * mu / sigma
    return float(mu * (sr * sr) / (sr * sr + param * param))


def load_strategy(path):
    # New module object per window, so global state cannot leak across windows.
    module_name = "strategy_" + str(abs(hash((path, os.urandom(8)))))
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod.getMyPosition


def calc_pl(prc, strategy_path, train_end, test_len, embargo=0, verbose=False):
    n_inst, n_days = prc.shape
    get_position = load_strategy(strategy_path)

    comm_rate = np.full(n_inst, DEFAULT_COMM, dtype=float)
    comm_rate[0] = INST0_COMM
    dlr_limit = np.full(n_inst, DEFAULT_LIMIT, dtype=float)
    dlr_limit[0] = INST0_LIMIT

    # We optionally skip some days after the training segment before scoring.
    # The strategy still only receives prices up to the current day.
    start_day = train_end + embargo
    end_day = start_day + test_len
    if start_day < 2 or end_day > n_days:
        raise ValueError(f"Invalid window: train_end={train_end}, embargo={embargo}, test_len={test_len}, n_days={n_days}")

    cash = 0.0
    cur_pos = np.zeros(n_inst, dtype=float)
    total_dvol = 0.0
    value = 0.0
    pending_comm = 0.0
    daily_pl = []

    for t in range(start_day, end_day + 1):
        hist = prc[:, :t]
        cur_prices = hist[:, -1]

        if t < end_day:
            raw_pos = get_position(hist)
            pos_limits = (dlr_limit / cur_prices).astype(int)
            new_pos = np.clip(raw_pos, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = np.array(cur_pos)

        delta = new_pos - cur_pos
        cash -= cur_prices.dot(delta) + pending_comm

        dvolumes = cur_prices * np.abs(delta)
        dvol = float(np.sum(dvolumes))
        total_dvol += dvol
        pending_comm = float(np.sum(dvolumes * comm_rate))

        cur_pos = np.array(new_pos, dtype=float)
        pos_value = float(cur_pos.dot(cur_prices))
        today_pl = cash + pos_value - value
        value = cash + pos_value

        if t > start_day:
            daily_pl.append(today_pl)
            if verbose:
                ret = value / total_dvol if total_dvol > 0 else 0.0
                print(f"Day {t:4d} PL {today_pl:10.2f} value {value:10.2f} dvol {total_dvol:12.0f} ret {ret:.5f}")

    pl = np.asarray(daily_pl, dtype=float)
    mu = float(pl.mean())
    sd = float(pl.std())
    shrp = float(np.sqrt(250.0) * mu / sd) if sd > 0 else 0.0
    ret = float(value / total_dvol) if total_dvol > 0 else 0.0
    return WindowResult(train_end, start_day, end_day, test_len, mu, sd, shrp, score(mu, sd), ret, total_dvol, value)


def make_windows(n_days, test_lens, min_train, step, embargo):
    windows = []
    for tl in test_lens:
        max_train_end = n_days - tl - embargo
        te = min_train
        while te <= max_train_end:
            windows.append((te, tl))
            te += step
        # Ensure the official last-window style split is included.
        if max_train_end >= min_train and (max_train_end, tl) not in windows:
            windows.append((max_train_end, tl))
    return sorted(set(windows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="Sigma0.py", help="Path to strategy .py containing getMyPosition")
    ap.add_argument("--prices", default="prices.txt", help="Path to whitespace-separated prices.txt")
    ap.add_argument("--test-windows", default="100,250", help="Comma-separated test lengths; use >=100 for real validation")
    ap.add_argument("--min-train", type=int, default=250, help="Minimum days before each test window")
    ap.add_argument("--step", type=int, default=50, help="Step between sliding train_end cutoffs")
    ap.add_argument("--embargo", type=int, default=0, help="Optional gap between train_end and first scored test day")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    prc, names = load_prices(args.prices)
    n_inst, n_days = prc.shape
    test_lens = [int(x) for x in args.test_windows.split(",") if x.strip()]

    print(f"Loaded {n_inst} instruments for {n_days} days")
    print(f"Strategy: {args.strategy}")
    print(f"Windows: test_lens={test_lens}, min_train={args.min_train}, step={args.step}, embargo={args.embargo}")

    windows = make_windows(n_days, test_lens, args.min_train, args.step, args.embargo)
    if not windows:
        raise SystemExit("No valid windows. Reduce min-train/test-window/embargo.")

    rows = []
    all_mu_weighted = []
    for train_end, test_len in windows:
        res = calc_pl(prc, args.strategy, train_end, test_len, args.embargo, args.verbose)
        rows.append(res)
        print(
            f"train_end={res.train_end:4d} test=({res.test_start+1:4d}-{res.test_end:4d}) "
            f"len={res.test_len:3d} mean={res.mean_pl:8.2f} std={res.std_pl:8.2f} "
            f"sharpe={res.sharpe:6.2f} score={res.score:8.2f} dvol={res.dvol:11.0f}"
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    print("===== window summary by test_len =====")
    summary = df.groupby("test_len").agg(
        n=("score", "count"),
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        min_score=("score", "min"),
        max_score=("score", "max"),
        mean_pl=("mean_pl", "mean"),
        mean_std=("std_pl", "mean"),
        mean_sharpe=("sharpe", "mean"),
    )
    print(summary.to_string(float_format=lambda x: f"{x:8.2f}"))

    print("===== all-window robustness =====")
    print(f"Mean window score:   {df['score'].mean():.2f}")
    print(f"Median window score: {df['score'].median():.2f}")
    print(f"Worst window score:  {df['score'].min():.2f}")
    print(f"Best window score:   {df['score'].max():.2f}")

    out_path = "window_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
