#!/usr/bin/env python3
"""
validate_overfitting.py

A comprehensive robustness and overfitting audit for a SIG Algothon strategy.

The script is designed for a strategy file exposing:

    getMyPosition(prcSoFar)

It uses only the supplied historical price data and performs five classes of
validation:

1. Chronological stability
   - restarted historical windows;
   - continuous 50/100/250-day block performance;
   - leave-one-time-block-out score sensitivity.

2. P&L concentration
   - per-instrument contribution;
   - score after removing each instrument's realised contribution;
   - score after removing ALGO;
   - score after removing the five largest contributors.

3. Mechanical stress
   - 1x, 2x, 3x and 5x commissions;
   - zero-, one- and two-day execution delay.

4. Placebo tests
   - independent circular shifts destroy cross-instrument timing;
   - common block permutation preserves same-day cross-section but destroys
     chronological order.

5. Relationship-decay simulations
   - generate days 1001-1500;
   - progressively mix multivariate historical blocks with independently
     sampled instrument histories;
   - retention=1 preserves historical cross-instrument structure;
   - retention=0 preserves each instrument's marginal behaviour but destroys
     cross-instrument dependence.

6. Parameter-neighbourhood tests
   - source-level one-at-a-time perturbations for recognised Sigma58 constants;
   - this is a stability audit, not a new optimisation search.

Important limitation
--------------------
No script can retroactively recreate a genuinely nested model-development
process unless the complete historical candidate-generation procedure is
available. This audit can expose fragility, concentration, parameter spikes
and dependence on preserved relationships, but cannot fully remove selection
bias from repeatedly inspecting the same 1,000 days.

Examples
--------
Core audit:

    python3 validate_overfitting.py \
        --strategy Sigma58.py \
        --prices prices_full_final.txt \
        --profile core

Full audit:

    python3 validate_overfitting.py \
        --strategy Sigma58.py \
        --prices prices_full_final.txt \
        --profile full \
        --workers 4

Faster smoke run:

    python3 validate_overfitting.py \
        --strategy Sigma58.py \
        --profile quick \
        --workers 4

Outputs are written below validation_output/ by default.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib.util
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Evaluator constants
# ---------------------------------------------------------------------------

SCORE_PARAM = 1.0
DEFAULT_COMMISSION = 0.0001
ALGO_COMMISSION = 0.00002
DEFAULT_LIMIT = 10_000.0
ALGO_LIMIT = 100_000.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trace:
    daily_pnl: np.ndarray
    instrument_daily_pnl: np.ndarray
    desired_positions: np.ndarray
    executed_positions: np.ndarray
    turnover: np.ndarray
    commission: np.ndarray
    test_start_day: int
    test_end_day: int


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def load_prices(path: str | Path) -> tuple[np.ndarray, list[str]]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline().strip().split()

    has_header = bool(first) and not all(
        is_number(token) for token in first
    )

    if has_header:
        values = np.loadtxt(path, skiprows=1)
        names = first
    else:
        values = np.loadtxt(path)
        names = [f"I{i}" for i in range(values.shape[1])]

    if values.ndim != 2:
        raise ValueError("Price file must be a two-dimensional table.")

    prices = values.T if values.shape[0] >= values.shape[1] else values

    if len(names) != prices.shape[0]:
        names = [f"I{i}" for i in range(prices.shape[0])]

    if np.any(~np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("All prices must be finite and positive.")

    return prices.astype(float), names


def score_from_daily_pnl(daily_pnl: np.ndarray) -> dict[str, float]:
    daily_pnl = np.asarray(daily_pnl, dtype=float)

    if daily_pnl.ndim != 1 or len(daily_pnl) == 0:
        return {
            "days": float(len(daily_pnl)),
            "mean_pl": float("nan"),
            "std_pl": float("nan"),
            "sharpe": float("nan"),
            "score": float("nan"),
            "total_pl": float("nan"),
            "max_drawdown": float("nan"),
            "positive_day_fraction": float("nan"),
        }

    mean_pl = float(daily_pnl.mean())
    std_pl = float(daily_pnl.std())

    sharpe = (
        math.sqrt(250.0) * mean_pl / std_pl
        if std_pl > 1e-12
        else 0.0
    )

    if mean_pl <= 0.0 or std_pl < 1e-10:
        score = mean_pl
    else:
        fraction = (
            sharpe * sharpe
            / (sharpe * sharpe + SCORE_PARAM * SCORE_PARAM)
        )
        score = mean_pl * fraction

    cumulative = np.cumsum(daily_pnl)
    running_peak = np.maximum.accumulate(
        np.concatenate([[0.0], cumulative])
    )[1:]
    drawdown = running_peak - cumulative

    return {
        "days": float(len(daily_pnl)),
        "mean_pl": mean_pl,
        "std_pl": std_pl,
        "sharpe": float(sharpe),
        "score": float(score),
        "total_pl": float(daily_pnl.sum()),
        "max_drawdown": float(drawdown.max(initial=0.0)),
        "positive_day_fraction": float(np.mean(daily_pnl > 0.0)),
    }


def write_rows(
    path: str | Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    if fieldnames is None:
        fieldnames = []

        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def format_metric_row(label: str, metrics: dict[str, float]) -> str:
    return (
        f"{label:34s} "
        f"score={metrics['score']:9.2f} "
        f"mean={metrics['mean_pl']:9.2f} "
        f"Sharpe={metrics['sharpe']:7.2f} "
        f"total={metrics['total_pl']:11.0f} "
        f"maxDD={metrics['max_drawdown']:10.0f}"
    )


def parse_float_list(text: str) -> list[float]:
    values = []

    for part in text.split(","):
        part = part.strip()

        if part:
            values.append(float(part))

    return values


# ---------------------------------------------------------------------------
# Strategy loading and tracing
# ---------------------------------------------------------------------------

def load_strategy(
    strategy_path: str | Path,
    unique_tag: str,
):
    strategy_path = Path(strategy_path).resolve()

    if not strategy_path.exists():
        raise FileNotFoundError(
            f"Strategy file not found: {strategy_path}"
        )

    strategy_directory = str(strategy_path.parent)

    if strategy_directory not in sys.path:
        sys.path.insert(0, strategy_directory)

    module_name = (
        f"_validation_strategy_{strategy_path.stem}_"
        f"{os.getpid()}_{unique_tag}"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        strategy_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {strategy_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    get_position = getattr(module, "getMyPosition", None)

    if not callable(get_position):
        raise AttributeError(
            f"{strategy_path.name} has no getMyPosition()."
        )

    reset = getattr(module, "reset_state", None)

    if callable(reset):
        reset()

    return get_position


def collect_desired_positions(
    prices: np.ndarray,
    strategy_path: str | Path,
    start_day_count: int,
    unique_tag: str,
) -> np.ndarray:
    """
    Collect the positions returned on each decision day.

    There are n_days - start_day_count decisions. The final price day is only
    used to realise the last held position.
    """
    prices = np.asarray(prices, dtype=float)
    n_instruments, n_days = prices.shape

    if not (1 <= start_day_count < n_days):
        raise ValueError("Invalid start_day_count.")

    get_position = load_strategy(
        strategy_path,
        unique_tag,
    )

    decisions = []

    for t in range(start_day_count, n_days):
        history = prices[:, :t]
        current_prices = history[:, -1]

        proposed = np.asarray(
            get_position(history),
            dtype=float,
        )

        if proposed.shape != (n_instruments,):
            raise ValueError(
                f"getMyPosition returned {proposed.shape}; "
                f"expected {(n_instruments,)}."
            )

        proposed = np.nan_to_num(
            proposed,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        limits = np.full(
            n_instruments,
            DEFAULT_LIMIT,
            dtype=float,
        )
        limits[0] = ALGO_LIMIT

        share_limits = (
            limits / np.maximum(current_prices, 1e-12)
        ).astype(int)

        proposed = np.clip(
            proposed,
            -share_limits,
            share_limits,
        ).astype(int)

        decisions.append(proposed)

    return np.asarray(decisions, dtype=int)


def account_positions(
    prices: np.ndarray,
    desired_positions: np.ndarray,
    start_day_count: int,
    commission_multiplier: float = 1.0,
    execution_delay: int = 0,
) -> Trace:
    """
    Apply official-style accounting to a fixed sequence of desired positions.

    execution_delay=1 means today's desired position is executed one decision
    day later. Initial delayed days remain flat.
    """
    prices = np.asarray(prices, dtype=float)
    desired_positions = np.asarray(
        desired_positions,
        dtype=int,
    )

    n_instruments, n_days = prices.shape
    n_decisions = n_days - start_day_count

    if desired_positions.shape != (
        n_decisions,
        n_instruments,
    ):
        raise ValueError(
            "desired_positions has the wrong shape: "
            f"{desired_positions.shape}"
        )

    if execution_delay < 0:
        raise ValueError("execution_delay cannot be negative.")

    commission_rate = np.full(
        n_instruments,
        DEFAULT_COMMISSION * commission_multiplier,
        dtype=float,
    )
    commission_rate[0] = (
        ALGO_COMMISSION * commission_multiplier
    )

    executed = np.zeros_like(
        desired_positions,
        dtype=int,
    )

    if execution_delay == 0:
        executed[:] = desired_positions
    elif execution_delay < n_decisions:
        executed[execution_delay:] = (
            desired_positions[:-execution_delay]
        )

    cash = np.zeros(n_instruments, dtype=float)
    current_position = np.zeros(
        n_instruments,
        dtype=float,
    )
    portfolio_value = np.zeros(
        n_instruments,
        dtype=float,
    )
    pending_commission = np.zeros(
        n_instruments,
        dtype=float,
    )

    instrument_daily_pnl = []
    turnover_rows = []
    commission_rows = []

    for offset, t in enumerate(
        range(start_day_count, n_days + 1)
    ):
        current_prices = prices[:, t - 1]

        if t < n_days:
            new_position = executed[offset].astype(float)
        else:
            new_position = current_position.copy()

        position_change = (
            new_position - current_position
        )

        cash -= (
            current_prices * position_change
            + pending_commission
        )

        dollar_turnover = (
            current_prices * np.abs(position_change)
        )
        pending_commission = (
            dollar_turnover * commission_rate
        )

        current_position = new_position

        new_portfolio_value = (
            cash + current_position * current_prices
        )
        today_pnl = (
            new_portfolio_value - portfolio_value
        )
        portfolio_value = new_portfolio_value

        if t > start_day_count:
            instrument_daily_pnl.append(today_pnl.copy())
            turnover_rows.append(dollar_turnover.copy())
            commission_rows.append(
                pending_commission.copy()
            )

    instrument_daily_pnl_array = np.asarray(
        instrument_daily_pnl,
        dtype=float,
    )
    daily_pnl = instrument_daily_pnl_array.sum(axis=1)

    return Trace(
        daily_pnl=daily_pnl,
        instrument_daily_pnl=instrument_daily_pnl_array,
        desired_positions=desired_positions,
        executed_positions=executed,
        turnover=np.asarray(turnover_rows, dtype=float),
        commission=np.asarray(
            commission_rows,
            dtype=float,
        ),
        test_start_day=start_day_count + 1,
        test_end_day=n_days,
    )


def run_trace(
    prices: np.ndarray,
    strategy_path: str | Path,
    start_day_count: int,
    unique_tag: str,
    commission_multiplier: float = 1.0,
    execution_delay: int = 0,
) -> Trace:
    desired = collect_desired_positions(
        prices,
        strategy_path,
        start_day_count,
        unique_tag,
    )

    return account_positions(
        prices,
        desired,
        start_day_count,
        commission_multiplier=commission_multiplier,
        execution_delay=execution_delay,
    )


# ---------------------------------------------------------------------------
# Chronological and concentration tests
# ---------------------------------------------------------------------------

def chronological_windows(
    prices: np.ndarray,
    strategy_path: str | Path,
    windows: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    rows = []

    for index, (start_day, end_day) in enumerate(windows):
        if start_day < 2 or end_day > prices.shape[1]:
            continue

        if start_day > end_day:
            continue

        subset = prices[:, :end_day]
        trace = run_trace(
            subset,
            strategy_path,
            start_day_count=start_day - 1,
            unique_tag=f"chronology_{index}_{start_day}_{end_day}",
        )
        metrics = score_from_daily_pnl(trace.daily_pnl)

        rows.append({
            "window": f"{start_day}-{end_day}",
            "start_day": start_day,
            "end_day": end_day,
            "days": int(metrics["days"]),
            **{
                key: value
                for key, value in metrics.items()
                if key != "days"
            },
        })

    return rows


def continuous_blocks(
    trace: Trace,
    block_sizes: Iterable[int],
) -> list[dict[str, Any]]:
    rows = []
    total_days = len(trace.daily_pnl)

    for block_size in block_sizes:
        block_size = int(block_size)

        if block_size <= 0:
            continue

        for offset in range(0, total_days, block_size):
            block = trace.daily_pnl[
                offset:offset + block_size
            ]

            if len(block) < max(20, block_size // 2):
                continue

            start_day = trace.test_start_day + offset
            end_day = start_day + len(block) - 1
            metrics = score_from_daily_pnl(block)

            rows.append({
                "block_size": block_size,
                "block": (
                    f"{start_day}-{end_day}"
                ),
                "start_day": start_day,
                "end_day": end_day,
                **metrics,
            })

    return rows


def leave_one_block_out(
    trace: Trace,
    block_size: int,
) -> list[dict[str, Any]]:
    rows = []
    total_days = len(trace.daily_pnl)

    for offset in range(0, total_days, block_size):
        end = min(offset + block_size, total_days)

        if end - offset < max(20, block_size // 2):
            continue

        retained = np.concatenate([
            trace.daily_pnl[:offset],
            trace.daily_pnl[end:],
        ])

        metrics = score_from_daily_pnl(retained)

        rows.append({
            "removed_block": (
                f"{trace.test_start_day + offset}-"
                f"{trace.test_start_day + end - 1}"
            ),
            "removed_total_pl": float(
                trace.daily_pnl[offset:end].sum()
            ),
            **metrics,
        })

    return rows


def concentration_tests(
    trace: Trace,
    names: list[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    instrument_rows = []
    exclusion_rows = []

    instrument_total = trace.instrument_daily_pnl.sum(
        axis=0
    )
    total_pl = float(instrument_total.sum())

    for instrument, name in enumerate(names):
        pnl = trace.instrument_daily_pnl[:, instrument]
        metrics = score_from_daily_pnl(pnl)

        instrument_rows.append({
            "instrument": instrument,
            "name": name,
            "total_pl": float(instrument_total[instrument]),
            "share_of_total_pl": (
                float(instrument_total[instrument] / total_pl)
                if abs(total_pl) > 1e-12
                else float("nan")
            ),
            "mean_pl": metrics["mean_pl"],
            "std_pl": metrics["std_pl"],
            "sharpe": metrics["sharpe"],
            "positive_day_fraction": (
                metrics["positive_day_fraction"]
            ),
            "turnover": float(
                trace.turnover[:, instrument].sum()
            ),
        })

        excluded = (
            trace.daily_pnl
            - trace.instrument_daily_pnl[:, instrument]
        )
        excluded_metrics = score_from_daily_pnl(excluded)

        exclusion_rows.append({
            "excluded_instrument": instrument,
            "excluded_name": name,
            "excluded_total_pl": float(
                instrument_total[instrument]
            ),
            **excluded_metrics,
        })

    profitable_order = np.argsort(
        instrument_total
    )[::-1]
    top_five = profitable_order[:5]

    without_top_five = (
        trace.daily_pnl
        - trace.instrument_daily_pnl[:, top_five].sum(
            axis=1
        )
    )
    without_algo = (
        trace.daily_pnl
        - trace.instrument_daily_pnl[:, 0]
    )

    summary = {
        "total_pl": total_pl,
        "top_five_indices": [
            int(index) for index in top_five
        ],
        "top_five_names": [
            names[int(index)] for index in top_five
        ],
        "top_five_total_pl": float(
            instrument_total[top_five].sum()
        ),
        "top_five_share": (
            float(
                instrument_total[top_five].sum()
                / total_pl
            )
            if abs(total_pl) > 1e-12
            else float("nan")
        ),
        "without_top_five": score_from_daily_pnl(
            without_top_five
        ),
        "without_algo": score_from_daily_pnl(
            without_algo
        ),
    }

    return instrument_rows, exclusion_rows, summary


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------

def stress_tests(
    prices: np.ndarray,
    strategy_path: str | Path,
    start_day_count: int,
    commission_multipliers: list[float],
    execution_delays: list[int],
) -> list[dict[str, Any]]:
    desired = collect_desired_positions(
        prices,
        strategy_path,
        start_day_count,
        unique_tag="stress_base",
    )

    rows = []

    for multiplier in commission_multipliers:
        for delay in execution_delays:
            trace = account_positions(
                prices,
                desired,
                start_day_count,
                commission_multiplier=multiplier,
                execution_delay=delay,
            )
            metrics = score_from_daily_pnl(
                trace.daily_pnl
            )

            rows.append({
                "commission_multiplier": multiplier,
                "execution_delay_days": delay,
                "total_turnover": float(
                    trace.turnover.sum()
                ),
                **metrics,
            })

    return rows


# ---------------------------------------------------------------------------
# Synthetic history and future generators
# ---------------------------------------------------------------------------

def geometric_block_indices(
    n_source: int,
    n_output: int,
    mean_block_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_source <= 0:
        raise ValueError("n_source must be positive.")

    continue_probability = np.clip(
        1.0 - 1.0 / max(mean_block_length, 1.0),
        0.0,
        0.999,
    )

    output = np.empty(n_output, dtype=int)
    current = int(rng.integers(0, n_source))

    for index in range(n_output):
        if index == 0:
            pass
        elif (
            rng.random() < continue_probability
            and current + 1 < n_source
        ):
            current += 1
        else:
            current = int(rng.integers(0, n_source))

        output[index] = current

    return output


def reconstruct_prices(
    initial_prices: np.ndarray,
    log_returns: np.ndarray,
) -> np.ndarray:
    initial_prices = np.asarray(
        initial_prices,
        dtype=float,
    )
    log_returns = np.asarray(
        log_returns,
        dtype=float,
    )

    log_path = (
        np.log(np.maximum(initial_prices, 1e-12))[None, :]
        + np.cumsum(log_returns, axis=0)
    )

    return np.vstack([
        initial_prices[None, :],
        np.exp(log_path),
    ]).T


def placebo_independent_shifts(
    prices: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Preserve each instrument's exact return sequence up to a circular shift,
    but destroy cross-instrument timing and lead-lag alignment.
    """
    rng = np.random.default_rng(seed)
    returns = np.diff(np.log(prices), axis=1).T
    shifted = np.empty_like(returns)

    for instrument in range(returns.shape[1]):
        shift = int(
            rng.integers(0, returns.shape[0])
        )
        shifted[:, instrument] = np.roll(
            returns[:, instrument],
            shift,
        )

    return reconstruct_prices(
        prices[:, 0],
        shifted,
    )


def placebo_common_block_permutation(
    prices: np.ndarray,
    seed: int,
    block_size: int = 20,
) -> np.ndarray:
    """
    Preserve within-block multivariate return vectors while randomising the
    order of chronological blocks.
    """
    rng = np.random.default_rng(seed)
    returns = np.diff(np.log(prices), axis=1).T

    blocks = [
        returns[start:start + block_size]
        for start in range(
            0,
            len(returns),
            block_size,
        )
    ]

    order = rng.permutation(len(blocks))
    permuted = np.concatenate(
        [blocks[index] for index in order],
        axis=0,
    )[:len(returns)]

    return reconstruct_prices(
        prices[:, 0],
        permuted,
    )


def relationship_decay_future(
    observed_prices: np.ndarray,
    future_days: int,
    retention: float,
    seed: int,
    mean_block_length: float,
) -> np.ndarray:
    """
    Generate a paired future with adjustable cross-instrument relationship
    retention.

    retention=1:
        all instruments use the same historical block index sequence.

    retention=0:
        each instrument uses an independent historical block sequence.

    Intermediate values blend the two and then restore each instrument's
    historical mean and volatility.
    """
    rng = np.random.default_rng(seed)
    returns = np.diff(
        np.log(observed_prices),
        axis=1,
    ).T

    n_source, n_instruments = returns.shape
    historical_mean = returns.mean(axis=0)
    historical_std = np.maximum(
        returns.std(axis=0),
        1e-12,
    )

    common_indices = geometric_block_indices(
        n_source,
        future_days,
        mean_block_length,
        rng,
    )
    common = returns[common_indices]

    independent = np.empty(
        (future_days, n_instruments),
        dtype=float,
    )

    for instrument in range(n_instruments):
        indices = geometric_block_indices(
            n_source,
            future_days,
            mean_block_length,
            rng,
        )
        independent[:, instrument] = (
            returns[indices, instrument]
        )

    retention = float(
        np.clip(retention, 0.0, 1.0)
    )
    independent_weight = math.sqrt(
        max(0.0, 1.0 - retention * retention)
    )

    blended = (
        retention * common
        + independent_weight * independent
    )

    # Preserve each instrument's empirical first two moments.
    blended_mean = blended.mean(axis=0)
    blended_std = np.maximum(
        blended.std(axis=0),
        1e-12,
    )

    blended = (
        (blended - blended_mean)
        * (historical_std / blended_std)
        + historical_mean
    )

    future = reconstruct_prices(
        observed_prices[:, -1],
        blended,
    )[:, 1:]

    return future


# ---------------------------------------------------------------------------
# Parallel synthetic tests
# ---------------------------------------------------------------------------

def _placebo_worker(
    task: tuple[
        str,
        int,
        str,
        str,
        int,
        int,
    ],
) -> dict[str, Any]:
    (
        kind,
        seed,
        strategy_path,
        prices_path,
        start_day_count,
        block_size,
    ) = task

    prices, _ = load_prices(prices_path)

    if kind == "independent_shift":
        synthetic = placebo_independent_shifts(
            prices,
            seed,
        )
    elif kind == "block_permutation":
        synthetic = placebo_common_block_permutation(
            prices,
            seed,
            block_size=block_size,
        )
    else:
        raise ValueError(f"Unknown placebo kind: {kind}")

    trace = run_trace(
        synthetic,
        strategy_path,
        start_day_count,
        unique_tag=f"placebo_{kind}_{seed}",
    )
    metrics = score_from_daily_pnl(trace.daily_pnl)

    return {
        "kind": kind,
        "seed": seed,
        **metrics,
    }


def _decay_worker(
    task: tuple[
        float,
        int,
        str,
        str,
        int,
        float,
    ],
) -> dict[str, Any]:
    (
        retention,
        seed,
        strategy_path,
        prices_path,
        future_days,
        mean_block_length,
    ) = task

    observed, _ = load_prices(prices_path)

    future = relationship_decay_future(
        observed,
        future_days,
        retention,
        seed,
        mean_block_length,
    )
    full = np.hstack([observed, future])

    trace = run_trace(
        full,
        strategy_path,
        start_day_count=observed.shape[1],
        unique_tag=(
            f"decay_{retention:.3f}_{seed}"
        ),
    )
    metrics = score_from_daily_pnl(trace.daily_pnl)

    return {
        "retention": retention,
        "seed": seed,
        **metrics,
    }


def run_parallel(
    worker,
    tasks: list[Any],
    workers: int,
    label: str,
) -> list[dict[str, Any]]:
    if not tasks:
        return []

    rows = []
    progress_every = max(1, len(tasks) // 10)

    if workers == 1:
        for completed, task in enumerate(tasks, start=1):
            rows.append(worker(task))

            if (
                completed % progress_every == 0
                or completed == len(tasks)
            ):
                print(
                    f"{label}: {completed}/{len(tasks)}"
                )
    else:
        with ProcessPoolExecutor(
            max_workers=workers
        ) as executor:
            future_map = {
                executor.submit(worker, task): task
                for task in tasks
            }

            completed = 0

            for future in as_completed(future_map):
                rows.append(future.result())
                completed += 1

                if (
                    completed % progress_every == 0
                    or completed == len(tasks)
                ):
                    print(
                        f"{label}: {completed}/{len(tasks)}"
                    )

    return rows


# ---------------------------------------------------------------------------
# Parameter perturbations
# ---------------------------------------------------------------------------

PARAMETER_NEIGHBOURHOODS: dict[str, list[Any]] = {
    "_primary_LLW": [1.1, 1.2, 1.4, 1.5],
    "_primary_LL_BLEND": [0.0, 0.05, 0.15, 0.20],
    "_primary_LL_THRESH": [0.0, 0.02, 0.06, 0.08],
    "_primary_LL_FULL_SV_SCALE": [1.4, 1.5, 1.7, 1.8],
    "_primary_LL_RECENT_SV_SCALE": [2.5, 2.75, 3.25, 3.5],
    "_primary_GW": [0.25, 0.40, 0.60, 0.75],
    "_primary_BASKET_ALPHA": [0.25, 0.30, 0.50, 0.55],
    "_primary_BASKET_GATE": [1.70, 1.80, 1.95, 2.05],
    "_primary_REV_GATE": [1.50, 1.625, 1.875, 2.00],
    "_primary_CF_GATE": [1.00, 1.10, 1.20, 1.30],
    "_CB_TRAIL_WINDOW": [40, 50, 80, 100],
    "_CB_MIN_SCALE": [0.20, 0.60, 0.80],
    "_CB_SHARPE_FLOOR": [-2.0, -4.0, -5.0],
}


def find_assignment_value(
    source: str,
    constant_name: str,
) -> str | None:
    pattern = re.compile(
        rf"(?m)^\s*{re.escape(constant_name)}\s*=\s*"
        rf"(?P<value>[^\n#]+?)\s*(?:#.*)?$"
    )
    match = pattern.search(source)

    return (
        match.group("value").strip()
        if match
        else None
    )


def patch_assignment(
    source: str,
    constant_name: str,
    new_value: Any,
) -> str:
    pattern = re.compile(
        rf"(?m)^(?P<prefix>\s*"
        rf"{re.escape(constant_name)}\s*=\s*)"
        rf"(?P<value>[^\n#]+?)"
        rf"(?P<suffix>\s*(?:#.*)?)$"
    )

    replacement_value = repr(new_value)

    patched, count = pattern.subn(
        lambda match: (
            match.group("prefix")
            + replacement_value
            + match.group("suffix")
        ),
        source,
        count=1,
    )

    if count != 1:
        raise ValueError(
            f"Could not uniquely patch {constant_name}."
        )

    return patched


def parameter_variant_worker(
    task: tuple[
        str,
        str,
        Any,
        str,
        int,
        str,
    ],
) -> dict[str, Any]:
    (
        strategy_source_path,
        constant_name,
        value,
        prices_path,
        start_day_count,
        temporary_directory,
    ) = task

    source_path = Path(strategy_source_path)
    source = source_path.read_text(
        encoding="utf-8",
    )
    patched = patch_assignment(
        source,
        constant_name,
        value,
    )

    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        f"{constant_name}_{value}",
    )
    variant_path = (
        Path(temporary_directory)
        / f"variant_{safe_name}.py"
    )
    variant_path.write_text(
        patched,
        encoding="utf-8",
    )

    prices, _ = load_prices(prices_path)

    trace = run_trace(
        prices,
        variant_path,
        start_day_count,
        unique_tag=safe_name,
    )
    metrics = score_from_daily_pnl(
        trace.daily_pnl
    )

    return {
        "parameter": constant_name,
        "value": value,
        **metrics,
    }


def parameter_stability_tests(
    strategy_path: str | Path,
    prices_path: str | Path,
    start_day_count: int,
    workers: int,
    max_parameters: int | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    strategy_path = Path(strategy_path).resolve()
    source = strategy_path.read_text(
        encoding="utf-8",
    )

    available = []

    for constant_name, values in (
        PARAMETER_NEIGHBOURHOODS.items()
    ):
        original = find_assignment_value(
            source,
            constant_name,
        )

        if original is not None:
            available.append(
                (constant_name, original, values)
            )

    if max_parameters is not None:
        available = available[:max_parameters]

    if not available:
        return [], []

    temporary_directory = tempfile.mkdtemp(
        prefix="sigma_parameter_validation_"
    )

    try:
        tasks = []

        for constant_name, original, values in available:
            for value in values:
                if repr(value) == original:
                    continue

                tasks.append((
                    str(strategy_path),
                    constant_name,
                    value,
                    str(Path(prices_path).resolve()),
                    start_day_count,
                    temporary_directory,
                ))

        rows = run_parallel(
            parameter_variant_worker,
            tasks,
            workers,
            "Parameter variants",
        )

        summary_rows = []

        for constant_name, original, _ in available:
            parameter_rows = [
                row
                for row in rows
                if row["parameter"] == constant_name
            ]

            if not parameter_rows:
                continue

            scores = np.asarray([
                row["score"]
                for row in parameter_rows
            ])

            summary_rows.append({
                "parameter": constant_name,
                "original_source_value": original,
                "variant_count": len(scores),
                "variant_score_mean": float(
                    scores.mean()
                ),
                "variant_score_min": float(
                    scores.min()
                ),
                "variant_score_max": float(
                    scores.max()
                ),
                "variant_score_std": float(
                    scores.std()
                ),
            })

        return rows, summary_rows
    finally:
        shutil.rmtree(
            temporary_directory,
            ignore_errors=True,
        )


# ---------------------------------------------------------------------------
# Summary diagnostics
# ---------------------------------------------------------------------------

def group_distribution(
    rows: list[dict[str, Any]],
    key: str,
    value: Any,
    metric: str = "score",
) -> dict[str, float] | None:
    values = np.asarray([
        float(row[metric])
        for row in rows
        if row[key] == value
    ])

    if len(values) == 0:
        return None

    quantiles = np.quantile(
        values,
        [0.05, 0.25, 0.50, 0.75, 0.95],
    )

    return {
        "count": float(len(values)),
        "mean": float(values.mean()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "positive_fraction": float(
            np.mean(values > 0.0)
        ),
    }


def build_summary(
    strategy_path: Path,
    prices_path: Path,
    baseline_metrics: dict[str, float],
    chronology_rows: list[dict[str, Any]],
    block_rows: list[dict[str, Any]],
    leave_block_rows: list[dict[str, Any]],
    concentration_summary: dict[str, Any] | None,
    stress_rows: list[dict[str, Any]],
    placebo_rows: list[dict[str, Any]],
    decay_rows: list[dict[str, Any]],
    parameter_summary_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "SIG ALGOTHON ROBUSTNESS / OVERFITTING AUDIT",
        "=" * 48,
        f"strategy: {strategy_path}",
        f"prices: {prices_path}",
        "",
        "BASELINE",
        format_metric_row(
            "Historical evaluation",
            baseline_metrics,
        ),
        "",
    ]

    if chronology_rows:
        lines.append("RESTARTED CHRONOLOGICAL WINDOWS")

        for row in chronology_rows:
            lines.append(
                format_metric_row(
                    row["window"],
                    row,
                )
            )

        lines.append("")

    if block_rows:
        fifty = [
            row for row in block_rows
            if int(row["block_size"]) == 50
        ]

        if fifty:
            scores = np.asarray([
                row["score"] for row in fifty
            ])
            lines.extend([
                "CONTINUOUS 50-DAY BLOCKS",
                (
                    f"count={len(scores)}  "
                    f"mean={scores.mean():.2f}  "
                    f"median={np.median(scores):.2f}  "
                    f"worst={scores.min():.2f}  "
                    f"positive={np.mean(scores > 0):.3f}"
                ),
                "",
            ])

    if leave_block_rows:
        scores = np.asarray([
            row["score"]
            for row in leave_block_rows
        ])
        lines.extend([
            "LEAVE-ONE-50-DAY-BLOCK-OUT",
            (
                f"score range after deleting one block: "
                f"{scores.min():.2f} to {scores.max():.2f}"
            ),
            "",
        ])

    if concentration_summary:
        lines.extend([
            "CONCENTRATION",
            (
                "Top five contributors: "
                + ", ".join(
                    concentration_summary[
                        "top_five_names"
                    ]
                )
            ),
            (
                f"Top-five share of total P&L: "
                f"{concentration_summary['top_five_share']:.3f}"
            ),
            format_metric_row(
                "After removing top five",
                concentration_summary[
                    "without_top_five"
                ],
            ),
            format_metric_row(
                "After removing ALGO",
                concentration_summary[
                    "without_algo"
                ],
            ),
            "",
        ])

    if stress_rows:
        lines.append("MECHANICAL STRESS")

        for row in stress_rows:
            label = (
                f"cost {row['commission_multiplier']}x, "
                f"delay {row['execution_delay_days']}d"
            )
            lines.append(
                format_metric_row(label, row)
            )

        lines.append("")

    if placebo_rows:
        lines.append("PLACEBO TESTS")

        for kind in (
            "independent_shift",
            "block_permutation",
        ):
            distribution = group_distribution(
                placebo_rows,
                "kind",
                kind,
            )

            if distribution:
                lines.append(
                    f"{kind:24s} "
                    f"mean={distribution['mean']:9.2f} "
                    f"p05={distribution['p05']:9.2f} "
                    f"median={distribution['median']:9.2f} "
                    f"p95={distribution['p95']:9.2f} "
                    f"P(score>0)="
                    f"{distribution['positive_fraction']:.3f}"
                )

        lines.append(
            "A high placebo score is a warning: the strategy may profit from "
            "generic drift/volatility or an evaluator artefact rather than "
            "the intended cross-instrument structure."
        )
        lines.append("")

    if decay_rows:
        lines.append("RELATIONSHIP-DECAY FUTURES")

        retentions = sorted({
            float(row["retention"])
            for row in decay_rows
        }, reverse=True)

        for retention in retentions:
            distribution = group_distribution(
                decay_rows,
                "retention",
                retention,
            )

            if distribution:
                lines.append(
                    f"retention={retention:4.2f}  "
                    f"mean={distribution['mean']:9.2f} "
                    f"p05={distribution['p05']:9.2f} "
                    f"median={distribution['median']:9.2f} "
                    f"p95={distribution['p95']:9.2f} "
                    f"P(score>0)="
                    f"{distribution['positive_fraction']:.3f}"
                )

        lines.append(
            "A gradual rather than cliff-like decline is more reassuring. "
            "Strong performance only at retention=1 indicates dependence on "
            "historical relationships remaining almost unchanged."
        )
        lines.append("")

    if parameter_summary_rows:
        lines.append("PARAMETER NEIGHBOURHOODS")

        for row in parameter_summary_rows:
            lines.append(
                f"{row['parameter']:30s} "
                f"mean={row['variant_score_mean']:9.2f} "
                f"min={row['variant_score_min']:9.2f} "
                f"max={row['variant_score_max']:9.2f} "
                f"sd={row['variant_score_std']:8.2f}"
            )

        lines.append(
            "Neighbouring variants should form a broad plateau. A large "
            "standard deviation or severe minimum is evidence of parameter "
            "fragility."
        )
        lines.append("")

    lines.extend([
        "HOW TO INTERPRET THE AUDIT",
        (
            "No individual failure proves that the strategy is overfit, and "
            "no collection of passes proves the historical score will repeat. "
            "The strongest warning combination is: high concentration, weak "
            "leave-block-out results, isolated parameter peaks, positive "
            "placebo scores, and rapid collapse as relationship retention "
            "falls."
        ),
        (
            "This audit does not remove the selection bias created by using "
            "the same 1,000 days to develop many strategy generations."
        ),
    ])

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a SIG Algothon strategy for historical fragility "
            "and research overfitting."
        )
    )

    parser.add_argument(
        "--strategy",
        required=True,
        help="Strategy Python file exposing getMyPosition().",
    )
    parser.add_argument(
        "--prices",
        default="prices_full_final.txt",
        help="Price file containing the available history.",
    )
    parser.add_argument(
        "--output",
        default="validation_output",
        help="Output directory.",
    )
    parser.add_argument(
        "--profile",
        choices=["quick", "core", "full"],
        default="core",
        help=(
            "quick: small synthetic samples; "
            "core: historical/concentration/stress only; "
            "full: all tests."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for synthetic and parameter tests.",
    )
    parser.add_argument(
        "--historical-test-days",
        type=int,
        default=500,
        help="Baseline evaluates the final N observed days.",
    )
    parser.add_argument(
        "--placebo-paths",
        type=int,
        default=None,
        help="Override placebo paths per placebo type.",
    )
    parser.add_argument(
        "--decay-paths",
        type=int,
        default=None,
        help="Override paths per relationship-retention level.",
    )
    parser.add_argument(
        "--future-days",
        type=int,
        default=500,
        help="Future days in relationship-decay simulations.",
    )
    parser.add_argument(
        "--retentions",
        default="1.0,0.75,0.5,0.25,0.0",
        help="Comma-separated relationship-retention levels.",
    )
    parser.add_argument(
        "--block-length",
        type=float,
        default=10.0,
        help="Mean bootstrap block length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9418,
        help="Master random seed.",
    )
    parser.add_argument(
        "--max-parameters",
        type=int,
        default=None,
        help=(
            "Limit how many recognised constants receive neighbourhood "
            "tests. Useful for a faster run."
        ),
    )

    args = parser.parse_args()

    if args.workers <= 0:
        parser.error("--workers must be positive.")

    if args.historical_test_days <= 0:
        parser.error("--historical-test-days must be positive.")

    return args


def main() -> None:
    args = parse_args()

    strategy_path = Path(args.strategy).resolve()
    prices_path = Path(args.prices).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    prices, names = load_prices(prices_path)
    n_instruments, n_days = prices.shape

    if args.historical_test_days >= n_days:
        raise ValueError(
            "--historical-test-days must be less than total days."
        )

    baseline_start = (
        n_days - args.historical_test_days
    )

    print(
        f"Loaded {n_instruments} instruments x {n_days} days"
    )
    print(
        f"Baseline: days {baseline_start + 1}-{n_days}"
    )

    start_time = time.time()

    baseline_trace = run_trace(
        prices,
        strategy_path,
        baseline_start,
        unique_tag="baseline",
    )
    baseline_metrics = score_from_daily_pnl(
        baseline_trace.daily_pnl
    )
    print(
        format_metric_row(
            "Baseline",
            baseline_metrics,
        )
    )

    # Historical chronology.
    default_windows = [
        (251, 500),
        (501, 750),
        (751, 1000),
        (501, 1000),
    ]
    windows = [
        (start, end)
        for start, end in default_windows
        if end <= n_days
    ]

    chronology_rows = chronological_windows(
        prices,
        strategy_path,
        windows,
    )
    write_rows(
        output / "chronological_windows.csv",
        chronology_rows,
    )

    block_rows = continuous_blocks(
        baseline_trace,
        block_sizes=(50, 100, 250),
    )
    write_rows(
        output / "continuous_blocks.csv",
        block_rows,
    )

    leave_block_rows = leave_one_block_out(
        baseline_trace,
        block_size=50,
    )
    write_rows(
        output / "leave_one_block_out.csv",
        leave_block_rows,
    )

    instrument_rows, exclusion_rows, concentration_summary = (
        concentration_tests(
            baseline_trace,
            names,
        )
    )
    write_rows(
        output / "instrument_contributions.csv",
        instrument_rows,
    )
    write_rows(
        output / "leave_one_instrument_out.csv",
        exclusion_rows,
    )

    stress_rows = stress_tests(
        prices,
        strategy_path,
        baseline_start,
        commission_multipliers=[
            1.0,
            2.0,
            3.0,
            5.0,
        ],
        execution_delays=[0, 1, 2],
    )
    write_rows(
        output / "cost_delay_stress.csv",
        stress_rows,
    )

    placebo_rows: list[dict[str, Any]] = []
    decay_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    parameter_summary_rows: list[dict[str, Any]] = []

    run_synthetic = args.profile in (
        "quick",
        "full",
    )
    run_parameters = args.profile == "full"

    if run_synthetic:
        if args.placebo_paths is None:
            placebo_paths = (
                3 if args.profile == "quick" else 20
            )
        else:
            placebo_paths = args.placebo_paths

        if args.decay_paths is None:
            decay_paths = (
                3 if args.profile == "quick" else 20
            )
        else:
            decay_paths = args.decay_paths

        master_rng = np.random.default_rng(
            args.seed
        )

        placebo_tasks = []

        for kind in (
            "independent_shift",
            "block_permutation",
        ):
            seeds = master_rng.integers(
                0,
                np.iinfo(np.uint32).max,
                size=placebo_paths,
                dtype=np.uint32,
            )

            for seed in seeds:
                placebo_tasks.append((
                    kind,
                    int(seed),
                    str(strategy_path),
                    str(prices_path),
                    baseline_start,
                    20,
                ))

        placebo_rows = run_parallel(
            _placebo_worker,
            placebo_tasks,
            args.workers,
            "Placebos",
        )
        write_rows(
            output / "placebo_results.csv",
            placebo_rows,
        )

        retentions = parse_float_list(
            args.retentions
        )
        decay_tasks = []

        # Paired seeds across retention levels.
        decay_seeds = master_rng.integers(
            0,
            np.iinfo(np.uint32).max,
            size=decay_paths,
            dtype=np.uint32,
        )

        for retention in retentions:
            for seed in decay_seeds:
                decay_tasks.append((
                    float(retention),
                    int(seed),
                    str(strategy_path),
                    str(prices_path),
                    args.future_days,
                    args.block_length,
                ))

        decay_rows = run_parallel(
            _decay_worker,
            decay_tasks,
            args.workers,
            "Relationship decay",
        )
        write_rows(
            output / "relationship_decay_results.csv",
            decay_rows,
        )

    if run_parameters:
        parameter_rows, parameter_summary_rows = (
            parameter_stability_tests(
                strategy_path,
                prices_path,
                baseline_start,
                args.workers,
                args.max_parameters,
            )
        )
        write_rows(
            output / "parameter_variants.csv",
            parameter_rows,
        )
        write_rows(
            output / "parameter_summary.csv",
            parameter_summary_rows,
        )

    summary = build_summary(
        strategy_path,
        prices_path,
        baseline_metrics,
        chronology_rows,
        block_rows,
        leave_block_rows,
        concentration_summary,
        stress_rows,
        placebo_rows,
        decay_rows,
        parameter_summary_rows,
    )

    summary_path = output / "validation_summary.txt"
    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary, end="")
    print()
    print(
        f"Completed in {time.time() - start_time:.1f}s"
    )
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()