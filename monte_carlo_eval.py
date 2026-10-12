#!/usr/bin/env python3
"""
monte_carlo_eval.py

Generate plausible price paths for days 1001-1500 from the observed
1-1000 price history, evaluate a submitted strategy on every path using
the official Algothon accounting rules, and aggregate the Monte Carlo
distribution.

The default generator is a data-driven regime/block model:

1. Fit a six-state model to historical multivariate log returns:
      bearish / neutral / bullish
      crossed with low / high volatility.
2. Estimate empirical Markov transitions between those states.
3. Resample whole 51-instrument return vectors in geometric blocks,
   preserving cross-sectional dependence, tails and short serial structure.
4. Draw a long-run drift vector from historical 250-day windows.
5. Balance simulations across bearish, neutral and bullish drift scenarios.
6. Draw historical volatility multipliers to create calm and volatile paths.

This is a scenario model, not a claim that the future is forecastable.
Its value is comparing strategies across many continuations consistent with
the observed data-generating behaviour.

Examples
--------
    python3 monte_carlo_eval.py \
        --strategy Sigma0.py \
        --prices prices_full_final.txt \
        --paths 300 \
        --workers 4

    python3 monte_carlo_eval.py \
        --strategy Sigma0.py \
        --paths 1000 \
        --directions balanced \
        --save-paths 10

Outputs
-------
    monte_carlo_results.csv
    monte_carlo_daily_summary.csv
    monte_carlo_summary.txt
    simulated_paths/path_*.txt       when --save-paths > 0

NumPy only.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Official evaluator constants
# ---------------------------------------------------------------------------

SCORE_PARAM = 1.0
DEFAULT_COMM = 0.0001
INST0_COMM = 0.00002
DEFAULT_LIMIT = 10_000.0
INST0_LIMIT = 100_000.0


# ---------------------------------------------------------------------------
# Price loading and scoring
# ---------------------------------------------------------------------------

def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def load_prices(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """
    Load a whitespace-delimited price file.

    Returns
    -------
    prices:
        Shape (instruments, days).
    names:
        Instrument names, or generated names when the file has no header.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip().split()

    has_header = bool(first_line) and not all(
        _is_number(token) for token in first_line
    )

    if has_header:
        values = np.loadtxt(path, skiprows=1)
        names = first_line
    else:
        values = np.loadtxt(path)
        names = [f"I{i}" for i in range(values.shape[1])]

    if values.ndim != 2:
        raise ValueError("Price file must contain a two-dimensional table.")

    # Source format is normally days x instruments.
    if values.shape[0] >= values.shape[1]:
        prices = values.T
    else:
        prices = values

    if len(names) != prices.shape[0]:
        names = [f"I{i}" for i in range(prices.shape[0])]

    if np.any(~np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("All prices must be finite and strictly positive.")

    return prices.astype(float), names


def official_score(
    mean_pl: float,
    std_pl: float,
    param: float = SCORE_PARAM,
) -> float:
    if mean_pl <= 0.0 or std_pl < 1e-10:
        return float(mean_pl)

    sharpe = math.sqrt(250.0) * mean_pl / std_pl
    fraction = sharpe * sharpe / (
        sharpe * sharpe + param * param
    )
    return float(mean_pl * fraction)


def summarise_pnl(
    daily_pnl: np.ndarray,
    total_dollar_volume: float,
    final_value: float,
) -> dict[str, float]:
    daily_pnl = np.asarray(daily_pnl, dtype=float)

    mean_pl = float(daily_pnl.mean())
    std_pl = float(daily_pnl.std())

    annualised_sharpe = (
        math.sqrt(250.0) * mean_pl / std_pl
        if std_pl > 1e-12
        else 0.0
    )

    portfolio_return = (
        final_value / total_dollar_volume
        if total_dollar_volume > 0.0
        else 0.0
    )

    return {
        "mean_pl": mean_pl,
        "std_pl": std_pl,
        "sharpe": annualised_sharpe,
        "score": official_score(mean_pl, std_pl),
        "total_dollar_volume": float(total_dollar_volume),
        "final_value": float(final_value),
        "return": float(portfolio_return),
    }


# ---------------------------------------------------------------------------
# Strategy loading and official-style evaluation
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

    strategy_dir = str(strategy_path.parent)

    if strategy_dir not in sys.path:
        sys.path.insert(0, strategy_dir)

    module_name = (
        f"_mc_strategy_{strategy_path.stem}_"
        f"{os.getpid()}_{unique_tag}"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        strategy_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load strategy file: {strategy_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    get_position = getattr(module, "getMyPosition", None)

    if not callable(get_position):
        raise AttributeError(
            f"{strategy_path.name} does not expose getMyPosition()."
        )

    # Optional compatibility with several strategy state conventions.
    for reset_name in ("reset_state", "_reset", "reset"):
        reset = getattr(module, reset_name, None)

        if callable(reset):
            reset()
            break

    return get_position


def evaluate_like_official(
    prices: np.ndarray,
    strategy_path: str | Path,
    start_day_count: int,
    unique_tag: str,
) -> tuple[dict[str, float], np.ndarray]:
    """
    Reproduce the official eval.py accounting loop.

    start_day_count is the number of observed days available when the first
    position for the test is requested. For a 1000-day history followed by
    500 simulated days, use start_day_count=1000.
    """
    prices = np.asarray(prices, dtype=float)
    n_instruments, n_days = prices.shape

    if not (1 <= start_day_count < n_days):
        raise ValueError(
            "start_day_count must be between 1 and n_days - 1."
        )

    get_position = load_strategy(
        strategy_path,
        unique_tag,
    )

    commission_rate = np.full(
        n_instruments,
        DEFAULT_COMM,
        dtype=float,
    )
    commission_rate[0] = INST0_COMM

    dollar_limit = np.full(
        n_instruments,
        DEFAULT_LIMIT,
        dtype=float,
    )
    dollar_limit[0] = INST0_LIMIT

    cash = 0.0
    current_position = np.zeros(
        n_instruments,
        dtype=float,
    )
    total_dollar_volume = 0.0
    portfolio_value = 0.0
    commission = 0.0
    daily_pnl: list[float] = []

    # This matches:
    #     for t in range(startDay, nt + 1)
    #     prcHistSoFar = prcHist[:, :t]
    for t in range(start_day_count, n_days + 1):
        history = prices[:, :t]
        current_prices = history[:, -1]

        if t < n_days:
            proposed = np.asarray(
                get_position(history),
                dtype=float,
            )

            if proposed.shape != (n_instruments,):
                raise ValueError(
                    "getMyPosition returned shape "
                    f"{proposed.shape}; expected {(n_instruments,)}."
                )

            proposed = np.nan_to_num(
                proposed,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            position_limits = (
                dollar_limit
                / np.maximum(current_prices, 1e-12)
            ).astype(int)

            new_position = np.clip(
                proposed,
                -position_limits,
                position_limits,
            ).astype(int)
        else:
            # Final day is a mark-to-market day only.
            new_position = current_position.copy()

        position_change = (
            new_position - current_position
        )

        # Preserve official eval.py fee timing exactly.
        cash -= (
            float(current_prices @ position_change)
            + commission
        )

        dollar_volumes = (
            current_prices
            * np.abs(position_change)
        )
        total_dollar_volume += float(
            dollar_volumes.sum()
        )
        commission = float(
            dollar_volumes @ commission_rate
        )

        current_position = np.asarray(
            new_position,
            dtype=float,
        )

        position_value = float(
            current_position @ current_prices
        )
        today_pnl = (
            cash
            + position_value
            - portfolio_value
        )
        portfolio_value = cash + position_value

        if t > start_day_count:
            daily_pnl.append(float(today_pnl))

    pnl_array = np.asarray(daily_pnl, dtype=float)

    if len(pnl_array) != n_days - start_day_count:
        raise RuntimeError(
            "Unexpected number of evaluated P&L days."
        )

    metrics = summarise_pnl(
        pnl_array,
        total_dollar_volume,
        portfolio_value,
    )

    return metrics, pnl_array


# ---------------------------------------------------------------------------
# Scenario generator fitting
# ---------------------------------------------------------------------------

def _rolling_sum(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    output = np.full(len(values), np.nan)
    cumsum = np.concatenate([[0.0], np.cumsum(values)])

    for end in range(window - 1, len(values)):
        output[end] = (
            cumsum[end + 1]
            - cumsum[end + 1 - window]
        )

    return output


def _rolling_std(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    output = np.full(len(values), np.nan)

    for end in range(window - 1, len(values)):
        output[end] = float(
            values[end + 1 - window:end + 1].std()
        )

    return output


def fit_generator(
    observed_prices: np.ndarray,
    trend_window: int,
    volatility_window: int,
    drift_window: int,
    block_length: float,
    transition_smoothing: float,
) -> dict[str, Any]:
    """
    Fit the regime/block continuation model from the observed history.
    """
    log_prices = np.log(
        np.maximum(observed_prices, 1e-12)
    )
    returns = np.diff(log_prices, axis=1).T
    n_return_days, n_instruments = returns.shape

    if n_return_days < max(
        trend_window,
        volatility_window,
        drift_window,
    ) + 20:
        raise ValueError(
            "Not enough history for the requested generator windows."
        )

    global_mean = returns.mean(axis=0)
    centred = returns - global_mean

    covariance = (
        centred.T @ centred
        / max(n_return_days - 1, 1)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )
    pc1 = eigenvectors[:, np.argmax(eigenvalues)]

    # Stable orientation: positive ALGO loading.
    if pc1[0] < 0.0:
        pc1 = -pc1

    factor = centred @ pc1

    trend = _rolling_sum(
        factor,
        trend_window,
    )
    factor_volatility = _rolling_std(
        factor,
        volatility_window,
    )

    valid = (
        np.isfinite(trend)
        & np.isfinite(factor_volatility)
    )

    trend_low, trend_high = np.quantile(
        trend[valid],
        [1.0 / 3.0, 2.0 / 3.0],
    )
    volatility_cut = float(
        np.median(factor_volatility[valid])
    )

    direction_state = np.ones(
        n_return_days,
        dtype=int,
    )
    direction_state[trend <= trend_low] = 0
    direction_state[trend >= trend_high] = 2

    volatility_state = (
        factor_volatility > volatility_cut
    ).astype(int)

    states = (
        direction_state
        + 3 * volatility_state
    )

    # Fill early undefined state observations with the first valid state.
    first_valid = int(np.flatnonzero(valid)[0])
    states[:first_valid] = states[first_valid]

    n_states = 6
    transition_counts = np.full(
        (n_states, n_states),
        float(transition_smoothing),
    )

    for current_state, next_state in zip(
        states[:-1],
        states[1:],
    ):
        transition_counts[
            int(current_state),
            int(next_state),
        ] += 1.0

    transition_matrix = (
        transition_counts
        / transition_counts.sum(
            axis=1,
            keepdims=True,
        )
    )

    state_pools: list[np.ndarray] = []

    for state in range(n_states):
        pool = np.flatnonzero(states == state)

        if len(pool) < 5:
            pool = np.arange(n_return_days)

        state_pools.append(pool.astype(int))

    # Rolling long-run drift vectors. Their projection onto PC1 defines
    # bearish, neutral and bullish scenario buckets.
    cumsum_returns = np.vstack([
        np.zeros((1, n_instruments)),
        np.cumsum(returns, axis=0),
    ])

    drift_vectors = []
    drift_projection = []

    for end in range(drift_window, n_return_days + 1):
        drift = (
            cumsum_returns[end]
            - cumsum_returns[end - drift_window]
        ) / drift_window

        drift_vectors.append(drift)
        drift_projection.append(
            float((drift - global_mean) @ pc1)
        )

    drift_vectors_array = np.asarray(
        drift_vectors,
        dtype=float,
    )
    drift_projection_array = np.asarray(
        drift_projection,
        dtype=float,
    )

    drift_low, drift_high = np.quantile(
        drift_projection_array,
        [1.0 / 3.0, 2.0 / 3.0],
    )

    drift_labels = np.ones(
        len(drift_projection_array),
        dtype=int,
    )
    drift_labels[
        drift_projection_array <= drift_low
    ] = 0
    drift_labels[
        drift_projection_array >= drift_high
    ] = 2

    drift_pools = [
        np.flatnonzero(drift_labels == label)
        for label in range(3)
    ]

    # Historical volatility multipliers based on rolling median
    # cross-sectional volatility.
    global_instrument_volatility = returns.std(
        axis=0,
    )
    global_median_volatility = max(
        float(np.median(global_instrument_volatility)),
        1e-12,
    )

    volatility_multipliers = []

    for end in range(
        volatility_window,
        n_return_days + 1,
    ):
        local_volatility = returns[
            end - volatility_window:end
        ].std(axis=0)

        volatility_multipliers.append(
            float(
                np.median(local_volatility)
                / global_median_volatility
            )
        )

    volatility_multipliers = np.clip(
        np.asarray(
            volatility_multipliers,
            dtype=float,
        ),
        0.70,
        1.40,
    )

    lower_clip = np.quantile(
        returns,
        0.001,
        axis=0,
    )
    upper_clip = np.quantile(
        returns,
        0.999,
        axis=0,
    )

    return {
        "returns": returns,
        "global_mean": global_mean,
        "pc1": pc1,
        "states": states,
        "transition_matrix": transition_matrix,
        "state_pools": state_pools,
        "initial_state": int(states[-1]),
        "drift_vectors": drift_vectors_array,
        "drift_pools": drift_pools,
        "volatility_multipliers": volatility_multipliers,
        "lower_clip": lower_clip,
        "upper_clip": upper_clip,
        "block_continue_probability": max(
            0.0,
            min(
                0.999,
                1.0 - 1.0 / max(
                    float(block_length),
                    1.0,
                ),
            ),
        ),
    }


# ---------------------------------------------------------------------------
# Future path generation
# ---------------------------------------------------------------------------

DIRECTION_NAMES = {
    0: "bear",
    1: "neutral",
    2: "bull",
}


def select_direction(
    path_index: int,
    mode: str,
    rng: np.random.Generator,
) -> int:
    if mode == "balanced":
        return path_index % 3

    if mode == "empirical":
        return int(rng.integers(0, 3))

    if mode == "bear":
        return 0

    if mode == "neutral":
        return 1

    if mode == "bull":
        return 2

    raise ValueError(f"Unknown direction mode: {mode}")


def simulate_future(
    observed_prices: np.ndarray,
    model: dict[str, Any],
    future_days: int,
    seed: int,
    direction: int,
    drift_strength: float,
    volatility_refresh_days: int,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)

    returns = model["returns"]
    global_mean = model["global_mean"]
    states = model["states"]
    transition_matrix = model["transition_matrix"]
    state_pools = model["state_pools"]

    drift_pool = model["drift_pools"][direction]

    if len(drift_pool) == 0:
        drift_pool = np.arange(
            len(model["drift_vectors"])
        )

    drift_index = int(rng.choice(drift_pool))
    scenario_drift = model["drift_vectors"][
        drift_index
    ]

    # Only apply part of the historical drift difference. The resampled
    # returns already contain their own conditional directional behaviour.
    drift_adjustment = (
        float(drift_strength)
        * (scenario_drift - global_mean)
    )

    volatility_scale = float(
        rng.choice(model["volatility_multipliers"])
    )

    future_log_returns = np.zeros(
        (future_days, observed_prices.shape[0]),
        dtype=float,
    )

    current_state = int(model["initial_state"])
    historical_index = int(
        rng.choice(state_pools[current_state])
    )

    continue_probability = float(
        model["block_continue_probability"]
    )

    for step in range(future_days):
        if (
            step > 0
            and volatility_refresh_days > 0
            and step % volatility_refresh_days == 0
        ):
            # Slow regime changes in overall volatility.
            new_scale = float(
                rng.choice(
                    model["volatility_multipliers"]
                )
            )
            volatility_scale = (
                0.5 * volatility_scale
                + 0.5 * new_scale
            )

        next_state = int(
            rng.choice(
                6,
                p=transition_matrix[current_state],
            )
        )

        continue_block = (
            rng.random() < continue_probability
        )
        next_historical_index = (
            historical_index + 1
        )

        if (
            continue_block
            and next_historical_index < len(returns)
            and int(states[next_historical_index])
            == next_state
        ):
            historical_index = next_historical_index
        else:
            historical_index = int(
                rng.choice(state_pools[next_state])
            )

        sampled_return = returns[historical_index]

        simulated_return = (
            global_mean
            + volatility_scale
            * (sampled_return - global_mean)
            + drift_adjustment
        )

        # Limit only extrapolation created by scaling. Historical sampled
        # observations remain inside these empirical tail bounds.
        simulated_return = np.clip(
            simulated_return,
            model["lower_clip"] * 1.15,
            model["upper_clip"] * 1.15,
        )

        future_log_returns[step] = simulated_return
        current_state = next_state

    last_log_price = np.log(
        np.maximum(
            observed_prices[:, -1],
            1e-12,
        )
    )

    future_log_prices = (
        last_log_price[None, :]
        + np.cumsum(
            future_log_returns,
            axis=0,
        )
    )

    future_prices = np.exp(future_log_prices).T

    diagnostics = {
        "direction_code": float(direction),
        "volatility_scale": float(volatility_scale),
        "pc1_drift": float(
            drift_adjustment @ model["pc1"]
        ),
        "mean_abs_log_return": float(
            np.mean(np.abs(future_log_returns))
        ),
    }

    return future_prices, diagnostics


def save_price_path(
    path: str | Path,
    names: list[str],
    full_prices: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(" ".join(names) + "\n")

        np.savetxt(
            handle,
            full_prices.T,
            fmt="%.6f",
            delimiter=" ",
        )


# ---------------------------------------------------------------------------
# Multiprocessing
# ---------------------------------------------------------------------------

_WORKER_CONTEXT: dict[str, Any] = {}


def _initialise_worker(
    observed_prices: np.ndarray,
    names: list[str],
    generator_model: dict[str, Any],
    strategy_path: str,
    future_days: int,
    direction_mode: str,
    drift_strength: float,
    volatility_refresh_days: int,
    save_paths: int,
    save_directory: str,
) -> None:
    global _WORKER_CONTEXT

    _WORKER_CONTEXT = {
        "observed_prices": observed_prices,
        "names": names,
        "generator_model": generator_model,
        "strategy_path": strategy_path,
        "future_days": future_days,
        "direction_mode": direction_mode,
        "drift_strength": drift_strength,
        "volatility_refresh_days": volatility_refresh_days,
        "save_paths": save_paths,
        "save_directory": save_directory,
    }


def _run_one_path(
    task: tuple[int, int],
) -> dict[str, Any]:
    path_index, seed = task
    context = _WORKER_CONTEXT

    rng = np.random.default_rng(seed)
    direction = select_direction(
        path_index,
        context["direction_mode"],
        rng,
    )

    future_prices, scenario = simulate_future(
        context["observed_prices"],
        context["generator_model"],
        context["future_days"],
        seed,
        direction,
        context["drift_strength"],
        context["volatility_refresh_days"],
    )

    full_prices = np.hstack([
        context["observed_prices"],
        future_prices,
    ])

    if path_index < context["save_paths"]:
        filename = (
            f"path_{path_index + 1:04d}_"
            f"{DIRECTION_NAMES[direction]}.txt"
        )

        save_price_path(
            Path(context["save_directory"])
            / filename,
            context["names"],
            full_prices,
        )

    metrics, daily_pnl = evaluate_like_official(
        full_prices,
        context["strategy_path"],
        start_day_count=(
            context["observed_prices"].shape[1]
        ),
        unique_tag=f"{path_index}_{seed}",
    )

    result: dict[str, Any] = {
        "path": path_index + 1,
        "seed": seed,
        "direction": DIRECTION_NAMES[direction],
        **scenario,
        **metrics,
        "daily_pnl": daily_pnl,
    }

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def percentile_summary(
    values: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)

    quantiles = np.quantile(
        values,
        [0.05, 0.25, 0.50, 0.75, 0.95],
    )

    return {
        "mean": float(values.mean()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def format_distribution(
    label: str,
    values: np.ndarray,
) -> str:
    summary = percentile_summary(values)

    return (
        f"{label:24s} "
        f"mean={summary['mean']:9.2f} "
        f"p05={summary['p05']:9.2f} "
        f"p25={summary['p25']:9.2f} "
        f"p50={summary['median']:9.2f} "
        f"p75={summary['p75']:9.2f} "
        f"p95={summary['p95']:9.2f}"
    )


def write_results_csv(
    path: str | Path,
    results: list[dict[str, Any]],
) -> None:
    fields = [
        "path",
        "seed",
        "direction",
        "direction_code",
        "volatility_scale",
        "pc1_drift",
        "mean_abs_log_return",
        "mean_pl",
        "std_pl",
        "sharpe",
        "score",
        "total_dollar_volume",
        "final_value",
        "return",
    ]

    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()

        for result in results:
            writer.writerow({
                field: result[field]
                for field in fields
            })


def write_daily_summary(
    path: str | Path,
    results: list[dict[str, Any]],
) -> None:
    daily_matrix = np.vstack([
        result["daily_pnl"]
        for result in results
    ])

    quantiles = np.quantile(
        daily_matrix,
        [0.05, 0.25, 0.50, 0.75, 0.95],
        axis=0,
    )

    mean = daily_matrix.mean(axis=0)

    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "future_day",
            "mean_pnl",
            "p05_pnl",
            "p25_pnl",
            "median_pnl",
            "p75_pnl",
            "p95_pnl",
        ])

        for day in range(daily_matrix.shape[1]):
            writer.writerow([
                day + 1,
                mean[day],
                quantiles[0, day],
                quantiles[1, day],
                quantiles[2, day],
                quantiles[3, day],
                quantiles[4, day],
            ])


def build_summary_text(
    args: argparse.Namespace,
    observed_prices: np.ndarray,
    historical_metrics: dict[str, float] | None,
    results: list[dict[str, Any]],
) -> str:
    scores = np.asarray([
        result["score"]
        for result in results
    ])
    mean_pnl = np.asarray([
        result["mean_pl"]
        for result in results
    ])
    sharpe = np.asarray([
        result["sharpe"]
        for result in results
    ])
    volume = np.asarray([
        result["total_dollar_volume"]
        for result in results
    ])

    lines = [
        "MONTE CARLO ALGOTHON EVALUATION",
        "=" * 36,
        f"strategy: {Path(args.strategy).resolve()}",
        f"prices: {Path(args.prices).resolve()}",
        (
            f"observed matrix: "
            f"{observed_prices.shape[0]} instruments x "
            f"{observed_prices.shape[1]} days"
        ),
        f"simulated future: {args.future_days} days",
        f"paths: {args.paths}",
        f"direction sampling: {args.directions}",
        (
            "generator: six-state direction/volatility Markov model "
            "+ multivariate geometric block bootstrap"
        ),
        "",
    ]

    if historical_metrics is not None:
        lines.extend([
            (
                f"HISTORICAL LAST {args.historical_test_days} DAYS"
            ),
            (
                f"score={historical_metrics['score']:.2f}  "
                f"meanPL={historical_metrics['mean_pl']:.2f}  "
                f"stdPL={historical_metrics['std_pl']:.2f}  "
                f"Sharpe={historical_metrics['sharpe']:.2f}  "
                f"volume={historical_metrics['total_dollar_volume']:.0f}"
            ),
            "",
        ])

    lines.extend([
        "ALL SIMULATED PATHS",
        format_distribution("Score", scores),
        format_distribution("Mean daily P&L", mean_pnl),
        format_distribution("Annualised Sharpe", sharpe),
        format_distribution("Dollar volume", volume),
        "",
        "SCORE BY DIRECTION",
    ])

    for direction in ("bear", "neutral", "bull"):
        direction_scores = np.asarray([
            result["score"]
            for result in results
            if result["direction"] == direction
        ])

        if len(direction_scores):
            lines.append(
                format_distribution(
                    direction.capitalize(),
                    direction_scores,
                )
            )

    lines.extend([
        "",
        "SCORE THRESHOLD PROBABILITIES",
    ])

    for threshold in args.thresholds:
        probability = float(
            np.mean(scores >= threshold)
        )
        lines.append(
            f"P(score >= {threshold:g}) = {probability:.3f}"
        )

    lines.extend([
        f"P(score > 0) = {float(np.mean(scores > 0.0)):.3f}",
        (
            "P(mean daily P&L > 0) = "
            f"{float(np.mean(mean_pnl > 0.0)):.3f}"
        ),
        "",
        "INTERPRETATION",
        (
            "These are conditional scenario results, not a genuine forecast "
            "of days 1001-1500. A strategy can look strong when the simulator "
            "reproduces structures it was designed to exploit. Compare "
            "strategies under the same seeds and inspect bearish, neutral and "
            "bullish subsets separately."
        ),
    ])

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate days 1001-1500 and evaluate a strategy "
            "over many Monte Carlo continuations."
        )
    )

    parser.add_argument(
        "--strategy",
        default="Sigma0.py",
        help="Python strategy file exposing getMyPosition().",
    )
    parser.add_argument(
        "--prices",
        default="prices_full_final.txt",
        help="Observed price file, normally containing days 1-1000.",
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=200,
        help="Number of Monte Carlo paths.",
    )
    parser.add_argument(
        "--future-days",
        type=int,
        default=500,
        help="Number of future days to simulate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9418,
        help="Master random seed.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes. Use 1 for easiest debugging.",
    )
    parser.add_argument(
        "--directions",
        choices=[
            "balanced",
            "empirical",
            "bear",
            "neutral",
            "bull",
        ],
        default="balanced",
        help=(
            "How to choose long-run drift scenarios. Balanced gives "
            "approximately one-third bear, neutral and bull paths."
        ),
    )
    parser.add_argument(
        "--trend-window",
        type=int,
        default=20,
        help="Days used to classify directional regimes.",
    )
    parser.add_argument(
        "--volatility-window",
        type=int,
        default=60,
        help="Days used to classify and sample volatility regimes.",
    )
    parser.add_argument(
        "--drift-window",
        type=int,
        default=250,
        help="Historical window used to sample long-run drift vectors.",
    )
    parser.add_argument(
        "--block-length",
        type=float,
        default=10.0,
        help="Mean geometric bootstrap block length.",
    )
    parser.add_argument(
        "--drift-strength",
        type=float,
        default=0.35,
        help=(
            "Fraction of sampled historical long-run drift difference "
            "added to each path."
        ),
    )
    parser.add_argument(
        "--volatility-refresh-days",
        type=int,
        default=60,
        help="How often a path partially refreshes its volatility scale.",
    )
    parser.add_argument(
        "--transition-smoothing",
        type=float,
        default=1.0,
        help="Laplace smoothing for regime transition probabilities.",
    )
    parser.add_argument(
        "--save-paths",
        type=int,
        default=0,
        help="Save the first N complete 1-1500 price files.",
    )
    parser.add_argument(
        "--save-directory",
        default="simulated_paths",
        help="Directory for saved price paths.",
    )
    parser.add_argument(
        "--results",
        default="monte_carlo_results.csv",
        help="Per-path result CSV.",
    )
    parser.add_argument(
        "--daily-results",
        default="monte_carlo_daily_summary.csv",
        help="Across-path daily P&L summary CSV.",
    )
    parser.add_argument(
        "--summary",
        default="monte_carlo_summary.txt",
        help="Text summary output.",
    )
    parser.add_argument(
        "--historical-test-days",
        type=int,
        default=500,
        help=(
            "Also evaluate the strategy on the observed final N days. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[500.0, 700.0, 800.0, 1000.0],
        help="Score thresholds for reported probabilities.",
    )

    args = parser.parse_args()

    if args.paths <= 0:
        parser.error("--paths must be positive.")

    if args.future_days <= 0:
        parser.error("--future-days must be positive.")

    if args.workers <= 0:
        parser.error("--workers must be positive.")

    if args.save_paths < 0:
        parser.error("--save-paths cannot be negative.")

    return args


def main() -> None:
    args = parse_args()

    prices_path = Path(args.prices).resolve()
    strategy_path = Path(args.strategy).resolve()

    observed_prices, names = load_prices(prices_path)

    print(
        f"Loaded {observed_prices.shape[0]} instruments "
        f"for {observed_prices.shape[1]} observed days"
    )
    print(
        f"Simulating {args.paths} paths x "
        f"{args.future_days} future days"
    )
    print(
        "Generator: directional/volatility regimes "
        "+ multivariate block bootstrap"
    )

    fit_start = time.time()

    generator_model = fit_generator(
        observed_prices,
        trend_window=args.trend_window,
        volatility_window=args.volatility_window,
        drift_window=args.drift_window,
        block_length=args.block_length,
        transition_smoothing=args.transition_smoothing,
    )

    print(
        f"Generator fitted in {time.time() - fit_start:.2f}s"
    )

    historical_metrics = None

    if args.historical_test_days > 0:
        if args.historical_test_days >= observed_prices.shape[1]:
            raise ValueError(
                "--historical-test-days must be smaller than "
                "the number of observed days."
            )

        historical_metrics, _ = evaluate_like_official(
            observed_prices,
            strategy_path,
            start_day_count=(
                observed_prices.shape[1]
                - args.historical_test_days
            ),
            unique_tag="historical",
        )

        print(
            f"Historical last-{args.historical_test_days}: "
            f"score {historical_metrics['score']:.2f}, "
            f"mean PL {historical_metrics['mean_pl']:.2f}, "
            f"Sharpe {historical_metrics['sharpe']:.2f}"
        )

    master_rng = np.random.default_rng(args.seed)
    path_seeds = master_rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=args.paths,
        dtype=np.uint32,
    )

    tasks = [
        (index, int(path_seeds[index]))
        for index in range(args.paths)
    ]

    worker_args = (
        observed_prices,
        names,
        generator_model,
        str(strategy_path),
        args.future_days,
        args.directions,
        args.drift_strength,
        args.volatility_refresh_days,
        min(args.save_paths, args.paths),
        str(Path(args.save_directory).resolve()),
    )

    results: list[dict[str, Any]] = []
    simulation_start = time.time()
    progress_every = max(1, args.paths // 20)

    if args.workers == 1:
        _initialise_worker(*worker_args)

        for completed, task in enumerate(tasks, start=1):
            results.append(_run_one_path(task))

            if (
                completed % progress_every == 0
                or completed == args.paths
            ):
                print(
                    f"Completed {completed}/{args.paths} paths"
                )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialise_worker,
            initargs=worker_args,
        ) as executor:
            future_map = {
                executor.submit(
                    _run_one_path,
                    task,
                ): task[0]
                for task in tasks
            }

            completed = 0

            for future in as_completed(future_map):
                results.append(future.result())
                completed += 1

                if (
                    completed % progress_every == 0
                    or completed == args.paths
                ):
                    print(
                        f"Completed {completed}/{args.paths} paths"
                    )

    results.sort(key=lambda item: item["path"])

    elapsed = time.time() - simulation_start
    print(f"Simulation/evaluation time: {elapsed:.1f}s")

    write_results_csv(
        args.results,
        results,
    )
    write_daily_summary(
        args.daily_results,
        results,
    )

    summary_text = build_summary_text(
        args,
        observed_prices,
        historical_metrics,
        results,
    )

    Path(args.summary).write_text(
        summary_text,
        encoding="utf-8",
    )

    print()
    print(summary_text, end="")
    print()
    print(f"Wrote {args.results}")
    print(f"Wrote {args.daily_results}")
    print(f"Wrote {args.summary}")

    if args.save_paths:
        print(
            f"Saved {min(args.save_paths, args.paths)} paths "
            f"under {args.save_directory}"
        )


if __name__ == "__main__":
    main()