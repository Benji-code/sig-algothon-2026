#!/usr/bin/env python
"""Algothon 2026 dual-window evaluator.

Expected files beside this script:
    Sigma0.py
    prices_full.txt

The price file must contain all 1,000 days. The evaluator reports two
independent scores:

1. Days 751-1000
   - The model first receives days 1-750.
   - It opens the first evaluated position at the day-750 close.
   - Exactly 250 daily P&Ls are scored.

2. Days 501-1000
   - The model first receives days 1-500.
   - It opens the first evaluated position at the day-500 close.
   - Exactly 500 daily P&Ls are scored.

Each window imports Sigma0.py into a fresh module instance so state from the
first evaluation cannot leak into the second evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


SCORE_DEFAULT_PARAM = 1.0

DEFAULT_COMM_RATE = 0.0001
INST0_COMM_RATE = 0.00002

DEFAULT_DOLLAR_POSITION_LIMIT = 10_000
INST0_DOLLAR_POSITION_LIMIT = 100_000

MODEL_FILENAME = "Sigma0.py"
DEFAULT_PRICES_FILENAME = "prices_full.txt"

# Change to True to reproduce the original evaluator's per-day output.
PRINT_DAILY_PNL = False


@dataclass(frozen=True)
class EvaluationResult:
    label: str
    start_day: int
    end_day: int
    pnl_days: int
    mean_pl: float
    return_on_turnover: float
    std_pl: float
    annualised_sharpe: float
    total_dollar_volume: float
    score: float


def locate_prices_file(script_dir: Path, requested: str | None) -> Path:
    """Resolve the supplied path or discover a prices_full*.txt file."""
    if requested:
        supplied = Path(requested).expanduser()
        candidates = [
            supplied,
            Path.cwd() / supplied,
            script_dir / supplied,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"Could not find price file: {requested}")

    exact = script_dir / DEFAULT_PRICES_FILENAME
    if exact.is_file():
        return exact.resolve()

    matches = sorted(script_dir.glob("prices_full*.txt"))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        names = "\n".join(f"  - {path.name}" for path in matches)
        raise FileNotFoundError(
            "Multiple prices_full*.txt files were found. Supply the desired "
            f"filename on the command line:\n{names}"
        )

    raise FileNotFoundError(
        f"Could not find {DEFAULT_PRICES_FILENAME} or any prices_full*.txt "
        f"beside {Path(__file__).name}."
    )


def load_prices_full(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load a day-by-instrument table and return instruments-by-day prices."""
    frame = pd.read_csv(path, sep=r"\s+", header=0, index_col=None)

    if frame.shape[0] < 1000:
        raise ValueError(
            f"{path.name} contains {frame.shape[0]} days; at least 1000 are required."
        )
    if frame.shape[1] < 1:
        raise ValueError(f"{path.name} contains no instruments.")

    values = frame.iloc[:1000].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contains NaN or infinite prices.")
    if np.any(values <= 0):
        raise ValueError(f"{path.name} contains non-positive prices.")

    return values.T, list(frame.columns)


def load_fresh_model(model_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    """Import Sigma0.py as a fresh module and return getMyPosition."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Could not find {MODEL_FILENAME} beside the evaluator: {model_path}"
        )

    # Allow Sigma0.py to import helper modules stored in the same directory.
    model_directory = str(model_path.parent)
    if model_directory not in sys.path:
        sys.path.insert(0, model_directory)

    module_name = f"_sigma0_eval_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {model_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    reset_state = getattr(module, "reset_state", None)
    if callable(reset_state):
        reset_state()

    get_position = getattr(module, "getMyPosition", None)
    if not callable(get_position):
        raise AttributeError(
            f"{MODEL_FILENAME} must define a callable getMyPosition(prcSoFar)."
        )

    return get_position


def charge_fees(dollar_volumes: np.ndarray, commission_rates: np.ndarray) -> float:
    """Calculate total commission for one day's trades."""
    return float(np.sum(dollar_volumes * commission_rates))


def calculate_score(
    mean_pl: float,
    std_pl: float,
    parameter: float = SCORE_DEFAULT_PARAM,
) -> float:
    """Competition score from daily-P&L mean and standard deviation."""
    if mean_pl <= 0 or std_pl < 1e-10:
        return float(mean_pl)

    sharpe = np.sqrt(250.0) * mean_pl / std_pl
    fraction = sharpe**2 / (sharpe**2 + parameter**2)
    return float(mean_pl * fraction)


def evaluate_window(
    prices: np.ndarray,
    get_position: Callable[[np.ndarray], np.ndarray],
    num_test_days: int,
    label: str,
    commission_rates: np.ndarray,
    dollar_position_limits: np.ndarray,
    print_daily: bool = False,
) -> EvaluationResult:
    """Evaluate one independent trailing window using official mechanics."""
    n_instruments, total_days = prices.shape

    if num_test_days <= 0 or num_test_days >= total_days:
        raise ValueError(
            f"num_test_days must be between 1 and {total_days - 1}; "
            f"received {num_test_days}."
        )

    start_index = total_days - num_test_days

    cash = 0.0
    current_position = np.zeros(n_instruments, dtype=int)
    total_dollar_volume = 0.0
    portfolio_value = 0.0
    pending_commission = 0.0
    daily_pnl: list[float] = []

    # At t=start_index the model sees all prices through that day and opens
    # the first evaluated position. P&L begins on the following mark.
    for t in range(start_index, total_days + 1):
        price_history_so_far = prices[:, :t]
        current_prices = price_history_so_far[:, -1]

        if t < total_days:
            proposed = np.asarray(get_position(price_history_so_far))
            if proposed.shape != (n_instruments,):
                raise ValueError(
                    f"getMyPosition returned shape {proposed.shape}; "
                    f"expected {(n_instruments,)}."
                )
            if not np.isfinite(proposed).all():
                raise ValueError("getMyPosition returned NaN or infinite positions.")

            share_limits = (dollar_position_limits / current_prices).astype(int)
            new_position = np.clip(
                proposed, -share_limits, share_limits
            ).astype(int)
        else:
            # The final day only marks the existing portfolio.
            new_position = current_position.copy()

        position_change = new_position - current_position

        # Preserve the original evaluator's delayed-commission convention:
        # commission generated by yesterday's trade is charged today.
        cash -= float(current_prices @ position_change) + pending_commission

        dollar_volumes = current_prices * np.abs(position_change)
        total_dollar_volume += float(np.sum(dollar_volumes))
        pending_commission = charge_fees(dollar_volumes, commission_rates)

        current_position = new_position
        marked_value = cash + float(current_position @ current_prices)
        today_pl = marked_value - portfolio_value
        portfolio_value = marked_value

        return_on_turnover = (
            portfolio_value / total_dollar_volume
            if total_dollar_volume > 0
            else 0.0
        )

        if t > start_index:
            daily_pnl.append(today_pl)
            if print_daily:
                print(
                    f"[{label}] Day {t}: value={portfolio_value:.2f} "
                    f"todayPL=${today_pl:.2f} "
                    f"$-traded={total_dollar_volume:.0f} "
                    f"return={return_on_turnover:.5f}"
                )

    pnl = np.asarray(daily_pnl, dtype=float)
    if pnl.size != num_test_days:
        raise RuntimeError(
            f"{label}: scored {pnl.size} P&L days; expected {num_test_days}."
        )

    mean_pl = float(np.mean(pnl))
    std_pl = float(np.std(pnl))
    annualised_sharpe = (
        float(np.sqrt(250.0) * mean_pl / std_pl) if std_pl > 0 else 0.0
    )

    return EvaluationResult(
        label=label,
        start_day=total_days - num_test_days + 1,
        end_day=total_days,
        pnl_days=int(pnl.size),
        mean_pl=mean_pl,
        return_on_turnover=float(return_on_turnover),
        std_pl=std_pl,
        annualised_sharpe=annualised_sharpe,
        total_dollar_volume=float(total_dollar_volume),
        score=calculate_score(mean_pl, std_pl),
    )


def print_result(result: EvaluationResult) -> None:
    """Print one clearly labelled score summary."""
    print()
    print("=" * 64)
    print(f"{result.label} — DAYS {result.start_day}-{result.end_day}")
    print("=" * 64)
    print(f"Scored PnL days: {result.pnl_days}")
    print(f"mean(PL): {result.mean_pl:.2f}")
    print(f"return: {result.return_on_turnover:.5f}")
    print(f"StdDev(PL): {result.std_pl:.2f}")
    print(f"annSharpe(PL): {result.annualised_sharpe:.4f}")
    print(f"totDvolume: {result.total_dollar_volume:.0f}")
    print(f"Score: {result.score:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Sigma0.py independently on days 751-1000 and 501-1000."
        )
    )
    parser.add_argument(
        "prices_file",
        nargs="?",
        default=None,
        help=(
            "Optional full 1000-day price file. When omitted, the evaluator "
            "uses prices_full.txt or a unique prices_full*.txt beside itself."
        ),
    )
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Print every evaluated day's P&L.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / MODEL_FILENAME
    prices_path = locate_prices_file(script_dir, args.prices_file)

    prices, instrument_names = load_prices_full(prices_path)
    n_instruments, total_days = prices.shape

    print(f"Model: {model_path.name}")
    print(f"Prices: {prices_path.name}")
    print(f"Loaded {n_instruments} instruments for {total_days} days")
    print(f"Instrument 0: {instrument_names[0]}")

    commission_rates = np.full(
        n_instruments, DEFAULT_COMM_RATE, dtype=float
    )
    commission_rates[0] = INST0_COMM_RATE

    dollar_position_limits = np.full(
        n_instruments, DEFAULT_DOLLAR_POSITION_LIMIT, dtype=float
    )
    dollar_position_limits[0] = INST0_DOLLAR_POSITION_LIMIT

    # Fresh Sigma0 module for the 250-day evaluation.
    get_position_751_1000 = load_fresh_model(model_path)
    result_751_1000 = evaluate_window(
        prices=prices,
        get_position=get_position_751_1000,
        num_test_days=250,
        label="250-DAY WINDOW",
        commission_rates=commission_rates,
        dollar_position_limits=dollar_position_limits,
        print_daily=PRINT_DAILY_PNL or args.daily,
    )

    # Import Sigma0.py again so no state from the first run leaks into the
    # continuous 500-day evaluation.
    get_position_501_1000 = load_fresh_model(model_path)
    result_501_1000 = evaluate_window(
        prices=prices,
        get_position=get_position_501_1000,
        num_test_days=500,
        label="500-DAY WINDOW",
        commission_rates=commission_rates,
        dollar_position_limits=dollar_position_limits,
        print_daily=PRINT_DAILY_PNL or args.daily,
    )

    print_result(result_751_1000)
    print_result(result_501_1000)

    print()
    print("=" * 64)
    print("HEADLINE SCORES")
    print("=" * 64)
    print(f"Days 751-1000: {result_751_1000.score:.6f}")
    print(f"Days 501-1000: {result_501_1000.score:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())