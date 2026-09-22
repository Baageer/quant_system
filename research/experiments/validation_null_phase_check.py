"""Is the validation-split null percentile an artifact of the window's start?

``random_portfolio_baseline --split validation`` gives the model one number and
the null 200, which is asymmetric: the null's spread absorbs selection luck, but
the model's single figure also carries *schedule* luck.  The repo already learned
this the hard way on the turnover study -- a nominally identical weekly cadence
swung 26pp on validation purely from which session of the week the book is struck
(``turnover_optimization_ab`` module docstring).

v1 rebalances on the ISO week, whose dates the calendar fixes, so there is no
free phase to sweep.  What *can* be perturbed is where the window opens: start
the backtest k sessions later and the initial book, the first rebalance date and
the cash-timing all move.  This script runs the **actual** arm (and optionally a
few random draws) across those shifts and reports the spread, so the null
percentile can be quoted with a noise floor instead of as a point estimate.

Two modes, one per axis, and they answer different questions:

``--cadence N --phases M``
    Counts the cadence in *sessions* and sweeps the starting phase.  This is how
    a "is the cadence itself special?" question gets asked, but the cadence it
    sweeps is no longer weekly.

``--weekday``
    Keeps the ISO-week cadence and moves which day of the week the book is
    struck (``make_weekday_rank_strategy``).  This is the only way to sweep
    phase *while holding cadence fixed to weekly*, so it is the one that speaks
    to whether v1's production Monday strike is schedule luck.  Result files are
    named ``validation_null_phase_weekday_*`` so the counter-phase artifacts are
    never overwritten.

Both are paired across trials: the RNG is reseeded per trial so every phase sees
the same permutations and the spread isolates phase from selection luck.  Use
``--random-seeds 60`` to match the published counter-phase N, or the two spreads
are not comparable.

Examples::

    python -m research.experiments.validation_null_phase_check --model-id zz1000_pit_outcome
    python -m research.experiments.validation_null_phase_check \\
        --model-id hs300_excess_v1_outcome --weekday --random-seeds 60 --tag weekday
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from research.experiments.random_portfolio_baseline import (
    _score_slots,
    _write_rank_score,
)
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.probability_trade_evaluation import _run_engine, summarize_backtest
from research.market_regime.tradability_ranking_evaluation import (
    apply_rank_selection,
    build_rank_market_data,
    make_rank_strategy,
    select_top_k,
)
from research.market_regime.trade_outcome_ranking_evaluation import SCORE_COLUMN, load_outcome_scores
from research.market_regime.turnover_optimization_ab import (
    make_periodic_rank_strategy,
    make_weekday_rank_strategy,
    phase_offsets,
)


WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri")


def weekday_strike_dates(dates, weekday: int):
    """The sessions a weekday-shifted ISO-week cadence would strike.

    Mirrors the rebalance test in ``make_weekday_rank_strategy``: the first
    session of each ISO week whose weekday is on or after ``weekday``.  A week
    with no such session (a lone-Monday week under ``weekday=4``) is skipped, so
    reporting these counts is what keeps "different weekday" from being silently
    read as "different cadence".
    """

    stamps = [pd.Timestamp(date) for date in dates]
    seen = {}
    for stamp in stamps:
        iso = stamp.isocalendar()
        seen.setdefault((int(iso[0]), int(iso[1])), []).append(stamp)
    return [min(day for day in days_of_week if day.weekday() >= weekday)
            for days_of_week in seen.values()
            if any(day.weekday() >= weekday for day in days_of_week)]


def _context(scores, start_date, max_positions, initial_capital, horizon, config_path,
             cadence=None, cadence_offset=0, weekday=None):
    return {
        "max_positions": max_positions,
        "target_notional": initial_capital / max_positions,
        "horizon": horizon,
        "initial_capital": initial_capital,
        "config_path": config_path,
        "start_date": start_date,
        "end_date": scores["date"].max(),
        "cadence": cadence,
        "cadence_offset": int(cadence_offset),
        "weekday": weekday,
    }


def _strategy(context):
    """Production ISO-week strategy, a session-counted cadence at a phase, or a weekday shift.

    ``cadence`` and ``weekday`` both None reproduces v1 exactly.  The session
    cadence makes the starting phase a free parameter.  ``weekday`` keeps the
    ISO-week clock but moves which day of the week the book is struck, which is
    the only way to sweep phase while holding the *cadence* fixed to weekly --
    the ISO-week dates themselves are pinned by the calendar.
    """

    cadence = context.get("cadence")
    weekday = context.get("weekday")
    if weekday is not None:
        return make_weekday_rank_strategy(
            context["max_positions"], context["target_notional"], context["horizon"],
            weekday=int(weekday),
        )
    if cadence is None:
        return make_rank_strategy(context["max_positions"], context["target_notional"],
                                  min_holding_days=context["horizon"])
    return make_periodic_rank_strategy(
        context["max_positions"], context["target_notional"], context["horizon"],
        rebalance_days=cadence, offset=context["cadence_offset"],
    )


def _run_one(prepared, context):
    results, trades = _run_engine(
        prepared,
        _strategy(context),
        context["initial_capital"],
        context["config_path"],
        context["start_date"],
        context["end_date"],
    )
    return summarize_backtest("phase", results, trades, context["initial_capital"])


def run_phase_check(
    model_id: str,
    dataset_dir: Path,
    split: str = "validation",
    offsets: Sequence[int] = (0, 1, 2, 3, 4),
    top_k: float = 0.1,
    horizon: int = 10,
    initial_capital: float = 500000.0,
    max_positions: int = 20,
    random_seeds: int = 0,
    seed: int = 20260915,
    cadence: Optional[int] = None,
    phases: int = 0,
    weekdays: Optional[Sequence[int]] = None,
    config_path: str = "config/settings.yaml",
    verbose: bool = True,
) -> Dict[str, object]:
    scores = load_outcome_scores(dataset_dir, model_id, split, horizon)
    raw_market_data, skipped = build_rank_market_data(scores, None, SCORE_COLUMN)
    slots = _score_slots(scores, raw_market_data)

    sessions = np.sort(scores["date"].unique())
    base_start = scores["date"].min()
    values = np.asarray(scores[SCORE_COLUMN], dtype=float)
    rng = np.random.default_rng(seed)

    # (window-opening offset, cadence, phase, weekday) -- one trial each.
    trials: List[tuple] = []
    for offset in offsets:
        # Window-opening trials keep the production ISO-week strategy: the point
        # is to move the window, not the cadence.
        trials.append((offset, None, None, None))
    if phases and cadence:
        for phase in phase_offsets(cadence, phases):
            trials.append((0, cadence, phase, None))
    for weekday in weekdays or ():
        # The weekday family holds the ISO-week *cadence* while moving which day
        # of the week the book is struck.  It is the only free phase for an
        # ISO-week arm, so for these arms it is the "phase average" axis.
        trials.append((0, None, None, int(weekday)))

    rows: List[Dict[str, object]] = []
    selected = select_top_k(scores, top_k, SCORE_COLUMN)
    for window_offset, trial_cadence, trial_phase, trial_weekday in trials:
        if window_offset >= len(sessions):
            continue
        start = pd.Timestamp(sessions[window_offset])
        context = _context(scores, start, max_positions, initial_capital, horizon, config_path,
                           cadence=trial_cadence, cadence_offset=trial_phase or 0,
                           weekday=trial_weekday)
        prepared = apply_rank_selection(raw_market_data, scores, selected)
        actual = _run_one(prepared, context)
        rebalances = None
        if trial_weekday is not None:
            rebalances = len(weekday_strike_dates(sessions[window_offset:], trial_weekday))
        # Which axis this trial perturbs.  Derived here because ``phase``/``weekday``
        # are both NaN for the window trials -- classifying on ``phase.isna()``
        # alone would file every weekday trial under "window".
        kind = ("weekday" if trial_weekday is not None
                else "phase" if trial_phase is not None else "window")
        row = {
            "kind": kind,
            "window_offset_sessions": int(window_offset),
            "cadence": trial_cadence,
            "phase": trial_phase,
            "weekday": trial_weekday,
            "start_date": str(start.date()),
            "rebalance_count": rebalances,
            "actual_total_return": float(actual["total_return"]),
            "actual_sharpe": float(actual["sharpe_zero_rate"]),
            "actual_max_drawdown": float(actual["max_drawdown"]),
            "random_mean": float("nan"),
            "random_sd": float("nan"),
            "percentile": float("nan"),
        }
        if random_seeds:
            # Reseed per trial so the same permutations are drawn under every
            # phase: the phases are then paired and the spread isolates phase
            # rather than mixing in selection luck.
            rng = np.random.default_rng(seed)
            probe = scores.copy()
            drawn = []
            for _ in range(random_seeds):
                shuffled = rng.permutation(values)
                probe[SCORE_COLUMN] = shuffled
                pick = select_top_k(probe, top_k, SCORE_COLUMN)
                prep = apply_rank_selection(raw_market_data, probe, pick)
                _write_rank_score(prep, shuffled, slots)
                drawn.append(float(_run_one(prep, context)["total_return"]))
            drawn_arr = np.asarray(drawn)
            row["random_mean"] = float(drawn_arr.mean())
            row["random_sd"] = float(drawn_arr.std(ddof=1)) if len(drawn_arr) > 1 else 0.0
            row["percentile"] = float((drawn_arr <= row["actual_total_return"]).mean())
        rows.append(row)
        if verbose:
            if kind == "weekday":
                label = "{} ({} rb)".format(WEEKDAY_NAMES[trial_weekday], rebalances)
            elif kind == "phase":
                label = "cadence={} phase={}".format(trial_cadence, trial_phase)
            else:
                label = "window+{}".format(window_offset)
            print(
                "  {:<16} start {}  actual {:>8.2%}  random {:>8.2%}  pctile {:>6}".format(
                    label, start.date(), row["actual_total_return"], row["random_mean"],
                    "{:.1%}".format(row["percentile"]) if random_seeds else "-",
                ),
                flush=True,
            )

    frame = pd.DataFrame(rows)
    windows = frame[frame["kind"] == "window"]
    phase_rows = frame[frame["kind"] == "phase"]
    weekday_rows = frame[frame["kind"] == "weekday"]

    def spread(part):
        if not len(part):
            return float("nan")
        return float((part["actual_total_return"].max()
                      - part["actual_total_return"].min()) * 100.0)

    # The family comparison B1 exists to make.  ``baseline_return`` is whichever
    # trial sits on the production calendar -- the Monday strike for the weekday
    # family, phase 0 for the counter family -- and ``rank`` is where it lands
    # among its siblings (1 = worst).  A baseline at the top of a wide family is
    # schedule luck; a baseline at the top of a *narrow* family is a real effect.
    weekday_baseline = weekday_rows[weekday_rows["weekday"] == 0]
    baseline_return = (float(weekday_baseline["actual_total_return"].iloc[0])
                       if len(weekday_baseline) else float("nan"))
    summary = {
        "model_id": model_id,
        "split": split,
        "top_k": top_k,
        "cadence": cadence,
        "phases": int(phases),
        "weekdays": sorted(int(v) for v in weekday_rows["weekday"]) if len(weekday_rows) else [],
        "random_seeds": int(random_seeds),
        "skipped_prices": skipped,
        "rows": rows,
        "window_spread_pp": spread(windows),
        "phase_spread_pp": spread(phase_rows),
        "phase_min_percentile": float(phase_rows["percentile"].min()) if len(phase_rows) else float("nan"),
        "phase_max_percentile": float(phase_rows["percentile"].max()) if len(phase_rows) else float("nan"),
        "weekday_spread_pp": spread(weekday_rows),
        "weekday_baseline_return": baseline_return,
        "weekday_baseline_rank": (
            int((weekday_rows["actual_total_return"] <= baseline_return).sum()) if len(weekday_baseline) else None
        ),
        "weekday_trials": int(len(weekday_rows)),
        "weekday_min_percentile": float(weekday_rows["percentile"].min()) if len(weekday_rows) else float("nan"),
        "weekday_max_percentile": float(weekday_rows["percentile"].max()) if len(weekday_rows) else float("nan"),
    }
    return summary


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--offsets", type=int, nargs="+", default=None,
                        help="Window-opening shifts to sweep. Defaults to 0..4, or to none "
                             "when --weekday is given (which is the usual pairing).")
    parser.add_argument("--top-k", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--random-seeds", type=int, default=0)
    parser.add_argument("--cadence", type=int, default=None,
                        help="Session-counted rebalance cadence. Omit for the production ISO week.")
    parser.add_argument("--phases", type=int, default=0,
                        help="Starting phases to sweep when --cadence is set.")
    parser.add_argument("--weekday", action="store_true",
                        help="Sweep Mon..Fri ISO-week strikes. Use --random-seeds 60 to match "
                             "the published counter-phase N so the two spreads are comparable.")
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    if args.offsets is not None:
        offsets = args.offsets
    else:
        offsets = [] if args.weekday else [0, 1, 2, 3, 4]
    weekdays = list(range(len(WEEKDAY_NAMES))) if args.weekday else None

    summary = run_phase_check(
        args.model_id,
        Path(args.dataset_dir),
        split=args.split,
        offsets=offsets,
        top_k=args.top_k,
        horizon=args.horizon,
        random_seeds=args.random_seeds,
        cadence=args.cadence,
        phases=args.phases,
        weekdays=weekdays,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # The weekday family gets its own name: the counter-phase JSON is a published
    # artifact and must not be overwritten by a differently-meant sweep.
    stem = "validation_null_phase_weekday" if args.weekday else "validation_null_phase"
    name = "{}_{}_{}{}.json".format(
        stem, args.model_id, args.split, "_{}".format(args.tag) if args.tag else ""
    )
    (out / name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    if offsets:
        print("spread across window openings: {:.2f}pp".format(summary["window_spread_pp"]))
    if summary["phase_spread_pp"] == summary["phase_spread_pp"]:
        print("spread across phases (cadence={}): {:.2f}pp   percentiles {:.1%}..{:.1%}".format(
            summary["cadence"], summary["phase_spread_pp"],
            summary["phase_min_percentile"], summary["phase_max_percentile"]))
    if summary["weekday_trials"]:
        print("spread across weekdays: {:.2f}pp   percentiles {:.1%}..{:.1%}".format(
            summary["weekday_spread_pp"],
            summary["weekday_min_percentile"], summary["weekday_max_percentile"]))
        print("production Monday strike: {:+.2%}  rank {}/{} among the five".format(
            summary["weekday_baseline_return"], summary["weekday_baseline_rank"],
            summary["weekday_trials"]))


if __name__ == "__main__":
    main()
