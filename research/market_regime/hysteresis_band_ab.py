"""Can a rank band cut turnover without throwing away alpha?

The previous step (``turnover_optimization_ab``) falsified every *blanket* way of
slowing v1 down: extending the minimum holding period or lengthening the rebalance
cadence cuts turnover 74% but destroys 33.56pp of frictionless gross return to save
2.76pp of cost.  The reason is structural -- those levers make the whole book stale
relative to a 10-day signal, with no regard for *which* names went stale.

A rank band is a different mechanism.  v1 buys the top ``entry_k`` fraction and
sells as soon as a name leaves it.  Here a held name is kept until it drops out of
a **wider** ``exit_k`` band, while entries still only come from the narrow band::

    entry_k == exit_k      -> exactly v1 (parity arm)
    entry_k <  exit_k      -> "sticky" names in [entry_k, exit_k]

The point is that a name is retained only *while it is still ranked near the top*
-- unlike a holding-period extension, which retains it no matter how far it falls.
Whether that middle ground pays is the question.

**The strategy function is reused byte-for-byte.**  ``make_rank_strategy`` reads
``entry_trade`` and ``hold_trade`` off the frame; v1 derives both from one
``eod_candidate``, this module derives ``hold_trade`` from a wider selection.  So
every difference is attributable to the band and nothing else.

Cadence stays ISO-week.  That used to be described here as "no rebalance-phase
freedom to average over" -- which was the bug, not a property.  The ISO-week
dates are pinned to Monday, but *which day of the week the book is struck* is
free, and v1 happens to sit on the luckiest one: the weekday family spans tens of
pp on validation (``docs/research/05_engine_and_methodology/weekday_phase_axis.md``).

So the validation grid is now run once per weekday and phase-averaged, selection
goes through ``selection_rule.select_configuration`` (phase average + phase floor
+ turnover floor + cross-fold sign), and the test grid runs v1's production
Monday strike alone -- the family is a measurement of how much edge is schedule,
never a licence to pick a luckier clock on test.

Example::

    python -m research.market_regime.hysteresis_band_ab
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.market_regime.probability_trade_evaluation import _run_engine, summarize_backtest
from research.market_regime.selection_rule import (
    fold_boundaries,
    fold_returns_from_equity,
    phase_excess_table,
    select_configuration,
)
from research.market_regime.tradability_ranking_evaluation import (
    build_rank_market_data,
    make_rank_strategy,
)
from research.market_regime.trade_outcome_ranking_evaluation import (
    SCORE_COLUMN,
    load_outcome_scores,
    outcome_quality,
)
from research.market_regime.turnover_optimization_ab import make_weekday_rank_strategy


BASELINE_ARM = "band_entry10_exit10"


def select_rank_band(
    scores: pd.DataFrame, entry_k: float, exit_k: float, score_column: str = SCORE_COLUMN
) -> Tuple[pd.Series, pd.Series]:
    """Daily membership for the narrow entry band and the wide hold band.

    Same estimator as ``tradability_ranking_evaluation.select_top_k`` (rank
    ``method="first"`` descending, cut at ``ceil(count * k)``) so that
    ``entry_k == exit_k`` reproduces v1's selection exactly.
    """

    for name, value in (("entry_k", entry_k), ("exit_k", exit_k)):
        if not 0 < value <= 1:
            raise ValueError("{} must be in (0, 1]".format(name))
    if exit_k < entry_k:
        raise ValueError("exit_k must be >= entry_k (the hold band must contain the entry band)")
    ranks = scores.groupby("date", sort=False)[score_column].rank(method="first", ascending=False)
    counts = scores.groupby("date", sort=False)[score_column].transform("size")
    entry = (ranks <= np.ceil(counts * entry_k)).astype(bool)
    held = (ranks <= np.ceil(counts * exit_k)).astype(bool)
    return entry, held


def apply_band_selection(
    raw_market_data: Dict[str, pd.DataFrame],
    scores: pd.DataFrame,
    entry_selected: pd.Series,
    hold_selected: pd.Series,
) -> Dict[str, pd.DataFrame]:
    """Shift EOD membership to next-session signals, with the two bands separate.

    Mirrors ``tradability_ranking_evaluation.apply_rank_selection`` apart from
    sourcing ``hold_trade`` from its own (wider) selection.
    """

    decisions = scores.loc[:, ["date", "symbol"]].copy()
    decisions["entry_candidate"] = np.asarray(entry_selected, dtype=bool)
    decisions["hold_candidate"] = np.asarray(hold_selected, dtype=bool)
    prepared = {}
    for symbol, prices in raw_market_data.items():
        frame = prices.copy()
        symbol_decisions = decisions[decisions["symbol"] == symbol].set_index("date")
        frame = frame.join(symbol_decisions, how="left")
        frame["entry_candidate"] = frame["entry_candidate"].fillna(False).astype(bool)
        frame["hold_candidate"] = frame["hold_candidate"].fillna(False).astype(bool)
        frame["entry_trade"] = frame["entry_candidate"].shift(1).fillna(False).astype(bool)
        frame["hold_trade"] = frame["hold_candidate"].shift(1).fillna(False).astype(bool)
        prepared[symbol] = frame
    return prepared


def band_specs(entry_values: Sequence[float], exit_values: Sequence[float]) -> List[Dict[str, object]]:
    """Every (entry, exit) pair with ``exit >= entry``, baseline first."""

    specs = [{"arm": BASELINE_ARM, "entry_k": 0.10, "exit_k": 0.10}]
    for entry_k in entry_values:
        for exit_k in exit_values:
            if exit_k < entry_k or (entry_k == 0.10 and exit_k == 0.10):
                continue
            specs.append({
                "arm": "band_entry{:02d}_exit{:02d}".format(int(round(entry_k * 100)), int(round(exit_k * 100))),
                "entry_k": float(entry_k),
                "exit_k": float(exit_k),
            })
    return specs


def evaluate_bands(
    scores: pd.DataFrame,
    raw_market_data: Dict[str, pd.DataFrame],
    specs: Sequence[Dict[str, object]],
    horizon: int,
    initial_capital: float,
    max_positions: int,
    config_path: str,
    label: str,
    frictionless: bool = False,
    collect_equity: bool = False,
    weekday: Optional[int] = None,
):
    start_date, end_date = scores["date"].min(), scores["date"].max()
    target_notional = initial_capital / max_positions
    started = time.time()
    rows = []
    equity = {}
    for index, spec in enumerate(specs, start=1):
        entry_selected, hold_selected = select_rank_band(scores, spec["entry_k"], spec["exit_k"])
        prepared = apply_band_selection(raw_market_data, scores, entry_selected, hold_selected)
        # The band machinery is cadence-agnostic; only which session is a
        # rebalance session changes.  ``weekday=None`` is the production ISO week
        # (Monday strike), which is what every earlier run of this grid used.
        strategy = (make_rank_strategy(max_positions, target_notional, min_holding_days=horizon)
                    if weekday is None else
                    make_weekday_rank_strategy(max_positions, target_notional, horizon,
                                               weekday=int(weekday)))
        results, trades = _run_engine(
            prepared, strategy, initial_capital, config_path, start_date, end_date, frictionless=frictionless)
        if collect_equity:
            equity[str(spec["arm"])] = results["portfolio_value"].astype(float).copy()
        performance = summarize_backtest(str(spec["arm"]), results, trades, initial_capital)
        # Entry-band quality keeps the outcome comparable across arms; the hold
        # band says how much wider the retention set is.
        rows.append({
            "split": label,
            "weekday": -1 if weekday is None else int(weekday),
            "arm": spec["arm"],
            "entry_k": spec["entry_k"],
            "exit_k": spec["exit_k"],
            **{key: value for key, value in outcome_quality(scores, entry_selected, horizon).items()
               if key != "selected_count"},
            "entry_selected_count": int(np.asarray(entry_selected, dtype=bool).sum()),
            "hold_selected_count": int(np.asarray(hold_selected, dtype=bool).sum()),
            **{key: value for key, value in performance.items() if key != "strategy"},
        })
        print("  [{}] {}/{} {}".format(label, index, len(specs), spec["arm"]), flush=True)
    table = pd.DataFrame(rows)
    print("[{}] {} runs in {:.1f}s".format(label, len(rows), time.time() - started), flush=True)
    return (table, equity) if collect_equity else table


def phase_average(grid: pd.DataFrame) -> pd.DataFrame:
    """Collapse one row per (arm, weekday) into one row per arm, keeping the spread.

    ``phase_min``/``phase_max`` are the across-weekday extremes of the net
    return.  For the production ISO-week arm these are the only phase axis that
    exists -- its rebalance dates are pinned by the calendar -- which is exactly
    why the weekday family is what makes a "beat the baseline in the worst
    phase" gate mean anything here.
    """

    keys = ["arm", "entry_k", "exit_k"]
    numeric = [column for column in grid.select_dtypes(include=[np.number]).columns
               if column not in keys + ["weekday"]]
    averaged = grid.groupby(keys, as_index=False)[numeric].mean()
    spread = grid.groupby(keys, as_index=False).agg(
        phase_min=("total_return", "min"),
        phase_max=("total_return", "max"),
        phases=("total_return", "size"),
    )
    return averaged.merge(spread, on=keys, how="left")


def fold_excess_table(
    equity: Dict[Tuple[str, int], pd.Series],
    folds: Sequence[Dict[str, pd.Timestamp]],
    initial_capital: float,
    baseline_arm: str = BASELINE_ARM,
) -> pd.DataFrame:
    """Per-fold return in excess of the baseline arm, averaged over weekdays.

    ``equity`` maps ``(arm, weekday)`` to a daily equity series.  Each arm's fold
    return is differenced against the *baseline's* fold return **on the same
    weekday** first and only then averaged, so the fold gate asks "does this arm
    beat v1 on the same clock" and not "does it beat v1 by being on a luckier
    clock".
    """

    per_fold = {key: fold_returns_from_equity(series, folds, initial_capital)
                for key, series in equity.items()}
    weekdays = sorted({weekday for _, weekday in equity})
    rows = []
    for arm in sorted({arm for arm, _ in equity}):
        blocks = []
        for weekday in weekdays:
            arm_returns = per_fold.get((arm, weekday))
            base_returns = per_fold.get((baseline_arm, weekday))
            if arm_returns is None or base_returns is None:
                continue
            blocks.append([value - base for value, base in zip(arm_returns, base_returns)])
        if not blocks:
            continue
        mean = np.mean(np.asarray(blocks, dtype=float), axis=0)
        rows.append({"arm": arm, **{"fold_{}".format(index): float(value)
                                    for index, value in enumerate(mean)}})
    return pd.DataFrame(rows)


def yearly_returns(equity: pd.Series, initial_capital: float) -> pd.Series:
    """Per-calendar-year net return, measured off the daily equity marks."""

    marks = pd.concat([pd.Series([initial_capital], index=[equity.index[0] - pd.Timedelta(days=1)]), equity])
    returns = marks.pct_change().dropna()
    return returns.groupby(returns.index.year).apply(lambda block: float((1.0 + block).prod() - 1.0))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="output/datasets")
    parser.add_argument("--model-id", default="hs300_excess_v1_outcome")
    parser.add_argument("--output-dir", default="output/datasets")
    parser.add_argument("--run-id", default="hysteresis_band_v1")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--config-path", default="config/settings.yaml")
    parser.add_argument("--entry-k", default="0.05,0.10")
    parser.add_argument("--exit-k", default="0.10,0.15,0.20,0.30,0.50,0.70,1.00")
    parser.add_argument("--weekdays", default="0,1,2,3,4",
                        help="ISO-week strike days for the validation grid. Each arm is "
                             "phase-averaged over them; the test grid always runs the "
                             "production Monday strike alone. Pass an empty string to "
                             "reproduce the pre-2026-09 single-schedule grid.")
    parser.add_argument("--fold-months", type=int, default=12,
                        help="Validation sub-window length for the cross-fold sign gate.")
    parser.add_argument("--turnover-floor-frac", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset_dir, output_dir = Path(args.dataset_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entry_values = [float(value) for value in args.entry_k.split(",") if value.strip()]
    exit_values = [float(value) for value in args.exit_k.split(",") if value.strip()]
    specs = band_specs(entry_values, exit_values)
    print("Arms: {}".format([spec["arm"] for spec in specs]))

    loads = {split: load_outcome_scores(dataset_dir, args.model_id, split, args.horizon)
             for split in ("validation", "test")}
    raw_market, skipped = {}, {}
    for split in ("validation", "test"):
        raw_market[split], skipped[split] = build_rank_market_data(
            loads[split], args.price_history_dir, SCORE_COLUMN)

    weekdays = [int(value) for value in args.weekdays.split(",") if value.strip()]
    if not weekdays:
        weekdays = [None]

    # Validation is averaged over the weekday family -- v1's Monday strike is one
    # draw from it, and on the single-schedule grid it was the luckiest one.
    validation_tables, equity = [], {}
    for weekday in weekdays:
        table, phase_equity = evaluate_bands(
            loads["validation"], raw_market["validation"], specs, args.horizon,
            args.initial_capital, args.max_positions, args.config_path, "validation",
            collect_equity=True, weekday=weekday)
        validation_tables.append(table)
        equity.update({(arm, -1 if weekday is None else weekday): series
                       for arm, series in phase_equity.items()})
    validation_raw = pd.concat(validation_tables, ignore_index=True)
    validation = phase_average(validation_raw) if len(weekdays) > 1 else validation_raw

    folds = fold_boundaries(pd.Timestamp(loads["validation"]["date"].min()),
                            pd.Timestamp(loads["validation"]["date"].max()), args.fold_months)
    fold_excess = fold_excess_table(equity, folds, args.initial_capital)

    # The baseline gets the same weekday family the candidates do, so the phase
    # gate is paired.  Without it the gate would compare each candidate's worst
    # weekday against the baseline's *average*, which v1's Monday strike sits
    # 25.9pp above -- an unpaired gate rejects arms that beat v1 every day.
    phase_excess = None
    if len(weekdays) > 1:
        phase_excess = phase_excess_table(
            validation_raw, validation_raw[validation_raw["arm"] == BASELINE_ARM],
            key_columns=("arm",))

    # Selection is by rule, not by rank order.  ``sort_values(...).iloc[0]`` used
    # to pick ``exit_k=1.00`` here -- an arm that traded 20 times in the window
    # and scored -35.39% on test.  See ``selection_rule``.
    outcome = select_configuration(
        validation,
        baseline_key={"arm": BASELINE_ARM},
        key_columns=("arm",),
        fold_excess=fold_excess,
        phase_excess=phase_excess,
        turnover_floor_frac=args.turnover_floor_frac,
    )
    if outcome.row is None:
        print("\nselection: NO CHANGE -- {}".format(outcome.reason))
        selected = validation[validation["arm"] == BASELINE_ARM].iloc[0]
    else:
        print("\nselection: {}".format(outcome.reason))
        selected = outcome.row
    ranked = outcome.table
    # The test grid always runs v1's own calendar: the weekday family is a
    # validation-side measurement of how much of an edge is schedule, not a
    # licence to pick a luckier clock on test.
    test = evaluate_bands(
        loads["test"], raw_market["test"], specs, args.horizon,
        args.initial_capital, args.max_positions, args.config_path, "test")
    # Test is touched exactly once, with the validation-frozen choice.
    selected_test = test[test["arm"] == selected["arm"]].reset_index(drop=True)
    baseline_test = test[test["arm"] == BASELINE_ARM].reset_index(drop=True)

    view = ["arm", "entry_k", "exit_k", "total_return", "turnover_multiple", "max_drawdown",
            "sharpe_zero_rate", "filled_trade_count", "hold_selected_count"]
    pd.set_option("display.width", 240)
    print("\n=== validation, weekday-averaged (selection is by rule; see selection_reason) ===")
    print(ranked[[column for column in view if column in ranked.columns]].round(4).to_string(index=False))
    print("\n=== test (all arms, post-hoc) ===")
    print(test.sort_values("total_return", ascending=False)[view].round(4).to_string(index=False))

    detail = ["split", "arm", "entry_k", "exit_k", "total_return", "annual_return", "annual_volatility",
              "sharpe_zero_rate", "max_drawdown", "turnover_multiple", "reported_cost", "filled_trade_count"]
    comparison = pd.concat([baseline_test, selected_test], ignore_index=True)
    print("\n=== test: v1 parity arm vs validation-selected ===")
    print(comparison[detail].round(4).to_string(index=False))

    # Cost decomposition + by-year dose-response.  Probing the whole entry_k=0.10
    # ladder (rather than just the winner) is the point: if the gain lives only at
    # one grid cell, the year-by-year and gross/net splits will show it.
    probe_names = {spec["arm"] for spec in specs if spec["entry_k"] == 0.10} | {str(selected["arm"])}
    probe_specs = [spec for spec in specs if spec["arm"] in probe_names]
    probe_net, probe_equity = evaluate_bands(
        loads["test"], raw_market["test"], probe_specs, args.horizon, args.initial_capital,
        args.max_positions, args.config_path, "test_probe_net", collect_equity=True)
    probe_gross = evaluate_bands(
        loads["test"], raw_market["test"], probe_specs, args.horizon, args.initial_capital,
        args.max_positions, args.config_path, "test_probe_frictionless", frictionless=True)
    probe = probe_net.merge(probe_gross, on=["arm", "entry_k", "exit_k"], suffixes=("_net", "_gross"))
    print("\n=== test cost decomposition (ISO-week cadence, production Monday strike) ===")
    print(probe[["arm", "entry_k", "exit_k", "total_return_gross", "total_return_net", "reported_cost_net",
                 "turnover_multiple_net"]].round(4).to_string(index=False))
    yearly = pd.DataFrame({
        arm: yearly_returns(series, args.initial_capital) for arm, series in probe_equity.items()
    })
    yearly.index.name = "year"
    print("\n=== test net return by calendar year ===")
    print(yearly.round(4).to_string())

    artifacts = {
        "validation_grid": validation,
        "validation_by_weekday": validation_raw,
        "test_grid": test,
        "test_selected": comparison,
        "cost_decomposition": probe,
        "test_by_year": yearly.reset_index(),
        # Each candidate with every gate's verdict, so a "no change" outcome can
        # be audited rather than taken on trust.
        "selection_gates": outcome.table.copy(),
    }
    paths = {}
    for name, table in artifacts.items():
        path = output_dir / "{}_{}.csv".format(args.run_id, name)
        table.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = str(path)

    manifest = {
        "analysis_id": args.run_id,
        "created_at": datetime.now().isoformat(),
        "question": "does a rank hysteresis band (narrow entry, wide exit) cut turnover without losing alpha",
        "model_id": args.model_id,
        "horizon": int(args.horizon),
        "max_positions": int(args.max_positions),
        "cadence": "iso_week, Monday strike on test; validation averaged over the weekday family",
        "validation_weekdays": weekdays,
        "fold_months": int(args.fold_months),
        "turnover_floor_frac": float(args.turnover_floor_frac),
        "arms": [{"arm": spec["arm"], "entry_k": spec["entry_k"], "exit_k": spec["exit_k"]} for spec in specs],
        "selection_objective": "selection_rule.select_configuration (phase average + phase floor + turnover floor + cross-fold sign)",
        "selection_reason": outcome.reason,
        "selection_gates_run": outcome.gates,
        "selected": {"arm": str(selected["arm"]), "entry_k": float(selected["entry_k"]),
                     "exit_k": float(selected["exit_k"])},
        "baseline_arm": BASELINE_ARM,
        "baseline_parity": {
            "validation_total_return": float(ranked.loc[ranked["arm"] == BASELINE_ARM, "total_return"].iloc[0]),
            "documented_v1_validation_total_return": 1.0925,
            "test_total_return": float(baseline_test["total_return"].iloc[0]) if len(baseline_test) else None,
            "documented_v1_test_total_return": 0.4598,
        },
        "note": "test_selected is the single frozen evaluation; the full test grid is inspection only",
        "skipped_prices": {split: len(value) for split, value in skipped.items()},
        "paths": paths,
    }
    manifest_path = output_dir / "{}_manifest.json".format(args.run_id)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\nmanifest={}".format(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
