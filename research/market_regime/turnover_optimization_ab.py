"""Does slowing v1 down pay for itself?

v1 turns over 67x a year and pays 1.85万 of a 50万 book in commission plus stamp
duty -- a 4.38pp drag on a 50.36% frictionless return, i.e. roughly a tenth of the
gross result.  Turnover is set by three things, only one of which v1 tunes:

* the rebalance cadence -- **hard-coded to the ISO week** in
  ``tradability_ranking_evaluation.make_rank_strategy``;
* ``min_holding_days`` -- v1 passes ``horizon`` (10 sessions);
* the exit rule -- a held name is sold only once it drops out of Top-K.

**Why every arm is phase-averaged.**  A first cut compared a session-counted
cadence against the ISO week and found that ``every-5-sessions`` -- nominally the
same weekly rhythm as the baseline -- returned +83.5% on validation against the
baseline's +109.2%.  The two differ only in *which* session of the week the book
is struck.  A 26pp swing from pure phase means that a single draw per arm cannot
tell "slower cadence" apart from "luckier calendar", and selecting the max of a
grid of such draws selects noise.  So each cadence arm is run from several evenly
spaced starting phases and the metrics are averaged; the spread across phases is
reported alongside as the noise floor that any claimed improvement must clear.

The ISO-week baseline has no free phase (the calendar fixes it), so it runs once.

Discipline: the configuration is chosen on **phase-averaged validation net
return** (ties broken by lower turnover), and the test split is run once with that
frozen choice.  The full test grid is also written out, labelled post-hoc.

Parity: the baseline arm calls the *original* ``make_rank_strategy`` unchanged and
reproduces v1 exactly -- validation net +109.25% (documented +109.2%), test net
+45.98%, Sharpe 0.8976, maxDD -13.82%, turnover 67.18x, cost 18504.42.

Example::

    python -m research.market_regime.turnover_optimization_ab
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
from research.market_regime.selection_rule import phase_excess_min, select_configuration
from research.market_regime.tradability_ranking_evaluation import (
    TOP_K_GRID,
    apply_rank_selection,
    build_rank_market_data,
    make_rank_strategy,
    select_top_k,
)
from research.market_regime.trade_outcome_ranking_evaluation import (
    SCORE_COLUMN,
    load_outcome_scores,
    outcome_quality,
)


BASELINE_ARM = "week_iso_hold10"
METRIC_COLUMNS = (
    "total_return", "annual_return", "annual_volatility", "sharpe_zero_rate",
    "max_drawdown", "turnover_multiple", "reported_cost", "filled_trade_count",
)


def phase_offsets(cadence: int, count: int) -> Tuple[int, ...]:
    """``count`` evenly spaced starting phases out of ``cadence`` possible ones."""

    if cadence <= 1:
        return (0,)
    count = max(1, min(count, cadence))
    return tuple(sorted({int(round(i * cadence / count)) % cadence for i in range(count)}))


def make_periodic_rank_strategy(
    max_positions: int, target_notional: float, min_holding_days: int,
    rebalance_days: int, offset: int = 0, lot_size: int = 100,
):
    """``make_rank_strategy`` with a session-counted, phase-shifted cadence.

    The body is intentionally a line-for-line copy of the original entry/exit
    logic; only the "is this a rebalance session?" test changes, so any
    performance difference is attributable to the cadence and phase alone.

    A rebalance fires on session ``i`` when ``(i - offset) % rebalance_days == 0``.
    ``offset=0`` acts on the very first session, matching the original's
    ``current_week = None`` behaviour.
    """

    entry_dates: Dict[str, pd.Timestamp] = {}
    state = {"sessions": 0}

    def strategy(date, relevant_data, positions):
        index = state["sessions"]
        state["sessions"] += 1
        if (index - offset) % rebalance_days != 0:
            return {}

        held = set(positions)
        for symbol in held:
            if symbol in relevant_data:
                entry_dates.setdefault(symbol, pd.Timestamp(date))
        for symbol in list(entry_dates):
            if symbol not in held:
                del entry_dates[symbol]
        signals = {}
        for symbol in sorted(held):
            frame = relevant_data.get(symbol)
            if frame is None or bool(frame.loc[date, "hold_trade"]):
                continue
            entry_date = entry_dates.get(symbol, pd.Timestamp(date))
            entry_position = int(frame.index.searchsorted(entry_date, side="left"))
            days = int(frame.index.get_loc(date)) - entry_position
            if days >= min_holding_days:
                signals[symbol] = {"action": "sell", "shares": positions[symbol], "reason": "tradability_rank_exit"}
        slots = max(max_positions - len(held), 0)
        candidates = []
        for symbol, frame in relevant_data.items():
            if symbol not in held and bool(frame.loc[date, "entry_trade"]):
                candidates.append((symbol, float(frame.loc[date, "rank_score"]), float(frame.loc[date, "close"])))
        for symbol, _, price in sorted(candidates, key=lambda item: (-item[1], item[0]))[:slots]:
            shares = int(np.floor(target_notional / price / lot_size) * lot_size) if price > 0 else 0
            if shares > 0:
                signals[symbol] = {"action": "buy", "shares": shares, "reason": "tradability_rank_entry_t_plus_1"}
        return signals

    return strategy


def make_weekday_rank_strategy(
    max_positions: int, target_notional: float, min_holding_days: int,
    weekday: int = 0, lot_size: int = 100,
):
    """``make_rank_strategy`` with the ISO week struck on a chosen weekday.

    The body is intentionally a line-for-line copy of the original entry/exit
    logic; only the "is this a rebalance session?" test changes, so any
    performance difference is attributable to the weekday alone.

    A session rebalances when it is the **first session of its ISO week whose
    weekday is on or after ``weekday``** (0=Mon .. 4=Fri).  ``weekday=0``
    reproduces the production ISO-week strategy exactly -- the first session of
    an ISO week is always >= Monday and the week guard fires once -- which is the
    parity check this family is built on.

    That "on or after" form keeps the *number* of rebalances comparable across
    targets: an ordinary Mon-Fri week strikes exactly once under any target, and
    only a holiday-shortened week can differ.  A week with no session on or after
    the target (e.g. a Monday-only week under ``weekday=4``) is skipped entirely,
    which is why callers should report the realised rebalance counts.
    """

    entry_dates: Dict[str, pd.Timestamp] = {}
    current_week = None

    def strategy(date, relevant_data, positions):
        nonlocal current_week
        stamp = pd.Timestamp(date)
        iso = stamp.isocalendar()
        week = (int(iso[0]), int(iso[1]))
        if week == current_week or stamp.weekday() < weekday:
            return {}
        current_week = week
        held = set(positions)
        for symbol in held:
            if symbol in relevant_data:
                entry_dates.setdefault(symbol, pd.Timestamp(date))
        for symbol in list(entry_dates):
            if symbol not in held:
                del entry_dates[symbol]
        signals = {}
        for symbol in sorted(held):
            frame = relevant_data.get(symbol)
            if frame is None or bool(frame.loc[date, "hold_trade"]):
                continue
            entry_date = entry_dates.get(symbol, pd.Timestamp(date))
            entry_position = int(frame.index.searchsorted(entry_date, side="left"))
            days = int(frame.index.get_loc(date)) - entry_position
            if days >= min_holding_days:
                signals[symbol] = {"action": "sell", "shares": positions[symbol], "reason": "tradability_rank_exit"}
        slots = max(max_positions - len(held), 0)
        candidates = []
        for symbol, frame in relevant_data.items():
            if symbol not in held and bool(frame.loc[date, "entry_trade"]):
                candidates.append((symbol, float(frame.loc[date, "rank_score"]), float(frame.loc[date, "close"])))
        for symbol, _, price in sorted(candidates, key=lambda item: (-item[1], item[0]))[:slots]:
            shares = int(np.floor(target_notional / price / lot_size) * lot_size) if price > 0 else 0
            if shares > 0:
                signals[symbol] = {"action": "buy", "shares": shares, "reason": "tradability_rank_entry_t_plus_1"}
        return signals

    return strategy


def build_arm_specs(
    min_holding_values: Sequence[int], rebalance_values: Sequence[int], phases: int
) -> List[Dict[str, object]]:
    """Baseline first, then every (hold, cadence) pair with its phase set."""

    specs: List[Dict[str, object]] = [{
        "arm": BASELINE_ARM, "min_holding_days": 10, "rebalance_days": None, "offsets": (0,),
    }]
    for min_holding in min_holding_values:
        for cadence in rebalance_values:
            specs.append({
                "arm": "periodic_hold{}_rebal{}".format(min_holding, cadence),
                "min_holding_days": int(min_holding),
                "rebalance_days": int(cadence),
                "offsets": phase_offsets(int(cadence), phases),
            })
    return specs


def baseline_phase_family(
    scores: pd.DataFrame,
    raw_market_data,
    top_k_grid: Sequence[float],
    horizon: int,
    initial_capital: float,
    max_positions: int,
    config_path: str,
    weekdays: Sequence[int],
) -> Dict[float, List[float]]:
    """The baseline ISO-week arm's own phase family, one set of returns per Top-K.

    Candidates on this grid are phase-averaged over a session-counted cadence;
    the baseline is a *single* calendar draw.  Comparing the two without giving
    the baseline a family of its own is what made the phase gate reject arms
    that beat the baseline on every phase -- see ``selection_rule``.

    These runs are reference material for the gate only.  They are not grid
    candidates: a weekday variant of the baseline is a different clock, and the
    test split always runs the production Monday one.
    """

    start_date, end_date = scores["date"].min(), scores["date"].max()
    target_notional = initial_capital / max_positions
    family: Dict[float, List[float]] = {}
    for top_k in top_k_grid:
        prepared = apply_rank_selection(raw_market_data, scores, select_top_k(scores, top_k, SCORE_COLUMN))
        returns = []
        for weekday in weekdays:
            strategy = make_weekday_rank_strategy(
                max_positions, target_notional, horizon, weekday=int(weekday))
            results, trades = _run_engine(
                prepared, strategy, initial_capital, config_path, start_date, end_date)
            returns.append(float(summarize_backtest(
                "baseline_phase_{}".format(weekday), results, trades, initial_capital)["total_return"]))
        family[float(top_k)] = returns
        print("  baseline phase family Top-K={}: {}".format(
            top_k, " ".join("{:.2%}".format(value) for value in returns)), flush=True)
    return family


def make_strategy(spec: Dict[str, object], offset: int, max_positions: int, target_notional: float):
    if spec["rebalance_days"] is None:
        return make_rank_strategy(max_positions, target_notional, min_holding_days=int(spec["min_holding_days"]))
    return make_periodic_rank_strategy(
        max_positions, target_notional, int(spec["min_holding_days"]), int(spec["rebalance_days"]), offset=offset,
    )


def evaluate_grid(
    scores: pd.DataFrame,
    raw_market_data,
    specs: Sequence[Dict[str, object]],
    top_k_grid: Sequence[float],
    horizon: int,
    initial_capital: float,
    max_positions: int,
    config_path: str,
    label: str,
    frictionless: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run every (arm phase, Top-K) pair over one split.

    Returns ``(averaged, raw)``.  ``apply_rank_selection`` depends only on
    ``top_k``, so Top-K is the *outer* loop: the prepared market data is built
    once per Top-K and dropped before the next one.  Holding all of them at once
    was enough to get the process killed partway through the grid.
    """

    start_date, end_date = scores["date"].min(), scores["date"].max()
    target_notional = initial_capital / max_positions
    draws = sum(len(spec["offsets"]) for spec in specs) * len(top_k_grid)
    started, done = time.time(), 0

    rows = []
    for top_k in top_k_grid:
        selected = select_top_k(scores, top_k, SCORE_COLUMN)
        prepared = apply_rank_selection(raw_market_data, scores, selected)
        for spec in specs:
            for offset in spec["offsets"]:
                strategy = make_strategy(spec, int(offset), max_positions, target_notional)
                results, trades = _run_engine(
                    prepared, strategy, initial_capital, config_path, start_date, end_date,
                    frictionless=frictionless)
                performance = summarize_backtest(str(spec["arm"]), results, trades, initial_capital)
                rows.append({
                    "split": label,
                    "arm": spec["arm"],
                    "min_holding_days": spec["min_holding_days"],
                    "rebalance_days": spec["rebalance_days"] if spec["rebalance_days"] is not None else 0,
                    "offset": int(offset),
                    "top_k": float(top_k),
                    **outcome_quality(scores, selected, horizon),
                    **{key: value for key, value in performance.items() if key != "strategy"},
                })
                done += 1
                if done % 25 == 0:
                    print("  [{}] {}/{} runs, {:.0f}s".format(label, done, draws, time.time() - started), flush=True)
        del prepared
    raw = pd.DataFrame(rows)

    # Phase-average the metrics; keep the across-phase spread as the noise floor.
    keys = ["split", "arm", "min_holding_days", "rebalance_days", "top_k"]
    averaged = raw.groupby(keys, as_index=False)[list(METRIC_COLUMNS)].mean()
    spread = raw.groupby(keys, as_index=False).agg(
        phase_min=("total_return", "min"),
        phase_max=("total_return", "max"),
        phases=("total_return", "size"),
    )
    # selected_* depend on top_k alone, but the merge keeps one row per configuration.
    quality = raw.groupby(keys, as_index=False)[
        ["selected_count", "selected_mean_excess_return", "selected_mean_mae",
         "selected_excess_over_universe"]].mean()
    averaged = averaged.merge(quality, on=keys, how="left").merge(spread, on=keys, how="left")
    return averaged, raw


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="output/datasets")
    parser.add_argument("--model-id", default="hs300_excess_v1_outcome")
    parser.add_argument("--output-dir", default="output/datasets")
    parser.add_argument("--run-id", default="turnover_ab_v1")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--config-path", default="config/settings.yaml")
    parser.add_argument("--min-holding", default="10,20,30")
    parser.add_argument("--rebalance-days", default="5,10,20")
    parser.add_argument("--phases", type=int, default=5, help="starting phases averaged per cadence arm")
    parser.add_argument("--top-k-grid", default=",".join(str(value) for value in TOP_K_GRID))
    parser.add_argument("--weekdays", default="0,1,2,3,4",
                        help="Weekday strikes used to give the ISO-week baseline the same phase "
                             "axis as the candidates, so the phase gate is paired. Pass an empty "
                             "string to skip the gate (it is then reported as not run).")
    parser.add_argument("--skip-test-grid", action="store_true", help="skip the post-hoc full test grid")
    parser.add_argument("--skip-cost-probe", action="store_true",
                        help="skip the frictionless decomposition of the baseline vs the slowest arm")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset_dir, output_dir = Path(args.dataset_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_holding_values = [int(value) for value in args.min_holding.split(",") if value.strip()]
    rebalance_values = [int(value) for value in args.rebalance_days.split(",") if value.strip()]
    top_k_grid = [float(value) for value in args.top_k_grid.split(",") if value.strip()]
    specs = build_arm_specs(min_holding_values, rebalance_values, args.phases)
    draws = sum(len(spec["offsets"]) for spec in specs) * len(top_k_grid)
    print("Arms/phases: {}".format([(spec["arm"], list(spec["offsets"])) for spec in specs]))
    print("Validation engine runs: {}".format(draws))

    loads = {split: load_outcome_scores(dataset_dir, args.model_id, split, args.horizon)
             for split in ("validation", "test")}
    raw_market, skipped = {}, {}
    for split in ("validation", "test"):
        raw_market[split], skipped[split] = build_rank_market_data(
            loads[split], args.price_history_dir, SCORE_COLUMN)

    started = time.time()
    validation_grid, validation_raw = evaluate_grid(
        loads["validation"], raw_market["validation"], specs, top_k_grid, args.horizon,
        args.initial_capital, args.max_positions, args.config_path, "validation",
    )
    print("Validation grid done in {:.1f}s".format(time.time() - started))

    weekdays = [int(value) for value in args.weekdays.split(",") if value.strip()]
    phase_excess = None
    if len(weekdays) > 1:
        family = baseline_phase_family(
            loads["validation"], raw_market["validation"], top_k_grid, args.horizon,
            args.initial_capital, args.max_positions, args.config_path, weekdays)
        phase_excess = pd.DataFrame([
            {"arm": arm, "top_k": float(top_k),
             "phase_excess_min": phase_excess_min(block["total_return"].tolist(), family[float(top_k)])}
            for (arm, top_k), block in validation_raw.groupby(["arm", "top_k"])
        ])

    # Selection is by rule, not by rank order: a candidate has to beat the
    # baseline on the phase average *and* on its worst paired phase, and keep
    # most of the baseline's turnover.  ``sort_values(...).iloc[0]`` used to pick
    # ``periodic_hold30_rebal5`` here -- +93.55% against the baseline's
    # +109.25%, i.e. it always returned something strictly worse.  See
    # ``selection_rule``.
    outcome = select_configuration(
        validation_grid,
        baseline_key={"arm": BASELINE_ARM, "top_k": 0.10},
        key_columns=("arm", "top_k"),
        phase_excess=phase_excess,
    )
    # ``ranked`` stays as the inspection view; it is no longer the decision.
    ranked = outcome.table
    if outcome.row is None:
        print("\nselection: NO CHANGE -- {}".format(outcome.reason))
        ranked_for_spec = validation_grid.sort_values(
            ["total_return", "turnover_multiple", "top_k"],
            ascending=[False, True, True]).reset_index(drop=True)
        selected = ranked_for_spec[ranked_for_spec["arm"] == BASELINE_ARM].iloc[0]
    else:
        print("\nselection: {}".format(outcome.reason))
        selected = outcome.row
    selected_spec = next(spec for spec in specs if spec["arm"] == selected["arm"])
    selected_top_k = float(selected["top_k"])

    view = ["arm", "min_holding_days", "rebalance_days", "top_k", "total_return", "phase_min",
            "phase_max", "turnover_multiple", "max_drawdown", "sharpe_zero_rate"]
    pd.set_option("display.width", 240)
    print("\n=== validation, phase-averaged (selection is by net return, then lower turnover) ===")
    print(ranked.head(12)[view].round(4).to_string(index=False))
    print("\n=== validation at v1's frozen Top-K=0.10 (no re-selection) ===")
    print(ranked[ranked["top_k"] == 0.10][view].round(4).to_string(index=False))

    # Test is touched exactly once, with the validation-frozen choice.
    selected_test, _ = evaluate_grid(
        loads["test"], raw_market["test"], [selected_spec], [selected_top_k], args.horizon,
        args.initial_capital, args.max_positions, args.config_path, "test_selected")
    baseline_test, _ = evaluate_grid(
        loads["test"], raw_market["test"], specs[:1], [0.10], args.horizon,
        args.initial_capital, args.max_positions, args.config_path, "test_baseline")

    comparison = pd.concat([baseline_test, selected_test], ignore_index=True)
    detail = ["split", "arm", "min_holding_days", "rebalance_days", "top_k", "offset", "total_return",
              "annual_return", "annual_volatility", "sharpe_zero_rate", "max_drawdown",
              "turnover_multiple", "reported_cost"]
    print("\n=== test: v1 baseline (ISO week, hold10, Top-K=0.10) vs validation-selected ===")
    print(comparison[[column for column in detail if column in comparison.columns]].round(4).to_string(index=False))

    paths = {"validation_grid": output_dir / "{}_{}.csv".format(args.run_id, "validation_grid"),
             "validation_raw": output_dir / "{}_{}.csv".format(args.run_id, "validation_raw"),
             "test_selected": output_dir / "{}_{}.csv".format(args.run_id, "test_selected"),
             "selection_gates": output_dir / "{}_{}.csv".format(args.run_id, "selection_gates")}
    validation_grid.to_csv(paths["validation_grid"], index=False, encoding="utf-8-sig")
    validation_raw.to_csv(paths["validation_raw"], index=False, encoding="utf-8-sig")
    comparison.to_csv(paths["test_selected"], index=False, encoding="utf-8-sig")
    # Every candidate with each gate's verdict, so a "no change" outcome can be
    # audited rather than taken on trust.
    outcome.table.to_csv(paths["selection_gates"], index=False, encoding="utf-8-sig")

    if not args.skip_test_grid:
        test_grid, test_raw = evaluate_grid(
            loads["test"], raw_market["test"], specs, top_k_grid, args.horizon,
            args.initial_capital, args.max_positions, args.config_path, "test_posthoc")
        paths["test_grid_posthoc"] = output_dir / "{}_{}.csv".format(args.run_id, "test_grid_posthoc")
        paths["test_raw_posthoc"] = output_dir / "{}_{}.csv".format(args.run_id, "test_raw_posthoc")
        test_grid.to_csv(paths["test_grid_posthoc"], index=False, encoding="utf-8-sig")
        test_raw.to_csv(paths["test_raw_posthoc"], index=False, encoding="utf-8-sig")
        print("\n=== test, phase-averaged, post-hoc (not used for selection) ===")
        print(test_grid.sort_values(["total_return"], ascending=False).head(12)[view].round(4).to_string(index=False))

    # Does slowing turnover destroy gross alpha, or merely fail to save enough cost?
    if not args.skip_cost_probe:
        probe_names = {BASELINE_ARM,
                       "periodic_hold{}_rebal{}".format(max(min_holding_values), max(rebalance_values))}
        probe_specs = [spec for spec in specs if spec["arm"] in probe_names]
        config_keys = ["arm", "min_holding_days", "rebalance_days", "top_k"]
        probe_net, _ = evaluate_grid(
            loads["test"], raw_market["test"], probe_specs, [0.10], args.horizon,
            args.initial_capital, args.max_positions, args.config_path, "test_probe_net")
        probe_gross, _ = evaluate_grid(
            loads["test"], raw_market["test"], probe_specs, [0.10], args.horizon,
            args.initial_capital, args.max_positions, args.config_path, "test_probe_frictionless",
            frictionless=True)
        probe = probe_net.merge(probe_gross, on=config_keys, suffixes=("_net", "_gross"))
        probe["gross_alpha_lost"] = probe["total_return_net"] - probe["total_return_gross"]
        print("\n=== test cost decomposition at Top-K=0.10, phase-averaged ===")
        print(probe[["arm", "rebalance_days", "total_return_net", "total_return_gross", "gross_alpha_lost",
                     "turnover_multiple_net", "reported_cost_net"]].round(4).to_string(index=False))
        path = output_dir / "{}_{}.csv".format(args.run_id, "cost_decomposition")
        probe.to_csv(path, index=False, encoding="utf-8-sig")
        paths["cost_decomposition"] = path

    manifest = {
        "analysis_id": args.run_id,
        "created_at": datetime.now().isoformat(),
        "question": "does slowing turnover (holding period / rebalance cadence / Top-K) improve net-of-cost return",
        "model_id": args.model_id,
        "horizon": int(args.horizon),
        "phases_per_arm": int(args.phases),
        "arms": [{"arm": spec["arm"], "min_holding_days": spec["min_holding_days"],
                  "rebalance_days": spec["rebalance_days"], "offsets": list(spec["offsets"])} for spec in specs],
        "top_k_grid": top_k_grid,
        "selection_objective": "selection_rule.select_configuration (phase average + phase floor + turnover floor)",
        "selection_reason": outcome.reason,
        "selection_gates_run": outcome.gates,
        "baseline_phase_weekdays": weekdays,
        "selected": {"arm": str(selected["arm"]), "min_holding_days": int(selected["min_holding_days"]),
                     "rebalance_days": int(selected["rebalance_days"]) or None, "top_k": selected_top_k},
        "baseline_arm": BASELINE_ARM,
        "baseline_parity": {
            "validation_total_return": float(ranked.loc[ranked["arm"] == BASELINE_ARM, "total_return"].iloc[0]),
            "documented_v1_validation_total_return": 1.0925,
        },
        "note": "test_selected is the single frozen evaluation; test_grid_posthoc is inspection only",
        "skipped_prices": {split: len(value) for split, value in skipped.items()},
        "paths": {name: str(path) for name, path in paths.items()},
    }
    manifest_path = output_dir / "{}_manifest.json".format(args.run_id)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\nmanifest={}".format(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
