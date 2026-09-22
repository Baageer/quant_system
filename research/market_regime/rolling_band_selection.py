"""Does *causally* choosing the band width beat not having a band at all?

The static experiment (``hysteresis_band_ab``) left exactly one question open.
On validation the net return rose monotonically with ``exit_k`` -- right up to
the degenerate ``exit_k=1.00`` cell, which buys once and never sells.  A single
argmax over one validation draw picked that cell and lost 35% on test.  That
makes the *selection rule* the suspect, not necessarily the band: one static
argmax on one draw is a noisy estimator, and "highest validation net return" has
no defence against a cell that has quietly become a different strategy.

So this module re-runs the decision the way it would actually have been made.
Walk forward through the test period; at each fold boundary choose ``exit_k``
using only the realized net returns **strictly before that boundary**; trade the
fold with the chosen band; chain the folds into one equity curve.  If the band is
real the chained curve should beat v1's (``exit_k=0.10``, i.e. no band) over the
same window.  If the exit50 spike was luck, the chain wanders around v1 the way
any argmax over noise does.

Entry stays at ``0.10`` -- the open question is the exit band, and widening the
grid in two dimensions at once would only add selection noise.

The rolling rule is reported twice, because the degeneracy is the whole point:

* ``rolling`` -- every candidate is eligible;
* ``rolling_floored`` -- arms whose *trailing* turnover is below
  ``--turnover-floor-frac`` of the strongest trailing turnover in the same
  window are dropped first.

The floor is causal (trailing window only) and purely a candidate filter -- it
never looks at the fold being traded.  At the default 0.25 it excludes only
``exit_k=1.00``, whose turnover is ~2% of v1's.

Reference points on the same folds: ``v1_fixed`` (never band), ``static_*`` (a
single validation argmax held for the whole period, with and without the floor),
``oracle_fixed`` (the best single arm for the whole window, post-hoc) and
``oracle_per_fold`` (the best arm in every single fold, post-hoc).  The two
oracles are upper bounds: if even they cannot beat v1 by much, no selection rule
can.

Example::

    python -m research.market_regime.rolling_band_selection
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.market_regime.hysteresis_band_ab import apply_band_selection, select_rank_band
from research.market_regime.probability_trade_evaluation import _run_engine
# Re-exported: ``fold_boundaries`` is a pure date utility that lived here until
# the selection rule needed it too, and importing it from this module would make
# selection_rule depend on a strategy module for a calendar helper.
from research.market_regime.selection_rule import fold_boundaries, select_configuration  # noqa: F401
from research.market_regime.tradability_ranking_evaluation import (
    build_rank_market_data,
    make_rank_strategy,
)
from research.market_regime.trade_outcome_ranking_evaluation import (
    SCORE_COLUMN,
    load_outcome_scores,
)


ENTRY_K = 0.10
CANDIDATE_EXIT_K = (0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00)
BASELINE_EXIT_K = 0.10
TRADING_DAYS = 252
FLOOR_SENSITIVITY = (0.0, 0.05, 0.25, 0.50)
FLOOR_VARIANT = "rolling_floored"


def arm_name(exit_k: float) -> str:
    return "band_entry{:02d}_exit{:02d}".format(int(round(ENTRY_K * 100)), int(round(exit_k * 100)))


def candidates() -> List[Tuple[float, str]]:
    return [(float(exit_k), arm_name(exit_k)) for exit_k in CANDIDATE_EXIT_K]


def daily_returns(equity: pd.Series, initial_capital: float) -> pd.Series:
    """Daily returns with the first session anchored on the initial capital."""

    marks = pd.concat([pd.Series([initial_capital], index=[equity.index[0] - pd.Timedelta(days=1)]), equity])
    return marks.pct_change().dropna()


def slice_returns(returns: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return returns[(returns.index >= start) & (returns.index < end)]


def window_return(returns: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    block = slice_returns(returns, start, end)
    if block.empty:
        return np.nan
    return float((1.0 + block).prod() - 1.0)


def window_turnover(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                    initial_capital: float) -> float:
    if trades.empty:
        return 0.0
    block = trades[(trades["date"] >= start) & (trades["date"] < end)]
    return float(block["trade_value"].sum() / initial_capital)


def run_arms(
    scores: pd.DataFrame,
    raw_market_data: Dict[str, pd.DataFrame],
    horizon: int,
    initial_capital: float,
    max_positions: int,
    config_path: str,
    label: str,
    frictionless: bool = False,
) -> Dict[str, Dict[str, object]]:
    """One continuous backtest per candidate band, keeping equity and filled trades."""

    start_date, end_date = scores["date"].min(), scores["date"].max()
    target_notional = initial_capital / max_positions
    started = time.time()
    out = {}
    for index, (exit_k, arm) in enumerate(candidates(), start=1):
        entry_selected, hold_selected = select_rank_band(scores, ENTRY_K, exit_k)
        prepared = apply_band_selection(raw_market_data, scores, entry_selected, hold_selected)
        strategy = make_rank_strategy(max_positions, target_notional, min_holding_days=horizon)
        results, trades = _run_engine(
            prepared, strategy, initial_capital, config_path, start_date, end_date,
            frictionless=frictionless)
        del prepared
        filled = trades[trades["status"] == "filled"].copy() if not trades.empty else trades
        if not filled.empty:
            filled["date"] = pd.to_datetime(filled["date"], errors="coerce")
            filled = filled.dropna(subset=["date"])
        out[arm] = {
            "returns": daily_returns(results["portfolio_value"].astype(float), initial_capital),
            "trades": filled,
            "equity": results["portfolio_value"].astype(float),
        }
        print("  [{}] {}/{} {}".format(label, index, len(CANDIDATE_EXIT_K), arm), flush=True)
    print("[{}] {} runs in {:.1f}s".format(label, len(out), time.time() - started), flush=True)
    return out


def build_fold_table(
    returns: Dict[str, pd.Series],
    trades: Dict[str, pd.DataFrame],
    folds: Sequence[Dict[str, pd.Timestamp]],
    panel_start: pd.Timestamp,
    initial_capital: float,
    lookback_months: int,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, str], pd.Series]]:
    """Per-fold trailing statistics and per-fold return blocks, for every arm."""

    rows, blocks = [], {}
    for index, fold in enumerate(folds):
        window_start = (fold["selection_end"] - pd.DateOffset(months=lookback_months)
                        if lookback_months else panel_start)
        row = {"fold": index + 1, "selection_end": fold["selection_end"],
               "trade_start": fold["trade_start"], "trade_end": fold["trade_end"]}
        for _exit_k, arm in candidates():
            row["trail_return_{}".format(arm)] = window_return(returns[arm], window_start, fold["selection_end"])
            row["trail_turnover_{}".format(arm)] = window_turnover(
                trades[arm], window_start, fold["selection_end"], initial_capital)
            block = slice_returns(returns[arm], fold["trade_start"], fold["trade_end"])
            blocks[(index, arm)] = block
            row["fold_return_{}".format(arm)] = (
                float((1.0 + block).prod() - 1.0) if len(block) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows), blocks


def choose_arms(fold_table: pd.DataFrame, arms: Sequence[str], floor_frac: float = 0.0) -> List[str]:
    """The trailing-window argmax per fold: net return first, then lower turnover."""

    chosen = []
    for _index, fold in fold_table.iterrows():
        trailing = {arm: float(fold["trail_return_{}".format(arm)]) for arm in arms}
        turnover = {arm: float(fold["trail_turnover_{}".format(arm)]) for arm in arms}
        eligible = [arm for arm in arms if np.isfinite(trailing[arm])]
        if not eligible:
            eligible = list(arms)
        if floor_frac > 0:
            floor = floor_frac * max(turnover[arm] for arm in eligible)
            floored = [arm for arm in eligible if turnover[arm] >= floor]
            if floored:
                eligible = floored
        chosen.append(max(eligible, key=lambda arm: (trailing[arm], -turnover[arm])))
    return chosen


def chain_blocks(blocks: Dict[Tuple[int, str], pd.Series], chosen: Sequence[str]) -> pd.Series:
    parts = [blocks[(index, arm)] for index, arm in enumerate(chosen)]
    parts = [part for part in parts if len(part)]
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).sort_index()


def chain_metrics(returns: pd.Series, initial_capital: float) -> Dict[str, float]:
    """Same conventions as ``summarize_backtest``, applied to a spliced curve."""

    if returns.empty:
        return {"total_return": np.nan, "annual_return": np.nan, "annual_volatility": np.nan,
                "sharpe_zero_rate": np.nan, "max_drawdown": np.nan, "days": 0}
    equity = (1.0 + returns).cumprod() * initial_capital
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    annual_return = float((1.0 + total_return) ** (TRADING_DAYS / len(returns)) - 1.0)
    volatility = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe_zero_rate": annual_return / volatility if volatility > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "days": int(len(returns)),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="output/datasets")
    parser.add_argument("--model-id", default="hs300_excess_v1_outcome")
    parser.add_argument("--output-dir", default="output/datasets")
    parser.add_argument("--run-id", default="rolling_band_v1")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--config-path", default="config/settings.yaml")
    parser.add_argument("--step-months", type=int, default=6)
    parser.add_argument("--lookback-months", type=int, default=0,
                        help="cap the trailing window; 0 keeps all available history")
    parser.add_argument("--turnover-floor-frac", type=float, default=0.25,
                        help="candidate arms need trailing turnover >= frac * the trailing maximum")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset_dir, output_dir = Path(args.dataset_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loads = {split: load_outcome_scores(dataset_dir, args.model_id, split, args.horizon)
             for split in ("validation", "test")}
    raw_market, skipped = {}, {}
    for split in ("validation", "test"):
        raw_market[split], skipped[split] = build_rank_market_data(
            loads[split], args.price_history_dir, SCORE_COLUMN)

    net, gross = {}, {}
    for split in ("validation", "test"):
        net[split] = run_arms(loads[split], raw_market[split], args.horizon, args.initial_capital,
                              args.max_positions, args.config_path, "{}_net".format(split))
        gross[split] = run_arms(loads[split], raw_market[split], args.horizon, args.initial_capital,
                                args.max_positions, args.config_path, "{}_gross".format(split),
                                frictionless=True)

    arms = [arm for _exit_k, arm in candidates()]
    returns = {arm: pd.concat([net["validation"][arm]["returns"], net["test"][arm]["returns"]]).sort_index()
               for arm in arms}
    gross_returns = {arm: pd.concat([gross["validation"][arm]["returns"], gross["test"][arm]["returns"]]).sort_index()
                     for arm in arms}
    trades = {arm: pd.concat([net["validation"][arm]["trades"], net["test"][arm]["trades"]], ignore_index=True)
              for arm in arms}

    val_start = returns[arms[0]].index.min()
    val_end = loads["validation"]["date"].max() + pd.Timedelta(days=1)
    validation_summary = pd.DataFrame([
        {"arm": arm, "exit_k": exit_k,
         "total_return": window_return(returns[arm], val_start, val_end),
         "turnover_multiple": window_turnover(trades[arm], val_start, val_end, args.initial_capital),
         "max_drawdown": float((net["validation"][arm]["equity"] /
                                net["validation"][arm]["equity"].cummax() - 1.0).min())}
        for exit_k, arm in candidates()])
    ranked_validation = validation_summary.sort_values(
        ["total_return", "turnover_multiple"], ascending=[False, True]).reset_index(drop=True)
    static_arm = str(ranked_validation.iloc[0]["arm"])
    val_floor = args.turnover_floor_frac * float(validation_summary["turnover_multiple"].max())
    floored_validation = validation_summary[
        validation_summary["turnover_multiple"] >= val_floor].sort_values(
        ["total_return", "turnover_multiple"], ascending=[False, True]).reset_index(drop=True)
    static_floored_arm = str(floored_validation.iloc[0]["arm"])

    # The two picks above are kept as *historical comparators*: they are what the
    # rank-the-grid selector does, and the module's point is partly to show where
    # it lands.  ``static_rule`` is the same static decision taken by the shared
    # rule instead, so the comparison table carries both.
    #
    # No phase gate (these arms have no phase axis) and no fold gate (the folds
    # are cut over the *test* period, so feeding them here would leak test).  The
    # outcome says both gates did not run rather than implying it cleared them.
    rule_outcome = select_configuration(
        validation_summary,
        baseline_key={"arm": arm_name(BASELINE_EXIT_K)},
        key_columns=("arm",),
        turnover_floor_frac=args.turnover_floor_frac,
    )
    static_rule_arm = (str(rule_outcome.row["arm"]) if rule_outcome.row is not None
                       else arm_name(BASELINE_EXIT_K))

    print("\n=== validation per arm (parity check vs hysteresis_band_ab) ===")
    print(validation_summary.round(4).to_string(index=False))
    print("static (no floor) -> {}   static (floor frac {}) -> {}".format(
        static_arm, args.turnover_floor_frac, static_floored_arm))
    print("static (rule)     -> {}   [{}]".format(static_rule_arm, rule_outcome.reason))

    # Walk forward over the test period; the trailing window is everything before
    # the fold boundary, i.e. validation plus the folds already traded.
    test_start = loads["test"]["date"].min()
    test_end = loads["test"]["date"].max()
    folds = fold_boundaries(test_start, test_end, args.step_months)
    fold_table, blocks = build_fold_table(returns, trades, folds, val_start,
                                          args.initial_capital, args.lookback_months)
    print("\n=== folds ({} x {} months, selection window ends at each boundary) ===".format(
        len(folds), args.step_months))

    chosen = {"v1_fixed": [arm_name(BASELINE_EXIT_K)] * len(folds),
              "static_selected": [static_arm] * len(folds),
              "static_floored": [static_floored_arm] * len(folds),
              "static_rule": [static_rule_arm] * len(folds),
              "rolling": choose_arms(fold_table, arms, 0.0),
              FLOOR_VARIANT: choose_arms(fold_table, arms, args.turnover_floor_frac)}
    # Post-hoc upper bounds: the best single arm over the whole window, and the
    # best arm in each fold.  Neither is implementable; both bound any rule.
    window_totals = {arm: float((1.0 + slice_returns(returns[arm], test_start,
                                                      test_end + pd.Timedelta(days=1))).prod() - 1.0)
                     for arm in arms}
    chosen["oracle_fixed"] = [max(arms, key=lambda arm: window_totals[arm])] * len(folds)
    chosen["oracle_per_fold"] = [max(arms, key=lambda arm: fold_table.loc[index, "fold_return_{}".format(arm)])
                                 for index in fold_table.index]
    # Floor sensitivity: the whole path is recomputed for each candidate floor.
    for frac in FLOOR_SENSITIVITY:
        chosen["rolling_floor{:03d}".format(int(round(frac * 100)))] = choose_arms(fold_table, arms, frac)

    gross_blocks = {(index, arm): slice_returns(gross_returns[arm], fold["trade_start"], fold["trade_end"])
                    for index, fold in enumerate(folds) for _exit_k, arm in candidates()}

    rows, equity_rows = [], []
    for variant, arms_chosen in chosen.items():
        chain = chain_blocks(blocks, arms_chosen)
        gross_chain = chain_blocks(gross_blocks, arms_chosen)
        stats = chain_metrics(chain, args.initial_capital)
        rows.append({
            "variant": variant,
            "distinct_arms": len(set(arms_chosen)),
            "switches": sum(1 for left, right in zip(arms_chosen, arms_chosen[1:]) if left != right),
            **stats,
            "gross_total_return": float((1.0 + gross_chain).prod() - 1.0) if len(gross_chain) else np.nan,
        })
        equity = (1.0 + chain).cumprod() * args.initial_capital
        equity_rows.extend({"variant": variant, "date": date, "equity": float(value)}
                           for date, value in equity.items())
    variants = pd.DataFrame(rows)

    choices = pd.DataFrame({"fold": fold_table["fold"], "trade_start": fold_table["trade_start"],
                            "trade_end": fold_table["trade_end"],
                            **{variant: arms_chosen for variant, arms_chosen in chosen.items()}})

    view = ["variant", "distinct_arms", "switches", "total_return", "annual_return",
            "annual_volatility", "sharpe_zero_rate", "max_drawdown", "gross_total_return"]
    pd.set_option("display.width", 240)
    print("\n=== chained test-period performance (net unless marked gross) ===")
    print(variants.sort_values("total_return", ascending=False)[view].round(4).to_string(index=False))
    print("\n=== chosen band per fold ===")
    print(choices.to_string(index=False))

    parity = variants[variants["variant"] == "v1_fixed"].iloc[0]
    # Risk figures are the post-A1 ones: the halt-valuation fix (2026-09-16) moved
    # v1 test Sharpe 0.8976 -> 0.9324 and maxDD -0.1382 -> -0.1096.  Return is
    # untouched by that fix.
    print("\nparity: v1_fixed chain total_return={:.4f} sharpe={:.4f} maxDD={:.4f} "
          "(documented v1, post-A1: 0.4598 / 0.9324 / -0.1096)".format(
              parity["total_return"], parity["sharpe_zero_rate"], parity["max_drawdown"]))

    artifacts = {"validation_summary": validation_summary, "folds": fold_table,
                 "variants": variants, "choices": choices, "equity": pd.DataFrame(equity_rows)}
    paths = {}
    for name, table in artifacts.items():
        path = output_dir / "{}_{}.csv".format(args.run_id, name)
        table.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = str(path)

    manifest = {
        "analysis_id": args.run_id,
        "created_at": datetime.now().isoformat(),
        "question": "does causally rolling the exit band beat not having a band",
        "model_id": args.model_id,
        "horizon": int(args.horizon),
        "max_positions": int(args.max_positions),
        "entry_k": ENTRY_K,
        "candidate_exit_k": list(CANDIDATE_EXIT_K),
        "step_months": int(args.step_months),
        "lookback_months": int(args.lookback_months),
        "turnover_floor_frac": float(args.turnover_floor_frac),
        "test_window": {"start": str(test_start.date()), "end": str(test_end.date()), "folds": len(folds)},
        "selection_rule": "trailing_window_net_return_then_lower_turnover",
        "static_selected_arm": static_arm,
        "static_floored_arm": static_floored_arm,
        "static_rule_arm": static_rule_arm,
        "static_rule_reason": rule_outcome.reason,
        "static_rule_gates": rule_outcome.gates,
        "chosen": {variant: list(arms_chosen) for variant, arms_chosen in chosen.items()},
        "baseline_arm": arm_name(BASELINE_EXIT_K),
        "baseline_parity": {
            "chain_total_return": float(parity["total_return"]),
            "chain_sharpe_zero_rate": float(parity["sharpe_zero_rate"]),
            "chain_max_drawdown": float(parity["max_drawdown"]),
            "documented_v1_test_total_return": 0.4598,
            "documented_v1_test_sharpe_zero_rate": 0.9324,
            "documented_v1_test_max_drawdown": -0.1096,
            "documented_values_note": "post-A1 (halt-valuation fix, 2026-09-16); return unchanged by it",
        },
        "note": ("oracle_fixed and oracle_per_fold are post-hoc upper bounds and are not "
                 "implementable; every other variant picks its arm from trailing data only"),
        "skipped_prices": {split: len(value) for split, value in skipped.items()},
        "paths": paths,
    }
    manifest_path = output_dir / "{}_manifest.json".format(args.run_id)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("manifest={}".format(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
