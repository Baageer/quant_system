"""Backtest the direct T+1 outcome model's daily Top-K ``entry_score``."""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from research.market_regime.passive_benchmark import (
    DEFAULT_BENCHMARK_FILE,
    append_index_arm,
    benchmark_comparison,
)
from research.market_regime.probability_trade_evaluation import (
    _run_engine,
    make_passive_strategy,
    select_passive_symbols,
    summarize_backtest,
)
from research.market_regime.tradability_labels import normalize_a_share_symbols
from research.market_regime.tradability_ranking_evaluation import (
    TOP_K_GRID,
    apply_rank_selection,
    build_rank_market_data,
    make_rank_strategy,
    select_top_k,
    select_top_k_on_validation,
)
from research.market_regime.trade_outcome_model import target_columns
from research.market_regime.output_paths import DATASET_DIR


SCORE_COLUMN = "entry_score"

#: The basket width each model is currently published with -- i.e. what is on
#: disk today.  The selection rule treats that incumbent as the bar a candidate
#: has to beat, so a re-run has to *state* it rather than rediscover it:
#: rediscovering it from the grid is exactly the ``sort_values(...).iloc[0]``
#: behaviour being removed, and reading it back out of this module's previous
#: output would fold the last run's result into the next one.
#:
#: Transcribed from the ``trade_outcome_ranking_*_manifest.json`` files in the
#: dataset directory on 2026-09-18.  Every value here was picked by the retired
#: rule, so this is a statement of fact about the incumbent, not a claim that
#: the incumbent is right.
#:
#: Re-checked against all eight published grids after wiring the rule in: every
#: one returns the recorded width unchanged, so no downstream artifact needs
#: regenerating.  Read that as a no-op, not as a verdict -- each published value
#: *is* its grid's argmax by construction, and gate 0 requires beating the
#: incumbent outright, so on exactly these grids the rule cannot move and the two
#: rules cannot disagree.  The check confirms the wiring runs on real grids; it
#: says nothing about whether the widths are good, because these grids cannot
#: ask.  The rule's value is prospective -- on a fresh grid whose incumbent is
#: not the argmax it can refuse a degenerate narrower basket, which is the case
#: the published grids never present.
PUBLISHED_TOP_K = {
    "hs300_ewbaseline_outcome": 0.10,
    "hs300_excess_short_outcome": 0.10,
    "hs300_excess_v1_outcome": 0.10,
    "hs300_pit_outcome": 0.05,
    "hs300_trade_outcome_v1": 0.10,
    "v1_robustness_walkforward": 0.05,
    "zz1000_excess_v1_outcome": 0.10,
    "zz1000_pit_outcome": 0.10,
}


def resolve_baseline_top_k(model_id: str, override: Optional[float] = None) -> float:
    """Return the incumbent basket width that a re-run has to beat.

    Fails loudly on an unknown model id rather than guessing.  A silent default
    would put a new pool's candidates in front of some other pool's bar, and the
    rule would then report "no change" -- which reads as *evidence the incumbent
    is sound* when in fact nothing was tested against it.
    """

    if override is not None:
        return float(override)
    if model_id not in PUBLISHED_TOP_K:
        raise ValueError(
            "No published top_k on record for model_id={!r}; pass baseline_top_k "
            "explicitly so the incumbent bar is stated rather than guessed. "
            "Recorded ids: {}".format(model_id, sorted(PUBLISHED_TOP_K))
        )
    return float(PUBLISHED_TOP_K[model_id])


def published_top_k(dataset_dir: Path, model_id: str) -> float:
    """The basket width a *published* run actually used, read from its manifest.

    This is the number a **replication** has to use -- a random null arm, a
    forward arm -- because the two sides are only like-for-like at the same
    width, and a mismatch does not announce itself: it arrives as apparent
    alpha, or as apparent decay.

    Read here rather than through :func:`resolve_baseline_top_k`, and note that
    the split is deliberate.  The selector must *not* consult this function:
    it is deciding whether to move off the incumbent, and folding its own last
    answer back into the next run is how a selector starts ratcheting.  A
    replica has the opposite job -- reproduce the published configuration
    exactly -- so for it the manifest is the correct source of truth.

    Raises rather than defaulting, and names the file it wanted.
    """

    path = dataset_dir / "trade_outcome_ranking_{}_manifest.json".format(model_id)
    if not path.exists():
        raise ValueError(
            "No published ranking manifest for model_id={!r} at {}; the basket "
            "width cannot be inferred, and a replica drawn at a guessed width is "
            "not comparable to the published arm.".format(model_id, path)
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return float(manifest["selected_top_k_validation_only"])


def load_outcome_scores(dataset_dir: Path, model_id: str, split: str, horizon: int) -> pd.DataFrame:
    """Load a direct outcome score table exported by ``trade_outcome_model``."""

    path = dataset_dir / "trade_outcome_{}_predictions_{}.csv".format(split, model_id)
    excess_column, mae_column = target_columns(horizon)
    frame = pd.read_csv(path, dtype={"symbol": str})
    required = {"date", "symbol", SCORE_COLUMN, excess_column, mae_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Outcome score file missing columns: {}".format(sorted(missing)))
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = normalize_a_share_symbols(frame["symbol"])
    frame[SCORE_COLUMN] = pd.to_numeric(frame[SCORE_COLUMN], errors="coerce")
    frame[excess_column] = pd.to_numeric(frame[excess_column], errors="coerce")
    frame[mae_column] = pd.to_numeric(frame[mae_column], errors="coerce")
    return frame.dropna(subset=["date", "symbol", SCORE_COLUMN, excess_column, mae_column]).sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)


def outcome_quality(scores: pd.DataFrame, selected: pd.Series, horizon: int) -> Dict[str, float]:
    """Report forward outcomes for an equal-weight daily selected basket."""

    excess_column, mae_column = target_columns(horizon)
    mask = np.asarray(selected, dtype=bool)
    selected_excess = np.asarray(scores.loc[mask, excess_column], dtype=float)
    selected_mae = np.asarray(scores.loc[mask, mae_column], dtype=float)
    universe_excess = np.asarray(scores[excess_column], dtype=float)
    return {
        "selected_count": int(mask.sum()),
        "selected_mean_excess_return": float(np.mean(selected_excess)),
        "selected_mean_mae": float(np.mean(selected_mae)),
        "selected_excess_over_universe": float(np.mean(selected_excess) - np.mean(universe_excess)),
    }


def evaluate_outcome_validation_grid(
    validation_scores: pd.DataFrame,
    raw_market_data: Dict[str, pd.DataFrame],
    horizon: int,
    initial_capital: float,
    max_positions: int,
    config_path: str,
) -> pd.DataFrame:
    """Evaluate every grid width on validation, sorted best-return-first.

    The sort is a *view* for reading the table and for the CSV on disk -- it is
    not a selection.  Picking ``iloc[0]`` here would always move off the
    incumbent, because a narrower basket mechanically raises both return and
    turnover, so the grid argmax is a construction artefact rather than evidence
    that the incumbent is wrong.  Selection is
    :func:`tradability_ranking_evaluation.select_top_k_on_validation`, which
    requires the incumbent to be beaten outright on every gate.
    """

    start_date, end_date = validation_scores["date"].min(), validation_scores["date"].max()
    target_notional = initial_capital / max_positions
    rows = []
    for top_k in TOP_K_GRID:
        selected = select_top_k(validation_scores, top_k, SCORE_COLUMN)
        prepared = apply_rank_selection(raw_market_data, validation_scores, selected)
        results, trades = _run_engine(
            prepared,
            make_rank_strategy(max_positions, target_notional, min_holding_days=horizon),
            initial_capital,
            config_path,
            start_date,
            end_date,
        )
        performance = summarize_backtest("top_k", results, trades, initial_capital)
        rows.append(
            {
                "top_k": top_k,
                **outcome_quality(validation_scores, selected, horizon),
                **{"validation_" + key: value for key, value in performance.items() if key != "strategy"},
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["validation_total_return", "validation_turnover_multiple", "top_k"], ascending=[False, True, True]
    ).reset_index(drop=True)


def run_outcome_ranking_evaluation(
    dataset_dir: Path,
    model_id: str,
    output_dir: Path,
    horizon: int = 10,
    initial_capital: float = 500000.0,
    max_positions: int = 20,
    price_history_dir: Optional[str] = None,
    config_path: str = "config/settings.yaml",
    benchmark_file: str = DEFAULT_BENCHMARK_FILE,
    baseline_top_k: Optional[float] = None,
):
    """Select a validation Top-K and evaluate its direct score on the test split.

    ``baseline_top_k`` is the incumbent width the candidates must beat; it
    defaults to the published value recorded in :data:`PUBLISHED_TOP_K` and may
    be overridden explicitly.  An unknown ``model_id`` with no override raises
    rather than guessing.
    """

    validation = load_outcome_scores(dataset_dir, model_id, "validation", horizon)
    test = load_outcome_scores(dataset_dir, model_id, "test", horizon)
    validation_raw, validation_skipped = build_rank_market_data(validation, price_history_dir, SCORE_COLUMN)
    if not validation_raw:
        raise ValueError("No validation price data")
    validation_grid = evaluate_outcome_validation_grid(
        validation, validation_raw, horizon, initial_capital, max_positions, config_path
    )
    incumbent_top_k = resolve_baseline_top_k(model_id, baseline_top_k)
    print("\nbaseline top_k (incumbent to beat): {}".format(incumbent_top_k))
    selected_top_k = select_top_k_on_validation(validation_grid, incumbent_top_k)
    test_selected = select_top_k(test, selected_top_k, SCORE_COLUMN)
    test_raw, test_skipped = build_rank_market_data(test, price_history_dir, SCORE_COLUMN)
    prepared = apply_rank_selection(test_raw, test, test_selected)
    start_date, end_date = test["date"].min(), test["date"].max()
    target_notional = initial_capital / max_positions
    net_results, net_trades = _run_engine(
        prepared,
        make_rank_strategy(max_positions, target_notional, min_holding_days=horizon),
        initial_capital,
        config_path,
        start_date,
        end_date,
    )
    free_results, free_trades = _run_engine(
        prepared,
        make_rank_strategy(max_positions, target_notional, min_holding_days=horizon),
        initial_capital,
        config_path,
        start_date,
        end_date,
        frictionless=True,
    )
    passive_symbols = select_passive_symbols(prepared, max_positions)
    passive_results, passive_trades = _run_engine(
        prepared, make_passive_strategy(passive_symbols, target_notional), initial_capital, config_path, start_date, end_date
    )
    metrics = pd.DataFrame(
        [
            summarize_backtest("trade_outcome_top_k_net", net_results, net_trades, initial_capital),
            summarize_backtest("trade_outcome_top_k_frictionless", free_results, free_trades, initial_capital),
            summarize_backtest("passive_equal_weight_sample_net", passive_results, passive_trades, initial_capital),
        ]
    )
    metrics = append_index_arm(metrics, net_results, initial_capital, benchmark_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "trade_outcome_ranking_{}".format(model_id)
    validation_grid.to_csv(output_dir / (prefix + "_validation_grid.csv"), index=False)
    test.assign(selected_top_k=np.asarray(test_selected, dtype=bool)).to_csv(
        output_dir / (prefix + "_test_scores.csv"), index=False
    )
    metrics.to_csv(output_dir / (prefix + "_backtest_metrics.csv"), index=False)
    net_results.to_csv(output_dir / (prefix + "_equity_net.csv"))
    free_results.to_csv(output_dir / (prefix + "_equity_frictionless.csv"))
    net_trades.to_csv(output_dir / (prefix + "_trades_net.csv"), index=False)
    manifest = {
        "model_id": model_id,
        "horizon": int(horizon),
        "score_column": SCORE_COLUMN,
        "selected_top_k_validation_only": selected_top_k,
        "baseline_top_k": incumbent_top_k,
        "selection_objective": "selection_rule.select_configuration vs top_k={}".format(incumbent_top_k),
        "test_forward_outcome_quality": outcome_quality(test, test_selected, horizon),
        "test_return_vs_benchmarks": benchmark_comparison(metrics),
        "execution": {
            "signal_timing": "EOD score at t, order at next session t+1",
            "min_holding_days": int(horizon),
            "rebalance": "weekly",
            "max_positions": max_positions,
            "initial_capital": initial_capital,
            "validation_skipped_prices": validation_skipped,
            "test_skipped_prices": test_skipped,
        },
    }
    manifest_path = output_dir / (prefix + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--model-id", default="hs300_trade_outcome_v1")
    parser.add_argument("--output-dir", default=DATASET_DIR)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--config-path", default="config/settings.yaml")
    parser.add_argument("--benchmark-file", default=DEFAULT_BENCHMARK_FILE)
    parser.add_argument(
        "--baseline-top-k",
        type=float,
        default=None,
        help="Incumbent basket width to beat; defaults to the published value for --model-id",
    )
    args = parser.parse_args(argv)
    manifest = run_outcome_ranking_evaluation(
        Path(args.dataset_dir), args.model_id, Path(args.output_dir), args.horizon, args.initial_capital,
        args.max_positions, args.price_history_dir, args.config_path, args.benchmark_file,
        args.baseline_top_k,
    )
    print("Selected validation Top-K={:.2%}".format(manifest["selected_top_k_validation_only"]))


if __name__ == "__main__":
    main()
