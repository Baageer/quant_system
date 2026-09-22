"""Run the 2026 forward window through the *published* execution chain.

Nothing here is allowed to re-select anything.  ``top_k``, ``min_holding_days``,
``max_positions`` and ``initial_capital`` are read out of each arm's published
``trade_outcome_ranking_{model_id}_manifest.json`` and echoed into this run's
manifest, so the forward arm is the published arm executed on new dates.  A
forward test whose parameters were chosen after seeing the forward window is not
a forward test.

Two things are worth spelling out because they change the numbers:

**The window is shorter than the calendar suggests.**  ``load_outcome_scores``
drops rows without a complete forward label, and a 10-day label needs 10
sessions after entry.  The engine's ``end_date`` is the last *labelled* score
date, so the traded window ends ~10 sessions before 2026-09-16 -- the same
publication lag that makes the published test end 2025-12-02 rather than
2025-12-31.  Same recipe, same lag, which is what makes the comparison fair.

**The universe is the static pool, PIT-filtered.**  Identical to the published
runs (see ``pit_pool_filter``); the consequence is recorded, not hidden.

Benchmark levels come from ``load_benchmark`` in ``build_forward_2026_samples``,
which merges the historical file with the 2026 ``index_daily`` fetch.  Each pool
is benchmarked against **its own** index (hs300 -> 000300, zz1000 -> 000852),
matching what the published manifests actually record -- that is a measured
fact, not a convention: 000300 gives +6.315% over the published window vs
+2.856% for 000852, and each published manifest reports the one belonging to its
own pool.

Example::

    python -m research.experiments.forward_2026_backtest --pools hs300 --arms frozen
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from research.experiments.build_forward_2026_samples import load_benchmark
from research.experiments.constant_exposure_arm import measure_arm
from research.experiments.random_portfolio_baseline import _score_slots, _write_rank_score
from research.market_regime.data_loader import DEFAULT_PRICE_HISTORY_DIR
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.passive_benchmark import INDEX_ARM_NAME, append_index_arm
from research.market_regime.probability_trade_evaluation import _run_engine, summarize_backtest
from research.market_regime.tradability_ranking_evaluation import (
    apply_rank_selection,
    build_rank_market_data,
    make_rank_strategy,
    select_top_k,
)
from research.market_regime.trade_outcome_ranking_evaluation import (
    SCORE_COLUMN,
    load_outcome_scores,
    published_top_k,
)

FORWARD_ROOT = Path("data/raw/tushare/forward_2026")
FORWARD_PRICE_DIR = FORWARD_ROOT / "price_history"

POOLS = {
    "hs300": {"model_id": "hs300_pit_outcome", "benchmark": "000300"},
    "zz1000": {"model_id": "zz1000_pit_outcome", "benchmark": "000852"},
}
ARMS = {"frozen": "", "boundary": "_boundary", "boundary_tv": "_boundary_tv"}

#: The forward window proper.  The score files also carry Nov-Dec 2025 rows so the
#: parity check has something to compare; those belong to the published test
#: window and must not be traded here.
FORWARD_START = pd.Timestamp("2026-01-01")
HORIZON = 10


def published_execution(dataset_dir: Path, model_id: str) -> Dict[str, object]:
    """The execution config the published run actually used.

    Read rather than hard-coded: the plan assumed both pools froze ``top_k`` at
    0.1, and the manifests say hs300 selected 0.05 against zz1000's 0.1.  A
    forward arm run at the wrong basket width would not be comparable to the
    published one, and the error would look like decay.
    """

    path = dataset_dir / "trade_outcome_ranking_{}_manifest.json".format(model_id)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    execution = manifest["execution"]
    return {
        "top_k": published_top_k(dataset_dir, model_id),
        "min_holding_days": int(execution["min_holding_days"]),
        "max_positions": int(execution["max_positions"]),
        "initial_capital": float(execution["initial_capital"]),
        "source": str(path),
    }


def prepare_scores(source_csv: Path, model_id: str, dataset_dir: Path) -> pd.DataFrame:
    """Stage a forward score file under the canonical name and load it.

    Writing it out rather than loading it directly means the forward arm is read
    by ``load_outcome_scores`` -- the same function the published arm used, with
    the same dtype handling, symbol normalisation and label-completeness filter.
    """

    frame = pd.read_csv(source_csv, dtype={"symbol": str}, low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame = frame.loc[frame["date"] >= FORWARD_START].reset_index(drop=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / "trade_outcome_test_predictions_{}.csv".format(model_id)
    frame.to_csv(target, index=False, encoding="utf-8-sig")
    return load_outcome_scores(dataset_dir, model_id, "test", HORIZON)


def run_arm(
    pool: str, arm: str, scores: pd.DataFrame, prices: Dict[str, pd.DataFrame],
    execution: Dict[str, object], benchmark_levels: pd.Series, config_path: str,
    exposure_draws: int, seed: int,
) -> Dict[str, object]:
    """One arm through the production chain, plus its pinned-exposure replay."""

    top_k = execution["top_k"]
    initial_capital = execution["initial_capital"]
    max_positions = execution["max_positions"]
    target_notional = initial_capital / max_positions

    selected = select_top_k(scores, top_k, SCORE_COLUMN)
    prepared = apply_rank_selection(prices, scores, selected)
    slots = _score_slots(scores, prices)
    _write_rank_score(prepared, np.asarray(scores[SCORE_COLUMN], dtype=float), slots)

    start_date, end_date = scores["date"].min(), scores["date"].max()
    strategy_args = dict(min_holding_days=int(execution["min_holding_days"]))
    net_results, net_trades = _run_engine(
        prepared, make_rank_strategy(max_positions, target_notional, **strategy_args),
        initial_capital, config_path, start_date, end_date)
    free_results, free_trades = _run_engine(
        prepared, make_rank_strategy(max_positions, target_notional, **strategy_args),
        initial_capital, config_path, start_date, end_date, frictionless=True)

    metrics = pd.DataFrame([
        summarize_backtest("trade_outcome_top_k_net", net_results, net_trades, initial_capital),
        summarize_backtest("trade_outcome_top_k_frictionless", free_results, free_trades, initial_capital),
    ])
    metrics = append_index_arm(metrics, net_results, initial_capital, levels=benchmark_levels)

    context = {
        "max_positions": max_positions,
        "target_notional": target_notional,
        "horizon": int(execution["min_holding_days"]),
        "config_path": config_path,
        "start_date": start_date,
        "end_date": end_date,
        "seed": seed,
        "shuffles": 20,
    }
    exposure = measure_arm(prepared, context, initial_capital)

    random_exposure = []
    if exposure_draws > 0:
        rng = np.random.default_rng(seed)
        values = np.asarray(scores[SCORE_COLUMN], dtype=float)
        probe = scores.copy()
        for _ in range(exposure_draws):
            shuffled = rng.permutation(values)
            probe[SCORE_COLUMN] = shuffled
            draw_selected = select_top_k(probe, top_k, SCORE_COLUMN)
            draw_prepared = apply_rank_selection(prices, probe, draw_selected)
            _write_rank_score(draw_prepared, shuffled, slots)
            measured = measure_arm(draw_prepared, context, initial_capital)
            random_exposure.append({
                "w_bar": measured["w_bar"],
                "actual": measured["actual"]["total_return"],
                "pinned_w_bar": measured["pinned_w_bar"]["total_return"],
                "pinned_full": measured["pinned_full"]["total_return"],
                "shuffled_mean": measured["shuffled_mean"],
                "pollution": measured["pollution"],
            })

    # ``summarize_backtest`` labels rows in a ``strategy`` column, and the index
    # row is named by the module constant -- which says "hs300" even when the
    # levels passed in are 000852's.  That wart is the published chain's, so it
    # is read rather than worked around; the manifest records which index this
    # actually is.
    row = metrics.set_index("strategy")
    net = row.loc["trade_outcome_top_k_net"]
    free = row.loc["trade_outcome_top_k_frictionless"]
    idx = row.loc[INDEX_ARM_NAME]
    result = {
        "pool": pool,
        "arm": arm,
        "top_k": top_k,
        "min_holding_days": int(execution["min_holding_days"]),
        "max_positions": max_positions,
        "rows": int(len(scores)),
        "symbols": int(scores["symbol"].nunique()),
        "date_min": str(start_date.date()),
        "date_max": str(end_date.date()),
        "trading_sessions": int(len(net_results)),
        "selected_count": int(np.asarray(selected, dtype=bool).sum()),
        "net_total_return": float(net["total_return"]),
        "net_annual_volatility": float(net["annual_volatility"]),
        "net_sharpe": float(net["sharpe_zero_rate"]),
        "net_max_drawdown": float(net["max_drawdown"]),
        "frictionless_total_return": float(free["total_return"]),
        "cost_drag": float(free["total_return"]) - float(net["total_return"]),
        "index_total_return": float(idx["total_return"]),
        "index_max_drawdown": float(idx["max_drawdown"]),
        "index_sharpe": float(idx["sharpe_zero_rate"]),
        "net_excess_vs_index": float(net["total_return"]) - float(idx["total_return"]),
        "exposure_w_bar": exposure["w_bar"],
        "exposure_pinned_w_bar": float(exposure["pinned_w_bar"]["total_return"]),
        "exposure_pinned_full": float(exposure["pinned_full"]["total_return"]),
        "exposure_pinned_sharpe": float(exposure["pinned_w_bar"]["sharpe_zero_rate"]),
        "exposure_shuffled_mean": exposure["shuffled_mean"],
        "exposure_pollution": exposure["pollution"],
    }
    # Trades are kept so the T+1 signal timing is auditable rather than assumed:
    # the first fill must fall strictly after the first signal date.
    result["first_trade_date"] = (str(pd.Timestamp(net_trades["date"].min()).date())
                                  if not net_trades.empty and "date" in net_trades.columns else None)
    result["signal_end_date"] = str(end_date.date())

    if random_exposure:
        draws = pd.DataFrame(random_exposure)
        result["exposure_random"] = {
            "draws": int(len(draws)),
            "actual_mean": float(draws["actual"].mean()),
            "pinned_w_bar_mean": float(draws["pinned_w_bar"].mean()),
            "pollution_mean": float(draws["pollution"].mean()),
        }
        result["model_minus_random_actual"] = result["net_total_return"] - result["exposure_random"]["actual_mean"]
        result["model_minus_random_pinned"] = (
            result["exposure_pinned_w_bar"] - result["exposure_random"]["pinned_w_bar_mean"])
    return result, metrics, net_results, net_trades


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", default="hs300,zz1000")
    parser.add_argument("--arms", default="frozen,boundary")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--forward-root", default=str(FORWARD_ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config-path", default="config/settings.yaml")
    parser.add_argument("--exposure-draws", type=int, default=20,
                        help="random arms used only to compare exposure pollution")
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    forward_root = Path(args.forward_root)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir
    staging = forward_root / "dataset_2026"
    output_dir.mkdir(parents=True, exist_ok=True)

    price_dirs = [str(DEFAULT_PRICE_HISTORY_DIR), str(forward_root / "price_history")]

    rows, metrics_all = [], {}
    for pool in [p.strip() for p in args.pools.split(",") if p.strip()]:
        model_id = POOLS[pool]["model_id"]
        execution = published_execution(dataset_dir, model_id)
        levels = load_benchmark(pool)
        print("\n=== [{}] {} top_k={} hold={} max_pos={} capital={:.0f} ===".format(
            pool, model_id, execution["top_k"], execution["min_holding_days"],
            execution["max_positions"], execution["initial_capital"]), flush=True)
        print("[{}] benchmark {} .. {} ({} levels)".format(
            pool, levels.index[0].date(), levels.index[-1].date(), len(levels)), flush=True)

        for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
            # ``score_forward_2026`` writes the score tables into the dataset dir,
            # beside every other score table, not into the raw forward tree.
            source = dataset_dir / "forward_2026_scores_{}_{}.csv".format(pool, arm)
            if not source.exists():
                print("[{}] {} missing {}, skipping".format(pool, arm, source), flush=True)
                continue
            arm_model_id = model_id + ARMS[arm]
            scores = prepare_scores(source, arm_model_id, staging)
            if scores.empty:
                print("[{}] {} has no labelled 2026 rows, skipping".format(pool, arm), flush=True)
                continue
            prices, skipped = build_rank_market_data(scores, None, SCORE_COLUMN,
                                                     price_history_dirs=price_dirs)
            print("[{}] {}: {} rows / {} symbols, {}..{}; prices for {} symbols ({} skipped)".format(
                pool, arm, len(scores), scores["symbol"].nunique(),
                scores["date"].min().date(), scores["date"].max().date(),
                len(prices), len(skipped)), flush=True)

            result, metrics, equity, trades = run_arm(
                pool, arm, scores, prices, execution, levels, args.config_path,
                args.exposure_draws, args.seed)
            result["price_symbols_skipped"] = skipped[:30]
            rows.append(result)
            metrics_all["{}_{}".format(pool, arm)] = metrics
            metrics.to_csv(output_dir / "forward_2026_backtest_metrics_{}_{}.csv".format(pool, arm),
                           index=False, encoding="utf-8-sig")
            equity.to_csv(output_dir / "forward_2026_equity_{}_{}.csv".format(pool, arm))
            trades.to_csv(output_dir / "forward_2026_trades_{}_{}.csv".format(pool, arm),
                          index=False, encoding="utf-8-sig")

            print("[{}] {} net {:>8.2%}  index {:>8.2%}  excess {:>8.2%}  maxDD {:>7.2%}  "
                  "cost {:.2%}".format(
                      pool, arm, result["net_total_return"], result["index_total_return"],
                      result["net_excess_vs_index"], result["net_max_drawdown"], result["cost_drag"]),
                  flush=True)
            print("[{}] {} w_bar {:.3f}  pinned(w_bar) {:>8.2%}  pinned(1.0) {:>8.2%}  "
                  "pollution {:>7.2%}".format(
                      pool, arm, result["exposure_w_bar"], result["exposure_pinned_w_bar"],
                      result["exposure_pinned_full"], result["exposure_pollution"]), flush=True)
            if "model_minus_random_pinned" in result:
                print("[{}] {} model-random: actual {:>8.2%}  pinned {:>8.2%}  (random weights {})".format(
                    pool, arm, result["model_minus_random_actual"],
                    result["model_minus_random_pinned"], result["exposure_random"]["draws"]), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "forward_2026_backtest_summary.csv", index=False, encoding="utf-8-sig")
    (forward_root / "backtest_manifest.json").write_text(json.dumps({
        "forward_start": str(FORWARD_START.date()),
        "horizon": HORIZON,
        "price_history_dirs": price_dirs,
        "config_path": args.config_path,
        "seed": args.seed,
        "exposure_draws": args.exposure_draws,
        "results": rows,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n=== forward 2026 summary ===")
    if frame.empty:
        print("(no arms ran; check the score files exist and have labelled 2026 rows)")
        return 2
    keep = ["pool", "arm", "date_min", "date_max", "net_total_return", "index_total_return",
            "net_excess_vs_index", "net_max_drawdown", "net_sharpe", "cost_drag",
            "exposure_w_bar", "exposure_pinned_w_bar", "exposure_pollution"]
    print(frame.loc[:, keep].to_string(index=False))
    print("\nsummary -> {}".format(output_dir / "forward_2026_backtest_summary.csv"))
    print("manifest -> {}".format(forward_root / "backtest_manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
