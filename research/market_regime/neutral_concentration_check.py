"""Is v1's neutral-regime concentration a real structure, or a test-period accident?

``docs/research/04_validation_and_diagnostics/v1_robustness_validation_summary.md`` §4 observed, on the **test** split only,
that v1's edge sits almost entirely in ``neutral`` market states (rolling arm:
neutral Sharpe 2.22 vs risk_off -0.04, risk_on 0.30) -- and flagged it as a
post-hoc pattern that had never been confirmed out of sample.

Two confirmations are run here, because "concentrated in neutral" has two very
different possible causes:

1. **Out-of-sample replication.**  The production chain never backtests the
   validation split (``run_outcome_ranking_evaluation`` uses it only to pick
   Top-K), so the validation window 2019-07 -> 2021-12 is an untraded sample of
   the same model.  If the same shape appears there, the pattern is not an
   artifact of the test window.
2. **Like-for-like null.**  A regime tilt can belong to the *market* rather than
   to the *model*: if randomly-scored portfolios also earn more in ``neutral``,
   then "v1's edge is concentrated in neutral" is just "neutral days are easy
   days", and carries no information about the model at all.  So the real arm is
   placed against N shuffled-score draws on the same split.

Example::

    python -m research.market_regime.neutral_concentration_check --draws 25
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from research.experiments.random_portfolio_baseline import _score_slots, _write_rank_score
from research.market_regime.market_environment import build_causal_market_features, load_benchmark_history
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.probability_trade_evaluation import _run_engine
from research.market_regime.tradability_ranking_evaluation import (
    apply_rank_selection,
    build_rank_market_data,
    make_rank_strategy,
    select_top_k,
)
from research.market_regime.trade_outcome_ranking_evaluation import SCORE_COLUMN, load_outcome_scores
from research.market_regime.v1_robustness_validation import _period_stats

INITIAL_CAPITAL = 500000.0
MAX_POSITIONS = 20


def state_map(benchmark_file: str) -> pd.Series:
    benchmark = load_benchmark_history(benchmark_file)
    market = build_causal_market_features(benchmark)[["market_state"]].reset_index()
    market.columns = ["date", "market_state"]
    return market.set_index("date")["market_state"]


def arm_states(
    scores: pd.DataFrame,
    raw_market_data: Dict[str, pd.DataFrame],
    slots: Dict[str, np.ndarray],
    values: np.ndarray,
    states: pd.Series,
    top_k: float,
    horizon: int,
    config_path: str,
) -> pd.DataFrame:
    """Backtest one score vector and split its net daily P&L by market state."""

    probe = scores.copy()
    probe[SCORE_COLUMN] = values
    selected = select_top_k(probe, top_k, SCORE_COLUMN)
    prepared = apply_rank_selection(raw_market_data, probe, selected)
    _write_rank_score(prepared, np.asarray(values, dtype=float), slots)

    results, _ = _run_engine(
        prepared,
        make_rank_strategy(MAX_POSITIONS, INITIAL_CAPITAL / MAX_POSITIONS, min_holding_days=horizon),
        INITIAL_CAPITAL,
        config_path,
        scores["date"].min(),
        scores["date"].max(),
    )
    equity = results.copy()
    if "date" not in equity.columns:
        equity = equity.reset_index()
    equity = equity[["date", "portfolio_value"]].copy()
    equity["date"] = pd.to_datetime(equity["date"])
    equity = equity.sort_values("date").reset_index(drop=True)
    equity["daily_return"] = equity["portfolio_value"].pct_change().fillna(0.0)
    equity["market_state"] = equity["date"].map(states)
    equity = equity.dropna(subset=["market_state"])

    rows = []
    for state, group in equity.groupby("market_state", sort=True):
        stats = _period_stats(np.asarray(group["daily_return"], dtype=float))
        rows.append({"market_state": state, **stats})
    return pd.DataFrame(rows)


def concentration(frame: pd.DataFrame) -> Dict[str, float]:
    """Scalars that say *how* concentrated in neutral a run was."""

    by_state = frame.set_index("market_state")
    neutral = float(by_state.loc["neutral", "total_return"])
    others = [float(by_state.loc[s, "total_return"]) for s in ("risk_off", "risk_on") if s in by_state.index]
    sharpe = {s: float(by_state.loc[s, "sharpe_zero_rate"]) for s in by_state.index}
    return {
        "neutral_total": neutral,
        "risk_off_total": float(by_state.loc["risk_off", "total_return"]) if "risk_off" in by_state.index else np.nan,
        "risk_on_total": float(by_state.loc["risk_on", "total_return"]) if "risk_on" in by_state.index else np.nan,
        "neutral_minus_other_mean": neutral - float(np.mean(others)) if others else np.nan,
        "neutral_sharpe": sharpe.get("neutral", np.nan),
        "other_sharpe_mean": float(np.mean([v for k, v in sharpe.items() if k != "neutral"]))
        if len(sharpe) > 1 else np.nan,
    }


def run(
    dataset_dir: Path,
    model_id: str,
    splits: Sequence[str] = ("validation", "test"),
    draws: int = 25,
    top_k: float = 0.1,
    horizon: int = 10,
    benchmark_file: str = "data/benchmarks/000300.csv",
    config_path: str = "config/settings.yaml",
    seed: int = 20260915,
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    states = state_map(benchmark_file)
    rng = np.random.default_rng(seed)
    report: Dict[str, object] = {}

    for split in splits:
        scores = load_outcome_scores(dataset_dir, model_id, split, horizon)
        raw_market_data, _ = build_rank_market_data(scores, None, SCORE_COLUMN)
        slots = _score_slots(scores, raw_market_data)
        real_values = np.asarray(scores[SCORE_COLUMN], dtype=float)

        real_frame = arm_states(scores, raw_market_data, slots, real_values, states, top_k, horizon, config_path)
        real_scalars = concentration(real_frame)

        null_rows: List[Dict[str, float]] = []
        for draw in range(draws):
            shuffled = rng.permutation(real_values)
            frame = arm_states(scores, raw_market_data, slots, shuffled, states, top_k, horizon, config_path)
            null_rows.append({"draw": draw, **concentration(frame)})
        null = pd.DataFrame(null_rows)

        percentile = float((null["neutral_minus_other_mean"] < real_scalars["neutral_minus_other_mean"]).mean())
        report[split] = {
            "real_by_state": real_frame,
            "real_scalars": real_scalars,
            "null": null,
            "percentile_vs_null": percentile,
            "window": [str(pd.Timestamp(scores["date"].min()).date()), str(pd.Timestamp(scores["date"].max()).date())],
        }

        print("\n===== split: {}   window {} -> {} =====".format(
            split, *report[split]["window"]))
        print("--- REAL v1 net P&L by market_state ---")
        print(real_frame.round(4).to_string(index=False))
        print("--- shuffled-score null ({} draws), same statistic ---".format(draws))
        print("neutral - mean(other two) total return:")
        print("    real v1      {:>9.2%}".format(real_scalars["neutral_minus_other_mean"]))
        print("    null mean    {:>9.2%}   sd {:>7.2%}".format(
            float(null["neutral_minus_other_mean"].mean()), float(null["neutral_minus_other_mean"].std())))
        print("    null range   {:>9.2%} .. {:>9.2%}".format(
            float(null["neutral_minus_other_mean"].min()), float(null["neutral_minus_other_mean"].max())))
        print("    v1 percentile vs null: {:.1%}".format(percentile))
        print("neutral Sharpe: real {:>6.2f}   null mean {:>6.2f}  (real other-states mean {:>6.2f}, null {:>6.2f})".format(
            real_scalars["neutral_sharpe"], float(null["neutral_sharpe"].mean()),
            real_scalars["other_sharpe_mean"], float(null["other_sharpe_mean"].mean())))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for split, payload in report.items():
            payload["real_by_state"].to_csv(
                output_dir / "neutral_concentration_{}_real.csv".format(split), index=False, encoding="utf-8-sig")
            payload["null"].to_csv(
                output_dir / "neutral_concentration_{}_null.csv".format(split), index=False, encoding="utf-8-sig")
        summary = {split: {"window": p["window"], "real": p["real_scalars"],
                           "percentile_vs_null": p["percentile_vs_null"]} for split, p in report.items()}
        (output_dir / "neutral_concentration_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--model-id", default="hs300_excess_v1_outcome")
    parser.add_argument("--splits", default="validation,test")
    parser.add_argument("--draws", type=int, default=25)
    parser.add_argument("--top-k", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--benchmark-file", default="data/benchmarks/000300.csv")
    parser.add_argument("--config-path", default="config/settings.yaml")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    args = parser.parse_args(argv)
    run(
        Path(args.dataset_dir), args.model_id,
        tuple(s.strip() for s in args.splits.split(",") if s.strip()),
        args.draws, args.top_k, args.horizon, args.benchmark_file,
        args.config_path, args.seed, Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
