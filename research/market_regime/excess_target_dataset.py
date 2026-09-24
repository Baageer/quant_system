"""Build tradability targets measured against the HS300 index.

This module replaces the event-centric trend labels (``up``/``down`` turning
points confirmed offline by ``labeler``) with honest, executable labels whose
magnitude and market beta are meaningful:

1. **T+1 fixed-entry excess** (primary, consumed by ``trade_outcome_model``).
   Entry is the next local trading session after the EOD feature date; the
   excess return of holding ``h`` sessions is measured against the HS300 index
   over the *same two sessions*.  Columns reuse the existing schema
   ``future_return_{h}d / future_excess_return_{h}d / future_mae_{h}d`` with
   h in {5, 10, 20}.
2. **Tradeable-window view** (auxiliary).  ``window_excess_{W}w_{h}d`` is the
   best h-day excess achievable by entering on any session in
   ``signal+1 .. signal+W``, together with the offset that achieves it and the
   MAE of that entry.  This is an upper bound on a sustainable buy point.
3. **Ordinal grade** ``window_grade_{W}w_{h}d`` bucketting the window excess
   into five buckets whose breakpoints are fit on training data only.

The rows and the 28 frozen features are copied verbatim from the source
``*_model_samples_*`` exports: only the supervised target semantics change.
Per-split eligibility purging reuses ``mark_auxiliary_label_eligibility`` so
no forward window crosses the original train/validation/test boundaries.

Example::

    python -m research.market_regime.excess_target_dataset \
        --dataset-dir output/datasets --dataset-id hs300_probability_dataset \
        --benchmark-file data/benchmarks/000300.csv \
        --horizon 10 --window 5 --run-id excess_hs300_v1
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from research.market_regime.output_paths import DATASET_DIR

from research.market_regime.tradability_labels import (
    add_auxiliary_labels,
    load_model_splits,
    load_price_histories,
    mark_auxiliary_label_eligibility,
)


DEFAULT_HORIZONS = (5, 10, 20)
DEFAULT_HORIZON = 10
DEFAULT_WINDOW = 5
BENCHMARK_NAME = "HS300"
#: Display names for the benchmark files, keyed by file stem.  The excess label
#: itself is always taken from ``--benchmark-file``; this only feeds the
#: human-readable strings, which used to say "HS300" no matter which index was
#: passed -- so every 中证1000 and 中证500 manifest described the wrong index.
BENCHMARK_NAMES = {
    "000300": "HS300",
    "000852": "中证1000",
    "000905": "中证500",
}


def benchmark_name(benchmark_file: Union[str, Path]) -> str:
    """Display name for a benchmark file, falling back to its stem."""

    stem = Path(str(benchmark_file)).stem
    return BENCHMARK_NAMES.get(stem, stem)


def window_columns(window: int, horizon: int) -> Tuple[str, str, str]:
    """Return the (excess, entry offset, MAE) column names for ``W``/``h``."""

    window, horizon = int(window), int(horizon)
    if window < 1 or horizon < 1:
        raise ValueError("window and horizon must be positive integers")
    return (
        "window_excess_{}w_{}d".format(window, horizon),
        "window_entry_offset_{}w_{}d".format(window, horizon),
        "window_mae_{}w_{}d".format(window, horizon),
    )


def window_grade_column(window: int, horizon: int) -> str:
    """Return the ordinal grade column name for ``W``/``h``."""

    return "window_grade_{}w_{}d".format(int(window), int(horizon))


def load_hs300_benchmark(path: Path) -> pd.Series:
    """Load the HS300 index close level as a date-indexed Series.

    The local benchmark CSV is quoted and starts with a UTF-8 BOM, so it is
    read with ``encoding="utf-8-sig"``.
    """

    data = pd.read_csv(path, encoding="utf-8-sig")
    data = data.loc[:, ~data.columns.astype(str).str.startswith("Unnamed")]
    required = {"date", "close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Benchmark CSV is missing columns: {}".format(sorted(missing)))
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["date", "close"]).sort_values("date")
    data = data.drop_duplicates("date", keep="last")
    if data.empty:
        raise ValueError("Benchmark CSV has no valid date/close rows")
    close = data.set_index("date")["close"]
    close.index.name = "date"
    if (close <= 0).any():
        raise ValueError("Benchmark close levels must be strictly positive")
    return close


def add_window_best_labels(
    samples: pd.DataFrame,
    price_histories: Mapping[str, pd.DataFrame],
    benchmark_close: pd.Series,
    window: int = DEFAULT_WINDOW,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """Append the best-entry tradeable-window labels.

    For every sample whose EOD feature date is ``t``, the candidate entries are
    sessions ``t+1 .. t+W`` (T+1 is therefore always a candidate).  A candidate
    enters at ``t+k`` and holds ``horizon`` sessions.  The best candidate
    maximises the benchmark-relative h-day excess.  Rows without a single
    complete candidate keep NaN values.
    """

    window, horizon = int(window), int(horizon)
    if window < 1 or horizon < 1:
        raise ValueError("window and horizon must be positive integers")
    excess_column, offset_column, mae_column = window_columns(window, horizon)
    result = samples.copy()
    for column in (excess_column, offset_column, mae_column):
        result[column] = np.nan
    benchmark = pd.Series(pd.to_numeric(benchmark_close, errors="coerce"))
    benchmark.index.name = "date"

    for symbol, index in tqdm(
        result.groupby("symbol", sort=False).groups.items(),
        total=result["symbol"].nunique(),
        desc="Adding window best-entry labels",
        unit="symbol",
    ):
        prices = price_histories.get(symbol)
        if prices is None or prices.empty:
            continue
        close = np.asarray(pd.to_numeric(prices["close"], errors="coerce"), dtype=float)
        length = len(close)
        sample_dates = pd.DatetimeIndex(result.loc[index, "date"])
        positions = prices.index.get_indexer(sample_dates)
        row_count = len(positions)
        # Forward minimum over [position, position + horizon] for MAE of any entry.
        forward_min = np.asarray(
            pd.Series(close[::-1]).rolling(horizon + 1, min_periods=horizon + 1).min(), dtype=float
        )[::-1]
        benchmark_level = np.asarray(benchmark.reindex(prices.index), dtype=float)
        excess_matrix = np.full((row_count, window), np.nan, dtype=float)
        mae_matrix = np.full((row_count, window), np.nan, dtype=float)
        for offset in range(1, window + 1):
            entries = positions + offset
            exits = entries + horizon
            valid = (positions >= 0) & (entries >= 0) & (entries < length) & (exits < length)
            selected = np.flatnonzero(valid)
            if selected.size == 0:
                continue
            entry_positions = entries[selected]
            exit_positions = exits[selected]
            entry_close = close[entry_positions]
            exit_close = close[exit_positions]
            benchmark_entry = benchmark_level[entry_positions]
            benchmark_exit = benchmark_level[exit_positions]
            good = (
                np.isfinite(entry_close)
                & np.isfinite(exit_close)
                & (entry_close > 0)
                & np.isfinite(benchmark_entry)
                & np.isfinite(benchmark_exit)
                & (benchmark_entry > 0)
            )
            returns = np.full(selected.shape, np.nan, dtype=float)
            mae_values = np.full(selected.shape, np.nan, dtype=float)
            if np.any(good):
                entry_positions_good = entry_positions[good]
                returns[good] = (
                    exit_close[good] / entry_close[good]
                    - benchmark_exit[good] / benchmark_entry[good]
                )
                mae_values[good] = forward_min[entry_positions_good] / entry_close[good] - 1.0
            excess_matrix[selected, offset - 1] = returns
            mae_matrix[selected, offset - 1] = mae_values
        has_candidate = ~np.isnan(excess_matrix).all(axis=1)
        if not has_candidate.any():
            continue
        candidate_rows = np.asarray(index)[has_candidate]
        valid_excess = excess_matrix[has_candidate]
        best = np.argmax(np.where(np.isnan(valid_excess), -np.inf, valid_excess), axis=1)
        rows = np.arange(valid_excess.shape[0])
        result.loc[candidate_rows, excess_column] = valid_excess[rows, best]
        result.loc[candidate_rows, offset_column] = best + 1.0
        result.loc[candidate_rows, mae_column] = mae_matrix[has_candidate][rows, best]
    return result


def fit_window_grade_thresholds(
    train_samples: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    horizon: int = DEFAULT_HORIZON,
) -> Dict[str, object]:
    """Fit the window-excess grade scale on eligible training rows only.

    The scale is ``m = median(|window_excess|)``; if ``m`` degenerates to zero
    the standard deviation of the window excess is used as a fallback.  Buckets
    are ``strong_up >= 2m``, ``up in [m, 2m)``, ``none in (-m, m)``,
    ``down in (-2m, -m]`` and ``strong_down <= -2m``.
    """

    window, horizon = int(window), int(horizon)
    excess_column, _, _ = window_columns(window, horizon)
    eligible = train_samples["auxiliary_label_eligible"].astype(bool).to_numpy()
    values = pd.to_numeric(train_samples[excess_column], errors="coerce").to_numpy(dtype=float)
    values = values[eligible & np.isfinite(values)]
    if values.size == 0:
        raise ValueError(
            "No eligible training rows with a finite {} to fit grade thresholds".format(excess_column)
        )
    m = float(np.median(np.abs(values)))
    if not np.isfinite(m) or m <= 0.0:
        m = float(np.std(values))
    if not np.isfinite(m) or m <= 0.0:
        m = float(np.mean(np.abs(values)))
    if not np.isfinite(m) or m <= 0.0:
        m = float(np.abs(values).max()) or 1.0
    return {
        "window": int(window),
        "horizon": int(horizon),
        "target_column": excess_column,
        "fit_population": "eligible_train_rows",
        "fit_row_count": int(values.size),
        "m": m,
        "buckets": {
            "strong_up": ">= 2m",
            "up": "[m, 2m)",
            "none": "(-m, m)",
            "down": "(-2m, -m]",
            "strong_down": "<= -2m",
        },
    }


def add_window_grade(
    samples: pd.DataFrame,
    thresholds: Mapping[str, object],
    window: int = DEFAULT_WINDOW,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """Append the ordinal ``window_grade_{W}w_{h}d`` column."""

    window, horizon = int(window), int(horizon)
    excess_column, _, _ = window_columns(window, horizon)
    grade_column = window_grade_column(window, horizon)
    result = samples.copy()
    m = float(thresholds["m"])
    values = pd.to_numeric(result[excess_column], errors="coerce")
    buckets = np.select(
        [
            values >= 2.0 * m,
            values >= m,
            values > -m,
            values > -2.0 * m,
        ],
        ["strong_up", "up", "none", "down"],
        default="strong_down",
    )
    result[grade_column] = pd.Series(
        np.asarray(buckets, dtype=object), index=result.index
    ).where(values.notna())
    return result


def summarize_splits(samples: pd.DataFrame, grade_column: str, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    """Return per-split counts by auxiliary eligibility and window grade."""

    graded = samples.copy()
    graded["_grade_present"] = graded[grade_column].notnull() & graded["auxiliary_label_eligible"]
    graded.loc[~graded["_grade_present"], grade_column] = "unavailable"
    summary = (
        graded.groupby(["source_split", "auxiliary_label_eligible", grade_column])
        .size()
        .rename("sample_count")
        .reset_index()
    )
    return summary.sort_values(
        ["source_split", "auxiliary_label_eligible", grade_column]
    ).reset_index(drop=True)


def build_excess_target_splits(
    dataset_dir: Path,
    dataset_id: str,
    benchmark_file: Path,
    price_history_dir: Optional[str] = None,
    horizon: int = DEFAULT_HORIZON,
    window: int = DEFAULT_WINDOW,
    limit_symbols: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, object]:
    """Build HS300-relative T+1 and window labels on the frozen model rows."""

    samples, source_manifest = load_model_splits(dataset_dir, dataset_id)
    if limit_symbols is not None:
        symbols = sorted(samples["symbol"].unique())[: int(limit_symbols)]
        samples = samples.loc[samples["symbol"].isin(symbols)]
    price_histories, skipped = load_price_histories(samples["symbol"], price_history_dir)
    benchmark_close = load_hs300_benchmark(benchmark_file)
    labeled = add_auxiliary_labels(
        samples,
        price_histories,
        horizons=DEFAULT_HORIZONS,
        benchmark_returns=benchmark_close,
    )
    labeled = mark_auxiliary_label_eligibility(labeled, source_manifest)
    labeled = add_window_best_labels(
        labeled, price_histories, benchmark_close, window=window, horizon=horizon
    )
    train_only = labeled.loc[labeled["source_split"] == "train"]
    grade_thresholds = fit_window_grade_thresholds(train_only, window=window, horizon=horizon)
    labeled = add_window_grade(labeled, grade_thresholds, window=window, horizon=horizon)
    splits = {}
    for split in ("train", "validation", "test"):
        frame = labeled.loc[labeled["source_split"] == split]
        splits[split] = frame.drop(columns=["source_split"])
    summary = summarize_splits(labeled, window_grade_column(window, horizon), horizon)
    return {
        "splits": splits,
        "skipped": skipped,
        "grade_thresholds": grade_thresholds,
        "source_manifest": source_manifest,
        "summary": summary,
        "labeled": labeled,
    }


def export_excess_targets(
    report: Mapping[str, object],
    output_dir: Path,
    dataset_dir: Path,
    dataset_id: str,
    benchmark_file: Path,
    run_id: str,
    horizon: int = DEFAULT_HORIZON,
    window: int = DEFAULT_WINDOW,
) -> Dict[str, str]:
    """Write per-split tradability-schema CSVs and a reproducible manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = report["splits"]
    source_manifest = report["source_manifest"]
    paths = {"splits": {}}
    for split, frame in splits.items():
        path = output_dir / "trend_probability_{}_tradability_samples_{}.csv".format(split, run_id)
        frame.to_csv(path, index=False)
        paths["splits"][split] = str(path)
    skipped = report["skipped"]
    skipped_path = output_dir / "excess_target_skipped_prices_{}.csv".format(run_id)
    skipped.to_csv(skipped_path, index=False)
    summary = report["summary"]
    summary_path = output_dir / "excess_target_split_summary_{}.csv".format(run_id)
    summary.to_csv(summary_path, index=False)
    grade_column = window_grade_column(window, horizon)
    split_stats = {}
    labeled = report["labeled"]
    for split in ("train", "validation", "test"):
        subset = labeled.loc[labeled["source_split"] == split]
        eligibility = subset["auxiliary_label_eligible"].astype(bool)
        split_stats[split] = {
            "source_rows": int(len(subset)),
            "eligible_rows": int(eligibility.sum()),
            "unavailable_rows": int((~eligibility).sum()),
        }
    manifest = {
        "dataset_id": run_id,
        "analysis_id": run_id,
        "created_at": datetime.now().isoformat(),
        "label_semantics": {
            "event_centric": False,
            "benchmark": "{} index daily close; excess measured over the same two local sessions".format(
                benchmark_name(benchmark_file)
            ),
            "benchmark_file": str(benchmark_file),
            "entry_rule": "T+1 fixed: next local trading session after EOD feature date",
            "horizons": list(DEFAULT_HORIZONS),
            "primary_horizon": int(horizon),
            "window": int(window),
            "window_best_definition": (
                "max over h-day excess for entry offsets 1..W after the signal; "
                "offset, excess and MAE of that best entry are exported"
            ),
            "grade_definition": {
                "target": grade_column,
                "fit_population": "eligible training rows only",
                "buckets": report["grade_thresholds"]["buckets"],
            },
        },
        "grade_thresholds": {
            key: report["grade_thresholds"][key]
            for key in ("window", "horizon", "target_column", "fit_population", "fit_row_count", "m", "buckets")
        },
        "source_dataset": {
            "dataset_id": dataset_id,
            "analysis_id": source_manifest.get("analysis_id"),
            "manifest": str(dataset_dir / "trend_probability_dataset_manifest_{}.json".format(dataset_id)),
            "feature_count": int(source_manifest.get("candidate_feature_count", 0)),
            "split_boundaries": {
                "train_end": source_manifest.get("train_end"),
                "validation_start": source_manifest.get("validation_start"),
                "validation_end": source_manifest.get("validation_end"),
                "test_start": source_manifest.get("test_start"),
            },
        },
        "eligibility_rule": "forward windows crossing validation_start/test_start are purged via mark_auxiliary_label_eligibility",
        "split_stats": split_stats,
        "paths": {
            "splits": paths["splits"],
            "skipped_prices": str(skipped_path),
            "split_summary": str(summary_path),
        },
        "price_symbol_count": int(splits["train"]["symbol"].nunique()),
        "skipped_price_symbol_count": int(len(skipped)),
    }
    manifest_path = output_dir / "excess_target_manifest_{}.json".format(run_id)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"manifest": str(manifest_path), "split_summary": str(summary_path), "skipped_prices": str(skipped_path)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--dataset-id", default="hs300_probability_dataset")
    parser.add_argument("--benchmark-file", default="data/benchmarks/000300.csv")
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--output-dir", default=DATASET_DIR)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--run-id", default="excess_hs300_v1")
    parser.add_argument("--limit-symbols", type=int, default=None, help="Smoke-test only the first N symbols.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = build_excess_target_splits(
        Path(args.dataset_dir),
        args.dataset_id,
        Path(args.benchmark_file),
        price_history_dir=args.price_history_dir,
        horizon=args.horizon,
        window=args.window,
        limit_symbols=args.limit_symbols,
        show_progress=not args.no_progress,
    )
    paths = export_excess_targets(
        report,
        Path(args.output_dir),
        Path(args.dataset_dir),
        args.dataset_id,
        Path(args.benchmark_file),
        args.run_id,
        horizon=args.horizon,
        window=args.window,
    )
    scale = report["grade_thresholds"]["m"]
    eligible = sum(
        int(frame["auxiliary_label_eligible"].astype(bool).sum())
        for frame in report["splits"].values()
    )
    symbols = report["splits"]["train"]["symbol"].nunique()
    print(
        "Built {benchmark}-relative targets for {symbols} symbols; "
        "eligible rows={eligible}; grade scale m={m:.6g}; manifest={manifest}".format(
            benchmark=benchmark_name(args.benchmark_file),
            symbols=symbols,
            eligible=eligible,
            m=scale,
            manifest=paths["manifest"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
