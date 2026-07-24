"""Add T+1-aligned tradability labels to an existing probability dataset.

The module never adds future outcomes as features.  It only appends supervised
targets whose availability is recorded explicitly, so auxiliary-model training
can purge samples whose longest forward window crosses a split boundary.

Example::

    D:\\Anaconda3\\python.exe -m research.market_regime.tradability_labels \
        --dataset-dir output/datasets --dataset-id hs300_probability_dataset
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from research.market_regime.data_loader import load_local_price_history


DEFAULT_HORIZONS = (5, 10, 20)
DEFAULT_VOLATILITY_WINDOW = 20


def load_model_splits(dataset_dir: Path, dataset_id: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load existing model samples while preserving A-share leading zeroes."""

    manifest_path = dataset_dir / "trend_probability_dataset_manifest_{}.json".format(dataset_id)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    path_keys = {
        "train": "train_model_samples",
        "validation": "validation_model_samples",
        "test": "test_model_samples",
    }
    frames = []
    for split, key in path_keys.items():
        path = Path(manifest["paths"][key])
        samples = pd.read_csv(path, dtype={"symbol": str})
        samples["source_split"] = split
        frames.append(samples)
    samples = pd.concat(frames, ignore_index=True)
    samples["date"] = pd.to_datetime(samples["date"], errors="coerce")
    samples["symbol"] = normalize_a_share_symbols(samples["symbol"])
    return samples.dropna(subset=["date", "symbol"]), manifest


def normalize_a_share_symbols(symbols: pd.Series) -> pd.Series:
    """Restore six-digit A-share codes lost to CSV numeric type inference."""

    normalized = symbols.astype(str).str.strip()
    numeric_mask = normalized.str.match(r"^\d+$")
    normalized.loc[numeric_mask] = normalized.loc[numeric_mask].str.zfill(6)
    return normalized


def load_price_histories(symbols, price_history_dir=None) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Load price histories once for label creation and benchmark construction."""

    histories = {}
    skipped = []
    for symbol in tqdm(sorted(set(symbols)), desc="Loading auxiliary-label prices", unit="symbol"):
        try:
            prices = load_local_price_history(symbol, price_history_dir=price_history_dir)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append({"symbol": symbol, "reason": str(exc)})
            continue
        histories[symbol] = prices.loc[:, ["close"]].copy()
    return histories, pd.DataFrame(skipped, columns=["symbol", "reason"])


def build_cross_section_benchmark(price_histories: Mapping[str, pd.DataFrame]) -> pd.Series:
    """Build a label-only equal-weight universe benchmark from local prices."""

    if not price_histories:
        return pd.Series(dtype=float, name="cross_section_benchmark")
    close_matrix = pd.DataFrame({symbol: prices["close"] for symbol, prices in price_histories.items()})
    daily_returns = close_matrix.pct_change()
    benchmark_returns = daily_returns.mean(axis=1, skipna=True).fillna(0.0)
    return (1.0 + benchmark_returns).cumprod().rename("cross_section_benchmark")


def add_auxiliary_labels(
    samples: pd.DataFrame,
    price_histories: Mapping[str, pd.DataFrame],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    volatility_window: int = DEFAULT_VOLATILITY_WINDOW,
) -> pd.DataFrame:
    """Append future return, excess return, volatility and MAE labels.

    Entry is the next trading session after the sample's EOD feature date.
    A horizon of ``h`` exits after ``h`` additional sessions.  The resulting
    label availability date is therefore the exit session, not the signal day.
    """

    horizons = tuple(sorted({int(value) for value in horizons}))
    if not horizons or horizons[0] < 1:
        raise ValueError("horizons must contain positive integers")
    result = samples.copy()
    for horizon in horizons:
        for column in (
            "entry_date_{}d".format(horizon),
            "exit_date_{}d".format(horizon),
            "future_return_{}d".format(horizon),
            "future_excess_return_{}d".format(horizon),
            "future_risk_adjusted_return_{}d".format(horizon),
            "future_mae_{}d".format(horizon),
        ):
            result[column] = np.nan
    benchmark = build_cross_section_benchmark(price_histories)
    for symbol, index in tqdm(
        result.groupby("symbol", sort=False).groups.items(),
        total=result["symbol"].nunique(),
        desc="Building auxiliary labels",
        unit="symbol",
    ):
        prices = price_histories.get(symbol)
        if prices is None or prices.empty:
            continue
        sample_dates = pd.DatetimeIndex(result.loc[index, "date"])
        positions = prices.index.get_indexer(sample_dates)
        close = np.asarray(pd.to_numeric(prices["close"], errors="coerce"), dtype=float)
        volatility = np.asarray(
            pd.Series(close, index=prices.index).pct_change().rolling(volatility_window).std(), dtype=float
        )
        for horizon in horizons:
            entry_positions = positions + 1
            exit_positions = entry_positions + horizon
            valid = (positions >= 0) & (entry_positions < len(close)) & (exit_positions < len(close))
            if not valid.any():
                continue
            valid_indices = np.asarray(index)[valid]
            valid_entry_positions = entry_positions[valid]
            valid_exit_positions = exit_positions[valid]
            entry_close = close[valid_entry_positions]
            exit_close = close[valid_exit_positions]
            good_prices = np.isfinite(entry_close) & np.isfinite(exit_close) & (entry_close > 0)
            if not good_prices.any():
                continue
            assign_indices = valid_indices[good_prices]
            entries = valid_entry_positions[good_prices]
            exits = valid_exit_positions[good_prices]
            returns = exit_close[good_prices] / entry_close[good_prices] - 1.0
            forward_min = np.asarray(
                pd.Series(close[::-1]).rolling(horizon + 1, min_periods=horizon + 1).min(), dtype=float
            )[::-1]
            mae = forward_min[entries] / entry_close[good_prices] - 1.0
            signal_volatility = volatility[positions[valid][good_prices]]
            risk_adjusted = returns / (signal_volatility * np.sqrt(horizon))
            entry_dates = prices.index[entries]
            exit_dates = prices.index[exits]
            benchmark_entry = np.asarray(benchmark.reindex(entry_dates), dtype=float)
            benchmark_exit = np.asarray(benchmark.reindex(exit_dates), dtype=float)
            benchmark_return = benchmark_exit / benchmark_entry - 1.0
            result.loc[assign_indices, "entry_date_{}d".format(horizon)] = entry_dates
            result.loc[assign_indices, "exit_date_{}d".format(horizon)] = exit_dates
            result.loc[assign_indices, "future_return_{}d".format(horizon)] = returns
            result.loc[assign_indices, "future_excess_return_{}d".format(horizon)] = returns - benchmark_return
            result.loc[assign_indices, "future_risk_adjusted_return_{}d".format(horizon)] = risk_adjusted
            result.loc[assign_indices, "future_mae_{}d".format(horizon)] = mae
    max_horizon = max(horizons)
    result["auxiliary_label_available_date"] = pd.to_datetime(
        result["exit_date_{}d".format(max_horizon)], errors="coerce"
    )
    return result


def mark_auxiliary_label_eligibility(samples: pd.DataFrame, manifest: Mapping[str, object]) -> pd.DataFrame:
    """Purge auxiliary labels that would cross the original split boundaries."""

    result = samples.copy()
    available = pd.to_datetime(result["auxiliary_label_available_date"], errors="coerce")
    validation_start = pd.Timestamp(manifest["validation_start"])
    test_start = pd.Timestamp(manifest["test_start"])
    eligible = available.notnull()
    train = result["source_split"] == "train"
    validation = result["source_split"] == "validation"
    eligible &= (~train | (available < validation_start))
    eligible &= (~validation | (available < test_start))
    result["auxiliary_label_eligible"] = eligible.astype(bool)
    result["auxiliary_label_reason"] = np.where(
        available.isnull(),
        "price_horizon_unavailable",
        np.where(
            train & (available >= validation_start),
            "purged_auxiliary_window",
            np.where(validation & (available >= test_start), "purged_auxiliary_window", "eligible"),
        ),
    )
    return result


def fit_tradability_thresholds(samples: pd.DataFrame, horizon: int = 10) -> Dict[str, float]:
    """Fit binary auxiliary-target thresholds on eligible training rows only."""

    columns = [
        "future_risk_adjusted_return_{}d".format(horizon),
        "future_excess_return_{}d".format(horizon),
        "future_mae_{}d".format(horizon),
    ]
    train_up = samples[
        (samples["source_split"] == "train")
        & samples["auxiliary_label_eligible"]
        & (samples["target"] == "up")
    ].copy()
    train_up = train_up.dropna(subset=columns)
    if train_up.empty:
        raise ValueError("No eligible training up samples for tradability threshold fitting")
    return {
        "horizon": int(horizon),
        "risk_adjusted_return_threshold": float(train_up[columns[0]].median()),
        "excess_return_threshold": 0.0,
        "mae_threshold": float(train_up[columns[2]].quantile(0.25)),
        "fit_population": "eligible_train_up_only",
        "fit_row_count": int(len(train_up)),
    }


def add_tradability_target(samples: pd.DataFrame, thresholds: Mapping[str, float]) -> pd.DataFrame:
    """Create the secondary ``tradable_up`` / ``not_tradable`` target."""

    result = samples.copy()
    horizon = int(thresholds["horizon"])
    valid = result["auxiliary_label_eligible"].copy()
    risk_adjusted = pd.to_numeric(result["future_risk_adjusted_return_{}d".format(horizon)], errors="coerce")
    excess = pd.to_numeric(result["future_excess_return_{}d".format(horizon)], errors="coerce")
    mae = pd.to_numeric(result["future_mae_{}d".format(horizon)], errors="coerce")
    valid &= risk_adjusted.notnull() & excess.notnull() & mae.notnull()
    tradable = (
        (result["target"] == "up")
        & (risk_adjusted >= float(thresholds["risk_adjusted_return_threshold"]))
        & (excess >= float(thresholds["excess_return_threshold"]))
        & (mae >= float(thresholds["mae_threshold"]))
    )
    result["tradability_target_{}d".format(horizon)] = np.where(
        valid, np.where(tradable, "tradable_up", "not_tradable"), "unavailable"
    )
    return result


def export_tradability_dataset(
    dataset_dir: Path,
    dataset_id: str,
    output_dir: Path,
    price_history_dir: Optional[str] = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    volatility_window: int = DEFAULT_VOLATILITY_WINDOW,
) -> Dict[str, object]:
    """Create per-split auxiliary-label exports and a reproducible manifest."""

    samples, source_manifest = load_model_splits(dataset_dir, dataset_id)
    histories, skipped = load_price_histories(samples["symbol"], price_history_dir)
    labeled = add_auxiliary_labels(samples, histories, horizons, volatility_window)
    labeled = mark_auxiliary_label_eligibility(labeled, source_manifest)
    thresholds = fit_tradability_thresholds(labeled, horizon=10)
    labeled = add_tradability_target(labeled, thresholds)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ("train", "validation", "test"):
        path = output_dir / "trend_probability_{}_tradability_samples_{}.csv".format(split, dataset_id)
        labeled[labeled["source_split"] == split].drop(["source_split"], axis=1).to_csv(path, index=False)
        paths[split] = str(path)
    skipped_path = output_dir / "trend_probability_tradability_skipped_prices_{}.csv".format(dataset_id)
    skipped.to_csv(skipped_path, index=False)
    summary = (
        labeled.groupby(["source_split", "auxiliary_label_eligible", "tradability_target_10d"])
        .size().rename("sample_count").reset_index()
    )
    summary_path = output_dir / "trend_probability_tradability_split_summary_{}.csv".format(dataset_id)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "dataset_id": dataset_id,
        "created_at": datetime.now().isoformat(),
        "source_manifest": str(dataset_dir / "trend_probability_dataset_manifest_{}.json".format(dataset_id)),
        "label_definition": {
            "execution": "entry at next local trading session after EOD feature date",
            "horizons": list(horizons),
            "volatility_window": int(volatility_window),
            "benchmark": "equal-weight local cross-sectional benchmark; label-only, not a feature",
            "max_horizon_label_available_date": "auxiliary_label_available_date",
            "split_rule": "training and validation rows crossing the next split boundary are purged for auxiliary training",
        },
        "tradability_thresholds_fit_on_train_only": thresholds,
        "paths": {"splits": paths, "skipped_prices": str(skipped_path), "split_summary": str(summary_path)},
        "price_symbol_count": len(histories),
        "skipped_price_symbol_count": int(len(skipped)),
    }
    manifest_path = output_dir / "trend_probability_tradability_manifest_{}.json".format(dataset_id)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def finalize_existing_exports(dataset_dir: Path, dataset_id: str, output_dir: Path) -> Dict[str, object]:
    """Build summary and manifest after an interrupted post-export stage."""

    source_manifest_path = dataset_dir / "trend_probability_dataset_manifest_{}.json".format(dataset_id)
    with source_manifest_path.open("r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    split_frames = []
    paths = {}
    for split in ("train", "validation", "test"):
        path = output_dir / "trend_probability_{}_tradability_samples_{}.csv".format(split, dataset_id)
        frame = pd.read_csv(path, dtype={"symbol": str})
        frame["source_split"] = split
        split_frames.append(frame)
        paths[split] = str(path)
    labeled = pd.concat(split_frames, ignore_index=True)
    thresholds = fit_tradability_thresholds(labeled, horizon=10)
    summary = (
        labeled.groupby(["source_split", "auxiliary_label_eligible", "tradability_target_10d"])
        .size().rename("sample_count").reset_index()
    )
    summary_path = output_dir / "trend_probability_tradability_split_summary_{}.csv".format(dataset_id)
    summary.to_csv(summary_path, index=False)
    skipped_path = output_dir / "trend_probability_tradability_skipped_prices_{}.csv".format(dataset_id)
    skipped_count = max(skipped_path.read_bytes().count(b"\n") - 1, 0) if skipped_path.is_file() else 0
    manifest = {
        "dataset_id": dataset_id,
        "created_at": datetime.now().isoformat(),
        "source_manifest": str(source_manifest_path),
        "label_definition": {
            "execution": "entry at next local trading session after EOD feature date",
            "horizons": list(DEFAULT_HORIZONS),
            "volatility_window": DEFAULT_VOLATILITY_WINDOW,
            "benchmark": "equal-weight local cross-sectional benchmark; label-only, not a feature",
            "max_horizon_label_available_date": "auxiliary_label_available_date",
            "split_rule": "training and validation rows crossing the next split boundary are purged for auxiliary training",
        },
        "tradability_thresholds_fit_on_train_only": thresholds,
        "paths": {"splits": paths, "skipped_prices": str(skipped_path), "split_summary": str(summary_path)},
        "price_symbol_count": int(labeled["symbol"].nunique() - skipped_count),
        "skipped_price_symbol_count": skipped_count,
        "finalized_from_existing_exports": True,
    }
    manifest_path = output_dir / "trend_probability_tradability_manifest_{}.json".format(dataset_id)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="output/datasets")
    parser.add_argument("--dataset-id", default="hs300_probability_dataset")
    parser.add_argument("--output-dir", default="output/datasets")
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--horizons", default="5,10,20")
    parser.add_argument("--volatility-window", type=int, default=DEFAULT_VOLATILITY_WINDOW)
    parser.add_argument("--finalize-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.finalize_existing:
        manifest = finalize_existing_exports(Path(args.dataset_dir), args.dataset_id, Path(args.output_dir))
    else:
        horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
        manifest = export_tradability_dataset(
            Path(args.dataset_dir), args.dataset_id, Path(args.output_dir), args.price_history_dir,
            horizons, args.volatility_window,
        )
    print(
        "Created tradability labels for {count} price symbols; "
        "thresholds={thresholds}".format(
            count=manifest["price_symbol_count"], thresholds=manifest["tradability_thresholds_fit_on_train_only"]
        )
    )


if __name__ == "__main__":
    main()
