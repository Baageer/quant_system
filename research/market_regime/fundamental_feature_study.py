"""Factor-discovery study for valuation / size / liquidity features vs HS300 excess.

The frozen 28 features of ``hs300_logistic_v2`` contain **no market-cap and no
turnover/valuation information at all**: ``ret_20d`` and ``volume_ratio_60`` are
the only faint proxies, and the local price CSVs carry a ``turnover`` column that
is entirely NaN.  This module fills that gap with the per-symbol valuation panel
cached at ``data/raw/akshare/valuation/{symbol}_history.csv`` (``stock_a_indicator_lg``:
``pe_ttm`` / ``pb`` / ``ps_ttm`` / ``dv_ttm`` / ``total_mv``), combined with the
locally cached price ``amount`` column.

Two properties matter for causality:

* the valuation panel is **point-in-time** -- each row's PE/PB is computed from the
  financials *disclosed as of that trade date*, so joining on the signal date ``t``
  carries no restatement look-ahead;
* ``amount`` and ``total_mv`` are both unadjusted, so their ratio is a real
  turnover rate.  Rolling windows use only bars up to ``t``.

The valuation cache ends 2025-08-04, so the tail of the test split (2025-08->12)
is uncovered; the manifest reports the per-split match rate.

Example::

    python -m research.market_regime.fundamental_feature_study \
        --dataset-id excess_hs300_v1 --run-id fundamental_study_v1 --horizon 10
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from research.market_regime.feature_study_core import (
    LABEL_COLUMN,
    MAE_COLUMN,
    SPLITS,
    evaluate_feature_panel,
    export_study,
    load_ohlcv_histories,
    load_study_panels,
    read_csv_with_fallback,
    surviving_candidates,
)


PREFIX = "fundamental_feature"
DEFAULT_VALUATION_DIR = "data/raw/akshare/valuation"
VALUATION_COLUMNS = ("pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_mv")
# Force price/amount onto the sample's calendar rather than requiring the vendor
# date to land exactly on the signal date; the tolerance only ever fills backward.
VALUATION_TOLERANCE = pd.Timedelta("7D")
# akshare's valuation file is truncated to the most recent N rows per symbol.
MIN_VALUATION_ROWS = 60

FUNDAMENTAL_FEATURE_SPECS = (
    # ---- size -------------------------------------------------------------
    {"name": "log_total_mv", "category": "size"},
    {"name": "log_amount_20", "category": "size"},
    # ---- cross-sectional value -------------------------------------------
    {"name": "ep_ttm", "category": "value"},
    {"name": "bp", "category": "value"},
    {"name": "sp_ttm", "category": "value"},
    {"name": "dv_ttm", "category": "value"},
    # ---- time-series value (cheap vs own history) -------------------------
    {"name": "ep_z_250", "category": "value_history"},
    {"name": "bp_z_250", "category": "value_history"},
    # ---- liquidity / turnover --------------------------------------------
    {"name": "turnover_mv", "category": "liquidity"},
    {"name": "turnover_mv_mean_20", "category": "liquidity"},
    {"name": "turnover_mv_std_20", "category": "liquidity"},
    {"name": "amihud_20", "category": "liquidity"},
)

FEATURE_NAMES = [spec["name"] for spec in FUNDAMENTAL_FEATURE_SPECS]

# The nearest things the frozen 28 have to size/liquidity.  Value factors should
# come out close to orthogonal to these.
EXISTING_PROXIES = (
    "volume_ratio_60",
    "ret_20d",
    "ret_10d",
    "momentum_10",
    "bollinger_bandwidth_20",
    "vwap_distance",
)


def load_valuation_history(symbol: str, valuation_dir: Path) -> Optional[pd.DataFrame]:
    """Load one symbol's point-in-time valuation panel, or None when absent."""

    path = Path(valuation_dir) / "{}_history.csv".format(symbol)
    if not path.exists():
        return None
    frame = read_csv_with_fallback(path, low_memory=False)
    if "trade_date" not in frame.columns:
        return None
    frame = frame.loc[:, [column for column in ("trade_date",) + VALUATION_COLUMNS if column in frame.columns]]
    frame["date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    for column in VALUATION_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = np.nan
    frame = frame.drop_duplicates("date", keep="last").sort_values("date")
    if len(frame) < MIN_VALUATION_ROWS:
        return None
    return frame.loc[:, ["date", *VALUATION_COLUMNS]]


def _positive_inverse(series: pd.Series) -> pd.Series:
    """1/x restricted to strictly positive x (negative earnings yields are not comparable)."""

    positive = series.where(series > 0)
    return 1.0 / positive


def compute_fundamental_features(prices: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
    """Compute causal size / value / liquidity features for one symbol.

    ``prices`` is date-indexed OHLCV (needs ``close`` and ``amount``);
    ``valuation`` is the point-in-time panel from :func:`load_valuation_history`.
    Both are joined backward onto the price calendar, so every feature at date
    ``t`` uses only information available at ``t``.
    """

    base = prices.reset_index()
    date_column = base.columns[0]
    base = base.rename(columns={date_column: "date"})
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base = base.dropna(subset=["date"]).sort_values("date")

    merged = pd.merge_asof(
        base, valuation.sort_values("date"), on="date",
        direction="backward", tolerance=VALUATION_TOLERANCE,
    ).set_index("date")

    close = pd.to_numeric(merged["close"], errors="coerce")
    amount = pd.to_numeric(merged.get("amount", pd.Series(np.nan, index=merged.index)), errors="coerce")
    pct_change = pd.to_numeric(merged.get("pct_change", pd.Series(np.nan, index=merged.index)), errors="coerce")
    total_mv = pd.to_numeric(merged["total_mv"], errors="coerce")

    features = pd.DataFrame(index=merged.index)

    # ---- size --------------------------------------------------------------
    features["log_total_mv"] = np.log(total_mv.where(total_mv > 0))
    features["log_amount_20"] = np.log(amount.rolling(20, min_periods=20).mean().where(lambda value: value > 0))

    # ---- cross-sectional value --------------------------------------------
    features["ep_ttm"] = _positive_inverse(pd.to_numeric(merged["pe_ttm"], errors="coerce"))
    features["bp"] = _positive_inverse(pd.to_numeric(merged["pb"], errors="coerce"))
    features["sp_ttm"] = _positive_inverse(pd.to_numeric(merged["ps_ttm"], errors="coerce"))
    features["dv_ttm"] = pd.to_numeric(merged["dv_ttm"], errors="coerce")

    # ---- time-series value -------------------------------------------------
    for source, target in (("ep_ttm", "ep_z_250"), ("bp", "bp_z_250")):
        series = features[source]
        mean = series.rolling(250, min_periods=120).mean()
        std = series.rolling(250, min_periods=120).std()
        features[target] = (series - mean) / std.where(std > 0)

    # ---- liquidity / turnover ---------------------------------------------
    # legulegu total_mv is in 万元 and tushare amount in 千元, so the raw ratio is
    # 10x the turnover fraction; the constant does not affect rank statistics but
    # is kept so the level is interpretable.
    turnover = amount / (total_mv * 10.0).where(total_mv > 0)
    features["turnover_mv"] = turnover
    features["turnover_mv_mean_20"] = turnover.rolling(20, min_periods=20).mean()
    features["turnover_mv_std_20"] = turnover.rolling(20, min_periods=20).std()
    features["amihud_20"] = (pct_change.abs() / amount.where(amount > 0)).rolling(20, min_periods=20).mean()

    return features.loc[:, FEATURE_NAMES]


def attach_fundamental_features(
    samples: pd.DataFrame, histories: Dict[str, pd.DataFrame], valuations: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Left-join the causal fundamental features onto the frozen sample rows."""

    parts = []
    for symbol, group in tqdm(samples.groupby("symbol", sort=False), desc="Computing fundamental features", unit="symbol"):
        prices = histories.get(symbol)
        valuation = valuations.get(symbol)
        if prices is None or prices.empty or valuation is None or valuation.empty:
            continue
        group = group.sort_values("date")
        features = compute_fundamental_features(prices, valuation).reindex(pd.DatetimeIndex(group["date"]))
        features = features.reset_index(drop=True)
        features["date"] = group["date"].to_numpy()
        features["symbol"] = symbol
        parts.append(features)
    if not parts:
        raise ValueError("No symbols produced fundamental features")
    return samples.merge(pd.concat(parts, ignore_index=True), on=["date", "symbol"], how="left")


def build_study(
    dataset_dir: Path,
    dataset_id: str,
    benchmark_file: Path,
    horizon: int,
    price_history_dir: Optional[str] = None,
    valuation_dir: Path = Path(DEFAULT_VALUATION_DIR),
    limit_symbols: Optional[int] = None,
) -> dict:
    """Compute features and run the full factor evaluation."""

    label = LABEL_COLUMN.format(int(horizon))
    samples, splits = load_study_panels(dataset_dir, dataset_id, horizon, limit_symbols=limit_symbols)
    histories, skipped = load_ohlcv_histories(samples["symbol"].unique(), price_history_dir)

    valuations = {}
    missing_valuation = []
    for symbol in samples["symbol"].unique():
        valuation = load_valuation_history(symbol, valuation_dir)
        if valuation is None:
            missing_valuation.append(symbol)
        else:
            valuations[symbol] = valuation

    panel = attach_fundamental_features(samples, histories, valuations)
    keep = ["date", "symbol", "source_split", label, MAE_COLUMN.format(int(horizon)), *EXISTING_PROXIES]
    panel = panel.loc[:, [column for column in keep if column in panel.columns] + FEATURE_NAMES]
    panel[FEATURE_NAMES] = panel[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")

    match_rate = {
        split: float(panel.loc[panel["source_split"] == split, "log_total_mv"].notna().mean()) for split in SPLITS
    }

    tables = evaluate_feature_panel(panel, FEATURE_NAMES, label, EXISTING_PROXIES, Path(benchmark_file))
    return {
        "question": "do size / valuation / liquidity features predict forward HS300-relative excess",
        "label": label,
        "horizon": int(horizon),
        "feature_names": FEATURE_NAMES,
        "feature_specs": list(FUNDAMENTAL_FEATURE_SPECS),
        "existing_proxies": list(EXISTING_PROXIES),
        "row_counts": {name: int(len(splits[name])) for name in SPLITS},
        "symbol_count": int(panel["symbol"].nunique()),
        "skipped_prices": skipped,
        "missing_rate": float(panel[FEATURE_NAMES].isnull().mean().mean()),
        "extra_manifest": {
            "valuation_dir": str(valuation_dir),
            "valuation_match_rate_by_split": match_rate,
            "symbols_without_valuation": missing_valuation,
            "skipped_price_symbols": skipped.to_dict("records"),
        },
        **tables,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="output/datasets")
    parser.add_argument("--dataset-id", default="excess_hs300_v1")
    parser.add_argument("--benchmark-file", default="data/benchmarks/000300.csv")
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--valuation-dir", default=DEFAULT_VALUATION_DIR)
    parser.add_argument("--output-dir", default="output/datasets")
    parser.add_argument("--run-id", default="fundamental_study_v1")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--limit-symbols", type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = build_study(
        Path(args.dataset_dir), args.dataset_id, Path(args.benchmark_file), args.horizon,
        price_history_dir=args.price_history_dir, valuation_dir=Path(args.valuation_dir),
        limit_symbols=args.limit_symbols,
    )
    paths = export_study(
        report, Path(args.output_dir), args.run_id, PREFIX,
        {"dataset_id": args.dataset_id, "horizon": args.horizon, "limit_symbols": args.limit_symbols},
    )
    survivors = surviving_candidates(report["ic_table"], report["group_table"], FEATURE_NAMES)
    count = int(survivors["survives"].sum()) if not survivors.empty else 0
    rates = report["extra_manifest"]["valuation_match_rate_by_split"]
    print("Fundamental feature study done. {} features, {} survive screening; match_rate={}; manifest={}".format(
        len(FEATURE_NAMES), count,
        {key: "{:.1%}".format(value) for key, value in rates.items()}, paths["manifest"]))
    if count:
        print("Survivors:\n{}".format(
            survivors[survivors["survives"]][["feature", "mean_ic", "icir", "win_rate", "top_bottom_spread"]].to_string(index=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
