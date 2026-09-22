"""Shared evaluation core for cross-sectional feature-discovery studies.

Both ``reversal_feature_study`` and ``fundamental_feature_study`` answer the same
question -- "does this causal feature predict the forward HS300-relative excess
``future_excess_return_{h}d``?" -- so the measurement layer lives here once and
the studies only supply their feature panels.

The ladder follows ``factors/factors_rule.md``: daily cross-sectional Rank IC
(mean / ICIR / win rate / t-stat per split) -> 5-quantile equal-weight forward
excess (Top-Bottom spread + monotonicity) -> stability by year and by causal
market state -> redundancy against the existing feature set.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from research.market_regime.data_loader import load_local_price_history
from research.market_regime.market_environment import build_causal_market_features, load_benchmark_history
from research.market_regime.tradability_labels import normalize_a_share_symbols


LABEL_COLUMN = "future_excess_return_{}d"
MAE_COLUMN = "future_mae_{}d"
SPLITS = ("train", "validation", "test")
TEST_SPLIT = "test"
YEAR_MIN_ROWS = 10
STATE_MIN_ROWS = 10
# A single pooled rank IC over a whole year/state would ignore the daily
# cross-section, so strata are always daily-IC-then-average.
SPLIT_MIN_ROWS = 20


def load_study_split(dataset_dir: Path, dataset_id: str, split: str, horizon: int) -> pd.DataFrame:
    """Load eligible rows of one split with the forward-excess label intact."""

    path = Path(dataset_dir) / "trend_probability_{}_tradability_samples_{}.csv".format(split, dataset_id)
    frame = pd.read_csv(path, dtype={"symbol": str}, low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = normalize_a_share_symbols(frame["symbol"])
    eligible = frame["auxiliary_label_eligible"].astype(str).str.lower().eq("true")
    label = LABEL_COLUMN.format(int(horizon))
    mae = MAE_COLUMN.format(int(horizon))
    valid = eligible & frame[label].notnull() & frame[mae].notnull()
    return frame.loc[valid].reset_index(drop=True)


def load_study_panels(
    dataset_dir: Path, dataset_id: str, horizon: int, limit_symbols: Optional[int] = None
) -> tuple:
    """Load and concatenate the three splits, tagging each row with its origin."""

    splits = {name: load_study_split(dataset_dir, dataset_id, name, horizon) for name in SPLITS}
    if limit_symbols:
        for name, frame in splits.items():
            symbols = sorted(frame["symbol"].unique())[: int(limit_symbols)]
            splits[name] = frame[frame["symbol"].isin(symbols)].reset_index(drop=True)
    panel = pd.concat([frame.assign(source_split=name) for name, frame in splits.items()], ignore_index=True)
    return panel, splits


def load_ohlcv_histories(symbols, price_history_dir: Optional[str] = None) -> tuple:
    """Load per-symbol OHLCV frames, skipping symbols without a local cache."""

    histories = {}
    skipped = []
    for symbol in sorted(set(symbols)):
        try:
            prices = load_local_price_history(symbol, price_history_dir=price_history_dir)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append({"symbol": symbol, "reason": str(exc)[:200]})
            continue
        histories[symbol] = prices
    return histories, pd.DataFrame(skipped, columns=["symbol", "reason"])


def read_csv_with_fallback(path, **kwargs) -> pd.DataFrame:
    """Read a cached CSV whose encoding may be utf-8, utf-8-sig, gbk or gb18030."""

    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def daily_rank_ics(dates, actual: np.ndarray, predicted: np.ndarray, min_rows: int = SPLIT_MIN_ROWS) -> np.ndarray:
    """Per-day cross-sectional Spearman IC between a feature and forward excess.

    Days are kept only when at least ``min_rows`` *paired, non-null* observations
    exist and neither side is constant; ``min_rows`` is per-day, so a stratum with
    too few names on a day simply drops that day.  Vectorised across days -- the
    per-day Python loop this replaces was the whole cost of a study run.
    """

    frame = pd.DataFrame({
        "date": pd.to_datetime(pd.Series(np.asarray(dates))).to_numpy(),
        "actual": np.asarray(actual, dtype=float),
        "predicted": np.asarray(predicted, dtype=float),
    }).dropna()
    if frame.empty:
        return np.asarray([], dtype=float)

    grouped = frame.groupby("date", sort=False)
    frame = frame[grouped["actual"].transform("size") >= min_rows]
    if frame.empty:
        return np.asarray([], dtype=float)

    date_key = frame["date"]
    grouped_actual = frame.groupby("date", sort=False)["actual"]
    grouped_predicted = frame.groupby("date", sort=False)["predicted"]
    usable = (grouped_actual.transform("nunique") >= 2) & (grouped_predicted.transform("nunique") >= 2)
    frame = frame[usable]
    if frame.empty:
        return np.asarray([], dtype=float)

    date_key = frame["date"]
    actual_rank = frame["actual"].groupby(date_key, sort=False).rank()
    predicted_rank = frame["predicted"].groupby(date_key, sort=False).rank()
    centered_actual = actual_rank - actual_rank.groupby(date_key, sort=False).transform("mean")
    centered_predicted = predicted_rank - predicted_rank.groupby(date_key, sort=False).transform("mean")
    covariance = (centered_actual * centered_predicted).groupby(date_key, sort=False).sum()
    denominator = np.sqrt(
        (centered_actual ** 2).groupby(date_key, sort=False).sum()
        * (centered_predicted ** 2).groupby(date_key, sort=False).sum()
    )
    ics = (covariance / denominator).replace([np.inf, -np.inf], np.nan).dropna()
    return ics.to_numpy(dtype=float)


def summarize_ic(ics: np.ndarray) -> Dict[str, float]:
    """Mean IC, ICIR, win rate and t-stat from a per-day IC vector."""

    if ics.size == 0:
        return {"mean_ic": np.nan, "icir": np.nan, "win_rate": np.nan, "t_stat": np.nan, "n_obs": 0}
    mean = float(np.mean(ics))
    std = float(np.std(ics, ddof=1)) if ics.size > 1 else np.nan
    return {
        "mean_ic": mean,
        "icir": mean / std if std and std > 0 else np.nan,
        "win_rate": float(np.mean(ics > 0)),
        "t_stat": mean / (std / np.sqrt(ics.size)) if std and std > 0 else np.nan,
        "n_obs": int(ics.size),
    }


def quantile_group_returns(dates, actual: np.ndarray, predicted: np.ndarray, n_q: int = 5) -> Dict[str, float]:
    """Equal-weight forward excess per cross-sectional quantile, spread and monotonicity."""

    frame = pd.DataFrame({"date": pd.to_datetime(pd.Series(np.asarray(dates))), "actual": actual, "predicted": predicted})
    buckets = {q: [] for q in range(n_q)}
    for _, group in frame.groupby("date", sort=False):
        if len(group) < n_q * 5:
            continue
        try:
            quantiles = pd.qcut(group["predicted"].rank(method="first"), n_q, labels=False, duplicates="drop")
        except ValueError:
            continue
        for q in range(n_q):
            mask = quantiles == q
            if mask.sum():
                buckets[q].append(float(group.loc[mask, "actual"].mean()))
    means = {q: float(np.mean(values)) if values else np.nan for q, values in buckets.items()}
    finite = [(q, means[q]) for q in range(n_q) if np.isfinite(means[q])]
    top_bottom = finite[-1][1] - finite[0][1] if len(finite) >= 2 else np.nan
    monotonic = np.nan
    if len(finite) >= 3:
        xs = np.asarray([q for q, _ in finite], dtype=float)
        ys = np.asarray([v for _, v in finite], dtype=float)
        if xs.std() > 0 and ys.std() > 0:
            monotonic = float(np.corrcoef(xs, ys)[0, 1])
    result = {"q{}_mean_excess".format(q): means[q] for q in range(n_q)}
    result["top_bottom_spread"] = top_bottom
    result["monotonic_corr"] = monotonic
    return result


def rank_corr_between(dates, a: np.ndarray, b: np.ndarray, min_rows: int = SPLIT_MIN_ROWS) -> float:
    """Mean daily Spearman correlation between two cross-sectional series."""

    ics = daily_rank_ics(dates, np.asarray(b, dtype=float), np.asarray(a, dtype=float), min_rows=min_rows)
    return float(np.mean(ics)) if ics.size else np.nan


def merge_market_state(frame: pd.DataFrame, benchmark_file: Path) -> pd.DataFrame:
    """Attach the causal three-state market regime by date (for stability strata)."""

    benchmark = load_benchmark_history(benchmark_file)
    market = build_causal_market_features(benchmark)[["market_state"]]
    market.index.name = "date"
    return frame.merge(market.reset_index(), on="date", how="left")


def _stratum_table(
    frame: pd.DataFrame, feature_names: Sequence[str], label: str, stratum: pd.Series, key_name: str, min_rows: int
) -> pd.DataFrame:
    """Daily-IC-then-average within each stratum value.

    ``frame`` and ``stratum`` must be positionally aligned; both are rebuilt on a
    fresh RangeIndex here so the grouping cannot silently pair the wrong rows.
    """

    base = frame.reset_index(drop=True)
    working = pd.DataFrame({
        "date": pd.to_datetime(base["date"]),
        "actual": pd.to_numeric(base[label], errors="coerce").to_numpy(dtype=float),
        "stratum": np.asarray(stratum)[: len(base)],
    })
    working = working.dropna(subset=["stratum"])
    rows = []
    for feature in feature_names:
        working["predicted"] = pd.to_numeric(base[feature], errors="coerce").to_numpy(dtype=float)[working.index]
        for value, group in working.groupby("stratum", sort=True):
            summary = summarize_ic(daily_rank_ics(group["date"], group["actual"].to_numpy(dtype=float),
                                                  group["predicted"].to_numpy(dtype=float), min_rows=min_rows))
            rows.append({"feature": feature, key_name: value, **summary})
    return pd.DataFrame(rows)


def evaluate_feature_panel(
    panel: pd.DataFrame,
    feature_names: Sequence[str],
    label: str,
    existing_proxies: Sequence[str],
    benchmark_file: Path,
) -> Dict[str, pd.DataFrame]:
    """Run the full IC / group / stability / redundancy ladder on one feature panel."""

    ic_rows = []
    for split in SPLITS:
        subset = panel[panel["source_split"] == split]
        actual = pd.to_numeric(subset[label], errors="coerce").to_numpy(dtype=float)
        for feature in feature_names:
            predicted = pd.to_numeric(subset[feature], errors="coerce").to_numpy(dtype=float)
            ic_rows.append({"split": split, "feature": feature, **summarize_ic(daily_rank_ics(subset["date"], actual, predicted))})
    ic_table = pd.DataFrame(ic_rows)

    test = panel[panel["source_split"] == TEST_SPLIT]
    test_actual = pd.to_numeric(test[label], errors="coerce").to_numpy(dtype=float)

    group_rows = []
    for feature in feature_names:
        predicted = pd.to_numeric(test[feature], errors="coerce").to_numpy(dtype=float)
        group_rows.append({"feature": feature, **quantile_group_returns(test["date"], test_actual, predicted)})
    group_table = pd.DataFrame(group_rows)

    year_table = _stratum_table(test, feature_names, label, test["date"].dt.year, "year", YEAR_MIN_ROWS)
    test_with_state = merge_market_state(test, benchmark_file)
    state_table = _stratum_table(
        test, feature_names, label, test_with_state["market_state"], "market_state", STATE_MIN_ROWS,
    )

    redundancy_rows = []
    for feature in feature_names:
        predicted = pd.to_numeric(test[feature], errors="coerce").to_numpy(dtype=float)
        for proxy in existing_proxies:
            if proxy not in test.columns:
                continue
            existing = pd.to_numeric(test[proxy], errors="coerce").to_numpy(dtype=float)
            redundancy_rows.append({
                "feature": feature, "existing_proxy": proxy,
                "mean_rank_corr": rank_corr_between(test["date"], predicted, existing),
            })
    redundancy_table = pd.DataFrame(redundancy_rows)

    return {
        "ic_table": ic_table,
        "group_table": group_table,
        "year_table": year_table,
        "state_table": state_table,
        "redundancy_table": redundancy_table,
    }


def surviving_candidates(
    ic_table: pd.DataFrame, group_table: pd.DataFrame, feature_names: Sequence[str],
    min_abs_ic: float = 0.02, min_icir: float = 0.3, min_win_rate: float = 0.52, min_abs_monotonic: float = 0.5,
) -> pd.DataFrame:
    """Apply the factors_rule.md screening thresholds, plus a sign-consistency gate.

    Thresholds are read off the test split, but a feature whose IC *sign* differs
    between validation and test has not been discovered -- its direction was only
    knowable in hindsight.  Such features are reported (``sign_consistent=False``)
    but do not survive.

    The thresholds themselves are direction-agnostic (``|IC|``, ``|ICIR|``, and how
    far the win rate sits from a coin flip), so a consistently *negative* predictor
    is judged on the same footing as a positive one.
    """

    columns = ["feature", "mean_ic", "validation_ic", "icir", "win_rate",
               "top_bottom_spread", "monotonic_corr", "sign_consistent", "survives"]
    test = ic_table[ic_table["split"] == TEST_SPLIT].set_index("feature")
    validation = ic_table[ic_table["split"] == "validation"].set_index("feature")
    groups = group_table.set_index("feature")
    rows = []
    for feature in feature_names:
        if feature not in test.index or feature not in groups.index:
            continue
        ic = test.loc[feature]
        group = groups.loc[feature]
        validation_ic = float(validation.loc[feature, "mean_ic"]) if feature in validation.index else np.nan
        sign_consistent = bool(
            np.isfinite(validation_ic) and np.isfinite(ic["mean_ic"])
            and np.sign(validation_ic) == np.sign(ic["mean_ic"])
            and abs(validation_ic) > min_abs_ic / 2
        )
        rows.append({
            "feature": feature,
            "mean_ic": ic["mean_ic"],
            "validation_ic": validation_ic,
            "icir": ic["icir"],
            "win_rate": ic["win_rate"],
            "top_bottom_spread": group["top_bottom_spread"],
            "monotonic_corr": group["monotonic_corr"],
            "sign_consistent": sign_consistent,
            "survives": bool(
                abs(ic["mean_ic"]) > min_abs_ic
                and abs(ic["icir"]) > min_icir
                and max(ic["win_rate"], 1.0 - ic["win_rate"]) > min_win_rate
                and abs(group["monotonic_corr"]) > min_abs_monotonic
                and sign_consistent
            ),
        })
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("mean_ic", key=abs, ascending=False).reset_index(drop=True)


def export_study(
    report: Mapping[str, object], output_dir: Path, run_id: str, prefix: str, metadata: Mapping[str, object]
) -> Dict[str, str]:
    """Write the evaluation tables, the survivor shortlist and a reproducible manifest."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("ic", "group", "year", "state", "redundancy"):
        table = report.get("{}_table".format(name))
        if table is None:
            continue
        path = output_dir / "{}_{}_{}.csv".format(prefix, name, run_id)
        table.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = str(path)

    survivors = surviving_candidates(report["ic_table"], report["group_table"], report["feature_names"])
    survivors_path = output_dir / "{}_survivors_{}.csv".format(prefix, run_id)
    survivors.to_csv(survivors_path, index=False, encoding="utf-8-sig")
    paths["survivors"] = str(survivors_path)

    manifest = {
        "analysis_id": run_id,
        "created_at": datetime.now().isoformat(),
        "question": report["question"],
        "label": report["label"],
        "feature_specs": report["feature_specs"],
        "feature_count": len(report["feature_names"]),
        "existing_proxies": list(report["existing_proxies"]),
        "overall_feature_missing_rate": report["missing_rate"],
        "row_counts": report["row_counts"],
        "symbol_count": report["symbol_count"],
        "screening_thresholds": "test |mean_ic|>0.02, icir>0.3, win_rate>0.52, |monotonic_corr|>0.5",
        "paths": paths,
        **dict(metadata),
    }
    if report.get("extra_manifest"):
        manifest.update(report["extra_manifest"])
    manifest_path = output_dir / "{}_manifest_{}.json".format(prefix, run_id)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["manifest"] = str(manifest_path)
    return paths
