"""Prepare leakage-controlled samples for the future trend probability model.

This module consumes the causal feature panel and the *offline* turning-point
events exported by ``feature_event_study``.  It keeps these responsibilities
separate: panel columns are model inputs, while events are used only to create
future labels and event-study based candidate feature reports.

Example::

    python -m research.market_regime.probability_dataset \
        --study-run-id causal_feature_smoke_v2 --run-id probability_smoke
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .feature_event_study import _effects_from_summary, summarize_turning_point_features


DEFAULT_TRAIN_END = "2018-12-31"
DEFAULT_VALIDATION_START = "2019-01-01"
DEFAULT_VALIDATION_END = "2021-12-31"
DEFAULT_TEST_START = "2022-01-01"
DEFAULT_HORIZON = 10
DEFAULT_EMBARGO_DAYS = 130

SAMPLE_METADATA_COLUMNS = (
    "date",
    "symbol",
    "target",
    "target_event_id",
    "target_event_date",
    "days_to_target",
    "label_available_date",
    "sample_weight",
)


def _as_timestamp(value: Union[str, pd.Timestamp]) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _normalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError("Feature panel is missing columns: {}".format(sorted(missing)))
    result = panel.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["symbol"] = result["symbol"].astype(str)
    result = result.dropna(subset=["date"]).sort_values(["symbol", "date"])
    if result.duplicated(["date", "symbol"]).any():
        raise ValueError("Feature panel has duplicate date × symbol rows")
    return result.reset_index(drop=True)


def _normalize_events(events: pd.DataFrame, min_event_confidence: float) -> pd.DataFrame:
    required = {"symbol", "event_date", "next_regime", "event_id"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError("Turning-point events are missing columns: {}".format(sorted(missing)))
    result = events.copy()
    result["symbol"] = result["symbol"].astype(str)
    result["event_date"] = pd.to_datetime(result["event_date"], errors="coerce")
    result = result.dropna(subset=["event_date"])
    result = result[result["next_regime"].isin(["up", "down"])].copy()
    if "pivot_confidence" in result.columns:
        confidence = pd.to_numeric(result["pivot_confidence"], errors="coerce")
        result = result[confidence >= float(min_event_confidence)].copy()
    return result.sort_values(["symbol", "event_date", "event_id"]).reset_index(drop=True)


def build_future_trend_samples(
    feature_panel: pd.DataFrame,
    turning_point_events: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    min_event_confidence: float = 0.0,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Label each sample as the first up/down event in ``(t, t + H]``.

    The returned ``target`` is ``up``, ``down``, ``none`` or ``unavailable``.
    ``unavailable`` marks the terminal H trading days and must not enter a
    model.  Positive samples belonging to the same event share total weight 1.
    """

    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    panel = _normalize_panel(feature_panel)
    events = _normalize_events(turning_point_events, min_event_confidence)
    parts = []
    for symbol, symbol_panel in tqdm(
        panel.groupby("symbol", sort=False),
        total=panel["symbol"].nunique(),
        desc="构建未来趋势标签",
        unit="标的",
        disable=not show_progress,
    ):
        data = symbol_panel.copy().reset_index(drop=True)
        dates = pd.DatetimeIndex(data["date"])
        symbol_events = events[events["symbol"] == str(symbol)]
        event_positions = dates.get_indexer(pd.DatetimeIndex(symbol_events["event_date"]))
        valid_events = symbol_events.iloc[np.flatnonzero(event_positions >= 0)].reset_index(drop=True)
        event_positions = event_positions[event_positions >= 0]
        order = np.argsort(event_positions, kind="mergesort")
        event_positions = event_positions[order]
        valid_events = valid_events.iloc[order].reset_index(drop=True)

        targets: List[str] = []
        event_ids: List[Optional[str]] = []
        event_dates: List[pd.Timestamp] = []
        days_to_target: List[float] = []
        label_available_dates: List[pd.Timestamp] = []
        for position, date in enumerate(dates):
            terminal_position = position + horizon
            if terminal_position >= len(dates):
                targets.append("unavailable")
                event_ids.append(None)
                event_dates.append(pd.NaT)
                days_to_target.append(np.nan)
                label_available_dates.append(pd.NaT)
                continue
            label_available_dates.append(dates[terminal_position])
            next_event_index = int(np.searchsorted(event_positions, position + 1, side="left"))
            if next_event_index < len(event_positions) and event_positions[next_event_index] <= terminal_position:
                event = valid_events.iloc[next_event_index]
                targets.append(str(event["next_regime"]))
                event_ids.append(str(event["event_id"]))
                event_dates.append(pd.Timestamp(event["event_date"]))
                days_to_target.append(float(event_positions[next_event_index] - position))
            else:
                targets.append("none")
                event_ids.append(None)
                event_dates.append(pd.NaT)
                days_to_target.append(np.nan)
        data["target"] = targets
        data["target_event_id"] = event_ids
        data["target_event_date"] = event_dates
        data["days_to_target"] = days_to_target
        data["label_available_date"] = label_available_dates
        data["sample_weight"] = 1.0
        positive = data["target_event_id"].notnull()
        positive_counts = data.loc[positive, "target_event_id"].value_counts()
        data.loc[positive, "sample_weight"] = data.loc[positive, "target_event_id"].map(
            lambda event_id: 1.0 / float(positive_counts[event_id])
        )
        parts.append(data)
    return pd.concat(parts, ignore_index=True) if parts else panel.assign(target=pd.Series(dtype=str))


def _embargo_start_date(dates: pd.DatetimeIndex, boundary: pd.Timestamp, embargo_days: int) -> pd.Timestamp:
    position = int(dates.searchsorted(boundary, side="left"))
    position = min(position + embargo_days, len(dates))
    return dates[position] if position < len(dates) else pd.NaT


def split_time_series_samples(
    samples: pd.DataFrame,
    train_end: Union[str, pd.Timestamp] = DEFAULT_TRAIN_END,
    validation_start: Union[str, pd.Timestamp] = DEFAULT_VALIDATION_START,
    validation_end: Union[str, pd.Timestamp] = DEFAULT_VALIDATION_END,
    test_start: Union[str, pd.Timestamp] = DEFAULT_TEST_START,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Assign train/validation/test partitions with label purge and embargo.

    Train and validation samples whose future label window reaches the next
    split are purged.  The first ``embargo_days`` trading sessions of
    validation and test are excluded per symbol, matching the plan's required
    feature-lookback plus label-horizon buffer.
    """

    if embargo_days < 0:
        raise ValueError("embargo_days cannot be negative")
    result = samples.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["label_available_date"] = pd.to_datetime(result["label_available_date"])
    result["split"] = "excluded"
    result["split_reason"] = "outside_period"
    train_end_date = _as_timestamp(train_end)
    validation_start_date = _as_timestamp(validation_start)
    validation_end_date = _as_timestamp(validation_end)
    test_start_date = _as_timestamp(test_start)
    if not train_end_date < validation_start_date <= validation_end_date < test_start_date:
        raise ValueError("Split boundaries must satisfy train < validation < test")

    result.loc[result["target"] == "unavailable", "split_reason"] = "label_unavailable"
    usable = result["target"] != "unavailable"
    symbol_groups = result.groupby("symbol", sort=False).groups
    for symbol, indices in tqdm(
        symbol_groups.items(),
        total=len(symbol_groups),
        desc="执行时间切分与隔离",
        unit="标的",
        disable=not show_progress,
    ):
        index = list(indices)
        dates = pd.DatetimeIndex(result.loc[index, "date"])
        validation_embargo_end = _embargo_start_date(dates, validation_start_date, embargo_days)
        test_embargo_end = _embargo_start_date(dates, test_start_date, embargo_days)
        subset = result.loc[index]
        train_mask = (
            usable.loc[index]
            & (subset["date"] <= train_end_date)
            & (subset["label_available_date"] < validation_start_date)
        )
        validation_mask = (
            usable.loc[index]
            & (subset["date"] >= validation_start_date)
            & (subset["date"] <= validation_end_date)
            & (subset["date"] >= validation_embargo_end)
            & (subset["label_available_date"] < test_start_date)
        )
        test_mask = usable.loc[index] & (subset["date"] >= test_start_date) & (subset["date"] >= test_embargo_end)
        result.loc[subset.index[train_mask], ["split", "split_reason"]] = ["train", "eligible"]
        result.loc[subset.index[validation_mask], ["split", "split_reason"]] = ["validation", "eligible"]
        result.loc[subset.index[test_mask], ["split", "split_reason"]] = ["test", "eligible"]

    train_boundary = usable & (result["date"] <= train_end_date) & (result["label_available_date"] >= validation_start_date)
    validation_boundary = (
        usable
        & (result["date"] >= validation_start_date)
        & (result["date"] <= validation_end_date)
        & (result["label_available_date"] >= test_start_date)
    )
    result.loc[train_boundary | validation_boundary, "split_reason"] = "purged_label_window"
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)


def select_event_study_candidates(
    observations: pd.DataFrame,
    selection_end: Union[str, pd.Timestamp] = DEFAULT_TRAIN_END,
    pre_event_start: int = -20,
    pre_event_end: int = -1,
    min_samples_per_day: int = 10,
    min_relative_days: int = 5,
    min_abs_effect: float = 0.20,
    min_direction_consistency: float = 0.60,
    show_progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select features using only pre-event, pre-validation event evidence.

    A feature qualifies for an up/down target when its event-minus-control
    standardized effect is sufficiently large and directionally consistent on
    multiple dates before the event.  The report retains every evaluated
    feature so later choices remain auditable.
    """

    required = {"event_date", "transition", "sample_group", "feature", "relative_day", "value"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError("Event observations are missing columns: {}".format(sorted(missing)))
    cutoff = _as_timestamp(selection_end)
    input_rows = observations.copy()
    input_rows["event_date"] = pd.to_datetime(input_rows["event_date"], errors="coerce")
    input_rows = input_rows[
        (input_rows["event_date"] <= cutoff)
        & (input_rows["relative_day"] >= pre_event_start)
        & (input_rows["relative_day"] <= pre_event_end)
    ].copy()
    _, effects = summarize_turning_point_features(input_rows)
    return select_candidate_features_from_effects(
        effects,
        pre_event_start=pre_event_start,
        pre_event_end=pre_event_end,
        min_samples_per_day=min_samples_per_day,
        min_relative_days=min_relative_days,
        min_abs_effect=min_abs_effect,
        min_direction_consistency=min_direction_consistency,
        show_progress=show_progress,
    )


def select_candidate_features_from_effects(
    effects: pd.DataFrame,
    pre_event_start: int = -20,
    pre_event_end: int = -1,
    min_samples_per_day: int = 10,
    min_relative_days: int = 5,
    min_abs_effect: float = 0.20,
    min_direction_consistency: float = 0.60,
    show_progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select candidate features from compact, already time-filtered effects."""

    if effects.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["feature", "selected"])
    effects = effects[
        (effects["relative_day"] >= pre_event_start) & (effects["relative_day"] <= pre_event_end)
    ].copy()
    effects = effects[effects["transition"].str.endswith(("_to_up", "_to_down"))].copy()
    effects["target"] = np.where(effects["transition"].str.endswith("_to_up"), "up", "down")
    effects["usable_day"] = (effects["count_event"] >= min_samples_per_day) & (effects["count_control"] >= min_samples_per_day)
    rows = []
    effect_groups = effects.groupby(["target", "transition", "feature"])
    for (target, transition, feature), group in tqdm(
        effect_groups,
        total=effect_groups.ngroups,
        desc="按事件效应筛选特征",
        unit="特征组",
        disable=not show_progress,
    ):
        usable = group[group["usable_day"] & group["standardized_effect"].notnull()]
        effect_values = np.asarray(usable["standardized_effect"], dtype=float)
        mean_effect = float(np.mean(effect_values)) if len(effect_values) else np.nan
        direction = float(np.sign(mean_effect)) if np.isfinite(mean_effect) else 0.0
        consistency = float(np.mean(np.sign(effect_values) == direction)) if direction and len(effect_values) else 0.0
        mean_abs_effect = float(np.mean(np.abs(effect_values))) if len(effect_values) else np.nan
        selected = bool(
            len(usable) >= min_relative_days
            and np.isfinite(mean_abs_effect)
            and mean_abs_effect >= min_abs_effect
            and consistency >= min_direction_consistency
        )
        rows.append(
            {
                "target": target,
                "transition": transition,
                "feature": feature,
                "usable_relative_days": int(len(usable)),
                "mean_standardized_effect": mean_effect,
                "mean_abs_standardized_effect": mean_abs_effect,
                "direction_consistency": consistency,
                "minimum_event_count": int(usable["count_event"].min()) if len(usable) else 0,
                "minimum_control_count": int(usable["count_control"].min()) if len(usable) else 0,
                "selected": selected,
            }
        )
    detail = pd.DataFrame(rows).sort_values(["selected", "mean_abs_standardized_effect"], ascending=[False, False])
    selected = detail[detail["selected"]].copy()
    candidate_rows = []
    selected_groups = selected.groupby("feature")
    for feature, group in tqdm(
        selected_groups,
        total=selected_groups.ngroups,
        desc="汇总事件候选特征",
        unit="特征",
        disable=not show_progress,
    ):
        candidate_rows.append(
            {
                "feature": feature,
                "target_count": int(group["target"].nunique()),
                "transition_count": int(group["transition"].nunique()),
                "best_abs_standardized_effect": float(group["mean_abs_standardized_effect"].max()),
                "minimum_direction_consistency": float(group["direction_consistency"].min()),
                "selected": True,
            }
        )
    candidates = pd.DataFrame(
        candidate_rows,
        columns=[
            "feature",
            "target_count",
            "transition_count",
            "best_abs_standardized_effect",
            "minimum_direction_consistency",
            "selected",
        ],
    )
    return detail.reset_index(drop=True), candidates.sort_values("best_abs_standardized_effect", ascending=False).reset_index(drop=True)


def prune_correlated_candidates(
    event_candidates: pd.DataFrame,
    train_samples: pd.DataFrame,
    max_candidates: int = 30,
    correlation_threshold: float = 0.85,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Greedily remove redundant candidates using training-period correlation only."""

    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")
    if not 0 < correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be in (0, 1]")
    columns = list(event_candidates.columns) + ["selection_rank", "selection_reason"]
    if event_candidates.empty:
        return pd.DataFrame(columns=columns)
    ranked = event_candidates.sort_values("best_abs_standardized_effect", ascending=False).copy()
    usable_features = [feature for feature in ranked["feature"] if feature in train_samples.columns]
    correlations = train_samples.loc[:, usable_features].apply(pd.to_numeric, errors="coerce").corr().abs()
    selected_features = []
    rows = []
    for _, row in tqdm(
        ranked.iterrows(),
        total=len(ranked),
        desc="训练期相关性去冗余",
        unit="特征",
        disable=not show_progress,
    ):
        output = row.to_dict()
        feature = row["feature"]
        output["selection_rank"] = np.nan
        if feature not in correlations.columns:
            output["selected"] = False
            output["selection_reason"] = "missing_from_training_samples"
        elif len(selected_features) >= max_candidates:
            output["selected"] = False
            output["selection_reason"] = "candidate_limit"
        else:
            correlated = [
                selected
                for selected in selected_features
                if correlations.loc[feature, selected] >= correlation_threshold
            ]
            if correlated:
                output["selected"] = False
                output["selection_reason"] = "correlated_with:{}".format(correlated[0])
            else:
                selected_features.append(feature)
                output["selected"] = True
                output["selection_rank"] = len(selected_features)
                output["selection_reason"] = "selected"
        rows.append(output)
    return pd.DataFrame(rows, columns=columns)


def apply_candidate_feature_set(samples: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Return model input columns plus labels using the final selected features."""

    if candidates.empty or "selected" not in candidates.columns:
        selected_features = []
    else:
        selected_features = candidates.loc[candidates["selected"], "feature"].tolist()
    available_features = [feature for feature in selected_features if feature in samples.columns]
    columns = [column for column in SAMPLE_METADATA_COLUMNS if column in samples.columns] + available_features
    return samples.loc[:, columns].copy()


def build_dataset_report(
    feature_panel: pd.DataFrame,
    events: pd.DataFrame,
    event_observations: Optional[pd.DataFrame],
    candidate_feature_effects: Optional[pd.DataFrame] = None,
    horizon: int = DEFAULT_HORIZON,
    min_event_confidence: float = 0.0,
    train_end: Union[str, pd.Timestamp] = DEFAULT_TRAIN_END,
    validation_start: Union[str, pd.Timestamp] = DEFAULT_VALIDATION_START,
    validation_end: Union[str, pd.Timestamp] = DEFAULT_VALIDATION_END,
    test_start: Union[str, pd.Timestamp] = DEFAULT_TEST_START,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    max_candidates: int = 30,
    correlation_threshold: float = 0.85,
    show_progress: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Build labelled samples, leakage-controlled splits and candidate reports."""

    samples = build_future_trend_samples(feature_panel, events, horizon, min_event_confidence, show_progress)
    split_samples = split_time_series_samples(
        samples, train_end, validation_start, validation_end, test_start, embargo_days, show_progress
    )
    if candidate_feature_effects is not None and not candidate_feature_effects.empty:
        candidate_detail, event_candidates = select_candidate_features_from_effects(candidate_feature_effects, show_progress=show_progress)
    elif event_observations is not None:
        candidate_detail, event_candidates = select_event_study_candidates(event_observations, train_end, show_progress=show_progress)
    else:
        raise ValueError("Need raw event observations or compact training_feature_effects for candidate selection")
    candidates = prune_correlated_candidates(
        event_candidates,
        split_samples[split_samples["split"] == "train"],
        max_candidates=max_candidates,
        correlation_threshold=correlation_threshold,
        show_progress=show_progress,
    )
    split_summary = (
        split_samples.groupby(["split", "target"])
        .size()
        .rename("sample_count")
        .reset_index()
        .sort_values(["split", "target"])
    )
    train_samples = split_samples[split_samples["split"] == "train"].copy()
    validation_samples = split_samples[split_samples["split"] == "validation"].copy()
    test_samples = split_samples[split_samples["split"] == "test"].copy()
    return {
        "all_samples": split_samples,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "test_samples": test_samples,
        "train_model_samples": apply_candidate_feature_set(train_samples, candidates),
        "validation_model_samples": apply_candidate_feature_set(validation_samples, candidates),
        "test_model_samples": apply_candidate_feature_set(test_samples, candidates),
        "excluded_samples": split_samples[split_samples["split"] == "excluded"].copy(),
        "split_summary": split_summary,
        "candidate_feature_detail": candidate_detail,
        "event_candidates": event_candidates,
        "candidate_features": candidates,
    }


def export_dataset_report(
    report: Mapping[str, pd.DataFrame],
    output_dir: Union[str, Path],
    run_id: Optional[str] = None,
    metadata: Optional[Mapping[str, object]] = None,
    show_progress: bool = False,
) -> Dict[str, Path]:
    """Write model-data tables and a manifest with immutable split settings."""

    identifier = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    names = {
        "all_samples": "trend_probability_all_samples",
        "train_samples": "trend_probability_train_samples",
        "validation_samples": "trend_probability_validation_samples",
        "test_samples": "trend_probability_test_samples",
        "train_model_samples": "trend_probability_train_model_samples",
        "validation_model_samples": "trend_probability_validation_model_samples",
        "test_model_samples": "trend_probability_test_model_samples",
        "excluded_samples": "trend_probability_excluded_samples",
        "split_summary": "trend_probability_split_summary",
        "candidate_feature_detail": "trend_probability_candidate_feature_detail",
        "event_candidates": "trend_probability_event_candidates",
        "candidate_features": "trend_probability_candidate_features",
    }
    paths = {}
    for key, prefix in tqdm(
        names.items(),
        total=len(names),
        desc="导出概率模型数据集",
        unit="文件",
        disable=not show_progress,
    ):
        path = destination / "{}_{}.csv".format(prefix, identifier)
        report.get(key, pd.DataFrame()).to_csv(path, index=False, encoding="utf-8-sig")
        paths[key] = path
    manifest = {
        "analysis_id": identifier,
        "created_at": datetime.now().isoformat(),
        "sample_metadata_columns": list(SAMPLE_METADATA_COLUMNS),
        "candidate_feature_count": int(report.get("candidate_features", pd.DataFrame()).get("selected", pd.Series(dtype=bool)).sum()),
        "selected_features": report.get("candidate_features", pd.DataFrame()).loc[
            lambda data: data.get("selected", pd.Series(False, index=data.index)), "feature"
        ].tolist() if not report.get("candidate_features", pd.DataFrame()).empty else [],
        "paths": {key: str(value) for key, value in paths.items()},
    }
    if metadata:
        manifest.update(dict(metadata))
    manifest_path = destination / "trend_probability_dataset_manifest_{}.json".format(identifier)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["manifest"] = manifest_path
    return paths


def _combine_compact_summaries(summary_frames: Sequence[pd.DataFrame], show_progress: bool = False) -> pd.DataFrame:
    """Pool batch-level means and standard deviations before deriving effects."""

    frames = [frame for frame in summary_frames if frame is not None and not frame.empty]
    columns = ["transition", "sample_group", "feature", "relative_day", "count", "mean", "median", "q25", "q75", "std"]
    if not frames:
        return pd.DataFrame(columns=columns)
    combined = pd.concat(frames, ignore_index=True)
    rows = []
    summary_groups = combined.groupby(["transition", "sample_group", "feature", "relative_day"], sort=False)
    for keys, group in tqdm(
        summary_groups,
        total=summary_groups.ngroups,
        desc="合并批次训练期汇总",
        unit="统计组",
        disable=not show_progress,
    ):
        counts = np.asarray(group["count"], dtype=float)
        means = np.asarray(group["mean"], dtype=float)
        stds = np.asarray(group["std"], dtype=float)
        total_count = int(np.sum(counts))
        mean = float(np.sum(counts * means) / total_count)
        stds = np.where(np.isfinite(stds), stds, 0.0)
        within_sum_squares = np.sum(np.maximum(counts - 1, 0) * stds ** 2)
        between_sum_squares = float(np.sum(counts * (means - mean) ** 2))
        std = float(np.sqrt((within_sum_squares + between_sum_squares) / (total_count - 1))) if total_count > 1 else np.nan
        rows.append(
            {
                "transition": keys[0],
                "sample_group": keys[1],
                "feature": keys[2],
                "relative_day": keys[3],
                "count": total_count,
                "mean": mean,
                "median": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "std": std,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _load_study_tables(
    output_dir: Union[str, Path], study_run_ids: Union[str, Sequence[str]], show_progress: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load and combine one or more independently exported feature-study batches."""

    root = Path(output_dir)
    run_ids = [study_run_ids] if isinstance(study_run_ids, str) else list(study_run_ids)
    paths = []
    for study_run_id in tqdm(
        run_ids,
        desc="检查事件研究批次",
        unit="批次",
        disable=not show_progress,
    ):
        paths.extend(
            [
                root / "trend_feature_panel_{}.csv".format(study_run_id),
                root / "trend_turning_point_events_{}.csv".format(study_run_id),
            ]
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing feature-event study exports: {}".format(missing))
    compact_summary_paths = [root / "trend_feature_training_summary_{}.csv".format(study_run_id) for study_run_id in run_ids]
    compact_paths = [root / "trend_feature_training_effects_{}.csv".format(study_run_id) for study_run_id in run_ids]
    observation_paths = [root / "trend_feature_event_observations_{}.csv".format(study_run_id) for study_run_id in run_ids]
    use_compact_summary = all(path.is_file() for path in compact_summary_paths)
    use_compact = all(path.is_file() for path in compact_paths)
    use_observations = all(path.is_file() for path in observation_paths)
    if not use_compact_summary and not use_compact and not use_observations:
        raise FileNotFoundError(
            "Missing compact training effects and legacy raw observations for study IDs: {}".format(run_ids)
        )
    panels = []
    events = []
    observations = []
    compact_effects = []
    compact_summaries = []
    for study_run_id in tqdm(
        run_ids,
        desc="读取事件研究批次",
        unit="批次",
        disable=not show_progress,
    ):
        panels.append(pd.read_csv(root / "trend_feature_panel_{}.csv".format(study_run_id)))
        events.append(pd.read_csv(root / "trend_turning_point_events_{}.csv".format(study_run_id)))
        if use_compact_summary:
            compact_summaries.append(pd.read_csv(root / "trend_feature_training_summary_{}.csv".format(study_run_id)))
        elif use_compact:
            compact_effects.append(pd.read_csv(root / "trend_feature_training_effects_{}.csv".format(study_run_id)))
        else:
            observations.append(pd.read_csv(root / "trend_feature_event_observations_{}.csv".format(study_run_id)))
    observation_frame = pd.concat(observations, ignore_index=True) if observations else None
    if compact_summaries:
        compact_frame = _effects_from_summary(_combine_compact_summaries(compact_summaries, show_progress))
    else:
        compact_frame = pd.concat(compact_effects, ignore_index=True) if compact_effects else None
    return pd.concat(panels, ignore_index=True), pd.concat(events, ignore_index=True), observation_frame, compact_frame


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare time-split samples for the trend probability model.")
    parser.add_argument("--study-run-id", required=True, nargs="+", help="One or more prior feature_event_study export IDs")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--min-event-confidence", type=float, default=0.0)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--validation-start", default=DEFAULT_VALIDATION_START)
    parser.add_argument("--validation-end", default=DEFAULT_VALIDATION_END)
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--embargo-days", type=int, default=DEFAULT_EMBARGO_DAYS)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--correlation-threshold", type=float, default=0.85)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    args = parser.parse_args(argv)
    show_progress = not args.no_progress
    panel, events, observations, compact_effects = _load_study_tables(args.output_dir, args.study_run_id, show_progress)
    report = build_dataset_report(
        panel,
        events,
        observations,
        candidate_feature_effects=compact_effects,
        horizon=args.horizon,
        min_event_confidence=args.min_event_confidence,
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.test_start,
        embargo_days=args.embargo_days,
        max_candidates=args.max_candidates,
        correlation_threshold=args.correlation_threshold,
        show_progress=show_progress,
    )
    metadata = {
        "source_study_run_ids": args.study_run_id,
        "horizon": args.horizon,
        "min_event_confidence": args.min_event_confidence,
        "train_end": args.train_end,
        "validation_start": args.validation_start,
        "validation_end": args.validation_end,
        "test_start": args.test_start,
        "embargo_days": args.embargo_days,
        "max_candidates": args.max_candidates,
        "correlation_threshold": args.correlation_threshold,
        "candidate_selection_period": "events through {} only".format(args.train_end),
    }
    paths = export_dataset_report(report, args.output_dir, args.run_id, metadata, show_progress)
    print("Prepared train/validation/test samples: {}/{}/{}".format(
        len(report["train_samples"]), len(report["validation_samples"]), len(report["test_samples"])
    ))
    selected_count = int(report["candidate_features"]["selected"].sum()) if not report["candidate_features"].empty else 0
    print("Selected {} candidate features. Manifest: {}".format(selected_count, paths["manifest"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
