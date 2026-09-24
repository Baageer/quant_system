import numpy as np
import pandas as pd
import pytest

from research.market_regime.probability_dataset import (
    _combine_compact_summaries,
    apply_candidate_feature_set,
    build_future_trend_samples,
    prune_correlated_candidates,
    select_event_study_candidates,
    split_time_series_samples,
)


def make_panel(periods=40):
    dates = pd.date_range("2018-12-03", periods=periods, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": ["AAA"] * periods,
            "feature_a": np.arange(periods, dtype=float),
            "feature_b": np.arange(periods, dtype=float) / 10,
        }
    )


def make_events(panel):
    dates = panel["date"]
    return pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "event_date": [dates.iloc[10], dates.iloc[26]],
            "next_regime": ["up", "down"],
            "event_id": ["up-event", "down-event"],
            "pivot_confidence": [0.8, 0.8],
        }
    )


def test_future_trend_labels_first_event_and_event_weights():
    panel = make_panel()
    samples = build_future_trend_samples(panel, make_events(panel), horizon=3)

    assert samples.loc[7:9, "target"].tolist() == ["up", "up", "up"]
    assert samples.loc[7:9, "target_event_id"].tolist() == ["up-event"] * 3
    assert samples.loc[7:9, "sample_weight"].sum() == 1.0
    assert samples.loc[37:39, "target"].tolist() == ["unavailable"] * 3


def test_time_split_purges_label_window_and_embargoes_boundary():
    panel = make_panel()
    samples = build_future_trend_samples(panel, make_events(panel), horizon=3)
    split = split_time_series_samples(
        samples,
        train_end="2018-12-21",
        validation_start="2018-12-24",
        validation_end="2019-01-11",
        test_start="2019-01-14",
        embargo_days=2,
    )

    train = split[split["split"] == "train"]
    validation = split[split["split"] == "validation"]
    test = split[split["split"] == "test"]
    assert (train["label_available_date"] < pd.Timestamp("2018-12-24")).all()
    assert validation["date"].min() >= pd.Timestamp("2018-12-26")
    assert (validation["label_available_date"] < pd.Timestamp("2019-01-14")).all()
    assert test["date"].min() >= pd.Timestamp("2019-01-16")
    assert "purged_label_window" in set(split["split_reason"])


def test_candidate_selection_uses_pre_event_training_observations_only():
    rows = []
    for sample in range(12):
        for relative_day in range(-5, 0):
            rows.extend(
                [
                    {"event_date": "2018-12-20", "transition": "down_to_up", "sample_group": "event", "feature": "strong", "relative_day": relative_day, "value": 2.0 + sample / 10},
                    {"event_date": "2018-12-20", "transition": "down_to_up", "sample_group": "control", "feature": "strong", "relative_day": relative_day, "value": sample / 20},
                    {"event_date": "2018-12-20", "transition": "down_to_up", "sample_group": "event", "feature": "weak", "relative_day": relative_day, "value": 1.0 + sample / 10},
                    {"event_date": "2018-12-20", "transition": "down_to_up", "sample_group": "control", "feature": "weak", "relative_day": relative_day, "value": 1.0 + sample / 10},
                ]
            )
    observations = pd.DataFrame(rows)
    detail, candidates = select_event_study_candidates(
        observations,
        selection_end="2018-12-31",
        min_samples_per_day=10,
        min_relative_days=5,
        min_abs_effect=0.2,
    )

    assert detail.loc[detail["feature"] == "strong", "selected"].iloc[0]
    assert "strong" in set(candidates["feature"])
    assert "weak" not in set(candidates["feature"])


def test_candidate_pruning_uses_training_correlation_and_limit():
    candidates = pd.DataFrame(
        {
            "feature": ["first", "duplicate", "independent"],
            "best_abs_standardized_effect": [1.0, 0.9, 0.8],
            "selected": [True, True, True],
        }
    )
    train_samples = pd.DataFrame(
        {
            "first": [1.0, 2.0, 3.0, 4.0],
            "duplicate": [2.0, 4.0, 6.0, 8.0],
            "independent": [1.0, 3.0, 2.0, 5.0],
        }
    )

    result = prune_correlated_candidates(candidates, train_samples, max_candidates=2, correlation_threshold=0.85)

    assert result.loc[result["feature"] == "first", "selected"].iloc[0]
    assert not result.loc[result["feature"] == "duplicate", "selected"].iloc[0]
    assert result.loc[result["feature"] == "independent", "selected"].iloc[0]


def test_model_sample_table_keeps_only_selected_features_and_metadata():
    samples = build_future_trend_samples(make_panel(), make_events(make_panel()), horizon=3)
    candidates = pd.DataFrame({"feature": ["feature_a", "feature_b"], "selected": [True, False]})

    model_samples = apply_candidate_feature_set(samples, candidates)

    assert "feature_a" in model_samples.columns
    assert "feature_b" not in model_samples.columns
    assert {"target", "sample_weight", "label_available_date"}.issubset(model_samples.columns)


def test_compact_batch_summaries_pool_counts_means_and_variance():
    first = pd.DataFrame(
        {
            "transition": ["down_to_up"], "sample_group": ["event"], "feature": ["feature"], "relative_day": [-1],
            "count": [2], "mean": [2.0], "median": [2.0], "q25": [1.5], "q75": [2.5], "std": [1.0],
        }
    )
    second = first.copy()
    second["mean"] = 6.0

    combined = _combine_compact_summaries([first, second])

    assert combined.iloc[0]["count"] == 4
    assert combined.iloc[0]["mean"] == 4.0
    assert combined.iloc[0]["std"] > 2.0


def _split_with_train_start(train_start):
    panel = make_panel()
    samples = build_future_trend_samples(panel, make_events(panel), horizon=3)
    return split_time_series_samples(
        samples,
        train_end="2018-12-21",
        validation_start="2018-12-24",
        validation_end="2019-01-11",
        test_start="2019-01-14",
        embargo_days=2,
        train_start=train_start,
    )


def test_train_start_defaults_to_an_unbounded_window():
    """``None`` has to leave the older three-way split byte for byte identical."""

    default = _split_with_train_start(None)
    explicit = _split_with_train_start(pd.Timestamp("1900-01-01"))

    assert default.equals(explicit)
    assert default["split"].value_counts().to_dict() == explicit["split"].value_counts().to_dict()


def test_train_start_truncates_only_the_front_of_train():
    unbounded = _split_with_train_start(None)
    truncated = _split_with_train_start("2018-12-10")

    train = truncated[truncated["split"] == "train"]
    assert len(train) < len(unbounded[unbounded["split"] == "train"])
    assert train["date"].min() >= pd.Timestamp("2018-12-10")
    # Validation and test are downstream of the training window, so narrowing
    # train's *left* edge must not move a single one of their rows.
    for split_name in ("validation", "test"):
        left = unbounded[unbounded["split"] == split_name].reset_index(drop=True)
        right = truncated[truncated["split"] == split_name].reset_index(drop=True)
        assert left.equals(right)


def test_train_start_after_train_end_is_rejected():
    with pytest.raises(ValueError):
        _split_with_train_start("2019-06-01")
