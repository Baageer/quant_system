import numpy as np
import pandas as pd

from research.market_regime.tradability_model import (
    TARGET_COLUMN,
    combine_entry_scores,
    train_tradability_experiment,
)


def make_samples(seed, size=60):
    random = np.random.RandomState(seed)
    feature_one = random.normal(size=size)
    feature_two = random.normal(size=size)
    target = np.where(feature_one + feature_two > 1.0, "tradable_up", "not_tradable")
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=size, freq="B"),
            "symbol": ["000001"] * size,
            "target": np.where(target == "tradable_up", "up", "none"),
            TARGET_COLUMN: target,
            "sample_weight": np.ones(size),
            "feature_one": feature_one,
            "feature_two": feature_two,
        }
    )


def test_binary_tradability_training_outputs_calibrated_probabilities():
    report = train_tradability_experiment(
        make_samples(1), make_samples(2), make_samples(3), ["feature_one", "feature_two"], c_values=(0.1, 1.0)
    )

    assert report["selected_c"] in {0.1, 1.0}
    assert report["test_predictions"]["p_tradable"].between(0, 1).all()
    assert {"tradability_raw", "tradability_calibrated"}.issubset(set(report["metrics"]["model"]))


def test_entry_score_uses_p_up_times_p_tradable_after_symbol_normalization():
    tradability = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "symbol": ["000001"],
            "target": ["up"],
            TARGET_COLUMN: ["tradable_up"],
            "sample_weight": [1.0],
            "p_tradable": [0.4],
        }
    )
    trend = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "symbol": ["000001"],
            "p_up": [0.25],
            "p_down": [0.1],
            "p_none": [0.65],
        }
    )

    combined, metrics = combine_entry_scores(tradability, trend, "test")

    assert combined.loc[0, "entry_score"] == 0.1
    assert metrics["matched_prediction_count"] == 1
