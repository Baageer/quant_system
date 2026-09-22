import numpy as np
import pandas as pd

from research.market_regime.logistic_model import (
    CLASSES,
    fit_preprocessor,
    train_logistic_experiment,
    transform_features,
)


def make_samples(seed, size=150):
    random = np.random.RandomState(seed)
    feature_one = random.normal(size=size)
    feature_two = random.normal(size=size)
    target = np.where(feature_one > 0.65, "up", np.where(feature_one < -0.65, "down", "none"))
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=size, freq="B"),
            "symbol": ["AAA"] * size,
            "target": target,
            "sample_weight": np.ones(size),
            "feature_one": feature_one,
            "feature_two": feature_two,
        }
    )


def test_preprocessor_uses_training_median_and_scale_only():
    train = make_samples(1, size=30)
    train.loc[0, "feature_two"] = np.nan
    test = make_samples(2, size=5)
    test.loc[0, "feature_one"] = np.nan

    preprocessor = fit_preprocessor(train, ["feature_one", "feature_two"])
    transformed = transform_features(test, preprocessor)

    assert transformed.shape == (5, 2)
    assert np.isfinite(transformed).all()
    assert preprocessor["medians"][1] == np.nanmedian(train["feature_two"])


def test_logistic_experiment_selects_validation_c_and_outputs_probabilities():
    report = train_logistic_experiment(
        make_samples(1),
        make_samples(2),
        make_samples(3),
        c_values=(0.1, 1.0),
        calibration="multinomial",
    )

    assert report["selected_c"] in {0.1, 1.0}
    assert set(report["test_predictions"]["predicted_target"]).issubset(CLASSES)
    probabilities = np.asarray(report["test_predictions"][["p_down", "p_none", "p_up"]], dtype=float)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert {"base_rate", "logistic_raw", "logistic_calibrated"}.issubset(set(report["metrics"]["model"]))
    assert len(report["coefficients"]) == len(CLASSES) * 2
