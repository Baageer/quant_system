import numpy as np
import pandas as pd

from research.market_regime.logistic_model import infer_feature_columns
from research.market_regime.market_environment import (
    build_causal_market_features,
    merge_market_features,
    stratify_probability_predictions,
)


def make_benchmark(size=180):
    dates = pd.date_range("2020-01-01", periods=size, freq="B")
    close = pd.Series(np.linspace(100.0, 160.0, size), index=dates)
    return pd.DataFrame({"close": close, "high": close + 1.0, "low": close - 1.0})


def test_market_features_are_causal_and_merge_by_date():
    benchmark = make_benchmark()
    full = build_causal_market_features(benchmark)
    prefix = build_causal_market_features(benchmark.iloc[:-15])

    pd.testing.assert_frame_equal(full.loc[prefix.index], prefix)
    assert full.loc[full.index[-1], "market_state"] == "risk_on"
    panel = pd.DataFrame(
        {"feature": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples(
            [(full.index[-2], "AAA"), (full.index[-1], "BBB")], names=["date", "symbol"]
        ),
    )
    merged = merge_market_features(panel, full)
    assert {"mkt_ret_20d", "market_state"}.issubset(merged.columns)
    assert merged.loc[(full.index[-1], "BBB"), "market_state"] == "risk_on"


def test_market_state_is_metadata_and_supports_stratified_metrics():
    samples = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=6, freq="B"),
            "symbol": ["AAA"] * 6,
            "target": ["down", "none", "up", "down", "none", "up"],
            "sample_weight": np.ones(6),
            "market_state": ["risk_on"] * 3 + ["risk_off"] * 3,
            "feature": np.arange(6, dtype=float),
        }
    )
    predictions = samples.loc[:, ["date", "symbol"]].copy()
    for column, values in zip(
        ("p_down", "p_none", "p_up"),
        np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]] * 2).T,
    ):
        predictions[column] = values

    def evaluator(targets, probabilities, weights, split, model):
        return {
            "split": split,
            "model": model,
            "sample_count": len(targets),
            "accuracy": float(np.mean(probabilities.max(axis=1))),
        }

    report = stratify_probability_predictions(predictions, samples, evaluator)
    assert set(report["market_state"]) == {"risk_on", "risk_off"}
    assert infer_feature_columns(samples) == ["feature"]
