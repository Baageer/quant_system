import numpy as np
import pandas as pd

from research.market_regime.feature_ablation import run_feature_ablation, select_feature_columns


def make_samples(seed, size=150):
    random = np.random.RandomState(seed)
    signal_feature = random.normal(size=size)
    noise_feature = random.normal(size=size)
    target = np.where(signal_feature > 0.65, "up", np.where(signal_feature < -0.65, "down", "none"))
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=size, freq="B"),
            "symbol": ["AAA"] * size,
            "target": target,
            "sample_weight": np.ones(size),
            "signal_feature": signal_feature,
            "noise_feature": noise_feature,
        }
    )


def test_select_feature_columns_preserves_expected_variants():
    feature_columns = ["first", "second", "third"]

    assert select_feature_columns(feature_columns, "full") == feature_columns
    assert select_feature_columns(feature_columns, "single_feature", "second") == ["second"]
    assert select_feature_columns(feature_columns, "leave_one_out", "second") == ["first", "third"]


def test_feature_ablation_reports_full_single_and_leave_one_out_results():
    results = run_feature_ablation(
        make_samples(1),
        make_samples(2),
        make_samples(3),
        modes=("single_feature", "leave_one_out"),
        features=("signal_feature",),
        c_values=(0.1, 1.0),
        calibration="none",
    )

    assert list(results["analysis_type"]) == ["full", "leave_one_out", "single_feature"]
    assert list(results["feature_count"]) == [2, 1, 1]
    assert results.loc[0, "feature"] is None
    assert results.loc[1, "feature"] == "signal_feature"
    assert results.loc[2, "feature"] == "signal_feature"
    assert "test_delta_vs_full_trend_macro_pr_auc" in results.columns
    assert "test_delta_vs_base_rate_trend_macro_pr_auc" in results.columns
    assert results.loc[0, "test_delta_vs_full_trend_macro_pr_auc"] == 0.0
