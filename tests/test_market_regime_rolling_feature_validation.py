import numpy as np
import pandas as pd

from research.market_regime.rolling_feature_validation import (
    build_rolling_folds,
    build_validation_frozen_candidates,
    run_rolling_feature_validation,
)


def make_ablation_metrics():
    return pd.DataFrame(
        [
            {"analysis_type": "full", "feature": None, "validation_trend_macro_pr_auc": 0.20, "validation_log_loss": 0.50},
            {"analysis_type": "single_feature", "feature": "alpha", "validation_trend_macro_pr_auc": 0.30, "validation_log_loss": 0.60},
            {"analysis_type": "single_feature", "feature": "beta", "validation_trend_macro_pr_auc": 0.25, "validation_log_loss": 0.40},
            {"analysis_type": "single_feature", "feature": "gamma", "validation_trend_macro_pr_auc": 0.10, "validation_log_loss": 0.30},
            {"analysis_type": "leave_one_out", "feature": "alpha", "validation_trend_macro_pr_auc": 0.19, "validation_log_loss": 0.55},
            {"analysis_type": "leave_one_out", "feature": "beta", "validation_trend_macro_pr_auc": 0.22, "validation_log_loss": 0.49},
            {"analysis_type": "leave_one_out", "feature": "gamma", "validation_trend_macro_pr_auc": 0.21, "validation_log_loss": 0.50},
        ]
    )


def make_samples(start, size=90):
    random = np.random.RandomState(start)
    alpha = random.normal(size=size)
    beta = random.normal(size=size)
    positions = np.arange(size) % 3
    target = np.asarray(["down", "none", "up"])[positions]
    dates = pd.date_range("2020-01-01", periods=size, freq="B") + pd.offsets.BDay(start)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": ["AAA"] * size,
            "target": target,
            "sample_weight": np.ones(size),
            "label_available_date": dates + pd.offsets.BDay(1),
            "alpha": alpha,
            "beta": beta,
        }
    )


def test_validation_frozen_candidates_use_validation_metrics_only():
    candidates, table = build_validation_frozen_candidates(make_ablation_metrics(), ["alpha", "beta", "gamma"], core_count=2)

    assert candidates["full"] == ["alpha", "beta", "gamma"]
    assert candidates["core_2"] == ["alpha", "beta"]
    assert candidates["validation_pruned"] == ["alpha"]
    assert set(table["candidate"]) == {"full", "core_2", "validation_pruned"}


def test_rolling_folds_and_models_follow_time_order():
    train = make_samples(0)
    validation = make_samples(90)
    test = make_samples(180)
    all_samples = pd.concat([train, validation, test], ignore_index=True)
    folds = build_rolling_folds(all_samples, test, calibration_days=20, test_days=20, min_test_days=20)
    assert folds[0]["calibration_end"] < folds[0]["test_start"]

    results = run_rolling_feature_validation(
        train,
        validation,
        test,
        {"full": ["alpha", "beta"], "core_1": ["alpha"]},
        calibration_days=20,
        test_days=20,
        min_test_days=20,
        c_values=(0.1,),
        calibration="none",
        candidate_names=("core_1",),
        fold_numbers=(1,),
    )

    assert set(results["candidate"]) == {"core_1"}
    assert set(results["fold"]) == {1}
    assert (pd.to_datetime(results["calibration_end"]) < pd.to_datetime(results["test_start"])).all()
    assert (results["train_sample_count"] > 0).all()
