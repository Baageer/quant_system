import numpy as np
import pandas as pd

from research.market_regime.trade_outcome_model import (
    decision_score,
    train_trade_outcome_experiment,
)
from research.market_regime.trade_outcome_ranking_evaluation import outcome_quality


def make_samples(seed, date_count=24, symbols_per_date=4):
    random = np.random.RandomState(seed)
    count = date_count * symbols_per_date
    feature_one = random.normal(size=count)
    feature_two = random.normal(size=count)
    excess_return = 0.02 * feature_one - 0.01 * feature_two + random.normal(scale=0.002, size=count)
    mae = -0.03 - 0.01 * np.maximum(-feature_one, 0) + random.normal(scale=0.001, size=count)
    return pd.DataFrame(
        {
            "date": np.repeat(pd.date_range("2020-01-01", periods=date_count, freq="B"), symbols_per_date),
            "symbol": ["{:06d}".format(index % symbols_per_date + 1) for index in range(count)],
            "future_excess_return_10d": excess_return,
            "future_mae_10d": mae,
            "feature_one": feature_one,
            "feature_two": feature_two,
        }
    )


def test_decision_score_penalizes_predicted_downside_only():
    scores = decision_score(np.array([0.03, 0.03]), np.array([-0.02, 0.01]), risk_penalty=0.5)

    assert np.allclose(scores, [0.02, 0.03])


def test_trade_outcome_training_outputs_direct_t_plus_one_rank_score():
    report = train_trade_outcome_experiment(
        make_samples(1),
        make_samples(2),
        make_samples(3),
        feature_columns=["feature_one", "feature_two"],
        horizon=10,
        alpha_values=(0.1, 1.0),
        risk_penalties=(0.0, 0.5),
        top_k=0.5,
    )

    predictions = report["test_predictions"]
    assert report["selected_alpha"] in {0.1, 1.0}
    assert report["selected_risk_penalty"] in {0.0, 0.5}
    assert {"predicted_excess_return_10d", "predicted_mae_10d", "entry_score"}.issubset(predictions.columns)
    assert predictions["entry_score"].notnull().all()
    assert report["metrics"].set_index("split").loc["test", "selected_count"] == 48


def test_outcome_quality_compares_selected_rows_with_the_daily_universe():
    scores = pd.DataFrame(
        {
            "future_excess_return_10d": [0.03, -0.01, 0.02, 0.00],
            "future_mae_10d": [-0.01, -0.05, -0.02, -0.03],
        }
    )

    quality = outcome_quality(scores, pd.Series([True, False, True, False]), horizon=10)

    assert quality["selected_count"] == 2
    assert quality["selected_mean_excess_return"] == 0.025
    assert np.isclose(quality["selected_excess_over_universe"], 0.015)
