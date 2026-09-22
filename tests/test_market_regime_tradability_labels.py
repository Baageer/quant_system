import pandas as pd

from research.market_regime.tradability_labels import (
    add_auxiliary_labels,
    add_tradability_target,
    fit_tradability_thresholds,
    mark_auxiliary_label_eligibility,
    normalize_a_share_symbols,
)


def make_samples():
    dates = pd.date_range("2020-01-01", periods=8, freq="B")
    return pd.DataFrame(
        {
            "date": [dates[0], dates[1], dates[2], dates[3]],
            "symbol": ["AAA"] * 4,
            "target": ["up", "up", "none", "up"],
            "source_split": ["train", "train", "validation", "test"],
        }
    )


def make_prices():
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    return {"AAA": pd.DataFrame({"close": range(100, 130)}, index=dates)}


def test_auxiliary_labels_use_next_trading_session_for_entry():
    labeled = add_auxiliary_labels(make_samples(), make_prices(), horizons=(2,), volatility_window=2)
    dates = pd.date_range("2020-01-01", periods=30, freq="B")

    assert labeled.loc[0, "entry_date_2d"] == dates[1]
    assert labeled.loc[0, "exit_date_2d"] == dates[3]
    assert labeled.loc[0, "future_return_2d"] == (103 / 101) - 1
    assert labeled.loc[0, "future_mae_2d"] == 0.0
    assert labeled.loc[0, "auxiliary_label_available_date"] == dates[3]


def test_symbol_normalization_restores_six_digit_a_share_codes():
    symbols = normalize_a_share_symbols(pd.Series(["1", "600000", "AAA"]))

    assert symbols.tolist() == ["000001", "600000", "AAA"]


def test_auxiliary_eligibility_purges_cross_boundary_windows():
    samples = make_samples()
    samples["auxiliary_label_available_date"] = pd.to_datetime(
        ["2020-01-03", "2020-01-08", "2020-01-06", "2020-01-07"]
    )
    manifest = {"validation_start": "2020-01-06", "test_start": "2020-01-07"}

    marked = mark_auxiliary_label_eligibility(samples, manifest)

    assert marked["auxiliary_label_eligible"].tolist() == [True, False, True, True]
    assert marked.loc[1, "auxiliary_label_reason"] == "purged_auxiliary_window"


def test_tradability_thresholds_fit_train_only_and_create_secondary_target():
    samples = make_samples()
    samples["auxiliary_label_eligible"] = True
    samples["future_risk_adjusted_return_10d"] = [2.0, 4.0, 99.0, 99.0]
    samples["future_excess_return_10d"] = [0.1, 0.2, 99.0, 99.0]
    samples["future_mae_10d"] = [-0.04, -0.01, -0.01, -0.01]

    thresholds = fit_tradability_thresholds(samples)
    labeled = add_tradability_target(samples, thresholds)

    assert thresholds["risk_adjusted_return_threshold"] == 3.0
    assert labeled.loc[1, "tradability_target_10d"] == "tradable_up"
    assert labeled.loc[2, "tradability_target_10d"] == "not_tradable"
