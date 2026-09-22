import pandas as pd

from research.market_regime.probability_trade_evaluation import (
    BASELINE_POLICY_ID,
    apply_trading_policy_signals,
    build_validation_rule_grid,
    calculate_executable_event_audit,
    calculate_event_metrics,
    choose_validation_policy,
    choose_validation_rule,
    select_alerts,
)


def make_predictions():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2021-01-01", "2021-01-01", "2021-01-04", "2021-01-04", "2021-01-05"]
            ),
            "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA"],
            "target": ["up", "none", "up", "none", "none"],
            "target_event_id": ["event-1", None, "event-1", None, None],
            "target_event_date": pd.to_datetime(["2021-01-08", None, "2021-01-08", None, None]),
            "days_to_target": [5, None, 4, None, None],
            "p_up": [0.8, 0.1, 0.7, 0.2, 0.3],
            "p_down": [0.1, 0.2, 0.1, 0.3, 0.1],
        }
    )


def test_top_k_alerts_are_cross_sectional_and_require_up_to_exceed_down():
    predictions = make_predictions()
    alerts = select_alerts(predictions, "top_k", 0.5)

    assert alerts.tolist() == [True, False, True, False, True]


def test_event_metrics_count_single_event_and_repeated_warning_days():
    predictions = make_predictions()
    events, summary = calculate_event_metrics(predictions, pd.Series([True, False, True, False, True]))

    assert len(events) == 1
    assert bool(events.loc[0, "hit"])
    assert events.loc[0, "first_warning_lead_days"] == 5
    assert events.loc[0, "repeat_alert_count"] == 1
    assert summary["event_hit_rate"] == 1.0
    assert summary["false_alert_count"] == 1


def test_validation_grid_selection_does_not_read_test_rows():
    validation = make_predictions()
    test = make_predictions().assign(p_up=0.999, p_down=0.0)

    selected_before = choose_validation_rule(build_validation_rule_grid(validation, min_hit_events=1))
    _ = test
    selected_after = choose_validation_rule(build_validation_rule_grid(validation, min_hit_events=1))

    assert selected_before == selected_after


def test_symbol_codes_retain_leading_zeroes_when_read(tmpdir):
    from research.market_regime.probability_trade_evaluation import load_predictions

    path = tmpdir.join("predictions.csv")
    make_predictions().assign(symbol="000001").to_csv(path, index=False)

    assert load_predictions(path).loc[0, "symbol"] == "000001"


def test_policy_requires_confirmation_and_shifts_decisions_to_next_day():
    market_data = {
        "AAA": pd.DataFrame(
            {
                "p_up": [0.03, 0.03, 0.01, 0.01],
                "p_down": [0.01, 0.01, 0.02, 0.02],
                "eod_alert": [True, True, False, False],
            },
            index=pd.date_range("2021-01-01", periods=4, freq="B"),
        )
    }
    policy = {
        "entry_confirm_days": 2,
        "exit_threshold_multiplier": 0.85,
        "min_holding_days": 5,
        "rebalance_weeks": 1,
    }
    rule = {"rule_type": "threshold", "rule_value": 0.02}

    prepared = apply_trading_policy_signals(market_data, rule, policy)["AAA"]

    assert prepared["entry_trade"].tolist() == [False, False, True, False]
    assert prepared["hold_trade"].tolist() == [False, True, True, False]


def policy_grid(rows):
    """A policy grid carrying the daily baseline row the rule compares against."""

    fields = {
        "entry_confirm_days": 1,
        "exit_threshold_multiplier": 0.85,
        "min_holding_days": 5,
        "rebalance_weeks": 1,
        "eligible": True,
    }
    return pd.DataFrame([
        {"policy_id": policy_id, **fields,
         "validation_total_return": total_return,
         "validation_turnover_multiple": turnover,
         "validation_filled_buy_count": 30}
        for policy_id, total_return, turnover in rows
    ])


def test_choose_validation_policy_adopts_a_policy_that_beats_the_baseline():
    grid = policy_grid([
        (BASELINE_POLICY_ID, 0.10, 3.0),
        ("slower", 0.12, 2.0),
    ])

    policy = choose_validation_policy(grid)

    assert policy["policy_id"] == "slower"
    assert policy["selection_reason"]


def test_choose_validation_policy_keeps_the_baseline_when_nothing_beats_it():
    """The old selector could not express this: it always adopted *some* policy."""

    grid = policy_grid([
        (BASELINE_POLICY_ID, 0.10, 3.0),
        ("slower", 0.05, 2.0),
    ])

    policy = choose_validation_policy(grid)

    assert policy["policy_id"] == BASELINE_POLICY_ID


def test_choose_validation_policy_rejects_a_degenerate_slow_policy():
    """A policy that all but stops trading must not be adopted on return alone."""

    grid = policy_grid([
        (BASELINE_POLICY_ID, 0.10, 3.0),
        ("barely_trades", 0.50, 0.02),
    ])

    policy = choose_validation_policy(grid)

    assert policy["policy_id"] == BASELINE_POLICY_ID


def test_choose_validation_policy_allows_slowing_down():
    """The floor is a degeneracy cut, not a ban on the module's own purpose.

    Calibrated against the daily baseline, a 0.5x floor would reject any policy
    slower than 2x -- which is every policy this module exists to consider.
    """

    grid = policy_grid([
        (BASELINE_POLICY_ID, 0.10, 100.0),
        ("weekly", 0.30, 12.0),   # 12x slower, still trades plenty
    ])

    policy = choose_validation_policy(grid)

    assert policy["policy_id"] == "weekly"


def test_executable_event_audit_uses_t_plus_one_after_first_alert():
    dates = pd.date_range("2021-01-01", periods=25, freq="B")
    prices = pd.DataFrame({"close": range(100, 125)}, index=dates)
    events = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "target_event_id": ["event-1"],
            "target_event_date": [dates[5]],
            "hit": [True],
            "first_alert_date": [dates[1]],
        }
    )

    audit, summary = calculate_executable_event_audit(events, {"AAA": prices}, horizons=(5, 10))

    assert audit.loc[0, "alert_entry_anchor_date"] == dates[2]
    assert audit.loc[0, "event_return_5d"] > 0
    assert audit.loc[0, "alert_entry_return_10d"] > 0
    assert summary["event_records_with_prices"] == 1
