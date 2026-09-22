import contextlib

import numpy as np
import pandas as pd

from research.market_regime.passive_benchmark import (
    INDEX_ARM_NAME,
    align_levels,
    append_index_arm,
    benchmark_comparison,
    index_arm_row,
    index_window_return,
)


@contextlib.contextmanager
def raises(exception, match=None):
    """Minimal stand-in for ``pytest.raises`` -- pytest is not installed here."""

    try:
        yield
    except exception as exc:
        if match is not None and match not in str(exc):
            raise AssertionError("{} not found in {!r}".format(match, str(exc)))
    else:
        raise AssertionError("{} was not raised".format(exception.__name__))


def make_levels(dates, values):
    return pd.Series([float(value) for value in values], index=pd.DatetimeIndex(dates))


SESSIONS = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
LEVELS = make_levels(SESSIONS, [3000.0, 3030.0, 2970.0, 3300.0])


def test_index_arm_row_is_the_index_return_times_the_capital_base():
    row = index_arm_row(LEVELS, SESSIONS, initial_capital=500000.0)

    assert row["strategy"] == INDEX_ARM_NAME
    assert np.isclose(row["total_return"], 3300.0 / 3000.0 - 1.0)
    assert np.isclose(row["final_value"], 500000.0 * 3300.0 / 3000.0)
    assert np.isclose(row["net_profit"], row["final_value"] - 500000.0)


def test_index_arm_row_never_reports_a_trade_or_a_cost():
    row = index_arm_row(LEVELS, SESSIONS, initial_capital=500000.0)

    assert row["filled_trade_count"] == 0
    assert row["filled_buy_count"] == 0
    assert row["rejected_buy_count"] == 0
    assert row["turnover_multiple"] == 0.0
    assert row["commission"] == 0.0
    assert row["stamp_duty"] == 0.0
    assert row["reported_cost"] == 0.0


def test_index_arm_annualization_matches_summarize_backtest_convention():
    """Same formula and same session count as the strategy arms, or the two
    annualized returns are not comparable."""

    row = index_arm_row(LEVELS, SESSIONS, initial_capital=500000.0)

    total = 3300.0 / 3000.0 - 1.0
    assert np.isclose(row["annual_return"], (1.0 + total) ** (252 / len(SESSIONS)) - 1.0)
    returns = pd.Series([3030.0, 2970.0, 3300.0], index=SESSIONS[1:]).pct_change().dropna()
    # pct_change on the level path, taken from the full level series:
    levels = pd.Series([3000.0, 3030.0, 2970.0, 3300.0], index=SESSIONS)
    daily = levels.pct_change().dropna()
    assert np.isclose(row["annual_volatility"], daily.std(ddof=1) * np.sqrt(252))
    assert row["annual_volatility"] > 0
    assert len(returns) == 2


def test_index_arm_drawdown_is_measured_on_the_level_path():
    row = index_arm_row(LEVELS, SESSIONS, initial_capital=500000.0)

    # Peak 3030 -> trough 2970.
    assert np.isclose(row["max_drawdown"], 2970.0 / 3030.0 - 1.0)


def test_index_arm_uses_only_the_sessions_it_is_given():
    """A shorter window must produce a different return, not the full-path one."""

    short = index_arm_row(LEVELS, SESSIONS[:3], initial_capital=500000.0)

    assert np.isclose(short["total_return"], 2970.0 / 3000.0 - 1.0)
    assert short["start_date"] == "2024-01-02"
    assert short["end_date"] == "2024-01-04"


def test_align_levels_rejects_a_window_whose_endpoint_the_index_does_not_reach():
    """The dangerous case: back-filling the tail would invent a flat return."""

    levels = make_levels(SESSIONS[:2], [3000.0, 3030.0])

    with raises(ValueError, match="endpoint"):
        align_levels(levels, SESSIONS)


def test_align_levels_rejects_a_window_with_too_many_interior_gaps():
    sessions = pd.bdate_range("2024-01-02", periods=40)
    levels = make_levels(sessions[:1].append(sessions[10:]), [3000.0] + [3000.0 + index for index in range(30)])

    with raises(ValueError, match="missing 9 of 40"):
        align_levels(levels, sessions)


def test_align_levels_forward_fills_an_isolated_gap():
    levels = make_levels([SESSIONS[0], SESSIONS[2], SESSIONS[3]], [3000.0, 2970.0, 3300.0])

    window = align_levels(levels, SESSIONS)

    assert window.loc[SESSIONS[1]] == 3000.0
    assert len(window) == len(SESSIONS)


def test_align_levels_rejects_an_empty_session_index():
    with raises(ValueError, match="empty"):
        align_levels(LEVELS, pd.DatetimeIndex([]))


def test_index_window_return_is_endpoint_to_endpoint():
    assert np.isclose(index_window_return(LEVELS, "2024-01-03", "2024-01-05"), 3300.0 / 3030.0 - 1.0)


def make_metrics_table():
    return pd.DataFrame([
        {"strategy": "top_k_net", "total_return": 0.40, "max_drawdown": -0.12, "sharpe_zero_rate": 0.9},
        {"strategy": "passive_equal_weight_sample_net", "total_return": 0.05,
         "max_drawdown": -0.26, "sharpe_zero_rate": 0.08},
    ])


def test_append_index_arm_keeps_the_sample_arm_and_adds_the_index():
    equity = pd.DataFrame({"portfolio_value": [500000.0] * len(SESSIONS)}, index=SESSIONS)

    table = append_index_arm(make_metrics_table(), equity, initial_capital=500000.0, levels=LEVELS)

    assert table["strategy"].tolist() == ["top_k_net", "passive_equal_weight_sample_net", INDEX_ARM_NAME]
    assert np.isclose(table.iloc[-1]["total_return"], 3300.0 / 3000.0 - 1.0)


def test_benchmark_comparison_reports_both_excesses_and_the_sample_gap():
    metrics = make_metrics_table()
    metrics = pd.concat([
        metrics,
        pd.DataFrame([{"strategy": INDEX_ARM_NAME, "total_return": 0.07,
                       "max_drawdown": -0.25, "sharpe_zero_rate": 0.11}]),
    ], ignore_index=True)

    comparison = benchmark_comparison(metrics, strategy="top_k_net")

    assert np.isclose(comparison["net_total_return_excess_vs_index"], 0.40 - 0.07)
    assert np.isclose(comparison["net_total_return_excess_vs_sample"], 0.40 - 0.05)
    # The quantity the old benchmark could not see: how wrong the sample was.
    assert np.isclose(comparison["sample_minus_index_gap"], 0.05 - 0.07)


def test_benchmark_comparison_names_the_missing_index_arm():
    with raises(ValueError, match="hs300_index_buy_hold"):
        benchmark_comparison(make_metrics_table(), strategy="top_k_net")
