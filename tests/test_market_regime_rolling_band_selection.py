import numpy as np
import pandas as pd

from research.market_regime.rolling_band_selection import (
    arm_name,
    chain_blocks,
    chain_metrics,
    choose_arms,
    daily_returns,
    fold_boundaries,
    window_return,
    window_turnover,
)


def test_arm_name_matches_the_hysteresis_module_convention():
    assert arm_name(0.10) == "band_entry10_exit10"
    assert arm_name(1.00) == "band_entry10_exit100"


def make_fold_table():
    return pd.DataFrame([
        {"fold": 1, "trail_return_a": 0.10, "trail_return_b": 0.20, "trail_return_c": 0.30,
         "trail_turnover_a": 10.0, "trail_turnover_b": 5.0, "trail_turnover_c": 1.0},
        {"fold": 2, "trail_return_a": 0.40, "trail_return_b": 0.20, "trail_return_c": 0.30,
         "trail_turnover_a": 10.0, "trail_turnover_b": 5.0, "trail_turnover_c": 1.0},
    ])


def test_choose_arms_takes_the_trailing_argmax_per_fold():
    assert choose_arms(make_fold_table(), ["a", "b", "c"], 0.0) == ["c", "a"]


def test_turnover_floor_excludes_a_lucky_low_turnover_arm():
    # c wins fold 1 on return but its trailing turnover is 10% of the best arm's.
    assert choose_arms(make_fold_table(), ["a", "b", "c"], 0.25) == ["b", "a"]


def test_turnover_breaks_a_return_tie_toward_the_narrower_band():
    table = pd.DataFrame([{"fold": 1, "trail_return_a": 0.20, "trail_return_b": 0.20,
                           "trail_turnover_a": 10.0, "trail_turnover_b": 5.0}])
    assert choose_arms(table, ["a", "b"], 0.0) == ["b"]


def test_fold_boundaries_tile_the_range_and_include_the_last_day():
    start, end = pd.Timestamp("2022-01-01"), pd.Timestamp("2023-06-15")
    folds = fold_boundaries(start, end, 6)

    assert [fold["trade_start"] for fold in folds] == [
        pd.Timestamp("2022-01-01"), pd.Timestamp("2022-07-01"), pd.Timestamp("2023-01-01")]
    # Contiguous: no gap and no overlap between consecutive folds.
    for left, right in zip(folds, folds[1:]):
        assert left["trade_end"] == right["trade_start"]
    # The final fold reaches one day past the last session so nothing is dropped.
    assert folds[-1]["trade_end"] == end + pd.Timedelta(days=1)
    assert folds[0]["selection_end"] == folds[0]["trade_start"]


def test_fold_boundaries_cover_a_range_shorter_than_one_step():
    folds = fold_boundaries(pd.Timestamp("2022-07-20"), pd.Timestamp("2022-09-01"), 6)

    assert len(folds) == 1
    assert folds[0]["trade_end"] == pd.Timestamp("2022-09-02")


def test_daily_returns_anchors_the_first_session_on_initial_capital():
    equity = pd.Series([100.0, 110.0], index=pd.to_datetime(["2021-01-04", "2021-01-05"]))

    returns = daily_returns(equity, 100.0)

    assert np.allclose(returns.to_numpy(), [0.0, 0.10])


def test_window_helpers_slice_half_open_and_scale_turnover():
    returns = pd.Series([0.0, 0.10, 0.10], index=pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"]))
    trades = pd.DataFrame({"date": pd.to_datetime(["2021-01-05", "2021-01-06"]), "trade_value": [1000.0, 3000.0]})

    # [start, end) -- the end date itself is excluded.
    assert np.isclose(window_return(returns, pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-06")), 0.10)
    assert window_return(returns, pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-04")) is np.nan or \
        np.isnan(window_return(returns, pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-04")))
    assert np.isclose(window_turnover(trades, pd.Timestamp("2021-01-05"), pd.Timestamp("2021-01-06"), 1000.0), 1.0)


def test_chain_blocks_concatenates_the_chosen_arm_per_fold_in_order():
    blocks = {
        (0, "a"): pd.Series([0.01], index=pd.to_datetime(["2021-01-04"])),
        (0, "b"): pd.Series([0.50], index=pd.to_datetime(["2021-01-04"])),
        (1, "a"): pd.Series([0.02], index=pd.to_datetime(["2021-01-05"])),
        (1, "b"): pd.Series([0.60], index=pd.to_datetime(["2021-01-05"])),
    }

    chained = chain_blocks(blocks, ["b", "a"])

    assert np.allclose(chained.to_numpy(), [0.50, 0.02])
    assert chained.index.is_monotonic_increasing


def test_chain_metrics_matches_a_known_geometric_path():
    returns = pd.Series([0.01] * 10, index=pd.bdate_range("2021-01-04", periods=10))

    stats = chain_metrics(returns, 100.0)

    assert np.isclose(stats["total_return"], 1.01 ** 10 - 1.0)
    assert np.isclose(stats["annual_volatility"], 0.0)  # a constant path has no dispersion
    assert stats["days"] == 10


def test_chain_metrics_reports_the_drawdown_of_the_spliced_path():
    returns = pd.Series([0.10, -0.50, 0.10], index=pd.bdate_range("2021-01-04", periods=3))

    stats = chain_metrics(returns, 100.0)

    # peak 1.10 -> trough 0.55 is a 50% drawdown.
    assert np.isclose(stats["max_drawdown"], -0.50)
