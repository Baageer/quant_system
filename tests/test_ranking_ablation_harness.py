"""The shared harness has two silent-failure modes worth pinning.

``fold_excess_table`` is the one that already shipped a bug of this shape once:
it returns arms in sorted order while callers build their grids in their own arm
order, so anything pairing the two by position attributes one arm's folds to
another.  It is pinned here to follow the arm *name*, and to refuse a baseline
that is not in the equity map rather than quietly producing an all-zero table.

``load_null_total_returns`` decides which "random" a strategy is compared against,
and the published nulls are not interchangeable -- two pools have same-width nulls
drawn on different windows.  Every check below exists because the alternative is
comparing a strategy to the wrong universe.
"""

import json

import numpy as np
import pandas as pd
import pytest

from research.experiments.ranking_ablation_harness import (
    fold_excess_table,
    load_null_total_returns,
)


def make_equity(dates, rates):
    return {
        arm: pd.Series(100000.0 * np.cumprod([1.0 + rate] * len(dates)), index=dates)
        for arm, rate in rates.items()
    }


def test_fold_excess_follows_the_arm_name_not_the_position():
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    equity = make_equity(dates, {"aaa": 0.01, "zzz": 0.0, "mmm": 0.02})
    folds = [
        {"trade_start": dates[0], "trade_end": dates[20]},
        {"trade_start": dates[20], "trade_end": dates[-1] + pd.Timedelta(days=1)},
    ]

    table = fold_excess_table(equity, folds, 100000.0, baseline_arm="zzz").set_index("arm")

    # ``zzz`` is both alphabetically last and the baseline, so a positional pair
    # would give it ``mmm``'s compounding numbers instead of the zero row.
    assert table.loc["zzz", "fold_0"] == pytest.approx(0.0, abs=1e-12)
    assert table.loc["zzz", "fold_1"] == pytest.approx(0.0, abs=1e-12)
    assert table.loc["aaa", "fold_0"] > 0
    assert table.loc["mmm", "fold_0"] > table.loc["aaa", "fold_0"]


def test_the_baseline_row_is_identically_zero():
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    equity = make_equity(dates, {"base": 0.005, "other": 0.004})
    folds = [{"trade_start": dates[0], "trade_end": dates[-1] + pd.Timedelta(days=1)}]

    table = fold_excess_table(equity, folds, 100000.0, baseline_arm="base").set_index("arm")

    fold_columns = [column for column in table.columns if column.startswith("fold_")]
    assert np.allclose(table.loc["base", fold_columns].astype(float), 0.0)


def test_an_unknown_baseline_is_named_rather_than_defaulted():
    """A default would silently produce an all-zero table for a caller who renamed
    its baseline arm -- indistinguishable from 'the arms are identical'."""

    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    equity = make_equity(dates, {"x": 0.01})
    folds = [{"trade_start": dates[0], "trade_end": dates[-1] + pd.Timedelta(days=1)}]

    with pytest.raises(ValueError) as excinfo:
        fold_excess_table(equity, folds, 100000.0, baseline_arm="missing")

    assert "missing" in str(excinfo.value)


def test_the_label_overrides_the_output_name_without_changing_the_column(monkeypatch):
    """A construction grid reuses one score column for every cell, so arm and
    label stop being the same string -- without ``label`` the whole performance
    table comes back with 36 rows all named after the column."""

    from research.experiments import ranking_ablation_harness as harness
    from research.market_regime.trade_outcome_ranking_evaluation import SCORE_COLUMN

    captured = {}
    monkeypatch.setattr(
        harness, "_run_engine",
        lambda *a, **k: (pd.DataFrame({"portfolio_value": [100000.0]}), pd.DataFrame()))

    def fake_summarize(name, results, trades, initial_capital):
        captured["name"] = name
        return {"total_return": 0.0}

    monkeypatch.setattr(harness, "summarize_backtest", fake_summarize)
    scores = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
        "symbol": ["000001.SZ", "000002.SZ"],
        SCORE_COLUMN: [1.0, 2.0],
    })

    harness.backtest_arm(
        scores, {}, {}, SCORE_COLUMN, scores["date"].min(), scores["date"].max(),
        0.5, 10, 100000.0, 2, "config/settings.yaml", label="tk0.1_mp20")

    assert captured["name"] == "tk0.1_mp20"


def write_null(tmp_path, model_id, tag, top_k, rows, returns=(0.01, 0.02), split=None,
               max_positions=None, rows_key="split_rows"):
    """``rows_key`` defaults to the key the *current* writer emits.

    The two spellings both exist in the wild: ``test_rows`` in summaries written
    before ``random_portfolio_baseline`` gained ``--split``, ``split_rows`` after.
    A fixture that only ever writes the legacy key is how the reader's inability to
    read the new one went unnoticed.
    """

    pd.DataFrame({"total_return": returns}).to_csv(
        tmp_path / "random_portfolio_{}_{}_draws.csv".format(model_id, tag), index=False)
    summary = {"model_id": model_id, "top_k": top_k, rows_key: rows}
    if split is not None:
        summary["split"] = split
    if max_positions is not None:
        summary["max_positions"] = max_positions
    (tmp_path / "random_portfolio_{}_{}_summary.json".format(model_id, tag)).write_text(
        json.dumps(summary), encoding="utf-8")


def test_the_null_is_returned_when_it_fits(tmp_path):
    write_null(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898)

    null = load_null_total_returns(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898)

    assert list(null) == [0.01, 0.02]


def test_a_null_at_another_width_is_rejected(tmp_path):
    write_null(tmp_path, "hs300_pit_outcome", "pit", 0.10, 218898)

    with pytest.raises(ValueError) as excinfo:
        load_null_total_returns(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898)

    assert "top_k" in str(excinfo.value)


def test_a_null_drawn_on_another_window_is_rejected(tmp_path):
    """HS300's forward-2026 null is also top_k=0.05; only the row count separates
    them, so a same-width null is not automatically the right null."""

    write_null(tmp_path, "hs300_pit_outcome", "pit", 0.05, 36767)

    with pytest.raises(ValueError) as excinfo:
        load_null_total_returns(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898)

    # The numbers themselves, not a key name -- the message names both keys, so
    # asserting on "test_rows" would pass for the wrong reason.
    assert "36767" in str(excinfo.value) and "218898" in str(excinfo.value)


def test_a_null_written_under_the_legacy_row_key_is_still_accepted(tmp_path):
    """Older summaries say ``test_rows``; they must keep working."""

    write_null(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898, rows_key="test_rows")

    null = load_null_total_returns(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898)

    assert list(null) == [0.01, 0.02]


def test_a_null_written_under_the_current_row_key_is_accepted(tmp_path):
    """The regression: the reader demanded ``test_rows`` while the writer emitted
    ``split_rows``, so *no* freshly drawn null could be loaded.

    Found when E8's own null came back as ``test_rows=None (need 218898)`` -- a
    missing key reported as a wrong value.
    """

    write_null(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898)   # split_rows

    null = load_null_total_returns(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898)

    assert list(null) == [0.01, 0.02]


def test_a_wrong_row_count_is_rejected_either_way_it_is_spelled(tmp_path):
    """Accepting both keys must not turn into accepting either number."""

    for key in ("split_rows", "test_rows"):
        write_null(tmp_path, "hs300_pit_outcome", "pit", 0.05, 999, rows_key=key)

        with pytest.raises(ValueError) as excinfo:
            load_null_total_returns(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898)

        assert "999" in str(excinfo.value)


def test_a_validation_split_null_is_rejected(tmp_path):
    write_null(tmp_path, "zz1000_pit_outcome", "zz1000_pit", 0.1, 543025, split="validation")

    with pytest.raises(ValueError) as excinfo:
        load_null_total_returns(tmp_path, "zz1000_pit_outcome", "zz1000_pit", 0.1, 543025)

    assert "split" in str(excinfo.value)


def test_a_null_drawn_at_another_cap_is_rejected(tmp_path):
    """Same width, same window, half the exposure per name: the drawn basket is
    the same but ``target_notional = capital / max_positions`` is not, so the
    percentile would compare the strategy against a smaller version of itself."""

    write_null(tmp_path, "hs300_pit_outcome", "e6_tk0.05_mp40", 0.05, 218898, max_positions=40)

    with pytest.raises(ValueError) as excinfo:
        load_null_total_returns(
            tmp_path, "hs300_pit_outcome", "e6_tk0.05_mp40", 0.05, 218898, max_positions=20)

    assert "max_positions" in str(excinfo.value)


def test_a_null_without_a_recorded_cap_is_rejected_when_the_caller_knows_its_own(tmp_path):
    """An absent key is not a match.  Reading it as one is the silent version."""

    write_null(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898)

    with pytest.raises(ValueError) as excinfo:
        load_null_total_returns(
            tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898, max_positions=20)

    assert "max_positions=absent" in str(excinfo.value)


def test_the_cap_is_not_checked_when_the_caller_does_not_name_one(tmp_path):
    """The published nulls predate the check; callers that know their cap pass it,
    the rest keep the old behaviour rather than failing on an older summary."""

    write_null(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898, max_positions=20)

    null = load_null_total_returns(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898)

    assert list(null) == [0.01, 0.02]


def test_a_missing_null_names_the_file(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        load_null_total_returns(tmp_path, "hs300_pit_outcome", "pit", 0.05, 218898)

    assert "hs300_pit_outcome" in str(excinfo.value)
