"""Tests for the E7 zz500 pool-expansion criteria.

Everything here runs on synthetic summaries written to ``tmp_path``; no real
dataset, price cache or engine run is touched.
"""

import json

import pandas as pd
import pytest

from research.experiments import pool_expansion_zz500 as e7
from research.experiments.pool_expansion_zz500 import (
    POOLS,
    Pool,
    collect_pool_row,
    criterion_1_null_percentile,
    criterion_2_split_completeness,
    criterion_3_separate_report,
    null_summary_path,
    read_null_summary,
)


def make_summary(percentile=0.99, model_id="zz500_pit_outcome", top_k=0.10,
                 seeds=200, total_return=0.25, split="test", rows_field="split_rows"):
    """A summary shaped like ``random_portfolio_baseline``'s output."""

    return {
        "model_id": model_id,
        "seeds": seeds,
        "seed": 20260915,
        "split": split,
        "top_k": top_k,
        "horizon": 10,
        "min_holding_days": 10,
        "max_positions": 20,
        rows_field: 281982,
        "actual": {
            "total_return": total_return,
            "annual_return": 0.1,
            "sharpe_zero_rate": 1.2,
            "max_drawdown": -0.15,
            "turnover_multiple": 12.0,
            "filled_buy_count": 100,
        },
        "random": {
            "total_return": {
                "mean": 0.05, "std": 0.09, "min": -0.2, "max": 0.4, "median": 0.04,
                "percentile_of_actual": percentile,
                "empirical_p_value": 0.005,
            }
        },
    }


def write_summary(dataset_dir, pool, split, **kwargs):
    path = null_summary_path(dataset_dir, pool, split)
    path.write_text(json.dumps(make_summary(split=split, **kwargs)), encoding="utf-8")
    return path


def test_percentile_is_read_off_the_summary_not_recomputed():
    result = criterion_1_null_percentile(make_summary(percentile=0.995))

    assert result["percentile_of_actual"] == pytest.approx(0.995)
    assert result["empirical_p_value"] == pytest.approx(0.005)
    assert result["seeds"] == 200


def test_percentile_exactly_at_the_bar_passes():
    """The bar is inclusive; a null with no spread sits exactly on it."""

    assert criterion_1_null_percentile(make_summary(percentile=0.95))["passed"]


def test_percentile_just_below_the_bar_fails():
    assert not criterion_1_null_percentile(make_summary(percentile=0.945))["passed"]


def test_reading_a_missing_null_summary_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_null_summary(tmp_path / "nope.json")


def test_a_summary_without_the_null_block_is_rejected(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text(json.dumps({"model_id": "zz500_pit_outcome"}), encoding="utf-8")

    with pytest.raises(ValueError):
        read_null_summary(path)


def test_an_unknown_split_has_no_null_tag():
    with pytest.raises(ValueError):
        null_summary_path("unused", POOLS["zz500"], "holdout")


def write_pit_audit(dataset_dir, pool, splits=("train", "validation", "test"), rows=1000):
    """A ``pit_pool_filter`` audit table with one row per split."""

    frame = pd.DataFrame({
        "split": list(splits),
        "kind": ["tradability_samples"] * len(splits),
        "rows_before": [rows] * len(splits),
        "rows_after": [rows // 2] * len(splits),
    })
    path = dataset_dir / "pit_pool_filter_{}.csv".format(pool.dataset_id)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_criterion_2_passes_only_when_both_windows_are_nulled(tmp_path):
    zz500 = POOLS["zz500"]
    write_pit_audit(tmp_path, zz500)
    write_summary(tmp_path, zz500, "test")
    write_summary(tmp_path, zz500, "validation")

    result = criterion_2_split_completeness(tmp_path, zz500)

    assert result["passed"]
    assert result["missing_nulls"] == []
    assert result["missing_splits"] == []


def test_criterion_2_does_not_ask_a_train_window_for_a_null():
    """A null is an out-of-sample backtest; "train" is a category error."""

    assert "train" not in e7.NULL_SPLITS
    assert set(e7.SAMPLE_SPLITS) == {"train", "validation", "test"}


def test_criterion_2_fails_when_validation_has_no_null_of_its_own(tmp_path):
    """The HS300 debt, reproduced: three splits of data but a hole in the nulls."""

    zz500 = POOLS["zz500"]
    write_pit_audit(tmp_path, zz500)
    write_summary(tmp_path, zz500, "test")

    result = criterion_2_split_completeness(tmp_path, zz500)

    assert not result["passed"]
    assert result["missing_nulls"] == ["validation"]


def test_criterion_2_fails_when_a_sample_split_is_absent(tmp_path):
    zz500 = POOLS["zz500"]
    write_pit_audit(tmp_path, zz500, splits=("train", "test"))
    write_summary(tmp_path, zz500, "test")
    write_summary(tmp_path, zz500, "validation")

    result = criterion_2_split_completeness(tmp_path, zz500)

    assert not result["passed"]
    assert result["missing_splits"] == ["validation"]


def test_criterion_2_flags_a_null_belonging_to_another_pool(tmp_path):
    zz500 = POOLS["zz500"]
    write_pit_audit(tmp_path, zz500)
    for split in ("validation", "test"):
        write_summary(tmp_path, zz500, split, model_id="hs300_pit_outcome")

    result = criterion_2_split_completeness(tmp_path, zz500)

    assert not result["passed"]
    assert sorted(result["mislabelled_nulls"]) == ["test", "validation"]


def test_comparison_rows_may_not_repeat_a_pool():
    rows = [{"pool": "zz500"}, {"pool": "zz500"}]

    with pytest.raises(ValueError):
        criterion_3_separate_report(rows)


def test_comparison_rows_need_a_pool_column():
    with pytest.raises(ValueError):
        criterion_3_separate_report([{"model_id": "zz500_pit_outcome"}])


def test_comparison_is_one_row_per_pool_with_no_aggregate_row():
    frame = criterion_3_separate_report([{"pool": "hs300"}, {"pool": "zz500"}])

    assert frame["pool"].tolist() == ["hs300", "zz500"]
    assert len(frame) == 2


def test_collect_pool_row_reads_either_row_field_name(tmp_path):
    """Published summaries say ``test_rows``; newer ones say ``split_rows``."""

    zz500 = POOLS["zz500"]
    write_summary(tmp_path, zz500, "test", rows_field="test_rows")

    row = collect_pool_row(tmp_path, zz500, "test")

    assert row["pool"] == "zz500"
    assert row["percentile_of_actual"] == pytest.approx(0.99)


def test_a_second_pool_named_differently_is_still_one_row_each(tmp_path):
    """Criterion 3 is structural: the report keeps pools apart by construction."""

    other = Pool(name="zz1000", model_id="zz1000_pit_outcome",
                 dataset_id="zz1000_pit_v1", benchmark="data/benchmarks/000852.csv",
                 tags={"test": "zz1000_pit"})
    write_summary(tmp_path, other, "test", model_id="zz1000_pit_outcome", percentile=0.98)

    row = collect_pool_row(tmp_path, other, "test")

    assert row["pool"] == "zz1000"


def test_building_a_report_with_nothing_on_disk_does_not_raise(tmp_path):
    """Every read is guarded, so a half-built pool reports failures, not a traceback."""

    report = e7.build_report(tmp_path, "test")

    assert report["criterion_1"]["missing"]
    assert not report["criterion_2"]["passed"]
    assert report["criterion_2"]["splits"]["train"]["rows_after"] == 0


def test_a_complete_report_renders_without_raising(tmp_path, capsys):
    """The renderer reads fields the criteria build.

    They were written separately: every criterion had a test and the renderer had
    none, so it read a ``null_median`` the percentile function never returned and
    the first real run died with a KeyError -- after the whole chain had been
    built.  A full-report render is the only test that covers the seam.
    """

    zz500 = POOLS["zz500"]
    write_pit_audit(tmp_path, zz500)
    for split in ("validation", "test"):
        write_summary(tmp_path, zz500, split)
    write_summary(tmp_path, POOLS["hs300"], "test", model_id="hs300_pit_outcome")

    report = e7.build_report(tmp_path, "test")
    e7.print_report(report)

    printed = capsys.readouterr().out
    assert "criterion 1" in printed
    assert "criterion 2" in printed
    assert "criterion 3" in printed
    assert report["criterion_1"]["passed"]
    # The report must not carry a "baseline arm percentile": the pre-registration
    # asked for the baseline arm's, which needs a null drawn for the baseline and
    # does not exist here.  Reporting criterion 1's own percentile under that name
    # is a tautology that reads as a check.
    assert "baseline_arm_percentile" not in report["criterion_1"]
    assert "clears it by" in printed


def test_the_written_table_is_the_one_the_report_printed(tmp_path, capsys):
    """Two artifacts, one frame: the CSV must not be a second computation.

    The plan named ``{criteria,summary}.{csv,json}``; nothing wrote either, so the
    report existed only as console scrollback.  The CSV is emitted from the frame
    the renderer already printed rather than rebuilt from disk, which is what
    stops the two files from disagreeing.
    """

    zz500 = POOLS["zz500"]
    write_pit_audit(tmp_path, zz500)
    for split in ("validation", "test"):
        write_summary(tmp_path, zz500, split)

    report = e7.build_report(tmp_path, "test")
    json_path, csv_path = e7.write_report(report, tmp_path / "out" / "e7_criteria.json")

    assert json_path.exists() and csv_path.exists()
    assert csv_path.name == "e7_criteria_table.csv"
    assert not report["comparison"].empty

    written = pd.read_csv(csv_path)
    assert written["pool"].tolist() == report["comparison"]["pool"].tolist()
    # Only zz500 has a summary here; the other two pools are error rows, which is
    # exactly the shape a half-built pool must produce rather than a traceback.
    assert written.loc[written["pool"] == "zz500", "percentile_of_actual"].iloc[0] == pytest.approx(0.99)
    assert written["percentile_of_actual"].isna().sum() == len(written) - 1
    assert json.loads(json_path.read_text(encoding="utf-8"))["pool"] == "zz500"


def test_report_only_does_not_start_the_pipeline(tmp_path, monkeypatch):
    """The whole point of ``--report-only`` is that it runs nothing."""

    calls = []
    monkeypatch.setattr(e7, "run_pipeline", lambda **kwargs: calls.append(kwargs) or 0)

    e7.main(["--dataset-dir", str(tmp_path), "--report-only"])

    assert calls == []


def test_the_default_invocation_prints_the_chain_instead_of_running_it(monkeypatch, capsys):
    """A bare call must never launch hours of pipeline by accident."""

    monkeypatch.setattr(e7, "run_pipeline", lambda **kwargs: 0)

    e7.main([])

    assert e7.PIPELINE_STEPS[0][0] == "-m"


def test_the_recorded_chain_pins_the_train_start_it_was_run_with():
    """The pool's train window starts at the index publication, not at the data."""

    joined = [" ".join(step) for step in e7.PIPELINE_STEPS]
    dataset_step = [step for step in joined if "probability_dataset" in step]

    assert dataset_step, "the chain must contain the dataset build"
    assert "--train-start 2007-01-31" in dataset_step[0]


def test_the_recorded_chain_uses_the_zz500_benchmark_everywhere_it_matters():
    joined = " ".join(" ".join(step) for step in e7.PIPELINE_STEPS)

    assert "data/benchmarks/000905.csv" in joined
    assert "data/ZZ_500.txt" in joined


def test_benchmark_display_name_follows_the_file_it_was_given():
    """Every 中证1000/中证500 manifest used to describe itself as HS300.

    The string is cosmetic -- the label always comes from ``--benchmark-file``
    -- but a manifest that names the wrong index is how a later reader
    concludes the wrong thing about which benchmark the excess is against.
    """

    from research.market_regime.excess_target_dataset import benchmark_name

    assert benchmark_name("data/benchmarks/000300.csv") == "HS300"
    assert benchmark_name("data/benchmarks/000905.csv") == "中证500"
    assert benchmark_name("data/benchmarks/000852.csv") == "中证1000"
    # An unknown file falls back to its stem rather than silently claiming HS300.
    assert benchmark_name("data/benchmarks/399905.csv") == "399905"
