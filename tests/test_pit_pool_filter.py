"""The PIT run must never inherit a stale path from the static dataset's manifest.

``load_model_splits`` reads its input paths out of the manifest, so a manifest
that merely gets copied under a new id would feed the *static* samples into a run
that reports itself as point-in-time.  That failure is silent whenever the old
path still resolves, which is why it is pinned here rather than left to a smoke run.
"""

import json

import pandas as pd
import pytest

from research.market_regime.pit_pool_filter import filter_dataset

SNAPSHOTS = {"20200123": ["000001.SZ"], "20200228": ["000001.SZ", "000002.SZ"]}

DATASET_ID = "src_dataset"
RUN_ID = "pit_run"


@pytest.fixture
def workspace(tmp_path):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    membership_dir = tmp_path / "membership"
    membership_dir.mkdir()

    pd.DataFrame([
        {"index_code": "000300.SH", "con_code": code, "trade_date": date, "weight": 1.0}
        for date, codes in SNAPSHOTS.items() for code in codes
    ]).to_csv(membership_dir / "000300.SH.csv", index=False, encoding="utf-8-sig")

    # 000001 is a member throughout; 000002 only from the second snapshot, so both
    # splits must shrink and the pre-membership row must be dropped.
    rows = [
        {"date": "2020-01-25", "symbol": "000001", "target": 1.0, "feature": 1.0},
        {"date": "2020-03-02", "symbol": "000001", "target": 2.0, "feature": 2.0},
        {"date": "2020-01-25", "symbol": "000002", "target": 3.0, "feature": 3.0},
        {"date": "2020-03-02", "symbol": "000002", "target": 4.0, "feature": 4.0},
    ]
    for split in ("train", "validation", "test"):
        for kind in ("tradability_samples", "model_samples"):
            pd.DataFrame(rows).to_csv(
                dataset_dir / "trend_probability_{}_{}_{}.csv".format(split, kind, DATASET_ID),
                index=False, encoding="utf-8-sig")

    # A plausible pre-move manifest: paths are relative and point at the *source* files
    dataset_manifest = {
        "analysis_id": DATASET_ID,
        "train_end": "2020-01-01",
        "paths": {
            "train_model_samples": "output\\datasets\\trend_probability_train_model_samples_{}.csv".format(DATASET_ID),
            "validation_model_samples": "output\\datasets\\trend_probability_validation_model_samples_{}.csv".format(DATASET_ID),
            "test_model_samples": "output\\datasets\\trend_probability_test_model_samples_{}.csv".format(DATASET_ID),
            "all_samples": "output\\datasets\\trend_probability_all_samples_{}.csv".format(DATASET_ID),
        },
    }
    tradability_manifest = {
        "dataset_id": DATASET_ID,
        "paths": {
            "splits": {
                split: "output\\datasets\\trend_probability_{}_tradability_samples_{}.csv".format(split, DATASET_ID)
                for split in ("train", "validation", "test")
            },
            "split_summary": "output\\datasets\\trend_probability_tradability_split_summary_{}.csv".format(DATASET_ID),
        },
    }
    (dataset_dir / "trend_probability_dataset_manifest_{}.json".format(DATASET_ID)).write_text(
        json.dumps(dataset_manifest), encoding="utf-8")
    (dataset_dir / "trend_probability_tradability_manifest_{}.json".format(DATASET_ID)).write_text(
        json.dumps(tradability_manifest), encoding="utf-8")

    return dataset_dir, membership_dir


def run_filter(workspace, **kwargs):
    dataset_dir, membership_dir = workspace
    return filter_dataset(dataset_dir, DATASET_ID, "hs300", RUN_ID, membership_dir, **kwargs)


def test_rows_are_restricted_to_members_at_the_time(workspace):
    dataset_dir, _ = workspace
    report = run_filter(workspace)
    assert set(report["rows_before"]) == {4}
    # 01-25 -> only the 01-23 snapshot, so 000001 alone; 03-02 -> the 02-28 snapshot,
    # which has added 000002, so both
    assert set(report["rows_after"]) == {3}

    kept = pd.read_csv(dataset_dir / "trend_probability_train_model_samples_{}.csv".format(RUN_ID),
                       dtype={"symbol": str})
    assert set(kept["symbol"]) == {"000001", "000002"}
    # 000002's pre-membership 01-25 row is gone, 000001's is kept
    assert set(zip(kept["symbol"], kept["date"])) == {("000001", "2020-01-25"), ("000001", "2020-03-02"),
                                                      ("000002", "2020-03-02")}


def test_model_sample_paths_in_the_manifest_point_at_the_pit_files(workspace):
    dataset_dir, _ = workspace
    run_filter(workspace)
    manifest = json.loads(
        (dataset_dir / "trend_probability_dataset_manifest_{}.json".format(RUN_ID)).read_text(encoding="utf-8"))

    for split in ("train", "validation", "test"):
        key = "{}_model_samples".format(split)
        path = manifest["paths"][key]
        assert RUN_ID in path
        assert DATASET_ID not in path
        # the whole point: the path the loader will follow must resolve
        assert pd.io.common.file_exists(path)


def test_nested_tradability_split_paths_are_rewritten(workspace):
    dataset_dir, _ = workspace
    run_filter(workspace)
    manifest = json.loads(
        (dataset_dir / "trend_probability_tradability_manifest_{}.json".format(RUN_ID)).read_text(encoding="utf-8"))

    for split, path in manifest["paths"]["splits"].items():
        assert path.endswith("trend_probability_{}_tradability_samples_{}.csv".format(split, RUN_ID))
        assert pd.io.common.file_exists(path)


def test_paths_without_a_pit_counterpart_are_reported_not_silently_kept(workspace):
    dataset_dir, _ = workspace
    run_filter(workspace)
    manifest = json.loads(
        (dataset_dir / "trend_probability_dataset_manifest_{}.json".format(RUN_ID)).read_text(encoding="utf-8"))

    # `all_samples` was never re-exported, so it must be flagged rather than left
    # looking like PIT data
    assert "all_samples" in manifest["pit_filter"]["paths_still_static"]
    assert "train_model_samples" not in manifest["pit_filter"]["paths_still_static"]


def test_provenance_records_the_pool_and_source(workspace):
    dataset_dir, _ = workspace
    run_filter(workspace)
    manifest = json.loads(
        (dataset_dir / "trend_probability_tradability_manifest_{}.json".format(RUN_ID)).read_text(encoding="utf-8"))

    assert manifest["pit_source_dataset_id"] == DATASET_ID
    assert manifest["pit_filter"]["pool"] == "hs300"
    assert manifest["pit_filter"]["run_id"] == RUN_ID
    # split boundaries are inherited, not recomputed
    assert manifest.get("dataset_id") == DATASET_ID
