"""E1's frozen inputs, and the anchor that decides whether it means anything.

``HISTGB_PARAMS`` is pre-registered as frozen: the whole experiment is void if the
hyperparameters are tuned afterwards, so the values are pinned here as literals.
That is the one place a literal is the right test -- unlike E2's arm orientation,
where a literal pinned a sign error, these numbers have no derivable form.

``anchor_check`` is the other half: it is what makes ``ridge`` the published arm
rather than a lookalike, so the failure modes that must raise (wrong row count,
misaligned keys, a recomputed score that does not match) are pinned too.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from research.experiments import model_form_reachability
from research.experiments.model_form_reachability import (
    ARMS,
    BASELINE_ARM,
    HISTGB_PARAMS,
    POOLS,
    anchor_check,
    build_arm_scores,
    load_published_model,
)


def test_histgb_params_are_the_preregistered_values():
    assert HISTGB_PARAMS == dict(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=500, l2_regularization=10.0,
        early_stopping=False, random_state=20260918,
    )


def test_the_baseline_arm_is_the_ridge():
    assert BASELINE_ARM == "ridge"
    assert ARMS == ("ridge", "histgb")


def test_the_pool_registry_names_the_published_dataset_ids():
    """A wrong ``dataset_id`` silently loads a different cross-section."""

    assert POOLS["hs300"]["dataset_id"] == "excess_hs300_pit_v1"
    assert POOLS["zz1000"]["dataset_id"] == "excess_zz1000_pit_v1"
    assert POOLS["hs300"]["model_id"] == "hs300_pit_outcome"
    assert POOLS["zz1000"]["model_id"] == "zz1000_pit_outcome"


def test_a_missing_pickle_is_named(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        load_published_model(tmp_path, "nope")

    assert "nope" in str(excinfo.value)


class _ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, features):
        return np.full(len(features), self.value, dtype=float)


def make_samples(rows):
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _bypass_preprocessing(monkeypatch):
    """The anchor only has to agree with the published *score*, so the feature
    transform is replaced by the identity -- a real preprocessor would need all
    28 columns to build a fixture that tests nothing extra."""

    monkeypatch.setattr(
        model_form_reachability, "transform_features", lambda samples, pre: samples)


def write_disk(tmp_path, model_id, samples, risk_penalty, scores):
    frame = samples.copy()
    frame["entry_score"] = scores
    frame.to_csv(tmp_path / "trade_outcome_test_predictions_{}.csv".format(model_id), index=False)


def test_the_anchor_accepts_a_reproduced_score(tmp_path):
    samples = make_samples([
        {"date": "2022-07-20", "symbol": "000001.SZ"},
        {"date": "2022-07-21", "symbol": "000002.SZ"},
    ])
    model = {
        "feature_columns": [], "preprocessor": {},
        "excess_model": _ConstantModel(0.01), "mae_model": _ConstantModel(-0.04),
    }
    # 0.01 - 0.25 * max(0.04, 0) == 0.0
    write_disk(tmp_path, "m", samples, 0.25, [0.0, 0.0])

    delta = anchor_check(model, samples, 10, 0.25, tmp_path, "m")

    assert delta == pytest.approx(0.0)


def test_the_anchor_rejects_a_score_that_does_not_match(tmp_path):
    samples = make_samples([{"date": "2022-07-20", "symbol": "000001.SZ"}])
    model = {
        "feature_columns": [], "preprocessor": {},
        "excess_model": _ConstantModel(0.01), "mae_model": _ConstantModel(-0.04),
    }
    write_disk(tmp_path, "m", samples, 0.25, [0.5])

    with pytest.raises(ValueError) as excinfo:
        anchor_check(model, samples, 10, 0.25, tmp_path, "m")

    assert "anchor FAILED" in str(excinfo.value)


def test_the_anchor_rejects_a_different_row_count(tmp_path):
    samples = make_samples([{"date": "2022-07-20", "symbol": "000001.SZ"}])
    model = {
        "feature_columns": [], "preprocessor": {},
        "excess_model": _ConstantModel(0.01), "mae_model": _ConstantModel(-0.04),
    }
    write_disk(tmp_path, "m", samples, 0.25, [0.0])
    longer = make_samples([
        {"date": "2022-07-20", "symbol": "000001.SZ"},
        {"date": "2022-07-21", "symbol": "000002.SZ"},
    ])

    with pytest.raises(ValueError) as excinfo:
        anchor_check(model, longer, 10, 0.25, tmp_path, "m")

    assert "row count differs" in str(excinfo.value)


def test_the_anchor_rejects_misaligned_keys(tmp_path):
    """Same length, different rows -- a length-only comparison would pass this."""

    model = {
        "feature_columns": [], "preprocessor": {},
        "excess_model": _ConstantModel(0.01), "mae_model": _ConstantModel(-0.04),
    }
    write_disk(tmp_path, "m", make_samples([
        {"date": "2022-07-20", "symbol": "000001.SZ"},
        {"date": "2022-07-21", "symbol": "000002.SZ"},
    ]), 0.25, [0.0, 0.0])
    other = make_samples([
        {"date": "2022-07-20", "symbol": "000001.SZ"},
        {"date": "2022-07-21", "symbol": "000009.SZ"},
    ])

    with pytest.raises(ValueError) as excinfo:
        anchor_check(model, other, 10, 0.25, tmp_path, "m")

    assert "keys differ" in str(excinfo.value)


def test_the_anchor_tolerates_row_order(tmp_path):
    """The disk file is sorted, so row *order* is presentation, not identity.

    The prediction is taken in the sorted order the diff happens in -- otherwise a
    caller passing an unsorted frame would be diffed against its own reshuffle.
    """

    disk_rows = [
        {"date": "2022-07-20", "symbol": "000001.SZ"},
        {"date": "2022-07-21", "symbol": "000002.SZ"},
    ]
    model = {
        "feature_columns": [], "preprocessor": {},
        "excess_model": _ConstantModel(0.01), "mae_model": _ConstantModel(-0.04),
    }
    write_disk(tmp_path, "m", make_samples(disk_rows), 0.25, [0.0, 0.0])

    delta = anchor_check(
        model, make_samples(list(reversed(disk_rows))), 10, 0.25, tmp_path, "m")

    assert delta == pytest.approx(0.0)
