"""Tests for the E8 A/B driver's gates and arm bookkeeping.

What is pinned here is the part that fails *quietly*: a gate reading the wrong
manifest, an arm whose feature list was rebuilt rather than appended, a criteria
table that accepts a treatment which only works on the window it was fitted to.
None of those raise, and all of them produce a number that looks like a result.

Synthetic frames and temporary manifests only -- no real dataset is touched, and
no engine run happens.
"""

import json

import numpy as np
import pandas as pd
import pytest

from research.experiments.alternative_data_main_ab import (
    ADOPTABLE_ARM,
    APPENDED_COLUMNS,
    ARMS,
    BOTH_ARM,
    FROZEN,
    HOLDER_COLUMN,
    MONEYFLOW_COLUMN,
    PUBLISHED_ARM,
    REPLAY_ARM,
    SELECTABLE_ARMS,
    TRAINABLE_ARMS,
    arm_dataset_id,
    arm_feature_columns,
    arm_model_id,
    arm_trend_model_id,
    attach_arm_scores,
    criteria_table,
    derive_trend_manifest,
    gate_frozen_params,
    check_appended_columns_exist,
    gate_replay,
    read_predictions,
)


PUBLISHED_COLUMNS = ["momentum_20", "volatility_60", "sma_ratio_5"]


def published_columns(pool):
    """As many names as the pre-registration froze for that pool.

    The count is itself a gated value, so a fixture that writes three columns
    cannot exercise the gate it is meant to exercise -- the feature count would
    fail first and hide whatever the test was actually about.
    """

    return ["f_{}".format(index) for index in range(FROZEN[pool]["features"])]


def write_manifests(tmp_path, pool="hs300", top_k_outcome=0.05, top_k_ranking=0.05,
                    alpha=100.0, risk_penalty=0.25, feature_columns=None):
    model_id = "hs300_pit_outcome" if pool == "hs300" else "zz1000_pit_outcome"
    outcome = {
        "selected_alpha": alpha,
        "selected_risk_penalty": risk_penalty,
        "feature_columns": list(feature_columns if feature_columns is not None
                                else published_columns(pool)),
        "top_k": top_k_outcome,
    }
    (tmp_path / "trade_outcome_manifest_{}.json".format(model_id)).write_text(
        json.dumps(outcome), encoding="utf-8")
    (tmp_path / "trade_outcome_ranking_{}_manifest.json".format(model_id)).write_text(
        json.dumps({"selected_top_k_validation_only": top_k_ranking}), encoding="utf-8")
    return model_id


def write_trend_manifest(tmp_path, trend_id="hs300_logistic_v2", feature_columns=None):
    (tmp_path / "trend_logistic_manifest_{}.json".format(trend_id)).write_text(
        json.dumps({"feature_columns": list(feature_columns or PUBLISHED_COLUMNS),
                    "analysis_id": trend_id, "something_else": 7}), encoding="utf-8")


# ---------------------------------------------------------------------------
# arm bookkeeping
# ---------------------------------------------------------------------------

def test_the_published_and_replay_arms_share_the_published_dataset_and_feature_list():
    """A_replay is only a replay if the *inputs* are the published ones.

    If it read the E8 dataset or the E8 feature list it would be a fifth arm, and
    the gate would be checking that the treatment reproduces the treatment.
    """

    for arm in (PUBLISHED_ARM, REPLAY_ARM):
        assert arm_dataset_id("hs300", arm) == "excess_hs300_pit_v1"
        assert arm_trend_model_id("hs300", arm) == "hs300_logistic_v2"


def test_replay_does_not_overwrite_the_model_it_is_checking_against():
    assert arm_model_id("hs300", REPLAY_ARM) != arm_model_id("hs300", PUBLISHED_ARM)


def test_appending_preserves_the_published_prefix_exactly():
    """The preprocessor's leading entries are indexed by position, so order is data."""

    for arm in (BOTH_ARM, "E8_mf", "E8_hn"):
        columns = arm_feature_columns(PUBLISHED_COLUMNS, arm)

        assert columns[:len(PUBLISHED_COLUMNS)] == PUBLISHED_COLUMNS


def test_the_two_signal_arm_carries_both_columns_in_the_registered_order():
    assert arm_feature_columns(PUBLISHED_COLUMNS, BOTH_ARM) == \
        PUBLISHED_COLUMNS + [MONEYFLOW_COLUMN, HOLDER_COLUMN]


def test_a_single_signal_arm_carries_exactly_one_column():
    """These arms exist to attribute a win, so each must isolate one signal."""

    assert arm_feature_columns(PUBLISHED_COLUMNS, "E8_mf") == PUBLISHED_COLUMNS + [MONEYFLOW_COLUMN]
    assert arm_feature_columns(PUBLISHED_COLUMNS, "E8_hn") == PUBLISHED_COLUMNS + [HOLDER_COLUMN]


def test_an_unknown_arm_is_refused_rather_than_treated_as_the_published_one():
    """Silently returning the published list would make a typo a no-op A/B."""

    with pytest.raises(ValueError, match="unknown arm"):
        arm_feature_columns(PUBLISHED_COLUMNS, "E8_bothh")


def test_every_arm_is_scored_from_a_distinct_model_id():
    ids = [arm_model_id("hs300", arm) for arm in ARMS]

    assert len(set(ids)) == len(ids)


def test_the_replay_arm_is_trainable_and_reachable_from_the_command_line():
    """H1 diffs a predictions file that only this arm produces.

    Found by running it: the ``train`` stage looped over ``TRAINED_ARMS``, which
    excludes the replay arm, so ``gates`` died on a missing file and H1 could not
    run at all.  Both halves have to hold -- reachable through ``--arms`` and
    actually trained by the loop -- or the gate is again unreachable.
    """

    assert REPLAY_ARM in TRAINABLE_ARMS
    assert REPLAY_ARM in SELECTABLE_ARMS
    assert REPLAY_ARM not in ARMS          # addressable, but never a default


# ---------------------------------------------------------------------------
# the derived trend manifest
# ---------------------------------------------------------------------------

def test_the_derived_manifest_edits_the_feature_list_and_declares_itself_derived(tmp_path):
    write_trend_manifest(tmp_path)
    target = "hs300_e8_both"

    derive_trend_manifest(tmp_path, "hs300_logistic_v2", target, PUBLISHED_COLUMNS + ["new"])

    written = json.loads((tmp_path / "trend_logistic_manifest_{}.json".format(target))
                         .read_text(encoding="utf-8"))
    assert written["feature_columns"] == PUBLISHED_COLUMNS + ["new"]
    assert written["derivation"]["derived_from"] == "hs300_logistic_v2"
    assert written["derivation"]["not_a_trained_model"] is True
    assert written["derivation"]["appended_columns"] == ["new"]
    # Everything not under test is inherited verbatim, so the file cannot be
    # mistaken for an independently trained model.
    assert written["something_else"] == 7


def test_the_derived_manifest_leaves_the_published_manifest_alone(tmp_path):
    """Overwriting the source would destroy the feature list every H3 check reads."""

    write_trend_manifest(tmp_path)

    derive_trend_manifest(tmp_path, "hs300_logistic_v2", "hs300_e8_both", ["only_one"])

    source = json.loads((tmp_path / "trend_logistic_manifest_hs300_logistic_v2.json")
                        .read_text(encoding="utf-8"))
    assert source["feature_columns"] == PUBLISHED_COLUMNS


# ---------------------------------------------------------------------------
# H1b: the frozen parameters
# ---------------------------------------------------------------------------

def test_the_frozen_parameters_gate_reproduces_the_registered_table(tmp_path):
    write_manifests(tmp_path, top_k_outcome=0.05, top_k_ranking=0.05)

    report = gate_frozen_params("hs300", tmp_path, expected=FROZEN["hs300"])

    assert report["ok"] is True
    assert report["actual"]["features"] == FROZEN["hs300"]["features"]


def test_top_k_comes_from_the_ranking_manifest_not_the_outcome_one(tmp_path):
    """The two disagree on ZZ1000 (0.05 against 0.10) and only one is usable.

    The published null was drawn at the ranking manifest's width, so taking the
    outcome manifest's value would produce a run whose returns are compared against
    a null of a different basket width -- and that arrives as apparent alpha, not as
    an error.  This test exists because the wrong source is the *plausible* one.
    """

    write_manifests(tmp_path, pool="zz1000", top_k_outcome=0.05, top_k_ranking=0.10,
                    alpha=FROZEN["zz1000"]["alpha"])

    report = gate_frozen_params("zz1000", tmp_path, expected=FROZEN["zz1000"])

    assert report["actual"]["top_k"] == 0.10
    assert report["ok"] is True


def test_a_drifted_frozen_parameter_is_a_gate_failure_not_a_quiet_rerun(tmp_path):
    """The experiment is only the registered experiment at the registered values."""

    write_manifests(tmp_path, alpha=10.0)          # HS300 is registered at 100.0

    report = gate_frozen_params("hs300", tmp_path, expected=FROZEN["hs300"])

    assert report["ok"] is False
    assert report["failing"]["alpha"] == {"expected": 100.0, "actual": 10.0}


def test_the_two_pools_freeze_different_alphas(tmp_path):
    """100 / 10, not one shared default -- copying either across is the trap."""

    assert FROZEN["hs300"]["alpha"] != FROZEN["zz1000"]["alpha"]
    assert FROZEN["hs300"]["top_k"] != FROZEN["zz1000"]["top_k"]

    write_manifests(tmp_path, pool="zz1000", alpha=10.0, top_k_ranking=0.10,
                    feature_columns=["f"] * 30)
    assert gate_frozen_params("zz1000", tmp_path, expected=FROZEN["zz1000"])["ok"] is True


# ---------------------------------------------------------------------------
# scoring: the join
# ---------------------------------------------------------------------------

def write_predictions(tmp_path, model_id, split, frame):
    path = tmp_path / "trade_outcome_{}_predictions_{}.csv".format(split, model_id)
    frame.to_csv(path, index=False)
    return path


def score_frame(rows):
    return pd.DataFrame({
        "date": [row[0] for row in rows],
        "symbol": [row[1] for row in rows],
        "entry_score": [row[2] for row in rows],
    })


def test_arm_scores_land_on_their_own_rows(tmp_path):
    """Merged positionally after a sort is how a score reaches the wrong stock."""

    samples = pd.DataFrame({"date": pd.to_datetime(["2024-01-03", "2024-01-02"]),
                            "symbol": ["600519", "000001"]})
    write_predictions(tmp_path, "m_baseline", "test",
                      score_frame([("2024-01-02", "000001", 1.0),
                                   ("2024-01-03", "600519", 2.0)]))
    write_predictions(tmp_path, "m_treat", "test",
                      score_frame([("2024-01-02", "000001", 10.0),
                                   ("2024-01-03", "600519", 20.0)]))

    scored = attach_arm_scores(tmp_path, {"A_published": "m_baseline", "E8_both": "m_treat"},
                               samples, "test")

    by_symbol = dict(zip(scored["symbol"], scored["E8_both"]))
    assert by_symbol["000001"] == 10.0
    assert by_symbol["600519"] == 20.0
    assert list(zip(scored["symbol"], scored["A_published"])) == \
        [("000001", 1.0), ("600519", 2.0)]


def test_an_unscored_row_is_refused_rather_than_left_as_a_nan(tmp_path):
    """``how="left"`` would drop the name from every basket and look like a holding."""

    samples = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                            "symbol": ["000001", "600519"]})
    write_predictions(tmp_path, "m_treat", "test", score_frame([("2024-01-02", "000001", 1.0)]))

    with pytest.raises(ValueError, match="not scored"):
        attach_arm_scores(tmp_path, {"E8_both": "m_treat"}, samples, "test")


def test_a_duplicated_key_is_refused_rather_than_fanning_the_frame_out(tmp_path):
    """A fan-out keeps every row count downstream plausible while doubling names."""

    samples = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "symbol": ["000001"]})
    write_predictions(tmp_path, "m_treat", "test",
                      score_frame([("2024-01-02", "000001", 1.0),
                                   ("2024-01-02", "000001", 2.0)]))

    with pytest.raises(ValueError):
        attach_arm_scores(tmp_path, {"E8_both": "m_treat"}, samples, "test")


def test_a_missing_predictions_file_names_the_model_it_wanted(tmp_path):
    with pytest.raises(FileNotFoundError, match="e8_both"):
        read_predictions(tmp_path, "excess_hs300_pit_e8_v1_e8_both", "test")


# ---------------------------------------------------------------------------
# H1: the replay gate
# ---------------------------------------------------------------------------

def test_the_replay_gate_passes_when_the_two_runs_agree(tmp_path):
    write_predictions(tmp_path, "hs300_pit_outcome", "test",
                      score_frame([("2024-01-02", "000001", 0.5),
                                   ("2024-01-03", "600519", -0.25)]))
    write_predictions(tmp_path, arm_model_id("hs300", REPLAY_ARM), "test",
                      score_frame([("2024-01-02", "000001", 0.5),
                                   ("2024-01-03", "600519", -0.25)]))

    report = gate_replay("hs300", tmp_path)

    assert report["ok"] is True
    assert report["max_abs_delta"] == 0.0


def test_the_replay_gate_fails_on_a_fifth_decimal_difference(tmp_path):
    """The gate is a tolerance of 1e-12, so it must not pass a real difference."""

    write_predictions(tmp_path, "hs300_pit_outcome", "test",
                      score_frame([("2024-01-02", "000001", 0.5)]))
    write_predictions(tmp_path, arm_model_id("hs300", REPLAY_ARM), "test",
                      score_frame([("2024-01-02", "000001", 0.50001)]))

    report = gate_replay("hs300", tmp_path)

    assert report["ok"] is False
    assert report["max_abs_delta"] == pytest.approx(1e-5)


def test_the_replay_gate_refuses_to_compare_two_different_cross_sections(tmp_path):
    """Equal row counts are not equal universes, and the diff would blame the fit."""

    write_predictions(tmp_path, "hs300_pit_outcome", "test",
                      score_frame([("2024-01-02", "000001", 0.5)]))
    write_predictions(tmp_path, arm_model_id("hs300", REPLAY_ARM), "test",
                      score_frame([("2024-01-02", "600519", 0.5)]))

    report = gate_replay("hs300", tmp_path)

    assert report["ok"] is False
    assert "different cross-sections" in report["error"]


# ---------------------------------------------------------------------------
# the criteria
# ---------------------------------------------------------------------------

def effects(validation, test, test_min=None):
    return {
        "E8_both": {
            "validation": {"phase_mean": validation, "phase_min": min(validation, 0.0),
                           "phase_max": validation, "phases": 5},
            "test": {"phase_mean": test, "phase_min": test if test_min is None else test_min,
                     "phase_max": test, "phases": 5},
        }
    }


def test_an_arm_that_clears_the_bar_on_both_windows_passes_all_three():
    verdict = criteria_table(effects(0.04, 0.03))["E8_both"]

    assert verdict["passes_all"] is True
    assert verdict["criterion_1_phase_mean"]["passes"] is True
    assert verdict["criterion_2_phase_min"]["passes"] is True


def test_a_test_mean_below_the_bar_fails_criterion_one():
    verdict = criteria_table(effects(0.04, 0.01))["E8_both"]

    assert verdict["criterion_1_phase_mean"]["passes"] is False
    assert verdict["passes_all"] is False


def test_a_positive_mean_hiding_a_negative_phase_fails_criterion_two():
    """The mean clearing the bar while one weekday loses is the phase-spread risk."""

    verdict = criteria_table(effects(0.04, 0.03, test_min=-0.01))["E8_both"]

    assert verdict["criterion_1_phase_mean"]["passes"] is True
    assert verdict["criterion_2_phase_min"]["passes"] is False
    assert verdict["passes_all"] is False


def test_a_win_on_test_alone_fails_criterion_three():
    """Validation is the window the model was fitted against -- a bar cleared there
    and nowhere else is the ordinary overfit, and so is the reverse."""

    verdict = criteria_table(effects(-0.05, 0.03))["E8_both"]

    assert verdict["criterion_3_same_sign"]["passes"] is False
    assert verdict["passes_all"] is False


def test_a_loss_on_both_windows_still_satisfies_criterion_three():
    """Criterion 3 asks about agreement, not about winning.  It cannot rescue an
    arm -- criteria 1 and 2 still fail it -- and reading it as a second bar would
    make a consistently bad arm look like a consistently good one."""

    verdict = criteria_table(effects(-0.05, -0.03))["E8_both"]

    assert verdict["criterion_3_same_sign"]["passes"] is True
    assert verdict["criterion_1_phase_mean"]["passes"] is False
    assert verdict["passes_all"] is False


def test_only_the_both_signals_arm_is_marked_adoptable():
    table = criteria_table({
        arm: {"validation": {"phase_mean": 0.04, "phase_min": 0.02},
              "test": {"phase_mean": 0.04, "phase_min": 0.02}}
        for arm in ARMS
    })

    assert [arm for arm, verdict in table.items() if verdict["adoptable"]] == [ADOPTABLE_ARM]


# ---------------------------------------------------------------------------
# column presence
# ---------------------------------------------------------------------------

def test_the_column_check_verifies_the_names_the_feature_build_actually_writes(tmp_path):
    """A name that disagrees with the written file would surface four minutes into
    training, once per arm, as an unrelated-looking loader error."""

    path = tmp_path / "trend_probability_test_tradability_samples_excess_hs300_pit_e8_v1.csv"
    pd.DataFrame({column: [1.0] for column in APPENDED_COLUMNS}).to_csv(path, index=False)

    assert check_appended_columns_exist("hs300", tmp_path)["ok"] is True

    pd.DataFrame({"alt_moneyflow_level": [1.0]}).to_csv(path, index=False)
    report = check_appended_columns_exist("hs300", tmp_path)

    assert report["ok"] is False
    assert report["missing"] == {"stk_holdernumber": HOLDER_COLUMN}
