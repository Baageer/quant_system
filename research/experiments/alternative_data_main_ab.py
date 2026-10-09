"""E8 main body: append the two gate-passing signals as features, retrain, and A/B.

Pre-registration: ``docs/research/nine_experiments/alternative_data_main_ab_2026-09.md``.

The experiment changes exactly one thing -- the feature set -- and this module is
mostly about *proving* that, because the failure mode here is not a crash, it is
a plausible number produced for the wrong reason.

Four arms per pool:

=================  ==========================================  ==================
arm                features                                   role
=================  ==========================================  ==================
``A_published``    the published 28 / 30, disk score verbatim  baseline
``E8_both``        published + both signals                    **only adoptable**
``E8_mf``          published + ``moneyflow``                   report only
``E8_hn``          published + ``stk_holdernumber``            report only
=================  ==========================================  ==================

Only ``E8_both`` may be adopted.  Picking whichever of the three arms looks best
would be selection on the very windows the verdict is read from, which is the
mistake E9 already made once; the single-signal arms exist so that a win can be
*attributed*, not so that one of them can be promoted.

Every arm is a **disk score**: the baseline is the published predictions file and
the other three are the predictions files the production CLI writes.  Nothing here
re-implements the scoring path, so no arm can differ from its siblings because
this module did the arithmetic differently.

Four hard gates run before any result is read.  Each guards a failure that leaves
the numbers looking fine, which is why each is a gate and not a printed sanity
check:

* **H1 -- the training path is inert.**  The published inputs are run through this
  module's own training call (``A_replay``) and the score diffed against the disk
  one.  If that is not zero, *every* arm below is a measurement of the pipeline.
  ``A_replay`` is never compared against anything else and never reported as a
  result.
* **H1b -- the frozen parameters.**  ``top_k`` is two different values depending
  on which manifest is asked, and ``selected_alpha`` differs between the pools
  (100 / 10).  A wrong pick does not error; it produces a run that is not
  comparable to the published null, and **H1 cannot catch it** -- the anchor tests
  the scoring path, which is unaffected by which ``top_k`` was used.
* **H2 -- the samples.**  The E8 dataset must be the published dataset plus two
  columns and nothing else, verified against the written file rather than the
  intent.
* **H3 -- the preprocessor.**  The first k entries of each treatment's medians,
  means and scales must equal the published ones.  This is what makes "appended"
  mean appended: it proves the original columns were standardised exactly as
  before, so a difference is attributable to the two new columns and not to a
  shifted scaler.

On cost: the pre-registration schedules a **fresh 200-draw null under the new model
id**.  That null is a function of ``(universe, top_k, seed)`` alone -- the same
rows, the same width -- so it recomputes a computation that already exists and its
value here is reproducibility, not a new distribution.  It is kept because it was
registered, and ``--skip-null`` exists so that the decision to spend it is made
explicitly rather than by omission.

Example::

    python -m research.experiments.alternative_data_main_ab --stage prep
    python -m research.experiments.alternative_data_main_ab --stage ab
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.experiments.alternative_data_features import (
    SIGNALS,
    sample_path,
    signal_column,
    verify_append_only,
)
from research.experiments.cross_sectional_representation_backtest import (
    FOLD_STEP_MONTHS,
    WEEKDAYS,
    phase_mean_equity,
)
from research.experiments.model_form_reachability import load_published_model
from research.experiments.random_portfolio_baseline import _score_slots
from research.experiments.rank_target_ab import paired_phase_excess
from research.experiments.ranking_ablation_harness import (
    backtest_arm,
    fold_excess_table,
    load_null_total_returns,
)
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.selection_rule import (
    fold_boundaries,
    phase_excess_table,
    select_configuration,
)
from research.market_regime.tradability_ranking_evaluation import build_rank_market_data
from research.market_regime.trade_outcome_model import load_outcome_split, main as train_main
from research.market_regime.trade_outcome_ranking_evaluation import (
    SCORE_COLUMN,
    published_top_k,
)

PUBLISHED_ARM = "A_published"
BOTH_ARM = "E8_both"
MF_ARM = "E8_mf"
HN_ARM = "E8_hn"
#: The single arm the pre-registration permits adopting.  The other two are
#: diagnostics: if ``E8_both`` wins while both single-signal arms lose, the win
#: came from neither signal and needs explaining before it means anything.
ADOPTABLE_ARM = BOTH_ARM
REPORT_ONLY_ARMS = (MF_ARM, HN_ARM)
#: Gate only, never a result: the published inputs run through this module's own
#: training call.  If it does not reproduce the disk score the path is not inert.
REPLAY_ARM = "A_replay"

ARMS = (PUBLISHED_ARM, BOTH_ARM, MF_ARM, HN_ARM)
TRAINED_ARMS = (BOTH_ARM, MF_ARM, HN_ARM)
ALL_MODEL_ARMS = (PUBLISHED_ARM,) + TRAINED_ARMS
#: What ``--arms`` accepts.  Wider than ``ARMS``: the replay arm is trainable and
#: gate-only, so it is addressable from the command line but never a default.
SELECTABLE_ARMS = ALL_MODEL_ARMS + (REPLAY_ARM,)
#: What the ``train`` stage can produce.  ``A_replay`` belongs here and was missing:
#: H1 diffs its predictions file against the published score, so if nothing trains it
#: the gate has no file to read and cannot run at all.  Its ``arm_*`` helpers already
#: route it to the published dataset and the published feature list, which is what
#: makes the training call a *replay* rather than a fourth treatment -- only the loop
#: could not reach it.
TRAINABLE_ARMS = TRAINED_ARMS + (REPLAY_ARM,)

HORIZON = 10
INITIAL_CAPITAL = 500000.0
MAX_POSITIONS = 20
#: H1.  A tolerance rather than bit equality: the replay refits the same Ridge on
#: the same rows, and the published score was produced by that same fit, so float
#: reassociation is the only admissible source of difference.
ANCHOR_TOLERANCE = 1e-12
#: H3.  Same argument one level down: the scaler is recomputed from the same
#: training rows over the same leading columns.
PREPROCESSOR_TOLERANCE = 1e-12
MIN_PHASE_MEAN = 0.02

APPENDED_COLUMNS = tuple("alt_{}_{}".format(clock.name, clock.form) for clock in SIGNALS)
#: ``moneyflow`` first, ``stk_holdernumber`` second -- the registered order.
MONEYFLOW_COLUMN, HOLDER_COLUMN = APPENDED_COLUMNS

#: The parameters frozen by the pre-registration, per pool, with the file each one
#: is read from at run time.  Written here as literals so a change on disk shows up
#: as a gate failure rather than as a silently different experiment.
FROZEN: Dict[str, Dict[str, object]] = {
    "hs300": {"top_k": 0.05, "risk_penalty": 0.25, "alpha": 100.0, "features": 28},
    "zz1000": {"top_k": 0.10, "risk_penalty": 0.25, "alpha": 10.0, "features": 30},
}

POOLS: Dict[str, Dict[str, object]] = {
    "hs300": {
        "model_id": "hs300_pit_outcome",
        "dataset_id": "excess_hs300_pit_v1",
        "e8_dataset_id": "excess_hs300_pit_e8_v1",
        "trend_model_id": "hs300_logistic_v2",
        "null_tag": "e8_pit",
    },
    "zz1000": {
        "model_id": "zz1000_pit_outcome",
        "dataset_id": "excess_zz1000_pit_v1",
        "e8_dataset_id": "excess_zz1000_pit_e8_v1",
        "trend_model_id": "zz1000_logistic_v1",
        "null_tag": "e8_pit",
    },
}


def arm_model_id(pool: str, arm: str) -> str:
    """The outcome model id this arm is scored from.

    ``A_replay`` gets its own id rather than overwriting the published model: the
    published pickle is the reference the gate compares against, and a run that
    could rewrite its own reference is not a gate.
    """

    spec = POOLS[pool]
    if arm == PUBLISHED_ARM:
        return str(spec["model_id"])
    if arm == REPLAY_ARM:
        return "{}_{}".format(spec["dataset_id"], arm.lower())
    return "{}_{}".format(spec["e8_dataset_id"], arm.lower())


def arm_dataset_id(pool: str, arm: str) -> str:
    """The samples this arm reads.  ``A_replay`` deliberately uses the published ones."""

    spec = POOLS[pool]
    if arm in (PUBLISHED_ARM, REPLAY_ARM):
        return str(spec["dataset_id"])
    return str(spec["e8_dataset_id"])


def arm_trend_model_id(pool: str, arm: str) -> str:
    """The trend manifest carrying this arm's feature list."""

    if arm in (PUBLISHED_ARM, REPLAY_ARM):
        return str(POOLS[pool]["trend_model_id"])
    return "{}_{}".format(POOLS[pool]["e8_dataset_id"], arm.lower())


def arm_feature_columns(published_columns: Sequence[str], arm: str) -> List[str]:
    """Published columns, in order, plus this arm's appended columns.

    Never a re-selection: the published list is copied verbatim and the new columns
    are appended, so the preprocessor's leading entries stay put and the only thing
    an arm varies is what it adds.
    """

    columns = list(published_columns)
    if arm in (PUBLISHED_ARM, REPLAY_ARM):
        return columns
    if arm not in TRAINED_ARMS:
        raise ValueError("unknown arm: {!r}".format(arm))
    if arm in (BOTH_ARM, MF_ARM):
        columns.append(MONEYFLOW_COLUMN)
    if arm in (BOTH_ARM, HN_ARM):
        columns.append(HOLDER_COLUMN)
    return columns


def derive_trend_manifest(dataset_dir: Path, source_trend_id: str, target_trend_id: str,
                          feature_columns: Sequence[str]) -> Dict[str, object]:
    """Write a feature-list carrier by copying the published manifest.

    This is **not** a trained model and says so in the file.  ``trade_outcome_model``
    reads the trend manifest only through ``load_trend_feature_columns``, which wants
    nothing but ``feature_columns`` -- and *that* is the point: training a fresh trend
    model would run its own candidate selection, which is exactly the re-selection the
    pre-registration forbids.  Copying the manifest edits the one field under test and
    leaves every other recorded value visibly inherited.
    """

    source_path = dataset_dir / "trend_logistic_manifest_{}.json".format(source_trend_id)
    target_path = dataset_dir / "trend_logistic_manifest_{}.json".format(target_trend_id)
    manifest = json.loads(source_path.read_text(encoding="utf-8"))
    published_columns = list(manifest["feature_columns"])
    manifest["feature_columns"] = list(feature_columns)
    manifest["analysis_id"] = target_trend_id
    manifest["derivation"] = {
        "derived_from": source_trend_id,
        "not_a_trained_model": True,
        "reason": ("Trade-outcome training reads this file only for feature_columns; "
                   "training a fresh trend model would re-run candidate selection and "
                   "could displace frozen features."),
        "published_feature_count": len(published_columns),
        "appended_columns": [column for column in feature_columns
                             if column not in set(published_columns)],
    }
    target_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(target_path), "feature_columns": list(feature_columns),
            "appended": manifest["derivation"]["appended_columns"]}


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def read_frozen_params(pool: str, dataset_dir: Path) -> Dict[str, object]:
    """Read the four frozen parameters off disk, each from its own file.

    ``top_k`` comes from ``published_top_k``, i.e. the **ranking** manifest, and
    deliberately not from the outcome manifest: the two disagree on ZZ1000
    (0.10 there, 0.05 here) and the ranking one is the value the published null was
    drawn at.  Taking the other would produce a run whose numbers are compared
    against a null drawn at a different basket width -- a mismatch that arrives
    looking like alpha rather than like an error.
    """

    spec = POOLS[pool]
    outcome = json.loads(
        (dataset_dir / "trade_outcome_manifest_{}.json".format(spec["model_id"]))
        .read_text(encoding="utf-8"))
    return {
        "top_k": float(published_top_k(dataset_dir, str(spec["model_id"]))),
        "risk_penalty": float(outcome["selected_risk_penalty"]),
        "alpha": float(outcome["selected_alpha"]),
        "features": len(outcome["feature_columns"]),
        "outcome_manifest": str(dataset_dir / "trade_outcome_manifest_{}.json".format(
            spec["model_id"])),
    }


def gate_frozen_params(pool: str, dataset_dir: Path,
                       expected: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    """H1b.  The values read off disk must be the ones the pre-registration froze."""

    expected = dict(FROZEN[pool] if expected is None else expected)
    actual = read_frozen_params(pool, dataset_dir)
    mismatches = {}
    for key, want in expected.items():
        got = actual[key]
        same = (abs(float(got) - float(want)) <= 1e-12 if isinstance(want, float)
                else int(got) == int(want))
        if not same:
            mismatches[key] = {"expected": want, "actual": got}
    return {"gate": "H1b", "pool": pool, "failing": mismatches, "actual": actual,
            "expected": expected, "ok": not mismatches}


def gate_append_only(pool: str, dataset_dir: Path) -> Dict[str, object]:
    """H2.  The E8 samples are the published samples plus the two registered columns."""

    spec = POOLS[pool]
    splits = {}
    for split in ("train", "validation", "test"):
        splits[split] = verify_append_only(
            sample_path(dataset_dir, str(spec["dataset_id"]), split),
            sample_path(dataset_dir, str(spec["e8_dataset_id"]), split),
            list(APPENDED_COLUMNS))
    return {"gate": "H2", "pool": pool, "splits": splits,
            "ok": all(entry["ok"] for entry in splits.values())}


def gate_preprocessor_prefix(pool: str, dataset_dir: Path, arm: str,
                             tolerance: float = PREPROCESSOR_TOLERANCE) -> Dict[str, object]:
    """H3.  The first k scaler entries are the published ones, entry for entry.

    Compares ``medians``/``means``/``scales`` positionally over the first k, where k
    is the published feature count.  If the appended columns had been *inserted* or
    the selection re-run, the prefix would still be the right length and the arrays
    would still be finite -- only the values would differ -- so the comparison is
    elementwise and the worst offender is reported rather than just a boolean.
    """

    spec = POOLS[pool]
    published = load_published_model(dataset_dir, str(spec["model_id"]))
    treatment = load_published_model(dataset_dir, arm_model_id(pool, arm))
    k = len(published["feature_columns"])
    worst = {}
    for key in ("medians", "means", "scales"):
        base = np.asarray(published["preprocessor"][key], dtype=float)[:k]
        other = np.asarray(treatment["preprocessor"][key], dtype=float)[:k]
        if base.shape != other.shape:
            worst[key] = "shape {} vs {}".format(base.shape, other.shape)
            continue
        worst[key] = float(np.max(np.abs(base - other))) if len(base) else 0.0
    ok = all(isinstance(value, float) and value <= tolerance for value in worst.values())
    return {"gate": "H3", "pool": pool, "arm": arm, "prefix_length": k,
            "max_abs_delta": worst, "tolerance": tolerance, "ok": ok,
            "appended_columns": [column for column in treatment["feature_columns"]
                                 if column not in set(published["feature_columns"])]}


def gate_replay(pool: str, dataset_dir: Path, split: str = "test",
                tolerance: float = ANCHOR_TOLERANCE) -> Dict[str, object]:
    """H1.  The published inputs through this module's training call reproduce the score.

    Both sides are the predictions files the same exporter wrote, so this diffs the
    *whole* arrow rather than a reconstructed score: dataset, feature list, alpha,
    penalty and ``top_k`` all flow in through the same arguments the other arms use.
    """

    spec = POOLS[pool]
    published = read_predictions(dataset_dir, str(spec["model_id"]), split)
    replay = read_predictions(dataset_dir, arm_model_id(pool, REPLAY_ARM), split)
    if len(published) != len(replay):
        return {"gate": "H1", "pool": pool, "split": split, "ok": False,
                "error": "row count {} vs {}".format(len(published), len(replay))}
    keys = ["date", "symbol"]
    a = published.sort_values(keys).reset_index(drop=True)
    b = replay.sort_values(keys).reset_index(drop=True)
    if not a[keys].equals(b[keys]):
        return {"gate": "H1", "pool": pool, "split": split, "ok": False,
                "error": "the two runs scored different cross-sections"}
    delta = float(np.max(np.abs(np.asarray(a[SCORE_COLUMN], dtype=float)
                                - np.asarray(b[SCORE_COLUMN], dtype=float))))
    return {"gate": "H1", "pool": pool, "split": split, "rows": int(len(a)),
            "max_abs_delta": delta, "tolerance": tolerance, "ok": delta <= tolerance}


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def read_predictions(dataset_dir: Path, model_id: str, split: str) -> pd.DataFrame:
    """One predictions file, keyed on ``(date, symbol)`` and typed like its siblings."""

    path = dataset_dir / "trade_outcome_{}_predictions_{}.csv".format(split, model_id)
    if not path.exists():
        raise FileNotFoundError(
            "no predictions for model_id={!r} split={!r}; train it first ({})".format(
                model_id, split, path))
    frame = pd.read_csv(path, dtype={"symbol": str}, low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if SCORE_COLUMN not in frame.columns:
        raise ValueError("{} has no {} column".format(path.name, SCORE_COLUMN))
    return frame.loc[:, ["date", "symbol", SCORE_COLUMN]]


def attach_arm_scores(dataset_dir: Path, model_ids: Mapping[str, str], samples: pd.DataFrame,
                      split: str) -> pd.DataFrame:
    """Merge one score column per arm onto ``samples``, on ``(date, symbol)``.

    ``validate="one_to_one"``: a duplicated key would fan the frame out and every
    downstream row count would still look plausible.  An unmatched row is raised
    rather than left as NaN -- a missing score would drop the name from every basket
    it belongs in and read as a strategy that simply held something else.
    """

    merged = samples.loc[:, ["date", "symbol"]].copy()
    for arm, model_id in model_ids.items():
        disk = read_predictions(dataset_dir, model_id, split).rename(
            columns={SCORE_COLUMN: arm})
        merged = merged.merge(disk, on=["date", "symbol"], how="left", validate="one_to_one")
        if len(merged) != len(samples):
            raise ValueError("{}: score join changed the row count {} -> {}".format(
                arm, len(samples), len(merged)))
        missing = int(merged[arm].isnull().sum())
        if missing:
            raise ValueError("{}: {} rows were not scored by {}".format(arm, missing, model_id))
    merged = merged.sort_values(["date", "symbol"]).reset_index(drop=True)
    out = samples.sort_values(["date", "symbol"]).reset_index(drop=True).copy()
    for arm in model_ids:
        out[arm] = np.asarray(merged[arm], dtype=float)
    return out


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train_arm(pool: str, arm: str, dataset_dir: Path, frozen: Mapping[str, object],
              output_dir: Optional[Path] = None) -> str:
    """Run the production CLI for one arm and return its model id.

    The CLI is called rather than reimplemented: the arms have to be exactly what
    the published pipeline produces, and a second implementation of it would make
    every arm a measurement of this module.  Alpha and risk penalty are handed in as
    one-element grids so the CLI's own selection can only return the frozen value --
    it still runs, it just has nothing to choose.
    """

    run_id = arm_model_id(pool, arm)
    argv = [
        "--dataset-dir", str(dataset_dir),
        "--dataset-id", arm_dataset_id(pool, arm),
        "--trend-model-id", arm_trend_model_id(pool, arm),
        "--run-id", run_id,
        "--output-dir", str(output_dir or dataset_dir),
        "--horizon", str(HORIZON),
        "--alphas", repr(float(frozen["alpha"])),
        "--risk-penalties", repr(float(frozen["risk_penalty"])),
        "--top-k", repr(float(frozen["top_k"])),
    ]
    train_main(argv)
    return run_id


def prepare_pool(pool: str, dataset_dir: Path) -> Dict[str, object]:
    """Write the three derived trend manifests.  Nothing is trained here."""

    spec = POOLS[pool]
    published = list(load_published_model(dataset_dir, str(spec["model_id"]))["feature_columns"])
    manifests = {}
    for arm in TRAINED_ARMS:
        columns = arm_feature_columns(published, arm)
        derived = derive_trend_manifest(dataset_dir, str(spec["trend_model_id"]),
                                       arm_trend_model_id(pool, arm), columns)
        manifests[arm] = {"appended": derived["appended"], "features": len(columns)}
    return {"pool": pool, "published_features": len(published), "manifests": manifests}


def check_appended_columns_exist(pool: str, dataset_dir: Path) -> Dict[str, object]:
    """Every arm's feature list must be present as a column in the E8 samples.

    Cheap, and it catches the one mistake ``derive_trend_manifest`` cannot: a name
    that disagrees with what the feature build actually wrote.  The loader would
    raise on it too, but it would raise once per arm, four minutes into training.
    """

    spec = POOLS[pool]
    header = pd.read_csv(sample_path(dataset_dir, str(spec["e8_dataset_id"]), "test"),
                         nrows=0).columns
    # The name comes from the feature build rather than being respelled here: a second
    # spelling is how the manifest and the file drift apart one character at a time.
    columns = {signal.name: signal_column(signal) for signal in SIGNALS}
    missing = {key: value for key, value in columns.items() if value not in header}
    return {"pool": pool, "columns": columns, "missing": missing, "ok": not missing}


# ---------------------------------------------------------------------------
# engine A/B
# ---------------------------------------------------------------------------

def evaluate_pool(pool: str, dataset_dir: Path, arms: Sequence[str] = ARMS,
                  horizon: int = HORIZON, initial_capital: float = INITIAL_CAPITAL,
                  max_positions: int = MAX_POSITIONS, price_history_dir: Optional[str] = None,
                  config_path: str = "config/settings.yaml",
                  price_history_dirs: Optional[Sequence[str]] = None,
                  weekdays: Sequence[int] = WEEKDAYS,
                  null: Optional[np.ndarray] = None,
                  baseline_arm: str = PUBLISHED_ARM) -> Dict[str, object]:
    """Score every arm on the shared cross-section and run the weekday family.

    The samples are loaded with the **published** feature columns and the arm scores
    are then joined onto them from disk.  Both halves of that are deliberate: the
    feature list fixes which rows count as eligible and is identical across arms (that
    is what makes this a one-variable A/B), and taking the scores from the exported
    files means the engine is handed the same numbers a replication of the published
    run would be handed, rather than a recomputation of them.
    """

    spec = POOLS[pool]
    frozen = read_frozen_params(pool, dataset_dir)
    top_k = float(frozen["top_k"])
    print("\n{}  model={}  top_k={} (ranking manifest)  alpha={}  risk_penalty={}  "
          "features={} (published)".format(pool, spec["model_id"], top_k, frozen["alpha"],
                                           frozen["risk_penalty"], frozen["features"]))

    model_ids = {arm: arm_model_id(pool, arm) for arm in arms}
    published_model = load_published_model(dataset_dir, str(spec["model_id"]))
    feature_columns = list(published_model["feature_columns"])

    scored, phase_rows, runs = {}, {}, {}
    raw, slots = {}, {}
    for split in ("validation", "test"):
        samples = load_outcome_split(dataset_dir, str(spec["e8_dataset_id"]), split,
                                     feature_columns, horizon)
        if split == "test":
            print("  test rows: {} (must equal the published split's count -- H4)".format(
                len(samples)))
        scored[split] = attach_arm_scores(dataset_dir, model_ids, samples, split)
        raw[split], skipped = build_rank_market_data(
            scored[split], price_history_dir, baseline_arm, price_history_dirs=price_history_dirs)
        if not raw[split]:
            raise ValueError("missing price data for {} {}".format(pool, split))
        slots[split] = _score_slots(scored[split], raw[split])
        print("  {} skipped prices: {}".format(split, skipped if skipped else "none"))

        rows = []
        for weekday in weekdays:
            for arm in arms:
                run = backtest_arm(
                    scored[split], raw[split], slots[split], arm,
                    scored[split]["date"].min(), scored[split]["date"].max(),
                    top_k, horizon, initial_capital, max_positions, config_path,
                    weekday=int(weekday), label=arm)
                runs[(split, arm, int(weekday))] = run
                rows.append({
                    "split": split, "weekday": int(weekday), "arm": arm,
                    **{key: value for key, value in run["performance"].items() if key != "strategy"},
                })
        phase_rows[split] = pd.DataFrame(rows)
        print("  {} engine runs: {} weekdays x {} arms".format(split, len(weekdays), len(arms)))

    effects = {}
    for arm in arms:
        if arm == baseline_arm:
            continue
        per_split = {}
        for split in ("validation", "test"):
            paired = paired_phase_excess(phase_rows[split], baseline_arm, arm)
            per_split[split] = {
                "phase_mean": float(paired.mean()),
                "phase_min": float(paired.min()),
                "phase_max": float(paired.max()),
                "phases": int(len(paired)),
            }
        effects[arm] = per_split

    verdicts = criteria_table(effects, baseline_arm=baseline_arm)
    print("\n  paired phase excess vs {} (percentage points):".format(baseline_arm))
    for arm in effects:
        validation, test = effects[arm]["validation"], effects[arm]["test"]
        print("    {:<11} validation {:+7.2f} (min {:+7.2f})   test {:+7.2f} "
              "(min {:+7.2f}, max {:+7.2f}){}".format(
                  arm, 100 * validation["phase_mean"], 100 * validation["phase_min"],
                  100 * test["phase_mean"], 100 * test["phase_min"], 100 * test["phase_max"],
                  "   [adoptable]" if arm == ADOPTABLE_ARM else "   [report only]"))

    summary = phase_rows["test"].groupby("arm", as_index=False).mean(numeric_only=True)
    percentiles = {}
    if null is not None:
        # The percentile alone is not readable: ">= 95th" has been shown twice now to
        # hold on a strong arm whatever the arm does, so the margin to the null's own
        # upper bound is reported beside it (E7's discipline).  Both are report-only.
        upper = float(null.max())
        for arm in arms:
            value = float(summary.loc[summary["arm"] == arm, "total_return"].iloc[0])
            percentiles[arm] = {
                "actual": value,
                "percentile": float((null <= value).mean()),
                "margin_to_upper_bound": value - upper,
            }
        print("\n  test vs the {}-draw null (mean {:+.2%}, max {:+.2%}):".format(
            len(null), null.mean(), upper))
        for arm in arms:
            entry = percentiles[arm]
            print("    {:<11} {:>8.2%}  pct={:>6.1%}  margin={:+8.2%}{}".format(
                arm, entry["actual"], entry["percentile"], entry["margin_to_upper_bound"],
                "   <-- the BASELINE, so this row cannot discriminate" if arm == baseline_arm
                else ""))

    gates = four_gates(pool, dataset_dir, scored, runs, phase_rows, arms, weekdays,
                       initial_capital, baseline_arm)
    print("\n  four gates (REPORTED ONLY -- not a criterion):")
    print("    {} -- {}".format("NO CHANGE" if gates.row is None else gates.row["arm"], gates.reason))

    return {
        "pool": pool, "top_k": top_k, "frozen": frozen, "effects": effects,
        "verdicts": verdicts, "percentiles": percentiles, "null_mean":
            (float(null.mean()) if null is not None else None),
        "four_gates": {"row": (None if gates.row is None else dict(gates.row)),
                       "reason": gates.reason},
        "phase_rows": {split: frame for split, frame in phase_rows.items()},
    }


def criteria_table(effects: Mapping[str, Mapping[str, Mapping[str, float]]],
                   baseline_arm: str = PUBLISHED_ARM) -> Dict[str, Dict[str, object]]:
    """The pre-registered three criteria, one verdict per arm.

    Criterion 3 is *same sign*, not "validation also clears the bar": validation is
    the window the model was fitted against and a treatment that only clears the bar
    there is the ordinary overfit.  What disqualifies an arm is the two windows
    disagreeing, which is a statement about the sign and nothing more.
    """

    verdicts = {}
    for arm, per_split in effects.items():
        test, validation = per_split["test"], per_split["validation"]
        c1 = test["phase_mean"] >= MIN_PHASE_MEAN
        c2 = test["phase_min"] > 0.0
        c3 = (test["phase_mean"] > 0) == (validation["phase_mean"] > 0)
        verdicts[arm] = {
            "criterion_1_phase_mean": {"value": test["phase_mean"], "threshold": MIN_PHASE_MEAN,
                                       "passes": bool(c1)},
            "criterion_2_phase_min": {"value": test["phase_min"], "threshold": 0.0,
                                      "passes": bool(c2)},
            "criterion_3_same_sign": {"test": test["phase_mean"],
                                      "validation": validation["phase_mean"],
                                      "passes": bool(c3)},
            "adoptable": arm == ADOPTABLE_ARM,
            "passes_all": bool(c1 and c2 and c3),
        }
    return verdicts


def four_gates(pool: str, dataset_dir: Path, scored: Mapping[str, pd.DataFrame],
               runs: Mapping[Tuple[str, str, int], Mapping[str, object]],
               phase_rows: Mapping[str, pd.DataFrame], arms: Sequence[str],
               weekdays: Sequence[int], initial_capital: float, baseline_arm: str):
    """The ``select_configuration`` gate set, on the validation window.

    Reported only.  The fold gate sits at its configured default and is known to be
    permanently closed there, so a "four gates" row is a description of the
    incumbent's footing and never a criterion -- see
    ``docs/research/02_model_and_target/improvement_experiment_plans_2026-09.md`` P0-4.
    """

    validation = phase_rows["validation"]
    folds = fold_boundaries(scored["validation"]["date"].min(),
                            scored["validation"]["date"].max(), FOLD_STEP_MONTHS)
    equity = {
        arm: phase_mean_equity({
            int(weekday): runs[("validation", arm, int(weekday))]["equity"]
            for weekday in weekdays})
        for arm in arms
    }
    fold_excess = fold_excess_table(equity, folds, initial_capital, baseline_arm=baseline_arm)
    phase_excess = phase_excess_table(validation, validation[validation["arm"] == baseline_arm],
                                      key_columns=("arm",))
    grid = pd.DataFrame([{
        "arm": arm,
        "validation_total_return": float(validation.groupby("arm")["total_return"].mean().loc[arm]),
        "validation_turnover_multiple": float(
            validation.groupby("arm")["turnover_multiple"].mean().loc[arm]),
    } for arm in arms])
    return select_configuration(
        grid, baseline_key={"arm": baseline_arm}, key_columns=("arm",),
        return_column="validation_total_return", turnover_column="validation_turnover_multiple",
        fold_excess=fold_excess, phase_excess=phase_excess)


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True,
                        choices=("prep", "train", "gates", "ab", "report"))
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--pools", nargs="+", default=sorted(POOLS), choices=sorted(POOLS))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(SELECTABLE_ARMS))
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--price-history-dirs", nargs="+", default=None)
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--skip-null", action="store_true",
                        help="do not reload the published null (it is not a criterion)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_dir)
    frozen = {pool: read_frozen_params(pool, dataset_dir) for pool in args.pools}

    if args.stage == "prep":
        report = {
            "frozen": {pool: gate_frozen_params(pool, dataset_dir) for pool in args.pools},
            "columns": {pool: check_appended_columns_exist(pool, dataset_dir)
                        for pool in args.pools},
            "append_only": {pool: gate_append_only(pool, dataset_dir) for pool in args.pools},
            "manifests": {pool: prepare_pool(pool, dataset_dir) for pool in args.pools},
        }
        for pool in args.pools:
            entry = report["frozen"][pool]
            print("\n{} frozen params: {}".format(
                pool, "OK" if entry["ok"] else "MISMATCH {}".format(entry["failing"])))
            print("  H2 append-only: {}".format(
                "OK" if report["append_only"][pool]["ok"] else "FAILED"))
            if not report["append_only"][pool]["ok"]:
                for split, detail in report["append_only"][pool]["splits"].items():
                    if not detail["ok"]:
                        print("    {}: differing={} unexpected={} order_ok={}".format(
                            split, detail["differing_shared_columns"],
                            detail["unexpected_new_columns"], detail["column_order_preserved"]))
            columns = report["columns"][pool]
            print("  appended columns present: {}".format("OK" if columns["ok"] else columns["missing"]))
            print("  manifests: {}".format(report["manifests"][pool]["manifests"]))
        ok = (all(entry["ok"] for entry in report["frozen"].values())
              and all(entry["ok"] for entry in report["append_only"].values())
              and all(entry["ok"] for entry in report["columns"].values()))
        return 0 if ok else 1

    if args.stage == "train":
        for pool in args.pools:
            for arm in TRAINABLE_ARMS:
                if arm not in args.arms:
                    continue
                print("\n== {} {} ==".format(pool, arm))
                train_arm(pool, arm, dataset_dir, frozen[pool])
        return 0

    if args.stage == "gates":
        replay, preprocessor = {}, {}
        for pool in args.pools:
            print("\n== {} ==".format(pool))
            replay[pool] = gate_replay(pool, dataset_dir)
            print("  H1 replay of the published score: max |delta| = {} -> {}".format(
                replay[pool].get("max_abs_delta", replay[pool].get("error")),
                "OK" if replay[pool]["ok"] else "FAILED"))
            preprocessor[pool] = {}
            for arm in TRAINED_ARMS:
                if arm not in args.arms:
                    continue
                entry = gate_preprocessor_prefix(pool, dataset_dir, arm)
                preprocessor[pool][arm] = entry
                print("  H3 {:<8} prefix of {} scaler entries: {} -> {}".format(
                    arm, entry["prefix_length"], entry["max_abs_delta"],
                    "OK" if entry["ok"] else "FAILED"))
        ok = (all(entry["ok"] for entry in replay.values())
              and all(entry["ok"] for pool in preprocessor.values()
                      for entry in pool.values()))
        return 0 if ok else 1

    if args.stage in ("ab", "report"):
        results = {}
        for pool in args.pools:
            null = None
            if not args.skip_null:
                # ``rows`` is the treatment cross-section's own row count, read from
                # the same loader the run uses -- ``load_null_total_returns`` rejects a
                # null drawn on any other split, and a row count taken from a
                # different loader would be a second opinion about the universe.
                columns = list(load_published_model(
                    dataset_dir, str(POOLS[pool]["model_id"]))["feature_columns"])
                rows = len(load_outcome_split(dataset_dir, str(POOLS[pool]["e8_dataset_id"]),
                                              "test", columns, HORIZON))
                null = load_null_total_returns(
                    dataset_dir, arm_model_id(pool, BOTH_ARM), str(POOLS[pool]["null_tag"]),
                    float(frozen[pool]["top_k"]), rows, max_positions=MAX_POSITIONS)
            results[pool] = evaluate_pool(
                pool, dataset_dir, arms=args.arms, price_history_dir=args.price_history_dir,
                config_path=args.config, price_history_dirs=args.price_history_dirs, null=null)
        output = Path(args.output) if args.output else \
            dataset_dir / "alternative_data_main_ab_report.json"
        payload = {pool: {key: value for key, value in result.items() if key != "phase_rows"}
                   for pool, result in results.items()}
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n[write] {}".format(output))

        print("\n== verdicts ==")
        for pool, result in results.items():
            for arm, verdict in result["verdicts"].items():
                print("  {:<8} {:<11} {}  ({} of 3)".format(
                    pool, arm,
                    "PASS" if verdict["passes_all"] else "FAIL",
                    sum(verdict[key]["passes"] for key in
                        ("criterion_1_phase_mean", "criterion_2_phase_min",
                         "criterion_3_same_sign"))))
        return 0

    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
