"""Score the 2026 forward samples with the frozen models, and prove the chain.

Two arms, per the approved plan:

* **A_frozen** (primary) -- the published pickle, applied as-is.  Nothing is
  refit: the pickle carries ``feature_columns`` + ``preprocessor``
  (median impute -> standardize) + two Ridge heads, and
  ``transform_features`` / ``build_outcome_predictions`` are called with them
  verbatim, so the score is produced by the same code path that produced the
  published ``trade_outcome_test_predictions_*.csv``.
* **B_boundary** (control) -- same recipe, refit on every published sample row
  at or before the boundary.  This separates "the model is stale" from "the
  alpha decayed": if A and B land in the same place, staleness is not the story.

The parity check is the load-bearing part
-----------------------------------------
A forward number is worthless if my feature panel, preprocessor and model
application differ from the published chain by even a little -- a silent recipe
drift would look exactly like decay.  So arm A re-scores the **published test
end date** and is compared, row by row, against the ``entry_score`` already on
disk for that date:

    same date x same symbols x |delta entry_score| < 1e-6

That is the only check that can distinguish "the alpha decayed" from "I built
the features slightly differently".  It runs before any result is reported, and
``--require-parity`` turns it into a hard gate.

Known-good reference point: 237 rows for hs300, 891 for zz1000 on 2025-12-02.

Example::

    python -m research.experiments.score_forward_2026 --limit-symbols 40
    python -m research.experiments.score_forward_2026 --require-parity
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.trade_outcome_model import (
    _fit_models,
    build_outcome_predictions,
    load_outcome_split,
)
from research.market_regime.logistic_model import fit_preprocessor, transform_features

FORWARD_ROOT = Path("data/raw/tushare/forward_2026")

#: pool -> (frozen model id, published dataset id it was trained from)
FROZEN_MODELS = {
    "hs300": ("hs300_pit_outcome", "excess_hs300_pit_v1"),
    "zz1000": ("zz1000_pit_outcome", "excess_zz1000_pit_v1"),
}

HORIZON = 10

#: The published test window's last session.  Arm A is re-scored here so it can
#: be compared against the stored predictions; that comparison is what licenses
#: the 2026 numbers.
PARITY_DATE = pd.Timestamp("2025-12-02")
PARITY_TOLERANCE = 1e-6
PARITY_EXPECTED_ROWS = {"hs300": 237, "zz1000": 891}

#: Splits each boundary-refit arm fits on.  All published rows are at or before
#: 2025-12-02, so "everything published" is exactly "everything at the boundary".
#:
#: ``boundary`` uses the test split too.  That is deliberate -- arm B's job is to
#: be the *most* optimistic version of a fresh model, which is the right
#: direction for a control -- but it leaves an obvious objection: maybe the edge
#: comes from having been fit on the test window rather than from being fresh.
#: ``boundary_tv`` therefore drops the test split, keeping recency (it still
#: trains through 2021 validation end, plus all of validation) and volume as the
#: only differences from ``frozen``.  If both boundary arms clear the band and
#: ``frozen`` does not, the reading is about refitting, not about that window.
BOUNDARY_SPLITS = {
    "boundary": ("train", "validation", "test"),
    "boundary_tv": ("train", "validation"),
}


def load_model(dataset_dir: Path, model_id: str) -> Dict[str, object]:
    """The published pickle: feature list, preprocessor, two Ridge heads."""

    path = dataset_dir / "trade_outcome_model_{}.pkl".format(model_id)
    with path.open("rb") as handle:
        model = pickle.load(handle)
    missing = [key for key in ("feature_columns", "preprocessor", "excess_model", "mae_model")
               if key not in model]
    if missing:
        raise ValueError("{} is missing {}".format(path, missing))
    return model


def load_manifest(dataset_dir: Path, model_id: str) -> Dict[str, object]:
    return json.loads((dataset_dir / "trade_outcome_manifest_{}.json".format(model_id))
                      .read_text(encoding="utf-8"))


def apply_model(samples: pd.DataFrame, model: Dict[str, object], risk_penalty: float,
                label: str, horizon: int = HORIZON) -> pd.DataFrame:
    """Score rows through the production transform + prediction path.

    ``transform_features`` reads ``preprocessor["feature_columns"]``, not the
    frame's columns, so a missing feature column fails loudly instead of being
    silently skipped -- which is the failure mode that would fake a decay.

    Rows are sorted by ``(date, symbol)`` *before* predicting, and that matters
    for more than tidiness.  ``build_outcome_predictions`` returns its frame
    sorted, while the raw predictions come back in ``samples`` order, so
    attaching them to the result is only correct when the two orders already
    agree.  The samples files are concatenations of per-batch frames, so they are
    not globally sorted; the two bare raw columns below were therefore silently
    scrambled onto the wrong rows -- a different scramble every time the window
    grew, because the frame's length changed the sort permutation.  Sorting up
    front makes the association order-independent by construction.  Values in
    ``entry_score`` and every ``_10d`` column are per-row functions and are
    untouched by this: they were always correct.
    """

    absent = [c for c in model["preprocessor"]["feature_columns"] if c not in samples.columns]
    if absent:
        raise ValueError("samples are missing {} frozen feature columns, e.g. {}".format(
            len(absent), absent[:8]))
    rows = samples.sort_values(["date", "symbol"]).reset_index(drop=True)
    features = transform_features(rows, model["preprocessor"])
    predicted_excess = model["excess_model"].predict(features)
    predicted_mae = model["mae_model"].predict(features)
    predictions = build_outcome_predictions(
        rows, predicted_excess, predicted_mae, horizon, risk_penalty, label)
    predictions["predicted_excess_return"] = predicted_excess
    predictions["predicted_mae"] = predicted_mae
    return predictions


def refit_at_boundary(dataset_dir: Path, dataset_id: str, feature_columns: Sequence[str],
                      alpha: float, risk_penalty: float, splits: Sequence[str],
                      horizon: int = HORIZON) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Arm B: refit the preprocessor and both heads at the boundary."""

    frames = []
    counts = {}
    for split in splits:
        part = load_outcome_split(dataset_dir, dataset_id, split, feature_columns, horizon)
        counts[split] = int(len(part))
        if not part.empty:
            frames.append(part)
    if not frames:
        raise SystemExit("no published samples available to refit {} at the boundary".format(dataset_id))
    boundary = pd.concat(frames, ignore_index=True)

    preprocessor = fit_preprocessor(boundary, feature_columns)
    features = transform_features(boundary, preprocessor)
    excess_column, mae_column = "future_excess_return_{}d".format(horizon), "future_mae_{}d".format(horizon)
    excess_model, mae_model = _fit_models(
        features,
        np.asarray(pd.to_numeric(boundary[excess_column], errors="raise"), dtype=float),
        np.asarray(pd.to_numeric(boundary[mae_column], errors="raise"), dtype=float),
        alpha,
    )
    model = {"feature_columns": list(feature_columns), "preprocessor": preprocessor,
             "excess_model": excess_model, "mae_model": mae_model}
    detail = {
        "rows": int(len(boundary)),
        "splits": list(splits),
        "rows_by_split": counts,
        "date_min": str(pd.to_datetime(boundary["date"]).min().date()),
        "date_max": str(pd.to_datetime(boundary["date"]).max().date()),
        "alpha": float(alpha),
        "risk_penalty": float(risk_penalty),
    }
    return model, detail


def _published_sample_lookup(dataset_dir: Path, dataset_id: str, date: pd.Timestamp,
                             symbols: Sequence[str]) -> pd.DataFrame:
    """The published *sample* rows (not predictions) for ``date``/``symbols``.

    Needed to classify a symbol that appears on my side but not in the stored
    predictions: either the published run never had the row (a real universe
    difference) or it had it and *dropped* it (a filter difference, which the
    published sample file records with its own eligibility flag).
    """

    path = dataset_dir / "trend_probability_test_tradability_samples_{}.csv".format(dataset_id)
    header = pd.read_csv(path, nrows=0).columns
    wanted = [c for c in ("date", "symbol", "auxiliary_label_eligible",
                          "auxiliary_label_reason", "future_excess_return_10d",
                          "future_mae_10d") if c in header]
    frame = pd.read_csv(path, dtype={"symbol": str}, low_memory=False, usecols=wanted)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    if "auxiliary_label_reason" not in frame.columns:
        frame["auxiliary_label_reason"] = "unknown"
    frame = frame.loc[(frame["date"] == date) & (frame["symbol"].isin(list(symbols)))]
    return frame


def parity_check(samples: pd.DataFrame, model: Dict[str, object], dataset_dir: Path,
                 pool: str, model_id: str, dataset_id: str, risk_penalty: float) -> dict:
    """Re-score the published test end date and diff against the stored scores.

    Four things are compared, not one:

    * ``entry_score`` -- the whole point: my features + preprocessor + model vs
      the published ones;
    * both **label** columns -- if my price frame or benchmark were built
      differently, the labels would move even when the scores agreed, so this
      catches a class of error the score comparison alone cannot;
    * the symbol sets -- via a join, so a matching count with mismatched symbols
      cannot pass;
    * any symbol only I have is looked up in the published *sample* file, so
      "the published run dropped this row" is distinguished from "the published
      run never had it".  Only the latter is a universe difference and fails.

    The last point is not hypothetical.  `002049.SZ` and `688012.SH` appear in
    my build on 2025-12-02 but not in the stored predictions, because the
    published samples file carries label values for those rows while flagging
    them ``auxiliary_label_eligible=False`` with a NaN availability date -- an
    internal inconsistency at its own boundary.  Both are outside the forward
    window, so neither affects a 2026 number; the check says so out loud instead
    of either failing on it or hiding it.
    """

    stored_path = dataset_dir / "trade_outcome_test_predictions_{}.csv".format(model_id)
    stored = pd.read_csv(stored_path, dtype={"symbol": str})
    stored["date"] = pd.to_datetime(stored["date"], errors="coerce")
    stored = stored.loc[stored["date"] == PARITY_DATE,
                        ["symbol", "entry_score", "predicted_excess_return_10d",
                         "future_excess_return_10d", "future_mae_10d"]]
    stored["symbol"] = stored["symbol"].astype(str).str.zfill(6)

    rows = samples.loc[pd.to_datetime(samples["date"]) == PARITY_DATE].copy()
    rows["symbol"] = rows["symbol"].astype(str).str.zfill(6)
    if rows.empty:
        return {"pool": pool, "model_id": model_id, "status": "NO_ROWS",
                "detail": "no forward samples on {}".format(PARITY_DATE.date())}

    mine = apply_model(rows, model, risk_penalty, "parity_check")
    mine = mine.loc[:, ["symbol", "entry_score", "predicted_excess_return_10d",
                        "future_excess_return_10d", "future_mae_10d"]]
    mine = mine.rename(columns={c: c + "_mine" for c in mine.columns if c != "symbol"})
    merged = stored.merge(mine, on="symbol", how="outer", indicator=True)

    both = merged.loc[merged["_merge"] == "both"]
    only_stored = merged.loc[merged["_merge"] == "left_only", "symbol"].tolist()
    only_mine = merged.loc[merged["_merge"] == "right_only", "symbol"].tolist()

    def worst_delta(column: str) -> float:
        if both.empty:
            return float("inf")
        a = both[column].to_numpy(dtype=float)
        b = both[column + "_mine"].to_numpy(dtype=float)
        return float(np.max(np.abs(a - b))) if len(a) else float("inf")

    worst = worst_delta("entry_score")
    worst_excess_pred = worst_delta("predicted_excess_return_10d")
    worst_label_excess = worst_delta("future_excess_return_10d")
    worst_label_mae = worst_delta("future_mae_10d")
    delta = np.abs(both["entry_score"].to_numpy(dtype=float)
                   - both["entry_score_mine"].to_numpy(dtype=float)) if len(both) else np.array([])

    # Classify the rows I have and the published run does not.
    explained, unexplained = [], []
    if only_mine:
        published = _published_sample_lookup(dataset_dir, dataset_id, PARITY_DATE, only_mine)
        flagged = dict(zip(published["symbol"], zip(published["auxiliary_label_eligible"],
                                                    published["auxiliary_label_reason"])))
        for symbol in only_mine:
            if symbol in flagged:
                eligible, reason = flagged[symbol]
                explained.append({"symbol": symbol, "eligible_in_published": bool(eligible),
                                  "reason": str(reason)})
            else:
                unexplained.append(symbol)

    expected = PARITY_EXPECTED_ROWS.get(pool)
    passed = (len(both) > 0 and not only_stored and not unexplained
              and worst <= PARITY_TOLERANCE and worst_label_excess <= PARITY_TOLERANCE
              and worst_label_mae <= PARITY_TOLERANCE
              and (expected is None or len(both) == expected))
    return {
        "pool": pool,
        "model_id": model_id,
        "parity_date": str(PARITY_DATE.date()),
        "status": "PASS" if passed else "FAIL",
        "stored_rows": int(len(stored)),
        "mine_rows": int(len(rows)),
        "matched_symbols": int(len(both)),
        "only_in_stored": only_stored[:20],
        "only_in_mine_explained": explained,
        "only_in_mine_unexplained": unexplained,
        "bit_identical_rows": int(np.sum(delta <= 1e-12)) if len(delta) else 0,
        "tolerance": PARITY_TOLERANCE,
        "max_abs_entry_score_delta": worst,
        "max_abs_pred_excess_delta": worst_excess_pred,
        "max_abs_label_excess_delta": worst_label_excess,
        "max_abs_label_mae_delta": worst_label_mae,
        "expected_rows": expected,
        "worst_offenders": (both.assign(delta=delta).nlargest(5, "delta")
                            .loc[:, ["symbol", "entry_score", "entry_score_mine", "delta"]]
                            .to_dict(orient="records")) if len(both) else [],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", default="hs300,zz1000")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--forward-root", default=str(FORWARD_ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit-symbols", type=int, default=None)
    parser.add_argument("--arms", default="frozen,boundary",
                        help="any of: frozen, boundary (train+val+test), boundary_tv (train+val)")
    parser.add_argument("--require-parity", action="store_true",
                        help="exit non-zero unless the parity check passes")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    forward_root = Path(args.forward_root)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    report, parity_rows = {}, []
    for pool in [p.strip() for p in args.pools.split(",") if p.strip()]:
        model_id, dataset_id = FROZEN_MODELS[pool]
        # ``build_forward_2026_samples`` writes the sample tables into the dataset
        # dir alongside every other published sample table, not into the raw tree.
        samples_path = dataset_dir / "forward_2026_samples_{}.csv".format(pool)
        samples = pd.read_csv(samples_path, dtype={"symbol": str}, low_memory=False)
        samples["date"] = pd.to_datetime(samples["date"], errors="coerce")
        samples["symbol"] = samples["symbol"].astype(str).str.zfill(6)
        if args.limit_symbols:
            keep = sorted(samples["symbol"].unique())[: args.limit_symbols]
            samples = samples.loc[samples["symbol"].isin(keep)].reset_index(drop=True)
        print("\n=== [{}] {} rows, {} symbols, {}..{} ===".format(
            pool, len(samples), samples["symbol"].nunique(),
            samples["date"].min().date(), samples["date"].max().date()), flush=True)

        manifest = load_manifest(dataset_dir, model_id)
        risk_penalty = float(manifest["selected_risk_penalty"])
        pool_report = {"model_id": model_id, "dataset_id": dataset_id,
                       "rows": int(len(samples)),
                       "symbols": int(samples["symbol"].nunique()),
                       "risk_penalty": risk_penalty, "arms": {}}

        if "frozen" in arms:
            frozen = load_model(dataset_dir, model_id)
            print("[{}] A_frozen: {} features".format(pool, len(frozen["feature_columns"])), flush=True)
            predictions = apply_model(samples, frozen, risk_penalty, "forward_frozen")
            path = output_dir / "forward_2026_scores_{}_frozen.csv".format(pool)
            predictions.to_csv(path, index=False, encoding="utf-8-sig")
            pool_report["arms"]["frozen"] = {
                "path": str(path), "rows": int(len(predictions)),
                "date_min": str(predictions["date"].min().date()),
                "date_max": str(predictions["date"].max().date()),
                "features": int(len(frozen["feature_columns"])),
            }
            print("[{}] wrote {}".format(pool, path), flush=True)

            check = parity_check(samples, frozen, dataset_dir, pool, model_id, dataset_id,
                                 risk_penalty)
            parity_rows.append(check)
            print("[{}] parity {}: {}/{} symbols matched, bit-identical {}, "
                  "max |d entry_score| = {:.3e}, max |d label| = {:.3e}".format(
                      pool, check["status"], check.get("matched_symbols"), check.get("stored_rows"),
                      check.get("bit_identical_rows"),
                      check.get("max_abs_entry_score_delta", float("nan")),
                      max(check.get("max_abs_label_excess_delta", 0.0),
                          check.get("max_abs_label_mae_delta", 0.0))), flush=True)
            for row in check.get("only_in_mine_explained", []):
                print("    only-in-mine (published dropped it): {}".format(row), flush=True)
            if check["status"] != "PASS":
                print("[{}] parity detail: only_stored={} unexplained={}".format(
                    pool, check.get("only_in_stored"),
                    check.get("only_in_mine_unexplained")), flush=True)
                for row in check.get("worst_offenders", []):
                    print("    {}".format(row), flush=True)

        for arm, splits in BOUNDARY_SPLITS.items():
            if arm not in arms:
                continue
            refit, detail = refit_at_boundary(dataset_dir, dataset_id, manifest["feature_columns"],
                                              float(manifest["selected_alpha"]), risk_penalty, splits)
            print("[{}] {} refit on {} rows ({}), alpha={}".format(
                pool, arm, detail["rows"], detail["rows_by_split"], detail["alpha"]), flush=True)
            predictions = apply_model(samples, refit, risk_penalty, "forward_" + arm)
            path = output_dir / "forward_2026_scores_{}_{}.csv".format(pool, arm)
            predictions.to_csv(path, index=False, encoding="utf-8-sig")
            pool_report["arms"][arm] = dict(detail, path=str(path), rows=int(len(predictions)))
            print("[{}] wrote {}".format(pool, path), flush=True)

        report[pool] = pool_report

    parity_dir = forward_root
    (parity_dir / "scoring_manifest.json").write_text(json.dumps({
        "horizon": HORIZON,
        "parity_date": str(PARITY_DATE.date()),
        "parity_tolerance": PARITY_TOLERANCE,
        "parity": parity_rows,
        "pools": report,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\nmanifest -> {}".format(parity_dir / "scoring_manifest.json"))

    failed = [row["pool"] for row in parity_rows if row["status"] != "PASS"]
    if failed:
        print("\n*** PARITY FAILED for {} -- forward numbers are not licensed ***".format(failed))
        return 2 if args.require_parity else 0
    if not parity_rows:
        # Arm B alone does not license anything: the parity check re-scores the
        # frozen model, so skipping arm A means the forward numbers were never
        # validated.  Say that instead of printing a pass that did not run.
        print("\nno parity check ran (arm A_frozen was not in --arms) -- the forward "
              "numbers are NOT licensed by this run")
        return 3 if args.require_parity else 0
    print("\nparity PASS for all pools: the forward chain matches the published one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
