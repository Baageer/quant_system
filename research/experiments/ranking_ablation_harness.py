"""Engine plumbing shared by the score-comparison experiments (E1, E2, E3...).

Every ablation in ``docs/research/02_model_and_target/improvement_experiment_plans_2026-09.md`` has the same shape:
build several candidate ``entry_score`` vectors over one cross-section, push each
of them through the *unmodified* production chain, and compare the resulting
equity curves against a baseline arm and a like-for-like random null.  Only the
score construction differs; the plumbing is this module.

Two traps live here, and both are silent:

* **The score has to be written into ``rank_score``, not just ``entry_score``.**
  ``build_rank_market_data`` attaches the score to each price frame, and
  ``make_rank_strategy`` breaks entry ties on that column -- so swapping
  ``entry_score`` alone would leave every arm ranked by the first arm's scores.
  ``_write_rank_score`` rewrites it from the arm's own vector, which is the same
  trick ``random_portfolio_baseline`` uses for its permutations.
* **Turnover gates compare within one window.**  ``fold_excess_table`` and the
  validation grid must both read validation against validation; reading the test
  column into the validation grid compares two windows.

Extracted from ``two_head_ablation.py`` once a second experiment needed it.  The
functions are unchanged there; that module re-exports them so its existing tests
keep pinning the same names.
"""

import json
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from research.experiments.random_portfolio_baseline import _write_rank_score
from research.market_regime.probability_trade_evaluation import _run_engine, summarize_backtest
from research.market_regime.selection_rule import fold_returns_from_equity
from research.market_regime.tradability_ranking_evaluation import (
    apply_rank_selection,
    make_rank_strategy,
    select_top_k,
)
from research.market_regime.turnover_optimization_ab import make_weekday_rank_strategy
from research.market_regime.trade_outcome_model import target_columns
from research.market_regime.trade_outcome_ranking_evaluation import SCORE_COLUMN


def backtest_arm(
    scores: pd.DataFrame,
    raw_market_data: Mapping[str, pd.DataFrame],
    slots: Dict[str, tuple],
    arm: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    top_k: float,
    horizon: int,
    initial_capital: float,
    max_positions: int,
    config_path: str,
    score_column: str = SCORE_COLUMN,
    weekday: Optional[int] = None,
    label: Optional[str] = None,
    selection_filter: Optional[Callable[[pd.Series], pd.Series]] = None,
) -> Dict[str, object]:
    """Run one arm through the unmodified production chain.

    ``arm`` names the column of ``scores`` holding this arm's vector; it is also
    the label the performance table carries, so callers should pass something
    that identifies the arm in the output.

    ``selection_filter`` post-processes the basket between ``select_top_k`` and
    the execution mapping, which is the only seam a *selection-side* candidate
    has: the scores, the engine, the sizing and the schedule are all untouched,
    so a difference in the equity curve is attributable to which names were
    allowed in.  It is a callable rather than a cap parameter because the harness
    has no business knowing what the filter is filtering on -- E5b caps by
    industry, and anything else that rewrites the basket belongs at this seam
    too.  The filter must return a boolean Series aligned to ``selected``.

    ``label`` overrides that output label without changing the column read.  A
    grid whose cells differ only in *construction* (``top_k``, ``max_positions``)
    reuses one score column for every cell, so arm and label stop being the same
    string and the performance table would otherwise be 36 rows all called
    ``entry_score``.

    ``weekday=None`` is the production ISO week.  A weekday 0..4 strikes the week
    on that day instead and exists so a candidate can be compared against the
    baseline over the whole *family* of schedules rather than on the one schedule
    that happens to be in production -- see ``hysteresis_band_ab`` and
    ``docs/research/05_engine_and_methodology/weekday_phase_axis.md``.  ``weekday=0`` reproduces the
    production strategy exactly.
    """

    frame = scores.copy()
    values = np.asarray(frame[arm], dtype=float)
    frame[score_column] = values
    selected = select_top_k(frame, top_k, score_column)
    if selection_filter is not None:
        filtered = selection_filter(selected)
        if len(filtered) != len(selected):
            raise ValueError(
                "selection_filter returned {} rows, expected {}".format(len(filtered), len(selected)))
        # Positional, because ``apply_rank_selection`` reads it with ``np.asarray``
        # and would silently take a misaligned index as gospel.
        selected = pd.Series(np.asarray(filtered, dtype=bool), index=selected.index)
    prepared = apply_rank_selection(raw_market_data, frame, selected)
    _write_rank_score(prepared, values, slots)
    strategy = (
        make_rank_strategy(max_positions, initial_capital / max_positions, min_holding_days=horizon)
        if weekday is None else
        make_weekday_rank_strategy(max_positions, initial_capital / max_positions, horizon,
                                   weekday=int(weekday))
    )
    results, trades = _run_engine(
        prepared,
        strategy,
        initial_capital,
        config_path,
        start_date,
        end_date,
    )
    return {
        "performance": summarize_backtest(
            arm if label is None else str(label), results, trades, initial_capital),
        "equity": results["portfolio_value"].astype(float),
        "selected": selected,
        "frame": frame,
        # The full engine frame, so a caller can audit *what was held* rather than
        # only what it was worth.  ``positions`` is ``{symbol: shares}`` at each
        # close and ``cash`` is the other half of ``portfolio_value`` -- together
        # they let a caller rebuild the equity curve and prove the holdings it
        # measured are the engine's, which a weights-only view cannot do.
        "results": results,
    }


def basket_downside(scores: pd.DataFrame, selected: pd.Series, horizon: int) -> Dict[str, float]:
    """What the selected names actually did -- the volatility-tilt caveat.

    A basket picked by a downside-aware score is partly a *low-realised-downside*
    basket.  These numbers say whether an arm won by picking better names or by
    holding quieter ones -- which is a risk exposure, not selection.
    """

    excess_column, mae_column = target_columns(horizon)
    mask = np.asarray(selected, dtype=bool)
    return {
        "selected_mean_future_mae": float(scores.loc[mask, mae_column].mean()),
        "selected_mean_future_excess": float(scores.loc[mask, excess_column].mean()),
        "universe_mean_future_mae": float(scores[mae_column].mean()),
    }


def fold_excess_table(
    equity: Mapping[str, pd.Series],
    folds: Sequence[Mapping[str, pd.Timestamp]],
    initial_capital: float,
    baseline_arm: str,
) -> pd.DataFrame:
    """Per-fold return of each arm minus ``baseline_arm``'s on the *same* fold.

    The baseline's own row is identically zero, which is what makes it useless to
    the fold gate -- and why gate 0 (``excess > 0``) is a separate gate rather
    than folded into this one.  ``baseline_arm`` is a parameter because the
    experiments name their baselines differently and a module-level default would
    silently produce an all-zero table for anyone who forgot to rename.
    """

    per_fold = {
        arm: fold_returns_from_equity(series, folds, initial_capital)
        for arm, series in equity.items()
    }
    if baseline_arm not in per_fold:
        raise ValueError("no equity series for baseline_arm={!r}; have {}".format(
            baseline_arm, sorted(per_fold)))
    baseline = per_fold[baseline_arm]
    rows = []
    for arm in sorted(per_fold):
        rows.append(
            {
                "arm": arm,
                **{
                    "fold_{}".format(index): float(value) - float(base)
                    for index, (value, base) in enumerate(zip(per_fold[arm], baseline))
                },
            }
        )
    return pd.DataFrame(rows)


def load_null_total_returns(
    dataset_dir: Path,
    model_id: str,
    tag: str,
    top_k: float,
    rows: int,
    max_positions: Optional[int] = None,
) -> np.ndarray:
    """The published null's ``total_return`` column, verified to fit this arm.

    The tag alone is not trusted.  Each pool also has same-width nulls drawn on
    *other* windows -- HS300's forward-2026 null is also ``top_k=0.05`` -- so the
    checks below are what makes this the null for the cross-section under test:
    same model, same basket width, and the same number of scored rows (the
    forward null has 36767 against the published test split's 218898).

    ``max_positions`` is the third leg of the same-width requirement and is
    checked whenever the caller passes it: the permutation draws a basket of
    ``ceil(n * top_k)`` names and the engine then caps it at ``max_positions``
    *and* sizes each name at ``capital / max_positions``, so two nulls with the
    same ``top_k`` and different caps are not interchangeable -- one of them
    holds the same names at half the exposure.  ``top_k`` and the row count alone
    would accept it.  It defaults to ``None`` because the published nulls'
    ``max_positions`` may be absent in an older summary; a caller that knows the
    cap it ran at should pass it.

    The row count is read from ``split_rows`` (current writer) or ``test_rows``
    (summaries written before the writer gained ``--split``); see the comment at
    the check.
    """

    draws = dataset_dir / "random_portfolio_{}_{}_draws.csv".format(model_id, tag)
    summary_path = dataset_dir / "random_portfolio_{}_{}_summary.json".format(model_id, tag)
    if not draws.exists() or not summary_path.exists():
        raise ValueError(
            "no published null for model_id={!r} tag={!r} at {}; run "
            "random_portfolio_baseline first".format(model_id, tag, draws))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    problems = []
    if summary.get("model_id") != model_id:
        problems.append("model_id={!r}".format(summary.get("model_id")))
    if abs(float(summary["top_k"]) - float(top_k)) > 1e-12:
        problems.append("top_k={} (need {})".format(summary["top_k"], top_k))
    # ``split_rows`` (current writer) or ``test_rows`` (summaries written before the
    # writer gained ``--split``).  The key was renamed and this reader was not, so
    # every null the current writer produces was rejected for a *missing key* rather
    # than for any mismatch -- found when E8's own null came back as
    # "test_rows=None (need 218898)".  ``pool_expansion_zz500`` had already worked
    # around the same drift at its own call site; the fix belongs here instead.
    # The numeric check is unchanged: whichever key is present must match.
    drawn_rows = summary.get("split_rows", summary.get("test_rows"))
    if int(drawn_rows if drawn_rows is not None else -1) != int(rows):
        problems.append("rows={} (need {}; looked at split_rows/test_rows)".format(
            drawn_rows, rows))
    if (summary.get("split") or "test") != "test":
        problems.append("split={!r}".format(summary.get("split")))
    if max_positions is not None:
        if "max_positions" not in summary:
            problems.append("max_positions=absent (need {})".format(max_positions))
        elif int(summary["max_positions"]) != int(max_positions):
            problems.append("max_positions={} (need {})".format(
                summary["max_positions"], max_positions))
    if problems:
        raise ValueError(
            "null {} is not for this cross-section: {}".format(summary_path.name, "; ".join(problems)))
    return np.asarray(pd.read_csv(draws)["total_return"], dtype=float)
