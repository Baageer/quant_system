"""How a configuration grid is allowed to pick a winner.

Every grid in this repo used to select the same way::

    grid.sort_values(["total_return", "turnover_multiple"]).iloc[0]

which picks *something* even when nothing on the grid beats the frozen baseline.
That is not a hypothetical.  ``hysteresis_band_ab`` ran it on validation and
selected ``exit_k=1.00`` -- an arm that traded 20 times in the whole window
(turnover 0.94x, essentially "never rebalance") -- which then scored -35.39% on
test.  ``turnover_optimization_ab`` ran it and selected
``periodic_hold30_rebal5`` at +93.55%, i.e. it selected a configuration strictly
*worse* than the +109.25% baseline it was supposed to improve on.

Two failure modes hide in that one line:

* **It never compares against the baseline.**  Ranking by absolute return is not
  the same as ranking by improvement; a candidate has to be shown to *beat*
  something, or "the best of a bad grid" gets adopted.
* **It is a point estimate.**  A single validation number carries schedule luck
  (``backtest-rebalance-phase-noise``), and a degenerate configuration can sit at
  the top of one draw.

:func:`select_configuration` replaces it with four gates, all of which a
candidate must clear -- and returns ``row=None`` ("change nothing") when none do:

0. **beats_baseline** -- strictly, always on.  ``excess > 0``, never folded into
   another gate: when gate 1 is skipped (no phase axis) folding it in there let
   the baseline clear its own turnover and fold gates and select itself.
1. **phase** -- the candidate beats the baseline in the phase *average* and in
   the *worst* phase.  A configuration that only wins on one schedule is a
   schedule, not an improvement.
2. **turnover** -- the candidate trades at least ``turnover_floor_frac`` of the
   baseline's turnover.  Necessary but not sufficient (``rolling_band_selection``
   added a floor and still chose below-baseline arms in most buckets), which is
   why it sits alongside the others rather than alone.
3. **folds** -- the candidate beats the baseline in at least
   ``min_fold_agreement`` of the validation sub-windows.  This is the "require
   the same sign across splits" gate; it is the one the validation null study
   would have supplied for free.

Gate 1's comparison must be **rank-paired**, and that is not a detail.  An
unpaired phase gate -- "does the candidate's worst phase beat the baseline's
average" -- is biased against every candidate, because the baseline is itself
one draw from a family.  On the hysteresis grid it rejected
``band_entry10_exit50``, an arm that beats v1 on *all five* weekdays, purely
because v1's Monday strike is the luckiest member of its own family (109.25%
against an 83.35% weekday average).  Gate 1 is therefore evaluated through
:func:`phase_excess_min`, which sorts both families and differences them k-th
against k-th -- so a candidate never loses points for the baseline having been
measured on a lucky schedule.

Gates 1 and 3 need data the caller may not have:

* ``phase_excess`` is ``None`` for any grid whose arms have no phase axis.
  **An ISO-week arm has none** -- its rebalance dates are pinned by the calendar
  -- so for those grids the phase axis has to come from the weekday family in
  ``turnover_optimization_ab.make_weekday_rank_strategy``, for the baseline as
  well as for the candidates.  Until it does, this gate cannot run, and the
  outcome says so.
* ``fold_excess`` is ``None`` when the caller did not collect per-arm equity.

Skipped gates are recorded in ``SelectionOutcome.gates`` and named in ``reason``
rather than being silently treated as passed, so a two-gate selection can never
be quoted as if it were three.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def fold_boundaries(start: pd.Timestamp, end: pd.Timestamp, step_months: int) -> List[Dict[str, pd.Timestamp]]:
    """Half-open folds ``[boundary, next_boundary)`` tiling ``start``..``end``.

    The final fold's exclusive end is pushed one day past ``end`` so the last
    session is traded (everything else uses ``<`` uniformly).
    """

    boundaries = []
    boundary = start
    while boundary < end:
        boundaries.append(boundary)
        boundary = boundary + pd.DateOffset(months=step_months)
    folds = []
    for index, boundary in enumerate(boundaries):
        stop = boundaries[index + 1] if index + 1 < len(boundaries) else end + pd.Timedelta(days=1)
        folds.append({"selection_end": boundary, "trade_start": boundary, "trade_end": stop})
    return folds


@dataclass
class SelectionOutcome:
    """The verdict, plus enough detail to audit it.

    ``row`` is the selected grid row, or ``None`` for "keep the baseline".  The
    ``None`` case is half the value of the rule: the old selector always
    returned a row, and on both grids it ran on, that row was worse than doing
    nothing.
    """

    row: Optional[pd.Series]
    reason: str
    table: pd.DataFrame
    gates: Dict[str, bool] = field(default_factory=dict)
    key_columns: Sequence[str] = ("arm",)

    @property
    def arm(self) -> Optional[str]:
        """The chosen config's name.

        Not every grid names its rows ``arm`` -- ``tradability_ranking_evaluation``
        is keyed by ``top_k`` alone -- so fall back to the key tuple rather than
        raising ``KeyError`` on a grid that selected perfectly well.
        """

        return None if self.row is None else _name_of(self.row, self.key_columns)

    @property
    def changed(self) -> bool:
        return self.row is not None


def _key_of(row: pd.Series, key_columns: Sequence[str]) -> tuple:
    return tuple(row[column] for column in key_columns)


def _name_of(row: pd.Series, key_columns: Sequence[str]) -> str:
    """A config's display name, for grids that do not carry an ``arm`` column."""

    if "arm" in row.index:
        return str(row["arm"])
    key = _key_of(row, key_columns)
    return str(key[0]) if len(key) == 1 else str(key)


def _locate_baseline(grid: pd.DataFrame, baseline_key: Mapping[str, object]) -> pd.Series:
    mask = pd.Series(True, index=grid.index)
    for column, value in baseline_key.items():
        if column not in grid.columns:
            raise KeyError("baseline_key names {!r}, absent from the grid".format(column))
        mask &= grid[column] == value
    matched = grid[mask]
    if len(matched) != 1:
        raise ValueError(
            "baseline_key {} matched {} rows; it must name exactly one".format(
                dict(baseline_key), len(matched)))
    return matched.iloc[0]


def phase_excess_min(candidate_returns: Sequence[float], baseline_returns: Sequence[float]) -> float:
    """Worst-case paired excess of a candidate's phase family over the baseline's.

    The two families are sorted and paired k-th with k-th, so the answer does not
    depend on an arbitrary alignment of two schedules that were never run in
    lockstep.  The value is "even the candidate's k-th worst phase beats the
    baseline's k-th worst phase", which at ``k = 0`` is the plain
    worst-candidate-phase against worst-baseline-phase comparison.

    Using ranks rather than levels is what keeps this gate from punishing a
    candidate for the baseline having been measured on a lucky schedule.
    """

    candidate = np.sort(np.asarray(candidate_returns, dtype=float))
    baseline = np.sort(np.asarray(baseline_returns, dtype=float))
    size = min(len(candidate), len(baseline))
    if size == 0:
        return float("nan")
    return float(np.min(candidate[:size] - baseline[:size]))


def phase_excess_table(
    candidate_phases: pd.DataFrame,
    baseline_phases: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    value_column: str = "total_return",
    out_column: str = "phase_excess_min",
) -> pd.DataFrame:
    """Per-config :func:`phase_excess_min`, keyed by ``key_columns``.

    ``candidate_phases`` holds one row per (config, phase) and
    ``baseline_phases`` one row per phase of the baseline's own family -- which
    for a grid whose arms are ISO-week means running the baseline through the
    weekday family as well.  No phase *identity* is needed on either side: the
    rank pairing in :func:`phase_excess_min` is alignment-free by construction.
    """

    baseline_values = baseline_phases[value_column].astype(float).tolist()
    rows = []
    for key, block in candidate_phases.groupby(list(key_columns), as_index=False):
        row = dict(zip(key_columns, key if isinstance(key, tuple) else (key,)))
        row[out_column] = phase_excess_min(block[value_column].astype(float).tolist(), baseline_values)
        row["phases"] = len(block)
        rows.append(row)
    return pd.DataFrame(rows)


def fold_returns_from_equity(
    equity: pd.Series,
    folds: Sequence[Mapping[str, pd.Timestamp]],
    initial_capital: float,
) -> List[float]:
    """Net return of each fold, marked off the equity curve.

    A fold's return is measured from the last mark *before* ``trade_start`` (or
    the initial capital, for the first fold) to the last mark strictly before
    ``trade_end``.  Using the pre-fold mark is what keeps the folds compounding
    to the whole-window return instead of double-counting the boundary session.
    """

    marks = equity.sort_index()
    values = []
    for fold in folds:
        start, stop = pd.Timestamp(fold["trade_start"]), pd.Timestamp(fold["trade_end"])
        prior = marks.index[marks.index < start]
        opening = float(marks.loc[prior[-1]]) if len(prior) else float(initial_capital)
        window = marks.index[marks.index < stop]
        if not len(window):
            values.append(float("nan"))
            continue
        values.append(float(marks.loc[window[-1]]) / opening - 1.0)
    return values


def select_configuration(
    grid: pd.DataFrame,
    *,
    baseline_key: Mapping[str, object],
    key_columns: Sequence[str] = ("arm",),
    return_column: str = "total_return",
    turnover_column: str = "turnover_multiple",
    phase_excess: Optional[pd.DataFrame] = None,
    phase_excess_column: str = "phase_excess_min",
    fold_excess: Optional[pd.DataFrame] = None,
    fold_prefix: str = "fold_",
    turnover_floor_frac: float = 0.5,
    min_fold_agreement: float = 1.0,
) -> SelectionOutcome:
    """Apply the three gates and return the winner, or ``None`` for "no change".

    ``grid`` is one row per configuration, already phase-averaged when the arms
    have a phase axis.  ``phase_excess`` is an optional frame keyed by
    ``key_columns`` holding each config's worst-case *paired* phase excess (see
    :func:`phase_excess_table`), and ``fold_excess`` an optional frame with one
    ``fold_*`` column per validation sub-window holding that fold's excess over
    the baseline on the same fold.  Both are already differenced against the
    baseline; neither is derived from the ``grid``'s absolute levels.
    """

    if return_column not in grid.columns:
        raise KeyError("grid has no {!r} column".format(return_column))
    baseline = _locate_baseline(grid, baseline_key)
    baseline_return = float(baseline[return_column])
    # The merge below resets the index, so the baseline is tracked by key rather
    # than by index label from here on.
    baseline_key_tuple = _key_of(baseline, key_columns)

    table = grid.copy()
    table["excess"] = table[return_column].astype(float) - baseline_return

    gates = {
        "phase": phase_excess is not None,
        "turnover": True,
        "folds": fold_excess is not None,
    }
    if phase_excess is not None:
        if phase_excess_column not in phase_excess.columns:
            raise ValueError(
                "phase_excess has no {!r} column; build it with "
                "phase_excess_table() so the comparison is paired".format(phase_excess_column))
        # Drop any colliding column first: the rule's own output table carries
        # these, and merging into them would silently rename ours to ``_x``/``_y``
        # and then fail (or worse, read the stale copy).
        table = table.drop(columns=[phase_excess_column], errors="ignore")
        table = table.merge(
            phase_excess[list(key_columns) + [phase_excess_column]], on=list(key_columns), how="left")

    if turnover_column not in grid.columns:
        raise KeyError("grid has no {!r} column".format(turnover_column))
    baseline_turnover = float(baseline[turnover_column])
    table["turnover_ratio"] = table[turnover_column].astype(float) / baseline_turnover

    table["fold_agreement"] = np.nan
    if fold_excess is not None:
        fold_columns = [column for column in fold_excess.columns if column.startswith(fold_prefix)]
        if not fold_columns:
            raise ValueError("fold_excess has no {!r}* columns".format(fold_prefix))
        table = table.drop(columns=fold_columns, errors="ignore")
        table = table.merge(
            fold_excess[list(key_columns) + fold_columns], on=list(key_columns), how="left")
        wins = (table[fold_columns] > 0).sum(axis=1)
        table["fold_agreement"] = wins / float(len(fold_columns))

    # Gate 1 is the phase *floor* only: a candidate that beats the baseline on
    # average but loses on its worst schedule is not an improvement yet.  The
    # average half of that sentence is gate 0 -- ``excess`` is already the
    # phase-averaged gap -- so repeating ``excess > 0`` here would make gate 1
    # merely imply gate 0 and mis-attribute every rejection to "phase".
    table["passes_phase"] = True
    if gates["phase"]:
        table["passes_phase"] = table[phase_excess_column] > 0
    table["passes_turnover"] = table["turnover_ratio"] >= float(turnover_floor_frac)
    table["passes_folds"] = True
    if gates["folds"]:
        table["passes_folds"] = table["fold_agreement"] >= float(min_fold_agreement)

    # Beating the baseline is a gate of its own, not part of gate 1: when the
    # phase axis is missing that gate is skipped, and folding the strict
    # improvement in there let the baseline clear its own turnover and fold
    # gates and select itself.
    table["passes_better"] = table["excess"] > 0
    table["passes"] = (table["passes_better"] & table["passes_phase"]
                       & table["passes_turnover"] & table["passes_folds"])
    survivors = table[table["passes"]]

    gate_columns = [("phase", "passes_phase"), ("turnover", "passes_turnover"), ("folds", "passes_folds")]
    is_baseline = pd.Series(
        [_key_of(row, key_columns) == baseline_key_tuple for _, row in table.iterrows()],
        index=table.index)
    table["reject_reason"] = ""
    for _, row in table[~table["passes"]].iterrows():
        if is_baseline.loc[row.name]:
            table.at[row.name, "reject_reason"] = "baseline"
            continue
        failed = [name for name, column in gate_columns if gates[name] and not row[column]]
        if not row["passes_better"]:
            failed.insert(0, "beats_baseline")
        table.at[row.name, "reject_reason"] = "+".join(failed)
    table = table.sort_values("excess", ascending=False).reset_index(drop=True)

    applied = ["beats_baseline"] + [name for name, ran in gates.items() if ran]
    skipped = [name for name, ran in gates.items() if not ran]

    if not len(survivors):
        detail = ", ".join(
            "{} {}".format(int((~table[column]).sum()), name)
            for name, column in gate_columns
            if gates[name] and int((~table[column]).sum()))
        reason = "no candidate beats the baseline on all of {}: {}".format(
            "+".join(applied) or "no gates", detail or "no rows")
        if skipped:
            reason += " (gates not run: {})".format(", ".join(skipped))
        return SelectionOutcome(row=None, reason=reason, table=table, gates=gates,
                                key_columns=tuple(key_columns))

    ranked = survivors.sort_values(["excess", "turnover_ratio"], ascending=[False, True])
    chosen = ranked.iloc[0]
    reason = "{} beats the baseline {:+.2%} -> {:+.2%} (gates: {}".format(
        _name_of(chosen, key_columns), baseline_return, float(chosen[return_column]),
        "+".join(applied))
    if gates["phase"]:
        reason += ", worst paired phase {:+.2%}".format(float(chosen[phase_excess_column]))
    reason += ", turnover {:.2f}x".format(float(chosen["turnover_ratio"]))
    if gates["folds"]:
        reason += ", folds {:.0%}".format(float(chosen["fold_agreement"]))
    reason += ")"
    if skipped:
        reason += " (gates not run: {})".format(", ".join(skipped))
    return SelectionOutcome(row=chosen, reason=reason, table=table, gates=gates,
                            key_columns=tuple(key_columns))
