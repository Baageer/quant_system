"""Run the engine's *own* selection and timing through a frictionless ledger.

``churn_replay_probe`` closes the accounting question (the engine's equity is
exactly its cash ledger) and the selection question (its buys are split across
exchanges the same way the pool is).  What is still open is whether the gap is
a property of the *strategy* or of the *engine's execution of it*.

So: take the identical ``prepared`` frames, the identical permuted score, and
the identical ``make_rank_strategy`` closure the engine calls -- then execute
every signal in plain numpy with fractional shares, unlimited cash and no
costs.  Same decisions, no lot rounding, no cash constraint, no commission.

If this lands near the engine, the strategy really does earn that much and
``pool_return_audit.simulate_churn`` is simply not the same strategy.  If it
lands near numpy's +2.15%, the engine is adding return somewhere in execution.

Example::

    python -m research.experiments.churn_frictionless_sim --draws 5
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from research.experiments.random_portfolio_baseline import (
    _score_slots,
    _write_rank_score,
    carry_forward_marks,
)
from research.market_regime.output_paths import DATASET_DIR
from research.market_regime.probability_trade_evaluation import _run_engine, summarize_backtest
from research.market_regime.tradability_ranking_evaluation import (
    apply_rank_selection,
    build_rank_market_data,
    make_rank_strategy,
    select_top_k,
)
from research.market_regime.trade_outcome_ranking_evaluation import SCORE_COLUMN, load_outcome_scores


def _date_index(prepared: Dict[str, pd.DataFrame]):
    mapping: Dict[pd.Timestamp, list] = {}
    for symbol, frame in prepared.items():
        for date in frame.index:
            mapping.setdefault(date, []).append(symbol)
    return mapping


def make_rank_strategy_slots_after_sells(max_positions: int, target_notional: float, min_holding_days: int = 5):
    """``make_rank_strategy`` with one change: free slots are counted after the sells.

    The production closure computes ``slots = max(max_positions - len(held), 0)``
    from the book *as it stood before* this week's sells, so a week that
    liquidates the book still buys nothing -- the freed cash waits until the
    next weekly decision.  This variant lets the same week redeploy.
    """

    entry_dates: Dict[str, pd.Timestamp] = {}
    current_week = None

    def strategy(date, relevant_data, positions):
        nonlocal current_week
        iso = pd.Timestamp(date).isocalendar()
        week = (int(iso[0]), int(iso[1]))
        if week == current_week:
            return {}
        current_week = week
        held = set(positions)
        for symbol in held:
            if symbol in relevant_data:
                entry_dates.setdefault(symbol, pd.Timestamp(date))
        for symbol in list(entry_dates):
            if symbol not in held:
                del entry_dates[symbol]
        signals = {}
        sells = 0
        for symbol in sorted(held):
            frame = relevant_data.get(symbol)
            if frame is None or bool(frame.loc[date, "hold_trade"]):
                continue
            entry_date = entry_dates.get(symbol, pd.Timestamp(date))
            entry_position = int(frame.index.searchsorted(entry_date, side="left"))
            days = int(frame.index.get_loc(date)) - entry_position
            if days >= min_holding_days:
                signals[symbol] = {"action": "sell", "shares": positions[symbol], "reason": "exit"}
                sells += 1
        slots = max(max_positions - (len(held) - sells), 0)
        candidates = []
        for symbol, frame in relevant_data.items():
            if symbol not in held and bool(frame.loc[date, "entry_trade"]):
                candidates.append((symbol, float(frame.loc[date, "rank_score"]), float(frame.loc[date, "close"])))
        for symbol, _, price in sorted(candidates, key=lambda item: (-item[1], item[0]))[:slots]:
            shares = int(np.floor(target_notional / price / 100) * 100) if price > 0 else 0
            if shares > 0:
                signals[symbol] = {"action": "buy", "shares": shares, "reason": "entry"}
        return signals

    return strategy


def frictionless_run(
    prepared: Dict[str, pd.DataFrame],
    max_positions: int,
    horizon: int,
    initial_capital: float,
    all_dates,
    strategy_factory=make_rank_strategy,
) -> Dict[str, float]:
    """The engine's strategy closure, executed with fractional shares."""

    target_notional = initial_capital / max_positions
    strategy = strategy_factory(max_positions, target_notional, min_holding_days=horizon)
    date_to_symbols = _date_index(prepared)
    marks = carry_forward_marks({s: f["close"] for s, f in prepared.items()}, all_dates)

    cash = initial_capital
    units: Dict[str, float] = {}
    exposure_sum = 0.0
    sessions = 0
    path = []

    for date in all_dates:
        relevant = {s: prepared[s] for s in date_to_symbols.get(date, [])}
        signals = strategy(date, relevant, units)
        for symbol in sorted(signals):
            signal = signals[symbol]
            frame = prepared[symbol]
            if date not in frame.index:
                continue
            price = frame.loc[date, "close"]
            if not np.isfinite(price) or price <= 0:
                continue
            if signal["action"] == "buy":
                shares = target_notional / price
                units[symbol] = units.get(symbol, 0.0) + shares
                cash -= shares * price
            else:
                held = units.get(symbol, 0.0)
                shares = min(held, float(signal["shares"]))
                if shares <= 0:
                    continue
                units[symbol] = held - shares
                cash += shares * price
                if units[symbol] <= 1e-12:
                    del units[symbol]

        stock_value = 0.0
        for symbol, shares in units.items():
            price = marks.at[date, symbol]
            if np.isfinite(price):
                stock_value += shares * price
        total = cash + stock_value
        exposure_sum += stock_value / total if total > 0 else 0.0
        sessions += 1
        path.append((date, total, stock_value / total if total > 0 else 0.0, len(units)))

    final_value = cash + stock_value
    return {
        "frictionless_return": float(final_value / initial_capital - 1.0),
        "mean_exposure": float(exposure_sum / sessions),
        "final_positions": len(units),
        "path": pd.DataFrame(path, columns=["date", "value", "exposure", "positions"]).set_index("date"),
    }


def _timing_split(path: pd.DataFrame, pool_daily: pd.Series) -> Dict[str, float]:
    """Is exposure higher on the days the pool falls or on the days it rises?"""

    daily = path["value"].pct_change()
    joined = pd.concat([daily.rename("p"), path["exposure"].rename("w"), pool_daily.rename("m")], axis=1).dropna()
    up = joined[joined["m"] > 0]
    down = joined[joined["m"] < 0]
    covariance = np.cov(joined["p"], joined["m"])
    beta = float(covariance[0, 1] / covariance[1, 1]) if covariance[1, 1] > 0 else float("nan")
    return {
        "mean_exposure_all": float(joined["w"].mean()),
        "mean_exposure_pool_up": float(up["w"].mean()),
        "mean_exposure_pool_down": float(down["w"].mean()),
        "beta_vs_pool": beta,
        "alpha_daily": float(joined["p"].mean() - beta * joined["m"].mean()),
        "alpha_annualized": float((1.0 + joined["p"].mean() - beta * joined["m"].mean()) ** 252 - 1.0),
    }


def compare(
    dataset_dir: Path,
    model_id: str,
    draws: int = 5,
    top_k: float = 0.1,
    horizon: int = 10,
    max_positions: int = 20,
    initial_capital: float = 500000.0,
    config_path: str = "config/settings.yaml",
    seed: int = 20260915,
    real_only: bool = False,
) -> pd.DataFrame:
    test = load_outcome_scores(dataset_dir, model_id, "test", horizon)
    raw_market_data, _ = build_rank_market_data(test, None, SCORE_COLUMN)
    slots = _score_slots(test, raw_market_data)
    values = np.asarray(test[SCORE_COLUMN], dtype=float)
    start_date, end_date = test["date"].min(), test["date"].max()
    rng = np.random.default_rng(seed)
    closes = pd.DataFrame({s: f["close"] for s, f in raw_market_data.items()}).sort_index()
    pool_daily = closes.pct_change(fill_method=None).mean(axis=1, skipna=True)

    rows = []
    for draw in ([-1] if real_only else range(draws)):
        # draw == -1 means "use the model's real scores": the control that says
        # whether this ledger is trustworthy at all.
        shuffled = values if draw < 0 else rng.permutation(values)
        probe = test.copy()
        probe[SCORE_COLUMN] = shuffled
        selected = select_top_k(probe, top_k, SCORE_COLUMN)
        prepared = apply_rank_selection(raw_market_data, probe, selected)
        _write_rank_score(prepared, shuffled, slots)
        all_dates = sorted({d for frame in prepared.values() for d in frame.index})

        results, trades = _run_engine(
            prepared,
            make_rank_strategy(max_positions, initial_capital / max_positions, min_holding_days=horizon),
            initial_capital,
            config_path,
            start_date,
            end_date,
        )
        engine_perf = summarize_backtest("random", results, trades, initial_capital)
        free = frictionless_run(prepared, max_positions, horizon, initial_capital, all_dates)
        patched = frictionless_run(
            prepared, max_positions, horizon, initial_capital, all_dates,
            strategy_factory=make_rank_strategy_slots_after_sells,
        )
        timing = _timing_split(free["path"], pool_daily)
        rows.append(
            {
                "draw": draw,
                "engine_return": float(engine_perf["total_return"]),
                "frictionless_return": free["frictionless_return"],
                "slots_after_sells": patched["frictionless_return"],
                "mean_exposure": free["mean_exposure"],
                "patched_exposure": patched["mean_exposure"],
                **timing,
            }
        )
        print(
            "draw {:>3}  engine={:>8.2%}  free={:>8.2%}  slots_after_sells={:>8.2%}  exp={:.0%}/{:.0%}  "
            "expo_up={:.0%} expo_down={:.0%}".format(
                draw, rows[-1]["engine_return"], rows[-1]["frictionless_return"],
                rows[-1]["slots_after_sells"], rows[-1]["mean_exposure"], rows[-1]["patched_exposure"],
                rows[-1]["mean_exposure_pool_up"], rows[-1]["mean_exposure_pool_down"],
            ), flush=True)
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--model-id", default="hs300_excess_v1_outcome")
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--top-k", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--config-path", default="config/settings.yaml")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--real-only", action="store_true", help="Control: run the model's real scores, no permutation.")
    args = parser.parse_args(argv)
    frame = compare(
        Path(args.dataset_dir), args.model_id, args.draws, args.top_k, args.horizon,
        args.max_positions, args.initial_capital, args.config_path, args.seed, args.real_only,
    )
    print()
    print(frame.to_string(index=False))
    print()
    print("engine mean       {:>9.2%}".format(frame["engine_return"].mean()))
    print("frictionless mean {:>9.2%}".format(frame["frictionless_return"].mean()))


if __name__ == "__main__":
    main()
