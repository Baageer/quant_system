"""Where does the random-score churn arm's excess return actually come from?

Established so far:

* the engine's random-20 arm (+15.85%, 200-draw mean) is **not** an accounting or
  execution artifact -- replaying the engine's own fills matches to 11 decimals,
  and an independent frictionless numpy ledger driven by the engine's own
  strategy closure reproduces the arm (+16.66% / +21.00% on shuffled scores);
* the same ledger fed the model's *real* scores reproduces the published v1
  result (+48.58% vs the engine's +45.98%), so the ledger is trustworthy;
* a plain numpy "random 20, equal weight, full replacement every N days" earns
  +2.15%, i.e. it tracks the *rebalanced* pool, not the arm.

So the excess is a property of the strategy closure, not of the engine.  This
probe asks the last question: is it the **selection** (which names are held on
which days) or the **weighting** (fixed 25k lots, never rebalanced, partial
cash)?

Two measurements, per draw, on shuffled scores:

1. **Same holdings, equal weights.**  Replay the identical signal stream, then
   mark the book to equal value every day instead of letting winners run.  If
   the return collapses toward the pool, the excess was the weight path.
2. **Forward returns of what was bought.**  Each buy's own fwd-21d return
   against the pool's cross-sectional fwd-21d return on the same date.  If the
   buys have no edge, the selection is genuinely neutral and only the weight
   path can be paying.

Example::

    python -m research.experiments.churn_attribution_probe --draws 3
"""

import argparse
import json
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
from research.market_regime.tradability_ranking_evaluation import (
    apply_rank_selection,
    build_rank_market_data,
    make_rank_strategy,
    select_top_k,
)
from research.market_regime.trade_outcome_ranking_evaluation import SCORE_COLUMN, load_outcome_scores


def ledger(
    prepared: Dict[str, pd.DataFrame],
    max_positions: int,
    horizon: int,
    initial_capital: float,
    all_dates,
    equal_weight: bool = False,
) -> Dict[str, object]:
    """Execute the production closure with fractional shares and no costs.

    ``equal_weight`` re-marks every held name to the same value each session
    once the signals have been applied -- identical holdings and identical cash,
    only the weight *within* the book differs.
    """

    target_notional = initial_capital / max_positions
    strategy = make_rank_strategy(max_positions, target_notional, min_holding_days=horizon)

    date_to_symbols: Dict[pd.Timestamp, list] = {}
    for symbol, frame in prepared.items():
        for date in frame.index:
            date_to_symbols.setdefault(date, []).append(symbol)

    price_of = {symbol: frame["close"] for symbol, frame in prepared.items()}
    marks = carry_forward_marks(price_of, all_dates)

    cash = initial_capital
    units: Dict[str, float] = {}
    buys = []
    exposure_sum = 0.0
    sessions = 0
    value_path = []

    for date in all_dates:
        relevant = {s: prepared[s] for s in date_to_symbols.get(date, [])}
        signals = strategy(date, relevant, units)
        for symbol in sorted(signals):
            signal = signals[symbol]
            series = price_of[symbol]
            if date not in series.index:
                continue
            price = series.loc[date]
            if not np.isfinite(price) or price <= 0:
                continue
            if signal["action"] == "buy":
                shares = target_notional / price
                units[symbol] = units.get(symbol, 0.0) + shares
                cash -= shares * price
                buys.append((symbol, date, float(price)))
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

        if equal_weight and len(units) > 0 and stock_value > 0:
            per_name = stock_value / len(units)
            for symbol in list(units):
                price = marks.at[date, symbol]
                units[symbol] = per_name / price if np.isfinite(price) and price > 0 else 0.0

        total = cash + stock_value
        exposure_sum += stock_value / total if total > 0 else 0.0
        sessions += 1
        value_path.append((date, total, stock_value / total if total > 0 else 0.0, len(units)))

    return {
        "return": float(total / initial_capital - 1.0),
        "mean_exposure": float(exposure_sum / sessions),
        "buys": buys,
        "path": pd.DataFrame(value_path, columns=["date", "value", "exposure", "positions"]).set_index("date"),
    }


def forward_returns(
    prepared: Dict[str, pd.DataFrame], buys, horizon: int
) -> Dict[str, float]:
    """Do the names the closure buys beat the pool over the next ``horizon`` days?"""

    closes = pd.DataFrame({s: f["close"] for s, f in prepared.items()}).sort_index()
    fwd = closes.shift(-horizon) / closes - 1.0
    pool_mean = fwd.mean(axis=1, skipna=True)

    bought, pool_at_buy = [], []
    for symbol, date, _ in buys:
        if date not in fwd.index:
            continue
        own = fwd.at[date, symbol] if symbol in fwd.columns else np.nan
        pool = pool_mean.get(date, np.nan)
        if np.isfinite(own) and np.isfinite(pool):
            bought.append(own)
            pool_at_buy.append(pool)

    bought = np.asarray(bought, dtype=float)
    pool_at_buy = np.asarray(pool_at_buy, dtype=float)
    return {
        "buys_scored": int(len(bought)),
        "mean_fwd_bought": float(bought.mean()) if len(bought) else float("nan"),
        "mean_fwd_pool_same_dates": float(pool_at_buy.mean()) if len(pool_at_buy) else float("nan"),
        "edge_per_buy": float((bought - pool_at_buy).mean()) if len(bought) else float("nan"),
        "t_stat": float(
            (bought - pool_at_buy).mean() / ((bought - pool_at_buy).std(ddof=1) / np.sqrt(len(bought)))
        ) if len(bought) > 1 else float("nan"),
    }


def run(
    dataset_dir: Path,
    model_id: str,
    draws: int = 3,
    top_k: float = 0.1,
    horizon: int = 10,
    max_positions: int = 20,
    initial_capital: float = 500000.0,
    seed: int = 20260915,
) -> pd.DataFrame:
    test = load_outcome_scores(dataset_dir, model_id, "test", horizon)
    raw_market_data, _ = build_rank_market_data(test, None, SCORE_COLUMN)
    slots = _score_slots(test, raw_market_data)
    values = np.asarray(test[SCORE_COLUMN], dtype=float)
    rng = np.random.default_rng(seed)

    closes = pd.DataFrame({s: f["close"] for s, f in raw_market_data.items()}).sort_index()
    pool_bh = float((closes.iloc[-1] / closes.iloc[0] - 1.0).mean())
    pool_median_bh = float((closes.iloc[-1] / closes.iloc[0] - 1.0).median())
    print("pool equal-weight buy-and-hold {:>8.2%}   median {:>8.2%}".format(pool_bh, pool_median_bh))

    rows = []
    for draw in range(draws):
        probe = test.copy()
        probe[SCORE_COLUMN] = rng.permutation(values)
        selected = select_top_k(probe, top_k, SCORE_COLUMN)
        prepared = apply_rank_selection(raw_market_data, probe, selected)
        _write_rank_score(prepared, np.asarray(probe[SCORE_COLUMN], dtype=float), slots)
        all_dates = sorted({d for frame in prepared.values() for d in frame.index})

        plain = ledger(prepared, max_positions, horizon, initial_capital, all_dates)
        equal = ledger(prepared, max_positions, horizon, initial_capital, all_dates, equal_weight=True)
        fwd = forward_returns(prepared, plain["buys"], horizon)

        rows.append(
            {
                "draw": draw,
                "return": plain["return"],
                "equal_weight_return": equal["return"],
                "mean_exposure": plain["mean_exposure"],
                "buys": len(plain["buys"]),
                **fwd,
            }
        )
        print(
            "draw {:>3}  plain={:>8.2%}  equal_weight={:>8.2%}  exp={:.0%}  buys={}  "
            "fwd_bought={:>7.2%}  fwd_pool={:>7.2%}  edge={:>7.3%} (t={:.2f})".format(
                draw, rows[-1]["return"], rows[-1]["equal_weight_return"], rows[-1]["mean_exposure"],
                rows[-1]["buys"], rows[-1]["mean_fwd_bought"], rows[-1]["mean_fwd_pool_same_dates"],
                rows[-1]["edge_per_buy"], rows[-1]["t_stat"],
            ),
            flush=True,
        )
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--model-id", default="hs300_excess_v1_outcome")
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--top-k", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args(argv)
    frame = run(
        Path(args.dataset_dir), args.model_id, args.draws, args.top_k,
        args.horizon, args.max_positions, args.initial_capital, args.seed,
    )
    print()
    print(frame.to_string(index=False))
    print()
    print("plain        mean {:>9.2%}".format(frame["return"].mean()))
    print("equal weight mean {:>9.2%}".format(frame["equal_weight_return"].mean()))
    print("edge per buy mean {:>9.4%}".format(frame["edge_per_buy"].mean()))
    print(json.dumps({"pool_bh": None}, default=str)[:0])


if __name__ == "__main__":
    main()
