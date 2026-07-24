"""Evaluate probability alerts and their tradable out-of-sample value.

The alert rule is selected exclusively on the validation prediction table.  It
is then frozen and applied once to the test prediction table, where this module
reports event-level warning quality and runs a next-session execution backtest.
The backtest deliberately delegates T+1, limit, suspension, slippage and fee
handling to :class:`backtest.engine.BacktestEngine`.

Example::

    D:\\Anaconda3\\python.exe -m research.market_regime.probability_trade_evaluation \
        --dataset-dir output/datasets --model-id hs300_logistic_v2
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from backtest.engine import BacktestEngine
from research.market_regime.data_loader import load_local_price_history


TOP_K_GRID = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)
THRESHOLD_QUANTILES = (0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999)
RETURN_AUDIT_HORIZONS = (5, 10, 20)
TRADING_POLICY_GRID = (
    {"policy_id": "daily_alert_baseline", "entry_confirm_days": 1, "exit_threshold_multiplier": 1.0, "min_holding_days": 0, "rebalance_weeks": 0},
    {"policy_id": "weekly_confirm_1", "entry_confirm_days": 1, "exit_threshold_multiplier": 0.85, "min_holding_days": 5, "rebalance_weeks": 1},
    {"policy_id": "weekly_confirm_2", "entry_confirm_days": 2, "exit_threshold_multiplier": 0.85, "min_holding_days": 5, "rebalance_weeks": 1},
    {"policy_id": "weekly_confirm_2_hold_10", "entry_confirm_days": 2, "exit_threshold_multiplier": 0.85, "min_holding_days": 10, "rebalance_weeks": 1},
    {"policy_id": "biweekly_confirm_2_hold_10", "entry_confirm_days": 2, "exit_threshold_multiplier": 0.85, "min_holding_days": 10, "rebalance_weeks": 2},
    {"policy_id": "weekly_confirm_2_wide_exit", "entry_confirm_days": 2, "exit_threshold_multiplier": 0.75, "min_holding_days": 10, "rebalance_weeks": 1},
)
REQUIRED_PREDICTION_COLUMNS = {
    "date",
    "symbol",
    "target",
    "target_event_id",
    "target_event_date",
    "days_to_target",
    "p_down",
    "p_up",
}


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and normalize one Logistic prediction export."""

    # Preserve A-share leading zeroes (for example ``000001``) before pandas
    # has a chance to infer the field as an integer.
    predictions = pd.read_csv(path, dtype={"symbol": str})
    missing = REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns)
    if missing:
        raise ValueError("Prediction file is missing columns: {}".format(sorted(missing)))
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
    predictions["target_event_date"] = pd.to_datetime(
        predictions["target_event_date"], errors="coerce"
    )
    predictions["symbol"] = predictions["symbol"].astype(str).str.strip()
    for column in ("p_up", "p_down", "days_to_target"):
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce")
    predictions = predictions.dropna(subset=["date", "symbol", "p_up", "p_down"])
    return predictions.sort_values(["date", "symbol"]).reset_index(drop=True)


def select_alerts(
    predictions: pd.DataFrame,
    rule_type: str,
    rule_value: float,
) -> pd.Series:
    """Return causal EOD alerts under a cross-sectional Top-K or threshold rule."""

    if rule_type == "top_k":
        if not 0 < rule_value <= 1:
            raise ValueError("top_k rule_value must be in (0, 1]")
        ranks = predictions.groupby("date", sort=False)["p_up"].rank(
            method="first", ascending=False
        )
        daily_counts = predictions.groupby("date", sort=False)["p_up"].transform("size")
        selected = ranks <= np.ceil(daily_counts * rule_value)
    elif rule_type == "threshold":
        selected = predictions["p_up"] >= rule_value
    else:
        raise ValueError("Unsupported rule_type: {}".format(rule_type))

    # An upward warning should not contradict the model's down probability.
    return (selected & (predictions["p_up"] > predictions["p_down"])).astype(bool)


def calculate_event_metrics(
    predictions: pd.DataFrame,
    alerts: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Measure warning quality once per future ``up`` turning event.

    Repeated alerts are counted within an event's available warning rows rather
    than treating each day before the same event as an independent success.
    """

    if len(predictions) != len(alerts):
        raise ValueError("predictions and alerts must have identical lengths")
    work = predictions.loc[:, [
        "date", "symbol", "target", "target_event_id", "target_event_date", "days_to_target"
    ]].copy()
    work["alert"] = np.asarray(alerts, dtype=bool)
    event_rows = work[
        (work["target"] == "up") & work["target_event_id"].notnull()
    ].copy()
    event_columns = [
        "symbol", "target_event_id", "target_event_date", "warning_rows",
        "alert_count", "hit", "first_alert_date", "first_warning_lead_days",
        "repeat_alert_count",
    ]
    if event_rows.empty:
        empty = pd.DataFrame(columns=event_columns)
        return empty, _empty_event_summary(int(work["alert"].sum()))

    event_rows["target_event_id"] = event_rows["target_event_id"].astype(str)
    event_records = []
    for (symbol, event_id), group in event_rows.groupby(["symbol", "target_event_id"], sort=False):
        group = group.sort_values("date")
        event_alerts = group[group["alert"]]
        hit = not event_alerts.empty
        first = event_alerts.iloc[0] if hit else None
        lead_days = float(first["days_to_target"]) if hit and _is_finite(first["days_to_target"]) else np.nan
        event_records.append(
            {
                "symbol": symbol,
                "target_event_id": event_id,
                "target_event_date": group["target_event_date"].dropna().iloc[0]
                if group["target_event_date"].notnull().any() else pd.NaT,
                "warning_rows": int(len(group)),
                "alert_count": int(len(event_alerts)),
                "hit": hit,
                "first_alert_date": first["date"] if hit else pd.NaT,
                "first_warning_lead_days": lead_days,
                "repeat_alert_count": max(int(len(event_alerts)) - 1, 0),
            }
        )
    events = pd.DataFrame(event_records, columns=event_columns)
    hit_events = events[events["hit"]]
    alert_count = int(work["alert"].sum())
    false_alert_count = int(work.loc[(work["target"] != "up") & work["alert"]].shape[0])
    hit_count = int(len(hit_events))
    event_count = int(len(events))
    event_recall = hit_count / event_count if event_count else np.nan
    event_precision = hit_count / alert_count if alert_count else 0.0
    event_f1 = _f1(event_precision, event_recall)
    summary = {
        "event_count": event_count,
        "hit_event_count": hit_count,
        "event_hit_rate": event_recall,
        "alert_count": alert_count,
        "false_alert_count": false_alert_count,
        "event_precision_per_alert": event_precision,
        "event_f1": event_f1,
        "mean_first_warning_lead_days": float(hit_events["first_warning_lead_days"].mean())
        if hit_count else np.nan,
        "median_first_warning_lead_days": float(hit_events["first_warning_lead_days"].median())
        if hit_count else np.nan,
        "mean_repeat_alert_count": float(hit_events["repeat_alert_count"].mean()) if hit_count else np.nan,
        "total_repeat_alert_count": int(events["repeat_alert_count"].sum()),
        "alerts_per_hit_event": alert_count / hit_count if hit_count else np.nan,
    }
    return events, summary


def calculate_executable_event_audit(
    events: pd.DataFrame,
    market_data: Mapping[str, pd.DataFrame],
    horizons: Sequence[int] = RETURN_AUDIT_HORIZONS,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Audit post-label and first-warning returns without changing model inputs.

    ``event_*`` measures whether the up-turning-point label itself is followed
    by a positive, investable close-to-close return. ``alert_entry_*`` measures
    the outcome from the next local trading session after the first model alert.
    The latter is the relevant quantity for a T+1 execution policy.
    """

    records = []
    for event in events.itertuples(index=False):
        prices = market_data.get(event.symbol)
        if prices is None or prices.empty:
            continue
        record = {
            "symbol": event.symbol,
            "target_event_id": event.target_event_id,
            "target_event_date": event.target_event_date,
            "hit": bool(event.hit),
            "first_alert_date": event.first_alert_date,
        }
        record.update(_forward_return_metrics(prices, event.target_event_date, "event", horizons, next_session=False))
        if bool(event.hit) and _timestamp_is_valid(event.first_alert_date):
            record.update(
                _forward_return_metrics(
                    prices, event.first_alert_date, "alert_entry", horizons, next_session=True
                )
            )
        records.append(record)
    audit = pd.DataFrame(records)
    return audit, summarize_executable_event_audit(audit, horizons)


def summarize_executable_event_audit(
    audit: pd.DataFrame,
    horizons: Sequence[int] = RETURN_AUDIT_HORIZONS,
) -> Dict[str, object]:
    """Summarize label viability and state a target-design diagnostic."""

    summary: Dict[str, object] = {"event_records_with_prices": int(len(audit))}
    for prefix in ("event", "alert_entry"):
        for horizon in horizons:
            column = "{}_return_{}d".format(prefix, horizon)
            values = pd.to_numeric(audit[column], errors="coerce").dropna() if column in audit else pd.Series(dtype=float)
            summary.update(
                {
                    "{}_{}_count".format(prefix, horizon): int(len(values)),
                    "{}_{}_mean".format(prefix, horizon): float(values.mean()) if len(values) else np.nan,
                    "{}_{}_median".format(prefix, horizon): float(values.median()) if len(values) else np.nan,
                    "{}_{}_positive_rate".format(prefix, horizon): float((values > 0).mean()) if len(values) else np.nan,
                }
            )
    label_median = summary.get("event_10_median", np.nan)
    label_positive_rate = summary.get("event_10_positive_rate", np.nan)
    entry_median = summary.get("alert_entry_10_median", np.nan)
    entry_positive_rate = summary.get("alert_entry_10_positive_rate", np.nan)
    label_viable = _is_finite(label_median) and _is_finite(label_positive_rate) and label_median > 0 and label_positive_rate >= 0.55
    entry_viable = _is_finite(entry_median) and _is_finite(entry_positive_rate) and entry_median > 0 and entry_positive_rate >= 0.55
    if not label_viable:
        recommendation = (
            "The event label is not consistently followed by a positive 10-session return. "
            "Prioritize a future risk-adjusted or benchmark-relative return target."
        )
    elif not entry_viable:
        recommendation = (
            "The event label has post-event return space, but first-alert T+1 entry does not. "
            "Keep the event study for interpretation and model entry timing, downside risk, or excess return."
        )
    else:
        recommendation = (
            "Both event and first-alert T+1 returns show positive 10-session viability. "
            "Retain the event target while validating portfolio construction and costs."
        )
    summary.update({"label_10d_viable": bool(label_viable), "alert_entry_10d_viable": bool(entry_viable), "target_recommendation": recommendation})
    return summary


def _forward_return_metrics(
    prices: pd.DataFrame,
    anchor_date,
    prefix: str,
    horizons: Sequence[int],
    next_session: bool,
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    if not _timestamp_is_valid(anchor_date):
        return result
    anchor = pd.Timestamp(anchor_date)
    start = int(prices.index.searchsorted(anchor, side="right" if next_session else "left"))
    if start >= len(prices):
        return result
    entry_price = float(prices["close"].iloc[start])
    if not _is_finite(entry_price) or entry_price <= 0:
        return result
    result["{}_anchor_date".format(prefix)] = prices.index[start]
    result["{}_anchor_close".format(prefix)] = entry_price
    for horizon in horizons:
        end = start + int(horizon)
        if end >= len(prices):
            continue
        path = pd.to_numeric(prices["close"].iloc[start : end + 1], errors="coerce").dropna()
        if path.empty:
            continue
        result["{}_return_{}d".format(prefix, horizon)] = float(path.iloc[-1] / entry_price - 1.0)
        result["{}_mfe_{}d".format(prefix, horizon)] = float(path.max() / entry_price - 1.0)
        result["{}_mae_{}d".format(prefix, horizon)] = float(path.min() / entry_price - 1.0)
    return result


def build_validation_rule_grid(
    validation_predictions: pd.DataFrame,
    min_hit_events: int = 25,
) -> pd.DataFrame:
    """Evaluate a small predeclared Top-K/threshold grid on validation only."""

    candidates = [("top_k", value, None) for value in TOP_K_GRID]
    for quantile in THRESHOLD_QUANTILES:
        candidates.append(("threshold", float(validation_predictions["p_up"].quantile(quantile)), quantile))

    records = []
    for rule_type, rule_value, quantile in candidates:
        alerts = select_alerts(validation_predictions, rule_type, rule_value)
        _, metrics = calculate_event_metrics(validation_predictions, alerts)
        records.append(
            {
                "rule_type": rule_type,
                "rule_value": rule_value,
                "source_quantile": quantile,
                "eligible": metrics["hit_event_count"] >= min_hit_events,
                **metrics,
            }
        )
    return pd.DataFrame(records).sort_values(
        ["eligible", "event_f1", "event_hit_rate", "event_precision_per_alert"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def choose_validation_rule(grid: pd.DataFrame) -> Dict[str, object]:
    """Freeze the first ranked eligible rule, with a safe fallback."""

    if grid.empty:
        raise ValueError("Validation rule grid is empty")
    eligible = grid[grid["eligible"]]
    selected = (eligible if not eligible.empty else grid).iloc[0]
    return {
        "rule_type": str(selected["rule_type"]),
        "rule_value": float(selected["rule_value"]),
        "selection_objective": "validation_event_f1",
        "validation_event_f1": float(selected["event_f1"]),
        "validation_event_hit_rate": float(selected["event_hit_rate"]),
        "validation_event_precision_per_alert": float(selected["event_precision_per_alert"]),
        "validation_hit_event_count": int(selected["hit_event_count"]),
    }


def build_market_data(
    predictions: pd.DataFrame,
    alerts: pd.Series,
    price_history_dir: Optional[str] = None,
) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, str]]]:
    """Load local OHLCV and shift EOD decisions to the next trading session."""

    decisions = predictions.loc[:, ["date", "symbol", "p_up", "p_down"]].copy()
    decisions["eod_alert"] = np.asarray(alerts, dtype=bool)
    start_date, end_date = decisions["date"].min(), decisions["date"].max()
    market_data: Dict[str, pd.DataFrame] = {}
    skipped: List[Dict[str, str]] = []
    for symbol in tqdm(sorted(decisions["symbol"].unique()), desc="Loading test prices", unit="symbol"):
        try:
            prices = load_local_price_history(symbol, price_history_dir=price_history_dir)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append({"symbol": symbol, "reason": str(exc)})
            continue
        prices = prices[(prices.index >= start_date) & (prices.index <= end_date)].copy()
        if prices.empty:
            skipped.append({"symbol": symbol, "reason": "no_prices_in_test_period"})
            continue
        symbol_decisions = decisions[decisions["symbol"] == symbol].set_index("date")
        prices = prices.join(symbol_decisions[["p_up", "p_down", "eod_alert"]], how="left")
        prices["p_up"] = prices["p_up"].fillna(0.0)
        prices["p_down"] = prices["p_down"].fillna(1.0)
        prices["eod_alert"] = prices["eod_alert"].fillna(False).astype(bool)
        market_data[symbol] = prices
    return market_data, skipped


def apply_trading_policy_signals(
    market_data: Mapping[str, pd.DataFrame],
    rule: Mapping[str, object],
    policy: Mapping[str, object],
) -> Dict[str, pd.DataFrame]:
    """Add causal entry and hold signals for one low-turnover trading policy."""

    entry_confirm_days = int(policy["entry_confirm_days"])
    exit_multiplier = float(policy["exit_threshold_multiplier"])
    if entry_confirm_days < 1 or not 0 < exit_multiplier <= 1:
        raise ValueError("Invalid trading policy")
    prepared = {}
    for symbol, prices in market_data.items():
        frame = prices.copy()
        confirmed_entry = (
            frame["eod_alert"].astype(int).rolling(entry_confirm_days, min_periods=entry_confirm_days).sum()
            >= entry_confirm_days
        )
        if rule["rule_type"] == "threshold":
            hold_eod = (
                (frame["p_up"] >= float(rule["rule_value"]) * exit_multiplier)
                & (frame["p_up"] > frame["p_down"])
            )
        else:
            hold_eod = frame["eod_alert"]
        # All information is known at the close of t; the engine sees it at t+1.
        frame["entry_trade"] = confirmed_entry.shift(1).fillna(False).astype(bool)
        frame["hold_trade"] = hold_eod.shift(1).fillna(False).astype(bool)
        prepared[symbol] = frame
    return prepared


def make_probability_strategy(
    max_positions: int,
    target_notional: float,
    policy: Mapping[str, object],
    lot_size: int = 100,
):
    """Create a confirmation, hysteresis and scheduled-rebalance callback."""

    min_holding_days = int(policy["min_holding_days"])
    rebalance_weeks = int(policy["rebalance_weeks"])
    entry_dates: Dict[str, pd.Timestamp] = {}
    latest_week = None
    rebalance_week_number = -1

    def is_rebalance_date(date):
        nonlocal latest_week, rebalance_week_number
        if rebalance_weeks == 0:
            return True
        iso = pd.Timestamp(date).isocalendar()
        current_week = (int(iso[0]), int(iso[1]))
        if current_week != latest_week:
            latest_week = current_week
            rebalance_week_number += 1
        return rebalance_week_number % rebalance_weeks == 0

    def holding_days(symbol, frame, date):
        entry_date = entry_dates.get(symbol)
        if entry_date is None or entry_date not in frame.index:
            return 0
        return max(int(frame.index.get_loc(date)) - int(frame.index.get_loc(entry_date)), 0)

    def strategy(date, relevant_data, positions):
        signals = {}
        held_symbols = set(positions)
        for symbol in held_symbols:
            entry_dates.setdefault(symbol, pd.Timestamp(date))
        for symbol in list(entry_dates):
            if symbol not in held_symbols:
                del entry_dates[symbol]
        if not is_rebalance_date(date):
            return signals
        for symbol in sorted(held_symbols):
            frame = relevant_data.get(symbol)
            if (
                frame is not None
                and not bool(frame.loc[date, "hold_trade"])
                and holding_days(symbol, frame, date) >= min_holding_days
            ):
                signals[symbol] = {"action": "sell", "shares": positions[symbol], "reason": "probability_exit"}

        available_slots = max(max_positions - len(held_symbols), 0)
        candidates = []
        for symbol, frame in relevant_data.items():
            if symbol not in held_symbols and bool(frame.loc[date, "entry_trade"]):
                candidates.append((symbol, float(frame.loc[date, "p_up"]), float(frame.loc[date, "close"])))
        for symbol, _, price in sorted(candidates, key=lambda item: (-item[1], item[0]))[:available_slots]:
            shares = int(np.floor(target_notional / price / lot_size) * lot_size) if price > 0 else 0
            if shares > 0:
                signals[symbol] = {"action": "buy", "shares": shares, "reason": "probability_entry_t_plus_1"}
        return signals

    return strategy


def make_passive_strategy(symbols: Sequence[str], target_notional: float, lot_size: int = 100):
    """Create a static equal-notional passive basket for the same engine."""

    selected = tuple(symbols)
    submitted = False

    def strategy(date, relevant_data, positions):
        nonlocal submitted
        if submitted:
            return {}
        signals = {}
        for symbol in selected:
            frame = relevant_data.get(symbol)
            if frame is None:
                continue
            price = float(frame.loc[date, "close"])
            shares = int(np.floor(target_notional / price / lot_size) * lot_size) if price > 0 else 0
            if shares > 0:
                signals[symbol] = {"action": "buy", "shares": shares, "reason": "passive_equal_weight_entry"}
        submitted = True
        return signals

    return strategy


def _run_engine(
    market_data: Mapping[str, pd.DataFrame],
    strategy,
    initial_capital: float,
    config_path: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    frictionless: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    engine = BacktestEngine(initial_capital=initial_capital, config_path=config_path)
    if frictionless:
        engine.commission_rate = 0.0
        engine.slippage = 0.0
        engine.min_commission = 0.0
        engine.stamp_duty_rate = 0.0
    results = engine.run(
        dict(market_data), strategy, str(start_date.date()), str(end_date.date()), show_progress=False
    )
    return results, engine.get_trades()


def summarize_backtest(
    name: str,
    results: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float,
) -> Dict[str, object]:
    """Return net-of-cost performance and execution diagnostics."""

    if results.empty:
        raise ValueError("Backtest results are empty")
    values = results["portfolio_value"].astype(float)
    daily_returns = values.pct_change().dropna()
    total_return = float(values.iloc[-1] / initial_capital - 1.0)
    annual_return = float((1.0 + total_return) ** (252 / len(values)) - 1.0) if len(values) else np.nan
    annual_volatility = float(daily_returns.std(ddof=1) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else np.nan
    drawdown = values / values.cummax() - 1.0
    filled = trades[trades.get("status", pd.Series(dtype=str)) == "filled"] if not trades.empty else trades
    rejected = trades[trades.get("status", pd.Series(dtype=str)) == "rejected"] if not trades.empty else trades
    trade_value = float(filled["trade_value"].sum()) if not filled.empty else 0.0
    commission = float(filled["commission"].sum()) if not filled.empty else 0.0
    stamp_duty = float(filled["stamp_duty"].sum()) if not filled.empty else 0.0
    filled_buy_count = int((filled["action"] == "buy").sum()) if not filled.empty else 0
    return {
        "strategy": name,
        "start_date": values.index.min().date().isoformat(),
        "end_date": values.index.max().date().isoformat(),
        "final_value": float(values.iloc[-1]),
        "net_profit": float(values.iloc[-1] - initial_capital),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_zero_rate": sharpe,
        "max_drawdown": float(drawdown.min()),
        "filled_trade_count": int(len(filled)),
        "filled_buy_count": filled_buy_count,
        "rejected_buy_count": int(len(rejected)),
        "turnover_multiple": trade_value / initial_capital,
        "commission": commission,
        "stamp_duty": stamp_duty,
        "reported_cost": commission + stamp_duty,
    }


def build_validation_policy_grid(
    market_data: Mapping[str, pd.DataFrame],
    rule: Mapping[str, object],
    initial_capital: float,
    max_positions: int,
    config_path: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    min_filled_buys: int = 20,
) -> pd.DataFrame:
    """Select low-turnover policy parameters from validation net returns only."""

    target_notional = initial_capital / max_positions
    records = []
    for policy in TRADING_POLICY_GRID:
        prepared = apply_trading_policy_signals(market_data, rule, policy)
        results, trades = _run_engine(
            prepared,
            make_probability_strategy(max_positions, target_notional, policy),
            initial_capital, config_path, start_date, end_date,
        )
        metrics = summarize_backtest(policy["policy_id"], results, trades, initial_capital)
        records.append({
            **policy,
            "eligible": metrics["filled_buy_count"] >= min_filled_buys,
            **{"validation_" + key: value for key, value in metrics.items() if key != "strategy"},
        })
    return pd.DataFrame(records).sort_values(
        ["eligible", "validation_total_return", "validation_turnover_multiple", "validation_max_drawdown"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)


def choose_validation_policy(grid: pd.DataFrame) -> Dict[str, object]:
    """Freeze the best eligible validation policy before touching test data."""

    if grid.empty:
        raise ValueError("Validation policy grid is empty")
    candidates = grid[grid["eligible"]]
    selected = (candidates if not candidates.empty else grid).sort_values(
        ["validation_total_return", "validation_turnover_multiple"],
        ascending=[False, True],
    ).iloc[0]
    return {
        "policy_id": str(selected["policy_id"]),
        "entry_confirm_days": int(selected["entry_confirm_days"]),
        "exit_threshold_multiplier": float(selected["exit_threshold_multiplier"]),
        "min_holding_days": int(selected["min_holding_days"]),
        "rebalance_weeks": int(selected["rebalance_weeks"]),
        "selection_objective": "validation_net_total_return_then_lower_turnover",
        "validation_total_return": float(selected["validation_total_return"]),
        "validation_turnover_multiple": float(selected["validation_turnover_multiple"]),
        "validation_filled_buy_count": int(selected["validation_filled_buy_count"]),
    }


def select_passive_symbols(symbols: Iterable[str], count: int) -> List[str]:
    """Choose a deterministic, universe-only passive sample without model ranks."""

    ordered = sorted(set(symbols))
    if not ordered:
        return []
    if len(ordered) <= count:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, count, dtype=int)
    return [ordered[index] for index in indices]


def run_evaluation(
    dataset_dir: Path,
    model_id: str,
    output_dir: Path,
    min_hit_events: int = 25,
    max_positions: int = 20,
    initial_capital: float = 500000.0,
    price_history_dir: Optional[str] = None,
    config_path: str = "config/settings.yaml",
) -> Dict[str, object]:
    """Select on validation, then evaluate alerts and trading only on test."""

    validation_path = dataset_dir / "trend_logistic_validation_predictions_{}.csv".format(model_id)
    test_path = dataset_dir / "trend_logistic_test_predictions_{}.csv".format(model_id)
    validation = load_predictions(validation_path)
    test = load_predictions(test_path)
    grid = build_validation_rule_grid(validation, min_hit_events=min_hit_events)
    rule = choose_validation_rule(grid)
    validation_alerts = select_alerts(validation, rule["rule_type"], rule["rule_value"])
    validation_market_data, validation_skipped_prices = build_market_data(
        validation, validation_alerts, price_history_dir=price_history_dir
    )
    if not validation_market_data:
        raise ValueError("No validation symbols had usable local price data")
    validation_start, validation_end = validation["date"].min(), validation["date"].max()
    policy_grid = build_validation_policy_grid(
        validation_market_data, rule, initial_capital, max_positions, config_path,
        validation_start, validation_end,
    )
    policy = choose_validation_policy(policy_grid)
    test_alerts = select_alerts(test, rule["rule_type"], rule["rule_value"])
    event_details, event_summary = calculate_event_metrics(test, test_alerts)

    raw_market_data, skipped_prices = build_market_data(test, test_alerts, price_history_dir=price_history_dir)
    if not raw_market_data:
        raise ValueError("No test symbols had usable local price data")
    executable_event_audit, executable_event_summary = calculate_executable_event_audit(
        event_details, raw_market_data
    )
    market_data = apply_trading_policy_signals(raw_market_data, rule, policy)
    start_date, end_date = test["date"].min(), test["date"].max()
    target_notional = initial_capital / max_positions
    probability_results, probability_trades = _run_engine(
        market_data,
        make_probability_strategy(max_positions, target_notional, policy),
        initial_capital, config_path, start_date, end_date,
    )
    frictionless_results, frictionless_trades = _run_engine(
        market_data,
        make_probability_strategy(max_positions, target_notional, policy),
        initial_capital, config_path, start_date, end_date, frictionless=True,
    )
    passive_symbols = select_passive_symbols(market_data, max_positions)
    passive_results, passive_trades = _run_engine(
        market_data,
        make_passive_strategy(passive_symbols, target_notional),
        initial_capital, config_path, start_date, end_date,
    )

    backtest_metrics = pd.DataFrame([
        summarize_backtest("probability_net", probability_results, probability_trades, initial_capital),
        summarize_backtest("probability_frictionless", frictionless_results, frictionless_trades, initial_capital),
        summarize_backtest("passive_equal_weight_sample_net", passive_results, passive_trades, initial_capital),
    ])
    net = backtest_metrics.set_index("strategy").loc["probability_net"]
    passive = backtest_metrics.set_index("strategy").loc["passive_equal_weight_sample_net"]
    frictionless = backtest_metrics.set_index("strategy").loc["probability_frictionless"]
    comparison = {
        "net_profit_vs_cash": float(net["net_profit"]),
        "net_total_return_vs_cash": float(net["total_return"]),
        "net_total_return_excess_vs_passive_sample": float(net["total_return"] - passive["total_return"]),
        "net_profit_excess_vs_passive_sample": float(net["net_profit"] - passive["net_profit"]),
        "estimated_friction_drag_return": float(frictionless["total_return"] - net["total_return"]),
        "net_gain_positive_vs_cash": bool(net["net_profit"] > 0),
        "net_excess_positive_vs_passive_sample": bool(net["total_return"] > passive["total_return"]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "trend_probability_trade_evaluation_{}".format(model_id)
    grid.to_csv(output_dir / (prefix + "_validation_rule_grid.csv"), index=False)
    policy_grid.to_csv(output_dir / (prefix + "_validation_policy_grid.csv"), index=False)
    event_details.to_csv(output_dir / (prefix + "_test_events.csv"), index=False)
    pd.DataFrame([event_summary]).to_csv(output_dir / (prefix + "_test_event_metrics.csv"), index=False)
    executable_event_audit.to_csv(output_dir / (prefix + "_test_executable_event_returns.csv"), index=False)
    pd.DataFrame([executable_event_summary]).to_csv(
        output_dir / (prefix + "_test_executable_event_summary.csv"), index=False
    )
    backtest_metrics.to_csv(output_dir / (prefix + "_backtest_metrics.csv"), index=False)
    probability_results.assign(strategy="probability_net").to_csv(output_dir / (prefix + "_equity_probability_net.csv"))
    frictionless_results.assign(strategy="probability_frictionless").to_csv(output_dir / (prefix + "_equity_probability_frictionless.csv"))
    passive_results.assign(strategy="passive_equal_weight_sample_net").to_csv(output_dir / (prefix + "_equity_passive.csv"))
    probability_trades.to_csv(output_dir / (prefix + "_trades_probability_net.csv"), index=False)
    frictionless_trades.to_csv(output_dir / (prefix + "_trades_probability_frictionless.csv"), index=False)
    passive_trades.to_csv(output_dir / (prefix + "_trades_passive.csv"), index=False)
    manifest = {
        "model_id": model_id,
        "validation_prediction_file": str(validation_path),
        "test_prediction_file": str(test_path),
        "rule": rule,
        "trading_policy": policy,
        "event_summary_test_only": event_summary,
        "executable_event_summary_test_only": executable_event_summary,
        "backtest_comparison_test_only": comparison,
        "execution": {
            "signal_timing": "EOD prediction at t; order submitted at next available session t+1",
            "engine": "BacktestEngine",
            "config_path": config_path,
            "initial_capital": initial_capital,
            "max_positions": max_positions,
            "target_notional": target_notional,
            "passive_symbols": passive_symbols,
            "validation_skipped_price_symbols": validation_skipped_prices,
            "prediction_symbol_count": int(test["symbol"].nunique()),
            "market_data_symbol_count": len(market_data),
            "market_data_coverage": len(market_data) / test["symbol"].nunique(),
            "skipped_price_symbols": skipped_prices,
        },
    }
    with (output_dir / (prefix + "_manifest.json")).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, default=_json_default)
    return manifest


def _empty_event_summary(alert_count: int) -> Dict[str, float]:
    return {
        "event_count": 0, "hit_event_count": 0, "event_hit_rate": np.nan,
        "alert_count": alert_count, "false_alert_count": alert_count,
        "event_precision_per_alert": 0.0, "event_f1": 0.0,
        "mean_first_warning_lead_days": np.nan, "median_first_warning_lead_days": np.nan,
        "mean_repeat_alert_count": np.nan, "total_repeat_alert_count": 0,
        "alerts_per_hit_event": np.nan,
    }


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _is_finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _timestamp_is_valid(value) -> bool:
    if value is None:
        return False
    try:
        return pd.Series([value]).notnull().iloc[0]
    except (TypeError, ValueError):
        return False


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError("Not JSON serializable: {}".format(type(value).__name__))


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="output/datasets")
    parser.add_argument("--model-id", default="hs300_logistic_v2")
    parser.add_argument("--output-dir", default="output/datasets")
    parser.add_argument("--min-hit-events", type=int, default=25)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=500000.0)
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--config-path", default="config/settings.yaml")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    manifest = run_evaluation(
        dataset_dir=Path(args.dataset_dir), model_id=args.model_id,
        output_dir=Path(args.output_dir), min_hit_events=args.min_hit_events,
        max_positions=args.max_positions, initial_capital=args.initial_capital,
        price_history_dir=args.price_history_dir, config_path=args.config_path,
    )
    rule = manifest["rule"]
    policy = manifest["trading_policy"]
    comparison = manifest["backtest_comparison_test_only"]
    print(
        "Selected validation rule: {rule_type}={rule_value:.8f}; policy={policy_id}; "
        "test event hit rate={hit:.2%}; net return={ret:.2%}; passive excess={excess:.2%}".format(
            rule_type=rule["rule_type"], rule_value=rule["rule_value"],
            policy_id=policy["policy_id"],
            hit=manifest["event_summary_test_only"]["event_hit_rate"],
            ret=comparison["net_total_return_vs_cash"],
            excess=comparison["net_total_return_excess_vs_passive_sample"],
        )
    )


if __name__ == "__main__":
    main()
