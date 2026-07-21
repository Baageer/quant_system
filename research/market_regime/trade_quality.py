"""回测持仓与离线行情阶段标签的核心质量指标。"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DAILY_POSITION_COLUMNS = [
    "date",
    "symbol",
    "shares",
    "close",
    "market_value",
    "portfolio_value",
    "weight",
]


def normalize_daily_positions(results, market_data):
    """将引擎每日 EOD 持仓字典展开为 date × symbol 明细。"""

    if results is None or results.empty or "positions" not in results.columns:
        return pd.DataFrame(columns=DAILY_POSITION_COLUMNS)

    rows = []
    for date, result_row in results.iterrows():
        positions = result_row.get("positions", {})
        if not isinstance(positions, dict):
            continue
        portfolio_value = _to_float(result_row.get("portfolio_value"))
        for symbol, shares in positions.items():
            shares = _to_float(shares)
            if shares is None or shares <= 0:
                continue
            close = _get_close(market_data.get(symbol), date)
            market_value = shares * close if close is not None else np.nan
            weight = market_value / portfolio_value if market_value == market_value and portfolio_value and portfolio_value > 0 else np.nan
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "symbol": str(symbol),
                    "shares": shares,
                    "close": close,
                    "market_value": market_value,
                    "portfolio_value": portfolio_value,
                    "weight": weight,
                }
            )
    return pd.DataFrame(rows, columns=DAILY_POSITION_COLUMNS).sort_values(["date", "symbol"]).reset_index(drop=True)


def build_regime_labels(market_data, labeler, progress_callback=None):
    """为回测实际加载的标的生成离线阶段、逐日状态和拐点表。"""

    segment_frames = []
    daily_frames = []
    pivot_frames = []
    skipped = []
    for symbol, data in market_data.items():
        _notify_progress(progress_callback, "生成 {} 的行情阶段标签".format(symbol))
        try:
            segments, daily, pivots = labeler.label(data, symbol)
        except ValueError as exc:
            skipped.append({"symbol": str(symbol), "reason": str(exc)})
            continue
        segment_frames.append(segments)
        daily_frames.append(daily)
        pivot_frames.append(pivots)

    return {
        "segments": _concat_tables(segment_frames),
        "daily_states": _concat_tables(daily_frames),
        "pivots": _concat_tables(pivot_frames),
        "skipped_symbols": skipped,
    }


def calculate_core_regime_metrics(results, daily_positions, daily_states, trades):
    """计算上涨覆盖率、下跌暴露和盘整换手的明细及汇总。"""

    overlap = _build_daily_overlap(daily_states, daily_positions)
    up_coverage = _build_segment_exposure(overlap, "up", exclude_open_segments=True)
    down_exposure = _build_segment_exposure(overlap, "down", exclude_open_segments=False)
    sideways_turnover = _build_sideways_turnover(results, daily_states, trades)
    summary = _build_summary(up_coverage, down_exposure, sideways_turnover)
    return {
        "daily_overlap": overlap,
        "up_coverage": up_coverage,
        "down_exposure": down_exposure,
        "sideways_turnover": sideways_turnover,
        "summary": summary,
    }


def build_regime_quality_report(results, market_data, labeler, trades, progress_callback=None):
    """构建可导出的阶段标签、持仓明细和三项核心质量指标。"""

    _notify_progress(progress_callback, "展开每日 EOD 持仓明细")
    daily_positions = normalize_daily_positions(results, market_data)
    labels = build_regime_labels(market_data, labeler, progress_callback=progress_callback)
    if not labels["daily_states"].empty and not results.empty:
        evaluation_dates = pd.DatetimeIndex(results.index)
        labels["daily_states"] = labels["daily_states"][labels["daily_states"]["date"].isin(evaluation_dates)].copy()
    _notify_progress(progress_callback, "对齐持仓、阶段并计算核心指标")
    metrics = calculate_core_regime_metrics(results, daily_positions, labels["daily_states"], trades)
    labels.update(metrics)
    labels["daily_positions"] = daily_positions
    labels["label_parameters"] = labeler.config.to_dict()
    return labels


def export_regime_quality_report(report, output_dir, run_id, progress_callback=None):
    """将标签快照和核心指标分别导出，并写入可追溯 manifest。"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    table_names = {
        "daily_positions": "daily_positions",
        "segments": "regime_segments",
        "daily_states": "regime_daily_states",
        "pivots": "regime_pivots",
        "daily_overlap": "regime_daily_overlap",
        "up_coverage": "regime_up_coverage",
        "down_exposure": "regime_down_exposure",
        "sideways_turnover": "regime_sideways_turnover",
        "summary": "regime_quality_summary",
    }
    paths = {}
    for key, prefix in table_names.items():
        _notify_progress(progress_callback, "写入 {}".format(prefix))
        file_path = output_path / "{}_{}.csv".format(prefix, run_id)
        report.get(key, pd.DataFrame()).to_csv(file_path, index=False, encoding="utf-8-sig")
        paths[key] = file_path

    segments = report.get("segments", pd.DataFrame())
    manifest = {
        "analysis_id": run_id,
        "created_at": datetime.now().isoformat(),
        "label_version": _first_value(segments, "label_version"),
        "label_parameters": report.get("label_parameters", {}),
        "segment_count": int(len(segments)),
        "daily_state_count": int(len(report.get("daily_states", pd.DataFrame()))),
        "skipped_symbols": report.get("skipped_symbols", []),
    }
    _notify_progress(progress_callback, "写入阶段质量 manifest")
    manifest_path = output_path / "regime_quality_manifest_{}.json".format(run_id)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["manifest"] = manifest_path
    return paths


def _build_daily_overlap(daily_states, daily_positions):
    columns = [
        "date", "symbol", "segment_id", "regime", "open_segment", "regime_confidence",
        "shares", "close", "market_value", "portfolio_value", "weight", "is_held",
    ]
    if daily_states is None or daily_states.empty:
        return pd.DataFrame(columns=columns)

    labels = daily_states.copy()
    labels["date"] = pd.to_datetime(labels["date"])
    positions = daily_positions.copy()
    if positions.empty:
        positions = pd.DataFrame(columns=DAILY_POSITION_COLUMNS)
    else:
        positions["date"] = pd.to_datetime(positions["date"])
    overlap = labels.merge(
        positions[["date", "symbol", "shares", "close", "market_value", "portfolio_value", "weight"]],
        on=["date", "symbol"],
        how="left",
    )
    overlap["shares"] = overlap["shares"].fillna(0.0)
    overlap["is_held"] = overlap["shares"] > 0
    overlap["weight"] = overlap["weight"].fillna(0.0)
    return overlap.reindex(columns=columns).sort_values(["symbol", "date"]).reset_index(drop=True)


def _build_segment_exposure(overlap, regime, exclude_open_segments):
    columns = [
        "symbol", "segment_id", "regime", "start_date", "end_date", "open_segment",
        "trading_days", "held_days", "exposure_rate", "average_weight", "max_weight",
    ]
    if overlap.empty:
        return pd.DataFrame(columns=columns)
    selected = overlap[overlap["regime"] == regime].copy()
    if exclude_open_segments:
        selected = selected[~selected["open_segment"].astype(bool)]
    rows = []
    for (symbol, segment_id), frame in selected.groupby(["symbol", "segment_id"]):
        held_days = int(frame["is_held"].sum())
        rows.append(
            {
                "symbol": symbol,
                "segment_id": segment_id,
                "regime": regime,
                "start_date": frame["date"].min(),
                "end_date": frame["date"].max(),
                "open_segment": bool(frame["open_segment"].iloc[0]),
                "trading_days": int(len(frame)),
                "held_days": held_days,
                "exposure_rate": held_days / float(len(frame)),
                "average_weight": float(frame["weight"].mean()),
                "max_weight": float(frame["weight"].max()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_sideways_turnover(results, daily_states, trades):
    columns = [
        "symbol", "segment_id", "start_date", "end_date", "trading_days", "open_segment",
        "trade_count", "buy_count", "sell_count", "turnover_value", "average_portfolio_value", "turnover_rate",
    ]
    if daily_states is None or daily_states.empty:
        return pd.DataFrame(columns=columns)
    filled_trades = _filled_trades(trades)
    result_values = results[["portfolio_value"]].copy() if results is not None and "portfolio_value" in results.columns else pd.DataFrame()
    if not result_values.empty:
        result_values.index = pd.to_datetime(result_values.index)
    rows = []
    selected = daily_states[daily_states["regime"] == "sideways"].copy()
    selected["date"] = pd.to_datetime(selected["date"])
    for (symbol, segment_id), frame in selected.groupby(["symbol", "segment_id"]):
        dates = pd.DatetimeIndex(frame["date"])
        segment_trades = filled_trades[(filled_trades["symbol"] == symbol) & (filled_trades["date"].isin(dates))]
        turnover_value = float(segment_trades["trade_value"].sum()) if "trade_value" in segment_trades.columns else 0.0
        average_value = float(result_values.loc[result_values.index.isin(dates), "portfolio_value"].mean()) if not result_values.empty else np.nan
        turnover_rate = turnover_value / average_value if average_value and average_value > 0 else np.nan
        rows.append(
            {
                "symbol": symbol,
                "segment_id": segment_id,
                "start_date": dates.min(),
                "end_date": dates.max(),
                "trading_days": int(len(frame)),
                "open_segment": bool(frame["open_segment"].iloc[0]),
                "trade_count": int(len(segment_trades)),
                "buy_count": int((segment_trades["action"] == "buy").sum()) if "action" in segment_trades.columns else 0,
                "sell_count": int((segment_trades["action"] == "sell").sum()) if "action" in segment_trades.columns else 0,
                "turnover_value": turnover_value,
                "average_portfolio_value": average_value,
                "turnover_rate": turnover_rate,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_summary(up_coverage, down_exposure, sideways_turnover):
    up_days = int(up_coverage["trading_days"].sum()) if not up_coverage.empty else 0
    up_held = int(up_coverage["held_days"].sum()) if not up_coverage.empty else 0
    down_days = int(down_exposure["trading_days"].sum()) if not down_exposure.empty else 0
    down_held = int(down_exposure["held_days"].sum()) if not down_exposure.empty else 0
    values = [
        ("up_coverage_rate", up_held / float(up_days) if up_days else np.nan),
        ("up_opportunity_days", up_days),
        ("up_held_days", up_held),
        ("down_exposure_rate", down_held / float(down_days) if down_days else np.nan),
        ("down_regime_days", down_days),
        ("down_held_days", down_held),
        ("sideways_trade_count", int(sideways_turnover["trade_count"].sum()) if not sideways_turnover.empty else 0),
        ("sideways_turnover_value", float(sideways_turnover["turnover_value"].sum()) if not sideways_turnover.empty else 0.0),
    ]
    return pd.DataFrame(values, columns=["metric", "value"])


def _filled_trades(trades):
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["date", "symbol", "action", "trade_value"])
    result = trades.copy()
    if "status" in result.columns:
        result = result[result["status"] == "filled"].copy()
    result["date"] = pd.to_datetime(result["date"])
    if "trade_value" not in result.columns:
        result["trade_value"] = 0.0
    return result


def _get_close(data, date):
    if data is None or data.empty or "close" not in data.columns:
        return None
    if date not in data.index:
        return None
    value = data.loc[date, "close"]
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    return _to_float(value)


def _to_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _concat_tables(frames):
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _first_value(frame, column):
    if frame is None or frame.empty or column not in frame.columns:
        return None
    return str(frame[column].iloc[0])


def _notify_progress(progress_callback, message):
    if progress_callback is not None:
        progress_callback(message)
