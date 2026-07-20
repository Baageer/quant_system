"""用于研究复盘的离线中期行情阶段标注器。

标签基于完整历史路径生成，包含未来信息，不能直接作为实盘信号。
"""

import numpy as np
import pandas as pd


class RegimeLabelerConfig:
    """第一版离线结构标签的固定参数。"""

    def __init__(
        self,
        min_segment_length=20,
        min_return=0.03,
        volatility_multiplier=0.75,
        min_efficiency_ratio=0.15,
        min_r_squared=0.10,
        boundary_window=5,
        smoothing_span=7,
        max_segment_length=120,
        volatility_horizon=60,
        dynamic_threshold_cap=0.20,
        segmentation_penalty=3.0,
        pivot_min_reversal=0.05,
        pivot_reversal_multiplier=2.0,
        pivot_volatility_window=20,
        label_version="market_regime_v1_1",
    ):
        self.min_segment_length = min_segment_length
        self.min_return = min_return
        self.volatility_multiplier = volatility_multiplier
        self.min_efficiency_ratio = min_efficiency_ratio
        self.min_r_squared = min_r_squared
        self.boundary_window = boundary_window
        self.smoothing_span = smoothing_span
        self.max_segment_length = max_segment_length
        self.volatility_horizon = volatility_horizon
        self.dynamic_threshold_cap = dynamic_threshold_cap
        self.segmentation_penalty = segmentation_penalty
        self.pivot_min_reversal = pivot_min_reversal
        self.pivot_reversal_multiplier = pivot_reversal_multiplier
        self.pivot_volatility_window = pivot_volatility_window
        self.label_version = label_version
        if self.min_segment_length < 2:
            raise ValueError("min_segment_length 必须至少为 2。")
        if self.min_return < 0:
            raise ValueError("min_return 不能为负数。")
        if self.volatility_multiplier < 0:
            raise ValueError("volatility_multiplier 不能为负数。")
        if not 0 <= self.min_efficiency_ratio <= 1:
            raise ValueError("min_efficiency_ratio 必须在 [0, 1] 内。")
        if not 0 <= self.min_r_squared <= 1:
            raise ValueError("min_r_squared 必须在 [0, 1] 内。")
        if self.boundary_window < 0:
            raise ValueError("boundary_window 不能为负数。")
        if self.smoothing_span < 1:
            raise ValueError("smoothing_span 必须至少为 1。")
        if self.max_segment_length < self.min_segment_length:
            raise ValueError("max_segment_length 不能小于 min_segment_length。")
        if self.volatility_horizon < 1:
            raise ValueError("volatility_horizon 必须至少为 1。")
        if not 0 < self.dynamic_threshold_cap <= 1:
            raise ValueError("dynamic_threshold_cap 必须在 (0, 1] 内。")
        if self.segmentation_penalty < 0:
            raise ValueError("segmentation_penalty 不能为负数。")
        if not 0 < self.pivot_min_reversal <= 1:
            raise ValueError("pivot_min_reversal 必须在 (0, 1] 内。")
        if self.pivot_reversal_multiplier < 0:
            raise ValueError("pivot_reversal_multiplier 不能为负数。")
        if self.pivot_volatility_window < 2:
            raise ValueError("pivot_volatility_window 必须至少为 2。")

    def to_dict(self):
        return {
            "min_segment_length": self.min_segment_length,
            "min_return": self.min_return,
            "volatility_multiplier": self.volatility_multiplier,
            "min_efficiency_ratio": self.min_efficiency_ratio,
            "min_r_squared": self.min_r_squared,
            "boundary_window": self.boundary_window,
            "smoothing_span": self.smoothing_span,
            "max_segment_length": self.max_segment_length,
            "volatility_horizon": self.volatility_horizon,
            "dynamic_threshold_cap": self.dynamic_threshold_cap,
            "segmentation_penalty": self.segmentation_penalty,
            "pivot_min_reversal": self.pivot_min_reversal,
            "pivot_reversal_multiplier": self.pivot_reversal_multiplier,
            "pivot_volatility_window": self.pivot_volatility_window,
            "label_version": self.label_version,
        }


def audit_price_data(data, close_column="close"):
    """返回输入行情的只读数据质量摘要，不修改原始数据。"""

    if close_column not in data.columns:
        raise ValueError(f"行情数据缺少 {close_column!r} 列。")

    dates = pd.to_datetime(data.index, errors="coerce")
    close = pd.to_numeric(data[close_column], errors="coerce")
    valid_dates = dates[~dates.isnull()]
    valid_close = close.dropna()

    return {
        "row_count": int(len(data)),
        "start_date": valid_dates.min() if len(valid_dates) else pd.NaT,
        "end_date": valid_dates.max() if len(valid_dates) else pd.NaT,
        "missing_date_count": int(dates.isnull().sum()),
        "duplicate_date_count": int(valid_dates.duplicated().sum()),
        "missing_close_count": int(close.isnull().sum()),
        "non_positive_close_count": int((valid_close <= 0).sum()),
        "is_date_sorted": bool(valid_dates.is_monotonic_increasing),
    }


class MarketRegimeLabeler:
    """以分段线性回归、三态分类和边界校准生成离线结构标签。"""

    def __init__(self, config=None):
        self.config = config or RegimeLabelerConfig()

    def label(
        self,
        data: pd.DataFrame,
        symbol: str,
        close_column: str = "close",
    ):
        """返回阶段表、逐日表和拐点事件表。

        输入必须使用日期索引和连续的正收盘价。最后一个阶段会标记为
        ``open_segment=True``，因为它的终点尚不能被后续数据确认。
        """

        close = self._prepare_close(data, close_column)
        if len(close) < self.config.min_segment_length:
            raise ValueError(
                "行情长度不足：至少需要 "
                f"{self.config.min_segment_length} 个交易日，实际为 {len(close)}。"
            )

        raw_log_close = np.log(np.asarray(close, dtype=float))
        analysis_log_close = self._smooth_log_price(raw_log_close)
        boundaries = self._find_candidate_boundaries(analysis_log_close)
        boundaries = self._calibrate_boundaries(boundaries, np.asarray(close, dtype=float))
        segments = self._build_segments(boundaries, close, analysis_log_close, raw_log_close)
        segments = self._merge_segments(segments, close, analysis_log_close, raw_log_close)

        segment_table = self._to_segment_table(segments, close, symbol)
        daily_table = self._to_daily_table(segment_table, close.index, symbol)
        pivot_table = self._to_pivot_table(segment_table, close, symbol)
        return segment_table, daily_table, pivot_table

    def audit(self, data, close_column="close"):
        """审计输入，并额外记录本次标签配置。"""

        report = audit_price_data(data, close_column)
        report["label_version"] = self.config.label_version
        report["parameters"] = self.config.to_dict()
        return report

    def _prepare_close(self, data, close_column):
        if close_column not in data.columns:
            raise ValueError(f"行情数据缺少 {close_column!r} 列。")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("行情数据必须使用 DatetimeIndex 作为日期索引。")

        close = pd.to_numeric(data[close_column], errors="coerce")
        if close.isnull().any():
            raise ValueError("close 包含缺失或非数值数据，请在标注前完成数据审计。")
        if (close <= 0).any():
            raise ValueError("close 必须全部为正数，才能进行对数价格分析。")
        if data.index.has_duplicates:
            raise ValueError("日期索引不能重复。")
        if not data.index.is_monotonic_increasing:
            raise ValueError("日期索引必须按升序排列。")
        return close.astype(float)

    def _smooth_log_price(self, raw_log_close):
        """使用单边 EWM 平滑价格，避免居中窗口引入未来信息。"""

        if self.config.smoothing_span == 1:
            return raw_log_close.copy()
        return pd.Series(raw_log_close).ewm(span=self.config.smoothing_span, adjust=False).mean().values

    def _find_candidate_boundaries(self, log_close):
        """用带惩罚的动态规划选择全局分段，边界采用左闭右开下标。"""

        length = len(log_close)
        minimum = self.config.min_segment_length
        maximum = min(self.config.max_segment_length, length)
        if length <= maximum:
            return [0, length]

        prefix = self._build_regression_prefix(log_close)
        base_variance = float(np.var(np.diff(log_close))) if length > 1 else 0.0
        penalty = self.config.segmentation_penalty * base_variance * minimum
        costs = np.full(length + 1, np.inf)
        previous = np.full(length + 1, -1, dtype=int)
        costs[0] = -penalty

        for end in range(minimum, length + 1):
            start_lower = max(0, end - maximum)
            start_upper = end - minimum
            for start in range(start_lower, start_upper + 1):
                if not np.isfinite(costs[start]):
                    continue
                candidate = costs[start] + self._linear_sse_from_prefix(prefix, start, end) + penalty
                if candidate < costs[end]:
                    costs[end] = candidate
                    previous[end] = start

        if previous[length] < 0:
            return [0, length]
        boundaries = [length]
        current = length
        while current > 0:
            current = int(previous[current])
            boundaries.append(current)
        return sorted(boundaries)

    @staticmethod
    def _build_regression_prefix(values):
        x_values = np.arange(len(values), dtype=float)
        return {
            "x": np.concatenate(([0.0], np.cumsum(x_values))),
            "x2": np.concatenate(([0.0], np.cumsum(x_values * x_values))),
            "y": np.concatenate(([0.0], np.cumsum(values))),
            "y2": np.concatenate(([0.0], np.cumsum(values * values))),
            "xy": np.concatenate(([0.0], np.cumsum(x_values * values))),
        }

    @staticmethod
    def _linear_sse_from_prefix(prefix, start, end):
        count = float(end - start)
        sum_x = prefix["x"][end] - prefix["x"][start]
        sum_x2 = prefix["x2"][end] - prefix["x2"][start]
        sum_y = prefix["y"][end] - prefix["y"][start]
        sum_y2 = prefix["y2"][end] - prefix["y2"][start]
        sum_xy = prefix["xy"][end] - prefix["xy"][start]
        denominator = count * sum_x2 - sum_x * sum_x
        if denominator <= np.finfo(float).eps:
            return max(0.0, sum_y2 - sum_y * sum_y / count)
        slope = (count * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / count
        sse = sum_y2 - 2 * slope * sum_xy - 2 * intercept * sum_y
        sse += slope * slope * sum_x2 + 2 * slope * intercept * sum_x + count * intercept * intercept
        return max(0.0, float(sse))

    @staticmethod
    def _linear_sse(values, start, end):
        y_values = values[start:end]
        if len(y_values) <= 2:
            return 0.0
        x_values = np.arange(len(y_values), dtype=float)
        slope, intercept = np.polyfit(x_values, y_values, 1)
        residuals = y_values - (slope * x_values + intercept)
        return float(np.dot(residuals, residuals))

    def _calibrate_boundaries(self, boundaries, close):
        """将所有方向切换边界校准到附近真实价格极值，同时保护最短长度。"""

        if len(boundaries) <= 2 or self.config.boundary_window == 0:
            return boundaries

        calibrated = boundaries.copy()
        minimum = self.config.min_segment_length
        raw_log_close = np.log(close)
        analysis_log_close = self._smooth_log_price(raw_log_close)
        labels = [
            self._segment_features(
                boundaries[index], boundaries[index + 1], close, analysis_log_close, raw_log_close
            )["regime"]
            for index in range(len(boundaries) - 1)
        ]
        for index in range(1, len(calibrated) - 1):
            previous_label, next_label = labels[index - 1], labels[index]
            pivot_type = self._transition_pivot_type(previous_label, next_label)
            if pivot_type is None:
                continue
            lower = max(calibrated[index - 1] + minimum, calibrated[index] - self.config.boundary_window)
            upper = min(calibrated[index + 1] - minimum, calibrated[index] + self.config.boundary_window)
            if lower > upper:
                continue
            candidate_boundaries = np.arange(lower, upper + 1)
            candidate_prices = close[candidate_boundaries - 1]
            local_index = np.argmax(candidate_prices) if pivot_type == "top" else np.argmin(candidate_prices)
            calibrated[index] = int(candidate_boundaries[local_index])
        return calibrated

    @staticmethod
    def _transition_pivot_type(previous_regime, next_regime):
        if previous_regime == next_regime:
            return None
        if previous_regime == "up" or next_regime == "down":
            return "top"
        if previous_regime == "down" or next_regime == "up":
            return "bottom"
        return None

    def _build_segments(
        self, boundaries, close, analysis_log_close, raw_log_close
    ):
        return [
            self._segment_features(start, end, np.asarray(close, dtype=float), analysis_log_close, raw_log_close)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

    def _merge_segments(
        self, segments, close, analysis_log_close, raw_log_close
    ):
        """合并同态或过短相邻区间，避免噪声造成频繁切换。"""

        prices = np.asarray(close, dtype=float)
        changed = True
        while changed and len(segments) > 1:
            changed = False
            for index in range(len(segments) - 1):
                combined_duration = segments[index + 1]["end"] - segments[index]["start"]
                can_merge_sideways = (
                    segments[index]["regime"] != "sideways"
                    or combined_duration <= self.config.max_segment_length
                )
                if segments[index]["regime"] == segments[index + 1]["regime"] and can_merge_sideways:
                    segments[index : index + 2] = [
                        self._segment_features(
                            segments[index]["start"], segments[index + 1]["end"], prices,
                            analysis_log_close, raw_log_close,
                        )
                    ]
                    changed = True
                    break
            if changed:
                continue

            short_index = next(
                (index for index, segment in enumerate(segments) if segment["duration"] < self.config.min_segment_length),
                None,
            )
            if short_index is None:
                continue
            choices = []
            if short_index > 0:
                start, end = segments[short_index - 1]["start"], segments[short_index]["end"]
                choices.append((self._merge_cost(segments, short_index - 1, short_index, analysis_log_close), short_index - 1, short_index))
            if short_index < len(segments) - 1:
                start, end = segments[short_index]["start"], segments[short_index + 1]["end"]
                choices.append((self._merge_cost(segments, short_index, short_index + 1, analysis_log_close), short_index, short_index + 1))
            _, left, right = min(choices, key=lambda item: item[0])
            segments[left : right + 1] = [
                self._segment_features(
                    segments[left]["start"], segments[right]["end"], prices,
                    analysis_log_close, raw_log_close,
                )
            ]
            changed = True
        return segments

    def _merge_cost(self, segments, left, right, analysis_log_close):
        start, end = segments[left]["start"], segments[right]["end"]
        cost = self._linear_sse(analysis_log_close, start, end)
        if segments[left]["regime"] != segments[right]["regime"]:
            cost *= 1.25
        return cost

    def _segment_features(
        self, start, end, close, analysis_log_close, raw_log_close
    ):
        segment_log = analysis_log_close[start:end]
        raw_segment_log = raw_log_close[start:end]
        segment_close = close[start:end]
        duration = end - start
        daily_returns = np.diff(raw_segment_log)
        volatility = float(np.std(daily_returns, ddof=0)) if len(daily_returns) else 0.0
        log_return = float(segment_log[-1] - segment_log[0])
        simple_return = float(segment_close[-1] / segment_close[0] - 1)
        x_values = np.arange(duration, dtype=float)
        slope, intercept = np.polyfit(x_values, segment_log, 1) if duration > 1 else (0.0, segment_log[0])
        fitted = slope * x_values + intercept
        total_variation = float(np.dot(segment_log - segment_log.mean(), segment_log - segment_log.mean()))
        residual_variation = float(np.dot(segment_log - fitted, segment_log - fitted))
        r_squared = 1.0 - residual_variation / total_variation if total_variation > 0 else 0.0
        path_length = float(np.abs(daily_returns).sum())
        efficiency_ratio = abs(log_return) / path_length if path_length > 0 else 0.0
        running_high = np.maximum.accumulate(segment_close)
        max_drawdown = float((segment_close / running_high - 1).min())
        dynamic_threshold = max(
            float(np.log1p(self.config.min_return)),
            self.config.volatility_multiplier
            * volatility
            * np.sqrt(min(max(duration - 1, 1), self.config.volatility_horizon)),
        )
        dynamic_threshold = min(dynamic_threshold, float(np.log1p(self.config.dynamic_threshold_cap)))
        slope_score = float(slope / volatility) if volatility > 0 else 0.0
        direction = 1 if log_return > 0 and slope > 0 else -1 if log_return < 0 and slope < 0 else 0
        evidence_count = sum(
            [
                abs(log_return) >= dynamic_threshold,
                efficiency_ratio >= self.config.min_efficiency_ratio,
                r_squared >= self.config.min_r_squared,
            ]
        )
        regime = "up" if direction > 0 and evidence_count >= 2 else "down" if direction < 0 and evidence_count >= 2 else "sideways"

        direction_strength = min(abs(log_return) / max(dynamic_threshold, np.finfo(float).eps), 1.0)
        efficiency_strength = min(efficiency_ratio / max(self.config.min_efficiency_ratio, np.finfo(float).eps), 1.0)
        r_squared_strength = min(max(r_squared, 0.0) / max(self.config.min_r_squared, np.finfo(float).eps), 1.0)
        confidence = float(np.clip(np.mean([direction_strength, efficiency_strength, r_squared_strength]), 0.0, 1.0))
        return {
            "start": start,
            "end": end,
            "duration": duration,
            "regime": regime,
            "start_price": float(segment_close[0]),
            "end_price": float(segment_close[-1]),
            "segment_return": simple_return,
            "log_return": log_return,
            "volatility": volatility,
            "dynamic_threshold": dynamic_threshold,
            "slope": float(slope),
            "slope_score": slope_score,
            "r_squared": float(r_squared),
            "efficiency_ratio": float(efficiency_ratio),
            "max_drawdown": max_drawdown,
            "confidence": confidence,
        }

    def _to_segment_table(self, segments, close, symbol):
        rows = []
        for segment_id, segment in enumerate(segments):
            row = {
                "symbol": str(symbol),
                "segment_id": segment_id,
                "start_date": close.index[segment["start"]],
                "end_date": close.index[segment["end"] - 1],
                "open_segment": segment_id == len(segments) - 1,
                "label_version": self.config.label_version,
            }
            row.update({key: value for key, value in segment.items() if key not in {"start", "end"}})
            rows.append(row)
        return pd.DataFrame(rows)

    def _to_daily_table(self, segments, dates, symbol):
        rows = []
        for segment in segments.itertuples(index=False):
            mask = (dates >= segment.start_date) & (dates <= segment.end_date)
            segment_dates = dates[mask]
            for offset, date in enumerate(segment_dates):
                rows.append(
                    {
                        "date": date,
                        "symbol": str(symbol),
                        "segment_id": segment.segment_id,
                        "regime": segment.regime,
                        "days_from_segment_start": offset,
                        "days_to_segment_end": len(segment_dates) - offset - 1,
                        "regime_confidence": segment.confidence,
                        "is_pivot": False,
                        "open_segment": segment.open_segment,
                        "label_version": self.config.label_version,
                    }
                )
        result = pd.DataFrame(rows)
        if len(segments) > 1:
            pivot_dates = segments["end_date"].iloc[:-1]
            result.loc[result["date"].isin(pivot_dates), "is_pivot"] = True
        return result

    def _to_pivot_table(self, segments, close, symbol):
        rows = []
        prices = np.asarray(close, dtype=float)
        dates = close.index
        daily_log_returns = np.diff(np.log(prices))
        for index in range(len(segments) - 1):
            previous, following = segments.iloc[index], segments.iloc[index + 1]
            pivot_type = self._transition_pivot_type(previous.regime, following.regime)
            if pivot_type is None:
                continue
            pivot_date = previous.end_date
            pivot_position = int(dates.get_loc(pivot_date))
            reversal_threshold = self._pivot_reversal_threshold(daily_log_returns, pivot_position)
            confirm_position = self._find_confirmation_position(prices, pivot_position, pivot_type, reversal_threshold)
            confirm_date = dates[confirm_position] if confirm_position is not None else pd.NaT
            confirmation_lag = confirm_position - pivot_position if confirm_position is not None else np.nan
            move_after = (
                float(prices[confirm_position] / prices[pivot_position] - 1)
                if confirm_position is not None
                else following.segment_return
            )
            rows.append(
                {
                    "symbol": str(symbol),
                    "pivot_date": pivot_date,
                    "pivot_type": pivot_type,
                    "confirm_date": confirm_date,
                    "confirmation_lag": confirmation_lag,
                    "previous_regime": previous.regime,
                    "next_regime": following.regime,
                    "move_before": previous.segment_return,
                    "move_after": move_after,
                    "reversal_threshold": reversal_threshold,
                    "pivot_confidence": float(min(previous.confidence, following.confidence)),
                    "label_version": self.config.label_version,
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "symbol", "pivot_date", "pivot_type", "confirm_date", "confirmation_lag",
                "previous_regime", "next_regime", "move_before", "move_after", "reversal_threshold",
                "pivot_confidence", "label_version",
            ],
        )

    def _pivot_reversal_threshold(self, daily_log_returns, pivot_position):
        start = max(0, pivot_position - self.config.pivot_volatility_window)
        local_returns = daily_log_returns[start:pivot_position]
        local_volatility = float(np.std(local_returns, ddof=0)) if len(local_returns) else 0.0
        return max(self.config.pivot_min_reversal, self.config.pivot_reversal_multiplier * local_volatility)

    @staticmethod
    def _find_confirmation_position(prices, pivot_position, pivot_type, reversal_threshold):
        pivot_price = prices[pivot_position]
        if pivot_type == "top":
            target = pivot_price * (1 - reversal_threshold)
            matches = np.flatnonzero(prices[pivot_position + 1 :] <= target)
        else:
            target = pivot_price * (1 + reversal_threshold)
            matches = np.flatnonzero(prices[pivot_position + 1 :] >= target)
        return int(pivot_position + 1 + matches[0]) if len(matches) else None
