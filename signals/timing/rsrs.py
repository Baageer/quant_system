"""
RSRS timing strategy.
"""
from typing import Dict, Tuple

import pandas as pd

from signals.indicators import rsrs


def _is_missing(value: float) -> bool:
    return value != value


def _apply_signal_delay(signal: pd.Series, signal_delay: int) -> pd.Series:
    if signal_delay <= 0:
        return signal.astype(int)
    return signal.shift(signal_delay).fillna(0).astype(int)


def _validate_required_columns(data: pd.DataFrame, required_columns) -> None:
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def rsrs_breakout_signal(
    data: pd.DataFrame,
    window: int = 18,
    zscore_window: int = 120,
    min_valid_window: int = 12,
    entry_zscore: float = 0.7,
    exit_zscore: float = 0.0,
    breakout_direction: str = "both",
    use_r2_weight: bool = True,
    use_beta_adjustment: bool = False,
    signal_delay: int = 0,
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    """
    RSRS breakout style timing signal.

    Signal semantics:
    - 1: long
    - -1: short / exit long
    - 0: flat
    """
    if breakout_direction not in {"up", "down", "both"}:
        raise ValueError("breakout_direction must be one of: up, down, both")

    _validate_required_columns(data, [high_col, low_col])
    beta, r2, zscore, score = rsrs(
        high=data[high_col],
        low=data[low_col],
        window=window,
        zscore_window=zscore_window,
        min_valid_window=min_valid_window,
        use_r2_weight=use_r2_weight,
        use_beta_adjustment=use_beta_adjustment,
    )

    position = pd.Series(0, index=data.index, dtype=int)
    current_pos = 0

    for i in range(len(data)):
        current_score = score.iloc[i]
        if _is_missing(current_score):
            position.iloc[i] = current_pos
            continue

        if current_pos == 0:
            if breakout_direction in {"up", "both"} and current_score >= entry_zscore:
                current_pos = 1
            elif breakout_direction in {"down", "both"} and current_score <= -entry_zscore:
                current_pos = -1
        elif current_pos == 1:
            if current_score <= exit_zscore:
                current_pos = 0
            elif breakout_direction in {"down", "both"} and current_score <= -entry_zscore:
                current_pos = -1
        else:
            if current_score >= -exit_zscore:
                current_pos = 0
            elif breakout_direction in {"up", "both"} and current_score >= entry_zscore:
                current_pos = 1

        position.iloc[i] = current_pos

    return _apply_signal_delay(position, signal_delay)


class RSRSStrategy:
    """RSRS strategy wrapper for backtest loader."""

    VALID_DIRECTIONS = {"up", "down", "both"}

    def __init__(
        self,
        window: int = 18,
        zscore_window: int = 120,
        min_valid_window: int = 12,
        entry_zscore: float = 0.7,
        exit_zscore: float = 0.0,
        breakout_direction: str = "both",
        use_r2_weight: bool = True,
        use_beta_adjustment: bool = False,
        signal_delay: int = 0,
        high_col: str = "high",
        low_col: str = "low",
    ):
        if window < 2:
            raise ValueError("window must be at least 2")
        if zscore_window < 10:
            raise ValueError("zscore_window must be at least 10")
        if min_valid_window < 2:
            raise ValueError("min_valid_window must be at least 2")
        if entry_zscore <= 0:
            raise ValueError("entry_zscore must be positive")
        if exit_zscore < 0:
            raise ValueError("exit_zscore must be non-negative")
        if breakout_direction not in self.VALID_DIRECTIONS:
            raise ValueError(
                f"breakout_direction must be one of {sorted(self.VALID_DIRECTIONS)}"
            )
        if signal_delay < 0:
            raise ValueError("signal_delay must be non-negative")

        self.window = window
        self.zscore_window = zscore_window
        self.min_valid_window = min_valid_window
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.breakout_direction = breakout_direction
        self.use_r2_weight = use_r2_weight
        self.use_beta_adjustment = use_beta_adjustment
        self.signal_delay = signal_delay
        self.high_col = high_col
        self.low_col = low_col

    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        return rsrs_breakout_signal(
            data=data,
            window=self.window,
            zscore_window=self.zscore_window,
            min_valid_window=self.min_valid_window,
            entry_zscore=self.entry_zscore,
            exit_zscore=self.exit_zscore,
            breakout_direction=self.breakout_direction,
            use_r2_weight=self.use_r2_weight,
            use_beta_adjustment=self.use_beta_adjustment,
            signal_delay=self.signal_delay,
            high_col=self.high_col,
            low_col=self.low_col,
        )

    def get_rsrs_values(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        _validate_required_columns(data, [self.high_col, self.low_col])
        beta, r2, zscore, score = rsrs(
            high=data[self.high_col],
            low=data[self.low_col],
            window=self.window,
            zscore_window=self.zscore_window,
            min_valid_window=self.min_valid_window,
            use_r2_weight=self.use_r2_weight,
            use_beta_adjustment=self.use_beta_adjustment,
        )
        return {
            "rsrs_beta": beta,
            "rsrs_r2": r2,
            "rsrs_zscore": zscore,
            "rsrs_score": score,
        }
