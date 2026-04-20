from research.param_search.conditions.bollinger_squeeze import (
    build_bollinger_squeeze_condition,
)
from research.param_search.conditions.rsrs_breakout import (
    build_rsrs_breakout_condition,
)


CONDITION_BUILDERS = {
    "bollinger_squeeze": build_bollinger_squeeze_condition,
    "rsrs_breakout": build_rsrs_breakout_condition,
}


BASE_CONDITION_PARAMS_MAP = {
    "bollinger_squeeze": {
        "window": 20,
        "num_std": 2,
        "squeeze_threshold": None,
        "squeeze_quantile": 0.05,
        "squeeze_lookback": 60,
        "require_breakout_confirmation": True,
        "breakout_direction": "up",  # up | down | both
        "breakout_buffer": 0.0,
        "breakout_max_wait": 10,
        "breakout_confirm_bars": 1,
        "use_volume_filter": True,
        "volume_window": 60,
        "volume_multiplier": 2,
        "use_trend_filter": True,
        "trend_window": 60,
        "trend_slope_window": 3,
        "use_supertrend_filter": False,
        "supertrend_atr_period": 10,
        "supertrend_multiplier": 3.0,
        "use_band_expansion_filter": False,
        "band_expansion_lookback": 1,
        "use_return_filter": False,
        "min_breakout_return": 0.0,
    },
    "rsrs_breakout": {
        "window": 20,
        "zscore_window": 90,
        "min_valid_window": 12,
        "entry_zscore": 1.0,
        "breakout_direction": "up",  # up | down | both
        "use_r2_weight": True,
        "use_beta_adjustment": False,
        "use_volume_filter": False,
        "volume_window": 20,
        "volume_multiplier": 2,
        "use_trend_filter": True,
        "trend_window": 60,
        "trend_slope_window": 3,
        "use_return_filter": False,
        "min_breakout_return": 0.0,
    },
}


CONDITION_PARAM_GRID_MAP = {
    "bollinger_squeeze": {
        # "window": [20, 30],
        "num_std": [2.0, 2.5, 3.0],
        # "squeeze_quantile": [0.05, 0.1, 0.15],
        # "squeeze_lookback": [40, 60],
        # "breakout_max_wait": [5, 10],
        # "breakout_confirm_bars": [1, 2],
        # "volume_multiplier": [1.5, 2.0, 2.5],
        # "volume_window": [20, 30, 60],
        # "trend_window": [20, 30, 60],
        # "trend_slope_window": [3,5],
        # "supertrend_atr_period": [5, 10, 15],
        # "supertrend_multiplier": [1.5, 3.0, 4.5],
        
    },
    "rsrs_breakout": {
        # "window": [16, 18, 20],
        # "zscore_window": [90, 120, 150],
        # "entry_zscore": [0.6, 0.8, 1.0],
        # "volume_window": [20, 30, 60],
        # "volume_multiplier": [1.2, 1.5, 2.0],
        # "trend_window": [20, 30, 60],
        # "trend_slope_window": [3, 5],
    },
}


def build_condition_frame(df, condition_name, params):
    if condition_name not in CONDITION_BUILDERS:
        available = ", ".join(sorted(CONDITION_BUILDERS.keys()))
        raise ValueError(f"Unsupported condition: {condition_name}. Available: {available}")
    return CONDITION_BUILDERS[condition_name](df, params)
