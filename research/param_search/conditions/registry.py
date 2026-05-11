from research.param_search.conditions.bollinger_squeeze import (
    build_bollinger_squeeze_condition,
    build_bollinger_squeeze_pullback_condition,
)
from research.param_search.conditions.rsrs_breakout import (
    build_rsrs_breakout_condition,
)


CONDITION_BUILDERS = {
    "bollinger_squeeze": build_bollinger_squeeze_condition,
    "bollinger_squeeze_pullback": build_bollinger_squeeze_pullback_condition,
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
    },
    "bollinger_squeeze_pullback": {
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
        "pullback_reference": "breakout_band",  # breakout_band | middle_band | breakout_close
        "pullback_max_wait": 10,
        "pullback_tolerance": 0.01,
        "require_rebound_confirmation": True,
        "invalidate_on_middle_cross": True,
    },
    "rsrs_breakout": {
        "window": 20,
        "zscore_window": 90,
        "min_valid_window": 12,
        "entry_zscore": 1.0,
        "breakout_direction": "up",  # up | down | both
        "use_r2_weight": True,
        "use_beta_adjustment": False,
    },
}


CONDITION_FILTER_PARAMS_MAP = {
    "volume": {
        "enabled": False,
        "window": 60,
        "multiplier": 2,
    },
    "trend": {
        "enabled": True,
        "window": 60,
        "slope_window": 3,
    },
    "supertrend": {
        "enabled": False,
        "atr_period": 10,
        "multiplier": 3.0,
    },
    "return": {
        "enabled": False,
        "min_breakout_return": 0.0,
    },
}


FILTER_PARAM_KEY_MAP = {
    "volume": {
        "enabled": "use_volume_filter",
        "window": "volume_window",
        "multiplier": "volume_multiplier",
    },
    "trend": {
        "enabled": "use_trend_filter",
        "window": "trend_window",
        "slope_window": "trend_slope_window",
    },
    "supertrend": {
        "enabled": "use_supertrend_filter",
        "atr_period": "supertrend_atr_period",
        "multiplier": "supertrend_multiplier",
    },
    "return": {
        "enabled": "use_return_filter",
        "min_breakout_return": "min_breakout_return",
    },
}


CONDITION_PARAM_GRID_MAP = {
    "bollinger_squeeze": {
        "window": [20, 30],
        # "num_std": [2.0, 2.5, 3.0],
        "squeeze_quantile": [0.05, 0.1, 0.15],
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
    "bollinger_squeeze_pullback": {
        # "num_std": [2.0, 2.5, 3.0],
        "pullback_reference": ["breakout_band", "middle_band"],
        # "pullback_max_wait": [5, 10, 15],
        # "pullback_tolerance": [0.005, 0.01, 0.015],
        # "volume_multiplier": [1.5, 2.0, 2.5],
        # "trend_window": [20, 30, 60],
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


def _flatten_filter_params(filter_params):
    flattened_params = {}
    for filter_name, params in filter_params.items():
        key_map = FILTER_PARAM_KEY_MAP.get(filter_name)
        if key_map is None:
            raise ValueError(f"Unsupported filter params: {filter_name}")
        for param_name, value in params.items():
            target_key = key_map.get(param_name)
            if target_key is None:
                raise ValueError(f"Unsupported {filter_name} filter param: {param_name}")
            flattened_params[target_key] = value
    return flattened_params


def get_condition_filter_params(overrides=None):
    filter_params = {
        filter_name: dict(params)
        for filter_name, params in CONDITION_FILTER_PARAMS_MAP.items()
    }
    for filter_name, params in (overrides or {}).items():
        if filter_name not in FILTER_PARAM_KEY_MAP:
            raise ValueError(f"Unsupported filter params: {filter_name}")
        merged_params = dict(filter_params.get(filter_name, {}))
        merged_params.update(params)
        filter_params[filter_name] = merged_params
    return _flatten_filter_params(filter_params)


def build_condition_params(condition_name, overrides=None):
    if condition_name not in BASE_CONDITION_PARAMS_MAP:
        available = ", ".join(sorted(BASE_CONDITION_PARAMS_MAP.keys()))
        raise ValueError(f"Unsupported condition: {condition_name}. Available: {available}")

    params = dict(BASE_CONDITION_PARAMS_MAP[condition_name])
    overrides = dict(overrides or {})
    nested_filter_overrides = {}
    for filter_name in tuple(FILTER_PARAM_KEY_MAP.keys()):
        nested_filter_params = overrides.pop(filter_name, None)
        if nested_filter_params is not None:
            if not isinstance(nested_filter_params, dict):
                raise ValueError(f"{filter_name} filter params must be a dict")
            nested_filter_overrides[filter_name] = nested_filter_params

    params.update(get_condition_filter_params(nested_filter_overrides))
    params.update(overrides)
    return params


def build_condition_frame(df, condition_name, params):
    if condition_name not in CONDITION_BUILDERS:
        available = ", ".join(sorted(CONDITION_BUILDERS.keys()))
        raise ValueError(f"Unsupported condition: {condition_name}. Available: {available}")
    return CONDITION_BUILDERS[condition_name](df, build_condition_params(condition_name, params))
