from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_EVENT_STUDY_DIR = Path("output/pattern_event_study")
REQUIRED_EVENT_WINDOW_COLUMNS = {
    "symbol",
    "event_date",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "relative_day",
}


def find_event_window_files(output_dir: Path | str = DEFAULT_EVENT_STUDY_DIR) -> list[str]:
    base_dir = Path(output_dir)
    if not base_dir.exists():
        return []
    return sorted(str(path) for path in base_dir.glob("event_windows_*.csv"))


def infer_event_meta_file(event_windows_file: Path | str) -> Path:
    event_windows_path = Path(event_windows_file)
    return event_windows_path.with_name(
        event_windows_path.name.replace("event_windows_", "event_meta_", 1)
    )


def read_csv_with_string_columns(path: Path | str, string_columns: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    dtype = {column: "string" for column in string_columns if column in header.columns}
    return pd.read_csv(path, dtype=dtype)


def validate_event_window_columns(event_windows: pd.DataFrame) -> None:
    missing_columns = sorted(REQUIRED_EVENT_WINDOW_COLUMNS - set(event_windows.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"event_windows 文件缺少必要字段: {missing_text}")


def load_event_study_data(
    event_windows_file: Path | str,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    event_windows = read_csv_with_string_columns(event_windows_file, ["symbol", "code"])
    if event_windows.empty:
        raise ValueError("event_windows 文件为空")

    validate_event_window_columns(event_windows)

    event_windows["symbol"] = event_windows["symbol"].astype(str).str.zfill(6)
    if "code" in event_windows.columns:
        event_windows["code"] = event_windows["code"].astype(str).str.zfill(6)
    event_windows["trade_date"] = pd.to_datetime(event_windows["trade_date"])
    event_windows["event_date"] = pd.to_datetime(event_windows["event_date"])

    event_windows = event_windows.sort_values(
        ["symbol", "event_date", "trade_date"]
    ).reset_index(drop=True)

    event_meta_file = infer_event_meta_file(event_windows_file)
    event_meta = None
    if event_meta_file.exists():
        event_meta = read_csv_with_string_columns(event_meta_file, ["symbol"])
        if not event_meta.empty:
            if "symbol" in event_meta.columns:
                event_meta["symbol"] = event_meta["symbol"].astype(str).str.zfill(6)
            if "event_date" in event_meta.columns:
                event_meta["event_date"] = pd.to_datetime(event_meta["event_date"])
            event_meta = event_meta.sort_values(["symbol", "event_date"]).reset_index(drop=True)

    event_catalog = build_event_catalog(event_windows, event_meta)
    return event_windows, event_meta, event_catalog


def build_event_catalog(
    event_windows: pd.DataFrame,
    event_meta: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if "event_close" not in event_windows.columns:
        event_close_frame = (
            event_windows[event_windows["relative_day"] == 0][["symbol", "event_date", "close"]]
            .rename(columns={"close": "event_close"})
            .drop_duplicates(subset=["symbol", "event_date"])
        )
        event_windows = event_windows.merge(
            event_close_frame,
            on=["symbol", "event_date"],
            how="left",
        )

    catalog = (
        event_windows.groupby(["symbol", "event_date"], as_index=False)
        .agg(
            window_start=("trade_date", "min"),
            window_end=("trade_date", "max"),
            bar_count=("trade_date", "size"),
            min_relative_day=("relative_day", "min"),
            max_relative_day=("relative_day", "max"),
            event_close=("event_close", "first"),
        )
        .sort_values(["symbol", "event_date"])
        .reset_index(drop=True)
    )

    if event_meta is not None and not event_meta.empty:
        meta_columns = [column for column in event_meta.columns if column not in {"event_close"}]
        catalog = catalog.merge(
            event_meta[meta_columns],
            on=["symbol", "event_date"],
            how="left",
        )

    return catalog


def get_symbol_event_catalog(event_catalog: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return (
        event_catalog[event_catalog["symbol"] == symbol]
        .sort_values("event_date")
        .reset_index(drop=True)
    )


def get_event_window(
    event_windows: pd.DataFrame,
    symbol: str,
    event_date: pd.Timestamp,
    visible_days: Optional[int] = None,
) -> pd.DataFrame:
    window = event_windows[
        (event_windows["symbol"] == symbol)
        & (event_windows["event_date"] == pd.Timestamp(event_date))
    ].sort_values("trade_date")

    if visible_days is not None:
        window = window[window["relative_day"].between(-visible_days, visible_days)]

    return window.reset_index(drop=True)
