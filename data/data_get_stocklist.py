"""Fetch Shanghai/Shenzhen A-share stock codes and save to txt files."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return False


try:
    import tushare as ts
except ModuleNotFoundError:  # pragma: no cover - runtime guard for minimal envs
    ts = None

try:
    import akshare as ak
except ModuleNotFoundError:  # pragma: no cover - runtime guard for minimal envs
    ak = None

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MARKET_DISPLAY = {
    "SH": "Shanghai",
    "SZ": "Shenzhen",
}
SH1000_INDEX_CODE = "000852.SH"
ZZ500_INDEX_CODE = "000905.SH"


def _require_tushare() -> None:
    if ts is None:
        raise ModuleNotFoundError(
            "tushare is required for this script. Install it with `pip install tushare`."
        )


def _require_akshare() -> None:
    if ak is None:
        raise ModuleNotFoundError(
            "akshare is required for this fallback. Install it with `pip install akshare`."
        )


def _numeric_sort(codes: Iterable[str]) -> List[str]:
    unique_codes = sorted(set(codes), key=lambda item: int(item))
    return unique_codes


def _is_market_a_share_code(code: str, market: str) -> bool:
    if len(code) != 6 or not code.isdigit():
        return False
    if market == "SH":
        return code.startswith("6")
    if market == "SZ":
        return code.startswith(("000", "001", "002", "003", "300", "301"))
    raise ValueError(f"Unsupported market: {market}")


def _is_a_share_code(code: str) -> bool:
    return _is_market_a_share_code(code, "SH") or _is_market_a_share_code(code, "SZ")


def _fetch_from_tushare_pro(token: str, market: str) -> List[str]:
    _require_tushare()
    exchange = "SSE" if market == "SH" else "SZSE"
    pro = ts.pro_api(token=token)
    data = pro.stock_basic(
        exchange=exchange,
        list_status="L",
        fields="ts_code,symbol,name,market",
    )
    if data is None or data.empty:
        raise RuntimeError(f"tushare pro returned empty stock_basic result for {exchange}.")

    # Keep A-share only and remove B-share rows when market labels are available.
    if "market" in data.columns:
        data = data[~data["market"].astype(str).str.contains("B", na=False)].copy()

    if "symbol" in data.columns:
        symbols = data["symbol"].astype(str).str.strip()
    elif "ts_code" in data.columns:
        symbols = data["ts_code"].astype(str).str.split(".", n=1).str[0].str.strip()
    else:
        raise RuntimeError("stock_basic result does not contain symbol/ts_code columns.")

    codes = [code for code in symbols if _is_market_a_share_code(code, market)]
    if not codes:
        raise RuntimeError(f"No {MARKET_DISPLAY[market]} A-share codes found from tushare pro stock_basic.")
    return _numeric_sort(codes)


def _fetch_from_tushare_legacy(market: str) -> List[str]:
    _require_tushare()
    data = ts.get_stock_basics()
    if data is None or data.empty:
        raise RuntimeError("tushare get_stock_basics returned empty result.")

    codes = [str(code).strip() for code in data.index]
    market_codes = [code for code in codes if _is_market_a_share_code(code, market)]
    if not market_codes:
        raise RuntimeError(f"No {MARKET_DISPLAY[market]} A-share codes found from tushare get_stock_basics.")
    return _numeric_sort(market_codes)


def _fetch_from_akshare(market: str) -> List[str]:
    _require_akshare()
    data = ak.stock_info_a_code_name()
    if data is None or data.empty or "code" not in data.columns:
        raise RuntimeError("akshare stock_info_a_code_name returned empty or invalid result.")

    codes = data["code"].astype(str).str.strip()
    market_codes = [code for code in codes if _is_market_a_share_code(code, market)]
    if not market_codes:
        raise RuntimeError(f"No {MARKET_DISPLAY[market]} A-share codes found from akshare.")
    return _numeric_sort(market_codes)


def _extract_code_from_text(value: object) -> str:
    text = str(value).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    if text.isdigit() and len(text) < 6:
        text = text.zfill(6)
    return text


def _looks_like_code_series(series) -> bool:
    sample = series.dropna().astype(str).str.strip().head(50)
    if sample.empty:
        return False
    normalized = sample.str.split(".", n=1).str[0]
    normalized = normalized.map(lambda item: item.zfill(6) if item.isdigit() and len(item) < 6 else item)
    valid_mask = normalized.str.fullmatch(r"\d{6}")
    valid_ratio = valid_mask.mean()
    if valid_ratio < 0.6:
        return False
    valid_series = normalized[valid_mask]
    return bool(valid_series.nunique() >= 10)


def _pick_code_column(data, candidates: Iterable[str]) -> str | None:
    columns = {str(col).strip().lower(): str(col) for col in data.columns}
    for candidate in candidates:
        name = columns.get(candidate.lower())
        if name:
            return name
    for col in data.columns:
        col_name = str(col)
        lower_name = col_name.lower()
        if "code" in lower_name:
            return col_name
    for col in data.columns:
        if _looks_like_code_series(data[col]):
            return str(col)
    return None


def _fetch_sh1000_from_tushare_pro(token: str) -> List[str]:
    _require_tushare()
    pro = ts.pro_api(token=token)
    end_date = datetime.now().strftime("%Y%m%d")

    # Expand the query window progressively in case the latest rebalance date
    # is not covered by a short lookback period.
    for lookback_days in (120, 400, 1200):
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")
        data = pro.index_weight(
            index_code=SH1000_INDEX_CODE,
            start_date=start_date,
            end_date=end_date,
            fields="con_code,trade_date",
        )
        if data is None or data.empty:
            continue

        if "trade_date" in data.columns:
            latest_trade_date = data["trade_date"].astype(str).max()
            data = data[data["trade_date"].astype(str) == latest_trade_date].copy()

        if "con_code" not in data.columns:
            raise RuntimeError("tushare pro index_weight result missing con_code column.")

        symbols = data["con_code"].astype(str).map(_extract_code_from_text)
        codes = [code for code in symbols if _is_a_share_code(code)]
        if codes:
            return _numeric_sort(codes)

    raise RuntimeError(f"tushare pro returned empty index_weight result for {SH1000_INDEX_CODE}.")


def _fetch_zz500_from_tushare_pro(token: str) -> List[str]:
    _require_tushare()
    pro = ts.pro_api(token=token)
    end_date = datetime.now().strftime("%Y%m%d")

    # Expand the query window progressively in case the latest rebalance date
    # is not covered by a short lookback period.
    for lookback_days in (120, 400, 1200):
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")
        data = pro.index_weight(
            index_code=ZZ500_INDEX_CODE,
            start_date=start_date,
            end_date=end_date,
            fields="con_code,trade_date",
        )
        if data is None or data.empty:
            continue

        if "trade_date" in data.columns:
            latest_trade_date = data["trade_date"].astype(str).max()
            data = data[data["trade_date"].astype(str) == latest_trade_date].copy()

        if "con_code" not in data.columns:
            raise RuntimeError("tushare pro index_weight result missing con_code column.")

        symbols = data["con_code"].astype(str).map(_extract_code_from_text)
        codes = [code for code in symbols if _is_a_share_code(code)]
        if codes:
            return _numeric_sort(codes)

    raise RuntimeError(f"tushare pro returned empty index_weight result for {ZZ500_INDEX_CODE}.")


def _fetch_sh1000_from_akshare() -> List[str]:
    _require_akshare()
    attempts = [
        ("index_stock_cons_csindex", {"symbol": "000852"}),
        ("index_stock_cons_weight_csindex", {"symbol": "000852"}),
        ("index_stock_cons", {"symbol": "000852"}),
        ("index_stock_cons", {"symbol": SH1000_INDEX_CODE}),
        ("index_stock_cons_csindex", {"symbol": SH1000_INDEX_CODE}),
    ]
    code_col_candidates = (
        "con_code",
        "code",
        "stock_code",
    )

    last_error = None
    for func_name, kwargs in attempts:
        func = getattr(ak, func_name, None)
        if func is None:
            continue
        try:
            data = func(**kwargs)
        except Exception as exc:  # pragma: no cover - depends on akshare runtime
            last_error = exc
            continue
        if data is None or data.empty:
            continue

        code_col = _pick_code_column(data, code_col_candidates)
        if not code_col:
            continue

        symbols = data[code_col].astype(str).map(_extract_code_from_text)
        codes = [code for code in symbols if _is_a_share_code(code)]
        if codes:
            return _numeric_sort(codes)

    detail = f": {last_error}" if last_error else "."
    raise RuntimeError(f"akshare failed to fetch SH1000 constituents{detail}")


def _fetch_zz500_from_akshare() -> List[str]:
    _require_akshare()
    attempts = [
        ("index_stock_cons_csindex", {"symbol": "000905"}),
        ("index_stock_cons_weight_csindex", {"symbol": "000905"}),
        ("index_stock_cons", {"symbol": "000905"}),
        ("index_stock_cons", {"symbol": ZZ500_INDEX_CODE}),
        ("index_stock_cons_csindex", {"symbol": ZZ500_INDEX_CODE}),
    ]
    code_col_candidates = (
        "con_code",
        "code",
        "stock_code",
    )

    last_error = None
    for func_name, kwargs in attempts:
        func = getattr(ak, func_name, None)
        if func is None:
            continue
        try:
            data = func(**kwargs)
        except Exception as exc:  # pragma: no cover - depends on akshare runtime
            last_error = exc
            continue
        if data is None or data.empty:
            continue

        code_col = _pick_code_column(data, code_col_candidates)
        if not code_col:
            continue

        symbols = data[code_col].astype(str).map(_extract_code_from_text)
        codes = [code for code in symbols if _is_a_share_code(code)]
        if codes:
            return _numeric_sort(codes)

    detail = f": {last_error}" if last_error else "."
    raise RuntimeError(f"akshare failed to fetch ZZ500 constituents{detail}")


def fetch_a_share_codes(market: str) -> List[str]:
    market = str(market).strip().upper()
    if market not in MARKET_DISPLAY:
        raise ValueError(f"Unsupported market: {market}. Expected one of SH, SZ.")

    token = os.getenv("TS_TOKEN") or os.getenv("TUSHARE_TOKEN")
    if token:
        try:
            return _fetch_from_tushare_pro(token, market)
        except Exception as exc:
            print(f"[warn] Tushare Pro failed: {exc}. Falling back to AkShare.")

    try:
        return _fetch_from_akshare(market)
    except Exception as exc:
        print(f"[warn] AkShare fallback failed: {exc}. Falling back to tushare legacy API.")

    return _fetch_from_tushare_legacy(market)


def fetch_sh_1000_codes() -> List[str]:
    token = os.getenv("TS_TOKEN") or os.getenv("TUSHARE_TOKEN")
    if token:
        try:
            return _fetch_sh1000_from_tushare_pro(token)
        except Exception as exc:
            print(f"[warn] Tushare Pro SH1000 fetch failed: {exc}. Falling back to AkShare.")

    return _fetch_sh1000_from_akshare()


def fetch_zz_500_codes() -> List[str]:
    token = os.getenv("TS_TOKEN") or os.getenv("TUSHARE_TOKEN")
    if token:
        try:
            return _fetch_zz500_from_tushare_pro(token)
        except Exception as exc:
            print(f"[warn] Tushare Pro ZZ500 fetch failed: {exc}. Falling back to AkShare.")

    return _fetch_zz500_from_akshare()


def fetch_sh_a_share_codes() -> List[str]:
    return fetch_a_share_codes("SH")


def fetch_sz_a_share_codes() -> List[str]:
    return fetch_a_share_codes("SZ")


def save_codes(codes: List[str], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(codes) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch A-share code lists.")
    parser.add_argument(
        "--scope",
        type=str,
        choices=("all", "sh1000", "zz500"),
        default="all",
        help="Universe selector: all (exchange-wide), sh1000 (CSI 1000), or zz500 (CSI 500).",
    )
    parser.add_argument(
        "--market",
        type=str,
        choices=("SH", "SZ"),
        default="SH",
        help="Market selector: SH for Shanghai, SZ for Shenzhen (default: SH).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path. Default: data/SH_ALL.txt/data/SZ_ALL.txt for --scope all, data/SH_1000.txt for --scope sh1000, or data/ZZ_500.txt for --scope zz500.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scope = str(args.scope).lower()

    if scope == "sh1000":
        output_file = args.output or (Path(__file__).resolve().parent / "SH_1000.txt")
        codes = fetch_sh_1000_codes()
        save_codes(codes, output_file)
        print(f"Saved {len(codes)} SH1000 constituent codes to: {output_file}")
        return

    if scope == "zz500":
        output_file = args.output or (Path(__file__).resolve().parent / "ZZ_500.txt")
        codes = fetch_zz_500_codes()
        save_codes(codes, output_file)
        print(f"Saved {len(codes)} ZZ500 constituent codes to: {output_file}")
        return

    market = str(args.market).upper()
    output_file = args.output or (Path(__file__).resolve().parent / f"{market}_ALL.txt")
    codes = fetch_a_share_codes(market)
    save_codes(codes, output_file)
    print(f"Saved {len(codes)} {MARKET_DISPLAY[market]} A-share codes to: {output_file}")


if __name__ == "__main__":
    main()
