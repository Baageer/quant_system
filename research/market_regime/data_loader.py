"""本地 HS300 行情缓存的只读加载工具。"""

from pathlib import Path

import pandas as pd


DEFAULT_PRICE_HISTORY_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "tushare"
    / "price_history"
)


def load_hs300_symbols(stock_file=None):
    """读取 HS300 六位代码列表，保留文件中的原始顺序。"""

    path = Path(stock_file) if stock_file else Path(__file__).resolve().parents[2] / "data" / "HS300.txt"
    if not path.is_file():
        raise FileNotFoundError("未找到 HS300 股票池文件：{}".format(path))
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def to_tushare_symbol(symbol):
    """将六位 A 股代码规范化为本地 TuShare 缓存文件名使用的代码。"""

    normalized = str(symbol).strip().upper()
    if not normalized:
        raise ValueError("股票代码不能为空。")
    if "." in normalized:
        return normalized
    return "{}.SH".format(normalized) if normalized[0] in {"5", "6", "9"} else "{}.SZ".format(normalized)


def find_local_price_files(symbol, price_history_dir=None, adjustment="raw_hfq_pct"):
    """返回指定复权口径下的全部本地快照文件。"""

    root = Path(price_history_dir) if price_history_dir else DEFAULT_PRICE_HISTORY_DIR
    directory = root / adjustment
    tushare_symbol = to_tushare_symbol(symbol)
    return sorted(directory.glob("{}_*.csv".format(tushare_symbol)))


def find_local_price_files_multi(symbol, price_history_dirs, adjustment="raw_hfq_pct"):
    """跨多个行情根目录查找同一标的的快照文件，顺序按传入顺序保留。"""

    files = []
    for root in price_history_dirs:
        files.extend(find_local_price_files(symbol, root, adjustment))
    return files


def load_local_price_history_multi(symbol, price_history_dirs, adjustment="raw_hfq_pct"):
    """合并多个行情根目录的同一标的快照，语义与 :func:`load_local_price_history` 一致。

    前向窗口需要把 2026 的拼接数据接在历史缓存后面，但**不能**改写历史缓存
    （已发布数字的可复现性依赖它）。所以两棵树在读取时显式合并：逐目录取文件、
    concat、按 ``date`` 排序去重（``keep="last"``，后传入的目录胜出）。

    与单目录版本的唯一区别是文件来源；排序/去重口径完全相同，因此对只存在于
    一个目录的标的，两者返回逐位相同的帧。
    """

    roots = [Path(root) for root in price_history_dirs]
    files = find_local_price_files_multi(symbol, roots, adjustment)
    if not files:
        raise FileNotFoundError(
            "未找到 {} 的 {} 本地行情缓存（查找目录：{}）。".format(
                to_tushare_symbol(symbol), adjustment, ", ".join(str(root) for root in roots))
        )

    frames = []
    for file_path in files:
        frame = pd.read_csv(file_path)
        frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
        if "date" not in frame.columns or "close" not in frame.columns:
            raise ValueError("行情文件缺少 date 或 close 列：{}".format(file_path))
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["date", "close"])
    data = data.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    data = data.set_index("date").sort_index()
    if data.empty:
        raise ValueError("本地行情不包含可用的日期和收盘价：{}".format(to_tushare_symbol(symbol)))
    return data


def load_local_price_history(symbol, price_history_dir=None, adjustment="raw_hfq_pct"):
    """合并同一标的的本地快照，并按日期返回标准 OHLC 数据。"""

    files = find_local_price_files(symbol, price_history_dir, adjustment)
    if not files:
        raise FileNotFoundError(
            "未找到 {} 的 {} 本地行情缓存。".format(to_tushare_symbol(symbol), adjustment)
        )

    frames = []
    for file_path in files:
        frame = pd.read_csv(file_path)
        frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
        if "date" not in frame.columns or "close" not in frame.columns:
            raise ValueError("行情文件缺少 date 或 close 列：{}".format(file_path))
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["date", "close"])
    data = data.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    data = data.set_index("date").sort_index()
    if data.empty:
        raise ValueError("本地行情不包含可用的日期和收盘价：{}".format(to_tushare_symbol(symbol)))
    return data
