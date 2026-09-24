"""Run the feature panel over the 中证1000 universe in small batches.

Why batches at all: a single 1000-symbol ``feature_event_study`` run holds every
panel in memory at once, and an earlier oversized run was killed by memory
pressure.  100 symbols per process keeps the peak bounded and makes a failure
resumable -- completed batches are skipped on a re-run.

Example::

    python -m research.experiments.run_zz1000_batches --batch-size 100
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from research.market_regime.output_paths import DATASET_DIR

DEFAULT_STOCK_FILE = "data/ZZ_1000.txt"
# The datasets live on D: (E: is FAT32 and cannot hold them); this default was
# the one site still pointing at the retired relative ``output/datasets``, which
# does not exist -- a batch run would have written its panels somewhere no
# downstream step reads and then failed to find them.
DEFAULT_OUTPUT_DIR = str(DATASET_DIR)
DEFAULT_RUN_PREFIX = "zz1000_batch"


def load_symbols(path: Path) -> List[str]:
    """Read a bare six-digit A-share code list, one per line."""

    if not path.is_file():
        raise FileNotFoundError("universe file not found: {}".format(path))
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def chunk(sequence: Sequence[str], size: int) -> List[List[str]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    return [list(sequence[index:index + size]) for index in range(0, len(sequence), size)]


def panel_path(output_dir: Path, run_id: str) -> Path:
    return output_dir / "trend_feature_panel_{}.csv".format(run_id)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-file", default=DEFAULT_STOCK_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--price-history-dir", default=None)
    parser.add_argument("--adjustment", default="raw_hfq_pct")
    parser.add_argument("--force", action="store_true", help="re-run batches whose panel already exists")
    parser.add_argument("--first-batch", type=int, default=1, help="1-based index of the first batch to run")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    symbols = load_symbols(Path(args.stock_file))
    batches = chunk(symbols, args.batch_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[universe] {} symbols -> {} batches of {}".format(len(symbols), len(batches), args.batch_size),
          flush=True)

    started = time.time()
    for index, batch in enumerate(batches, start=1):
        if index < args.first_batch:
            continue
        run_id = "{}_{:02d}".format(args.run_prefix, index)
        panel = panel_path(output_dir, run_id)
        if panel.exists() and not args.force:
            print("[skip] {} already built ({})".format(run_id, panel), flush=True)
            continue
        command = [sys.executable, "-u", "-m", "research.market_regime.feature_event_study",
                   "--symbols"] + batch + [
                   "--output-dir", args.output_dir,
                   "--run-id", run_id,
                   "--adjustment", args.adjustment]
        if args.price_history_dir:
            command += ["--price-history-dir", args.price_history_dir]
        batch_started = time.time()
        print("\n[batch {}/{}] {} ({} symbols)".format(index, len(batches), run_id, len(batch)), flush=True)
        completed = subprocess.run(command)
        if completed.returncode != 0:
            print("[fail] {} exited {}".format(run_id, completed.returncode), flush=True)
            return completed.returncode
        print("[batch {}/{}] {} done in {:.0f}s".format(
            index, len(batches), run_id, time.time() - batch_started), flush=True)

    print("\n[all done] {} batches in {:.0f}s".format(len(batches), time.time() - started), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
