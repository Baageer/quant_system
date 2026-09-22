"""Where the market-regime pipeline reads and writes its datasets.

The project checkout sits on ``E:``, a FAT32 removable volume.  FAT32 caps a
single file at 4 GB and cannot hold junctions, and the 中证1000 exports already
reached 3.9 GB, so the datasets live on the ``D:`` NTFS volume instead.

Set ``QUANT_OUTPUT_ROOT`` to relocate them.

The main pipeline -- ``feature_event_study``, ``probability_dataset``,
``logistic_model``, ``excess_target_dataset``, ``trade_outcome_model``,
``trade_outcome_ranking_evaluation`` and ``passive_benchmark`` -- defaults to
these paths.  Scripts from closed studies still default to the in-repo
``output``; pass them ``--output-dir`` by hand if they are ever re-run.
"""

import os

OUTPUT_ROOT = os.environ.get("QUANT_OUTPUT_ROOT", r"D:\temp\quant_output")
DATASET_DIR = os.path.join(OUTPUT_ROOT, "datasets")
