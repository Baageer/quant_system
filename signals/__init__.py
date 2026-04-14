"""Signals package compatibility helpers."""

import pandas as pd


# pandas 0.20.x does not expose `pd.isna`, while the codebase and tests use it.
if not hasattr(pd, "isna"):
    pd.isna = pd.isnull
