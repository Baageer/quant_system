"""Ad-hoc driver: pytest is not installed in this env (see AGENTS/testing notes)."""

import glob
import importlib
import os
import sys

# Running this file directly puts ``tests/`` on sys.path, not the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pattern = sys.argv[1] if len(sys.argv) > 1 else "tests/test_market_regime_*.py"
total = fails = 0
for path in sorted(glob.glob(pattern)):
    name = os.path.splitext(path.replace(os.sep, "/"))[0].replace("/", ".")
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print("IMPORTFAIL", name, type(exc).__name__, str(exc)[:140])
        fails += 1
        continue
    for fn in sorted(n for n in dir(module) if n.startswith("test_")):
        total += 1
        try:
            getattr(module, fn)()
        except Exception as exc:
            fails += 1
            print("FAIL", name, fn, type(exc).__name__, str(exc)[:180])
print("total", total, "fails", fails)
raise SystemExit(1 if fails else 0)
