"""CAI-Vision scripts package.

Each module in this package is a thin command-line entry point. The
domain logic (datasets, model construction, metrics, etc.) lives in the
top-level `cai` package; scripts only do argument parsing and
orchestration.

Run a script either way:
    python scripts/evaluate_model.py [args...]
    python -m scripts.evaluate_model [args...]

The `-m` form is recommended because it works regardless of the current
working directory.
"""
