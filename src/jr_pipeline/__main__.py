"""``python -m jr_pipeline`` launches the Junior CLI.

The CLI itself lives with the other operator surfaces in ``apps_and_interfaces/``,
outside this package; this shim keeps every documented invocation working (the
SLURM deployment scripts and the run docs all say ``python -m jr_pipeline <command>``).
An installed checkout (``pip install -e .``) makes both importable; with a bare
PYTHONPATH, use ``PYTHONPATH=src:.`` from the repo root.
"""
try:
    from apps_and_interfaces.command_line_interface import main
except ImportError as missing_interface:
    raise SystemExit(
        "The Junior CLI lives in apps_and_interfaces/, which is not importable here.\n"
        "From the repo root:  pip install -e .   (or run with PYTHONPATH=src:. )\n"
        f"(import error: {missing_interface})"
    ) from missing_interface

if __name__ == "__main__":
    main()
