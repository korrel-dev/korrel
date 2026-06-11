"""Make-free front door for the fidelity benchmark.

Reproduces ``make benchmark`` exactly, step for step, without requiring
``make``: sync the benchmarks uv project, run the selftest, recompute labels
from the frozen transcripts, then run the fidelity harness over every frozen
transcript. Each step is the same ``uv run --project .`` invocation the
Makefile issues, so the two front doors produce identical
``results/results.json`` output.

Usage (offline, no keys, any platform):

    cd benchmarks
    python run.py              # full benchmark: sync -> selftest -> labels -> fidelity
    python run.py --selftest   # sync -> selftest only (no frozen transcripts needed)

Prerequisites: uv on PATH (or set the UV environment variable to the binary),
matching the Makefile's UV variable. On Windows run
``git config --global core.longpaths true`` first; tau2's checkout has paths
exceeding MAX_PATH. This script uses only the standard library; the benchmark
itself runs inside the uv-managed Python 3.13 environment.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent

# Mirrors the Makefile's `UV ?= uv`.
UV = os.environ.get("UV", "uv")


def _step(label: str, args: list[str]) -> None:
    """Echo and run one benchmark step from benchmarks/; exit on failure."""
    print(f"--- {label} ---", flush=True)
    result = subprocess.run([UV, *args], cwd=BENCHMARKS_DIR)
    if result.returncode != 0:
        print(
            f"error: step failed with exit code {result.returncode}: "
            f"{UV} {' '.join(args)}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Make-free fidelity benchmark runner. Reproduces `make benchmark` "
            "(sync, selftest, labels, fidelity) offline with no keys."
        ),
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help=(
            "Run only the selftest against the committed synthetic fixtures "
            "(mirrors `make selftest`)."
        ),
    )
    args = parser.parse_args()

    # make sync
    _step("syncing the benchmarks environment", ["sync", "--project", "."])

    # make selftest
    _step(
        "running selftest (synthetic fixture, no frozen transcripts needed)",
        ["run", "--project", ".", "python", "fidelity/run_fidelity.py", "--selftest"],
    )
    if args.selftest:
        return

    # make labels
    _step(
        "computing labels from frozen transcripts",
        ["run", "--project", ".", "python", "regenerate/compute_labels.py"],
    )

    # make benchmark (the fidelity harness itself)
    _step(
        "running fidelity harness",
        ["run", "--project", ".", "python", "fidelity/run_fidelity.py"],
    )


if __name__ == "__main__":
    main()
