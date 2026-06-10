"""Pytest configuration for benchmarks/tests.

Sets TAU2_DATA_DIR before any test module imports tau2.  tau2.utils.utils reads
the env var at module load time, so the resolution must happen before the first
import tau2 statement in any test file.

All tests in this directory run offline (no network, no API keys).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the benchmarks directory is on sys.path so that fidelity.* imports work.
BENCHMARKS_DIR = Path(__file__).parent.parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

# Set TAU2_DATA_DIR before any tau2 import.
from fidelity._tau2_data import ensure_tau2_data_dir  # noqa: E402

ensure_tau2_data_dir()
