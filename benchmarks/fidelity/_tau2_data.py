"""Resolve tau2 data directory for the benchmarks environment.

tau2 ships its data under data/ in the repo. When installed from the git SHA
pin, the data lives in the uv git cache checkout, not in site-packages.

tau2 pin: v1.0.0 @ 17e07b1da2bbc0cadfddeea36412686e0604127b

Resolution order:
1. TAU2_DATA_DIR environment variable (explicit override).
2. Walk Python's sys.path to find tau2/utils/utils.py; walk up from there to
   find data/tau2/domains/retail/tasks.json (works when installed from source
   or TAU2_DATA_DIR already set by utils.py).
3. Search the uv git cache for the pinned commit SHA (cross-platform:
   %LOCALAPPDATA%/uv/cache on Windows, ~/.cache/uv on Linux/macOS).
4. Raises RuntimeError with a clear message if none of the above works.

IMPORTANT: call `ensure_tau2_data_dir()` BEFORE importing tau2. tau2.utils.utils
reads TAU2_DATA_DIR at module load time (not inside a function), so the env var
must be set before any `import tau2` statement.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Pinned commit SHA for tau2-bench v1.0.0.
# Verified on 2026-06-10 via gh api repos/sierra-research/tau2-bench/git/refs/tags/v1.0.0
TAU2_COMMIT_SHA = "17e07b1da2bbc0cadfddeea36412686e0604127b"
TAU2_SHORT_SHA = TAU2_COMMIT_SHA[:7]  # "17e07b1"


def _uv_cache_dirs() -> list[Path]:
    """Return candidate uv cache roots for the current platform."""
    candidates = []
    # Windows
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "uv" / "cache")
    # Linux / macOS
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        candidates.append(Path(xdg) / "uv")
    home = Path.home()
    candidates.append(home / ".cache" / "uv")
    candidates.append(home / "Library" / "Caches" / "uv")  # macOS legacy
    return candidates


def find_tau2_data_dir() -> Path:
    """Return the tau2 data/ directory path without importing tau2.

    Resolution order documented in module docstring.
    Raises RuntimeError if not found.
    """
    # 1. Explicit env var
    env_val = os.environ.get("TAU2_DATA_DIR")
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p
        raise RuntimeError(
            f"TAU2_DATA_DIR={env_val!r} is set but does not exist."
        )

    # 2. Walk up from tau2/utils/utils.py in sys.path
    tau2_utils_file: Path | None = None
    for sp in sys.path:
        c = Path(sp) / "tau2" / "utils" / "utils.py"
        if c.exists():
            tau2_utils_file = c
            break
    if tau2_utils_file is None:
        # Also check the local .venv
        benchmarks_dir = Path(__file__).parent.parent
        for venv_candidate in [
            benchmarks_dir / ".venv" / "Lib" / "site-packages",
            benchmarks_dir / ".venv" / "lib" / "python3.13" / "site-packages",
        ]:
            c = venv_candidate / "tau2" / "utils" / "utils.py"
            if c.exists():
                tau2_utils_file = c
                break

    if tau2_utils_file is not None:
        current = tau2_utils_file.parent
        for _ in range(12):
            candidate = current / "data"
            retail_check = candidate / "tau2" / "domains" / "retail" / "tasks.json"
            if retail_check.exists():
                return candidate
            current = current.parent

    # 3. Search uv git cache for the pinned SHA
    for cache_root in _uv_cache_dirs():
        git_checkouts = cache_root / "git-v0" / "checkouts"
        if not git_checkouts.exists():
            continue
        # Each repo gets a dir; inside are dirs named by short SHA
        for repo_dir in git_checkouts.iterdir():
            if not repo_dir.is_dir():
                continue
            for checkout_dir in repo_dir.iterdir():
                if not checkout_dir.is_dir():
                    continue
                if checkout_dir.name.startswith(TAU2_SHORT_SHA):
                    candidate = checkout_dir / "data"
                    retail_check = (
                        candidate / "tau2" / "domains" / "retail" / "tasks.json"
                    )
                    if retail_check.exists():
                        return candidate

    raise RuntimeError(
        "Could not locate tau2 data/ directory. "
        "Set TAU2_DATA_DIR to the path containing "
        "data/tau2/domains/retail/tasks.json. "
        f"Searched sys.path (tau2/utils/utils.py) and uv git cache "
        f"for commit prefix {TAU2_SHORT_SHA!r}."
    )


def ensure_tau2_data_dir() -> Path:
    """Find the data directory and set TAU2_DATA_DIR if not already set.

    Must be called BEFORE importing tau2 (tau2.utils.utils reads the env var
    at module load time).
    """
    data_dir = find_tau2_data_dir()
    if not os.environ.get("TAU2_DATA_DIR"):
        os.environ["TAU2_DATA_DIR"] = str(data_dir)
    return data_dir
