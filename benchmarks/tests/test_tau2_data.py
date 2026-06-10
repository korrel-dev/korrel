"""Tests for benchmarks/fidelity/_tau2_data.py.

Covers:
- TAU2_COMMIT_SHA and TAU2_SHORT_SHA constants
- _uv_cache_dirs: returns platform-appropriate candidates
- find_tau2_data_dir: TAU2_DATA_DIR env var paths (existing and nonexistent)
- ensure_tau2_data_dir: sets the env var exactly once
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from fidelity._tau2_data import (
    TAU2_COMMIT_SHA,
    TAU2_SHORT_SHA,
    _uv_cache_dirs,
    ensure_tau2_data_dir,
    find_tau2_data_dir,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_commit_sha_is_pinned_value() -> None:
    assert TAU2_COMMIT_SHA == "17e07b1da2bbc0cadfddeea36412686e0604127b"


def test_short_sha_is_prefix_of_full_sha() -> None:
    assert TAU2_COMMIT_SHA.startswith(TAU2_SHORT_SHA)


def test_short_sha_length() -> None:
    # Short SHA must be non-empty and shorter than the full SHA.
    assert 1 <= len(TAU2_SHORT_SHA) < len(TAU2_COMMIT_SHA)


# ---------------------------------------------------------------------------
# _uv_cache_dirs
# ---------------------------------------------------------------------------


def test_uv_cache_dirs_returns_list_of_paths() -> None:
    dirs = _uv_cache_dirs()
    assert isinstance(dirs, list)
    assert all(isinstance(d, Path) for d in dirs)


def test_uv_cache_dirs_nonempty() -> None:
    # At minimum the fallback ~/.cache/uv path is always included.
    dirs = _uv_cache_dirs()
    assert len(dirs) >= 1


def test_uv_cache_dirs_includes_localappdata_on_windows(tmp_path: Path) -> None:
    """When LOCALAPPDATA is set, the Windows path is in the list."""
    fake_local = str(tmp_path / "AppData" / "Local")
    with patch.dict(os.environ, {"LOCALAPPDATA": fake_local}):
        dirs = _uv_cache_dirs()
    expected = Path(fake_local) / "uv" / "cache"
    assert expected in dirs


def test_uv_cache_dirs_includes_xdg_when_set(tmp_path: Path) -> None:
    """When XDG_CACHE_HOME is set, the XDG path is included."""
    fake_xdg = str(tmp_path / "xdg-cache")
    with patch.dict(os.environ, {"XDG_CACHE_HOME": fake_xdg}):
        dirs = _uv_cache_dirs()
    expected = Path(fake_xdg) / "uv"
    assert expected in dirs


# ---------------------------------------------------------------------------
# find_tau2_data_dir: TAU2_DATA_DIR env var paths
# ---------------------------------------------------------------------------


def test_find_tau2_data_dir_raises_when_env_var_points_to_nonexistent_path(
    tmp_path: Path,
) -> None:
    """TAU2_DATA_DIR set to a path that does not exist -> RuntimeError."""
    nonexistent = str(tmp_path / "does_not_exist")
    with patch.dict(os.environ, {"TAU2_DATA_DIR": nonexistent}, clear=False):
        with pytest.raises(RuntimeError, match="TAU2_DATA_DIR"):
            find_tau2_data_dir()


def test_find_tau2_data_dir_returns_path_when_env_var_exists(tmp_path: Path) -> None:
    """TAU2_DATA_DIR pointing to an existing directory is returned as-is."""
    # Create the minimal structure that find_tau2_data_dir returns
    # (the retail check is only in the sys.path / cache branches,
    # not in the env-var branch which just tests existence).
    fake_data = tmp_path / "data"
    fake_data.mkdir()
    saved = os.environ.pop("TAU2_DATA_DIR", None)
    os.environ["TAU2_DATA_DIR"] = str(fake_data)
    try:
        result = find_tau2_data_dir()
        assert result == fake_data
    finally:
        if saved is not None:
            os.environ["TAU2_DATA_DIR"] = saved
        else:
            os.environ.pop("TAU2_DATA_DIR", None)


def test_find_tau2_data_dir_succeeds_in_current_environment() -> None:
    """find_tau2_data_dir must succeed in the benchmarks venv (data is present)."""
    result = find_tau2_data_dir()
    assert result.exists()
    retail_check = result / "tau2" / "domains" / "retail" / "tasks.json"
    assert retail_check.exists(), (
        f"tau2 retail tasks.json not found at {retail_check}"
    )


# ---------------------------------------------------------------------------
# ensure_tau2_data_dir
# ---------------------------------------------------------------------------


def test_ensure_tau2_data_dir_sets_env_var() -> None:
    """ensure_tau2_data_dir always leaves TAU2_DATA_DIR set to a valid path."""
    # Save and clear the existing value to test the setter path.
    saved = os.environ.pop("TAU2_DATA_DIR", None)
    try:
        result = ensure_tau2_data_dir()
        assert os.environ.get("TAU2_DATA_DIR") == str(result)
    finally:
        if saved is not None:
            os.environ["TAU2_DATA_DIR"] = saved
        else:
            # Leave it set since conftest already set it; restore the found path.
            ensure_tau2_data_dir()


def test_ensure_tau2_data_dir_does_not_overwrite_existing_env_var(
    tmp_path: Path,
) -> None:
    """If TAU2_DATA_DIR is already set to a valid path, ensure_tau2_data_dir
    does not overwrite it."""
    fake_data = tmp_path / "data"
    fake_data.mkdir()
    saved = os.environ.pop("TAU2_DATA_DIR", None)
    os.environ["TAU2_DATA_DIR"] = str(fake_data)
    try:
        result = ensure_tau2_data_dir()
        # The env var must still point to the pre-set fake path.
        assert os.environ["TAU2_DATA_DIR"] == str(fake_data)
        assert result == fake_data
    finally:
        if saved is not None:
            os.environ["TAU2_DATA_DIR"] = saved
        else:
            os.environ.pop("TAU2_DATA_DIR", None)
        # Restore valid env var for subsequent tests.
        ensure_tau2_data_dir()


def test_ensure_tau2_data_dir_returns_path_object() -> None:
    result = ensure_tau2_data_dir()
    assert isinstance(result, Path)
