"""Offline validation harness for the korrel scenarios gallery.

Run by the ``gallery`` CI job and reproducible locally:

    uv run --extra verifiers --extra openenv python gallery/_ci_check.py

For every entry under ``gallery/`` (excluding files that start with ``_`` and
``TEMPLATE.py``) it:

- runs the entry with ``korrel run`` and asserts the documented outcome
  (``pass`` -> exit 0; ``tool_error`` -> exit non-zero with a named tool-failure
  line on stderr),
- for every export target marked in ``gallery/index.json``, runs
  ``korrel export`` into a temp dir, asserts it succeeds, and loads the
  generated package to confirm it constructs offline.

No ANTHROPIC_API_KEY is set and no network is needed. Contributed entries run
here too, so this harness stays strictly offline and secret-free.

Per-entry expectations come from ``gallery/index.json``. An entry not listed
there defaults to ``{"run": "pass", "export": []}``: it must exit 0 offline and
is not export-checked.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

GALLERY_DIR = Path(__file__).resolve().parent
INDEX_PATH = GALLERY_DIR / "index.json"


def _discover_entries() -> list[Path]:
    """Return the runnable gallery entry files, sorted by name."""
    entries = [
        p
        for p in sorted(GALLERY_DIR.glob("*.py"))
        if not p.name.startswith("_") and p.name != "TEMPLATE.py"
    ]
    return entries


def _load_index() -> dict[str, dict[str, Any]]:
    """Map each entry file name to its expectation dict."""
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    index: dict[str, dict[str, Any]] = {}
    for row in data.get("entries", []):
        if "file" not in row:
            raise AssertionError(f"index.json: entry row is missing 'file': {row!r}")
        index[row["file"]] = row
    return index


def _korrel(*args: str) -> subprocess.CompletedProcess:
    """Invoke the korrel CLI in a subprocess and capture its result."""
    return subprocess.run(
        [sys.executable, "-m", "korrel.cli", *args],
        capture_output=True,
        text=True,
    )


def _import_from_path(module_name: str, file_path: Path) -> Any:
    """Import a Python file as a uniquely named module and return it."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Run checks
# ---------------------------------------------------------------------------


def _check_run(entry: Path, expected: str) -> None:
    result = _korrel("run", str(entry))
    if expected == "pass":
        if result.returncode != 0:
            raise AssertionError(
                f"{entry.name}: expected exit 0, got {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
    elif expected == "tool_error":
        if result.returncode == 0:
            raise AssertionError(
                f"{entry.name}: expected a non-zero exit for a raising tool, got 0"
            )
        stderr = result.stderr
        if "error: tool" not in stderr or "raised" not in stderr:
            raise AssertionError(
                f"{entry.name}: expected a named tool-failure line on stderr, got:\n{stderr}"
            )
    else:
        raise AssertionError(f"{entry.name}: unknown run expectation {expected!r}")


# ---------------------------------------------------------------------------
# Export + load checks
# ---------------------------------------------------------------------------


def _load_verifiers_env(out_dir: Path) -> None:
    """Construct the exported verifiers environment to confirm it loads."""
    env_modules = [p for p in out_dir.glob("*.py") if p.name != "_scenario.py"]
    if len(env_modules) != 1:
        raise AssertionError(
            f"{out_dir.name}: expected exactly one verifiers env module, "
            f"found {[p.name for p in env_modules]}"
        )
    module = _import_from_path(f"_gallery_vf_{out_dir.name}", env_modules[0])
    env = module.load_environment()
    if env is None:
        raise AssertionError(f"{out_dir.name}: load_environment() returned None")


def _load_openenv_env(out_dir: Path) -> None:
    """Import the generated openenv package and run reset() to confirm it loads."""
    # The generated server module falls back to a top-level ``from models import``
    # when it is not imported as a package, so the package root must be on the
    # path and a module named ``models`` must resolve to THIS package's models.
    # Both are scoped to this call and undone in the finally block, so the bare
    # ``models`` name never lingers in sys.modules to shadow another package, and
    # sys.path does not accumulate a temp-dir entry per entry.
    out_dir_str = str(out_dir)
    sys.path.insert(0, out_dir_str)
    sys.modules.pop("models", None)
    try:
        _import_from_path("models", out_dir / "models.py")

        env_files = list((out_dir / "server").glob("*_environment.py"))
        if not env_files:
            raise AssertionError(f"no openenv environment module written in {out_dir}")
        env_module = _import_from_path(f"_gallery_oe_{out_dir.name}", env_files[0])

        environment = env_module.KorrelEnvironment()
        observation = environment.reset()
        if not getattr(observation, "messages", None):
            raise AssertionError(
                f"{out_dir.name}: openenv reset() returned no messages"
            )
    finally:
        sys.modules.pop("models", None)
        if out_dir_str in sys.path:
            sys.path.remove(out_dir_str)


_LOADERS = {
    "verifiers": _load_verifiers_env,
    "openenv": _load_openenv_env,
}


def _check_export(entry: Path, target: str, tmp_root: Path) -> None:
    out_dir = tmp_root / f"{entry.stem}_{target}"
    result = _korrel("export", str(entry), "--to", target, "--out", str(out_dir))
    if result.returncode != 0:
        raise AssertionError(
            f"{entry.name}: export to {target} failed (exit {result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    _LOADERS[target](out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    index = _load_index()
    entries = _discover_entries()
    if not entries:
        print("gallery: no entries found", file=sys.stderr)
        return 1

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for entry in entries:
            expected = index.get(entry.name, {"run": "pass", "export": []})
            run_expectation = expected.get("run", "pass")
            export_targets = expected.get("export", [])

            try:
                _check_run(entry, run_expectation)
                line = f"  run({run_expectation:<10}) ok"
                for target in export_targets:
                    _check_export(entry, target, tmp_root)
                    line += f"  export:{target} ok"
                print(f"{entry.name:<28}{line}")
            except AssertionError as exc:
                failures.append(str(exc))
                print(f"{entry.name:<28}  FAILED")

    print()
    if failures:
        print(f"gallery check FAILED ({len(failures)} entr{'y' if len(failures) == 1 else 'ies'}):")
        for f in failures:
            print(f"\n--- {f}")
        return 1

    print(f"gallery check passed: {len(entries)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
