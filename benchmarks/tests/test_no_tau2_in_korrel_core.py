"""Guard test: tau2 must not leak into korrel core source.

korrel's core (src/korrel/) must be importable without tau2 installed.
tau2 is a benchmarks-only dependency; any import of tau2 in core would break
the root test suite and the published PyPI package.

This test:
1. Greps every .py file under src/korrel/ for the text "tau2" (import or
   reference).  A single hit fails the test.
2. Imports korrel in a subprocess with tau2 absent from sys.path to confirm
   the import does not raise (belt-and-suspenders).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

KORREL_SRC = Path(__file__).parent.parent.parent / "src" / "korrel"


# ---------------------------------------------------------------------------
# Static grep check
# ---------------------------------------------------------------------------


def _collect_tau2_references() -> list[tuple[Path, int, str]]:
    """Return (file, line_number, line_text) for every tau2 reference in korrel src."""
    hits: list[tuple[Path, int, str]] = []
    for py_file in sorted(KORREL_SRC.rglob("*.py")):
        for lineno, line in enumerate(
            py_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "tau2" in line:
                hits.append((py_file, lineno, line.strip()))
    return hits


def test_no_tau2_reference_in_korrel_src() -> None:
    """No file under src/korrel/ may reference tau2 in any way."""
    hits = _collect_tau2_references()
    if hits:
        report = "\n".join(
            f"  {f.relative_to(KORREL_SRC)}:{n}: {l}" for f, n, l in hits
        )
        pytest.fail(
            f"tau2 reference(s) found in korrel core source:\n{report}\n"
            "tau2 is a benchmarks-only dependency and must not be imported in core."
        )


# ---------------------------------------------------------------------------
# Runtime import check
# ---------------------------------------------------------------------------


def test_korrel_imports_without_tau2_on_path(tmp_path: Path) -> None:
    """Importing korrel must not trigger any import of tau2 at module load time.

    Writes a small helper script that blocks tau2 in sys.modules and then
    imports korrel core.  Success means korrel has no top-level tau2 dependency.
    """
    script = tmp_path / "_check_no_tau2.py"
    script.write_text(
        "import sys, types\n"
        "\n"
        "class _Blocker(types.ModuleType):\n"
        "    def __getattr__(self, name):\n"
        "        raise ImportError('tau2 blocked')\n"
        "\n"
        "sys.modules['tau2'] = _Blocker('tau2')\n"
        "\n"
        "import korrel\n"
        "import korrel.types\n"
        "import korrel.rubric\n"
        "import korrel.scenario\n"
        "import korrel.tools\n"
        "print('ok')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "korrel import touched tau2 at module load time.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "ok" in result.stdout


def test_korrel_fidelity_convert_imports_without_tau2_top_level() -> None:
    """_convert.py defers all tau2 imports; the module must be importable without tau2.

    This verifies the module docstring claim: 'All tau2 imports are deferred to
    keep this module importable in environments where tau2 is not installed.'
    """
    # Check the source statically: no top-level 'import tau2' or 'from tau2'
    convert_file = Path(__file__).parent.parent / "fidelity" / "_convert.py"
    source = convert_file.read_text(encoding="utf-8")
    lines = source.splitlines()
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        # Skip lines inside functions/classes (indented) -- those are deferred
        if stripped.startswith(("import tau2", "from tau2")) and not line.startswith(" "):
            pytest.fail(
                f"_convert.py line {lineno}: top-level tau2 import found: {line!r}\n"
                "All tau2 imports in _convert.py must be deferred (inside functions)."
            )


def test_korrel_scenario_retail_imports_without_tau2_top_level() -> None:
    """scenario_retail.py defers tau2 imports inside the reward function."""
    scenario_file = Path(__file__).parent.parent / "fidelity" / "scenario_retail.py"
    source = scenario_file.read_text(encoding="utf-8")
    lines = source.splitlines()
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith(("import tau2", "from tau2")) and not line.startswith(" "):
            pytest.fail(
                f"scenario_retail.py line {lineno}: top-level tau2 import: {line!r}\n"
                "tau2 imports must be deferred inside the reward function."
            )
