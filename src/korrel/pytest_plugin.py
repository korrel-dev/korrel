"""korrel pytest plugin.

Registered via the ``pytest11`` entry-point group so no user-side
conftest is required. Discovers ``*_scenario.py`` files (or whatever
glob ``korrel_glob`` / ``--korrel-glob`` says), imports them, and runs
each scenario as a pytest item.

Each scenario file must expose a module-level ``scenario`` (a
``Scenario`` instance) and a module-level ``adapter`` (any callable
matching the ``AgentAdapter`` protocol). If ``adapter`` is absent, the
item is reported as an error immediately.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Optional

import pytest

from .cli import load_module_from_path, write_transcript_for
from .runtime import RunResult, run_scenario
from .scenario import Scenario


# ---------------------------------------------------------------------------
# Plugin hooks
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser, pluginmanager: Any) -> None:
    """Register the ``korrel_glob`` ini option and ``--korrel-glob`` flag."""
    parser.addini(
        "korrel_glob",
        default="*_scenario.py",
        help="Glob pattern selecting scenario modules for the korrel plugin.",
    )
    parser.addoption(
        "--korrel-glob",
        default=None,
        metavar="GLOB",
        help="Override the korrel_glob ini option.",
    )


def pytest_collect_file(
    file_path: Path,
    parent: pytest.Collector,
) -> Optional[pytest.Collector]:
    """Return a KorrelFile collector when file_path matches the glob."""
    config = parent.config
    glob: str = config.getoption("--korrel-glob", default=None)
    if glob is None:
        glob = config.getini("korrel_glob") or "*_scenario.py"

    if fnmatch.fnmatch(file_path.name, glob):
        return KorrelFile.from_parent(parent, path=file_path)
    return None


# ---------------------------------------------------------------------------
# Collector and item
# ---------------------------------------------------------------------------


class KorrelFile(pytest.File):
    """Collects all scenario+adapter pairs from a scenario module."""

    def collect(self):
        try:
            module = load_module_from_path(self.path)
        except Exception as exc:
            raise pytest.UsageError(
                f"korrel: cannot import {self.path}: {exc}"
            ) from exc

        # Yield one item per scenario+adapter found. In v0.1 a module
        # exposes one pair; the structure supports more later.
        scenario = getattr(module, "scenario", None)
        adapter = getattr(module, "adapter", None)

        if scenario is None:
            raise pytest.UsageError(
                f"korrel: {self.path} has no module-level 'scenario' attribute."
            )

        yield KorrelItem.from_parent(
            self,
            name=getattr(scenario, "id", str(self.path.stem)),
            scenario=scenario,
            adapter=adapter,
        )


class KorrelScenarioFailed(AssertionError):
    """Raised by KorrelItem.runtest() when a scenario does not pass."""

    def __init__(
        self,
        scenario_id: str,
        score: float,
        threshold: float,
        failed_functions: list[str],
        transcript_path: Path,
    ) -> None:
        self.scenario_id = scenario_id
        self.score = score
        self.threshold = threshold
        self.failed_functions = failed_functions
        self.transcript_path = transcript_path
        super().__init__(
            f"scenario {scenario_id!r} failed: "
            f"score={score:.4f} < threshold={threshold:.4f}"
        )


class KorrelItem(pytest.Item):
    """A single runnable scenario item."""

    def __init__(
        self,
        *,
        name: str,
        parent: KorrelFile,
        scenario: Scenario,
        adapter: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, parent=parent, **kwargs)
        self._scenario = scenario
        self._adapter = adapter
        self._result: Optional[RunResult] = None
        self._transcript_path: Optional[Path] = None

    def runtest(self) -> None:
        if self._adapter is None:
            raise pytest.UsageError(
                f"korrel: {self.fspath} has no module-level 'adapter' attribute. "
                "A scenario file in a CI gate must include an adapter."
            )
        if not callable(self._adapter):
            raise pytest.UsageError(
                f"korrel: 'adapter' in {self.fspath} is not callable "
                f"(got {type(self._adapter).__name__})."
            )

        result = run_scenario(self._scenario, self._adapter)
        self._result = result

        # Write transcript next to the test for debug access.
        out_dir = Path(".korrel")
        self._transcript_path = write_transcript_for(result, self._scenario.id, out_dir)

        if not result.passed:
            threshold = (
                self._scenario.rubric.pass_threshold
                if self._scenario.rubric is not None
                else 0.5
            )
            raise KorrelScenarioFailed(
                scenario_id=self._scenario.id,
                score=result.score,
                threshold=threshold,
                failed_functions=result.failed_functions,
                transcript_path=self._transcript_path,
            )

    def repr_failure(self, excinfo: Any, style: Any = None) -> str:
        exc = excinfo.value
        if isinstance(exc, KorrelScenarioFailed):
            lines = [
                f"{'scenario':<10}: {exc.scenario_id}",
                f"{'score':<10}: {exc.score:.4f}",
                f"{'threshold':<10}: {exc.threshold:.4f}",
                f"{'status':<10}: fail",
            ]
            if exc.failed_functions:
                lines.append(f"{'failed':<10}: {', '.join(exc.failed_functions)}")
            lines.append(f"{'transcript':<10}: {exc.transcript_path}")
            return "\n".join(lines)
        return super().repr_failure(excinfo)

    def reportinfo(self) -> tuple[Any, Optional[int], str]:
        return self.fspath, None, f"scenario: {self._scenario.id}"
