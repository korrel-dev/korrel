"""Regression guard: the OpenEnv exporter emits a pip-installable package even
when the scenario id is not a valid Python identifier. A hyphenated id is
sanitized to the module form for every Python-path position (packages,
package-dir, the project.scripts entry point, the README import); the
hyphenated form is reserved for the PyPI distribution name.
"""

import ast
from pathlib import Path

from korrel import MockTool, Persona, Scenario
from korrel.exporters.openenv import write_openenv_env


def _hyphenated_scenario() -> Scenario:
    return Scenario(
        id="support-refund",
        system="You are a support agent.",
        persona=Persona(goal="Get a refund.", behavior="Polite."),
        opening_message="I want a refund.",
        tools=[
            MockTool(
                name="lookup_order",
                schema={"type": "function", "function": {"name": "lookup_order"}},
                respond=lambda arguments, state: {"ok": True},
            )
        ],
        max_turns=3,
    )


def test_openenv_export_sanitizes_hyphenated_id(tmp_path: Path) -> None:
    out = write_openenv_env(_hyphenated_scenario(), tmp_path / "env")
    pyproject = (out / "pyproject.toml").read_text(encoding="utf-8")
    assert 'packages = ["support_refund", "support_refund.server"]' in pyproject
    assert 'package-dir = { "support_refund" = ".", "support_refund.server" = "server" }' in pyproject
    assert 'server = "support_refund.server.app:main"' in pyproject
    assert '"support-refund"' not in pyproject
    assert "support-refund.server" not in pyproject
    assert 'name = "openenv-support-refund"' in pyproject
    readme = (out / "README.md").read_text(encoding="utf-8")
    python_block = readme.split("```python", 1)[1].split("```", 1)[0]
    ast.parse(python_block)
