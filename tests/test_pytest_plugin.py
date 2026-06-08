"""Tests for the korrel pytest plugin."""

import textwrap
from pathlib import Path

import pytest

# pytester is a built-in pytest plugin that is not activated by default.
pytest_plugins = ["pytester"]


PASSING_SCENARIO = textwrap.dedent("""\
    from korrel import Scenario, Rubric
    from korrel.persona import Persona
    from korrel.types import Message

    def always_one(completion, info, **kwargs):
        return 1.0

    scenario = Scenario(
        id="plugin_pass",
        system="test",
        persona=Persona(goal="g", behavior="b"),
        opening_message="hi",
        rubric=Rubric(funcs=[always_one], pass_threshold=0.5),
    )

    class _Adapter:
        def __call__(self, messages, tools):
            return Message(role="assistant", content="ok")

    adapter = _Adapter()
""")

FAILING_SCENARIO = textwrap.dedent("""\
    from korrel import Scenario, Rubric
    from korrel.persona import Persona
    from korrel.types import Message

    def always_zero(completion, info, **kwargs):
        return 0.0

    scenario = Scenario(
        id="plugin_fail",
        system="test",
        persona=Persona(goal="g", behavior="b"),
        opening_message="hi",
        rubric=Rubric(funcs=[always_zero], pass_threshold=0.5),
    )

    class _Adapter:
        def __call__(self, messages, tools):
            return Message(role="assistant", content="nope")

    adapter = _Adapter()
""")

NO_ADAPTER_SCENARIO = textwrap.dedent("""\
    from korrel import Scenario
    from korrel.persona import Persona

    scenario = Scenario(
        id="no_adapter",
        system="test",
        persona=Persona(goal="g", behavior="b"),
        opening_message="hi",
    )
""")


def test_plugin_collects_passing_scenario(pytester: pytest.Pytester):
    pytester.makefile(".py", plugin_pass_scenario=PASSING_SCENARIO)
    result = pytester.runpytest("--korrel-glob", "*_scenario.py", "-v")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*plugin_pass*PASSED*"])


def test_plugin_collects_failing_scenario(pytester: pytest.Pytester):
    pytester.makefile(".py", plugin_fail_scenario=FAILING_SCENARIO)
    result = pytester.runpytest("--korrel-glob", "*_scenario.py", "-v")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*plugin_fail*FAILED*"])


def test_plugin_failure_output_contains_score(pytester: pytest.Pytester):
    pytester.makefile(".py", plugin_fail_scenario=FAILING_SCENARIO)
    result = pytester.runpytest("--korrel-glob", "*_scenario.py", "-v")
    # repr_failure should show score and threshold.
    result.stdout.fnmatch_lines(["*score*"])
    result.stdout.fnmatch_lines(["*threshold*"])


def test_plugin_failure_shows_failed_functions(pytester: pytest.Pytester):
    pytester.makefile(".py", plugin_fail_scenario=FAILING_SCENARIO)
    result = pytester.runpytest("--korrel-glob", "*_scenario.py", "-v")
    result.stdout.fnmatch_lines(["*always_zero*"])


def test_plugin_no_adapter_raises_usage_error(pytester: pytest.Pytester):
    pytester.makefile(".py", no_adapter_scenario=NO_ADAPTER_SCENARIO)
    result = pytester.runpytest("--korrel-glob", "*_scenario.py", "-v")
    # Item is collected but runtest fails with UsageError -> ERROR.
    assert result.ret != 0


def test_plugin_does_not_collect_test_files(pytester: pytest.Pytester):
    # A plain test_*.py file should not be collected by the korrel plugin.
    pytester.makepyfile(
        test_plain="""\
def test_something():
    assert 1 + 1 == 2
"""
    )
    pytester.makefile(".py", plugin_pass_scenario=PASSING_SCENARIO)
    result = pytester.runpytest("--korrel-glob", "*_scenario.py", "-v")
    # The plain test runs via normal pytest collection; the scenario is also collected.
    result.assert_outcomes(passed=2)


def test_plugin_korrel_glob_ini_option(pytester: pytest.Pytester):
    pytester.makeini("[pytest]\nkorrel_glob = *_scene.py\n")
    pytester.makefile(".py", my_scene=PASSING_SCENARIO)
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)


def test_plugin_help_shows_korrel_glob(pytester: pytest.Pytester):
    result = pytester.runpytest("--help")
    result.stdout.fnmatch_lines(["*--korrel-glob*"])


def test_plugin_passing_writes_transcript(pytester: pytest.Pytester, tmp_path: Path):
    pytester.makefile(".py", plugin_pass_scenario=PASSING_SCENARIO)
    pytester.runpytest("--korrel-glob", "*_scenario.py", "-v")
    # Transcript is written to .korrel/ relative to the CWD where pytest ran.
    korrel_dir = Path(pytester.path) / ".korrel"
    transcripts = list(korrel_dir.glob("*.transcript.json")) if korrel_dir.exists() else []
    assert len(transcripts) >= 1
