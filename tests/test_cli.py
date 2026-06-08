"""Tests for the CLI module and shared helpers."""

import json
import sys
from pathlib import Path

import pytest

from korrel import adapter_from_provider
from korrel.cli import (
    load_module_from_path,
    load_scenario_and_adapter,
    write_transcript_for,
)
from korrel.runtime import RunResult, Transcript, FailureCluster
from korrel.rubric import RubricResult


# ---------------------------------------------------------------------------
# Fixtures: mini scenario modules written to tmp_path
# ---------------------------------------------------------------------------


PASSING_MODULE = """\
from korrel import Scenario, Rubric
from korrel.persona import Persona
from korrel.types import Message

def always_one(completion, info, **kwargs):
    return 1.0

scenario = Scenario(
    id="cli_pass_test",
    system="test",
    persona=Persona(goal="g", behavior="b"),
    opening_message="hello",
    rubric=Rubric(funcs=[always_one], pass_threshold=0.5),
)

class _FakeAdapter:
    def __call__(self, messages, tools):
        return Message(role="assistant", content="done")

adapter = _FakeAdapter()
"""

FAILING_MODULE = """\
from korrel import Scenario, Rubric
from korrel.persona import Persona
from korrel.types import Message

def always_zero(completion, info, **kwargs):
    return 0.0

scenario = Scenario(
    id="cli_fail_test",
    system="test",
    persona=Persona(goal="g", behavior="b"),
    opening_message="hello",
    rubric=Rubric(funcs=[always_zero], pass_threshold=0.5),
)

class _FakeAdapter:
    def __call__(self, messages, tools):
        return Message(role="assistant", content="nope")

adapter = _FakeAdapter()
"""

NO_ADAPTER_MODULE = """\
from korrel import Scenario
from korrel.persona import Persona

scenario = Scenario(
    id="no_adapter",
    system="test",
    persona=Persona(goal="g", behavior="b"),
    opening_message="hello",
)
"""

WRONG_TYPE_MODULE = """\
scenario = "not a Scenario"
adapter = lambda m, t: None
"""


# ---------------------------------------------------------------------------
# load_module_from_path
# ---------------------------------------------------------------------------


def test_load_module_from_path_loads_attributes(tmp_path):
    f = tmp_path / "my_mod.py"
    f.write_text("X = 42\n", encoding="utf-8")
    mod = load_module_from_path(f)
    assert mod.X == 42


def test_load_module_from_path_missing_file_raises(tmp_path):
    with pytest.raises(Exception):
        load_module_from_path(tmp_path / "does_not_exist.py")


# ---------------------------------------------------------------------------
# load_scenario_and_adapter
# ---------------------------------------------------------------------------


def test_load_scenario_and_adapter_ok(tmp_path):
    f = tmp_path / "pass_scenario.py"
    f.write_text(PASSING_MODULE, encoding="utf-8")
    mod = load_module_from_path(f)
    scenario, adapter = load_scenario_and_adapter(mod, "scenario", "adapter")
    from korrel import Scenario as S
    assert isinstance(scenario, S)
    assert callable(adapter)


def test_load_scenario_and_adapter_missing_scenario(tmp_path):
    f = tmp_path / "no_scenario.py"
    f.write_text("adapter = lambda m, t: None\n", encoding="utf-8")
    mod = load_module_from_path(f)
    with pytest.raises(AttributeError, match="scenario"):
        load_scenario_and_adapter(mod, "scenario", "adapter")


def test_load_scenario_and_adapter_missing_adapter(tmp_path):
    f = tmp_path / "no_adapter_scenario.py"
    f.write_text(NO_ADAPTER_MODULE, encoding="utf-8")
    mod = load_module_from_path(f)
    with pytest.raises(AttributeError, match="adapter"):
        load_scenario_and_adapter(mod, "scenario", "adapter")


def test_load_scenario_and_adapter_wrong_scenario_type(tmp_path):
    f = tmp_path / "wrong_type_scenario.py"
    f.write_text(WRONG_TYPE_MODULE, encoding="utf-8")
    mod = load_module_from_path(f)
    with pytest.raises(TypeError, match="Scenario"):
        load_scenario_and_adapter(mod, "scenario", "adapter")


# ---------------------------------------------------------------------------
# write_transcript_for
# ---------------------------------------------------------------------------


def _fake_run_result() -> RunResult:
    from korrel.types import Message as Msg
    transcript = Transcript(messages=[Msg(role="user", content="hi")], turns=[], seed=0)
    rubric_result = RubricResult(score=1.0, passed=True, scores={}, failed_functions=[])
    return RunResult(
        transcript=transcript,
        score=1.0,
        passed=True,
        failed_functions=[],
        clusters=[],
        rubric_result=rubric_result,
    )


def test_write_transcript_for_creates_file(tmp_path):
    result = _fake_run_result()
    path = write_transcript_for(result, "my_scenario", tmp_path)
    assert path.exists()
    assert path.name == "my_scenario.transcript.json"


def test_write_transcript_for_creates_out_dir(tmp_path):
    result = _fake_run_result()
    out = tmp_path / "nested" / "output"
    path = write_transcript_for(result, "s1", out)
    assert out.exists()
    assert path.exists()


def test_write_transcript_for_valid_json(tmp_path):
    result = _fake_run_result()
    path = write_transcript_for(result, "s1", tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "messages" in data
    assert "turns" in data


# ---------------------------------------------------------------------------
# CLI main() integration
# ---------------------------------------------------------------------------


def test_cli_run_pass_exits_zero(tmp_path, capsys):
    f = tmp_path / "pass_scenario.py"
    f.write_text(PASSING_MODULE, encoding="utf-8")
    out_dir = tmp_path / "out"

    from korrel.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(f), "--out", str(out_dir)])
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert "pass" in captured.out
    assert "cli_pass_test" in captured.out
    assert "transcript" in captured.out


def test_cli_run_fail_exits_nonzero(tmp_path, capsys):
    f = tmp_path / "fail_scenario.py"
    f.write_text(FAILING_MODULE, encoding="utf-8")
    out_dir = tmp_path / "out"

    from korrel.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(f), "--out", str(out_dir)])
    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "fail" in captured.out


def test_cli_run_writes_transcript(tmp_path):
    f = tmp_path / "pass_scenario.py"
    f.write_text(PASSING_MODULE, encoding="utf-8")
    out_dir = tmp_path / "transcripts"

    from korrel.cli import main
    with pytest.raises(SystemExit):
        main(["run", str(f), "--out", str(out_dir)])

    transcript_file = out_dir / "cli_pass_test.transcript.json"
    assert transcript_file.exists()


def test_cli_run_missing_file_exits_nonzero(tmp_path, capsys):
    from korrel.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(tmp_path / "nope.py")])
    assert exc_info.value.code == 1


def test_cli_run_no_adapter_exits_nonzero(tmp_path, capsys):
    f = tmp_path / "no_adapter_scenario.py"
    f.write_text(NO_ADAPTER_MODULE, encoding="utf-8")

    from korrel.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(f), "--out", str(tmp_path / "out")])
    assert exc_info.value.code == 1


def test_cli_run_with_seed_override(tmp_path):
    f = tmp_path / "pass_scenario.py"
    f.write_text(PASSING_MODULE, encoding="utf-8")
    out_dir = tmp_path / "out"

    from korrel.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(f), "--out", str(out_dir), "--seed", "42"])
    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# adapter_from_provider
# ---------------------------------------------------------------------------


def test_adapter_from_provider_wraps_callable():
    from korrel.types import Message as Msg

    class FakeProvider:
        def complete(self, messages, *, system=None, tools=None, **kwargs):
            return Msg(role="assistant", content="wrapped")

    adapter = adapter_from_provider(FakeProvider())
    result = adapter([Msg(role="user", content="hi")], [])
    assert result.content == "wrapped"
    assert result.role == "assistant"


def test_adapter_from_provider_exported_from_init():
    import korrel
    assert hasattr(korrel, "adapter_from_provider")
    assert callable(korrel.adapter_from_provider)
