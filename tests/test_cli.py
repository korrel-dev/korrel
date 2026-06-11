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


@pytest.mark.parametrize(
    "malicious_id",
    [
        "../../etc/crontab",
        "../escape",
        "a/b/c",
        "nested\\windows\\path",
        "/",
        "\\",
    ],
)
def test_write_transcript_for_id_cannot_escape_out_dir(tmp_path, malicious_id):
    # Security review K9: an author-controlled scenario id must not let the
    # transcript file escape out_dir, on either POSIX or Windows. The output
    # filename is platform-dependent (backslash is a separator on Windows but a
    # valid filename character on POSIX), so the stable invariant is that the
    # parent stays out_dir and a file is produced.
    result = _fake_run_result()
    out = tmp_path / "out"
    path = write_transcript_for(result, malicious_id, out)
    assert path.parent == out
    assert path.exists()


@pytest.mark.parametrize("degenerate_id", ["", ".", ".."])
def test_write_transcript_for_degenerate_id_falls_back(tmp_path, degenerate_id):
    # An id that reduces to no usable filename component on every platform
    # (Path(...).name == "") falls back to a default inside out_dir rather than
    # writing out_dir itself.
    result = _fake_run_result()
    out = tmp_path / "out"
    path = write_transcript_for(result, degenerate_id, out)
    assert path.parent == out
    assert path.name == "scenario.transcript.json"
    assert path.exists()


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


# ---------------------------------------------------------------------------
# Additional gap coverage
# ---------------------------------------------------------------------------


NON_CALLABLE_ADAPTER_MODULE = """\
from korrel import Scenario
from korrel.persona import Persona

scenario = Scenario(
    id="bad_adapter",
    system="test",
    persona=Persona(goal="g", behavior="b"),
    opening_message="hello",
)
adapter = 42
"""


def test_load_scenario_and_adapter_non_callable_adapter(tmp_path):
    f = tmp_path / "bad_adapter_scenario.py"
    f.write_text(NON_CALLABLE_ADAPTER_MODULE, encoding="utf-8")
    mod = load_module_from_path(f)
    with pytest.raises(TypeError, match="adapter"):
        load_scenario_and_adapter(mod, "scenario", "adapter")


def test_cli_run_scenario_attr_override(tmp_path, capsys):
    # Write a module where the scenario lives under a non-default attribute name.
    src = """\
from korrel import Scenario, Rubric
from korrel.persona import Persona
from korrel.types import Message

def always_one(completion, info, **kwargs):
    return 1.0

my_scenario = Scenario(
    id="attr_override_test",
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
    f = tmp_path / "attr_scenario.py"
    f.write_text(src, encoding="utf-8")
    out_dir = tmp_path / "out"

    from korrel.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(f), "--out", str(out_dir), "--scenario-attr", "my_scenario"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "attr_override_test" in captured.out


def test_cli_run_adapter_attr_override(tmp_path, capsys):
    src = """\
from korrel import Scenario, Rubric
from korrel.persona import Persona
from korrel.types import Message

def always_one(completion, info, **kwargs):
    return 1.0

scenario = Scenario(
    id="adapter_attr_override",
    system="test",
    persona=Persona(goal="g", behavior="b"),
    opening_message="hello",
    rubric=Rubric(funcs=[always_one], pass_threshold=0.5),
)

class _FakeAdapter:
    def __call__(self, messages, tools):
        return Message(role="assistant", content="done")

my_adapter = _FakeAdapter()
"""
    f = tmp_path / "adapter_attr_scenario.py"
    f.write_text(src, encoding="utf-8")
    out_dir = tmp_path / "out"

    from korrel.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(f), "--out", str(out_dir), "--adapter-attr", "my_adapter"])
    assert exc_info.value.code == 0


def test_cli_run_wrong_scenario_type_exits_nonzero(tmp_path, capsys):
    f = tmp_path / "wrong_type_scenario.py"
    f.write_text(WRONG_TYPE_MODULE, encoding="utf-8")

    from korrel.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(f), "--out", str(tmp_path / "out")])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


def test_cli_run_non_callable_adapter_exits_nonzero(tmp_path, capsys):
    f = tmp_path / "noncallable_scenario.py"
    f.write_text(NON_CALLABLE_ADAPTER_MODULE, encoding="utf-8")

    from korrel.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(f), "--out", str(tmp_path / "out")])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


def test_cli_run_fail_output_shows_failed_functions(tmp_path, capsys):
    f = tmp_path / "fail_scenario.py"
    f.write_text(FAILING_MODULE, encoding="utf-8")
    out_dir = tmp_path / "out"

    from korrel.cli import main
    with pytest.raises(SystemExit):
        main(["run", str(f), "--out", str(out_dir)])
    captured = capsys.readouterr()
    # The failing function name must appear in the output.
    assert "always_zero" in captured.out


def test_cli_run_transcript_file_is_valid_json(tmp_path):
    f = tmp_path / "pass_scenario.py"
    f.write_text(PASSING_MODULE, encoding="utf-8")
    out_dir = tmp_path / "transcripts"

    from korrel.cli import main
    with pytest.raises(SystemExit):
        main(["run", str(f), "--out", str(out_dir)])

    transcript_file = out_dir / "cli_pass_test.transcript.json"
    assert transcript_file.exists()
    data = json.loads(transcript_file.read_text(encoding="utf-8"))
    assert "messages" in data
    assert "turns" in data
    assert isinstance(data["messages"], list)


def test_cli_run_score_printed(tmp_path, capsys):
    f = tmp_path / "pass_scenario.py"
    f.write_text(PASSING_MODULE, encoding="utf-8")
    out_dir = tmp_path / "out"

    from korrel.cli import main
    with pytest.raises(SystemExit):
        main(["run", str(f), "--out", str(out_dir)])
    captured = capsys.readouterr()
    assert "score" in captured.out


def test_cli_no_subcommand_exits_zero(capsys):
    from korrel.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 0


def test_cli_run_transcript_path_printed(tmp_path, capsys):
    f = tmp_path / "pass_scenario.py"
    f.write_text(PASSING_MODULE, encoding="utf-8")
    out_dir = tmp_path / "out"

    from korrel.cli import main
    with pytest.raises(SystemExit):
        main(["run", str(f), "--out", str(out_dir)])
    captured = capsys.readouterr()
    assert "transcript" in captured.out


# ---------------------------------------------------------------------------
# Clean error surfaces: missing key, raising tool, generic failure
# ---------------------------------------------------------------------------


MISSING_KEY_MODULE = """\
from korrel import Scenario
from korrel.persona import Persona
from korrel.providers import MissingAPIKeyError

scenario = Scenario(
    id="missing_key_test",
    system="test",
    persona=Persona(goal="g", behavior="b"),
    opening_message="hello",
)

class _KeylessAdapter:
    def __call__(self, messages, tools):
        raise MissingAPIKeyError(
            "No API key found in ANTHROPIC_API_KEY. Korrel reads provider "
            "keys from the environment at call time and stores none. Set the "
            "variable or pass an instantiated client."
        )

adapter = _KeylessAdapter()
"""

RAISING_TOOL_MODULE = """\
from korrel import Message, MockTool, Scenario, ToolCall, ToolFunction
from korrel.persona import Persona

def explode(args, state):
    raise ValueError("boom from the mock tool")

scenario = Scenario(
    id="cli_tool_error_test",
    system="test",
    persona=Persona(goal="g", behavior="b"),
    opening_message="hello",
    tools=[MockTool(
        name="exploder",
        schema={
            "type": "function",
            "function": {
                "name": "exploder",
                "description": "Always raises.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        respond=explode,
    )],
)

class _Adapter:
    def __call__(self, messages, tools):
        return Message(
            role="assistant",
            tool_calls=[ToolCall(
                id="c1",
                function=ToolFunction(name="exploder", arguments="{}"),
            )],
        )

adapter = _Adapter()
"""

GENERIC_FAILURE_MODULE = """\
from korrel import Scenario
from korrel.persona import Persona

scenario = Scenario(
    id="generic_failure_test",
    system="test",
    persona=Persona(goal="g", behavior="b"),
    opening_message="hello",
)

class _BrokenAdapter:
    def __call__(self, messages, tools):
        raise ConnectionError("simulated transport failure")

adapter = _BrokenAdapter()
"""


def test_cli_run_missing_key_prints_single_error_line(tmp_path, capsys):
    f = tmp_path / "missing_key_scenario.py"
    f.write_text(MISSING_KEY_MODULE, encoding="utf-8")

    from korrel.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(f), "--out", str(tmp_path / "out")])
    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    error_lines = [l for l in captured.err.splitlines() if l.startswith("error:")]
    assert len(error_lines) == 1
    assert "ANTHROPIC_API_KEY" in error_lines[0]
    assert "Traceback" not in captured.err


def test_cli_run_raising_tool_prints_error_and_writes_transcript(tmp_path, capsys):
    f = tmp_path / "tool_error_scenario.py"
    f.write_text(RAISING_TOOL_MODULE, encoding="utf-8")
    out_dir = tmp_path / "out"

    from korrel.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(f), "--out", str(out_dir)])
    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    error_lines = [l for l in captured.err.splitlines() if l.startswith("error:")]
    assert len(error_lines) == 1
    assert "tool 'exploder' raised" in error_lines[0]
    assert "boom from the mock tool" in error_lines[0]
    assert "Traceback" not in captured.err

    transcript_file = out_dir / "cli_tool_error_test.transcript.json"
    assert transcript_file.exists()
    data = json.loads(transcript_file.read_text(encoding="utf-8"))
    roles = [m["role"] for m in data["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_cli_run_generic_failure_prints_single_error_line(tmp_path, capsys):
    f = tmp_path / "generic_failure_scenario.py"
    f.write_text(GENERIC_FAILURE_MODULE, encoding="utf-8")

    from korrel.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(f), "--out", str(tmp_path / "out")])
    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    error_lines = [l for l in captured.err.splitlines() if l.startswith("error:")]
    assert len(error_lines) == 1
    assert "run failed: ConnectionError: simulated transport failure" in error_lines[0]
    assert "Traceback" not in captured.err


def test_missing_api_key_error_is_runtime_error_subclass():
    from korrel.providers import MissingAPIKeyError

    assert issubclass(MissingAPIKeyError, RuntimeError)


def test_provider_get_client_raises_missing_api_key_error(monkeypatch):
    from korrel.providers import AnthropicProvider, MissingAPIKeyError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = AnthropicProvider()
    with pytest.raises(MissingAPIKeyError) as exc_info:
        provider._get_client()
    # The message instructs the user and never echoes a key value.
    assert "ANTHROPIC_API_KEY" in str(exc_info.value)


@pytest.mark.parametrize("scenario_attr,adapter_attr,expected_exit", [
    ("missing_scenario", "adapter", 1),
    ("scenario", "missing_adapter", 1),
])
def test_cli_run_missing_attribute_exits_nonzero(tmp_path, capsys, scenario_attr, adapter_attr, expected_exit):
    src = """\
from korrel import Scenario, Rubric
from korrel.persona import Persona
from korrel.types import Message

def always_one(completion, info, **kwargs):
    return 1.0

scenario = Scenario(
    id="attr_test",
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
    f = tmp_path / "attr_test_scenario.py"
    f.write_text(src, encoding="utf-8")

    from korrel.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(f), "--out", str(tmp_path / "out"),
              "--scenario-attr", scenario_attr,
              "--adapter-attr", adapter_attr])
    assert exc_info.value.code == expected_exit
