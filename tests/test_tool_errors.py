"""Tests for tool-failure capture: ToolExecutionError and the plugin surface."""

import json
import textwrap
from pathlib import Path

import pytest

from conftest import ScriptedAdapter, ScriptedPersona

from korrel import (
    Message,
    MockTool,
    Scenario,
    ToolCall,
    ToolExecutionError,
    ToolFunction,
    run_scenario,
)
from korrel.persona import Persona

pytest_plugins = ["pytester"]


def _raising_tool() -> MockTool:
    def explode(args, state):
        raise ValueError("boom from the mock tool")

    return MockTool(
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
    )


def _tool_call_message() -> Message:
    return Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="c1",
                function=ToolFunction(name="exploder", arguments=json.dumps({})),
            )
        ],
    )


def _scenario() -> Scenario:
    return Scenario(
        id="tool_error_test",
        system="test system",
        persona=Persona(goal="g", behavior="b"),
        opening_message="hello",
        tools=[_raising_tool()],
        max_turns=2,
    )


def test_raising_tool_raises_tool_execution_error():
    adapter = ScriptedAdapter([_tool_call_message()])
    persona = ScriptedPersona([])

    with pytest.raises(ToolExecutionError) as exc_info:
        run_scenario(_scenario(), adapter, persona=persona, seed=1)

    exc = exc_info.value
    assert exc.tool_name == "exploder"
    assert isinstance(exc.original, ValueError)
    assert "boom from the mock tool" in str(exc)


def test_tool_execution_error_carries_partial_transcript():
    adapter = ScriptedAdapter([_tool_call_message()])
    persona = ScriptedPersona([])

    with pytest.raises(ToolExecutionError) as exc_info:
        run_scenario(_scenario(), adapter, persona=persona, seed=3)

    transcript = exc_info.value.transcript
    roles = [m.role for m in transcript.messages]
    # System + opening user message + the assistant tool call that triggered
    # the failure. No tool message: the tool raised instead of responding.
    assert roles == ["system", "user", "assistant"]
    assert transcript.messages[-1].tool_calls is not None
    assert transcript.messages[-1].tool_calls[0].function.name == "exploder"
    assert transcript.seed == 3
    # The in-progress turn is included so the failing call is visible.
    assert len(transcript.turns) == 1
    assert transcript.turns[0].index == 0


def test_tool_error_on_second_turn_keeps_prior_turns():
    # First turn completes normally; the tool raises on the second turn.
    adapter = ScriptedAdapter(
        [
            Message(role="assistant", content="turn one answer"),
            _tool_call_message(),
        ]
    )
    persona = ScriptedPersona(["follow up"])

    with pytest.raises(ToolExecutionError) as exc_info:
        run_scenario(_scenario(), adapter, persona=persona, seed=1)

    transcript = exc_info.value.transcript
    assert len(transcript.turns) == 2
    assert transcript.turns[0].messages[-1].content == "turn one answer"


RAISING_SCENARIO = textwrap.dedent("""\
    import json
    from korrel import Message, MockTool, Rubric, Scenario, ToolCall, ToolFunction
    from korrel.persona import Persona

    def explode(args, state):
        raise ValueError("boom from the mock tool")

    def always_one(completion, info, **kwargs):
        return 1.0

    scenario = Scenario(
        id="plugin_tool_error",
        system="test",
        persona=Persona(goal="g", behavior="b"),
        opening_message="hi",
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
        rubric=Rubric(funcs=[always_one], pass_threshold=0.5),
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
""")


def test_plugin_reports_clean_tool_failure(pytester: pytest.Pytester):
    pytester.makefile(".py", plugin_tool_error_scenario=RAISING_SCENARIO)
    result = pytester.runpytest("--korrel-glob", "*_scenario.py", "-v")
    result.assert_outcomes(failed=1)
    # The failure block names the tool and the transcript path, with no
    # raw traceback dump.
    result.stdout.fnmatch_lines(["*exploder*"])
    result.stdout.fnmatch_lines(["*transcript*"])
    assert "Traceback (most recent call last)" not in result.stdout.str()


def test_plugin_writes_partial_transcript_on_tool_failure(pytester: pytest.Pytester):
    pytester.makefile(".py", plugin_tool_error_scenario=RAISING_SCENARIO)
    pytester.runpytest("--korrel-glob", "*_scenario.py", "-v")
    transcript_file = (
        Path(pytester.path) / ".korrel" / "plugin_tool_error.transcript.json"
    )
    assert transcript_file.exists()
    data = json.loads(transcript_file.read_text(encoding="utf-8"))
    roles = [m["role"] for m in data["messages"]]
    assert roles == ["system", "user", "assistant"]
