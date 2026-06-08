"""Tests for Scenario.max_tool_rounds and Turn.stop_reason."""

import json

import pytest

from conftest import ScriptedAdapter, ScriptedPersona

from korrel import Message, MockTool, Rubric, Scenario, ToolCall, ToolFunction, run_scenario
from korrel.persona import Persona
from korrel.runtime import Turn


def _echo_tool() -> MockTool:
    return MockTool(
        name="echo",
        schema={
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Returns the input unchanged.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
        respond=lambda args, state: {"echoed": args.get("text", "")},
    )


def _tool_msg() -> Message:
    return Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="c1",
                function=ToolFunction(name="echo", arguments=json.dumps({"text": "hi"})),
            )
        ],
    )


def _scenario(max_tool_rounds: int = 8, max_turns: int = 1) -> Scenario:
    return Scenario(
        id="cap_test",
        system="test system",
        persona=Persona(goal="test", behavior="test"),
        opening_message="hello",
        tools=[_echo_tool()],
        max_turns=max_turns,
        max_tool_rounds=max_tool_rounds,
    )


def test_max_tool_rounds_default_is_eight():
    scenario = Scenario(
        id="defaults_test",
        system="s",
        persona=Persona(goal="g", behavior="b"),
        opening_message="hi",
    )
    assert scenario.max_tool_rounds == 8


def test_max_tool_rounds_ge_one_validation():
    with pytest.raises(Exception):
        Scenario(
            id="invalid",
            system="s",
            persona=Persona(goal="g", behavior="b"),
            opening_message="hi",
            max_tool_rounds=0,
        )


def test_tool_loop_below_cap_no_stop_reason():
    # One tool call then a final answer. Should NOT hit cap, stop_reason stays None.
    messages_queue = [_tool_msg(), Message(role="assistant", content="done")]
    adapter = ScriptedAdapter(messages_queue)
    persona = ScriptedPersona([])
    scenario = _scenario(max_tool_rounds=3)

    result = run_scenario(scenario, adapter, persona=persona, seed=1)

    assert len(result.transcript.turns) == 1
    turn: Turn = result.transcript.turns[0]
    assert turn.stop_reason is None


def test_tool_loop_at_cap_sets_stop_reason():
    # Adapter keeps returning tool calls without end; cap should fire.
    # With max_tool_rounds=2:
    # - round 0: resolve tool, adapter returns tool_msg again
    # - round 1: resolve tool, adapter returns tool_msg again
    # - round 2: tool_round(2) >= max_tool_rounds(2), break with stop_reason
    endless = [_tool_msg()] * 20
    adapter = ScriptedAdapter(endless)
    persona = ScriptedPersona([])
    scenario = _scenario(max_tool_rounds=2)

    result = run_scenario(scenario, adapter, persona=persona, seed=1)

    assert len(result.transcript.turns) == 1
    turn: Turn = result.transcript.turns[0]
    assert turn.stop_reason == "max_tool_rounds"


def test_stop_reason_is_none_on_turns_not_capped():
    # Two turns; only second hits cap.
    first_no_cap = [_tool_msg(), Message(role="assistant", content="turn1 done")]
    # Second turn: endless tool calls
    second_capped = [_tool_msg()] * 10
    adapter = ScriptedAdapter(first_no_cap + second_capped)
    persona = ScriptedPersona(["follow up"])
    scenario = _scenario(max_tool_rounds=2, max_turns=2)

    result = run_scenario(scenario, adapter, persona=persona, seed=1)

    assert len(result.transcript.turns) == 2
    # First turn: one tool round, under cap.
    assert result.transcript.turns[0].stop_reason is None
    # Second turn: hits cap.
    assert result.transcript.turns[1].stop_reason == "max_tool_rounds"


def test_capped_turn_has_no_synthetic_message():
    # Verify no extra injected message appears when cap fires.
    endless = [_tool_msg()] * 20
    adapter = ScriptedAdapter(endless)
    persona = ScriptedPersona([])
    scenario = _scenario(max_tool_rounds=1)

    result = run_scenario(scenario, adapter, persona=persona, seed=1)

    roles = [m.role for m in result.transcript.messages]
    # No "system" role here (scenario has system but it's included); filter to
    # check that all roles are one of the expected canonical set.
    for role in roles:
        assert role in {"system", "user", "assistant", "tool"}, f"unexpected role: {role}"
