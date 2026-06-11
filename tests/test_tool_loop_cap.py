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


# ---------------------------------------------------------------------------
# Additional gap coverage
# ---------------------------------------------------------------------------


def test_stop_reason_none_when_no_tool_calls():
    # An adapter that never emits tool calls must leave stop_reason as None.
    adapter = ScriptedAdapter([Message(role="assistant", content="plain answer")])
    persona = ScriptedPersona([])
    scenario = _scenario(max_tool_rounds=3)

    result = run_scenario(scenario, adapter, persona=persona, seed=0)

    assert len(result.transcript.turns) == 1
    assert result.transcript.turns[0].stop_reason is None


def test_boundary_exactly_at_cap_with_one_round():
    # max_tool_rounds=1: the adapter is called, emits a tool call,
    # tool is resolved, adapter is called again (round 0 completes),
    # now tool_round(1) >= max_tool_rounds(1) -> cap fires on the NEXT
    # tool call attempt.  So: open + round-0 + round-0-assistant +
    # second tool call from adapter triggers cap.
    # Sequence: tool_msg, tool_msg (second), plain (never reached)
    adapter = ScriptedAdapter([_tool_msg(), _tool_msg(), Message(role="assistant", content="done")])
    persona = ScriptedPersona([])
    scenario = _scenario(max_tool_rounds=1)

    result = run_scenario(scenario, adapter, persona=persona, seed=0)

    assert result.transcript.turns[0].stop_reason == "max_tool_rounds"


def test_boundary_one_below_cap_no_stop_reason():
    # max_tool_rounds=2: adapter does exactly 1 tool round then final answer.
    # tool_round never reaches 2 -> no cap.
    adapter = ScriptedAdapter([_tool_msg(), Message(role="assistant", content="done")])
    persona = ScriptedPersona([])
    scenario = _scenario(max_tool_rounds=2)

    result = run_scenario(scenario, adapter, persona=persona, seed=0)

    assert result.transcript.turns[0].stop_reason is None


@pytest.mark.parametrize("cap", [1, 2, 3, 5])
def test_cap_fires_for_various_limits(cap):
    # For any cap value, an adapter that keeps emitting tool calls must
    # eventually hit it and set stop_reason.
    adapter = ScriptedAdapter([_tool_msg()] * (cap + 5))
    persona = ScriptedPersona([])
    scenario = _scenario(max_tool_rounds=cap)

    result = run_scenario(scenario, adapter, persona=persona, seed=0)

    assert result.transcript.turns[0].stop_reason == "max_tool_rounds"


def test_capped_turn_message_count_is_deterministic():
    # With max_tool_rounds=2 and an endless stream of tool calls:
    # - initial assistant (tool call) + tool result + round-1 assistant (tool call) +
    #   tool result + round-2 assistant (tool call): cap fires.
    # That means messages in the turn = 1(user) + 1(asst) + 1(tool) + 1(asst) + 1(tool) + 1(asst)
    # = 6, but depends on runtime details. What matters: count is fixed, not growing.
    adapter = ScriptedAdapter([_tool_msg()] * 30)
    persona = ScriptedPersona([])
    scenario = _scenario(max_tool_rounds=2)

    result = run_scenario(scenario, adapter, persona=persona, seed=0)

    turn = result.transcript.turns[0]
    assert turn.stop_reason == "max_tool_rounds"
    # Run again with a fresh adapter: message count must be identical (deterministic).
    adapter2 = ScriptedAdapter([_tool_msg()] * 30)
    result2 = run_scenario(scenario, adapter2, persona=persona, seed=0)
    assert len(result2.transcript.turns[0].messages) == len(turn.messages)


def test_max_tool_rounds_negative_raises():
    with pytest.raises(Exception):
        Scenario(
            id="neg",
            system="s",
            persona=Persona(goal="g", behavior="b"),
            opening_message="hi",
            max_tool_rounds=-1,
        )


# ---------------------------------------------------------------------------
# Run-level cap markers (Transcript.stop_reason, RunResult surfacing)
# ---------------------------------------------------------------------------


def test_run_stop_reason_max_turns_when_persona_never_ends():
    # The persona queue never empties before max_turns: the turn budget cuts
    # the run, and the run-level marker says so.
    adapter = ScriptedAdapter(
        [Message(role="assistant", content=f"answer {i}") for i in range(3)]
    )
    persona = ScriptedPersona(["again", "more", "still going"])
    scenario = _scenario(max_turns=3)

    result = run_scenario(scenario, adapter, persona=persona, seed=1)

    assert result.transcript.stop_reason == "max_turns"
    assert result.stop_reason == "max_turns"


def test_run_stop_reason_persona_ended():
    # The persona returns None before the budget: a natural end, not a cap.
    adapter = ScriptedAdapter(
        [
            Message(role="assistant", content="one"),
            Message(role="assistant", content="two"),
        ]
    )
    persona = ScriptedPersona([])
    scenario = _scenario(max_turns=5)

    result = run_scenario(scenario, adapter, persona=persona, seed=1)

    assert result.transcript.stop_reason == "persona_ended"
    assert result.stop_reason == "persona_ended"


def test_run_stop_reason_max_turns_at_one_turn():
    # max_turns=1 ends on the turn budget without invoking the persona.
    adapter = ScriptedAdapter([Message(role="assistant", content="only")])
    persona = ScriptedPersona(["never used"])
    scenario = _scenario(max_turns=1)

    result = run_scenario(scenario, adapter, persona=persona, seed=1)

    assert result.stop_reason == "max_turns"
    assert persona.calls == 0


def test_tool_rounds_capped_surfaced_on_run_result():
    # A turn that hits max_tool_rounds is counted on RunResult; the per-turn
    # stop_reason behavior is unchanged.
    adapter = ScriptedAdapter([_tool_msg()] * 20)
    persona = ScriptedPersona([])
    scenario = _scenario(max_tool_rounds=2)

    result = run_scenario(scenario, adapter, persona=persona, seed=1)

    assert result.transcript.turns[0].stop_reason == "max_tool_rounds"
    assert result.tool_rounds_capped == 1


def test_tool_rounds_capped_zero_when_under_cap():
    adapter = ScriptedAdapter([_tool_msg(), Message(role="assistant", content="done")])
    persona = ScriptedPersona([])
    scenario = _scenario(max_tool_rounds=3)

    result = run_scenario(scenario, adapter, persona=persona, seed=1)

    assert result.tool_rounds_capped == 0


def test_tool_rounds_capped_counts_multiple_turns():
    # Both turns hit the cap: the count reflects it.
    adapter = ScriptedAdapter([_tool_msg()] * 40)
    persona = ScriptedPersona(["follow up"])
    scenario = _scenario(max_tool_rounds=1, max_turns=2)

    result = run_scenario(scenario, adapter, persona=persona, seed=1)

    assert result.tool_rounds_capped == 2
