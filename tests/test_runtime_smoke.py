"""End-to-end smoke test of the simulation loop with deterministic fakes."""

import json

from conftest import ScriptedAdapter, ScriptedPersona

from korrel import Message, ToolCall, ToolFunction, run_scenario


def build_adapter() -> ScriptedAdapter:
    tool_call_turn = Message(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id="call_1",
                function=ToolFunction(
                    name="lookup_order",
                    arguments=json.dumps({"order_id": "A1001"}),
                ),
            )
        ],
    )
    confirm_turn = Message(
        role="assistant",
        content="Your refund of $49.99 for order A1001 has been processed.",
    )
    closing_turn = Message(
        role="assistant",
        content="You are welcome. The refund will appear in a few days.",
    )
    return ScriptedAdapter([tool_call_turn, confirm_turn, closing_turn])


def test_runtime_smoke_runs_loop_end_to_end():
    from support_refund import scenario

    adapter = build_adapter()
    persona = ScriptedPersona(["Thank you, that is all I needed."])

    result = run_scenario(scenario, adapter, persona=persona, seed=7)

    # Transcript shape: system + user opening + assistant tool call + tool result
    # + assistant confirm + user follow-up + assistant closing.
    roles = [m.role for m in result.transcript.messages]
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert "tool" in roles
    assert roles[-1] == "assistant"

    # The mock tool was invoked: a tool message references the assistant call.
    tool_messages = [m for m in result.transcript.messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call_1"
    order = json.loads(tool_messages[0].content)
    assert order["order_id"] == "A1001"

    # The persona produced the follow-up user turn.
    assert persona.calls == 1

    # model_calls is the exact count of loop-issued model calls: one per agent
    # (adapter) invocation, one per user-simulator (persona) call. This run has
    # one tool round, so the adapter is called three times (opening turn, tool
    # follow-up, closing turn) and the persona once: four calls total.
    assert result.model_calls == adapter.calls + persona.calls
    assert result.model_calls == 4

    # The transcript records the seed and the simulator model.
    assert result.transcript.seed == 7
    assert result.transcript.model == "fake-persona"

    # Turns: max_turns is 2, so two turns recorded.
    assert len(result.transcript.turns) == 2

    # The rubric produced a score and a pass/fail.
    assert isinstance(result.score, float)
    assert result.passed is True
    assert result.failed_functions == []


def test_unknown_tool_call_yields_error_result():
    bad_call = Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="call_x",
                function=ToolFunction(name="does_not_exist", arguments="{}"),
            )
        ],
    )
    final = Message(role="assistant", content="Sorry, I could not do that.")
    adapter = ScriptedAdapter([bad_call, final])
    persona = ScriptedPersona([])

    from support_refund import scenario

    result = run_scenario(scenario, adapter, persona=persona, seed=1, )
    tool_messages = [m for m in result.transcript.messages if m.role == "tool"]
    assert len(tool_messages) == 1
    payload = json.loads(tool_messages[0].content)
    assert "error" in payload


def _no_tool_scenario(max_turns: int):
    from korrel import Persona, Rubric, Scenario

    def always_one(completion, info, **kwargs):
        return 1.0

    return Scenario(
        id="no_tool",
        system="sys",
        persona=Persona(goal="g", behavior="b"),
        opening_message="hello",
        max_turns=max_turns,
        seed=1,
        rubric=Rubric(funcs=[always_one], pass_threshold=0.5),
    )


def test_model_calls_no_tool_round_is_two_t_minus_one():
    # A scenario with no tools and no judge over T turns issues T agent calls
    # and T-1 user-simulator calls: the 2T-1 rule of thumb. Here T=3 -> 5.
    scenario = _no_tool_scenario(max_turns=3)
    adapter = ScriptedAdapter(
        [
            Message(role="assistant", content="one"),
            Message(role="assistant", content="two"),
            Message(role="assistant", content="three"),
        ]
    )
    persona = ScriptedPersona(["again", "more"])

    result = run_scenario(scenario, adapter, persona=persona, seed=1)

    # No tool messages: the count is pure agent + persona calls.
    assert all(m.role != "tool" for m in result.transcript.messages)
    assert adapter.calls == 3
    assert persona.calls == 2
    assert result.model_calls == adapter.calls + persona.calls
    assert result.model_calls == 5  # 2 * 3 - 1
