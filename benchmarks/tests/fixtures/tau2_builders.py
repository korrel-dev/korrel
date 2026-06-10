"""Factory helpers that build tau2 message objects and SimulationRuns for tests.

All helpers are pure (no network, no API keys).  They import tau2 at the top
level, so this module must be imported only after conftest.py has called
ensure_tau2_data_dir() to set TAU2_DATA_DIR.
"""

from __future__ import annotations

from typing import Any

from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.orchestrator.modes import CommunicationMode


def make_tool_call(
    tc_id: str = "tc1",
    name: str = "search",
    arguments: dict[str, Any] | None = None,
    requestor: str = "assistant",
) -> ToolCall:
    """Return a minimal tau2 ToolCall."""
    return ToolCall(
        id=tc_id,
        name=name,
        arguments=arguments if arguments is not None else {"q": "value"},
        requestor=requestor,
    )


def make_tool_message(
    tc_id: str = "tc1",
    content: str = "tool result",
    requestor: str = "assistant",
) -> ToolMessage:
    """Return a minimal tau2 ToolMessage."""
    return ToolMessage(
        id=tc_id,
        role="tool",
        content=content,
        requestor=requestor,
    )


def make_multi_tool_message(
    tool_messages: list[ToolMessage] | None = None,
) -> MultiToolMessage:
    """Return a tau2 MultiToolMessage containing two ToolMessages by default."""
    if tool_messages is None:
        tool_messages = [
            make_tool_message("tc1", "result1"),
            make_tool_message("tc2", "result2"),
        ]
    return MultiToolMessage(role="tool", tool_messages=tool_messages)


def make_minimal_message_sequence() -> list[Any]:
    """Return a minimal realistic half-duplex message sequence.

    system -> user -> assistant (with tool call) -> tool -> assistant (final)
    """
    tc = make_tool_call("tc-a", "lookup_order", {"order_id": "123"})
    return [
        SystemMessage(role="system", content="You are a retail support agent."),
        UserMessage(role="user", content="Where is my order?"),
        AssistantMessage(role="assistant", content=None, tool_calls=[tc]),
        ToolMessage(id="tc-a", role="tool", content="Order is shipped.", requestor="assistant"),
        AssistantMessage(role="assistant", content="Your order has been shipped."),
    ]


def make_simulation_run(
    task_id: str = "0",
    messages: list[Any] | None = None,
    termination_reason: TerminationReason = TerminationReason.AGENT_STOP,
    run_id: str = "test-run",
    seed: int = 300,
) -> SimulationRun:
    """Return a minimal SimulationRun suitable for evaluate_simulation."""
    if messages is None:
        messages = [
            UserMessage(role="user", content="Hello"),
            AssistantMessage(role="assistant", content="I cannot help with that."),
        ]
    return SimulationRun(
        id=run_id,
        task_id=task_id,
        start_time="1970-01-01T00:00:00",
        end_time="1970-01-01T00:00:01",
        duration=1.0,
        termination_reason=termination_reason,
        messages=messages,
        mode=CommunicationMode.HALF_DUPLEX.value,
        seed=seed,
    )
