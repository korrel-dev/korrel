"""Failure-mode showcase: a mock tool that raises.

Pattern: korrel surfaces a tool failure as a clean error plus a partial
transcript, instead of hiding it or dumping a raw traceback. When the mock tool
raises, ``korrel run`` exits non-zero and prints
``error: tool '<name>' raised: <message>`` on stderr, then writes the transcript
built up to the failing call.

This entry is EXPECTED to exit non-zero. The gallery CI job asserts that the
failure is surfaced as a named tool error, not that the run passes. See
gallery/index.json (run: tool_error) and gallery/CONTRIBUTING.md.

The other single-turn entries print ``stop reason : max_turns`` on a normal run.
This entry exits through the tool-failure path instead: its stderr carries the
``error: tool '<name>' raised`` line and the partial-transcript path, with no
summary line.

Offline/needs-key: runs offline with no key.

Live run (needs a key): replace the scripted ``adapter`` with
``adapter_from_provider(AnthropicProvider())`` and set ANTHROPIC_API_KEY. The
tool still raises (it models a real downstream failure), so a live run surfaces
the same named tool error.

Export: run-only entry (see gallery/index.json).
"""

from typing import Any

from korrel import MockTool, Persona, Rubric, Scenario
from korrel.types import Message, ToolCall, ToolFunction


def charge_card(arguments: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """A tool that raises to model a downstream failure (a gateway timeout)."""
    raise RuntimeError("payment gateway timeout")


charge_tool = MockTool(
    name="charge_card",
    schema={
        "type": "function",
        "function": {
            "name": "charge_card",
            "description": "Charge the customer's card for an amount.",
            "parameters": {
                "type": "object",
                "properties": {"amount": {"type": "number"}},
                "required": ["amount"],
            },
        },
    },
    respond=charge_card,
)


def charge_succeeded(
    completion: list[Message], info: dict[str, Any], **kwargs: Any
) -> float:
    """Reward for a successful charge. Unreached here: the tool raises first."""
    for message in completion:
        if message.role == "assistant" and message.content and "charged" in message.content.lower():
            return 1.0
    return 0.0


def scripted_agent(*turns: Message):
    """Return an adapter that replays a fixed queue of assistant messages."""
    queue = list(turns)

    def _adapter(messages: list[Message], tools: list[dict]) -> Message:
        return queue.pop(0) if queue else Message(role="assistant", content="")

    return _adapter


scenario = Scenario(
    id="tool_failure_showcase",
    system="You are a checkout agent. Charge the customer's card to complete the order.",
    persona=Persona(goal="Complete the purchase."),
    opening_message="Please charge my card and complete the order.",
    tools=[charge_tool],
    max_turns=1,
    info={"amount": 49.99},
    rubric=Rubric(funcs=[charge_succeeded], pass_threshold=0.5),
)

adapter = scripted_agent(
    Message(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id="call_1",
                function=ToolFunction(
                    name="charge_card",
                    arguments='{"amount": 49.99}',
                ),
            )
        ],
    ),
)

# Go live (needs a key): swap the scripted adapter for a real agent. The tool
# still raises, so the live run surfaces the same named tool error.
#   from korrel import adapter_from_provider
#   from korrel.providers import AnthropicProvider
#   adapter = adapter_from_provider(AnthropicProvider())
