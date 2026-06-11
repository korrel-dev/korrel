"""Multi-turn persona with one mock tool.

Pattern: a persona-driven conversation where the agent verifies an account with
a mock tool, then confirms a password reset. The simulated user supplies the
username on the second turn. Both sides are scripted so the run is offline and
deterministic: the scripted persona replays fixed user turns (no model call, no
key), and the scripted adapter replays fixed agent turns. This is the core
korrel shape: the simulated user drives the conversation across turns.

Offline/needs-key: runs offline with no key.

Live run (needs a key): replace ``ScriptedPersona(...).load([...])`` with a
plain ``Persona(goal=..., behavior=...)`` and the scripted adapter with
``adapter_from_provider(AnthropicProvider())``, then set ANTHROPIC_API_KEY. The
scenario goal, tool, and rubric are unchanged; the user-simulator and the agent
become live.

Export: run-only entry (see gallery/index.json). The persona is scripted for
the offline gate; export the live-persona variant when training.
"""

import json
from typing import Any, Optional

from pydantic import PrivateAttr

from korrel import MockTool, Persona, Rubric, Scenario
from korrel.types import Message, ToolCall, ToolFunction


class ScriptedPersona(Persona):
    """A user-simulator that replays fixed user turns. No model call, no key.

    Subclasses Persona so it satisfies the ``Scenario.persona`` type while
    overriding ``next_message`` to return scripted text instead of calling a
    model. Swap it for a plain Persona to drive a live multi-turn conversation.
    """

    _replies: list[str] = PrivateAttr(default_factory=list)
    _index: int = PrivateAttr(default=0)

    def load(self, replies: list[str]) -> "ScriptedPersona":
        self._replies = list(replies)
        return self

    def next_message(self, messages: list[Message]) -> Optional[str]:
        i = self._index
        self._index += 1
        return self._replies[i] if i < len(self._replies) else None


ACCOUNTS = {
    "jordan": {"username": "jordan", "verified": True, "email": "jordan@example.com"},
}


def lookup_account(arguments: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    state["looked_up"] = True
    account = ACCOUNTS.get(arguments.get("username", ""))
    return account if account is not None else {"error": "account not found"}


account_tool = MockTool(
    name="lookup_account",
    schema={
        "type": "function",
        "function": {
            "name": "lookup_account",
            "description": "Look up an account by its username to verify it exists.",
            "parameters": {
                "type": "object",
                "properties": {"username": {"type": "string"}},
                "required": ["username"],
            },
        },
    },
    respond=lookup_account,
)


def confirmed_reset(
    completion: list[Message], info: dict[str, Any], **kwargs: Any
) -> float:
    """Return 1.0 when an assistant turn confirms a reset for the named account."""
    username = info["username"].lower()
    for message in completion:
        if message.role == "assistant" and message.content:
            text = message.content.lower()
            if "reset" in text and username in text:
                return 1.0
    return 0.0


def scripted_agent(*turns: Message):
    """Return an adapter that replays a fixed queue of assistant messages."""
    queue = list(turns)

    def _adapter(messages: list[Message], tools: list[dict]) -> Message:
        return queue.pop(0) if queue else Message(role="assistant", content="")

    return _adapter


scenario = Scenario(
    id="password_reset",
    system=(
        "You are an account-support agent. Reset a password only after "
        "verifying the account with the lookup_account tool."
    ),
    persona=ScriptedPersona(
        goal="Reset the password for the account 'jordan'.",
        behavior="Cooperative. Provides the username when asked.",
    ).load(["My username is jordan."]),
    opening_message="I'm locked out and need to reset my password.",
    tools=[account_tool],
    max_turns=2,
    seed=11,
    info={"username": "jordan"},
    rubric=Rubric(funcs=[confirmed_reset], pass_threshold=0.5),
)

adapter = scripted_agent(
    Message(
        role="assistant",
        content="I can help reset your password. What is your username?",
    ),
    Message(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id="call_1",
                function=ToolFunction(
                    name="lookup_account",
                    arguments=json.dumps({"username": "jordan"}),
                ),
            )
        ],
    ),
    Message(
        role="assistant",
        content=(
            "Thanks Jordan. I verified the account and sent a password reset "
            "link to your email."
        ),
    ),
)

# Go live (needs a key): swap the scripted persona and adapter for live ones.
#   from korrel import adapter_from_provider
#   from korrel.providers import AnthropicProvider
#   scenario.persona = Persona(
#       goal="Reset the password for the account 'jordan'.",
#       behavior="Cooperative. Provides the username when asked.",
#   )
#   adapter = adapter_from_provider(AnthropicProvider())
