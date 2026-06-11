"""Single-turn gate, scripted, zero key.

Pattern: a one-exchange scenario scored by a plain reward function, with no
tools and no judge. At ``max_turns=1`` the persona is constructed but never
invoked, so the run makes no model call and needs no key. This is the smallest
entry that proves the CI gate runs offline.

Offline/needs-key: runs offline with no key (the scripted adapter answers
locally).

Live run (needs a key): replace the scripted ``adapter`` with a real agent via
``adapter_from_provider(AnthropicProvider())`` and set ANTHROPIC_API_KEY. The
scenario (persona, rubric) is unchanged; only the agent becomes live.

Export: verified against verifiers and openenv (see gallery/index.json).
"""

from typing import Any

from korrel import Persona, Rubric, Scenario
from korrel.types import Message


def states_closing_time(
    completion: list[Message], info: dict[str, Any], **kwargs: Any
) -> float:
    """Return 1.0 when an assistant turn states the published closing time.

    The ``**kwargs`` is part of the reward-function contract: verifiers invokes
    reward functions with a merged kwargs mapping, so a function without it
    breaks when the scenario is exported.
    """
    closing = info["closing_time"]
    for message in completion:
        if message.role == "assistant" and message.content and closing in message.content:
            return 1.0
    return 0.0


def scripted_agent(*turns: Message):
    """Return an adapter that replays a fixed queue of assistant messages.

    A scripted adapter keeps the gate deterministic and offline: no provider,
    no key. The reusable artifact is the scenario; the adapter is the local
    stand-in for the agent under test.
    """
    queue = list(turns)

    def _adapter(messages: list[Message], tools: list[dict]) -> Message:
        return queue.pop(0) if queue else Message(role="assistant", content="")

    return _adapter


scenario = Scenario(
    id="support_hours",
    system="You are a support assistant. State the published support hours accurately.",
    # Persona is required by the type. At max_turns=1 the simulated user is
    # never invoked, so this run makes no model call and needs no key.
    persona=Persona(goal="Find out when support closes on Friday."),
    opening_message="What time does support close on Friday?",
    max_turns=1,
    info={"closing_time": "8 pm Eastern"},
    rubric=Rubric(funcs=[states_closing_time], pass_threshold=0.5),
)

adapter = scripted_agent(
    Message(role="assistant", content="Support closes at 8 pm Eastern on Friday."),
)

# Go live (needs a key): swap the scripted adapter for a real agent.
#   from korrel import adapter_from_provider
#   from korrel.providers import AnthropicProvider
#   adapter = adapter_from_provider(AnthropicProvider())
