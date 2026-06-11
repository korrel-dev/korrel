"""TEMPLATE: copy this file to gallery/<your_entry>.py and fill it in.

Replace this docstring with: the agent-testing pattern your entry demonstrates
(one or two sentences), whether a live run needs a key, and which export targets
it is verified against. Keep the prose declarative. No em-dashes, no emoji, no
exclamation marks.

A gallery entry is a self-contained scenario module that runs offline with no
key. It exposes two module-level names that ``korrel run`` and the gallery CI
job read: ``scenario`` (a Scenario) and ``adapter`` (a scripted callable). The
scenario (persona, tools, rubric) is the reusable artifact; the scripted adapter
is the local stand-in for the agent under test, so the gate is deterministic and
needs no provider and no key.

This template is skipped by the CI harness (it is named TEMPLATE.py).
"""

from typing import Any

from korrel import Persona, Rubric, Scenario
from korrel.types import Message


def meets_goal(completion: list[Message], info: dict[str, Any], **kwargs: Any) -> float:
    """Score the transcript against ground-truth ``info``.

    Reward signatures mirror verifiers: ``(completion, info, **kwargs) -> float``.
    Keep ``**kwargs``: verifiers calls reward functions with a merged kwargs
    mapping, so a function without it breaks when the scenario is exported.
    """
    expected = info["expected"]
    for message in completion:
        if message.role == "assistant" and message.content and expected in message.content:
            return 1.0
    return 0.0


def scripted_agent(*turns: Message):
    """Return an adapter that replays a fixed queue of assistant messages.

    For a tool-calling turn, build the message with ``tool_calls`` (see
    gallery/order_triage.py). For a multi-turn persona conversation, subclass
    Persona with a scripted ``next_message`` (see gallery/password_reset.py).
    """
    queue = list(turns)

    def _adapter(messages: list[Message], tools: list[dict]) -> Message:
        return queue.pop(0) if queue else Message(role="assistant", content="")

    return _adapter


scenario = Scenario(
    id="template_entry",
    system="You are an assistant. Describe what the agent under test must do.",
    # Persona is required by the type. At max_turns=1 it is never invoked, so the
    # run makes no model call. For a multi-turn entry, use a scripted Persona.
    persona=Persona(goal="State what the simulated user wants."),
    opening_message="The first user message.",
    max_turns=1,
    info={"expected": "the string the reward function looks for"},
    rubric=Rubric(funcs=[meets_goal], pass_threshold=0.5),
)

adapter = scripted_agent(
    Message(role="assistant", content="A reply containing the string the reward function looks for"),
)

# Go live (needs a key): swap the scripted adapter for a real agent.
#   from korrel import adapter_from_provider
#   from korrel.providers import AnthropicProvider
#   adapter = adapter_from_provider(AnthropicProvider())
