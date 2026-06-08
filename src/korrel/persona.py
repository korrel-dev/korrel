"""Persona: the LLM-driven user-simulator.

A Persona plays the user across a multi-turn conversation. Given the
conversation so far it produces the next user message. It is LLM-driven, not
scripted, and defaults to Claude with a bring-your-own key. For tests, inject a
fake user-simulator (any object exposing ``next_message``) through the
``persona`` argument of ``run_scenario``.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .providers import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, AnthropicProvider
from .types import Message

_SIMULATOR_SYSTEM = (
    "You are simulating a human user talking to an assistant. Stay in character "
    "for the whole conversation. Pursue the goal and follow the behavior below. "
    "Produce only the user's next message, with no preamble, labels, or quotation "
    "marks.\n\nGoal: {goal}\n\nBehavior: {behavior}"
)


class Persona(BaseModel):
    """A user-simulator driven by an LLM.

    ``goal`` is what the simulated user wants; ``behavior`` describes how they
    act. The remaining fields configure the simulator's own model call. Pass an
    instantiated client through ``client`` to reuse a connection; otherwise the
    key is read from the environment at call time.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    goal: str
    behavior: str = ""
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 1.0
    client: Any = Field(default=None, exclude=True, repr=False)

    def _provider(self) -> AnthropicProvider:
        return AnthropicProvider(
            model=self.model,
            client=self.client,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def _render_conversation(self, messages: list[Message]) -> str:
        lines: list[str] = []
        for msg in messages:
            if msg.role == "system":
                continue
            if msg.role == "assistant":
                if msg.content:
                    lines.append(f"Assistant: {msg.content}")
            elif msg.role == "user":
                lines.append(f"User: {msg.content or ''}")
            elif msg.role == "tool":
                lines.append(f"(tool result: {msg.content or ''})")
        return "\n".join(lines)

    def next_message(self, messages: list[Message]) -> Optional[str]:
        """Return the simulated user's next message given the history."""

        system = _SIMULATOR_SYSTEM.format(goal=self.goal, behavior=self.behavior)
        convo = self._render_conversation(messages)
        prompt = (
            f"Conversation so far:\n{convo}\n\n"
            "Write the user's next message."
        )
        reply = self._provider().complete(
            [Message(role="user", content=prompt)],
            system=system,
        )
        return (reply.content or "").strip()
