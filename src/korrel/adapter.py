"""AgentAdapter: the user-supplied agent under test.

An adapter is any callable that takes the conversation so far as canonical
``Message`` objects plus the available tool schemas, and returns the next
assistant ``Message`` (which may carry ``tool_calls``). The user wires their
own agent and their own keys here; Korrel holds none of them.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Message, ToolSchema


@runtime_checkable
class AgentAdapter(Protocol):
    def __call__(
        self, messages: list[Message], tools: list[ToolSchema]
    ) -> Message:
        ...
