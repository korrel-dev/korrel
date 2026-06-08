"""MockTool: a programmable stand-in for a real tool the agent can call."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

# respond(arguments, state) -> tool result. ``arguments`` is the parsed call
# payload; ``state`` is a mutable dict scoped to one scenario run. The return
# value is serialized into the tool message content.
RespondFn = Callable[[dict[str, Any], dict[str, Any]], Any]


class MockTool(BaseModel):
    """A mock tool definition.

    ``schema`` is a chat-completions tool schema (the shape passed to an
    adapter). ``respond`` is plain Python: it receives the parsed call
    arguments and a mutable per-run state dict, and returns the tool result.
    Programmability is the reason the scenario format is code-first.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    schema: dict[str, Any]
    respond: RespondFn

    def call(self, arguments: dict[str, Any], state: dict[str, Any]) -> Any:
        return self.respond(arguments, state)
