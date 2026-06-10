"""MockTool: a programmable stand-in for a real tool the agent can call."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

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

    The value is stored on the ``schema_`` field with a ``schema`` alias, so
    the public surface stays ``MockTool(schema=...)`` for construction and
    ``tool.schema`` for reads, while the field name itself does not shadow the
    deprecated ``BaseModel.schema`` method (which otherwise emits a UserWarning
    on every import).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    schema_: dict[str, Any] = Field(alias="schema")
    respond: RespondFn

    @property
    def schema(self) -> dict[str, Any]:
        """The chat-completions tool schema (public read alias for ``schema_``)."""
        return self.schema_

    def call(self, arguments: dict[str, Any], state: dict[str, Any]) -> Any:
        return self.respond(arguments, state)
