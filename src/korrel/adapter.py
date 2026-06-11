"""AgentAdapter: the user-supplied agent under test.

An adapter is any callable that takes the conversation so far as canonical
``Message`` objects plus the available tool schemas, and returns the next
assistant ``Message`` (which may carry ``tool_calls``). The user wires their
own agent and their own keys here; Korrel holds none of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .types import Message, ToolSchema

if TYPE_CHECKING:
    from .providers import Provider


@runtime_checkable
class AgentAdapter(Protocol):
    def __call__(
        self, messages: list[Message], tools: list[ToolSchema]
    ) -> Message:
        ...


def adapter_from_provider(provider: "Provider") -> AgentAdapter:
    """Wrap a Provider as an AgentAdapter.

    Returns a callable that delegates to ``provider.complete(messages,
    tools=tools)``. The caller supplies their own provider and their own
    keys; Korrel holds none.

    The provider is attached to the returned callable as ``.provider`` so
    callers (the CLI ``--model`` flag, the resolved-model summary line) can
    read or override the agent model. A custom adapter that does not expose
    ``.provider`` reports its model as unknown.
    """

    def _adapter(messages: list[Message], tools: list[ToolSchema]) -> Message:
        return provider.complete(messages, tools=tools)

    # The closure type does not declare .provider; the CLI --model path reads
    # this dynamic attribute.
    _adapter.provider = provider  # type: ignore[attr-defined]
    return _adapter  # type: ignore[return-value]
