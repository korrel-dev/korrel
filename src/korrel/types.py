"""Canonical transcript types for Korrel.

These types are provider-neutral and wire-compatible with the OpenAI
chat-completions message and tool-call schema. An assistant message carries a
``tool_calls`` array; each tool call is ``{id, type: "function", function:
{name, arguments}}`` where ``arguments`` is a JSON-encoded string (not a dict).
Tool results are messages with ``role: "tool"`` and a matching
``tool_call_id``.

This is the v0.2 verifiers/OpenEnv export target. The attribute names and the
wire shape are part of that mapping surface, so they are not changed casually.
Provider-specific blocks (for example Anthropic ``tool_use`` and
``tool_result`` blocks) are translated into these canonical types in
``providers.py``; nothing here depends on the ``openai`` package.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel

Role = Literal["system", "user", "assistant", "tool"]

# A tool definition passed to an adapter, in chat-completions tool-schema shape:
# {"type": "function", "function": {"name": str, "description": str,
#  "parameters": {json schema}}}.
ToolSchema = dict[str, Any]


class ToolFunction(BaseModel):
    """The function payload of a tool call.

    ``arguments`` is a JSON-encoded string, matching the chat-completions wire
    format. Parse it with ``json.loads`` to recover the call arguments.
    """

    name: str
    arguments: str


class ToolCall(BaseModel):
    """A single tool call emitted by an assistant message."""

    id: str
    type: Literal["function"] = "function"
    function: ToolFunction


class Message(BaseModel):
    """A canonical transcript message.

    ``content`` is the text body (may be ``None`` when an assistant message
    only carries tool calls). ``tool_calls`` is present on assistant messages
    that call tools. ``tool_call_id`` links a ``role: "tool"`` message back to
    the assistant tool call it answers. ``name`` is the optional speaker name.
    """

    role: Role
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
