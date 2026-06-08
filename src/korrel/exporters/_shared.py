"""korrel.exporters._shared: conversion helpers shared across exporters.

These helpers convert between the canonical Korrel message types and the flat
message shapes used by downstream frameworks (verifiers, OpenEnv). None of them
import a framework at definition time.

Confirmed against: korrel canonical types as of v0.2 (src/korrel/types.py).
"""

from __future__ import annotations

import json
from typing import Any, Optional


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a message or tool call object.

    The value may arrive as a pydantic object attribute or a plain dict entry.
    Unlike ``getattr(...) or dict.get(...)``, this does not treat a falsy-but-
    present value (an empty string content) as missing: only ``None`` or a
    truly absent key falls back to ``default``.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    value = getattr(obj, key, default)
    return default if value is None else value


def _content_to_str(content: Any) -> str:
    """Narrow MessageContent (str | list[ContentPart]) to str.

    Content may be a list of content parts (text, image, audio). Korrel
    canonical content is Optional[str]. String content passes through; list
    content is serialized as JSON per the spec (lossy edge: content-shape
    narrowing documented in the mapping spec).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # List of content parts: join text parts, serialize non-text as JSON.
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
            else:
                parts.append(json.dumps(part))
        elif hasattr(part, "type"):
            if getattr(part, "type", None) == "text":
                parts.append(getattr(part, "text", ""))
            else:
                parts.append(
                    json.dumps(
                        part.model_dump() if hasattr(part, "model_dump") else str(part)
                    )
                )
        else:
            parts.append(str(part))
    return "".join(parts)


def _to_korrel_messages(messages: list[Any]) -> list[Any]:
    """Convert a list of flat messages to korrel canonical Messages.

    Field-by-field per the mapping spec (Section: Reward-function value shim):
    - system/user/assistant without tool calls: content narrowed to str.
    - assistant with tool calls: FLAT ToolCall{id,name,arguments} ->
      NESTED korrel.ToolCall{id,type:"function",function:{name,arguments}}.
    - tool result: ToolMessage{role:"tool",tool_call_id,content}.
    ``arguments`` stays a JSON string on both sides.
    """
    from ..types import Message, ToolCall, ToolFunction

    result: list[Message] = []
    for msg in messages:
        role = _field(msg, "role")
        if role is None:
            continue

        if role == "system":
            result.append(
                Message(role="system", content=_content_to_str(_field(msg, "content")))
            )

        elif role == "user":
            result.append(
                Message(role="user", content=_content_to_str(_field(msg, "content")))
            )

        elif role == "assistant":
            content = _field(msg, "content")
            raw_tool_calls = _field(msg, "tool_calls")
            korrel_tool_calls: Optional[list[ToolCall]] = None
            if raw_tool_calls:
                korrel_tool_calls = []
                for tc in raw_tool_calls:
                    tc_id = _field(tc, "id", "") or ""
                    tc_name = _field(tc, "name", "") or ""
                    tc_args = _field(tc, "arguments", "{}") or "{}"
                    korrel_tool_calls.append(
                        ToolCall(
                            id=tc_id,
                            type="function",
                            function=ToolFunction(name=tc_name, arguments=tc_args),
                        )
                    )
            result.append(
                Message(
                    role="assistant",
                    content=_content_to_str(content) if content is not None else None,
                    tool_calls=korrel_tool_calls or None,
                )
            )

        elif role == "tool":
            result.append(
                Message(
                    role="tool",
                    content=_content_to_str(_field(msg, "content")),
                    tool_call_id=_field(msg, "tool_call_id"),
                )
            )

    return result
