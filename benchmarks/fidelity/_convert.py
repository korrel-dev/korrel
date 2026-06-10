"""Bidirectional conversion between tau2 messages and korrel canonical messages.

tau2 source confirmed at sierra-research/tau2-bench tag v1.0.0,
commit 17e07b1da2bbc0cadfddeea36412686e0604127b.

tau2 half-duplex message types (from src/tau2/data_model/message.py):

  SystemMessage
    role: "system"
    content: Optional[str]

  AssistantMessage (ParticipantMessageBase)
    role: "assistant"
    content: Optional[str]
    tool_calls: Optional[list[ToolCall]]

  UserMessage (ParticipantMessageBase)
    role: "user"
    content: Optional[str]
    tool_calls: Optional[list[ToolCall]]

  ToolMessage
    id: str          (matches the id of the tool call it answers)
    role: "tool"
    content: Optional[str]
    requestor: "user" | "assistant"
    error: bool

  MultiToolMessage
    role: "tool"
    tool_messages: list[ToolMessage]

  tau2 ToolCall: id: str, name: str, arguments: dict, requestor: str

korrel canonical types (from src/korrel/types.py):

  Message: role: Role, content: Optional[str],
           tool_calls: Optional[list[ToolCall]],
           tool_call_id: Optional[str], name: Optional[str]
  ToolCall: id: str, type: "function", function: ToolFunction
  ToolFunction: name: str, arguments: str  (JSON-encoded string, not dict)

Conversion rules:
  tau2 SystemMessage     -> korrel Message(role="system", content=...)
  tau2 UserMessage       -> korrel Message(role="user", content=...,
                              tool_calls=[...] if user has tool calls)
  tau2 AssistantMessage  -> korrel Message(role="assistant", content=...,
                              tool_calls=[...] if present)
  tau2 ToolMessage       -> korrel Message(role="tool", content=...,
                              tool_call_id=msg.id)
  tau2 MultiToolMessage  -> one korrel Message per inner ToolMessage
                            (flattened, since korrel has no multi-tool type)

tau2 ToolMessage note: the field is `id` (not `tool_call_id`); it matches
the `id` of the AssistantMessage ToolCall that triggered the tool. The field
`name` does not exist on tau2 ToolMessage. The korrel `name` field on a
"tool" message is used for the tool name in some adapters; it is set to None
in the reverse direction and populated from the ToolCall match where known.

All tau2 imports are deferred to keep this module importable in environments
where tau2 is not installed (e.g., root korrel tests).
"""

from __future__ import annotations

import json
from typing import Any

from korrel.types import Message, ToolCall, ToolFunction


# ---------------------------------------------------------------------------
# tau2 -> korrel
# ---------------------------------------------------------------------------


def tau2_tool_call_to_korrel(tc: Any) -> ToolCall:
    """Convert one tau2 ToolCall to a korrel ToolCall.

    tau2 ToolCall.arguments is a dict; korrel ToolFunction.arguments is a
    JSON-encoded string (chat-completions wire format).
    """
    return ToolCall(
        id=tc.id or "",
        type="function",
        function=ToolFunction(
            name=tc.name,
            arguments=json.dumps(tc.arguments, ensure_ascii=False),
        ),
    )


def tau2_message_to_korrel(msg: Any) -> list[Message]:
    """Convert one tau2 half-duplex message to a list of korrel canonical Messages.

    Returns a list because MultiToolMessage flattens to multiple korrel messages.
    Returns a single-element list for all other message types.

    Handles all five concrete tau2 message types confirmed from
    tau2 v1.0.0 src/tau2/data_model/message.py:
      SystemMessage, UserMessage, AssistantMessage,
      ToolMessage, MultiToolMessage.

    Audio messages (is_audio=True, half-duplex retail only uses text):
    content is set to "[audio]" so the message remains representable.
    """
    role = msg.role

    # MultiToolMessage: flatten to individual tool messages
    if hasattr(msg, "tool_messages"):
        result = []
        for tm in msg.tool_messages:
            result.extend(tau2_message_to_korrel(tm))
        return result

    # ToolMessage: role="tool", id matches the tool call id
    if role == "tool":
        return [
            Message(
                role="tool",
                content=getattr(msg, "content", None),
                # tau2 ToolMessage uses .id, not .tool_call_id
                tool_call_id=getattr(msg, "id", None),
                # tau2 ToolMessage has no .name field
                name=None,
            )
        ]

    # SystemMessage
    if role == "system":
        return [Message(role="system", content=getattr(msg, "content", None))]

    # UserMessage / AssistantMessage (ParticipantMessageBase subclasses)
    raw_tool_calls = getattr(msg, "tool_calls", None)
    korrel_tool_calls: list[ToolCall] | None = None
    if raw_tool_calls:
        korrel_tool_calls = [tau2_tool_call_to_korrel(tc) for tc in raw_tool_calls]

    content = getattr(msg, "content", None)
    # Audio-only messages: substitute a placeholder (not used in half-duplex retail)
    if getattr(msg, "is_audio", False) and not content:
        content = "[audio]"

    return [
        Message(
            role=role,  # type: ignore[arg-type]
            content=content,
            tool_calls=korrel_tool_calls,
        )
    ]


def tau2_messages_to_korrel(messages: list[Any]) -> list[Message]:
    """Convert a full tau2 half-duplex trajectory to korrel canonical messages.

    Filters out None entries. Flattens MultiToolMessage into individual
    "tool" messages.
    """
    result: list[Message] = []
    for m in messages:
        if m is None:
            continue
        result.extend(tau2_message_to_korrel(m))
    return result


# ---------------------------------------------------------------------------
# korrel -> tau2
# ---------------------------------------------------------------------------


def korrel_to_tau2_tool_call(tc: ToolCall, requestor: str = "assistant") -> Any:
    """Convert one korrel ToolCall back to a tau2 ToolCall.

    korrel arguments is a JSON string; tau2 arguments is a dict.
    """
    from tau2.data_model.message import ToolCall as Tau2ToolCall

    try:
        args = json.loads(tc.function.arguments)
    except (json.JSONDecodeError, ValueError):
        args = {"_raw": tc.function.arguments}

    return Tau2ToolCall(
        id=tc.id,
        name=tc.function.name,
        arguments=args,
        requestor=requestor,
    )


def korrel_message_to_tau2(msg: Message) -> Any:
    """Convert one korrel canonical Message back to the appropriate tau2 message type.

    Uses the concrete tau2 message subclasses so that the evaluator's
    isinstance checks and attribute access work correctly.

    tau2 ToolMessage uses `id` (not `tool_call_id`) and has no `name` field.
    """
    from tau2.data_model.message import AssistantMessage as Tau2AssistantMessage
    from tau2.data_model.message import SystemMessage as Tau2SystemMessage
    from tau2.data_model.message import ToolMessage as Tau2ToolMessage
    from tau2.data_model.message import UserMessage as Tau2UserMessage

    role = msg.role

    if role == "system":
        return Tau2SystemMessage(role="system", content=msg.content)

    if role == "tool":
        # tau2 ToolMessage.id matches the assistant tool-call id
        return Tau2ToolMessage(
            id=msg.tool_call_id or "",
            role="tool",
            content=msg.content or "",
            requestor="assistant",
        )

    tau2_tcs = None
    if msg.tool_calls:
        tau2_tcs = [korrel_to_tau2_tool_call(tc, requestor=role) for tc in msg.tool_calls]

    if role == "assistant":
        return Tau2AssistantMessage(
            role="assistant",
            content=msg.content,
            tool_calls=tau2_tcs,
        )

    # role == "user"
    return Tau2UserMessage(
        role="user",
        content=msg.content,
        tool_calls=tau2_tcs,
    )


def korrel_messages_to_tau2(messages: list[Message]) -> list[Any]:
    """Convert a korrel canonical transcript back to tau2 message objects."""
    return [korrel_message_to_tau2(m) for m in messages]
