"""Provider resolution and the Anthropic <-> canonical mapping.

The default provider is Claude via the native ``anthropic`` SDK. Providers are
pluggable: a caller may pass an already-instantiated client, or a model name
(optionally with a base URL) and let Korrel build the client, reading the API
key from the environment at call time.

Keys are never stored. Nothing here writes a key to disk, logs it, or holds it
on a field. The provider stores the name of the environment variable to read,
not the key value, and reads it only when a request is made.

The mapping is built against the installed ``anthropic`` shape:
``ToolUseBlock`` carries ``id``, ``name`` and ``input`` (a dict);
``TextBlock`` carries ``text``; tool results are sent back as ``tool_result``
blocks inside a user message keyed by ``tool_use_id``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Protocol, runtime_checkable

from .types import Message, ToolCall, ToolFunction, ToolSchema

# Default Claude model for Korrel's own LLM calls (Persona simulator, judge).
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"


class MissingAPIKeyError(RuntimeError):
    """No API key found in the environment variable the provider reads.

    Subclasses ``RuntimeError`` so existing callers catching ``RuntimeError``
    keep working. The message never contains a key value.
    """


@runtime_checkable
class Provider(Protocol):
    """Something the loop and helpers can call to get a completion."""

    def complete(
        self,
        messages: list[Message],
        *,
        system: Optional[str] = None,
        tools: Optional[list[ToolSchema]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Message:
        ...


def tool_schema_to_anthropic(schema: ToolSchema) -> dict[str, Any]:
    """Map a chat-completions tool definition to an Anthropic tool param."""

    fn = schema.get("function", schema)
    out: dict[str, Any] = {
        "name": fn["name"],
        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
    }
    if fn.get("description"):
        out["description"] = fn["description"]
    return out


def to_anthropic_request(
    messages: list[Message],
    *,
    system: Optional[str] = None,
    tools: Optional[list[ToolSchema]] = None,
) -> dict[str, Any]:
    """Build the Anthropic request body from canonical messages.

    System messages are folded into the top-level ``system`` string.
    Consecutive ``role: "tool"`` messages are merged into one user message
    carrying multiple ``tool_result`` blocks, as the Messages API expects.
    """

    system_parts: list[str] = []
    if system:
        system_parts.append(system)

    api_messages: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            api_messages.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for msg in messages:
        if msg.role == "system":
            if msg.content:
                system_parts.append(msg.content)
            continue

        if msg.role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "",
                    "content": msg.content or "",
                }
            )
            continue

        flush_tool_results()

        if msg.role == "user":
            api_messages.append(
                {"role": "user", "content": [{"type": "text", "text": msg.content or ""}]}
            )
        elif msg.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for call in msg.tool_calls or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.function.name,
                        "input": json.loads(call.function.arguments or "{}"),
                    }
                )
            api_messages.append({"role": "assistant", "content": blocks})

    flush_tool_results()

    body: dict[str, Any] = {"messages": api_messages}
    if system_parts:
        body["system"] = "\n\n".join(system_parts)
    if tools:
        body["tools"] = [tool_schema_to_anthropic(t) for t in tools]
    return body


def from_anthropic_response(response: Any) -> Message:
    """Map an Anthropic Messages response to a canonical assistant message."""

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(getattr(block, "text", ""))
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=getattr(block, "id", ""),
                    function=ToolFunction(
                        name=getattr(block, "name", ""),
                        arguments=json.dumps(getattr(block, "input", {}) or {}),
                    ),
                )
            )

    content = "".join(text_parts) or None
    return Message(
        role="assistant",
        content=content,
        tool_calls=tool_calls or None,
    )


class AnthropicProvider:
    """Claude provider via the native ``anthropic`` SDK.

    Pass an already-built ``client`` to use it directly, or leave it unset to
    have one constructed on first use from ``api_key_env`` (read at call time)
    and ``base_url``. The key value is never stored on the instance.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        client: Any = None,
        base_url: Optional[str] = None,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        self._client = client
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import anthropic

        key = os.environ.get(self.api_key_env)
        if not key:
            raise MissingAPIKeyError(
                f"No API key found in {self.api_key_env}. Korrel reads provider "
                "keys from the environment at call time and stores none. Set the "
                "variable or pass an instantiated client."
            )
        kwargs: dict[str, Any] = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return anthropic.Anthropic(**kwargs)

    def complete(
        self,
        messages: list[Message],
        *,
        system: Optional[str] = None,
        tools: Optional[list[ToolSchema]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Message:
        client = self._get_client()
        body = to_anthropic_request(messages, system=system, tools=tools)
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            **body,
        )
        return from_anthropic_response(response)

    @property
    def params(self) -> dict[str, Any]:
        """Recorded request parameters (never the key)."""

        return {
            "provider": "anthropic",
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "base_url": self.base_url,
        }


def resolve_provider(spec: Any = None) -> Provider:
    """Turn a provider spec into something the loop can call.

    Accepts: ``None`` (default Claude), an object that already implements
    ``complete`` (used as-is), an instantiated ``anthropic`` client (wrapped),
    a model-name string, or a dict of ``AnthropicProvider`` keyword arguments.
    """

    if spec is None:
        return AnthropicProvider()
    if hasattr(spec, "complete"):
        return spec
    if hasattr(spec, "messages"):  # an instantiated anthropic-style client
        return AnthropicProvider(client=spec)
    if isinstance(spec, str):
        return AnthropicProvider(model=spec)
    if isinstance(spec, dict):
        return AnthropicProvider(**spec)
    raise TypeError(f"Cannot resolve provider from {type(spec)!r}")
