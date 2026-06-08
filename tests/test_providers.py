"""Offline tests for the Anthropic <-> canonical mapping (no network)."""

import json
from types import SimpleNamespace

from korrel import Message, ToolCall, ToolFunction
from korrel.providers import (
    from_anthropic_response,
    to_anthropic_request,
    tool_schema_to_anthropic,
)


def test_canonical_to_anthropic_request_shapes():
    messages = [
        Message(role="system", content="be helpful"),
        Message(role="user", content="look up A1"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=ToolFunction(
                        name="lookup", arguments=json.dumps({"id": "A1"})
                    ),
                )
            ],
        ),
        Message(role="tool", content='{"ok": true}', tool_call_id="call_1"),
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "look up",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    body = to_anthropic_request(messages, tools=tools)

    assert body["system"] == "be helpful"
    api = body["messages"]
    # user, assistant(tool_use), user(tool_result)
    assert api[0]["role"] == "user"
    assert api[1]["role"] == "assistant"
    tool_use = api[1]["content"][0]
    assert tool_use["type"] == "tool_use"
    assert tool_use["id"] == "call_1"
    assert tool_use["input"] == {"id": "A1"}  # decoded from the JSON string
    assert api[2]["role"] == "user"
    tool_result = api[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "call_1"
    assert body["tools"][0]["name"] == "lookup"
    assert body["tools"][0]["input_schema"] == {"type": "object", "properties": {}}


def test_tool_schema_mapping_uses_input_schema():
    schema = {
        "type": "function",
        "function": {
            "name": "f",
            "description": "d",
            "parameters": {"type": "object"},
        },
    }
    out = tool_schema_to_anthropic(schema)
    assert out == {
        "name": "f",
        "description": "d",
        "input_schema": {"type": "object"},
    }


def test_anthropic_response_to_canonical_assistant_message():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="here you go"),
            SimpleNamespace(
                type="tool_use", id="toolu_9", name="lookup", input={"id": "A1"}
            ),
        ]
    )
    message = from_anthropic_response(response)
    assert message.role == "assistant"
    assert message.content == "here you go"
    assert message.tool_calls is not None
    call = message.tool_calls[0]
    assert call.id == "toolu_9"
    assert call.type == "function"
    assert call.function.name == "lookup"
    # arguments is a JSON-encoded string per the canonical contract.
    assert json.loads(call.function.arguments) == {"id": "A1"}
