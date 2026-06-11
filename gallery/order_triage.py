"""Multi-tool, multi-round: a tool loop within one turn.

Pattern: the agent calls two mock tools in sequence inside a single turn. It
looks up an order to get a tracking id, then checks the carrier with that id,
then states the delivery status. ``max_tool_rounds`` bounds the loop; here the
agent uses two rounds. The persona is present but never invoked at
``max_turns=1``, so the run is offline with no key.

Offline/needs-key: runs offline with no key.

Live run (needs a key): replace the scripted ``adapter`` with
``adapter_from_provider(AnthropicProvider())`` and set ANTHROPIC_API_KEY. A live
agent reads the tracking id from the first tool result and passes it to the
second tool on its own.

Export: verified against verifiers and openenv (see gallery/index.json).
"""

import json
from typing import Any

from korrel import MockTool, Persona, Rubric, Scenario
from korrel.types import Message, ToolCall, ToolFunction

ORDERS = {
    "A1001": {"order_id": "A1001", "tracking_id": "TRK42"},
}
SHIPMENTS = {
    "TRK42": {"tracking_id": "TRK42", "status": "out for delivery"},
}


def lookup_order(arguments: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    state["order_looked_up"] = True
    order = ORDERS.get(arguments.get("order_id", ""))
    return order if order is not None else {"error": "order not found"}


def track_shipment(arguments: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    state["shipment_tracked"] = True
    shipment = SHIPMENTS.get(arguments.get("tracking_id", ""))
    return shipment if shipment is not None else {"error": "tracking id not found"}


order_tool = MockTool(
    name="lookup_order",
    schema={
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by its id to get its tracking id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    respond=lookup_order,
)

shipment_tool = MockTool(
    name="track_shipment",
    schema={
        "type": "function",
        "function": {
            "name": "track_shipment",
            "description": "Get the delivery status for a tracking id.",
            "parameters": {
                "type": "object",
                "properties": {"tracking_id": {"type": "string"}},
                "required": ["tracking_id"],
            },
        },
    },
    respond=track_shipment,
)


def states_delivery_status(
    completion: list[Message], info: dict[str, Any], **kwargs: Any
) -> float:
    """Return 1.0 when an assistant turn states the delivery status."""
    status = info["status"].lower()
    for message in completion:
        if message.role == "assistant" and message.content and status in message.content.lower():
            return 1.0
    return 0.0


def scripted_agent(*turns: Message):
    """Return an adapter that replays a fixed queue of assistant messages."""
    queue = list(turns)

    def _adapter(messages: list[Message], tools: list[dict]) -> Message:
        return queue.pop(0) if queue else Message(role="assistant", content="")

    return _adapter


scenario = Scenario(
    id="order_triage",
    system=(
        "You are a logistics agent. Look up the order to get its tracking id, "
        "check the carrier with track_shipment, then state the delivery status."
    ),
    persona=Persona(goal="Find out where order A1001 is."),
    opening_message="Where is my order A1001?",
    tools=[order_tool, shipment_tool],
    max_turns=1,
    max_tool_rounds=4,
    info={"status": "out for delivery"},
    rubric=Rubric(funcs=[states_delivery_status], pass_threshold=0.5),
)

adapter = scripted_agent(
    Message(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id="call_1",
                function=ToolFunction(
                    name="lookup_order",
                    arguments=json.dumps({"order_id": "A1001"}),
                ),
            )
        ],
    ),
    Message(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id="call_2",
                function=ToolFunction(
                    name="track_shipment",
                    arguments=json.dumps({"tracking_id": "TRK42"}),
                ),
            )
        ],
    ),
    Message(
        role="assistant",
        content="Your order A1001 is out for delivery and should arrive today.",
    ),
)

# Go live (needs a key): swap the scripted adapter for a real agent.
#   from korrel import adapter_from_provider
#   from korrel.providers import AnthropicProvider
#   adapter = adapter_from_provider(AnthropicProvider())
