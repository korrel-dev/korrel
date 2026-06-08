"""A refund-support scenario: persona, an orders mock tool, and a rubric.

This is the scenario definition only. The end-to-end run against real Claude
arrives with the CLI in dispatch B. The smoke test drives this scenario through
``run_scenario`` with a fake adapter and a fake persona, so it runs with no
network and no keys.
"""

from typing import Any

from korrel import MockTool, Persona, Rubric, Scenario
from korrel.types import Message

ORDERS = {
    "A1001": {"order_id": "A1001", "item": "wireless mouse", "amount": 49.99, "refundable": True},
}


def lookup_order(arguments: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    state["lookup_called"] = True
    order = ORDERS.get(arguments.get("order_id", ""))
    return order if order is not None else {"error": "order not found"}


orders_tool = MockTool(
    name="lookup_order",
    schema={
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by its id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    respond=lookup_order,
)


def confirmed_refund_amount(completion: list[Message], info: dict[str, Any]) -> float:
    """1.0 if an assistant turn confirms the refund and states the amount."""

    amount = f"{info['amount']:.2f}"
    for message in completion:
        if message.role == "assistant" and message.content:
            text = message.content.lower()
            if "refund" in text and amount in message.content:
                return 1.0
    return 0.0


scenario = Scenario(
    id="support_refund",
    system="You are a support agent. Resolve refund requests using the lookup_order tool.",
    persona=Persona(
        goal="Get a refund for order A1001, a wireless mouse that arrived broken.",
        behavior="Polite but firm. Provides the order id when asked.",
    ),
    opening_message="Hi, my order A1001 arrived broken and I would like a refund.",
    tools=[orders_tool],
    max_turns=2,
    seed=7,
    info={"order_id": "A1001", "amount": 49.99, "refundable": True},
    rubric=Rubric(funcs=[confirmed_refund_amount], pass_threshold=0.5),
)
