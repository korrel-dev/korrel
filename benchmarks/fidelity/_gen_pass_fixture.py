"""Generate and verify synthetic_task0_pass.run.json.

Constructs a passing SimulationRun for tau2 retail task 0 by replaying
all 5 gold actions with correct arguments and exact tool responses.
Verifies that evaluate_simulation with ALL_IGNORE_BASIS returns 1.0.

Checked in as a generation script: re-run it if
fixtures/synthetic_task0_pass.run.json needs to be regenerated. Nothing
imports this module; all work happens inside main().

Run from benchmarks/:
  uv run --project . python fidelity/_gen_pass_fixture.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).parent.parent


def main() -> int:
    if str(BENCHMARKS_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARKS_DIR))

    from fidelity._tau2_data import ensure_tau2_data_dir

    ensure_tau2_data_dir()

    from tau2.data_model.message import (
        AssistantMessage,
        SystemMessage,
        ToolCall as Tau2ToolCall,
        ToolMessage,
        UserMessage,
    )
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.domains.retail.environment import get_environment, get_tasks
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.modes import CommunicationMode

    env = get_environment(solo_mode=False)

    def _get_content(name: str, args: dict) -> str:
        tc = Tau2ToolCall(id="x", name=name, arguments=args, requestor="assistant")
        return env.get_response(tc).content

    c1 = _get_content("find_user_id_by_name_zip", {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"})
    c2 = _get_content("get_order_details", {"order_id": "#W2378156"})
    c3 = _get_content("get_product_details", {"product_id": "1656367028"})
    c4 = _get_content("get_product_details", {"product_id": "4896585277"})
    c5 = _get_content(
        "exchange_delivered_order_items",
        {
            "order_id": "#W2378156",
            "item_ids": ["1151293680", "4983901480"],
            "new_item_ids": ["7706410293", "7747408585"],
            "payment_method_id": "credit_card_9513926",
        },
    )

    messages = [
        SystemMessage(role="system", content="You are a retail customer support agent."),
        UserMessage(role="user", content="I need to exchange two items in my order."),
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                Tau2ToolCall(
                    id="tc_p01",
                    name="find_user_id_by_name_zip",
                    arguments={"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"},
                    requestor="assistant",
                )
            ],
        ),
        ToolMessage(id="tc_p01", role="tool", content=c1, requestor="assistant", error=False),
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                Tau2ToolCall(
                    id="tc_p02",
                    name="get_order_details",
                    arguments={"order_id": "#W2378156"},
                    requestor="assistant",
                )
            ],
        ),
        ToolMessage(id="tc_p02", role="tool", content=c2, requestor="assistant", error=False),
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                Tau2ToolCall(
                    id="tc_p03",
                    name="get_product_details",
                    arguments={"product_id": "1656367028"},
                    requestor="assistant",
                ),
                Tau2ToolCall(
                    id="tc_p04",
                    name="get_product_details",
                    arguments={"product_id": "4896585277"},
                    requestor="assistant",
                ),
            ],
        ),
        ToolMessage(id="tc_p03", role="tool", content=c3, requestor="assistant", error=False),
        ToolMessage(id="tc_p04", role="tool", content=c4, requestor="assistant", error=False),
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                Tau2ToolCall(
                    id="tc_p05",
                    name="exchange_delivered_order_items",
                    arguments={
                        "order_id": "#W2378156",
                        "item_ids": ["1151293680", "4983901480"],
                        "new_item_ids": ["7706410293", "7747408585"],
                        "payment_method_id": "credit_card_9513926",
                    },
                    requestor="assistant",
                )
            ],
        ),
        ToolMessage(id="tc_p05", role="tool", content=c5, requestor="assistant", error=False),
        AssistantMessage(
            role="assistant",
            content="I have successfully processed the exchange for order #W2378156.",
            tool_calls=None,
        ),
        UserMessage(role="user", content="Thank you."),
    ]

    run = SimulationRun(
        id="synthetic-fixture-task0-pass",
        task_id="0",
        start_time="1970-01-01T00:00:00",
        end_time="1970-01-01T00:00:05",
        duration=5.0,
        termination_reason=TerminationReason.AGENT_STOP,
        messages=messages,
        mode=CommunicationMode.HALF_DUPLEX.value,
        seed=301,
    )

    tasks = get_tasks(task_split_name=None)
    task0 = next(t for t in tasks if t.id == "0")

    reward_info = evaluate_simulation(
        simulation=run,
        task=task0,
        evaluation_type=EvaluationType.ALL_IGNORE_BASIS,
        solo_mode=False,
        domain="retail",
        mode=CommunicationMode.HALF_DUPLEX,
        env_kwargs=None,
    )
    print(f"reward = {reward_info.reward}")
    print(f"reward_breakdown = {reward_info.reward_breakdown}")
    if reward_info.db_check:
        print(f"db_match = {reward_info.db_check.db_match}")
    if reward_info.action_checks:
        for ac in reward_info.action_checks:
            print(f"  action {ac.action.name}: match={ac.action_match}")

    if reward_info.reward <= 0.0:
        print("ERROR: reward is not > 0")
        return 1

    out = run.model_dump(mode="json")
    fixture_path = BENCHMARKS_DIR / "fidelity" / "fixtures" / "synthetic_task0_pass.run.json"
    with open(fixture_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"Fixture written: {fixture_path}")
    print(f"Messages: {len(out['messages'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
