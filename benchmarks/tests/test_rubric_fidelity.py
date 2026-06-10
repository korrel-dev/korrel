"""Tests for benchmarks/fidelity/scenario_retail.py rubric fidelity.

Covers:
- scenario object structure and required Korrel types
- tau2_retail_reward return value matches direct evaluate_simulation result
- Determinism: scoring the same transcript 20 times yields bit-identical floats
- Unknown termination_reason falls back to UNEXPECTED_ERROR (reward=0.0)
- Missing required keys in info raise KeyError
- Rubric wraps tau2_retail_reward with pass_threshold=1.0
"""

from __future__ import annotations

from typing import Any

import pytest

from tau2.data_model.message import AssistantMessage, UserMessage, ToolCall, ToolMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.domains.retail.environment import get_tasks as retail_get_tasks
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.modes import CommunicationMode

from fidelity._convert import tau2_messages_to_korrel
from fidelity.scenario_retail import scenario, tau2_retail_reward
from korrel.rubric import Rubric
from korrel.scenario import Scenario
from korrel.types import Message

from tests.fixtures.tau2_builders import make_simulation_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_TASKS = retail_get_tasks(task_split_name=None)
_TASK_BY_ID = {t.id: t for t in _ALL_TASKS}


def _direct_reward(
    task_id: str,
    messages: list[Any],
    termination_reason: TerminationReason,
) -> float:
    """Call evaluate_simulation directly and return the float reward."""
    run = make_simulation_run(task_id=task_id, messages=messages, termination_reason=termination_reason)
    reward_info = evaluate_simulation(
        simulation=run,
        task=_TASK_BY_ID[task_id],
        evaluation_type=EvaluationType.ALL_IGNORE_BASIS,
        solo_mode=False,
        domain="retail",
        mode=CommunicationMode.HALF_DUPLEX,
        env_kwargs=None,
    )
    return float(reward_info.reward)


def _rubric_reward(
    task_id: str,
    messages: list[Any],
    termination_reason: TerminationReason,
) -> float:
    """Score via the scenario rubric and return the float score."""
    korrel_msgs = tau2_messages_to_korrel(messages)
    info: dict[str, Any] = {
        "task_id": task_id,
        "termination_reason": termination_reason.value,
    }
    return tau2_retail_reward(korrel_msgs, info)


# ---------------------------------------------------------------------------
# Scenario structure
# ---------------------------------------------------------------------------


def test_scenario_is_korrel_scenario_instance() -> None:
    assert isinstance(scenario, Scenario)


def test_scenario_id() -> None:
    assert scenario.id == "tau2-retail-fidelity"


def test_scenario_rubric_is_rubric_instance() -> None:
    assert isinstance(scenario.rubric, Rubric)


def test_scenario_rubric_pass_threshold() -> None:
    """pass_threshold must be 1.0 (tau2 product-of-components)."""
    assert scenario.rubric.pass_threshold == 1.0


def test_scenario_rubric_contains_tau2_retail_reward() -> None:
    assert tau2_retail_reward in scenario.rubric.funcs


def test_scenario_has_stub_tool() -> None:
    assert len(scenario.tools) >= 1


def test_scenario_has_system_prompt() -> None:
    assert len(scenario.system) > 0


# ---------------------------------------------------------------------------
# Rubric reproduction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_id", ["0", "1", "2"])
def test_rubric_matches_direct_evaluate_simulation_agent_stop(task_id: str) -> None:
    """Rubric score == direct evaluate_simulation reward for AGENT_STOP."""
    messages = [
        UserMessage(role="user", content="I need help with my order."),
        AssistantMessage(role="assistant", content="I cannot assist with that."),
    ]
    expected = _direct_reward(task_id, messages, TerminationReason.AGENT_STOP)
    actual = _rubric_reward(task_id, messages, TerminationReason.AGENT_STOP)
    assert actual == expected


@pytest.mark.parametrize("task_id", ["0", "1"])
def test_rubric_matches_direct_evaluate_simulation_user_stop(task_id: str) -> None:
    """Rubric score == direct evaluate_simulation reward for USER_STOP."""
    messages = [
        UserMessage(role="user", content="Goodbye."),
        AssistantMessage(role="assistant", content="Have a great day."),
    ]
    expected = _direct_reward(task_id, messages, TerminationReason.USER_STOP)
    actual = _rubric_reward(task_id, messages, TerminationReason.USER_STOP)
    assert actual == expected


@pytest.mark.parametrize(
    "termination_reason",
    [
        TerminationReason.MAX_STEPS,
        TerminationReason.TIMEOUT,
        TerminationReason.UNEXPECTED_ERROR,
    ],
)
def test_non_stop_termination_yields_zero(
    termination_reason: TerminationReason,
) -> None:
    """evaluate_simulation returns 0.0 when termination is not AGENT_STOP or USER_STOP."""
    messages = [
        UserMessage(role="user", content="Hello."),
        AssistantMessage(role="assistant", content="Hi."),
    ]
    expected = _direct_reward("0", messages, termination_reason)
    actual = _rubric_reward("0", messages, termination_reason)
    assert actual == expected
    # The value is expected to be 0.0 for non-stop terminations.
    assert actual == 0.0


def test_rubric_reward_with_tool_call_sequence() -> None:
    """A trajectory with a tool call round-trips correctly and matches direct eval."""
    tc = ToolCall(
        id="tc1",
        name="get_order_details",
        arguments={"order_id": "ORD-999"},
        requestor="assistant",
    )
    messages = [
        UserMessage(role="user", content="Where is order ORD-999?"),
        AssistantMessage(role="assistant", content=None, tool_calls=[tc]),
        ToolMessage(id="tc1", role="tool", content="Order ORD-999 is in transit.", requestor="assistant"),
        AssistantMessage(role="assistant", content="Your order is in transit."),
    ]
    expected = _direct_reward("0", messages, TerminationReason.AGENT_STOP)
    actual = _rubric_reward("0", messages, TerminationReason.AGENT_STOP)
    assert actual == expected


def test_rubric_reward_unknown_termination_maps_to_unexpected_error() -> None:
    """Unknown termination_reason string falls back to UNEXPECTED_ERROR -> 0.0."""
    messages = [
        UserMessage(role="user", content="Hello."),
        AssistantMessage(role="assistant", content="Hi."),
    ]
    korrel_msgs = tau2_messages_to_korrel(messages)
    info: dict[str, Any] = {
        "task_id": "0",
        "termination_reason": "this_is_not_a_valid_reason",
    }
    reward = tau2_retail_reward(korrel_msgs, info)
    # evaluate_simulation with UNEXPECTED_ERROR should return 0.0
    assert reward == 0.0


def test_rubric_missing_task_id_raises_key_error() -> None:
    messages = [UserMessage(role="user", content="Hi")]
    korrel_msgs = tau2_messages_to_korrel(messages)
    with pytest.raises(KeyError):
        tau2_retail_reward(korrel_msgs, {"termination_reason": "agent_stop"})


def test_rubric_missing_termination_reason_raises_key_error() -> None:
    messages = [UserMessage(role="user", content="Hi")]
    korrel_msgs = tau2_messages_to_korrel(messages)
    with pytest.raises(KeyError):
        tau2_retail_reward(korrel_msgs, {"task_id": "0"})


def test_rubric_nonexistent_task_id_raises_value_error() -> None:
    messages = [UserMessage(role="user", content="Hi")]
    korrel_msgs = tau2_messages_to_korrel(messages)
    info: dict[str, Any] = {
        "task_id": "NONEXISTENT_TASK_9999",
        "termination_reason": "agent_stop",
    }
    with pytest.raises(ValueError, match="not found"):
        tau2_retail_reward(korrel_msgs, info)


# ---------------------------------------------------------------------------
# Rubric.score integration
# ---------------------------------------------------------------------------


def test_rubric_score_via_scenario_rubric_matches_direct_reward() -> None:
    """scenario.rubric.score returns a RubricResult whose .score matches direct eval."""
    messages = [
        UserMessage(role="user", content="Help please."),
        AssistantMessage(role="assistant", content="Sorry, cannot help."),
    ]
    korrel_msgs = tau2_messages_to_korrel(messages)
    info: dict[str, Any] = {
        "task_id": "0",
        "termination_reason": TerminationReason.AGENT_STOP.value,
    }
    direct = _direct_reward("0", messages, TerminationReason.AGENT_STOP)
    result = scenario.rubric.score(korrel_msgs, info)
    assert result.score == direct


def test_rubric_score_below_threshold_fails() -> None:
    """A trajectory that does not satisfy all criteria scores < 1.0 and fails."""
    messages = [
        UserMessage(role="user", content="I need help."),
        AssistantMessage(role="assistant", content="Sorry, I cannot help."),
    ]
    korrel_msgs = tau2_messages_to_korrel(messages)
    info: dict[str, Any] = {
        "task_id": "0",
        "termination_reason": TerminationReason.AGENT_STOP.value,
    }
    result = scenario.rubric.score(korrel_msgs, info)
    # A minimal unhelpful response always fails the reward criteria.
    assert result.score < 1.0
    assert result.passed is False


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_20_identical_runs() -> None:
    """Scoring the same transcript 20 times must produce bit-identical floats."""
    messages = [
        UserMessage(role="user", content="Can I return this?"),
        AssistantMessage(role="assistant", content="I cannot process returns."),
    ]
    korrel_msgs = tau2_messages_to_korrel(messages)
    info: dict[str, Any] = {
        "task_id": "0",
        "termination_reason": TerminationReason.AGENT_STOP.value,
    }
    scores = [tau2_retail_reward(korrel_msgs, info) for _ in range(20)]
    assert len(set(scores)) == 1, f"Non-deterministic scores: {set(scores)}"


def test_determinism_via_rubric_score_20_runs() -> None:
    """scenario.rubric.score is also deterministic over 20 calls."""
    messages = [
        UserMessage(role="user", content="What is my balance?"),
        AssistantMessage(role="assistant", content="I do not have that information."),
    ]
    korrel_msgs = tau2_messages_to_korrel(messages)
    info: dict[str, Any] = {
        "task_id": "0",
        "termination_reason": TerminationReason.AGENT_STOP.value,
    }
    scores = [scenario.rubric.score(korrel_msgs, info).score for _ in range(20)]
    assert len(set(scores)) == 1, f"Non-deterministic scores: {set(scores)}"
