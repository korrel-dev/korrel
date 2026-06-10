"""Korrel Scenario for the tau2 retail fidelity benchmark.

The Rubric reward function reproduces tau2's gold reward by construction:
it converts korrel canonical messages back to tau2 message objects,
reconstructs the SimulationRun fields the evaluator needs from `info`,
looks up the tau2 Task by task_id from the pinned tau2 data, and calls
tau2's evaluate_simulation with EvaluationType.ALL_IGNORE_BASIS.

Fidelity claim: for every frozen transcript in transcripts/retail/,
  scenario.rubric.score(completion, info).score == gold_reward
to exact float equality.

Source pins confirmed on 2026-06-10:
  tau2-bench v1.0.0 @ 17e07b1da2bbc0cadfddeea36412686e0604127b
  evaluate_simulation signature: (simulation, task, evaluation_type,
    solo_mode, domain, mode=HALF_DUPLEX, env_kwargs=None) -> RewardInfo

Evaluation type: EvaluationType.ALL_IGNORE_BASIS
  Reason: 38 of 40 benchmark tasks have NL_ASSERTION in reward_basis.
  EvaluationType.ALL raises ValueError when NL_ASSERTION is in the basis
  but is not being evaluated (confirmed from evaluator.py source).
  EvaluationType.ALL_WITH_NL_ASSERTIONS requires a live LLM call (WIP).
  EvaluationType.ALL_IGNORE_BASIS multiplies ENV*ACTION*COMMUNICATE rewards
  unconditionally, ignoring each task's declared reward_basis. This is the
  correct choice for offline, no-key fidelity verification.
  Caveat: ALL_IGNORE_BASIS omits the NL_ASSERTION component. The fidelity
  claim covers only the structural (DB, action, communicate) components.
  This caveat is documented in the README and in run_fidelity.py output.

All tau2 imports are deferred inside the reward function so this module
can be imported in the korrel root env without tau2 installed; the reward
function itself requires tau2 (benchmarks env only).
"""

from __future__ import annotations

from typing import Any

from korrel.persona import Persona
from korrel.rubric import Rubric
from korrel.scenario import Scenario
from korrel.tools import MockTool
from korrel.types import Message

from ._convert import korrel_messages_to_tau2


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def tau2_retail_reward(
    completion: list[Message],
    info: dict[str, Any],
    **kwargs: Any,
) -> float:
    """Reproduce tau2's gold reward for a retail task trajectory.

    Parameters
    ----------
    completion:
        Korrel canonical messages for the trajectory.  These must be the
        full half-duplex message list originally produced by tau2's runner
        and stored in SimulationRun.messages.  The fidelity harness
        populates this from the frozen *.run.json via tau2_messages_to_korrel.

    info:
        Metadata dict carrying the non-message fields the evaluator needs.
        Required keys:
          "task_id"            : str   -- retail task id (e.g. "0")
          "termination_reason" : str   -- tau2 TerminationReason value
                                         (e.g. "agent_stop")
        Optional keys (passed through to SimulationRun constructor for
        completeness; not read by evaluate_simulation directly):
          "run_id"             : str   -- original SimulationRun.id
          "seed"               : int   -- seed used for generation
          "domain"             : str   -- "retail" (default if absent)

    Returns
    -------
    float
        The reward from tau2.evaluator.evaluator.evaluate_simulation with
        EvaluationType.ALL_IGNORE_BASIS, domain="retail", solo_mode=False,
        mode=HALF_DUPLEX.  Returns 0.0 if termination_reason is not in
        {AGENT_STOP, USER_STOP} (matching evaluate_simulation's own check).
    """
    # Deferred tau2 imports: tau2 is a benchmarks-only dependency.
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.modes import CommunicationMode

    task_id: str = info["task_id"]
    termination_reason_str: str = info["termination_reason"]

    # Resolve TerminationReason; fall back to UNEXPECTED_ERROR if unknown.
    try:
        termination_reason = TerminationReason(termination_reason_str)
    except ValueError:
        termination_reason = TerminationReason.UNEXPECTED_ERROR

    # Look up the tau2 Task from the pinned retail data.
    task = _load_retail_task(task_id)

    # Reconstruct tau2 messages from korrel canonical messages.
    tau2_msgs = korrel_messages_to_tau2(completion)

    # Build a minimal SimulationRun sufficient for evaluate_simulation.
    # Fields not used by the evaluator are given placeholder values.
    run = SimulationRun(
        id=info.get("run_id", "fidelity-harness"),
        task_id=task_id,
        start_time="1970-01-01T00:00:00",
        end_time="1970-01-01T00:00:01",
        duration=1.0,
        termination_reason=termination_reason,
        messages=tau2_msgs,
        mode=CommunicationMode.HALF_DUPLEX.value,
        seed=info.get("seed", 300),
    )

    # EvaluationType.ALL_IGNORE_BASIS: multiplies ENV*ACTION*COMMUNICATE
    # rewards unconditionally, without requiring NL_ASSERTION (LLM-based, WIP).
    # Rationale: 38/40 benchmark tasks have NL_ASSERTION in reward_basis;
    # EvaluationType.ALL raises ValueError for those tasks when NL_ASSERTION
    # is not evaluated. ALL_IGNORE_BASIS is the correct offline type.
    # Caveat: the NL_ASSERTION component is not scored here.
    reward_info = evaluate_simulation(
        simulation=run,
        task=task,
        evaluation_type=EvaluationType.ALL_IGNORE_BASIS,
        solo_mode=False,
        domain="retail",
        mode=CommunicationMode.HALF_DUPLEX,
        env_kwargs=None,
    )
    return float(reward_info.reward)


def _load_retail_task(task_id: str) -> Any:
    """Load a retail Task by id from the pinned tau2 data.

    Raises ValueError if the task is not found.
    """
    from tau2.domains.retail.environment import get_tasks as retail_get_tasks

    # get_tasks returns all tasks in the base split when split_name="base".
    # We iterate to find the matching id.
    tasks = retail_get_tasks(task_split_name=None)  # all 114 tasks
    for task in tasks:
        if task.id == task_id:
            return task
    raise ValueError(
        f"Retail task '{task_id}' not found in tau2 data. "
        "Ensure tau2 is installed from the pinned SHA and TAU2_DATA_DIR is "
        "set (or tau2 was installed from the git source pin which ships data/)."
    )


# ---------------------------------------------------------------------------
# Korrel Scenario
# ---------------------------------------------------------------------------
#
# The Persona and MockTool definitions below are minimal stubs; they satisfy
# the Korrel Scenario constructor but are not exercised during the fidelity
# check (which replays frozen transcripts, not live LLM calls).
#
# The rubric is the load-bearing component: it wraps tau2_retail_reward with
# a pass threshold of 1.0 (since tau2's product-of-components reward only
# reaches 1.0 when all required criteria are met).

_STUB_PERSONA = Persona(
    goal=(
        "You are a customer contacting a retail support agent. "
        "Follow the task instructions provided in `info`."
    ),
    behavior="",
)

_STUB_TOOL = MockTool(
    name="noop",
    schema={
        "type": "function",
        "function": {
            "name": "noop",
            "description": "Placeholder; not called during fidelity replay.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    respond=lambda args, state: "noop",
)

scenario = Scenario(
    id="tau2-retail-fidelity",
    system=(
        "You are a retail customer support agent. Use the available tools to "
        "help the customer with their request. When you are done, end the "
        "conversation."
    ),
    persona=_STUB_PERSONA,
    opening_message="Hello, I need help with my order.",
    tools=[_STUB_TOOL],
    rubric=Rubric(
        funcs=[tau2_retail_reward],
        # pass threshold = 1.0: tau2's reward is a product of components;
        # any partial failure yields reward < 1.0.
        pass_threshold=1.0,
    ),
)
