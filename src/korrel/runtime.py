"""The simulation loop.

``run_scenario`` drives a scenario end to end: it emits the opening user
message, lets the adapter (the agent under test) respond, resolves any tool
calls against the scenario's mock tools, lets the Persona produce the next user
message, and repeats until the agent stops or ``max_turns`` is reached. The
transcript is then scored with the scenario's rubric.

Determinism: every run takes a seed and records the model and request
parameters. The seed pins scenario setup and any sampling Korrel controls. LLM
calls are not bit-reproducible; provider nondeterminism is outside the seed.
"""

from __future__ import annotations

import json
import random
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from .adapter import AgentAdapter
from .rubric import RubricResult
from .scenario import Scenario
from .types import Message, ToolCall


class Turn(BaseModel):
    """One round of the conversation: the messages added during it."""

    index: int
    messages: list[Message]
    stop_reason: Optional[str] = None


class ToolExecutionError(Exception):
    """A mock tool raised while resolving an agent tool call.

    Carries the failing tool's name, the original exception, and the partial
    transcript built up to the failure (system plus every message up to and
    including the assistant tool call that triggered it), so the run context
    is preserved exactly when it is most useful for debugging.
    """

    def __init__(
        self,
        tool_name: str,
        original: BaseException,
        transcript: "Transcript",
    ) -> None:
        self.tool_name = tool_name
        self.original = original
        self.transcript = transcript
        super().__init__(
            f"tool {tool_name!r} raised {type(original).__name__}: {original}"
        )


class Transcript(BaseModel):
    """The full record of a run."""

    messages: list[Message]
    turns: list[Turn]
    model: Optional[str] = None
    params: dict[str, Any] = {}
    seed: int = 0


class FailureCluster(BaseModel):
    """A minimal grouping of failures by function and a cheap signature.

    Semantic clustering is deferred to v0.2; this groups by failed rubric
    function plus a coarse score-bucket signature so the interface is stable.
    """

    function: str
    signature: str
    count: int


class RunResult(BaseModel):
    """The result of ``run_scenario``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transcript: Transcript
    score: float
    passed: bool
    failed_functions: list[str]
    clusters: list[FailureCluster]
    rubric_result: RubricResult
    # Count of model calls the simulation loop itself issued: one per agent
    # (adapter) invocation, one per user-simulator (persona) call. Exact for the
    # documented adapter_from_provider path (one provider call per invocation).
    # Excludes calls made inside a custom adapter's own internals (Korrel cannot
    # see them) and the judge's scoring-time call (it runs inside rubric.score,
    # outside the loop). Every counted call is billed to the user's provider key.
    model_calls: int = 0


def _tool_schemas(scenario: Scenario) -> list[dict[str, Any]]:
    return [tool.schema for tool in scenario.tools]


def _resolve_tool_call(
    call: ToolCall,
    tools_by_name: dict[str, Any],
    state: dict[str, Any],
) -> Message:
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        arguments = {}

    tool = tools_by_name.get(call.function.name)
    if tool is None:
        result: Any = {"error": f"unknown tool: {call.function.name}"}
    else:
        result = tool.call(arguments, state)

    content = result if isinstance(result, str) else json.dumps(result)
    return Message(role="tool", content=content, tool_call_id=call.id)


def _build_transcript(
    messages: list[Message],
    turns: list[Turn],
    user_sim: Any,
    run_seed: int,
) -> Transcript:
    model = getattr(user_sim, "model", None)
    params = getattr(user_sim, "params", {}) if hasattr(user_sim, "params") else {}
    return Transcript(
        messages=messages,
        turns=turns,
        model=model,
        params=params,
        seed=run_seed,
    )


def _cluster_failures(rubric_result: RubricResult) -> list[FailureCluster]:
    clusters: list[FailureCluster] = []
    for name in rubric_result.failed_functions:
        value = rubric_result.scores.get(name, 0.0)
        signature = "zero" if value <= 0.0 else "below_threshold"
        clusters.append(FailureCluster(function=name, signature=signature, count=1))
    return clusters


def run_scenario(
    scenario: Scenario,
    adapter: AgentAdapter,
    *,
    persona: Any = None,
    seed: Optional[int] = None,
) -> RunResult:
    """Run a scenario against an adapter and score the transcript.

    The transcript is scored with ``scenario.rubric`` if one is attached.
    ``persona`` overrides ``scenario.persona`` (used to inject a fake
    user-simulator in tests). ``seed`` overrides ``scenario.seed``.
    """

    run_seed = seed if seed is not None else scenario.seed
    user_sim = persona if persona is not None else scenario.persona
    # The RNG pins any sampling Korrel itself controls. Provider sampling is not
    # covered by the seed (documented nondeterminism).
    rng = random.Random(run_seed)  # noqa: F841 - reserved for Korrel-controlled sampling

    tools_by_name = {tool.name: tool for tool in scenario.tools}
    tool_schemas = _tool_schemas(scenario)
    state: dict[str, Any] = {}

    messages: list[Message] = []
    if scenario.system:
        messages.append(Message(role="system", content=scenario.system))

    turns: list[Turn] = []
    pending_user: Optional[Message] = Message(
        role="user", content=scenario.opening_message
    )

    # Count of model calls the loop issues (agent + user-simulator). Kept local
    # to this run and carried out on RunResult; no module-level state, so it
    # cannot leak across runs.
    model_calls = 0

    for turn_index in range(scenario.max_turns):
        turn_messages: list[Message] = []
        if pending_user is not None:
            messages.append(pending_user)
            turn_messages.append(pending_user)
            pending_user = None

        assistant = adapter(messages, tool_schemas)
        model_calls += 1
        messages.append(assistant)
        turn_messages.append(assistant)

        tool_round = 0
        turn_stop_reason: Optional[str] = None
        while assistant.tool_calls:
            if tool_round >= scenario.max_tool_rounds:
                turn_stop_reason = "max_tool_rounds"
                break
            for call in assistant.tool_calls:
                try:
                    tool_message = _resolve_tool_call(call, tools_by_name, state)
                except Exception as exc:
                    # A raising tool still fails the run (unchanged semantics),
                    # but the transcript built so far is attached instead of
                    # lost. The in-progress turn is included so the failing
                    # assistant tool call is visible.
                    partial_turns = turns + [
                        Turn(index=turn_index, messages=turn_messages)
                    ]
                    raise ToolExecutionError(
                        tool_name=call.function.name,
                        original=exc,
                        transcript=_build_transcript(
                            messages, partial_turns, user_sim, run_seed
                        ),
                    ) from exc
                messages.append(tool_message)
                turn_messages.append(tool_message)
            assistant = adapter(messages, tool_schemas)
            model_calls += 1
            messages.append(assistant)
            turn_messages.append(assistant)
            tool_round += 1

        turns.append(Turn(index=turn_index, messages=turn_messages, stop_reason=turn_stop_reason))

        if turn_index == scenario.max_turns - 1:
            break

        next_user = user_sim.next_message(messages)
        model_calls += 1
        if not next_user:
            break
        pending_user = Message(role="user", content=next_user)

    transcript = _build_transcript(messages, turns, user_sim, run_seed)

    if scenario.rubric is None:
        rubric_result = RubricResult(
            score=0.0, passed=False, scores={}, failed_functions=[]
        )
    else:
        rubric_result = scenario.rubric.score(messages, scenario.info)

    clusters = _cluster_failures(rubric_result)
    return RunResult(
        transcript=transcript,
        score=rubric_result.score,
        passed=rubric_result.passed,
        failed_functions=rubric_result.failed_functions,
        clusters=clusters,
        rubric_result=rubric_result,
        model_calls=model_calls,
    )
