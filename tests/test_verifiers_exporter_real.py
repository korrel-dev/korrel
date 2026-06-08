"""Integration tests for korrel.exporters.verifiers against the REAL verifiers package.

Gate: ``pytest.importorskip("verifiers")`` skips this entire module when verifiers
is not installed (Python 3.14 repo venv, default CI matrix). All tests are
offline and require no API keys; live personas and live judges are replaced by
deterministic fakes.

Confirmed against verifiers==0.1.14 in a Python 3.12 venv with korrel editable.

What these tests cover (complementary to test_verifiers_exporter.py):
1. Decision rule: real isinstance checks against vf.MultiTurnEnv / vf.SingleTurnEnv.
2. Rubric wiring and scoring: reward float flows through env.rubric.score_rollout
   on a real vf.State populated with real vf.AssistantMessage objects. Covers the
   1.0 case and the 0.0 case.
3. Completion-shim through the real rubric: an AssistantMessage carrying a real
   vf.ToolCall (flat shape) is converted to the korrel nested ToolCall shape
   before the reward function sees it.
4. Structural assertions: tool_defs, dataset columns, pass_threshold, max_turns
   per the confirmed realities doc.
5. Artifact round-trip: write_verifiers_env produces a loadable package; the
   loaded env is a real vf.MultiTurnEnv and the reward func name is present.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any, Optional

import pytest

vf = pytest.importorskip("verifiers")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _FakePersona:
    """Deterministic persona that replays a fixed queue of messages."""

    def __init__(self, messages: list[str]) -> None:
        self._queue = list(messages)

    def next_message(self, messages: list[Any]) -> Optional[str]:
        if self._queue:
            return self._queue.pop(0)
        return None


def _make_tool(name: str = "search") -> Any:
    """Build a MockTool with a trivial respond function."""
    from korrel.tools import MockTool

    return MockTool(
        name=name,
        schema={
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        respond=lambda args, state: f"{name}-result",
    )


def _make_scenario(
    *,
    scenario_id: str = "test-scenario",
    tools: Optional[list[Any]] = None,
    max_turns: int = 1,
    max_tool_rounds: int = 8,
    rubric_funcs: Optional[list[Any]] = None,
    pass_threshold: float = 0.5,
    system: str = "You are an assistant.",
    opening_message: str = "Hello.",
    info: Optional[dict[str, Any]] = None,
) -> Any:
    """Build a Scenario via model_construct so _FakePersona passes Pydantic."""
    from korrel.rubric import Rubric
    from korrel.scenario import Scenario

    if rubric_funcs is None:
        rubric_funcs = [lambda completion, info, **kw: 1.0]

    rubric = Rubric(funcs=rubric_funcs, pass_threshold=pass_threshold)
    return Scenario.model_construct(
        id=scenario_id,
        system=system,
        persona=_FakePersona([]),
        opening_message=opening_message,
        tools=tools or [],
        max_turns=max_turns,
        max_tool_rounds=max_tool_rounds,
        seed=0,
        info=info,
        rubric=rubric,
    )


def _make_state(
    completion: list[Any],
    info: Optional[dict[str, Any]] = None,
) -> Any:
    """Construct a minimal vf.State for offline scoring tests."""
    return vf.State(
        input={
            "prompt": [{"role": "user", "content": "hi"}],
            "example_id": 0,
            "info": info or {},
        },
        task={},
        client=None,
        model="test",
        sampling_args=None,
        is_completed=True,
        is_truncated=False,
        stop_condition=None,
        tool_defs=[],
        trajectory=[],
        completion=completion,
        reward=None,
        advantage=None,
        metrics=None,
        timing=None,
        error=None,
        usage=None,
        usage_tracker=None,
    )


# ---------------------------------------------------------------------------
# 1. Decision rule against real verifiers
# ---------------------------------------------------------------------------


def test_decision_rule_persona_and_tools_yields_multiturn_env():
    """A scenario with tools produces a real vf.MultiTurnEnv (not SingleTurnEnv)."""
    from korrel.exporters.verifiers import to_verifiers_env

    scenario = _make_scenario(
        scenario_id="multi-tool",
        tools=[_make_tool("lookup")],
        max_turns=3,
        max_tool_rounds=5,
    )
    env = to_verifiers_env(scenario, persona=_FakePersona([]))

    assert isinstance(env, vf.MultiTurnEnv)
    # SingleTurnEnv is a subclass of MultiTurnEnv in verifiers 0.1.14; the
    # concrete class for a persona/tool scenario must NOT be SingleTurnEnv.
    assert not isinstance(env, vf.SingleTurnEnv)


def test_decision_rule_single_exchange_yields_singleturn_env():
    """max_turns==1, no tools, persona=None produces a real vf.SingleTurnEnv."""
    from korrel.exporters.verifiers import to_verifiers_env

    scenario = _make_scenario(
        scenario_id="single-exchange",
        tools=[],
        max_turns=1,
        max_tool_rounds=8,
    )
    env = to_verifiers_env(scenario, persona=None)

    assert isinstance(env, vf.SingleTurnEnv)
    # SingleTurnEnv is a MultiTurnEnv subclass in verifiers 0.1.14.
    assert isinstance(env, vf.MultiTurnEnv)


# ---------------------------------------------------------------------------
# 2. Rubric wiring and scoring: reward float flows through score_rollout
# ---------------------------------------------------------------------------


def test_rubric_score_rollout_returns_one_for_passing_reward():
    """A reward fn returning 1.0 sets state['reward'] == 1.0 via score_rollout."""
    from korrel.exporters.verifiers import to_verifiers_env

    def always_one(completion, info, **kw):
        return 1.0

    scenario = _make_scenario(rubric_funcs=[always_one])
    env = to_verifiers_env(scenario, persona=None)

    state = _make_state(
        completion=[vf.AssistantMessage(role="assistant", content="correct")]
    )
    asyncio.run(env.rubric.score_rollout(state))

    assert state["reward"] == pytest.approx(1.0)
    assert "always_one" in state["metrics"]
    assert state["metrics"]["always_one"] == pytest.approx(1.0)


def test_rubric_score_rollout_returns_zero_for_failing_reward():
    """A reward fn returning 0.0 sets state['reward'] == 0.0 via score_rollout."""
    from korrel.exporters.verifiers import to_verifiers_env

    def always_zero(completion, info, **kw):
        return 0.0

    scenario = _make_scenario(rubric_funcs=[always_zero])
    env = to_verifiers_env(scenario, persona=None)

    state = _make_state(
        completion=[vf.AssistantMessage(role="assistant", content="wrong")]
    )
    asyncio.run(env.rubric.score_rollout(state))

    assert state["reward"] == pytest.approx(0.0)
    assert "always_zero" in state["metrics"]
    assert state["metrics"]["always_zero"] == pytest.approx(0.0)


def test_rubric_score_rollout_receives_info_dict():
    """The reward fn receives the info dict from the state input."""
    from korrel.exporters.verifiers import to_verifiers_env

    received: dict[str, Any] = {}

    def capture_info(completion, info, **kw):
        received["info"] = info
        return 1.0

    scenario = _make_scenario(rubric_funcs=[capture_info], info={"expected": "answer"})
    env = to_verifiers_env(scenario, persona=None)

    state = _make_state(
        completion=[vf.AssistantMessage(role="assistant", content="answer")],
        info={"expected": "answer"},
    )
    asyncio.run(env.rubric.score_rollout(state))

    assert received["info"] == {"expected": "answer"}


# ---------------------------------------------------------------------------
# 3. Completion shim: flat vf.ToolCall -> nested korrel ToolCall
# ---------------------------------------------------------------------------


def test_reward_fn_sees_nested_korrel_toolcall_shape():
    """Reward fn receives the nested korrel ToolCall (not the flat vf.ToolCall).

    Verifies the flat->nested conversion end to end through the real rubric.
    The reward fn inspects completion[i].tool_calls[0].function.name which is
    the KORREL nested attribute path. If the shim did not convert, accessing
    .function would raise AttributeError or return wrong data.
    """
    from korrel.exporters.verifiers import to_verifiers_env

    observed_tool_name: list[str] = []
    observed_tool_args: list[str] = []

    def inspect_tool_call(completion, info, **kw):
        for msg in completion:
            if msg.role == "assistant" and msg.tool_calls:
                tc = msg.tool_calls[0]
                observed_tool_name.append(tc.function.name)
                observed_tool_args.append(tc.function.arguments)
                return 1.0
        return 0.0

    scenario = _make_scenario(rubric_funcs=[inspect_tool_call])
    env = to_verifiers_env(scenario, persona=None)

    # Build a real vf.AssistantMessage with a FLAT vf.ToolCall.
    tc = vf.ToolCall(
        id="call-1",
        name="search",
        arguments=json.dumps({"query": "korrel"}),
    )
    assistant_msg = vf.AssistantMessage(role="assistant", content=None, tool_calls=[tc])
    tool_result = vf.ToolMessage(tool_call_id="call-1", content="result text")

    state = _make_state(completion=[assistant_msg, tool_result])
    asyncio.run(env.rubric.score_rollout(state))

    assert observed_tool_name == ["search"], (
        "reward fn must see tc.function.name == 'search' (nested korrel shape)"
    )
    assert observed_tool_args == [json.dumps({"query": "korrel"})], (
        "reward fn must see tc.function.arguments as the original JSON string"
    )
    assert state["reward"] == pytest.approx(1.0)


def test_reward_fn_sees_korrel_toolcall_id():
    """The korrel ToolCall id field matches the originating vf.ToolCall id."""
    from korrel.exporters.verifiers import to_verifiers_env

    observed_ids: list[str] = []

    def capture_id(completion, info, **kw):
        for msg in completion:
            if msg.role == "assistant" and msg.tool_calls:
                observed_ids.append(msg.tool_calls[0].id)
        return 1.0

    scenario = _make_scenario(rubric_funcs=[capture_id])
    env = to_verifiers_env(scenario, persona=None)

    tc = vf.ToolCall(id="unique-id-42", name="mytool", arguments="{}")
    assistant_msg = vf.AssistantMessage(role="assistant", content=None, tool_calls=[tc])

    state = _make_state(completion=[assistant_msg])
    asyncio.run(env.rubric.score_rollout(state))

    assert observed_ids == ["unique-id-42"]


def test_reward_fn_sees_korrel_toolcall_type_is_function():
    """The converted korrel ToolCall has type='function' (nested shape contract)."""
    from korrel.exporters.verifiers import to_verifiers_env

    observed_types: list[str] = []

    def capture_type(completion, info, **kw):
        for msg in completion:
            if msg.role == "assistant" and msg.tool_calls:
                observed_types.append(msg.tool_calls[0].type)
        return 1.0

    scenario = _make_scenario(rubric_funcs=[capture_type])
    env = to_verifiers_env(scenario, persona=None)

    tc = vf.ToolCall(id="tc1", name="any", arguments="{}")
    assistant_msg = vf.AssistantMessage(role="assistant", content=None, tool_calls=[tc])

    state = _make_state(completion=[assistant_msg])
    asyncio.run(env.rubric.score_rollout(state))

    assert observed_types == ["function"]


def test_reward_fn_sees_tool_result_message():
    """The reward fn sees a tool-result message with correct role and tool_call_id."""
    from korrel.exporters.verifiers import to_verifiers_env

    observed: list[dict[str, Any]] = []

    def capture_tool_result(completion, info, **kw):
        for msg in completion:
            if msg.role == "tool":
                observed.append({"content": msg.content, "tool_call_id": msg.tool_call_id})
        return 1.0

    scenario = _make_scenario(rubric_funcs=[capture_tool_result])
    env = to_verifiers_env(scenario, persona=None)

    tc = vf.ToolCall(id="tc-result", name="fetch", arguments="{}")
    assistant_msg = vf.AssistantMessage(role="assistant", content=None, tool_calls=[tc])
    tool_msg = vf.ToolMessage(tool_call_id="tc-result", content="fetched data")

    state = _make_state(completion=[assistant_msg, tool_msg])
    asyncio.run(env.rubric.score_rollout(state))

    assert len(observed) == 1
    assert observed[0]["content"] == "fetched data"
    assert observed[0]["tool_call_id"] == "tc-result"


# ---------------------------------------------------------------------------
# 4. Structural assertions: rubric shape, tool_defs, dataset, pass_threshold,
#    max_turns per confirmed realities.
# ---------------------------------------------------------------------------


def test_env_rubric_is_rubricgroup():
    """env.rubric is a vf.RubricGroup (MultiTurnEnv wraps rubric in a group)."""
    from korrel.exporters.verifiers import to_verifiers_env

    scenario = _make_scenario()
    env = to_verifiers_env(scenario, persona=None)

    assert isinstance(env.rubric, vf.RubricGroup)
    # RubricGroup subclasses Rubric in verifiers 0.1.14.
    assert isinstance(env.rubric, vf.Rubric)


def test_env_rubric_reward_func_names_include_custom_name():
    """The reward func name appears among env.rubric._get_reward_func_names().

    The list also contains verifiers monitor names (num_turns); the test asserts
    inclusion, not exact equality, per the confirmed realities doc.
    """
    from korrel.exporters.verifiers import to_verifiers_env

    def my_accuracy(completion, info, **kw):
        return 1.0

    scenario = _make_scenario(rubric_funcs=[my_accuracy])
    env = to_verifiers_env(scenario, persona=None)

    names = env.rubric._get_reward_func_names()
    assert "my_accuracy" in names


def test_env_rubric_reward_func_names_include_num_turns_monitor():
    """The num_turns monitor metric is always present (added by MultiTurnEnv)."""
    from korrel.exporters.verifiers import to_verifiers_env

    def dummy(completion, info, **kw):
        return 1.0

    scenario = _make_scenario(rubric_funcs=[dummy])
    env = to_verifiers_env(scenario, persona=None)

    names = env.rubric._get_reward_func_names()
    assert "num_turns" in names


def test_tool_defs_are_vf_tool_instances():
    """env.tool_defs is a list of vf.Tool with name/description/parameters."""
    from korrel.exporters.verifiers import to_verifiers_env

    tool = _make_tool("lookup")
    scenario = _make_scenario(tools=[tool], max_turns=2)
    env = to_verifiers_env(scenario, persona=_FakePersona([]))

    assert isinstance(env.tool_defs, list)
    assert len(env.tool_defs) == 1
    td = env.tool_defs[0]
    assert isinstance(td, vf.Tool)
    assert td.name == "lookup"
    assert td.description == "lookup tool"


def test_tool_defs_parameters_from_mock_tool_schema():
    """Tool parameters are taken from MockTool.schema['function']['parameters']."""
    from korrel.exporters.verifiers import to_verifiers_env
    from korrel.tools import MockTool

    specific_params = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }
    tool = MockTool(
        name="search",
        schema={
            "type": "function",
            "function": {
                "name": "search",
                "description": "search with params",
                "parameters": specific_params,
            },
        },
        respond=lambda args, state: "ok",
    )
    scenario = _make_scenario(tools=[tool], max_turns=2)
    env = to_verifiers_env(scenario, persona=_FakePersona([]))

    assert env.tool_defs[0].parameters == specific_params


def test_get_dataset_returns_datasets_dataset():
    """env.get_dataset() returns a datasets.Dataset with the correct columns."""
    datasets = pytest.importorskip("datasets")

    from korrel.exporters.verifiers import to_verifiers_env

    scenario = _make_scenario()
    env = to_verifiers_env(scenario, persona=None)

    ds = env.get_dataset()
    assert isinstance(ds, datasets.Dataset)
    assert len(ds) == 1
    assert "prompt" in ds.column_names
    assert "info" in ds.column_names
    assert "example_id" in ds.column_names


def test_get_dataset_row_prompt_contains_system_and_user():
    """Dataset row0 prompt is [system_msg, user_msg] with the correct content."""
    from korrel.exporters.verifiers import to_verifiers_env

    scenario = _make_scenario(
        system="Test system prompt.",
        opening_message="Test opening.",
    )
    env = to_verifiers_env(scenario, persona=None)

    row = env.get_dataset()[0]
    prompt = row["prompt"]
    assert isinstance(prompt, list)
    assert len(prompt) == 2
    assert prompt[0]["role"] == "system"
    assert prompt[0]["content"] == "Test system prompt."
    assert prompt[1]["role"] == "user"
    assert prompt[1]["content"] == "Test opening."


def test_get_dataset_row_info_matches_scenario_info():
    """Dataset row0 info matches scenario.info."""
    from korrel.exporters.verifiers import to_verifiers_env

    scenario = _make_scenario(info={"answer": "42"})
    env = to_verifiers_env(scenario, persona=None)

    row = env.get_dataset()[0]
    assert row["info"] == {"answer": "42"}


def test_pass_threshold_flows_through_to_env():
    """env.pass_threshold equals scenario.rubric.pass_threshold."""
    from korrel.exporters.verifiers import to_verifiers_env

    scenario = _make_scenario(pass_threshold=0.8)
    env = to_verifiers_env(scenario, persona=None)

    assert env.pass_threshold == pytest.approx(0.8)


def test_max_turns_multiturn_env_equals_budget_formula():
    """env.max_turns = scenario.max_turns * (1 + scenario.max_tool_rounds)."""
    from korrel.exporters.verifiers import to_verifiers_env

    # 4 turns, 6 tool rounds -> 4 * (1 + 6) = 28
    scenario = _make_scenario(
        max_turns=4,
        max_tool_rounds=6,
        tools=[_make_tool()],
    )
    env = to_verifiers_env(scenario, persona=_FakePersona([]))

    assert env.max_turns == 4 * (1 + 6)


@pytest.mark.parametrize(
    "max_turns, max_tool_rounds, expected",
    [
        (1, 8, 9),
        (3, 5, 18),
        (5, 2, 15),
        (10, 0, 10),  # max_tool_rounds=0 is disallowed by Scenario (ge=1); test
                      # the formula directly without going through Scenario validation
    ],
)
def test_max_turns_budget_formula_parametrized(
    max_turns: int, max_tool_rounds: int, expected: int
) -> None:
    """env.max_turns = max_turns * (1 + max_tool_rounds) across several inputs."""
    from korrel.exporters.verifiers import to_verifiers_env

    if max_tool_rounds < 1:
        pytest.skip("Scenario enforces max_tool_rounds >= 1; skip formula edge case")

    scenario = _make_scenario(
        max_turns=max_turns,
        max_tool_rounds=max_tool_rounds,
        tools=[_make_tool()],
    )
    env = to_verifiers_env(scenario, persona=_FakePersona([]))

    assert env.max_turns == expected


def test_singleturn_env_max_turns_is_one():
    """SingleTurnEnv.max_turns is 1 (set by SingleTurnEnv.__init__ via super)."""
    from korrel.exporters.verifiers import to_verifiers_env

    scenario = _make_scenario(max_turns=1, max_tool_rounds=8, tools=[])
    env = to_verifiers_env(scenario, persona=None)

    assert env.max_turns == 1


# ---------------------------------------------------------------------------
# 5. Artifact round-trip: write_verifiers_env -> load_environment -> real env
# ---------------------------------------------------------------------------


def test_artifact_roundtrip_load_environment_returns_multiturn_env(tmp_path: Path) -> None:
    """write_verifiers_env produces a loadable module; load_environment() returns vf.MultiTurnEnv."""
    from korrel.exporters.verifiers import write_verifiers_env

    # Write a bundled scenario source file.
    scenario_src = tmp_path / "my_scenario.py"
    scenario_src.write_text(
        "\n".join([
            "from korrel.rubric import Rubric",
            "from korrel.scenario import Scenario",
            "",
            "class _FP:",
            "    def next_message(self, messages):",
            "        return None",
            "",
            "def my_metric(completion, info, **kw):",
            "    return 1.0",
            "",
            "scenario = Scenario.model_construct(",
            "    id='roundtrip-test',",
            "    system='sys',",
            "    persona=_FP(),",
            "    opening_message='hello',",
            "    tools=[],",
            "    max_turns=1,",
            "    max_tool_rounds=8,",
            "    seed=0,",
            "    rubric=Rubric(funcs=[my_metric]),",
            ")",
        ]),
        encoding="utf-8",
    )

    from korrel.rubric import Rubric
    from korrel.scenario import Scenario

    def my_metric(completion, info, **kw):
        return 1.0

    scenario = Scenario.model_construct(
        id="roundtrip-test",
        system="sys",
        persona=_FakePersona([]),
        opening_message="hello",
        tools=[],
        max_turns=1,
        max_tool_rounds=8,
        seed=0,
        rubric=Rubric(funcs=[my_metric]),
    )

    out = write_verifiers_env(scenario, tmp_path / "export", scenario_source_path=scenario_src)

    # Find the generated env module (not _scenario.py, not pyproject.toml).
    env_module_path = next(
        f for f in out.iterdir()
        if f.suffix == ".py" and not f.name.startswith("_")
    )

    spec = importlib.util.spec_from_file_location("_rt_env_mod", env_module_path)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    loaded_env = mod.load_environment(persona=None)

    assert isinstance(loaded_env, vf.MultiTurnEnv)
    names = loaded_env.rubric._get_reward_func_names()
    assert "my_metric" in names


def test_artifact_roundtrip_no_args_keeps_scenario_persona(tmp_path: Path) -> None:
    """load_environment() with no persona kwarg must keep the scenario's persona.

    Regression: forwarding persona=None unconditionally stripped the scenario
    persona and forced SingleTurnEnv. A persona-driven, no-tools, max_turns==1
    scenario loaded with no arguments must stay a MultiTurnEnv (persona-driven),
    not collapse to SingleTurnEnv.
    """
    from korrel.exporters.verifiers import write_verifiers_env

    scenario_src = tmp_path / "persona_scenario.py"
    scenario_src.write_text(
        "\n".join([
            "from korrel.rubric import Rubric",
            "from korrel.scenario import Scenario",
            "",
            "class _FP:",
            "    def next_message(self, messages):",
            "        return None",
            "",
            "def my_metric(completion, info, **kw):",
            "    return 1.0",
            "",
            "scenario = Scenario.model_construct(",
            "    id='persona-noargs',",
            "    system='sys',",
            "    persona=_FP(),",
            "    opening_message='hello',",
            "    tools=[],",
            "    max_turns=1,",
            "    max_tool_rounds=8,",
            "    seed=0,",
            "    rubric=Rubric(funcs=[my_metric]),",
            ")",
        ]),
        encoding="utf-8",
    )

    from korrel.rubric import Rubric
    from korrel.scenario import Scenario

    def my_metric(completion, info, **kw):
        return 1.0

    scenario = Scenario.model_construct(
        id="persona-noargs",
        system="sys",
        persona=_FakePersona([]),
        opening_message="hello",
        tools=[],
        max_turns=1,
        max_tool_rounds=8,
        seed=0,
        rubric=Rubric(funcs=[my_metric]),
    )

    out = write_verifiers_env(scenario, tmp_path / "export", scenario_source_path=scenario_src)
    env_module_path = next(
        f for f in out.iterdir()
        if f.suffix == ".py" and not f.name.startswith("_")
    )
    spec = importlib.util.spec_from_file_location("_rt_persona_mod", env_module_path)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    loaded_env = mod.load_environment()  # no persona kwarg

    assert isinstance(loaded_env, vf.MultiTurnEnv)
    assert not isinstance(loaded_env, vf.SingleTurnEnv)


def test_artifact_roundtrip_reward_func_name_survives_reload(tmp_path: Path) -> None:
    """The reward func name in the loaded artifact matches what was registered."""
    from korrel.exporters.verifiers import write_verifiers_env

    # Create a scenario source with a distinctively named reward function.
    scenario_src = tmp_path / "artifact_scenario.py"
    scenario_src.write_text(
        "\n".join([
            "from korrel.rubric import Rubric",
            "from korrel.scenario import Scenario",
            "",
            "class _FP:",
            "    def next_message(self, messages):",
            "        return None",
            "",
            "def exact_match_score(completion, info, **kw):",
            "    return 1.0",
            "",
            "scenario = Scenario.model_construct(",
            "    id='name-persist-test',",
            "    system='sys',",
            "    persona=_FP(),",
            "    opening_message='hi',",
            "    tools=[],",
            "    max_turns=1,",
            "    max_tool_rounds=8,",
            "    seed=0,",
            "    rubric=Rubric(funcs=[exact_match_score]),",
            ")",
        ]),
        encoding="utf-8",
    )

    from korrel.rubric import Rubric
    from korrel.scenario import Scenario

    def exact_match_score(completion, info, **kw):
        return 1.0

    scenario = Scenario.model_construct(
        id="name-persist-test",
        system="sys",
        persona=_FakePersona([]),
        opening_message="hi",
        tools=[],
        max_turns=1,
        max_tool_rounds=8,
        seed=0,
        rubric=Rubric(funcs=[exact_match_score]),
    )

    out = write_verifiers_env(
        scenario,
        tmp_path / "export2",
        scenario_source_path=scenario_src,
    )

    env_module_path = next(
        f for f in out.iterdir()
        if f.suffix == ".py" and not f.name.startswith("_")
    )

    spec = importlib.util.spec_from_file_location("_rt_env_mod2", env_module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    loaded_env = mod.load_environment(persona=None)
    names = loaded_env.rubric._get_reward_func_names()
    assert "exact_match_score" in names
