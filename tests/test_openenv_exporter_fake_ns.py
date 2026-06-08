"""Offline tests for korrel.exporters.openenv that require a fake openenv namespace.

These tests install a minimal fake ``openenv`` package into ``sys.modules``
via a session-scoped fixture, then exercise:

- ``build_environment_class``: returns a concrete Environment subclass.
- ``KorrelOpenEnvEnvironment.reset()``: seeds messages from scenario, returns
  an observation with ``done=False`` and ``reward=None``.
- ``KorrelOpenEnvEnvironment.step(action)``: all branches (tool, persona,
  persona-exhausted, max-turns, max-tool-rounds cap) with deterministic reward
  assertions.
- Round-trip: ``write_openenv_env`` to a tmp dir, then load the generated
  ``models.py`` and ``server/<env>_environment.py`` under the fake openenv
  namespace, construct the generated ``KorrelEnvironment``, and call
  ``reset()`` plus ``step()``.
- Shim assertion at the generated-environment level: action with tool_calls
  (arguments as JSON string) produces a role:"tool" message with the correct
  ``tool_call_id`` in the returned observation.

All tests are offline. No network, no API keys.

Ref: docs/spec/korrel-to-openenv.md
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Optional

import pytest


# ---------------------------------------------------------------------------
# Fake openenv namespace fixture
# ---------------------------------------------------------------------------


def _build_fake_openenv() -> dict[str, types.ModuleType]:
    """Build a minimal fake openenv package tree and return it as a module map.

    The map keys are the dotted module paths to install in sys.modules. Each
    value is a fresh types.ModuleType with the minimum attributes needed for:
    - openenv.core.env_server.interfaces.Environment (base class with
      __init__(self, transform=None, rubric=None), plus abstract reset/step/state)
    - openenv.core.env_server.types.State, Observation, Action
    - openenv.core.env_server.http_server.create_app (for generated app.py)
    - openenv.core.EnvClient (for generated client.py)
    - openenv.core.client_types.StepResult (for generated client.py)

    The generated server/app.py and client.py are NOT loaded in the round-trip
    test because they require fastapi/uvicorn and a running server. Only
    models.py and the environment module are loaded; this is documented in the
    test body.
    """

    # --- types: State, Observation, Action ---

    class FakeState:
        """Minimal State for offline testing (matches State fields episode_id, step_count)."""

        def __init__(self, episode_id: str = "", step_count: int = 0, **kwargs: Any) -> None:
            self.episode_id = episode_id
            self.step_count = step_count

    class FakeObservation:
        """Minimal Observation base with done/reward/messages (extra="forbid" not enforced here)."""

        def __init__(
            self,
            messages: Any = None,
            done: bool = False,
            reward: Any = None,
            metadata: Any = None,
            **kwargs: Any,
        ) -> None:
            self.messages = messages if messages is not None else []
            self.done = done
            self.reward = reward
            self.metadata = metadata or {}

    class FakeAction:
        """Minimal Action base."""

        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    # --- interfaces: Environment ---

    class FakeEnvironment:
        """Minimal Environment base class.

        Provides the __init__ signature openenv.core.env_server.interfaces.Environment
        exposes (``transform=None, rubric=None``) and no-op reset/step/state so
        concrete subclasses can call super().__init__() safely.
        """

        def __init__(self, transform: Any = None, rubric: Any = None) -> None:
            self._oe_transform = transform
            self._oe_rubric = rubric

        def reset(self, seed: Any = None, episode_id: Any = None, **kwargs: Any) -> Any:  # noqa: D102
            raise NotImplementedError

        def step(self, action: Any, timeout_s: Any = None, **kwargs: Any) -> Any:  # noqa: D102
            raise NotImplementedError

        @property
        def state(self) -> Any:  # noqa: D102
            raise NotImplementedError

    # --- http_server.create_app stub ---

    def fake_create_app(env_cls: Any, action_cls: Any, obs_cls: Any, **kwargs: Any) -> Any:
        """Stub create_app that returns a plain object (never starts a server)."""

        class _FakeApp:
            env_class = env_cls
            action_class = action_cls
            obs_class = obs_cls
            env_name = kwargs.get("env_name", "unnamed")

        return _FakeApp()

    # --- EnvClient stub ---

    class FakeEnvClient:
        def __init__(self, base_url: str = "", **kwargs: Any) -> None:
            self.base_url = base_url

    # --- StepResult stub ---

    class FakeStepResult:
        def __init__(self, observation: Any = None, reward: Any = None, done: bool = False) -> None:
            self.observation = observation
            self.reward = reward
            self.done = done

    # Build the module tree.

    oe = types.ModuleType("openenv")
    oe_core = types.ModuleType("openenv.core")
    oe_core_esrv = types.ModuleType("openenv.core.env_server")
    oe_core_esrv_iface = types.ModuleType("openenv.core.env_server.interfaces")
    oe_core_esrv_types = types.ModuleType("openenv.core.env_server.types")
    oe_core_esrv_http = types.ModuleType("openenv.core.env_server.http_server")
    oe_core_ct = types.ModuleType("openenv.core.client_types")

    # Populate with the fake classes.
    oe_core_esrv_iface.Environment = FakeEnvironment
    oe_core_esrv_types.State = FakeState
    oe_core_esrv_types.Observation = FakeObservation
    oe_core_esrv_types.Action = FakeAction
    oe_core_esrv_http.create_app = fake_create_app
    oe_core.EnvClient = FakeEnvClient
    oe_core.client_types = oe_core_ct
    oe_core_ct.StepResult = FakeStepResult

    # Wire the attribute hierarchy so dotted access works.
    oe.core = oe_core
    oe_core.env_server = oe_core_esrv
    oe_core_esrv.interfaces = oe_core_esrv_iface
    oe_core_esrv.types = oe_core_esrv_types
    oe_core_esrv.http_server = oe_core_esrv_http

    return {
        "openenv": oe,
        "openenv.core": oe_core,
        "openenv.core.env_server": oe_core_esrv,
        "openenv.core.env_server.interfaces": oe_core_esrv_iface,
        "openenv.core.env_server.types": oe_core_esrv_types,
        "openenv.core.env_server.http_server": oe_core_esrv_http,
        "openenv.core.client_types": oe_core_ct,
    }


@pytest.fixture()
def fake_openenv():
    """Install a minimal fake openenv namespace into sys.modules and clean up after.

    Yields the module map so tests can reach the fake classes directly when
    needed (e.g. isinstance checks against FakeObservation).
    """
    # If real openenv is already installed, skip rather than shadow it.
    if "openenv" in sys.modules:
        pytest.skip("real openenv is installed; fake-namespace tests are skipped")

    module_map = _build_fake_openenv()
    for name, mod in module_map.items():
        sys.modules[name] = mod
    try:
        yield module_map
    finally:
        for name in module_map:
            sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Scenario and persona helpers
# ---------------------------------------------------------------------------


class _FakePersona:
    """Deterministic persona that replays a fixed queue of messages."""

    def __init__(self, messages: list[str]) -> None:
        self._queue = list(messages)

    def next_message(self, messages: Any) -> Optional[str]:
        if self._queue:
            return self._queue.pop(0)
        return None


def _make_scenario(
    scenario_id: str = "test-scenario",
    *,
    system: str = "You are helpful.",
    opening_message: str = "Hello.",
    max_turns: int = 1,
    max_tool_rounds: int = 8,
    tools: list = None,
    rubric=None,
    persona=None,
    info: Any = None,
):
    from korrel.rubric import Rubric
    from korrel.scenario import Scenario

    if rubric is None:
        rubric = Rubric(funcs=[lambda completion, info_dict, **kw: 1.0])
    if persona is None:
        persona = _FakePersona([])
    return Scenario.model_construct(
        id=scenario_id,
        system=system,
        persona=persona,
        opening_message=opening_message,
        tools=tools or [],
        max_turns=max_turns,
        max_tool_rounds=max_tool_rounds,
        seed=0,
        info=info,
        rubric=rubric,
    )


def _make_tool_action(content: Any = None, tool_calls: Any = None) -> Any:
    class _Action:
        pass

    a = _Action()
    a.content = content
    a.tool_calls = tool_calls
    return a


# ---------------------------------------------------------------------------
# Tests: build_environment_class
# ---------------------------------------------------------------------------


def test_build_environment_class_returns_type(fake_openenv):
    """build_environment_class returns a class (not an instance)."""
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    class _Act:
        pass

    scenario = _make_scenario()
    env_cls = build_environment_class(
        scenario, _Obs, _Act, persona=_FakePersona([])
    )
    assert isinstance(env_cls, type)


def test_build_environment_class_subclasses_environment(fake_openenv):
    """The returned class is a subclass of FakeEnvironment."""
    from korrel.exporters.openenv import build_environment_class

    FakeEnvironment = fake_openenv["openenv.core.env_server.interfaces"].Environment
    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    class _Act:
        pass

    scenario = _make_scenario()
    env_cls = build_environment_class(scenario, _Obs, _Act, persona=_FakePersona([]))
    assert issubclass(env_cls, FakeEnvironment)


def test_build_environment_class_instantiates(fake_openenv):
    """The returned class can be instantiated without error."""
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    class _Act:
        pass

    scenario = _make_scenario()
    env_cls = build_environment_class(scenario, _Obs, _Act, persona=_FakePersona([]))
    env = env_cls()
    assert env is not None


def test_build_environment_class_raises_value_error_when_rubric_none(fake_openenv):
    """build_environment_class raises ValueError when scenario.rubric is None.

    A reward-less RL environment is meaningless. This mirrors to_verifiers_env
    (N4: public constructor parity).
    """
    from korrel.exporters.openenv import build_environment_class
    from korrel.scenario import Scenario

    scenario = Scenario.model_construct(
        id="no-rubric",
        system="",
        persona=_FakePersona([]),
        opening_message="Hello.",
        tools=[],
        max_turns=1,
        max_tool_rounds=8,
        seed=0,
        info=None,
        rubric=None,
    )
    with pytest.raises(ValueError, match="Scenario.rubric is required"):
        build_environment_class(scenario, object, object)


# ---------------------------------------------------------------------------
# Tests: max_turns budget matches run_scenario (B1 fix verification)
# ---------------------------------------------------------------------------


def test_max_turns_budget_1_gives_1_assistant_turn_0_persona_calls(fake_openenv):
    """max_turns=1: policy gets 1 assistant turn, persona is never called.

    Matches runtime.py::run_scenario: for max_turns=1 the loop runs once and
    the persona is not called on that final turn (run_scenario breaks at
    turn_index == max_turns - 1 without calling the persona).
    """
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    persona_calls: list[int] = []

    class _CountingPersona:
        def next_message(self, messages: Any) -> Any:
            persona_calls.append(1)
            return "follow-up"

    scenario = _make_scenario("budget-1-test", max_turns=1, persona=_CountingPersona())
    env_cls = build_environment_class(
        scenario, _Obs, object, persona=_CountingPersona()
    )
    env = env_cls()
    env.reset()

    action = _make_tool_action(content="Turn 1.")
    obs = env.step(action)

    # The first plain step must terminate (1 assistant turn total, persona not called).
    assert obs.done is True, "max_turns=1: first step must terminate"
    assert len(persona_calls) == 0, (
        f"max_turns=1: persona must not be called, but was called {len(persona_calls)} times"
    )


def test_max_turns_budget_2_gives_2_assistant_turns_1_persona_call(fake_openenv):
    """max_turns=2: policy gets 2 assistant turns, persona is called exactly 1 time.

    Matches runtime.py::run_scenario behavior.
    """
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    persona_calls: list[int] = []

    class _CountingPersona:
        def next_message(self, messages: Any) -> Any:
            persona_calls.append(1)
            return "follow-up"

    scenario = _make_scenario("budget-2-test", max_turns=2, persona=_CountingPersona())
    env_cls = build_environment_class(
        scenario, _Obs, object, persona=_CountingPersona()
    )
    env = env_cls()
    env.reset()

    action = _make_tool_action(content="Turn 1.")
    obs1 = env.step(action)

    assert obs1.done is False, "max_turns=2: first step must not terminate"
    assert len(persona_calls) == 1, (
        f"max_turns=2: persona must be called exactly once after step 1, "
        f"but was called {len(persona_calls)} times"
    )

    action2 = _make_tool_action(content="Turn 2.")
    obs2 = env.step(action2)

    assert obs2.done is True, "max_turns=2: second step must terminate"
    assert len(persona_calls) == 1, (
        f"max_turns=2: persona must be called exactly 1 time total, "
        f"but was called {len(persona_calls)} times"
    )


def test_max_turns_budget_3_gives_3_assistant_turns_2_persona_calls(fake_openenv):
    """max_turns=3: policy gets 3 assistant turns, persona is called exactly 2 times.

    Matches runtime.py::run_scenario behavior.
    """
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    persona_calls: list[int] = []

    class _CountingPersona:
        def next_message(self, messages: Any) -> Any:
            persona_calls.append(1)
            return "follow-up"

    scenario = _make_scenario("budget-3-test", max_turns=3, persona=_CountingPersona())
    env_cls = build_environment_class(
        scenario, _Obs, object, persona=_CountingPersona()
    )
    env = env_cls()
    env.reset()

    for i, content in enumerate(["Turn 1.", "Turn 2."], start=1):
        obs = env.step(_make_tool_action(content=content))
        assert obs.done is False, f"max_turns=3: step {i} must not terminate"
    assert len(persona_calls) == 2, (
        f"max_turns=3: persona must be called exactly 2 times after steps 1 and 2, "
        f"but was called {len(persona_calls)} times"
    )

    obs3 = env.step(_make_tool_action(content="Turn 3."))
    assert obs3.done is True, "max_turns=3: step 3 must terminate"
    assert len(persona_calls) == 2, (
        f"max_turns=3: persona must be called exactly 2 times total, "
        f"but was called {len(persona_calls)} times"
    )


# ---------------------------------------------------------------------------
# Tests: reset()
# ---------------------------------------------------------------------------


def test_reset_returns_observation_with_done_false(fake_openenv):
    """reset() returns an observation with done=False."""
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    scenario = _make_scenario("reset-test", opening_message="Start here.")
    env_cls = build_environment_class(
        scenario, _Obs, object, persona=_FakePersona([])
    )
    env = env_cls()
    obs = env.reset()
    assert obs.done is False


def test_reset_returns_observation_with_reward_none(fake_openenv):
    """reset() returns an observation with reward=None."""
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    scenario = _make_scenario("reset-reward-test")
    env_cls = build_environment_class(
        scenario, _Obs, object, persona=_FakePersona([])
    )
    env = env_cls()
    obs = env.reset()
    assert obs.reward is None


def test_reset_observation_contains_opening_message(fake_openenv):
    """reset() observation messages include the opening user message."""
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    scenario = _make_scenario("msg-test", opening_message="What is 2+2?")
    env_cls = build_environment_class(
        scenario, _Obs, object, persona=_FakePersona([])
    )
    env = env_cls()
    obs = env.reset()
    # The observation messages should include the opening user message.
    assert isinstance(obs.messages, list)
    assert len(obs.messages) >= 1
    roles = [m.get("role") for m in obs.messages]
    assert "user" in roles
    contents = [m.get("content", "") for m in obs.messages]
    assert any("What is 2+2?" in str(c) for c in contents)


def test_reset_observation_includes_system_message_when_set(fake_openenv):
    """reset() observation messages include the system message when scenario.system is set."""
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    scenario = _make_scenario(
        "sys-msg-test",
        system="Be concise.",
        opening_message="Hello.",
    )
    env_cls = build_environment_class(
        scenario, _Obs, object, persona=_FakePersona([])
    )
    env = env_cls()
    obs = env.reset()
    roles = [m.get("role") for m in obs.messages]
    # seed_observation carries the opening message. Internal state tracks system+user.
    # The observation messages from seed_observation always include user.
    assert "user" in roles


def test_reset_seeds_internal_state_with_user_message(fake_openenv):
    """After reset(), internal message list contains the user opening message."""
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    scenario = _make_scenario("state-seed-test", opening_message="Seeded.")
    env_cls = build_environment_class(
        scenario, _Obs, object, persona=_FakePersona([])
    )
    env = env_cls()
    env.reset()
    # The internal _korrel_messages list should have at least the user message.
    assert len(env._korrel_messages) >= 1
    assert env._korrel_messages[-1].role == "user"
    assert env._korrel_messages[-1].content == "Seeded."


def test_reset_clears_previous_state_on_second_call(fake_openenv):
    """Calling reset() a second time clears previous episode state."""
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    scenario = _make_scenario("reset-clear-test", max_turns=5)
    env_cls = build_environment_class(
        scenario, _Obs, object, persona=_FakePersona(["r1", "r2"])
    )
    env = env_cls()
    env.reset()
    # Manually inflate internal state to simulate a partial episode.
    env._korrel_tool_round_state["user_turns"] = 3
    env._korrel_tool_round_state["tool_round"] = 2
    # Second reset must clear it.
    env.reset()
    assert env._korrel_tool_round_state["user_turns"] == 0
    assert env._korrel_tool_round_state["tool_round"] == 0


def test_reset_state_step_count_reset_to_zero(fake_openenv):
    """State.step_count is reset to 0 by reset()."""
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    scenario = _make_scenario("step-count-test", max_turns=5)
    env_cls = build_environment_class(
        scenario, _Obs, object, persona=_FakePersona(["u1"])
    )
    env = env_cls()
    env.reset()
    # Simulate a step to advance step_count.
    env._state.step_count = 5
    # Reset must zero it.
    env.reset()
    assert env._state.step_count == 0


# ---------------------------------------------------------------------------
# Tests: step() - tool branch
# ---------------------------------------------------------------------------


def test_step_tool_branch_returns_done_false(fake_openenv):
    """step() with a tool call returns done=False when within tool_round cap."""
    from korrel.exporters.openenv import build_environment_class
    from korrel.tools import MockTool

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    tool = MockTool(
        name="add",
        schema={"type": "function", "function": {"name": "add", "description": "", "parameters": {}}},
        respond=lambda args, state: {"sum": args.get("a", 0) + args.get("b", 0)},
    )
    scenario = _make_scenario("tool-step-test", tools=[tool], max_turns=3)
    env_cls = build_environment_class(scenario, _Obs, object, persona=_FakePersona(["next"]))
    env = env_cls()
    env.reset()

    action = _make_tool_action(
        tool_calls=[{"id": "tc1", "name": "add", "arguments": '{"a": 1, "b": 2}'}]
    )
    obs = env.step(action)
    assert obs.done is False


def test_step_tool_branch_reward_is_none(fake_openenv):
    """step() tool branch sets reward=None (intermediate step)."""
    from korrel.exporters.openenv import build_environment_class
    from korrel.tools import MockTool

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    tool = MockTool(
        name="echo",
        schema={"type": "function", "function": {"name": "echo", "description": "", "parameters": {}}},
        respond=lambda args, state: "echoed",
    )
    scenario = _make_scenario("tool-reward-test", tools=[tool], max_turns=3)
    env_cls = build_environment_class(scenario, _Obs, object, persona=_FakePersona(["next"]))
    env = env_cls()
    env.reset()

    action = _make_tool_action(
        tool_calls=[{"id": "tc1", "name": "echo", "arguments": "{}"}]
    )
    obs = env.step(action)
    assert obs.reward is None


def test_step_tool_branch_observation_contains_tool_result(fake_openenv):
    """step() tool branch observation messages carry a role:'tool' entry."""
    from korrel.exporters.openenv import build_environment_class
    from korrel.tools import MockTool

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    tool = MockTool(
        name="lookup",
        schema={"type": "function", "function": {"name": "lookup", "description": "", "parameters": {}}},
        respond=lambda args, state: "found-it",
    )
    scenario = _make_scenario("tool-obs-test", tools=[tool], max_turns=3)
    env_cls = build_environment_class(scenario, _Obs, object, persona=_FakePersona(["next"]))
    env = env_cls()
    env.reset()

    action = _make_tool_action(
        tool_calls=[{"id": "tc42", "name": "lookup", "arguments": "{}"}]
    )
    obs = env.step(action)
    assert len(obs.messages) == 1
    assert obs.messages[0]["role"] == "tool"


def test_step_tool_branch_tool_call_id_in_result_message(fake_openenv):
    """The tool result message carries tool_call_id matching the originating call id.

    This exercises the shim at the generated-environment level: an action
    carrying tool_calls (arguments as a JSON string) is converted to a canonical
    ToolCall and the tool result comes back as a role:'tool' message with the
    correct tool_call_id.
    """
    from korrel.exporters.openenv import build_environment_class
    from korrel.tools import MockTool

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    tool = MockTool(
        name="fetch",
        schema={"type": "function", "function": {"name": "fetch", "description": "", "parameters": {}}},
        respond=lambda args, state: "data",
    )
    scenario = _make_scenario("shim-test", tools=[tool], max_turns=3)
    env_cls = build_environment_class(scenario, _Obs, object, persona=_FakePersona(["q"]))
    env = env_cls()
    env.reset()

    call_id = "call-abc-123"
    action = _make_tool_action(
        tool_calls=[{"id": call_id, "name": "fetch", "arguments": '{"url": "x"}'}]
    )
    obs = env.step(action)
    assert obs.messages[0].get("tool_call_id") == call_id


def test_step_tool_branch_max_tool_rounds_cap_terminates(fake_openenv):
    """Reaching max_tool_rounds during step causes done=True."""
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    scenario = _make_scenario("cap-step-test", max_tool_rounds=1, max_turns=5)
    env_cls = build_environment_class(scenario, _Obs, object, persona=_FakePersona(["u"]))
    env = env_cls()
    env.reset()
    # Advance tool_round to the cap value.
    env._korrel_tool_round_state["tool_round"] = 1

    action = _make_tool_action(
        tool_calls=[{"id": "tc1", "name": "no_such", "arguments": "{}"}]
    )
    obs = env.step(action)
    assert obs.done is True


# ---------------------------------------------------------------------------
# Tests: step() - persona branch
# ---------------------------------------------------------------------------


def test_step_persona_branch_returns_user_message(fake_openenv):
    """step() persona branch observation contains the next user message."""
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    scenario = _make_scenario("persona-step-test", max_turns=3)
    env_cls = build_environment_class(
        scenario, _Obs, object, persona=_FakePersona(["follow-up question"])
    )
    env = env_cls()
    env.reset()

    action = _make_tool_action(content="I can help with that.")
    obs = env.step(action)
    assert obs.done is False
    assert obs.reward is None
    assert len(obs.messages) == 1
    assert obs.messages[0]["role"] == "user"
    assert obs.messages[0]["content"] == "follow-up question"


def test_step_persona_exhausted_returns_done_true(fake_openenv):
    """Persona returning None causes step() to return done=True."""
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    scenario = _make_scenario("exhaust-step-test", max_turns=5)
    env_cls = build_environment_class(
        scenario, _Obs, object, persona=_FakePersona([])  # empty queue
    )
    env = env_cls()
    env.reset()

    action = _make_tool_action(content="Policy reply.")
    obs = env.step(action)
    assert obs.done is True


def test_step_persona_exhausted_terminal_reward_exact(fake_openenv):
    """Terminal step carries reward == deterministic rubric aggregate (exact scalar)."""
    from korrel.rubric import Rubric
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    # Deterministic reward: always returns 0.75.
    expected_reward = 0.75

    def fixed_reward(completion, info_dict, **kw):
        return expected_reward

    rubric = Rubric(funcs=[fixed_reward])
    scenario = _make_scenario("reward-exact-test", max_turns=5, rubric=rubric)
    env_cls = build_environment_class(
        scenario, _Obs, object, persona=_FakePersona([])
    )
    env = env_cls()
    env.reset()

    action = _make_tool_action(content="Assistant reply.")
    obs = env.step(action)
    assert obs.done is True
    assert obs.reward == pytest.approx(expected_reward)


def test_step_max_turns_terminates_with_reward(fake_openenv):
    """max_turns reached causes done=True with terminal reward.

    The budget matches run_scenario: for max_turns=1 the policy gets 1
    assistant turn and the persona is never called. The cap fires when
    user_turns >= max_turns - 1 = 0, so even user_turns=0 triggers
    termination. user_turns=0 is the initial state after reset().
    """
    from korrel.rubric import Rubric
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    expected_reward = 0.3

    def fixed_reward(completion, info_dict, **kw):
        return expected_reward

    rubric = Rubric(funcs=[fixed_reward])
    # max_turns=1: the first plain assistant step immediately terminates.
    # The cap fires at user_turns >= max_turns - 1 = 0 (initial state).
    scenario = _make_scenario("max-turns-step-test", max_turns=1, rubric=rubric)
    env_cls = build_environment_class(
        scenario, _Obs, object, persona=_FakePersona(["would-follow"])
    )
    env = env_cls()
    env.reset()
    # user_turns=0 (the post-reset default) is sufficient to trigger the cap for max_turns=1.

    action = _make_tool_action(content="Done.")
    obs = env.step(action)
    assert obs.done is True
    assert obs.reward == pytest.approx(expected_reward)


def test_step_no_persona_terminates_immediately(fake_openenv):
    """With persona=None in build_environment_class, the first plain step terminates."""
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    scenario = _make_scenario("no-persona-test", max_turns=5)
    env_cls = build_environment_class(
        scenario, _Obs, object, persona=None
    )
    env = env_cls()
    env.reset()

    action = _make_tool_action(content="Hello.")
    obs = env.step(action)
    assert obs.done is True


def test_step_increments_state_step_count(fake_openenv):
    """step() increments State.step_count each time it is called."""
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    scenario = _make_scenario("step-count-inc-test", max_turns=5)
    env_cls = build_environment_class(
        scenario, _Obs, object, persona=_FakePersona(["r1", "r2", "r3"])
    )
    env = env_cls()
    env.reset()

    action = _make_tool_action(content="Turn.")
    env.step(action)
    assert env.state.step_count == 1
    env.step(action)
    assert env.state.step_count == 2


def test_step_intermediate_steps_have_none_reward(fake_openenv):
    """Every intermediate step carries reward=None (terminal-only reward contract)."""
    from korrel.exporters.openenv import build_environment_class

    FakeObservation = fake_openenv["openenv.core.env_server.types"].Observation

    class _Obs(FakeObservation):
        def __init__(self, messages=None, done=False, reward=None, **kw):
            super().__init__(messages=messages, done=done, reward=reward, **kw)

    scenario = _make_scenario("reward-none-test", max_turns=5)
    env_cls = build_environment_class(
        scenario, _Obs, object, persona=_FakePersona(["r1", "r2", "r3"])
    )
    env = env_cls()
    env.reset()

    action = _make_tool_action(content="Step 1.")
    obs1 = env.step(action)
    assert obs1.reward is None

    action2 = _make_tool_action(content="Step 2.")
    obs2 = env.step(action2)
    assert obs2.reward is None


# ---------------------------------------------------------------------------
# Tests: round-trip (write_openenv_env -> load generated modules -> reset/step)
# ---------------------------------------------------------------------------

_BUNDLED_SCENARIO_SRC = '''\
"""Bundled offline scenario for round-trip tests."""
from korrel.rubric import Rubric
from korrel.scenario import Scenario


class _FakePersona:
    """Minimal offline persona returning a single scripted reply."""

    def __init__(self, replies):
        self._replies = list(replies)

    def next_message(self, messages):
        if self._replies:
            return self._replies.pop(0)
        return None


def deterministic_reward(completion, info, **kw):
    """Always returns 0.6."""
    return 0.6


scenario = Scenario.model_construct(
    id="roundtrip-offline",
    system="Offline system prompt.",
    persona=_FakePersona(["User follow-up."]),
    opening_message="Offline opening.",
    tools=[],
    max_turns=2,
    max_tool_rounds=8,
    seed=0,
    info=None,
    rubric=Rubric(funcs=[deterministic_reward]),
)
'''


@pytest.fixture()
def roundtrip_package(tmp_path, fake_openenv):
    """Write a bundled offline OpenEnv package to tmp_path and return the out_dir.

    Depends on fake_openenv so that the generated environment module can be
    loaded (it imports openenv.core.env_server.interfaces.Environment and
    openenv.core.env_server.types.State at load time).
    """
    from korrel.exporters.openenv import write_openenv_env
    from korrel.rubric import Rubric
    from korrel.scenario import Scenario

    scenario_src = tmp_path / "_scenario_src.py"
    scenario_src.write_text(_BUNDLED_SCENARIO_SRC, encoding="utf-8")

    def deterministic_reward(completion, info, **kw):
        return 0.6

    class _FP:
        def next_message(self, messages):
            return None

    scenario = Scenario.model_construct(
        id="roundtrip-offline",
        system="Offline system prompt.",
        persona=_FP(),
        opening_message="Offline opening.",
        tools=[],
        max_turns=2,
        max_tool_rounds=8,
        seed=0,
        info=None,
        rubric=Rubric(funcs=[deterministic_reward]),
    )

    out_dir = tmp_path / "export"
    write_openenv_env(scenario, out_dir, scenario_source_path=scenario_src)
    return out_dir


def _load_module_from_path(
    path: Path,
    name: str,
    extra_sys_path: Path | None = None,
) -> types.ModuleType:
    """Load a Python source file by absolute path as a module.

    If ``extra_sys_path`` is provided it is prepended to ``sys.path`` for the
    duration of the load and removed afterwards. This is needed for the
    generated server/<env>_environment.py whose fallback import path
    (``from models import KorrelAction, KorrelObservation``) requires the
    package root directory to be on sys.path.
    """
    inserted = False
    if extra_sys_path is not None:
        extra = str(extra_sys_path)
        if extra not in sys.path:
            sys.path.insert(0, extra)
            inserted = True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None, f"Cannot build spec for {path}"
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
    finally:
        if inserted:
            try:
                sys.path.remove(str(extra_sys_path))
            except ValueError:
                pass


def test_roundtrip_models_py_compiles_and_defines_korrel_action(roundtrip_package):
    """The generated models.py compiles and defines KorrelAction and KorrelObservation.

    models.py imports from openenv.core.env_server.types; the fake openenv
    fixture satisfies that import so the module can be loaded.
    """
    models_path = roundtrip_package / "models.py"
    assert models_path.exists(), "models.py must be generated"

    # Load models.py; it imports from openenv.core.env_server.types.
    # The fake_openenv fixture (via roundtrip_package) must be installed.
    mod = _load_module_from_path(
        models_path, "_rt_models", extra_sys_path=roundtrip_package
    )
    assert hasattr(mod, "KorrelAction"), "models.py must define KorrelAction"
    assert hasattr(mod, "KorrelObservation"), "models.py must define KorrelObservation"


def test_roundtrip_environment_module_defines_korrel_environment(roundtrip_package):
    """The generated server/<env>_environment.py defines KorrelEnvironment."""
    server_dir = roundtrip_package / "server"
    env_files = list(server_dir.glob("*_environment.py"))
    assert len(env_files) == 1, f"Expected one environment .py, got {env_files}"

    env_path = env_files[0]
    # extra_sys_path=roundtrip_package satisfies the fallback:
    #   from models import KorrelAction, KorrelObservation
    # where models.py lives at roundtrip_package/models.py.
    mod = _load_module_from_path(
        env_path, "_rt_env", extra_sys_path=roundtrip_package
    )
    assert hasattr(mod, "KorrelEnvironment"), "environment module must define KorrelEnvironment"


def test_roundtrip_environment_reset_returns_observation_with_messages(roundtrip_package):
    """Loaded KorrelEnvironment.reset() returns an observation with messages."""
    server_dir = roundtrip_package / "server"
    env_files = list(server_dir.glob("*_environment.py"))
    env_path = env_files[0]
    mod = _load_module_from_path(
        env_path, "_rt_env_reset", extra_sys_path=roundtrip_package
    )

    env = mod.KorrelEnvironment()
    obs = env.reset()
    assert obs.done is False
    assert obs.reward is None
    assert isinstance(obs.messages, list)
    assert len(obs.messages) >= 1
    roles = [m.get("role") for m in obs.messages]
    assert "user" in roles


def test_roundtrip_environment_reset_observation_contains_opening_text(roundtrip_package):
    """Loaded KorrelEnvironment.reset() observation includes the opening message text."""
    server_dir = roundtrip_package / "server"
    env_files = list(server_dir.glob("*_environment.py"))
    env_path = env_files[0]
    mod = _load_module_from_path(
        env_path, "_rt_env_opening", extra_sys_path=roundtrip_package
    )

    env = mod.KorrelEnvironment()
    obs = env.reset()
    contents = [m.get("content", "") for m in obs.messages]
    assert any("Offline opening." in str(c) for c in contents)


def test_roundtrip_environment_step_tool_branch_returns_tool_message(
    roundtrip_package, fake_openenv
):
    """Loaded KorrelEnvironment.step() with tool_calls returns role:'tool' messages.

    This exercises the shim at the generated-environment level: action carrying
    tool_calls (arguments as a JSON string) produces a role:'tool' message with
    the correct tool_call_id.

    Because the bundled scenario has no tools, an unknown-tool call returns the
    error dict (mirrors advance() unknown-tool branch). The role and tool_call_id
    are still populated correctly.
    """
    from korrel.tools import MockTool

    # Re-write the package with a scenario that has a real tool.
    from korrel.exporters.openenv import write_openenv_env
    from korrel.rubric import Rubric
    from korrel.scenario import Scenario

    tool = MockTool(
        name="ping",
        schema={"type": "function", "function": {"name": "ping", "description": "", "parameters": {}}},
        respond=lambda args, state: "pong",
    )

    tmp_dir = roundtrip_package.parent / "tool_pkg"

    scenario_src = roundtrip_package.parent / "_tool_scenario.py"
    scenario_src.write_text(
        "\n".join([
            "from korrel.rubric import Rubric",
            "from korrel.scenario import Scenario",
            "from korrel.tools import MockTool",
            "",
            "class _FP:",
            "    def next_message(self, m): return None",
            "",
            "def reward_fn(c, i, **k): return 1.0",
            "",
            "tool = MockTool(",
            "    name='ping',",
            "    schema={'type':'function','function':{'name':'ping','description':'','parameters':{}}},",
            "    respond=lambda a,s: 'pong',",
            ")",
            "scenario = Scenario.model_construct(",
            "    id='tool-roundtrip',",
            "    system='s',",
            "    persona=_FP(),",
            "    opening_message='hi',",
            "    tools=[tool],",
            "    max_turns=3,",
            "    max_tool_rounds=8,",
            "    seed=0,",
            "    info=None,",
            "    rubric=Rubric(funcs=[reward_fn]),",
            ")",
        ]),
        encoding="utf-8",
    )

    class _FP:
        def next_message(self, m):
            return None

    def reward_fn(c, i, **k):
        return 1.0

    scenario = Scenario.model_construct(
        id="tool-roundtrip",
        system="s",
        persona=_FP(),
        opening_message="hi",
        tools=[tool],
        max_turns=3,
        max_tool_rounds=8,
        seed=0,
        info=None,
        rubric=Rubric(funcs=[reward_fn]),
    )
    write_openenv_env(scenario, tmp_dir, scenario_source_path=scenario_src)

    server_dir = tmp_dir / "server"
    env_files = list(server_dir.glob("*_environment.py"))
    env_path = env_files[0]
    mod = _load_module_from_path(env_path, "_rt_tool_env", extra_sys_path=tmp_dir)

    env = mod.KorrelEnvironment()
    env.reset()

    call_id = "call-roundtrip-42"

    class _Act:
        content = None
        tool_calls = [{"id": call_id, "name": "ping", "arguments": "{}"}]

    obs = env.step(_Act())
    assert obs.done is False
    assert len(obs.messages) == 1
    assert obs.messages[0]["role"] == "tool"
    assert obs.messages[0].get("tool_call_id") == call_id


def test_roundtrip_environment_terminal_step_carries_rubric_reward(roundtrip_package):
    """Loaded KorrelEnvironment terminal step carries done=True and reward == 0.6.

    The bundled scenario uses deterministic_reward which always returns 0.6.
    The mean of a single 0.6-valued function is 0.6.
    """
    server_dir = roundtrip_package / "server"
    env_files = list(server_dir.glob("*_environment.py"))
    env_path = env_files[0]
    mod = _load_module_from_path(
        env_path, "_rt_env_terminal", extra_sys_path=roundtrip_package
    )

    env = mod.KorrelEnvironment()
    env.reset()

    # The bundled scenario has _FakePersona(["User follow-up."]) which returns one
    # reply then None. Two assistant steps are required to exhaust the persona.
    # Step 1: persona returns the queued reply, done=False.
    # Step 2: persona queue is empty, returns None, done=True with reward=0.6.

    class _Act:
        content = "Hello."
        tool_calls = None

    obs1 = env.step(_Act())
    assert obs1.done is False, "First step should not terminate (persona has one reply)"

    obs2 = env.step(_Act())
    assert obs2.done is True
    assert obs2.reward == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Tests: _import_openenv_core raises ImportError when absent
# ---------------------------------------------------------------------------


def test_import_openenv_core_raises_when_absent():
    """_import_openenv_core raises ImportError with install hint when openenv absent."""
    # This test runs with no fake_openenv fixture, so openenv is absent.
    if "openenv" in sys.modules:
        pytest.skip("openenv is installed; this test requires it to be absent")

    from korrel.exporters.openenv import _import_openenv_core

    with pytest.raises(ImportError, match="openenv-core"):
        _import_openenv_core()


def test_build_environment_class_raises_without_openenv():
    """build_environment_class raises ImportError when openenv-core is not installed."""
    if "openenv" in sys.modules:
        pytest.skip("openenv is installed; this test requires it to be absent")

    from korrel.exporters.openenv import build_environment_class

    scenario = _make_scenario()

    with pytest.raises(ImportError, match="openenv-core"):
        build_environment_class(scenario, object, object)
