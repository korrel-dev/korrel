"""Integration tests for korrel.exporters.openenv against real openenv-core.

Gate: ``pytest.importorskip("openenv")`` skips this entire module when
openenv-core is not installed (Python 3.14 repo venv, default CI matrix).
All tests are offline: no network, no API keys, no running server. Live
personas and live judges are replaced by deterministic fakes.

Confirmed against: openenv-core==0.3.0. All interface claims cite the module
and symbol they were read from.

What these tests cover (complementary to test_openenv_exporter.py and
test_openenv_exporter_fake_ns.py):

1. ``build_environment_class`` with a fake persona and deterministic reward
   builds an Environment whose ``reset()`` returns a REAL
   ``openenv.core.env_server.types.Observation`` instance with ``done=False``
   and ``reward=None``.
2. Terminal ``step()`` returns a real Observation with ``done=True`` and
   ``reward == expected scalar``.
3. Round-trip: ``write_openenv_env`` to tmp with a bundled offline scenario,
   load the generated environment module, construct/reset/step, assert the
   real Observation type and the done/reward fields.
4. Optional: ``create_app(...)`` builds a FastAPI app object without starting
   a server (import-level only, no HTTP calls).

Ref: docs/spec/korrel-to-openenv.md
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Optional

import pytest

# Gate: skip the entire module when openenv is not installed.
# In the default repo venv (Python 3.14, no openenv-core) this line causes the
# module to be collected but all tests are skipped. In CI's openenv job
# (openenv-core>=0.3.0 installed) the module runs normally.
openenv = pytest.importorskip("openenv")


# ---------------------------------------------------------------------------
# Helpers
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
    scenario_id: str = "test-real-openenv",
    *,
    system: str = "You are helpful.",
    opening_message: str = "Hello.",
    max_turns: int = 2,
    max_tool_rounds: int = 8,
    tools: list = None,
    rubric=None,
    persona=None,
    info: Any = None,
):
    """Build a minimal Scenario via model_construct (bypasses Pydantic validation)."""
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


def _make_observation_action_cls():
    """Return (ObsCls, ActCls) subclasses of the real openenv base types.

    These are minimal pydantic subclasses that add the fields the exporter
    expects (messages on Observation, content/tool_calls on Action).

    The real openenv.core.env_server.types.Observation and Action set
    ``extra="forbid"``, so declared fields must be listed explicitly.
    """
    from openenv.core.env_server.types import Action, Observation
    from pydantic import Field

    class _Obs(Observation):
        messages: list = Field(default_factory=list)

    class _Act(Action):
        content: Optional[str] = Field(default=None)
        tool_calls: Optional[list] = Field(default=None)

    return _Obs, _Act


def _plain_action(content: Optional[str] = None, tool_calls: Any = None) -> Any:
    """Build a plain object action that carries content and tool_calls attributes."""

    class _Action:
        pass

    a = _Action()
    a.content = content
    a.tool_calls = tool_calls
    return a


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


# ---------------------------------------------------------------------------
# 1. build_environment_class with real openenv types
# ---------------------------------------------------------------------------


def test_build_environment_class_returns_env_subclass():
    """build_environment_class returns a type that is a subclass of real Environment.

    Confirmed against: openenv.core.env_server.interfaces.Environment
    (openenv-core 0.3.0, core/env_server/interfaces.py:
    class Environment(ABC, Generic[ActT, ObsT, StateT])).
    """
    from openenv.core.env_server.interfaces import Environment

    from korrel.exporters.openenv import build_environment_class

    _Obs, _Act = _make_observation_action_cls()
    scenario = _make_scenario()
    env_cls = build_environment_class(scenario, _Obs, _Act, persona=_FakePersona([]))
    assert issubclass(env_cls, Environment)


def test_reset_returns_real_observation_instance():
    """reset() returns a real openenv.core.env_server.types.Observation instance.

    Confirmed against: openenv.core.env_server.types.Observation (openenv-core
    0.3.0, core/env_server/types.py: class Observation(BaseModel) with
    done: bool = False, reward: bool | int | float | None = None).
    """
    from openenv.core.env_server.types import Observation

    from korrel.exporters.openenv import build_environment_class

    _Obs, _Act = _make_observation_action_cls()
    scenario = _make_scenario()
    env_cls = build_environment_class(scenario, _Obs, _Act, persona=_FakePersona([]))
    env = env_cls()
    obs = env.reset()
    assert isinstance(obs, Observation)


def test_reset_done_false_reward_none_on_real_observation():
    """reset() real Observation has done=False and reward=None."""
    from korrel.exporters.openenv import build_environment_class

    _Obs, _Act = _make_observation_action_cls()
    scenario = _make_scenario("reset-real-check", opening_message="Opener.")
    env_cls = build_environment_class(scenario, _Obs, _Act, persona=_FakePersona([]))
    env = env_cls()
    obs = env.reset()
    assert obs.done is False
    assert obs.reward is None


def test_reset_observation_messages_contain_opening_user_message():
    """reset() real Observation messages carry the opening user message."""
    from korrel.exporters.openenv import build_environment_class

    _Obs, _Act = _make_observation_action_cls()
    scenario = _make_scenario("reset-msg-check", opening_message="What day is it?")
    env_cls = build_environment_class(
        scenario, _Obs, _Act, persona=_FakePersona([])
    )
    env = env_cls()
    obs = env.reset()
    roles = [m.get("role") for m in obs.messages]
    assert "user" in roles
    contents = [m.get("content", "") for m in obs.messages]
    assert any("What day is it?" in str(c) for c in contents)


# ---------------------------------------------------------------------------
# 2. Terminal step returns real Observation with done=True and exact reward
# ---------------------------------------------------------------------------


def test_terminal_step_returns_real_observation():
    """Terminal step() returns a real Observation instance."""
    from openenv.core.env_server.types import Observation

    from korrel.exporters.openenv import build_environment_class

    _Obs, _Act = _make_observation_action_cls()
    scenario = _make_scenario(max_turns=3)
    env_cls = build_environment_class(scenario, _Obs, _Act, persona=_FakePersona([]))
    env = env_cls()
    env.reset()

    obs = env.step(_plain_action(content="Policy reply."))
    assert isinstance(obs, Observation)


def test_terminal_step_done_true_with_exact_reward():
    """Terminal step() real Observation has done=True and reward == expected scalar.

    Uses a deterministic reward function that always returns 0.7. The rubric
    mean over a single function is 0.7 exactly.
    """
    from korrel.rubric import Rubric
    from korrel.exporters.openenv import build_environment_class

    _Obs, _Act = _make_observation_action_cls()
    expected = 0.7

    def fixed_reward(completion, info_dict, **kw):
        return expected

    rubric = Rubric(funcs=[fixed_reward])
    scenario = _make_scenario("terminal-real-test", max_turns=5, rubric=rubric)
    env_cls = build_environment_class(
        scenario, _Obs, _Act, persona=_FakePersona([])  # empty queue
    )
    env = env_cls()
    env.reset()

    obs = env.step(_plain_action(content="Policy reply."))
    assert obs.done is True
    assert obs.reward == pytest.approx(expected)


def test_terminal_step_intermediate_reward_is_none():
    """Non-terminal steps carry reward=None (terminal-only reward contract)."""
    from korrel.exporters.openenv import build_environment_class

    _Obs, _Act = _make_observation_action_cls()
    scenario = _make_scenario(max_turns=5)
    env_cls = build_environment_class(
        scenario, _Obs, _Act, persona=_FakePersona(["r1", "r2", "r3"])
    )
    env = env_cls()
    env.reset()

    obs = env.step(_plain_action(content="Turn."))
    assert obs.reward is None
    assert obs.done is False


def test_terminal_step_persona_exhaust_reward_is_float():
    """Terminal reward is a float (not bool, not None) after persona exhaustion."""
    from korrel.exporters.openenv import build_environment_class

    _Obs, _Act = _make_observation_action_cls()
    scenario = _make_scenario(max_turns=5)
    env_cls = build_environment_class(
        scenario, _Obs, _Act, persona=_FakePersona([])
    )
    env = env_cls()
    env.reset()

    obs = env.step(_plain_action(content="Step."))
    assert obs.done is True
    assert isinstance(obs.reward, (int, float))
    assert not isinstance(obs.reward, bool)


# ---------------------------------------------------------------------------
# 3. Round-trip: write_openenv_env -> load generated environment -> reset/step
# ---------------------------------------------------------------------------

_BUNDLED_SCENARIO_REAL = """\
\"\"\"Bundled offline scenario for real-openenv round-trip tests.\"\"\"
from korrel.rubric import Rubric
from korrel.scenario import Scenario


class _FakePersona:
    def __init__(self, replies):
        self._replies = list(replies)

    def next_message(self, messages):
        if self._replies:
            return self._replies.pop(0)
        return None


def deterministic_reward(completion, info, **kw):
    \"\"\"Always returns 0.8.\"\"\"
    return 0.8


scenario = Scenario.model_construct(
    id="roundtrip-real-openenv",
    system="Real-openenv system prompt.",
    persona=_FakePersona([]),
    opening_message="Real-openenv opening.",
    tools=[],
    max_turns=2,
    max_tool_rounds=8,
    seed=0,
    info=None,
    rubric=Rubric(funcs=[deterministic_reward]),
)
"""


@pytest.fixture()
def real_roundtrip_package(tmp_path):
    """Write a bundled offline OpenEnv package to tmp_path and return the out_dir.

    Uses real openenv-core types for models.py (openenv.core.env_server.types).
    The bundled scenario is fully offline (fake persona, deterministic reward).
    """
    from korrel.exporters.openenv import write_openenv_env
    from korrel.rubric import Rubric
    from korrel.scenario import Scenario

    scenario_src = tmp_path / "_scenario_src.py"
    scenario_src.write_text(_BUNDLED_SCENARIO_REAL, encoding="utf-8")

    def deterministic_reward(completion, info, **kw):
        return 0.8

    class _FP:
        def next_message(self, messages):
            return None

    scenario = Scenario.model_construct(
        id="roundtrip-real-openenv",
        system="Real-openenv system prompt.",
        persona=_FP(),
        opening_message="Real-openenv opening.",
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


def test_roundtrip_models_py_action_observation_are_real_subclasses(
    real_roundtrip_package,
):
    """Generated models.py KorrelAction/KorrelObservation are real subclass instances.

    openenv.core.env_server.types.Action and .Observation are the base classes
    (openenv-core 0.3.0, core/env_server/types.py).
    """
    from openenv.core.env_server.types import Action, Observation

    models_path = real_roundtrip_package / "models.py"
    mod = _load_module_from_path(
        models_path, "_rt_real_models", extra_sys_path=real_roundtrip_package
    )

    assert issubclass(mod.KorrelAction, Action), (
        "KorrelAction must be a subclass of openenv.core.env_server.types.Action"
    )
    assert issubclass(mod.KorrelObservation, Observation), (
        "KorrelObservation must be a subclass of openenv.core.env_server.types.Observation"
    )


def test_roundtrip_environment_reset_returns_real_observation_isinstance(
    real_roundtrip_package,
):
    """Loaded KorrelEnvironment.reset() returns a real Observation isinstance."""
    from openenv.core.env_server.types import Observation

    server_dir = real_roundtrip_package / "server"
    env_files = list(server_dir.glob("*_environment.py"))
    assert len(env_files) == 1, f"Expected one environment .py, got {env_files}"
    env_path = env_files[0]

    mod = _load_module_from_path(
        env_path, "_rt_real_env_reset", extra_sys_path=real_roundtrip_package
    )
    env = mod.KorrelEnvironment()
    obs = env.reset()

    assert isinstance(obs, Observation), (
        "reset() must return a real openenv.core.env_server.types.Observation instance"
    )
    assert obs.done is False
    assert obs.reward is None


def test_roundtrip_environment_terminal_step_done_and_reward(
    real_roundtrip_package,
):
    """Loaded KorrelEnvironment terminal step has done=True and reward==0.8.

    The bundled scenario uses deterministic_reward returning 0.8.
    """
    from openenv.core.env_server.types import Observation

    server_dir = real_roundtrip_package / "server"
    env_files = list(server_dir.glob("*_environment.py"))
    env_path = env_files[0]

    mod = _load_module_from_path(
        env_path, "_rt_real_env_terminal", extra_sys_path=real_roundtrip_package
    )
    env = mod.KorrelEnvironment()
    env.reset()

    # The bundled scenario has _FakePersona([]) with an empty queue. A plain
    # assistant step immediately exhausts the persona and triggers termination.

    class _Act:
        content = "Policy reply."
        tool_calls = None

    obs = env.step(_Act())
    assert isinstance(obs, Observation)
    assert obs.done is True
    assert obs.reward == pytest.approx(0.8)


def test_roundtrip_environment_intermediate_step_done_false(
    real_roundtrip_package,
):
    """Loaded KorrelEnvironment step with persona reply returns done=False."""
    from korrel.exporters.openenv import write_openenv_env
    from korrel.rubric import Rubric
    from korrel.scenario import Scenario

    # Build a package with a persona that has one reply so the first step is
    # non-terminal.
    tmp_dir = real_roundtrip_package.parent / "nonterminal_pkg"
    scenario_src = real_roundtrip_package.parent / "_nonterminal_scenario.py"
    scenario_src.write_text(
        "\n".join([
            "from korrel.rubric import Rubric",
            "from korrel.scenario import Scenario",
            "",
            "class _FP:",
            "    def __init__(self):",
            "        self._q = ['reply-text']",
            "    def next_message(self, m):",
            "        if self._q: return self._q.pop(0)",
            "        return None",
            "",
            "def reward_fn(c, i, **k): return 1.0",
            "",
            "scenario = Scenario.model_construct(",
            "    id='nonterminal-roundtrip',",
            "    system='sys',",
            "    persona=_FP(),",
            "    opening_message='hi',",
            "    tools=[],",
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
        def __init__(self):
            self._q = ["reply-text"]

        def next_message(self, m):
            if self._q:
                return self._q.pop(0)
            return None

    def reward_fn(c, i, **k):
        return 1.0

    scenario = Scenario.model_construct(
        id="nonterminal-roundtrip",
        system="sys",
        persona=_FP(),
        opening_message="hi",
        tools=[],
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
    mod = _load_module_from_path(
        env_path, "_rt_real_nonterminal", extra_sys_path=tmp_dir
    )
    env = mod.KorrelEnvironment()
    env.reset()

    class _Act:
        content = "Hello."
        tool_calls = None

    obs = env.step(_Act())
    assert obs.done is False
    assert obs.reward is None


# ---------------------------------------------------------------------------
# 4. create_app builds a FastAPI app object without starting a server
# ---------------------------------------------------------------------------


def test_create_app_builds_fastapi_app_without_starting_server(real_roundtrip_package):
    """create_app(KorrelEnvironment, KorrelAction, KorrelObservation) returns an app.

    Confirmed against: openenv.core.env_server.http_server.create_app
    (openenv-core 0.3.0, core/env_server/http_server.py:
    def create_app(env, action_cls, observation_cls, env_name=None, ...) -> FastAPI).
    No HTTP server is started; only the app object is inspected.

    This test is marked xfail-on-missing-fastapi so that environments with
    openenv-core but without fastapi installed do not fail the suite outright.
    """
    fastapi = pytest.importorskip("fastapi", reason="fastapi is required for create_app")

    from openenv.core.env_server.http_server import create_app

    # Load the generated models.py and environment module to get real subclasses.
    models_path = real_roundtrip_package / "models.py"
    models_mod = _load_module_from_path(
        models_path, "_rt_app_models", extra_sys_path=real_roundtrip_package
    )
    server_dir = real_roundtrip_package / "server"
    env_files = list(server_dir.glob("*_environment.py"))
    env_path = env_files[0]
    env_mod = _load_module_from_path(
        env_path, "_rt_app_env", extra_sys_path=real_roundtrip_package
    )

    KorrelAction = models_mod.KorrelAction
    KorrelObservation = models_mod.KorrelObservation
    KorrelEnvironment = env_mod.KorrelEnvironment

    app = create_app(
        KorrelEnvironment,
        KorrelAction,
        KorrelObservation,
        env_name="roundtrip-real-openenv",
    )
    # The returned object is a FastAPI app instance.
    assert isinstance(app, fastapi.FastAPI), (
        f"create_app must return a FastAPI instance, got {type(app)}"
    )
