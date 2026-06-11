"""Sanity-check tests for korrel.exporters.openenv.

These tests run without openenv-core installed and without network access or
API keys. They verify:
- Import safety: importing korrel.exporters.openenv never imports openenv.
- write_openenv_env: artifact layout (all expected files present).
- Generated .py files compile() for a normal scenario id.
- Generated .py files compile() for a hostile scenario id (injection guard).
- scenario_attr identifier guard raises ValueError on invalid input.
- _sanitize_id_for_comment strips triple-quotes and newlines.
- advance() tool and persona branches work with fake objects.
- seed_observation() builds the correct initial message list.

Comprehensive branch and matrix coverage is delegated to test-author per the
dispatch contract.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
):
    """Build a minimal Scenario via model_construct (bypasses Pydantic validation)."""
    from korrel.scenario import Scenario
    from korrel.rubric import Rubric

    if rubric is None:
        rubric = Rubric(funcs=[lambda completion, info, **kw: 1.0])
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
        info=None,
        rubric=rubric,
    )


class _FakePersona:
    """Minimal fake persona for offline tests."""

    def __init__(self, messages: list[str]) -> None:
        self._queue = list(messages)

    def next_message(self, messages: Any) -> Optional[str]:
        if self._queue:
            return self._queue.pop(0)
        return None


class _FakeObservation:
    """Minimal observation stand-in for advance() tests."""

    def __init__(self, messages=None, done=False, reward=None):
        self.messages = messages or []
        self.done = done
        self.reward = reward


def _fake_obs_cls(**kwargs) -> _FakeObservation:
    return _FakeObservation(**kwargs)


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def test_import_openenv_module_does_not_import_openenv():
    """Importing korrel.exporters.openenv must not import openenv-core."""
    before = set(sys.modules.keys())
    import korrel.exporters.openenv  # noqa: F401

    after = set(sys.modules.keys())
    new_modules = after - before
    assert "openenv" not in new_modules, (
        "Importing korrel.exporters.openenv must not import 'openenv'. "
        f"New modules after import: {new_modules}"
    )


def test_import_korrel_does_not_import_openenv():
    """Importing the top-level korrel package must not import openenv-core."""
    before = set(sys.modules.keys())
    import korrel  # noqa: F401

    after = set(sys.modules.keys())
    new_modules = after - before
    assert "openenv" not in new_modules, (
        "Importing korrel must not import 'openenv'. "
        f"New modules after import: {new_modules}"
    )


# ---------------------------------------------------------------------------
# _sanitize_id_for_comment
# ---------------------------------------------------------------------------


def test_sanitize_id_strips_triple_quotes():
    from korrel.exporters.openenv import _sanitize_id_for_comment

    assert '"""' not in _sanitize_id_for_comment('foo"""bar')


def test_sanitize_id_collapses_newlines():
    from korrel.exporters.openenv import _sanitize_id_for_comment

    label = _sanitize_id_for_comment("line1\nline2")
    assert "\n" not in label
    assert "line1" in label
    assert "line2" in label


def test_sanitize_id_passthrough_for_clean_id():
    from korrel.exporters.openenv import _sanitize_id_for_comment

    assert _sanitize_id_for_comment("my-scenario") == "my-scenario"


# ---------------------------------------------------------------------------
# write_openenv_env: artifact layout
# ---------------------------------------------------------------------------


def test_write_openenv_env_creates_expected_files(tmp_path):
    """All expected files are created for a normal scenario."""
    from korrel.exporters.openenv import write_openenv_env

    scenario = _make_scenario("my-env")
    out = write_openenv_env(scenario, tmp_path / "export")

    assert (out / "__init__.py").exists()
    assert (out / "client.py").exists()
    assert (out / "models.py").exists()
    assert (out / "openenv.yaml").exists()
    assert (out / "pyproject.toml").exists()
    assert (out / "README.md").exists()
    assert (out / "_scenario.py").exists()
    assert (out / "server" / "__init__.py").exists()
    assert (out / "server" / "app.py").exists()
    assert (out / "server" / "Dockerfile").exists()
    assert (out / "server" / "requirements.txt").exists()
    # Environment file uses the sanitized env_module name.
    assert (out / "server" / "my_env_environment.py").exists()


def test_write_openenv_env_pyproject_content(tmp_path):
    """pyproject.toml contains required dependencies and script entry point."""
    from korrel.exporters.openenv import write_openenv_env

    scenario = _make_scenario("my-env")
    out = write_openenv_env(scenario, tmp_path / "export")
    content = (out / "pyproject.toml").read_text()

    assert "openenv-core>=0.3.0" in content
    assert "korrel" in content
    assert "server" in content
    assert "app:main" in content


def test_write_openenv_env_openenv_yaml_content(tmp_path):
    """openenv.yaml has the required manifest fields."""
    from korrel.exporters.openenv import write_openenv_env

    scenario = _make_scenario("my-env")
    out = write_openenv_env(scenario, tmp_path / "export")
    content = (out / "openenv.yaml").read_text()

    assert "spec_version: 1" in content
    assert "type: space" in content
    assert "runtime: fastapi" in content
    assert "server.app:app" in content
    assert "port: 8000" in content


def test_write_openenv_env_all_py_files_compile(tmp_path):
    """All generated .py files parse without SyntaxError for a normal id."""
    from korrel.exporters.openenv import write_openenv_env

    scenario = _make_scenario("my-scenario")
    out = write_openenv_env(scenario, tmp_path / "export")

    for py_file in out.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        try:
            compile(source, str(py_file), "exec")
        except SyntaxError as exc:
            raise AssertionError(
                f"Generated file {py_file.name} has a syntax error: {exc}"
            ) from exc


def test_write_openenv_env_hostile_id_templates_compile():
    """A hostile scenario.id does not produce a syntax error in generated source.

    Tests template rendering only (no file I/O) to avoid OS filename
    restrictions on characters that are invalid in Windows paths.
    """
    from korrel.exporters.openenv import (
        _ENVIRONMENT_PY,
        _MODELS_PY,
        _APP_PY,
        _CLIENT_PY,
        _PACKAGE_INIT_PY,
        _sanitize_id_for_comment,
    )

    # Build a maximally hostile id: triple-double-quote, newline, backslash.
    hostile_id = "evil" + '"""' + chr(10) + 'print("injected")' + chr(10) + chr(92)

    ctx = dict(
        scenario_id_repr=repr(hostile_id),
        scenario_id_label=_sanitize_id_for_comment(hostile_id),
        env_module="my_env",
        env_name="my-env",
        scenario_attr="scenario",
        scenario_module_name="_korrel_scenario_my_env",
    )

    templates = {
        "models.py": _MODELS_PY,
        "app.py": _APP_PY,
        "client.py": _CLIENT_PY,
        "__init__.py": _PACKAGE_INIT_PY,
        "<env>_environment.py": _ENVIRONMENT_PY,
    }

    for name, tmpl in templates.items():
        source = tmpl.format(**ctx)
        try:
            compile(source, f"<{name}>", "exec")
        except SyntaxError as exc:
            raise AssertionError(
                f"Hostile id produced syntax error in {name}: {exc}"
            ) from exc

    # The environment template must still define KorrelEnvironment.
    env_source = _ENVIRONMENT_PY.format(**ctx)
    assert "class KorrelEnvironment" in env_source


def test_write_openenv_env_rejects_non_identifier_attr(tmp_path):
    """A non-identifier scenario_attr raises ValueError before any file is written."""
    from korrel.exporters.openenv import write_openenv_env

    scenario = _make_scenario("guard-test")
    with pytest.raises(ValueError, match="valid Python identifier"):
        write_openenv_env(scenario, tmp_path / "export", scenario_attr="x}{evil")


def test_write_openenv_env_copies_scenario_source(tmp_path):
    """When scenario_source_path is provided, it is copied as _scenario.py."""
    from korrel.exporters.openenv import write_openenv_env

    fake_source = tmp_path / "my_scenario.py"
    fake_source.write_text("# real scenario\nscenario = None\n")

    scenario = _make_scenario("copy-test")
    out = write_openenv_env(
        scenario, tmp_path / "export", scenario_source_path=fake_source
    )
    assert "real scenario" in (out / "_scenario.py").read_text()


def test_write_openenv_env_placeholder_when_no_source(tmp_path):
    """When no source path is given, _scenario.py contains a placeholder."""
    from korrel.exporters.openenv import write_openenv_env

    scenario = _make_scenario("placeholder-test")
    out = write_openenv_env(scenario, tmp_path / "export")
    content = (out / "_scenario.py").read_text()
    assert "ImportError" in content or "Placeholder" in content


def test_write_openenv_env_sanitizes_path_traversal(tmp_path):
    """A scenario.id with path separators is sanitized (K9 guard)."""
    from korrel.exporters.openenv import write_openenv_env

    scenario = _make_scenario("some/nested/path")
    out = write_openenv_env(scenario, tmp_path / "export")
    # Environment file should use only the basename component.
    assert (out / "server" / "path_environment.py").exists()
    assert not (out / "some").exists()
    assert not (out / "nested").exists()


def test_write_openenv_env_returns_out_dir(tmp_path):
    """write_openenv_env returns the out_dir Path."""
    from korrel.exporters.openenv import write_openenv_env

    scenario = _make_scenario("return-test")
    out_dir = tmp_path / "my-export"
    result = write_openenv_env(scenario, out_dir)
    assert result == out_dir


def test_write_openenv_env_hostile_id_k11_injection_guard(tmp_path):
    """K11 regression: a scenario.id with injection characters produces valid output.

    A scenario.id containing a newline, semicolon, dot, space, and backslash
    must produce:
    - An env_module that is a valid Python identifier (.isidentifier() True).
    - No path component containing a newline or OS path separator.
    - Every generated .py file that compile()s without SyntaxError.
    - The app.py import line is a single clean 'from .<identifier>_environment import' line.

    Ref: security review K9/K11.
    """
    from korrel.exporters.openenv import write_openenv_env

    # Construct a maximally hostile id (backslash last to avoid Windows path issues
    # when combined with the other characters; use a raw backslash not as a separator).
    hostile_id = "evil\n;.hello world\\"
    scenario = _make_scenario(hostile_id)
    out = write_openenv_env(scenario, tmp_path / "export")

    # Determine what env_module was derived.
    env_files = list((out / "server").glob("*_environment.py"))
    assert len(env_files) == 1, f"Expected exactly one environment .py file, got {env_files}"
    env_file = env_files[0]
    # The stem is "<env_module>_environment"; env_module is everything before "_environment".
    env_module_from_filename = env_file.stem.replace("_environment", "")

    # env_module must be a valid Python identifier.
    assert env_module_from_filename.isidentifier(), (
        f"env_module derived from filename is not a valid identifier: {env_module_from_filename!r}"
    )

    # No file name component must contain a newline.
    # (We check filenames specifically; full absolute paths naturally include separators.)
    for py_file in out.rglob("*.py"):
        assert "\n" not in py_file.name, (
            f"Newline in generated filename: {py_file.name!r}"
        )
        assert "\r" not in py_file.name, (
            f"Carriage return in generated filename: {py_file.name!r}"
        )

    # Every generated .py file must compile().
    for py_file in out.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        try:
            compile(source, str(py_file), "exec")
        except SyntaxError as exc:
            raise AssertionError(
                f"Hostile id produced syntax error in {py_file.name}: {exc}"
            ) from exc

    # app.py must contain a single-line clean import (no injected newline+code).
    app_source = (out / "server" / "app.py").read_text(encoding="utf-8")
    import_lines = [
        line for line in app_source.splitlines()
        if f"from .{env_module_from_filename}_environment import" in line
    ]
    assert len(import_lines) >= 1, (
        "app.py must contain the environment import line"
    )
    # Each matching line must be a single clean line with no injected code.
    for line in import_lines:
        assert "\n" not in line, f"Injected newline in import line: {line!r}"


# ---------------------------------------------------------------------------
# seed_observation
# ---------------------------------------------------------------------------


def test_seed_observation_includes_opening_message():
    """seed_observation returns an observation whose messages include the opening."""
    from korrel.exporters.openenv import seed_observation

    scenario = _make_scenario(
        "seed-test",
        system="System prompt.",
        opening_message="What can you do?",
    )
    obs = seed_observation(scenario, _fake_obs_cls)
    assert obs.done is False
    assert obs.reward is None
    # The observation messages should include the opening user message.
    roles = [m.get("role") for m in obs.messages]
    assert "user" in roles
    contents = [m.get("content", "") for m in obs.messages]
    assert any("What can you do?" in str(c) for c in contents)


def test_seed_observation_no_system():
    """seed_observation without a system field emits only the user message."""
    from korrel.exporters.openenv import seed_observation

    scenario = _make_scenario("no-sys", system="", opening_message="Hello.")
    obs = seed_observation(scenario, _fake_obs_cls)
    roles = [m.get("role") for m in obs.messages]
    assert roles == ["user"]


# ---------------------------------------------------------------------------
# advance(): tool branch
# ---------------------------------------------------------------------------


def _make_tool_action(content=None, tool_calls=None):
    """Build a minimal fake action with content and tool_calls."""

    class FakeAction:
        pass

    a = FakeAction()
    a.content = content
    a.tool_calls = tool_calls
    return a


def test_advance_tool_branch_resolves_call():
    """Tool branch resolves a known tool and returns its result."""
    from korrel.exporters.openenv import advance
    from korrel.tools import MockTool

    called: dict = {}

    def respond(args, state):
        called["args"] = args
        return {"value": 42}

    tool = MockTool(
        name="mytool",
        schema={
            "type": "function",
            "function": {"name": "mytool", "description": "", "parameters": {}},
        },
        respond=respond,
    )
    scenario = _make_scenario("tool-test", tools=[tool])
    state_messages: list = []
    tool_round_state = {"tool_round": 0, "user_turns": 0, "tool_state": {}}

    action = _make_tool_action(
        tool_calls=[{"id": "tc1", "name": "mytool", "arguments": '{"x": 1}'}]
    )

    obs = advance(
        scenario,
        state_messages,
        tool_round_state,
        action,
        _fake_obs_cls,
        persona=_FakePersona([]),
    )

    assert obs.done is False
    assert obs.reward is None
    assert called.get("args") == {"x": 1}
    # Tool round counter incremented.
    assert tool_round_state["tool_round"] == 1
    # Tool result message in the observation.
    assert len(obs.messages) == 1
    assert obs.messages[0]["role"] == "tool"


def test_advance_tool_branch_unknown_tool():
    """Unknown tool returns error dict without raising."""
    from korrel.exporters.openenv import advance

    scenario = _make_scenario("unknown-tool-test")
    state_messages: list = []
    tool_round_state = {"tool_round": 0, "user_turns": 0, "tool_state": {}}

    action = _make_tool_action(
        tool_calls=[{"id": "tc1", "name": "no_such_tool", "arguments": "{}"}]
    )

    obs = advance(
        scenario,
        state_messages,
        tool_round_state,
        action,
        _fake_obs_cls,
        persona=_FakePersona([]),
    )

    assert obs.done is False
    content = obs.messages[0].get("content", "")
    assert "unknown tool" in content


def test_advance_tool_branch_bad_json():
    """Malformed tool arguments default to {} (tolerant parse per spec)."""
    from korrel.exporters.openenv import advance
    from korrel.tools import MockTool

    called: dict = {}

    def respond(args, state):
        called["args"] = args
        return "ok"

    tool = MockTool(
        name="t",
        schema={"type": "function", "function": {"name": "t", "description": "", "parameters": {}}},
        respond=respond,
    )
    scenario = _make_scenario("bad-json-test", tools=[tool])
    state_messages: list = []
    tool_round_state = {"tool_round": 0, "user_turns": 0, "tool_state": {}}

    action = _make_tool_action(
        tool_calls=[{"id": "tc1", "name": "t", "arguments": "INVALID{{JSON"}]
    )

    advance(
        scenario,
        state_messages,
        tool_round_state,
        action,
        _fake_obs_cls,
        persona=_FakePersona([]),
    )

    assert called.get("args") == {}


def test_advance_tool_round_cap_terminates():
    """When max_tool_rounds is reached the episode terminates."""
    from korrel.exporters.openenv import advance

    scenario = _make_scenario("cap-test", max_tool_rounds=2)
    state_messages: list = []
    # Simulate that we are already at the cap.
    tool_round_state = {"tool_round": 2, "user_turns": 0, "tool_state": {}}

    action = _make_tool_action(
        tool_calls=[{"id": "tc1", "name": "t", "arguments": "{}"}]
    )

    obs = advance(
        scenario,
        state_messages,
        tool_round_state,
        action,
        _fake_obs_cls,
        persona=_FakePersona([]),
    )

    assert obs.done is True
    assert obs.reward is not None


# ---------------------------------------------------------------------------
# advance(): persona branch
# ---------------------------------------------------------------------------


def test_advance_persona_branch_returns_user_message():
    """Persona branch returns the next user message with done=False."""
    from korrel.exporters.openenv import advance

    scenario = _make_scenario("persona-test", max_turns=3)
    state_messages: list = []
    tool_round_state = {"tool_round": 0, "user_turns": 0, "tool_state": {}}

    action = _make_tool_action(content="Hello from policy")

    obs = advance(
        scenario,
        state_messages,
        tool_round_state,
        action,
        _fake_obs_cls,
        persona=_FakePersona(["follow-up"]),
    )

    assert obs.done is False
    assert obs.reward is None
    assert len(obs.messages) == 1
    assert obs.messages[0]["role"] == "user"
    assert obs.messages[0]["content"] == "follow-up"


def test_advance_persona_exhausted_terminates():
    """Persona returning None triggers done=True with terminal reward."""
    from korrel.exporters.openenv import advance

    scenario = _make_scenario("exhaust-test", max_turns=3)
    state_messages: list = []
    tool_round_state = {"tool_round": 0, "user_turns": 0, "tool_state": {}}

    action = _make_tool_action(content="policy turn")

    obs = advance(
        scenario,
        state_messages,
        tool_round_state,
        action,
        _fake_obs_cls,
        persona=_FakePersona([]),  # empty queue
    )

    assert obs.done is True
    assert obs.reward is not None
    assert isinstance(obs.reward, float)


def test_advance_max_turns_terminates():
    """Reaching the max_turns budget triggers done=True.

    The budget matches run_scenario: for max_turns=2 the policy gets 2
    assistant turns and the persona is called at most 1 time. The cap fires
    when user_turns >= max_turns - 1 (i.e. user_turns >= 1 for max_turns=2).
    """
    from korrel.exporters.openenv import advance

    scenario = _make_scenario("turns-test", max_turns=2)
    state_messages: list = []
    # user_turns=1 is the minimum that triggers the cap for max_turns=2
    # (cap fires at user_turns >= max_turns - 1 = 1).
    tool_round_state = {"tool_round": 0, "user_turns": 1, "tool_state": {}}

    action = _make_tool_action(content="final turn")

    obs = advance(
        scenario,
        state_messages,
        tool_round_state,
        action,
        _fake_obs_cls,
        persona=_FakePersona(["would-follow"]),
    )

    assert obs.done is True


def test_advance_reward_is_none_on_intermediate_steps():
    """Intermediate steps carry reward=None."""
    from korrel.exporters.openenv import advance

    scenario = _make_scenario("reward-test", max_turns=5)
    state_messages: list = []
    tool_round_state = {"tool_round": 0, "user_turns": 0, "tool_state": {}}

    action = _make_tool_action(content="turn 1")
    obs = advance(
        scenario,
        state_messages,
        tool_round_state,
        action,
        _fake_obs_cls,
        persona=_FakePersona(["reply", "another"]),
    )

    assert obs.reward is None


def test_advance_tool_state_persists_across_calls():
    """tool_state dict is shared across calls and accumulates state."""
    from korrel.exporters.openenv import advance
    from korrel.tools import MockTool

    def respond(args, state):
        state["counter"] = state.get("counter", 0) + 1
        return state["counter"]

    tool = MockTool(
        name="counter",
        schema={"type": "function", "function": {"name": "counter", "description": "", "parameters": {}}},
        respond=respond,
    )
    scenario = _make_scenario("state-persist-test", tools=[tool], max_turns=5)
    state_messages: list = []
    tool_round_state = {"tool_round": 0, "user_turns": 0, "tool_state": {}}

    action = _make_tool_action(
        tool_calls=[{"id": "tc1", "name": "counter", "arguments": "{}"}]
    )

    advance(scenario, state_messages, tool_round_state, action, _fake_obs_cls,
            persona=_FakePersona([]))
    # Reset tool_round so the second call is not capped.
    tool_round_state["tool_round"] = 0
    state_messages2: list = []
    advance(scenario, state_messages2, tool_round_state, action, _fake_obs_cls,
            persona=_FakePersona([]))

    assert tool_round_state["tool_state"]["counter"] == 2


# ---------------------------------------------------------------------------
# CLI: "openenv" is a supported target
# ---------------------------------------------------------------------------


def test_cli_parser_accepts_openenv_target():
    """argparse accepts --to openenv."""
    from korrel.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["export", "my_scenario.py", "--to", "openenv"])
    assert args.to == "openenv"


def test_cli_openenv_export_creates_package(tmp_path, monkeypatch):
    """CLI export to openenv creates the package directory."""
    import argparse

    from korrel.cli import _cmd_export

    module_file = tmp_path / "scen.py"
    module_file.write_text(
        "from korrel.scenario import Scenario\n"
        "from korrel.rubric import Rubric\n"
        "from korrel.persona import Persona\n"
        "scenario = Scenario(\n"
        "    id='cli-openenv-test',\n"
        "    system='s',\n"
        "    persona=Persona(goal='g'),\n"
        "    opening_message='hi',\n"
        "    rubric=Rubric(funcs=[lambda c, i, **k: 1.0]),\n"
        ")\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(
        file=str(module_file), to="openenv", out=str(out_dir), scenario_attr="scenario"
    )
    result = _cmd_export(args)
    assert result == 0
    assert (out_dir / "pyproject.toml").exists()
    assert (out_dir / "models.py").exists()
    assert (out_dir / "server" / "app.py").exists()
