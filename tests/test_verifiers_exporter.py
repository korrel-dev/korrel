"""Offline tests for the korrel.exporters.verifiers module.

All tests run without verifiers installed and without network access or API
keys. Where the verifiers environment class is needed, fake substitutes that
mirror the minimal interface are used.

Test matrix:
- Import safety: importing korrel.exporters.verifiers never imports verifiers.
- Error on missing verifiers: calling to_verifiers_env raises ImportError.
- _to_korrel_messages: field-by-field conversion for all message roles.
- _wrap_reward_fn: sync and async reward fn wrapping.
- _safe_filename_stem reuse in write_verifiers_env.
- write_verifiers_env: artifact layout (pyproject.toml, env_module.py, _scenario.py).
- CLI export subcommand: argparse accepts expected arguments.
- SingleTurn vs MultiTurn decision rule.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from korrel.types import Message


# ---------------------------------------------------------------------------
# Import-safety test: importing the exporter module must not import verifiers.
# ---------------------------------------------------------------------------


def test_import_exporter_module_does_not_import_verifiers():
    """Importing korrel.exporters.verifiers must not trigger 'import verifiers'."""
    # Guard: if verifiers happens to be installed in the current env, skip.
    # The module-level guard is the actual contract; we test it by ensuring
    # the 'verifiers' module is not in sys.modules after importing the exporter
    # (unless it was already there before).
    before = set(sys.modules.keys())
    import korrel.exporters.verifiers  # noqa: F401

    after = set(sys.modules.keys())
    new_modules = after - before
    assert "verifiers" not in new_modules, (
        "Importing korrel.exporters.verifiers must not import the 'verifiers' package. "
        f"New modules after import: {new_modules}"
    )


def test_import_korrel_does_not_import_verifiers():
    """Importing the top-level korrel package must not import verifiers."""
    before = set(sys.modules.keys())
    import korrel  # noqa: F401

    after = set(sys.modules.keys())
    new_modules = after - before
    assert "verifiers" not in new_modules, (
        "Importing korrel must not import 'verifiers'. "
        f"New modules after import: {new_modules}"
    )


# ---------------------------------------------------------------------------
# Error test: to_verifiers_env raises ImportError when verifiers is absent.
# ---------------------------------------------------------------------------


def _make_minimal_scenario(
    tools=None,
    max_turns=1,
    rubric=None,
    persona=None,
    system="You are an assistant.",
    opening_message="Hello.",
    info=None,
):
    """Build a minimal Scenario using fakes, no live imports.

    ``Scenario.persona`` is typed as ``Persona`` and validated by Pydantic. A
    fake persona cannot be placed in this field directly. Tests that need a fake
    persona pass it as the ``persona`` override to ``to_verifiers_env`` instead.
    Here we use ``Scenario.model_construct`` to bypass validation so offline
    tests never require the live ``Persona`` model.
    """
    from korrel.scenario import Scenario
    from korrel.rubric import Rubric

    if rubric is None:
        rubric = Rubric(funcs=[lambda completion, info, **kw: 1.0])

    if persona is None:
        persona = _FakePersona([])

    # model_construct bypasses Pydantic validation, allowing _FakePersona in
    # the persona field and None in rubric. This is safe for offline tests
    # because we never serialize or round-trip the scenario.
    return Scenario.model_construct(
        id="test-scenario",
        system=system,
        persona=persona,
        opening_message=opening_message,
        tools=tools or [],
        max_turns=max_turns,
        max_tool_rounds=8,
        seed=0,
        info=info,
        rubric=rubric,
    )


class _FakePersona:
    """Minimal fake persona for tests; never makes model calls."""

    def __init__(self, messages: list[str]) -> None:
        self._queue = list(messages)

    def next_message(self, messages: list[Message]) -> Optional[str]:
        if self._queue:
            return self._queue.pop(0)
        return None


def test_to_verifiers_env_raises_import_error_when_verifiers_absent():
    """to_verifiers_env raises ImportError with install hint when verifiers absent."""
    from korrel.exporters.verifiers import to_verifiers_env

    scenario = _make_minimal_scenario()

    # Patch _import_verifiers to simulate verifiers being absent.
    with patch("korrel.exporters.verifiers._import_verifiers") as mock_import:
        mock_import.side_effect = ImportError("korrel's verifiers exporter requires")
        with pytest.raises(ImportError, match="verifiers"):
            to_verifiers_env(scenario)


def test_to_verifiers_env_raises_value_error_without_rubric():
    """to_verifiers_env raises ValueError when scenario.rubric is None."""
    from korrel.exporters.verifiers import to_verifiers_env
    from korrel.scenario import Scenario

    # Use model_construct to bypass Pydantic validation and allow rubric=None.
    scenario = Scenario.model_construct(
        id="no-rubric",
        system="sys",
        persona=_FakePersona([]),
        opening_message="hi",
        tools=[],
        max_turns=1,
        max_tool_rounds=8,
        seed=0,
        rubric=None,
    )

    # Even when verifiers would be available, the rubric check fires first.
    with pytest.raises(ValueError, match="rubric"):
        to_verifiers_env(scenario)


# ---------------------------------------------------------------------------
# _to_korrel_messages: field-by-field conversion tests.
# ---------------------------------------------------------------------------


def _make_vf_message(role: str, **fields: Any) -> Any:
    """Build a plain-dict stand-in for a verifiers Message."""
    msg: dict[str, Any] = {"role": role}
    msg.update(fields)
    return msg


def test_to_korrel_messages_system():
    from korrel.exporters.verifiers import _to_korrel_messages

    msgs = [_make_vf_message("system", content="You are helpful.")]
    result = _to_korrel_messages(msgs)
    assert len(result) == 1
    assert result[0].role == "system"
    assert result[0].content == "You are helpful."


def test_to_korrel_messages_user():
    from korrel.exporters.verifiers import _to_korrel_messages

    msgs = [_make_vf_message("user", content="Hello!")]
    result = _to_korrel_messages(msgs)
    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].content == "Hello!"


def test_to_korrel_messages_assistant_no_tool_calls():
    from korrel.exporters.verifiers import _to_korrel_messages

    msgs = [_make_vf_message("assistant", content="I can help.")]
    result = _to_korrel_messages(msgs)
    assert len(result) == 1
    assert result[0].role == "assistant"
    assert result[0].content == "I can help."
    assert result[0].tool_calls is None


def test_to_korrel_messages_assistant_with_tool_calls():
    """FLAT vf.ToolCall{id,name,arguments} -> NESTED korrel.ToolCall."""
    from korrel.exporters.verifiers import _to_korrel_messages

    vf_tool_call = {"id": "tc1", "name": "search", "arguments": '{"query": "hello"}'}
    msgs = [_make_vf_message("assistant", content=None, tool_calls=[vf_tool_call])]
    result = _to_korrel_messages(msgs)
    assert len(result) == 1
    assert result[0].role == "assistant"
    assert result[0].tool_calls is not None
    assert len(result[0].tool_calls) == 1
    tc = result[0].tool_calls[0]
    assert tc.id == "tc1"
    assert tc.type == "function"
    assert tc.function.name == "search"
    assert tc.function.arguments == '{"query": "hello"}'


def test_to_korrel_messages_tool_result():
    from korrel.exporters.verifiers import _to_korrel_messages

    msgs = [_make_vf_message("tool", content="result text", tool_call_id="tc1")]
    result = _to_korrel_messages(msgs)
    assert len(result) == 1
    assert result[0].role == "tool"
    assert result[0].content == "result text"
    assert result[0].tool_call_id == "tc1"


def test_to_korrel_messages_list_content():
    """List content is narrowed to string (lossy for non-text parts)."""
    from korrel.exporters.verifiers import _to_korrel_messages

    content_parts = [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]
    msgs = [_make_vf_message("user", content=content_parts)]
    result = _to_korrel_messages(msgs)
    assert len(result) == 1
    assert "Hello" in result[0].content
    assert "world" in result[0].content


def test_to_korrel_messages_mixed_conversation():
    """Full conversation with system, user, assistant+tool_calls, tool, assistant."""
    from korrel.exporters.verifiers import _to_korrel_messages

    vf_tc = {"id": "call_1", "name": "lookup", "arguments": '{"key": "x"}'}
    msgs = [
        _make_vf_message("system", content="sys"),
        _make_vf_message("user", content="question"),
        _make_vf_message("assistant", content=None, tool_calls=[vf_tc]),
        _make_vf_message("tool", content="answer", tool_call_id="call_1"),
        _make_vf_message("assistant", content="final answer"),
    ]
    result = _to_korrel_messages(msgs)
    assert len(result) == 5
    assert result[0].role == "system"
    assert result[1].role == "user"
    assert result[2].role == "assistant"
    assert result[2].tool_calls[0].function.name == "lookup"
    assert result[3].role == "tool"
    assert result[3].tool_call_id == "call_1"
    assert result[4].role == "assistant"
    assert result[4].content == "final answer"


# ---------------------------------------------------------------------------
# _wrap_reward_fn: sync and async wrapping.
# ---------------------------------------------------------------------------


def test_wrap_reward_fn_sync():
    """Wrapped sync fn receives list[korrel.Message] and returns float."""
    from korrel.exporters.verifiers import _wrap_reward_fn

    received: dict[str, Any] = {}

    def my_reward(completion, info, **kwargs):
        received["completion"] = completion
        received["info"] = info
        return 0.75

    wrapped = _wrap_reward_fn(my_reward, "my_reward")
    assert wrapped.__name__ == "my_reward"

    vf_msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    result = wrapped(completion=vf_msgs, info={"key": "val"})
    assert result == 0.75
    # The korrel fn received list[Message], not raw dicts.
    assert isinstance(received["completion"], list)
    assert all(isinstance(m, Message) for m in received["completion"])
    assert received["info"] == {"key": "val"}


def test_wrap_reward_fn_async():
    """Wrapped async fn is itself async and returns float."""
    from korrel.exporters.verifiers import _wrap_reward_fn
    import inspect

    async def async_reward(completion, info, **kwargs):
        return 0.5

    wrapped = _wrap_reward_fn(async_reward, "async_reward")
    assert wrapped.__name__ == "async_reward"
    assert inspect.iscoroutinefunction(wrapped)

    vf_msgs = [{"role": "user", "content": "hi"}]
    result = asyncio.run(wrapped(completion=vf_msgs, info={}))
    assert result == 0.5


def test_wrap_reward_fn_preserves_name():
    """Wrapped fn __name__ matches the provided name."""
    from korrel.exporters.verifiers import _wrap_reward_fn

    def exact_match(completion, info, **kwargs):
        return 1.0

    wrapped = _wrap_reward_fn(exact_match, "exact_match")
    assert wrapped.__name__ == "exact_match"


def test_wrap_reward_fn_coerces_to_float():
    """Integer return is coerced to float."""
    from korrel.exporters.verifiers import _wrap_reward_fn

    def integer_reward(completion, info, **kwargs):
        return 1

    wrapped = _wrap_reward_fn(integer_reward, "integer_reward")
    result = wrapped(completion=[], info={})
    assert isinstance(result, float)
    assert result == 1.0


# ---------------------------------------------------------------------------
# SingleTurn vs MultiTurn decision rule.
# ---------------------------------------------------------------------------


class _FakeVF:
    """Minimal verifiers namespace substitute for testing the decision rule."""

    class Tool:
        def __init__(self, name, description, parameters, strict=None):
            self.name = name
            self.description = description
            self.parameters = parameters

    class Rubric:
        def __init__(self, funcs=None, weights=None, parser=None):
            self.funcs = funcs or []
            self.weights = weights or []

    class _SingleTurnEnv:
        _class_name = "SingleTurnEnv"

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _MultiTurnEnv:
        _class_name = "MultiTurnEnv"

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    SingleTurnEnv = _SingleTurnEnv
    MultiTurnEnv = _MultiTurnEnv

    @staticmethod
    def UserMessage(content):
        return {"role": "user", "content": content}

    @staticmethod
    def ToolMessage(tool_call_id, content):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _patch_imports_for_decision_test(monkeypatch, fake_vf):
    """Patch _import_verifiers and _import_datasets for decision rule tests."""
    fake_datasets = MagicMock()
    fake_datasets.Dataset.from_list.return_value = MagicMock()
    monkeypatch.setattr("korrel.exporters.verifiers._import_verifiers", lambda: fake_vf)
    monkeypatch.setattr("korrel.exporters.verifiers._import_datasets", lambda: fake_datasets)
    return fake_datasets


def test_decision_rule_single_turn(monkeypatch):
    """max_turns==1, no tools, persona=None -> SingleTurnEnv."""
    from korrel.exporters.verifiers import to_verifiers_env
    from korrel.rubric import Rubric
    from korrel.scenario import Scenario

    class _VF(_FakeVF):
        class SingleTurnEnv:
            used = False

            def __init__(self, **kwargs):
                _VF.SingleTurnEnv.used = True
                self.env_id = kwargs.get("env_id", "")

        MultiTurnEnv = _FakeVF.MultiTurnEnv

    _patch_imports_for_decision_test(monkeypatch, _VF)

    scenario = Scenario.model_construct(
        id="single",
        system="sys",
        persona=_FakePersona([]),
        opening_message="hi",
        tools=[],
        max_turns=1,
        max_tool_rounds=8,
        seed=0,
        rubric=Rubric(funcs=[lambda c, i, **k: 1.0]),
    )
    # persona=None override means SingleTurnEnv (no persona follow-up, no tools).
    to_verifiers_env(scenario, persona=None)
    assert _VF.SingleTurnEnv.used


def test_decision_rule_multi_turn_tools(monkeypatch):
    """Has tools -> MultiTurnEnv even with max_turns==1."""
    from korrel.exporters.verifiers import to_verifiers_env
    from korrel.rubric import Rubric
    from korrel.scenario import Scenario
    from korrel.tools import MockTool

    class _VF(_FakeVF):
        class MultiTurnEnv:
            used = False

            def __init__(self, **kwargs):
                _VF.MultiTurnEnv.used = True
                self.env_id = kwargs.get("env_id", "")

            _korrel_persona = None
            _korrel_tools_by_name = {}
            _korrel_max_tool_rounds = 8

        SingleTurnEnv = _FakeVF._SingleTurnEnv

    _patch_imports_for_decision_test(monkeypatch, _VF)

    # _build_multiturn_env_class must return something that accepts the kwargs.
    monkeypatch.setattr(
        "korrel.exporters.verifiers._build_multiturn_env_class",
        lambda vf: _VF.MultiTurnEnv,
    )

    tool = MockTool(
        name="mytool",
        schema={"type": "function", "function": {"name": "mytool", "description": "a tool", "parameters": {}}},
        respond=lambda args, state: "ok",
    )
    scenario = Scenario.model_construct(
        id="multi-tool",
        system="sys",
        persona=_FakePersona([]),
        opening_message="hi",
        tools=[tool],
        max_turns=1,
        max_tool_rounds=8,
        seed=0,
        rubric=Rubric(funcs=[lambda c, i, **k: 1.0]),
    )
    to_verifiers_env(scenario, persona=None)
    assert _VF.MultiTurnEnv.used


def test_decision_rule_multi_turn_persona(monkeypatch):
    """Has persona -> MultiTurnEnv."""
    from korrel.exporters.verifiers import to_verifiers_env
    from korrel.rubric import Rubric
    from korrel.scenario import Scenario

    class _VF(_FakeVF):
        class MultiTurnEnv:
            used = False

            def __init__(self, **kwargs):
                _VF.MultiTurnEnv.used = True
                self.env_id = kwargs.get("env_id", "")

            _korrel_persona = None
            _korrel_tools_by_name = {}
            _korrel_max_tool_rounds = 8

        SingleTurnEnv = _FakeVF._SingleTurnEnv

    _patch_imports_for_decision_test(monkeypatch, _VF)

    monkeypatch.setattr(
        "korrel.exporters.verifiers._build_multiturn_env_class",
        lambda vf: _VF.MultiTurnEnv,
    )

    scenario = Scenario.model_construct(
        id="multi-persona",
        system="sys",
        persona=_FakePersona([]),
        opening_message="hi",
        tools=[],
        max_turns=1,
        max_tool_rounds=8,
        seed=0,
        rubric=Rubric(funcs=[lambda c, i, **k: 1.0]),
    )
    # Pass a fake persona with a follow-up message as the override.
    to_verifiers_env(scenario, persona=_FakePersona(["follow-up"]))
    assert _VF.MultiTurnEnv.used


def test_decision_rule_multi_turn_max_turns(monkeypatch):
    """max_turns > 1 -> MultiTurnEnv."""
    from korrel.exporters.verifiers import to_verifiers_env
    from korrel.rubric import Rubric
    from korrel.scenario import Scenario

    class _VF(_FakeVF):
        class MultiTurnEnv:
            used = False

            def __init__(self, **kwargs):
                _VF.MultiTurnEnv.used = True
                self.env_id = kwargs.get("env_id", "")

            _korrel_persona = None
            _korrel_tools_by_name = {}
            _korrel_max_tool_rounds = 8

        SingleTurnEnv = _FakeVF._SingleTurnEnv

    _patch_imports_for_decision_test(monkeypatch, _VF)

    monkeypatch.setattr(
        "korrel.exporters.verifiers._build_multiturn_env_class",
        lambda vf: _VF.MultiTurnEnv,
    )

    scenario = Scenario.model_construct(
        id="multi-turns",
        system="sys",
        persona=_FakePersona([]),
        opening_message="hi",
        tools=[],
        max_turns=3,
        max_tool_rounds=8,
        seed=0,
        rubric=Rubric(funcs=[lambda c, i, **k: 1.0]),
    )
    to_verifiers_env(scenario, persona=None)
    assert _VF.MultiTurnEnv.used


# ---------------------------------------------------------------------------
# env_response branches (fake environment exercised without verifiers).
# ---------------------------------------------------------------------------


def _make_fake_multiturn_class():
    """Build a KorrelMultiTurnEnv using a fake vf namespace for offline tests."""
    import korrel.exporters.verifiers as exporter

    class _FakeState(dict):
        pass

    class FakeVFNamespace:
        class MultiTurnEnv:
            def __init__(self, **kwargs):
                pass

        @staticmethod
        def UserMessage(content):
            return {"role": "user", "content": content}

        @staticmethod
        def ToolMessage(tool_call_id, content):
            return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    cls = exporter._build_multiturn_env_class(FakeVFNamespace)
    return cls, _FakeState


def test_env_response_tool_branch():
    """Tool branch: resolves tool call and returns ToolMessage."""
    from korrel.tools import MockTool
    import asyncio

    cls, FakeState = _make_fake_multiturn_class()
    env = cls.__new__(cls)

    called_with: dict[str, Any] = {}

    def respond(args, state):
        called_with["args"] = args
        called_with["state"] = state
        return {"result": "ok"}

    tool = MockTool(
        name="mytool",
        schema={"type": "function", "function": {"name": "mytool", "description": "", "parameters": {}}},
        respond=respond,
    )
    env._korrel_tools_by_name = {"mytool": tool}
    env._korrel_persona = _FakePersona([])
    env._korrel_max_tool_rounds = 8

    state = FakeState()

    vf_tool_call = {"id": "tc1", "name": "mytool", "arguments": '{"x": 1}'}
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": None, "tool_calls": [vf_tool_call]},
    ]

    result = asyncio.run(env.env_response(messages, state))

    assert len(result) == 1
    assert result[0]["role"] == "tool"
    assert result[0]["tool_call_id"] == "tc1"
    assert called_with["args"] == {"x": 1}
    assert state["_korrel_tool_round"] == 1


def test_env_response_tool_branch_unknown_tool():
    """Unknown tool returns error dict, does not raise."""
    import asyncio

    cls, FakeState = _make_fake_multiturn_class()
    env = cls.__new__(cls)
    env._korrel_tools_by_name = {}
    env._korrel_persona = _FakePersona([])
    env._korrel_max_tool_rounds = 8

    state = FakeState()

    vf_tool_call = {"id": "tc1", "name": "unknown_tool", "arguments": "{}"}
    messages = [{"role": "assistant", "content": None, "tool_calls": [vf_tool_call]}]

    result = asyncio.run(env.env_response(messages, state))

    assert len(result) == 1
    content = result[0]["content"]
    assert "unknown tool" in content


def test_env_response_tool_branch_bad_json():
    """Malformed tool arguments default to {} (tolerant per spec)."""
    import asyncio

    cls, FakeState = _make_fake_multiturn_class()
    env = cls.__new__(cls)

    called_with: dict[str, Any] = {}

    def respond(args, state):
        called_with["args"] = args
        return "ok"

    from korrel.tools import MockTool

    tool = MockTool(
        name="t",
        schema={"type": "function", "function": {"name": "t", "description": "", "parameters": {}}},
        respond=respond,
    )
    env._korrel_tools_by_name = {"t": tool}
    env._korrel_persona = _FakePersona([])
    env._korrel_max_tool_rounds = 8

    state = FakeState()
    vf_tool_call = {"id": "tc1", "name": "t", "arguments": "INVALID JSON {{{"}
    messages = [{"role": "assistant", "content": None, "tool_calls": [vf_tool_call]}]

    asyncio.run(env.env_response(messages, state))

    assert called_with["args"] == {}


def test_env_response_tool_round_cap():
    """When max_tool_rounds is reached, env sets final_env_response and returns []."""
    import asyncio

    cls, FakeState = _make_fake_multiturn_class()
    env = cls.__new__(cls)
    env._korrel_tools_by_name = {}
    env._korrel_persona = _FakePersona([])
    env._korrel_max_tool_rounds = 2

    state = FakeState()
    state["_korrel_tool_round"] = 2  # already at cap

    vf_tool_call = {"id": "tc1", "name": "t", "arguments": "{}"}
    messages = [{"role": "assistant", "content": None, "tool_calls": [vf_tool_call]}]

    result = asyncio.run(env.env_response(messages, state))

    assert result == []
    assert state.get("final_env_response") is not None


def test_env_response_persona_branch_returns_user_message():
    """Persona branch: non-empty response returns [UserMessage]."""
    import asyncio

    cls, FakeState = _make_fake_multiturn_class()
    env = cls.__new__(cls)
    env._korrel_tools_by_name = {}
    env._korrel_persona = _FakePersona(["follow-up question"])
    env._korrel_max_tool_rounds = 8

    state = FakeState()
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "tool_calls": None},
    ]

    result = asyncio.run(env.env_response(messages, state))

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "follow-up question"


def test_env_response_persona_exhausted_sets_final():
    """Persona exhaustion sets state['final_env_response'] and returns []."""
    import asyncio

    cls, FakeState = _make_fake_multiturn_class()
    env = cls.__new__(cls)
    env._korrel_tools_by_name = {}
    env._korrel_persona = _FakePersona([])  # empty queue
    env._korrel_max_tool_rounds = 8

    state = FakeState()
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    result = asyncio.run(env.env_response(messages, state))

    assert result == []
    assert state.get("final_env_response") is not None


def test_env_response_tool_state_persists():
    """The korrel tool state dict persists across calls (same dict reference)."""
    import asyncio

    cls, FakeState = _make_fake_multiturn_class()
    env = cls.__new__(cls)

    persistent_state: dict[str, Any] = {}

    def respond(args, state):
        state["counter"] = state.get("counter", 0) + 1
        return state["counter"]

    from korrel.tools import MockTool

    tool = MockTool(
        name="counter",
        schema={"type": "function", "function": {"name": "counter", "description": "", "parameters": {}}},
        respond=respond,
    )
    env._korrel_tools_by_name = {"counter": tool}
    env._korrel_persona = _FakePersona([])
    env._korrel_max_tool_rounds = 8

    state = FakeState()
    vf_tc = {"id": "tc1", "name": "counter", "arguments": "{}"}
    messages = [{"role": "assistant", "content": None, "tool_calls": [vf_tc]}]

    asyncio.run(env.env_response(messages, state))
    asyncio.run(env.env_response(messages, state))

    # The tool state is stored in state["_korrel_tool_state"].
    assert state["_korrel_tool_state"]["counter"] == 2


# ---------------------------------------------------------------------------
# write_verifiers_env: artifact layout tests.
# ---------------------------------------------------------------------------


def _make_scenario_for_artifact(scenario_id: str, **kwargs):
    """Build a Scenario for artifact emitter tests using model_construct."""
    from korrel.scenario import Scenario
    from korrel.rubric import Rubric

    rubric = kwargs.pop("rubric", Rubric(funcs=[lambda c, i, **k: 1.0]))
    return Scenario.model_construct(
        id=scenario_id,
        system=kwargs.pop("system", "sys"),
        persona=_FakePersona([]),
        opening_message=kwargs.pop("opening_message", "hello"),
        tools=[],
        max_turns=1,
        max_tool_rounds=8,
        seed=0,
        rubric=rubric,
        **kwargs,
    )


def test_write_verifiers_env_creates_files(tmp_path):
    """Emitter creates pyproject.toml, env_module.py, and _scenario.py."""
    from korrel.exporters.verifiers import write_verifiers_env

    scenario = _make_scenario_for_artifact("my-test-scenario")
    out = write_verifiers_env(scenario, tmp_path / "export")

    assert (out / "pyproject.toml").exists()
    # env_module name: safe_filename_stem("my-test-scenario") = "my-test-scenario"
    # -> replace("-","_") = "my_test_scenario"
    assert (out / "my_test_scenario.py").exists()
    assert (out / "_scenario.py").exists()


def test_write_verifiers_env_pyproject_content(tmp_path):
    """pyproject.toml contains required fields and dependencies."""
    from korrel.exporters.verifiers import write_verifiers_env

    scenario = _make_scenario_for_artifact("my-env")
    out = write_verifiers_env(scenario, tmp_path / "export")
    content = (out / "pyproject.toml").read_text()

    assert "verifiers>=0.1.14" in content
    assert "korrel" in content
    assert "hatchling" in content
    assert ">=3.10" in content


def test_write_verifiers_env_module_has_load_environment(tmp_path):
    """Generated env module defines load_environment."""
    from korrel.exporters.verifiers import write_verifiers_env

    scenario = _make_scenario_for_artifact("my-env")
    out = write_verifiers_env(scenario, tmp_path / "export")
    env_module_path = out / "my_env.py"
    content = env_module_path.read_text()

    assert "def load_environment" in content
    assert "to_verifiers_env" in content


def test_write_verifiers_env_copies_scenario_source(tmp_path):
    """When scenario_source_path is provided, it is copied as _scenario.py."""
    from korrel.exporters.verifiers import write_verifiers_env

    # Create a fake scenario source file.
    fake_source = tmp_path / "my_scenario_source.py"
    fake_source.write_text("# fake scenario source\nscenario = None\n")

    scenario = _make_scenario_for_artifact("my-env")
    out = write_verifiers_env(
        scenario,
        tmp_path / "export",
        scenario_source_path=fake_source,
    )
    copied = out / "_scenario.py"
    assert "fake scenario source" in copied.read_text()


def test_write_verifiers_env_placeholder_when_no_source(tmp_path):
    """When no source path is provided, _scenario.py contains an informative placeholder."""
    from korrel.exporters.verifiers import write_verifiers_env

    scenario = _make_scenario_for_artifact("my-env")
    out = write_verifiers_env(scenario, tmp_path / "export")
    content = (out / "_scenario.py").read_text()
    assert "ImportError" in content or "Placeholder" in content


def test_write_verifiers_env_sanitizes_id(tmp_path):
    """Scenario ids with path separators are sanitized."""
    from korrel.exporters.verifiers import write_verifiers_env

    scenario = _make_scenario_for_artifact("some/nested/path")
    out = write_verifiers_env(scenario, tmp_path / "export")
    # After sanitization, the module is named after the basename only.
    assert (out / "path.py").exists()
    # No directory component of the id leaked into the package.
    assert not (out / "some").exists() and not (out / "nested").exists()


def test_write_verifiers_env_rejects_non_identifier_attr(tmp_path):
    """A non-identifier scenario_attr is rejected before any source is generated.

    scenario_attr is interpolated into the generated module template; a value
    that is not a plain Python identifier could inject braces or a misleading
    attribute name (security review K10).
    """
    import pytest

    from korrel.exporters.verifiers import write_verifiers_env

    scenario = _make_scenario_for_artifact("attr-guard")
    with pytest.raises(ValueError, match="valid Python identifier"):
        write_verifiers_env(
            scenario, tmp_path / "export", scenario_attr="x}{scenario_id"
        )


def test_cmd_export_default_out_dir_cannot_escape(tmp_path, monkeypatch, capsys):
    """A traversal scenario.id cannot push the default out_dir outside .korrel/export.

    Regression for the v0.1 K9 path-traversal class: when --out is omitted, the
    default directory name is built from scenario.id and must be sanitized to a
    single path component.
    """
    import argparse

    from korrel.cli import _cmd_export

    module_file = tmp_path / "scen.py"
    module_file.write_text(
        "from korrel.scenario import Scenario\n"
        "from korrel.rubric import Rubric\n"
        "from korrel.persona import Persona\n"
        "scenario = Scenario(\n"
        "    id='../../escapee',\n"
        "    system='s',\n"
        "    persona=Persona(goal='g'),\n"
        "    opening_message='hi',\n"
        "    rubric=Rubric(funcs=[lambda c, i, **k: 1.0]),\n"
        ")\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(
        file=str(module_file), to="verifiers", out=None, scenario_attr="scenario"
    )
    assert _cmd_export(args) == 0

    export_root = tmp_path / ".korrel" / "export"
    produced = [p for p in export_root.iterdir() if p.is_dir()]
    assert produced, "no export directory created"
    # Every produced directory stays directly under .korrel/export (no ..).
    for path in produced:
        assert path.resolve().parent == export_root.resolve()
    # Nothing was written above tmp_path.
    assert not (tmp_path.parent / "escapee").exists()


def test_cmd_export_rejects_non_identifier_attr(tmp_path, capsys):
    """--scenario-attr that is not a Python identifier exits 1 before loading."""
    import argparse

    from korrel.cli import _cmd_export

    dummy = tmp_path / "scen.py"
    dummy.write_text("scenario = None\n", encoding="utf-8")
    args = argparse.Namespace(
        file=str(dummy), to="verifiers", out=str(tmp_path / "out"),
        scenario_attr="x}{evil",
    )
    assert _cmd_export(args) == 1
    assert "identifier" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# Rubric weight normalization: 1/n uniform weights.
# ---------------------------------------------------------------------------


def test_rubric_uniform_weights(monkeypatch):
    """Rubric is created with uniform 1/n weights."""
    from korrel.exporters.verifiers import to_verifiers_env
    from korrel.rubric import Rubric
    from korrel.scenario import Scenario

    captured: dict[str, Any] = {}

    class _CapturingRubric:
        def __init__(self, funcs=None, weights=None, parser=None):
            captured["weights"] = weights

    class _VF(_FakeVF):
        Rubric = _CapturingRubric

        class SingleTurnEnv:
            def __init__(self, **kwargs):
                self.env_id = kwargs.get("env_id", "")

        class MultiTurnEnv(_FakeVF.MultiTurnEnv):
            _korrel_persona = None
            _korrel_tools_by_name = {}
            _korrel_max_tool_rounds = 8

    _patch_imports_for_decision_test(monkeypatch, _VF)

    def r1(completion, info, **kw):
        return 1.0

    def r2(completion, info, **kw):
        return 0.5

    scenario = Scenario.model_construct(
        id="weighted",
        system="sys",
        persona=_FakePersona([]),
        opening_message="hi",
        tools=[],
        max_turns=1,
        max_tool_rounds=8,
        seed=0,
        rubric=Rubric(funcs=[r1, r2]),
    )
    to_verifiers_env(scenario, persona=None)

    weights = captured["weights"]
    assert len(weights) == 2
    assert abs(weights[0] - 0.5) < 1e-9
    assert abs(weights[1] - 0.5) < 1e-9
    assert abs(sum(weights) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# verifiers max_turns budget test.
# ---------------------------------------------------------------------------


def test_vf_max_turns_budget(monkeypatch):
    """verifiers max_turns = scenario.max_turns * (1 + scenario.max_tool_rounds)."""
    from korrel.exporters.verifiers import to_verifiers_env
    from korrel.rubric import Rubric
    from korrel.scenario import Scenario

    captured: dict[str, Any] = {}

    class _CapturingMultiTurnEnv:
        def __init__(self, **kwargs):
            captured["max_turns"] = kwargs.get("max_turns")
            self.env_id = kwargs.get("env_id", "")

        _korrel_persona = None
        _korrel_tools_by_name = {}
        _korrel_max_tool_rounds = 8

    class _VF(_FakeVF):
        class SingleTurnEnv(_FakeVF._SingleTurnEnv):
            pass

    _patch_imports_for_decision_test(monkeypatch, _VF)
    monkeypatch.setattr(
        "korrel.exporters.verifiers._build_multiturn_env_class",
        lambda vf: _CapturingMultiTurnEnv,
    )

    scenario = Scenario.model_construct(
        id="budget",
        system="sys",
        persona=_FakePersona(["follow"]),
        opening_message="hi",
        tools=[],
        max_turns=3,
        max_tool_rounds=5,
        seed=0,
        rubric=Rubric(funcs=[lambda c, i, **k: 1.0]),
    )
    # Pass a non-None persona override so the MultiTurnEnv branch is taken.
    to_verifiers_env(scenario, persona=_FakePersona(["follow"]))

    # Expected: 3 * (1 + 5) = 18
    assert captured["max_turns"] == 18


# ---------------------------------------------------------------------------
# CLI argument parsing test (parses only; does not run the export).
# ---------------------------------------------------------------------------


def test_cli_export_parser_accepts_args():
    """argparse accepts 'export' subcommand with expected flags."""
    from korrel.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        ["export", "my_scenario.py", "--to", "verifiers", "--out", "./out"]
    )
    assert args.command == "export"
    assert args.file == "my_scenario.py"
    assert args.to == "verifiers"
    assert args.out == "./out"


def test_cli_export_parser_default_scenario_attr():
    """--scenario-attr defaults to 'scenario'."""
    from korrel.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["export", "f.py", "--to", "verifiers"])
    assert args.scenario_attr == "scenario"


def test_cli_export_unsupported_target(tmp_path, capsys):
    """Unsupported --to value prints error and returns exit code 1."""
    from korrel.cli import _cmd_export
    import argparse

    # Create a dummy file so the file-not-found check passes.
    dummy = tmp_path / "dummy.py"
    dummy.write_text("scenario = None\n")

    args = argparse.Namespace(
        file=str(dummy),
        to="not-a-real-target",
        out=str(tmp_path / "out"),
        scenario_attr="scenario",
    )
    result = _cmd_export(args)
    assert result == 1
    captured = capsys.readouterr()
    assert "unsupported" in captured.err.lower()


# ---------------------------------------------------------------------------
# Regression: scenario.id with triple-quote / newline / backslash injection.
# ---------------------------------------------------------------------------


def test_env_module_template_hostile_id_produces_valid_python():
    """A scenario.id containing triple-quotes, newlines, and backslashes must
    not break the syntax of the generated env module.

    The fix routes scenario.id through repr() for the _SCENARIO_ID assignment
    (outside any string region) and through _sanitize_id_for_comment for the
    comment header. Neither path allows the raw id to appear inside a string
    literal or to inject arbitrary text into the generated source.

    This test exercises the template rendering directly (no file I/O) so it
    runs on all platforms without hitting OS filename restrictions.

    Regression for the code-injection / syntax-break bug reported in Dispatch D.
    """
    from korrel.exporters.verifiers import (
        _ENV_MODULE_TEMPLATE,
        _sanitize_id_for_comment,
    )

    # Construct a maximally hostile id: triple-double-quote, newline, backslash.
    hostile_id = 'evil"""\nprint("injected")\n\\'

    # Render the template exactly as write_verifiers_env does.
    source = _ENV_MODULE_TEMPLATE.format(
        scenario_id_repr=repr(hostile_id),
        scenario_id_label=_sanitize_id_for_comment(hostile_id),
        scenario_attr="scenario",
    )

    # Key assertion: the generated source must parse without SyntaxError.
    try:
        compile(source, "<generated>", "exec")
    except SyntaxError as exc:
        raise AssertionError(
            f"Generated module has a syntax error: {exc}\n\nSource:\n{source}"
        ) from exc

    # The module must still define load_environment (structural integrity check).
    assert "def load_environment" in source, (
        "load_environment not found in generated module after hostile id injection."
    )

    # The _SCENARIO_ID assignment must be present and use repr() output, meaning
    # the id is enclosed in a valid Python string literal (not injected raw).
    assert "_SCENARIO_ID = " in source, "_SCENARIO_ID assignment missing from generated source."

    # The raw hostile payload must NOT appear verbatim as an unquoted sequence.
    # The comment line has the sanitized label (no triple-quote, no newline).
    # Find the comment line and confirm it does not contain triple-double-quote.
    comment_line = next(
        (ln for ln in source.splitlines() if ln.startswith("# Generated verifiers")),
        None,
    )
    assert comment_line is not None, "header comment line not found in generated source"
    assert '"""' not in comment_line, (
        f"Unsanitized triple-double-quote found in the comment header line: {comment_line!r}"
    )
    assert "\n" not in comment_line, (
        "Newline found inside the header comment line (should be single line)"
    )


def test_write_verifiers_env_hostile_id_produces_valid_python(tmp_path):
    """write_verifiers_env with a hostile scenario.id produces a valid module.

    Uses an id that is valid as a filesystem path component (no quotes or
    newlines in the id itself) but contains a backslash sequence that would
    have caused a syntax error in the old raw-interpolation code path.

    This test exercises the full artifact emitter path (file I/O included)
    and confirms that compile() succeeds on the generated file.

    Regression for the code-injection / syntax-break bug reported in Dispatch D.
    """
    from korrel.exporters.verifiers import write_verifiers_env

    # Use a backslash-containing id that is safe as a path component (single
    # backslash is a path separator on Windows, so _safe_filename_stem will
    # extract only the final component; the repr() of the resulting string
    # will still include escaped backslash sequences in the generated file).
    hostile_id = "my-scenario\\extra"

    scenario = _make_scenario_for_artifact(hostile_id)
    out = write_verifiers_env(
        scenario,
        tmp_path / "export",
        scenario_attr="scenario",
    )

    # Locate the generated env module (name is derived from the sanitized id).
    py_files = [p for p in out.iterdir() if p.suffix == ".py" and p.name != "_scenario.py"]
    assert py_files, "no env module was written"
    env_module_path = py_files[0]

    source = env_module_path.read_text(encoding="utf-8")

    # Key assertion: the generated source must parse without SyntaxError.
    try:
        compile(source, str(env_module_path), "exec")
    except SyntaxError as exc:
        raise AssertionError(
            f"Generated module has a syntax error: {exc}\n\nSource:\n{source}"
        ) from exc

    # The module must still define load_environment (structural integrity check).
    assert "def load_environment" in source, (
        "load_environment not found in generated module after hostile id."
    )
