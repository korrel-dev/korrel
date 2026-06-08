"""Korrel to verifiers exporter (v0.2).

Translates a Korrel ``Scenario`` into a ``verifiers`` environment. All
verifiers imports are deferred to function bodies; importing this module does
not require ``verifiers`` to be installed. Only calling the public functions
does.

Confirmed against: ``verifiers==0.1.14`` (``verifiers-0.1.14.dist-info/METADATA``).
verifiers requires Python <3.14,>=3.10, so this exporter cannot be exercised
in the Korrel repo's Python 3.14 venv. Generated artifacts run in a separate
environment that satisfies verifiers' constraint.

Mapping summary (per ``docs/spec/korrel-to-verifiers.md``):

- Scenario with persona, tools, or max_turns>1  ->  MultiTurnEnv subclass.
- Scenario with max_turns==1, no tools, no persona follow-up  ->  SingleTurnEnv.
- Rubric reward functions are wrapped to convert the verifiers completion
  (list of flat vf.Message) to the Korrel canonical form (list of korrel.Message)
  before calling the Korrel reward function.
- Weights are set to 1/n for n reward functions so the verifiers weighted sum
  equals Korrel's arithmetic mean.
- The per-turn tool-round cap from scenario.max_tool_rounds is enforced inside
  env_response via a per-turn counter stored in verifiers state.
- verifiers max_turns is set to scenario.max_turns * (1 + scenario.max_tool_rounds)
  to cover Korrel's worst-case step budget.
- The artifact emitter writes a pip-installable package without importing
  verifiers; only loading the generated package requires verifiers.
"""

from __future__ import annotations

import inspect
import json
import shutil
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ..scenario import Scenario

if TYPE_CHECKING:
    # Type-only: these are never imported at module load when verifiers is absent.
    import verifiers as vf

# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

_VERIFIERS_INSTALL_HINT = (
    "korrel's verifiers exporter requires the 'verifiers' package. "
    "Install it with: pip install 'korrel[verifiers]'  "
    "(or: uv add 'korrel[verifiers]'). "
    "verifiers requires Python <3.14."
)


def _import_verifiers() -> Any:
    """Import verifiers lazily; raise a clear error if absent."""
    try:
        import verifiers as vf  # type: ignore[import-not-found]

        return vf
    except ImportError as exc:
        raise ImportError(_VERIFIERS_INSTALL_HINT) from exc


def _import_datasets() -> Any:
    """Import datasets lazily (present when verifiers is installed)."""
    try:
        import datasets  # type: ignore[import-not-found]

        return datasets
    except ImportError as exc:
        raise ImportError(
            "korrel's verifiers exporter requires the 'datasets' package, "
            "which ships as a verifiers dependency. "
            "Install it with: pip install 'korrel[verifiers]'."
        ) from exc


# ---------------------------------------------------------------------------
# Internal conversion helpers (no verifiers import at definition time)
# ---------------------------------------------------------------------------


def _field(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a verifiers message or tool call.

    The value may arrive as a pydantic object attribute or a plain dict entry.
    Unlike ``getattr(...) or dict.get(...)``, this does not treat a falsy-but-
    present value (an empty string content) as missing: only ``None`` or a
    truly absent key falls back to ``default``.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    value = getattr(obj, key, default)
    return default if value is None else value


def _content_to_str(content: Any) -> str:
    """Narrow verifiers MessageContent (str | list[ContentPart]) to str.

    Verifiers content may be a list of content parts (text, image, audio).
    Korrel canonical content is Optional[str]. String content passes through;
    list content is serialized as JSON per the spec (lossy edge: content-shape
    narrowing documented in the mapping spec).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # List of content parts: join text parts, serialize non-text as JSON.
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
            else:
                parts.append(json.dumps(part))
        elif hasattr(part, "type"):
            if getattr(part, "type", None) == "text":
                parts.append(getattr(part, "text", ""))
            else:
                parts.append(json.dumps(part.model_dump() if hasattr(part, "model_dump") else str(part)))
        else:
            parts.append(str(part))
    return "".join(parts)


def _to_korrel_messages(vf_messages: list[Any]) -> list[Any]:
    """Convert a list of verifiers Messages to korrel canonical Messages.

    Field-by-field per the mapping spec (Section: Reward-function value shim):
    - system/user/assistant without tool calls: content narrowed to str.
    - assistant with tool calls: FLAT vf.ToolCall{id,name,arguments} ->
      NESTED korrel.ToolCall{id,type:"function",function:{name,arguments}}.
    - tool result: ToolMessage{role:"tool",tool_call_id,content}.
    ``arguments`` stays a JSON string on both sides.
    """
    from ..types import Message, ToolCall, ToolFunction

    result: list[Message] = []
    for msg in vf_messages:
        role = _field(msg, "role")
        if role is None:
            continue

        if role == "system":
            result.append(Message(role="system", content=_content_to_str(_field(msg, "content"))))

        elif role == "user":
            result.append(Message(role="user", content=_content_to_str(_field(msg, "content"))))

        elif role == "assistant":
            content = _field(msg, "content")
            vf_tool_calls = _field(msg, "tool_calls")
            korrel_tool_calls: Optional[list[ToolCall]] = None
            if vf_tool_calls:
                korrel_tool_calls = []
                for tc in vf_tool_calls:
                    tc_id = _field(tc, "id", "") or ""
                    tc_name = _field(tc, "name", "") or ""
                    tc_args = _field(tc, "arguments", "{}") or "{}"
                    korrel_tool_calls.append(
                        ToolCall(
                            id=tc_id,
                            type="function",
                            function=ToolFunction(name=tc_name, arguments=tc_args),
                        )
                    )
            result.append(
                Message(
                    role="assistant",
                    content=_content_to_str(content) if content is not None else None,
                    tool_calls=korrel_tool_calls or None,
                )
            )

        elif role == "tool":
            result.append(
                Message(
                    role="tool",
                    content=_content_to_str(_field(msg, "content")),
                    tool_call_id=_field(msg, "tool_call_id"),
                )
            )

    return result


# ---------------------------------------------------------------------------
# Reward function wrapper
# ---------------------------------------------------------------------------


def _wrap_reward_fn(korrel_fn: Any, name: str) -> Any:
    """Wrap a Korrel reward fn for use in a verifiers Rubric.

    The Korrel reward fn has signature (completion, info, **kwargs) -> float
    (sync or async). verifiers calls the wrapped fn with a merged kwargs dict
    that includes ``completion`` (list of vf.Message) and ``info`` (dict).

    The wrapper:
    1. Converts vf.Messages -> list[korrel.Message] via _to_korrel_messages.
    2. Calls the Korrel fn with the converted completion and info.
    3. If the Korrel fn returns an awaitable, the wrapper is async.
    4. Coerces the result to float.
    5. Sets __name__ to the Korrel fn name so verifiers metrics are keyed correctly.
    """

    if inspect.iscoroutinefunction(korrel_fn):

        async def async_wrapper(completion: Any = None, info: Any = None, **kwargs: Any) -> float:
            korrel_messages = _to_korrel_messages(completion or [])
            result = await korrel_fn(korrel_messages, info or {}, **kwargs)
            return float(result)

        async_wrapper.__name__ = name
        return async_wrapper

    else:

        def sync_wrapper(completion: Any = None, info: Any = None, **kwargs: Any) -> float:
            korrel_messages = _to_korrel_messages(completion or [])
            result = korrel_fn(korrel_messages, info or {}, **kwargs)
            # Handle the case where a sync function returns an awaitable (edge case).
            if inspect.isawaitable(result):
                import asyncio

                result = asyncio.run(result)
            return float(result)

        sync_wrapper.__name__ = name
        return sync_wrapper


# ---------------------------------------------------------------------------
# MultiTurnEnv class factory
# ---------------------------------------------------------------------------


def _build_multiturn_env_class(vf: Any) -> type:
    """Return a KorrelMultiTurnEnv class that subclasses vf.MultiTurnEnv.

    Defined inside a factory so that ``class KorrelMultiTurnEnv(vf.MultiTurnEnv)``
    is only evaluated when verifiers is available.

    env_response logic (per the mapping spec, Section: Mock tools to env_response):

    Tool branch (last assistant message has tool_calls):
    - Mirrors runtime.py::_resolve_tool_call exactly.
    - Tracks per-user-turn tool rounds via state["_korrel_tool_round"].
    - Stops returning tool results once max_tool_rounds is reached.

    Persona branch (no tool calls in last assistant message):
    - Converts conversation to korrel Messages.
    - Calls persona.next_message(messages).
    - Returns [vf.UserMessage(content=...)] if non-empty.
    - Sets state["final_env_response"] to [] (truthy-for-has_final_env_response
      check: has_final_env_response fires when state["final_env_response"] is not None)
      and returns [] if persona returns None/empty.

    State layout:
    - state["_korrel_tool_state"]: mutable dict passed to MockTool.call (per-run).
    - state["_korrel_tool_round"]: int counting tool rounds within the current user turn.
    - state["_korrel_user_turn"]: int counting user turns, used to reset tool_round.
    """

    class KorrelMultiTurnEnv(vf.MultiTurnEnv):  # type: ignore[misc]
        """A verifiers MultiTurnEnv that drives a Korrel Scenario.

        Constructed via ``to_verifiers_env``; do not instantiate directly.
        """

        # Populated by to_verifiers_env after construction.
        _korrel_persona: Any = None
        _korrel_tools_by_name: dict = {}
        _korrel_max_tool_rounds: int = 8

        async def env_response(
            self, messages: Any, state: Any, **kwargs: Any
        ) -> Any:
            """Drive one environment step for a Korrel scenario.

            Branches on whether the last assistant message carries tool_calls.
            """
            # Ensure per-run mutable state dicts exist.
            if "_korrel_tool_state" not in state:
                state["_korrel_tool_state"] = {}
            if "_korrel_tool_round" not in state:
                state["_korrel_tool_round"] = 0
            if "_korrel_user_turn" not in state:
                state["_korrel_user_turn"] = 0

            tool_state = state["_korrel_tool_state"]

            # Find the last assistant message.
            last_assistant = None
            for msg in reversed(messages):
                role = getattr(msg, "role", None)
                if role is None and isinstance(msg, dict):
                    role = msg.get("role")
                if role == "assistant":
                    last_assistant = msg
                    break

            has_tool_calls = False
            if last_assistant is not None:
                tc = getattr(last_assistant, "tool_calls", None)
                if tc is None and isinstance(last_assistant, dict):
                    tc = last_assistant.get("tool_calls")
                has_tool_calls = bool(tc)

            if has_tool_calls:
                # Tool branch: resolve each tool call.
                # Check per-turn tool-round cap.
                if state["_korrel_tool_round"] >= self._korrel_max_tool_rounds:
                    # Cap reached: signal completion via final_env_response.
                    state["final_env_response"] = []
                    return []

                tc_list = getattr(last_assistant, "tool_calls", None)
                if tc_list is None and isinstance(last_assistant, dict):
                    tc_list = last_assistant.get("tool_calls", [])

                tool_messages: list[Any] = []
                for tc in (tc_list or []):
                    tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else "")
                    tc_name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else "")
                    tc_args_str = getattr(tc, "arguments", None) or (tc.get("arguments") if isinstance(tc, dict) else "{}")

                    # Mirror runtime.py::_resolve_tool_call: tolerant JSON parse.
                    try:
                        arguments = json.loads(tc_args_str or "{}")
                    except json.JSONDecodeError:
                        arguments = {}

                    tool = self._korrel_tools_by_name.get(tc_name or "")
                    if tool is None:
                        result: Any = {"error": f"unknown tool: {tc_name}"}
                    else:
                        result = tool.call(arguments, tool_state)

                    content = result if isinstance(result, str) else json.dumps(result)
                    tool_messages.append(
                        vf.ToolMessage(tool_call_id=tc_id or "", content=content)
                    )

                state["_korrel_tool_round"] += 1
                return tool_messages

            else:
                # Persona branch: advance to the next user turn.
                # Reset tool round counter for this new user turn.
                state["_korrel_tool_round"] = 0
                state["_korrel_user_turn"] = state.get("_korrel_user_turn", 0) + 1

                korrel_messages = _to_korrel_messages(messages)
                next_msg = self._korrel_persona.next_message(korrel_messages)

                if next_msg:
                    return [vf.UserMessage(content=next_msg)]
                else:
                    # Persona exhausted: signal termination.
                    state["final_env_response"] = []
                    return []

    return KorrelMultiTurnEnv


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_PERSONA_NOT_SET = object()


def to_verifiers_env(
    scenario: Scenario,
    *,
    persona: Any = _PERSONA_NOT_SET,
) -> "vf.Environment":
    """Translate a Korrel Scenario into a verifiers Environment.

    Parameters
    ----------
    scenario:
        The Korrel Scenario to translate.
    persona:
        Optional persona override. When provided (including ``None``) it
        replaces ``scenario.persona`` as the user-simulator inside the
        generated environment. Pass a fake persona in tests to avoid live
        model calls. Pass ``None`` explicitly to produce a ``SingleTurnEnv``
        (no environment turn) when the scenario would otherwise select
        ``MultiTurnEnv`` because of its ``persona`` field.

    Returns
    -------
    vf.Environment
        A constructed verifiers environment (MultiTurnEnv or SingleTurnEnv)
        ready for rollout generation.

    Raises
    ------
    ImportError
        If the ``verifiers`` package is not installed.
    ValueError
        If ``scenario.rubric`` is None (a reward-less RL environment is
        meaningless).
    """
    # Rubric check fires before the verifiers import so it works even when
    # verifiers is absent (useful for validation in the authoring workflow).
    if scenario.rubric is None:
        raise ValueError(
            "Scenario.rubric is required to export to verifiers. "
            "A reward-less RL environment is meaningless. "
            "Attach a Rubric with at least one reward function to the Scenario."
        )

    vf = _import_verifiers()
    datasets_lib = _import_datasets()

    # _PERSONA_NOT_SET: caller did not provide an override; use scenario.persona.
    # Any other value (including None): use as the active persona.
    if persona is _PERSONA_NOT_SET:
        active_persona = scenario.persona
    else:
        active_persona = persona

    # --- Decision rule: SingleTurnEnv vs MultiTurnEnv ---
    has_tools = bool(scenario.tools)
    has_persona_follow_up = active_persona is not None
    is_single_turn = (
        scenario.max_turns == 1
        and not has_tools
        and not has_persona_follow_up
    )

    # --- Dataset row ---
    prompt: list[dict[str, str]] = []
    if scenario.system:
        prompt.append({"role": "system", "content": scenario.system})
    prompt.append({"role": "user", "content": scenario.opening_message})

    row: dict[str, Any] = {
        "prompt": prompt,
        "info": scenario.info or {},
    }
    dataset = datasets_lib.Dataset.from_list([row])

    # --- Tool definitions ---
    tool_defs: list[Any] = []
    tools_by_name: dict[str, Any] = {}
    for mock_tool in scenario.tools:
        func_schema = mock_tool.schema.get("function", {})
        tool_defs.append(
            vf.Tool(
                name=func_schema.get("name", mock_tool.name),
                description=func_schema.get("description", ""),
                parameters=func_schema.get("parameters", {}),
            )
        )
        tools_by_name[mock_tool.name] = mock_tool

    # --- Rubric ---
    all_korrel_fns = list(scenario.rubric.funcs)
    if scenario.rubric.judge is not None:
        all_korrel_fns.append(scenario.rubric.judge)

    n = len(all_korrel_fns)
    if n == 0:
        raise ValueError(
            "Scenario.rubric has no reward functions. "
            "Add at least one reward function to export to verifiers."
        )

    uniform_weight = 1.0 / n
    wrapped_fns: list[Any] = []
    for i, fn in enumerate(all_korrel_fns):
        fn_name = getattr(fn, "__name__", None) or getattr(fn, "name", None)
        if not fn_name or fn_name == "<lambda>":
            fn_name = f"reward_{i}"
        wrapped_fns.append(_wrap_reward_fn(fn, fn_name))

    vf_rubric = vf.Rubric(
        funcs=wrapped_fns,
        weights=[uniform_weight] * n,
    )

    # --- verifiers max_turns budget (spec: max_turns * (1 + max_tool_rounds)) ---
    vf_max_turns = scenario.max_turns * (1 + scenario.max_tool_rounds)

    if is_single_turn:
        env = vf.SingleTurnEnv(
            dataset=dataset,
            rubric=vf_rubric,
            tool_defs=tool_defs if tool_defs else None,
            pass_threshold=scenario.rubric.pass_threshold,
            env_id=scenario.id,
        )
    else:
        KorrelMultiTurnEnv = _build_multiturn_env_class(vf)
        env = KorrelMultiTurnEnv(
            dataset=dataset,
            rubric=vf_rubric,
            tool_defs=tool_defs if tool_defs else None,
            max_turns=vf_max_turns,
            pass_threshold=scenario.rubric.pass_threshold,
            env_id=scenario.id,
        )
        # Attach Korrel-specific attributes after construction.
        env._korrel_persona = active_persona
        env._korrel_tools_by_name = tools_by_name
        env._korrel_max_tool_rounds = scenario.max_tool_rounds

    return env


# ---------------------------------------------------------------------------
# Artifact emitter
# ---------------------------------------------------------------------------

_PYPROJECT_TEMPLATE = """\
[project]
name = "{env_id}"
description = "Korrel scenario exported as a verifiers environment."
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "verifiers>=0.1.14",
    "korrel",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["{env_module}.py", "_scenario.py", "pyproject.toml"]
"""

_ENV_MODULE_TEMPLATE = '''\
"""Generated verifiers environment for Korrel scenario: {scenario_id}.

This module was produced by ``korrel.exporters.verifiers.write_verifiers_env``.
It imports the original scenario source (copied as ``_scenario.py``) and
translates it to a verifiers Environment via ``to_verifiers_env``.

The scenario source contains Python callables (tools, persona, reward functions)
that cannot be serialized to JSON, so the original source is included in the
package. Any changes to the original scenario file must be propagated here by
re-running ``korrel export``.

Built against: verifiers==0.1.14.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_scenario():
    """Load the scenario object from the bundled _scenario.py."""
    src = Path(__file__).parent / "_scenario.py"
    spec = importlib.util.spec_from_file_location("_scenario", src)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load scenario from {{src}}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    scenario = getattr(mod, {scenario_attr!r}, None)
    if scenario is None:
        raise AttributeError(
            f"_scenario.py has no attribute {scenario_attr!r}. "
            "Check that the scenario attribute name matches the original file."
        )
    return scenario


def load_environment(**kwargs):
    """Return a verifiers Environment for the bundled Korrel scenario.

    Accepts an optional ``persona`` keyword argument to override the
    scenario's default persona (useful for testing without live model calls).
    """
    from korrel.exporters.verifiers import to_verifiers_env

    scenario = _load_scenario()
    # Only override the scenario's own persona when the caller actually passes
    # one. Forwarding persona=None unconditionally would strip the scenario
    # persona and force the single-turn path for persona-driven scenarios.
    if "persona" in kwargs:
        return to_verifiers_env(scenario, persona=kwargs["persona"])
    return to_verifiers_env(scenario)
'''


def write_verifiers_env(
    scenario: Scenario,
    out_dir: Path,
    *,
    scenario_source_path: Optional[Path] = None,
    scenario_attr: str = "scenario",
) -> Path:
    """Write a pip-installable verifiers environment package for a Korrel Scenario.

    Produces:
      ``<out_dir>/pyproject.toml``
      ``<out_dir>/<env_module>.py``
      ``<out_dir>/_scenario.py``  (copy of the original scenario source)

    The package declares dependencies ``verifiers>=0.1.14`` and ``korrel``.
    Install it with ``pip install -e <out_dir>`` in a Python <3.14 environment,
    then use ``vf.load_environment("<env_id>")`` to construct the environment.

    Parameters
    ----------
    scenario:
        The Korrel Scenario whose id and definition to export.
    out_dir:
        Directory where the package files are written. Created if absent.
    scenario_source_path:
        Path to the Python file that defines the Scenario. Its content is
        copied into the package as ``_scenario.py``. When None, the emitter
        writes a placeholder that raises ImportError at runtime.
    scenario_attr:
        The module-level attribute name for the Scenario in the source file
        (default: ``"scenario"``).

    Returns
    -------
    Path
        The ``out_dir`` containing the generated package.

    Notes
    -----
    This function does NOT import verifiers. It writes text files only, so
    the artifact can be produced on Python 3.14 where verifiers cannot be
    installed. Only running ``load_environment()`` from the generated package
    requires verifiers.
    """
    from ..cli import _safe_filename_stem

    # scenario_attr is interpolated into the generated module source. Reject any
    # value that is not a plain Python identifier so it cannot inject braces into
    # the format template or a misleading attribute name into the artifact
    # (security review K10). The CLI validates this too; the guard is repeated
    # here because write_verifiers_env is a public entry point.
    if not scenario_attr.isidentifier():
        raise ValueError(
            f"scenario_attr must be a valid Python identifier, got {scenario_attr!r}."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # env_id: sanitize scenario.id with the same rule used in cli.py.
    raw_id = _safe_filename_stem(scenario.id)
    # Convert to a valid Python identifier for the module name.
    env_id = raw_id.replace(" ", "-").replace("_", "-")
    env_module = env_id.replace("-", "_")

    # pyproject.toml
    pyproject_path = out_dir / "pyproject.toml"
    pyproject_path.write_text(
        _PYPROJECT_TEMPLATE.format(env_id=env_id, env_module=env_module),
        encoding="utf-8",
    )

    # <env_module>.py
    env_module_path = out_dir / f"{env_module}.py"
    env_module_path.write_text(
        _ENV_MODULE_TEMPLATE.format(
            scenario_id=scenario.id,
            scenario_attr=scenario_attr,
        ),
        encoding="utf-8",
    )

    # _scenario.py: copy the original source or write a placeholder.
    scenario_dest = out_dir / "_scenario.py"
    if scenario_source_path is not None and Path(scenario_source_path).exists():
        shutil.copy2(scenario_source_path, scenario_dest)
    else:
        placeholder = textwrap.dedent(
            f"""\
            # Placeholder: the original scenario source was not available when
            # this package was generated. Replace this file with the real source
            # that defines a module-level ``{scenario_attr}`` attribute.
            raise ImportError(
                "The scenario source was not bundled during export. "
                "Copy the original scenario file here as _scenario.py "
                "and ensure it defines a module-level {scenario_attr!r} attribute."
            )
            """
        )
        scenario_dest.write_text(placeholder, encoding="utf-8")

    return out_dir
