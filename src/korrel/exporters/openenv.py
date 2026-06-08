"""Korrel to OpenEnv exporter (v0.3).

Translates a Korrel ``Scenario`` into an OpenEnv environment package. All
openenv imports are deferred to function bodies; importing this module does not
require ``openenv-core`` to be installed. The text emitter (``write_openenv_env``)
imports no openenv at all and can run on any Python >= 3.10 regardless of
whether openenv-core is present.

Confirmed against: openenv-core==0.3.0 (``importlib.metadata.version
("openenv-core") == "0.3.0"``; confirmed in a throwaway Python 3.14 venv and
cross-checked in Python 3.12). All openenv interface claims cite the module
and symbol they were read from.

Mapping summary (per ``docs/spec/korrel-to-openenv.md``):

- ``reset(seed, episode_id)`` builds the conversation seed from
  ``scenario.system`` and ``scenario.opening_message`` and returns a Korrel
  Observation carrying the opening user message, ``done=False``,
  ``reward=None``.
- ``step(action)`` branches on whether the action carries tool_calls:
    Tool branch: resolve each call (mirroring ``runtime.py::_resolve_tool_call``
    exactly), return tool-result messages, ``done=False``, ``reward=None``.
    Persona branch: advance persona, return the user message (or terminate).
  Termination (persona exhausted or ``max_turns`` reached): ``done=True``,
  ``reward = scenario.rubric.score(messages, info).score``.
- Reward is terminal only. Every intermediate step carries ``reward=None``.
- The OpenEnv ``rubric`` base class (``action, observation`` signature) is NOT
  used; the Korrel ``Rubric.score(completion, info)`` path is called directly.
- Author-defined ``Action`` subclass carries assistant ``content`` and
  ``tool_calls`` (each tool call has ``id``, ``name``, ``arguments`` as a JSON
  string per the canonical type). Author-defined ``Observation`` subclass
  carries environment reply ``messages`` plus inherited ``done`` and ``reward``.
"""

from __future__ import annotations

import inspect
import json
import shutil
import textwrap
from pathlib import Path
from typing import Any, Optional

from ..scenario import Scenario
from ._shared import _content_to_str, _field, _to_korrel_messages  # noqa: F401

# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

_OPENENV_INSTALL_HINT = (
    "korrel's openenv exporter requires the 'openenv-core' package. "
    "Install it with: pip install 'korrel[openenv]'  "
    "(or: uv add 'korrel[openenv]'). "
    "openenv-core requires Python >=3.10."
)


def _import_openenv_core() -> Any:
    """Import openenv.core lazily; raise a clear error if absent.

    Always imports from ``openenv.core``, never from the deprecated alias
    ``openenv_core`` (openenv-core 0.3.0: the top-level ``openenv_core`` module
    emits a DeprecationWarning on import).
    """
    try:
        import openenv.core as oe_core  # type: ignore[import-not-found]

        return oe_core
    except ImportError as exc:
        raise ImportError(_OPENENV_INSTALL_HINT) from exc


# ---------------------------------------------------------------------------
# Server-side adapter helpers
# These implement reset/step logic against Korrel canonical types. All openenv
# imports are deferred so generated packages may call helpers without the host
# environment needing openenv installed at import time.
# ---------------------------------------------------------------------------

_PERSONA_NOT_SET = object()


def seed_observation(
    scenario: Scenario,
    observation_cls: Any,
) -> Any:
    """Build the reset observation for ``scenario``.

    Mirrors how ``run_scenario`` seeds ``messages``: a system message when
    ``scenario.system`` is set, then the opening user message.

    Returns an observation instance (``observation_cls``) with ``done=False``,
    ``reward=None``, and ``messages`` carrying the opening user message.

    ``openenv.core.env_server.types.Observation`` is the base of
    ``observation_cls``; ``done`` and ``reward`` are inherited fields on it
    (``core/env_server/types.py``: ``done: bool = False``,
    ``reward: bool | int | float | None = None``).
    """
    from ..types import Message

    seed: list[Message] = []
    if scenario.system:
        seed.append(Message(role="system", content=scenario.system))
    seed.append(Message(role="user", content=scenario.opening_message))

    messages_dicts = [m.model_dump(exclude_none=True) for m in seed]
    return observation_cls(
        messages=messages_dicts,
        done=False,
        reward=None,
    )


def advance(
    scenario: Scenario,
    state_messages: list[Any],
    tool_round_state: dict[str, Any],
    action: Any,
    observation_cls: Any,
    *,
    persona: Any = _PERSONA_NOT_SET,
) -> Any:
    """Advance the environment by one step and return the next observation.

    Parameters
    ----------
    scenario:
        The Korrel Scenario driving this episode.
    state_messages:
        The mutable list of canonical ``Message`` objects for the episode. This
        list is appended to in place.
    tool_round_state:
        A mutable dict with the following keys:
        - ``"tool_round"``: int counting tool rounds in the current user turn.
        - ``"user_turns"``: int counting completed user turns.
        - ``"tool_state"``: mutable dict passed to ``MockTool.call``.
        Both counts are reset or incremented by this function.
    action:
        The author-defined ``Action`` subclass instance from the policy. Fields:
        - ``content``: ``Optional[str]``, the assistant text (may be None when
          the action only carries tool calls).
        - ``tool_calls``: ``Optional[list[dict]]``, each dict has ``id``,
          ``name``, ``arguments`` (JSON string).
    observation_cls:
        The author-defined ``Observation`` subclass to instantiate.
    persona:
        Optional persona override. When provided (including ``None``) it
        replaces ``scenario.persona``.

    Returns
    -------
    observation_cls instance
        ``done=False, reward=None`` on intermediate steps.
        ``done=True, reward=<float>`` on the terminal step (persona exhausted
        or ``scenario.max_turns`` reached).

    Notes
    -----
    The step branch mirrors ``runtime.py::_resolve_tool_call`` and the inner
    loop of ``run_scenario`` exactly, per the spec (Section: Step to
    environment reply).
    """
    from ..types import Message, ToolCall, ToolFunction

    # Resolve the active persona.
    if persona is _PERSONA_NOT_SET:
        active_persona = scenario.persona
    else:
        active_persona = persona

    tools_by_name = {t.name: t for t in scenario.tools}

    # Convert the action to a canonical assistant Message and append it.
    action_content: Optional[str] = getattr(action, "content", None)
    raw_tool_calls: Optional[list[Any]] = getattr(action, "tool_calls", None)

    korrel_tool_calls: Optional[list[ToolCall]] = None
    if raw_tool_calls:
        korrel_tool_calls = []
        for tc in raw_tool_calls:
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

    assistant_msg = Message(
        role="assistant",
        content=action_content,
        tool_calls=korrel_tool_calls or None,
    )
    state_messages.append(assistant_msg)

    # --- Tool branch ---
    if korrel_tool_calls:
        tool_round = tool_round_state.get("tool_round", 0)

        if tool_round >= scenario.max_tool_rounds:
            # Cap reached: terminate the episode (mirror run_scenario's break
            # with stop_reason="max_tool_rounds").
            reward = _compute_reward(scenario, state_messages)
            return observation_cls(
                messages=[],
                done=True,
                reward=reward,
            )

        tool_result_messages: list[Message] = []
        for call in korrel_tool_calls:
            # Mirror runtime.py::_resolve_tool_call exactly.
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            tool = tools_by_name.get(call.function.name)
            if tool is None:
                result: Any = {"error": f"unknown tool: {call.function.name}"}
            else:
                result = tool.call(arguments, tool_round_state["tool_state"])

            content = result if isinstance(result, str) else json.dumps(result)
            tool_msg = Message(role="tool", content=content, tool_call_id=call.id)
            tool_result_messages.append(tool_msg)
            state_messages.append(tool_msg)

        tool_round_state["tool_round"] = tool_round + 1

        result_dicts = [m.model_dump(exclude_none=True) for m in tool_result_messages]
        return observation_cls(
            messages=result_dicts,
            done=False,
            reward=None,
        )

    # --- Persona branch ---
    # Reset the tool round counter for this new user turn.
    tool_round_state["tool_round"] = 0
    user_turns = tool_round_state.get("user_turns", 0)

    # Check if the user-turn count has reached max_turns.
    if user_turns >= scenario.max_turns:
        reward = _compute_reward(scenario, state_messages)
        return observation_cls(
            messages=[],
            done=True,
            reward=reward,
        )

    if active_persona is None:
        # No persona configured: terminate after the first assistant turn.
        reward = _compute_reward(scenario, state_messages)
        return observation_cls(
            messages=[],
            done=True,
            reward=reward,
        )

    next_user_text = active_persona.next_message(state_messages)
    tool_round_state["user_turns"] = user_turns + 1

    if not next_user_text:
        # Persona exhausted.
        reward = _compute_reward(scenario, state_messages)
        return observation_cls(
            messages=[],
            done=True,
            reward=reward,
        )

    user_msg = Message(role="user", content=next_user_text)
    state_messages.append(user_msg)

    return observation_cls(
        messages=[user_msg.model_dump(exclude_none=True)],
        done=False,
        reward=None,
    )


def _compute_reward(scenario: Scenario, messages: list[Any]) -> float:
    """Compute the terminal scalar from the rubric.

    Returns 0.0 when ``scenario.rubric`` is None (no rubric configured).
    Uses ``Rubric.score(messages, info).score`` (arithmetic mean of all reward
    functions) per the spec (Section: Reward to terminal observation reward).
    """
    if scenario.rubric is None:
        return 0.0
    result = scenario.rubric.score(messages, scenario.info or {})
    return float(result.score)


def build_environment_class(
    scenario: Scenario,
    observation_cls: Any,
    action_cls: Any,
    *,
    persona: Any = _PERSONA_NOT_SET,
) -> type:
    """Return a concrete ``Environment`` subclass for ``scenario``.

    All openenv imports are deferred into this function body.

    Parameters
    ----------
    scenario:
        The Korrel Scenario to adapt.
    observation_cls:
        The author-defined Observation subclass (subclass of
        ``openenv.core.env_server.types.Observation``).
    action_cls:
        The author-defined Action subclass.
    persona:
        Optional persona override for testing without live model calls.

    Returns
    -------
    type
        A concrete ``Environment`` subclass.

    Raises
    ------
    ImportError
        If ``openenv-core`` is not installed.
    """
    oe_core = _import_openenv_core()
    Environment = oe_core.env_server.interfaces.Environment
    State = oe_core.env_server.types.State

    # Resolve the active persona once at class-build time so tests can inject
    # a fake without touching the scenario object.
    if persona is _PERSONA_NOT_SET:
        active_persona = scenario.persona
    else:
        active_persona = persona

    class KorrelOpenEnvEnvironment(Environment):  # type: ignore[misc]
        """OpenEnv Environment driven by a Korrel Scenario."""

        SUPPORTS_CONCURRENT_SESSIONS: bool = True

        def __init__(self) -> None:
            super().__init__()
            from uuid import uuid4

            self._korrel_messages: list[Any] = []
            self._korrel_tool_round_state: dict[str, Any] = {
                "tool_round": 0,
                "user_turns": 0,
                "tool_state": {},
            }
            self._state = State(episode_id=str(uuid4()), step_count=0)

        def reset(
            self,
            seed: Optional[int] = None,
            episode_id: Optional[str] = None,
            **kwargs: Any,
        ) -> Any:
            """Reset the episode and return the opening observation."""
            from uuid import uuid4

            self._korrel_messages = []
            self._korrel_tool_round_state = {
                "tool_round": 0,
                "user_turns": 0,
                "tool_state": {},
            }
            ep_id = episode_id if episode_id is not None else str(uuid4())
            self._state = State(episode_id=ep_id, step_count=0)

            obs = seed_observation(scenario, observation_cls)
            # Populate the internal message list from the reset observation.
            from ..types import Message

            if scenario.system:
                self._korrel_messages.append(
                    Message(role="system", content=scenario.system)
                )
            self._korrel_messages.append(
                Message(role="user", content=scenario.opening_message)
            )

            return obs

        def step(
            self,
            action: Any,
            timeout_s: Optional[float] = None,
            **kwargs: Any,
        ) -> Any:
            """Advance the episode by one action."""
            self._state.step_count += 1
            obs = advance(
                scenario,
                self._korrel_messages,
                self._korrel_tool_round_state,
                action,
                observation_cls,
                persona=active_persona,
            )
            return obs

        @property
        def state(self) -> Any:
            """Return the current episode State."""
            return self._state

    KorrelOpenEnvEnvironment.__name__ = "KorrelOpenEnvEnvironment"
    KorrelOpenEnvEnvironment.__qualname__ = "KorrelOpenEnvEnvironment"
    return KorrelOpenEnvEnvironment


# ---------------------------------------------------------------------------
# Scenario id safety helpers (shared pattern with verifiers exporter)
# ---------------------------------------------------------------------------


def _sanitize_id_for_comment(scenario_id: str) -> str:
    """Return a display-safe single-line label for use in a comment or header.

    Strips leading/trailing whitespace, collapses newlines to a space, and
    removes every occurrence of triple-double-quote so the label cannot close
    or escape from any surrounding string region in the generated module.
    The exact scenario id is always preserved separately via repr().
    """
    label = scenario_id.strip().replace("\n", " ").replace("\r", " ")
    label = label.replace('"""', "")
    return label


# ---------------------------------------------------------------------------
# Text templates for the generated package
# ---------------------------------------------------------------------------

# fmt: off

_MODELS_PY = '''\
# Generated by korrel.exporters.openenv.write_openenv_env.
# Built against: openenv-core==0.3.0.
#
# Action and Observation models for the Korrel scenario: {scenario_id_label}
# Do not edit this file by hand; re-run ``korrel export`` to regenerate it.

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class KorrelAction(Action):
    """Action sent by the RL policy for the Korrel scenario.

    Fields mirror the canonical Korrel assistant message shape
    (src/korrel/types.py, class Message and class ToolCall).

    ``content`` is the assistant text reply (None when the turn only calls
    tools).  ``tool_calls`` is the list of tool calls; each entry is a dict
    with keys ``id``, ``name``, and ``arguments`` where ``arguments`` is a
    JSON-encoded string (canonical ToolFunction.arguments shape).
    """

    content: Optional[str] = Field(default=None, description="Assistant text content")
    tool_calls: Optional[List[Dict]] = Field(
        default=None,
        description=(
            "Tool calls from the assistant. Each dict has id, name, and "
            "arguments (JSON-encoded string)."
        ),
    )


class KorrelObservation(Observation):
    """Observation returned by the environment for the Korrel scenario.

    ``messages`` is a list of message dicts (each with at least a ``role``
    key) representing the environment reply for this step. On a tool turn the
    list contains the tool-result messages; on a persona turn it contains the
    next user message; on the terminal step the list is empty.

    ``done`` and ``reward`` are inherited from
    ``openenv.core.env_server.types.Observation``.
    """

    messages: List[Dict] = Field(
        default_factory=list,
        description="Environment reply messages for this step.",
    )
'''

_ENVIRONMENT_PY = '''\
# Generated by korrel.exporters.openenv.write_openenv_env.
# Built against: openenv-core==0.3.0.
#
# Environment implementation for Korrel scenario: {scenario_id_label}
# Do not edit this file by hand; re-run ``korrel export`` to regenerate it.

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# _SCENARIO_ID holds the exact, unmodified scenario id.
_SCENARIO_ID = {scenario_id_repr}

try:
    from ..models import KorrelAction, KorrelObservation
except ImportError:
    from models import KorrelAction, KorrelObservation


def _load_scenario():
    """Load the scenario object from the bundled _scenario.py."""
    src = Path(__file__).parent.parent / "_scenario.py"
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


class KorrelEnvironment(Environment):
    """OpenEnv Environment driven by the bundled Korrel Scenario.

    reset() seeds the conversation from the scenario.
    step(action) advances the episode by one policy turn.
    state returns the current episode State.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._scenario = _load_scenario()
        self._messages: list[Any] = []
        self._tool_round_state: dict[str, Any] = {{
            "tool_round": 0,
            "user_turns": 0,
            "tool_state": {{}},
        }}
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> KorrelObservation:
        """Reset the episode and return the opening observation."""
        from korrel.exporters.openenv import seed_observation
        from korrel.types import Message

        ep_id = episode_id if episode_id is not None else str(uuid4())
        self._state = State(episode_id=ep_id, step_count=0)
        self._messages = []
        self._tool_round_state = {{
            "tool_round": 0,
            "user_turns": 0,
            "tool_state": {{}},
        }}

        obs = seed_observation(self._scenario, KorrelObservation)
        if self._scenario.system:
            self._messages.append(
                Message(role="system", content=self._scenario.system)
            )
        self._messages.append(
            Message(role="user", content=self._scenario.opening_message)
        )
        return obs

    def step(
        self,
        action: KorrelAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> KorrelObservation:
        """Advance the episode by one policy action."""
        from korrel.exporters.openenv import advance

        self._state.step_count += 1
        return advance(
            self._scenario,
            self._messages,
            self._tool_round_state,
            action,
            KorrelObservation,
        )

    @property
    def state(self) -> State:
        """Return the current episode State."""
        return self._state
'''

_APP_PY = '''\
# Generated by korrel.exporters.openenv.write_openenv_env.
# Built against: openenv-core==0.3.0.
#
# FastAPI application for the Korrel scenario environment: {scenario_id_label}

"""
FastAPI application for the Korrel scenario environment.

Endpoints:
    POST /reset  -- Reset the environment
    POST /step   -- Execute an action
    GET  /state  -- Get current state
    GET  /schema -- Get action/observation schemas
    WS   /ws     -- WebSocket endpoint for persistent sessions

Usage (development):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

Usage (direct):
    python -m server.app
    uv run --project . server
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core>=0.3.0"
    ) from exc

try:
    from ..models import KorrelAction, KorrelObservation
    from .{env_module}_environment import KorrelEnvironment
except ImportError:
    from models import KorrelAction, KorrelObservation
    from server.{env_module}_environment import KorrelEnvironment

app = create_app(
    KorrelEnvironment,
    KorrelAction,
    KorrelObservation,
    env_name="{env_name}",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for ``uv run --project . server`` and direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parsed = parser.parse_args()
    main(port=parsed.port)
'''

_SERVER_INIT_PY = '''\
# Generated by korrel.exporters.openenv.write_openenv_env.
"""Server package for the Korrel scenario environment."""
'''

_PACKAGE_INIT_PY = '''\
# Generated by korrel.exporters.openenv.write_openenv_env.
"""Korrel scenario environment package: {env_name}."""

from .client import KorrelEnv
from .models import KorrelAction, KorrelObservation

__all__ = ["KorrelAction", "KorrelObservation", "KorrelEnv"]
'''

_CLIENT_PY = '''\
# Generated by korrel.exporters.openenv.write_openenv_env.
# Built against: openenv-core==0.3.0.
"""Client for the Korrel scenario environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import KorrelAction, KorrelObservation


class KorrelEnv(EnvClient[KorrelAction, KorrelObservation, State]):
    """Client for the Korrel scenario environment server.

    Usage:
        with KorrelEnv(base_url="http://localhost:8000") as env:
            result = env.reset()
            result = env.step(KorrelAction(content="Hello"))
    """

    def _step_payload(self, action: KorrelAction) -> Dict:
        payload: Dict = {{}}
        if action.content is not None:
            payload["content"] = action.content
        if action.tool_calls is not None:
            payload["tool_calls"] = action.tool_calls
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[KorrelObservation]:
        obs_data = payload.get("observation", {{}})
        observation = KorrelObservation(
            messages=obs_data.get("messages", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {{}}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
'''

_OPENENV_YAML = '''\
spec_version: 1
name: {env_name}
type: space
runtime: fastapi
app: server.app:app
port: 8000
'''

_PYPROJECT_TOML = '''\
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "openenv-{env_name}"
description = "Korrel scenario exported as an OpenEnv environment."
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "openenv-core>=0.3.0",
    "korrel",
]

[project.scripts]
server = "{env_name}.server.app:main"

[tool.setuptools]
include-package-data = true
packages = ["{env_name}", "{env_name}.server"]
package-dir = {{ "{env_name}" = ".", "{env_name}.server" = "server" }}
'''

_README_MD = '''\
# {env_name} -- Korrel scenario environment

Generated by `korrel export --to openenv`. Built against openenv-core==0.3.0.

## Quick start

```bash
# Start the server locally
uvicorn server.app:app --reload --port 8000
```

```python
from {env_name} import KorrelAction, KorrelEnv

with KorrelEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    # result.observation.messages contains the opening user message
    result = env.step(KorrelAction(content="Hello"))
    # step until result.done is True
```

## Deploy to Hugging Face Spaces

```bash
openenv push --secret ANTHROPIC_API_KEY=<your-key>
```

The environment persona and judge make live model calls inside the container.
Supply the API key via `openenv push --secret`; do not write the key into any
file.

## Project structure

```
{env_name}/
    __init__.py
    client.py
    models.py
    openenv.yaml
    pyproject.toml
    README.md
    _scenario.py          (bundled Korrel scenario source)
    server/
        __init__.py
        {env_name}_environment.py
        app.py
        Dockerfile
        requirements.txt
```
'''

_DOCKERFILE = '''\
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${{BASE_IMAGE}} AS builder

WORKDIR /app

RUN apt-get update && \\
    apt-get install -y --no-install-recommends git && \\
    rm -rf /var/lib/apt/lists/*

ARG ENV_NAME={env_name}

COPY . /app/env

WORKDIR /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \\
        curl -LsSf https://astral.sh/uv/install.sh | sh && \\
        mv /root/.local/bin/uv /usr/local/bin/uv && \\
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \\
    fi

RUN --mount=type=cache,target=/root/.cache/uv \\
    uv sync --no-install-project --no-editable

RUN --mount=type=cache,target=/root/.cache/uv \\
    uv sync --no-editable

FROM ${{BASE_IMAGE}}

WORKDIR /app

COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# ANTHROPIC_API_KEY must be supplied at runtime via:
#   openenv push --secret ANTHROPIC_API_KEY=<your-key>
# Never write a key into this file.

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
'''

_REQUIREMENTS_TXT = '''\
openenv-core>=0.3.0
korrel
fastapi>=0.115.0
uvicorn>=0.24.0
'''

# fmt: on


# ---------------------------------------------------------------------------
# Public API: artifact emitter
# ---------------------------------------------------------------------------


def write_openenv_env(
    scenario: Scenario,
    out_dir: Path,
    *,
    scenario_source_path: Optional[Path] = None,
    scenario_attr: str = "scenario",
) -> Path:
    """Write a pip-installable OpenEnv environment package for a Korrel Scenario.

    Produces the ``openenv init`` file set with Korrel-specific content:

    ::

        <out_dir>/
            __init__.py
            client.py
            models.py
            openenv.yaml
            pyproject.toml
            README.md
            _scenario.py      (copy of the original source, or a placeholder)
            server/
                __init__.py
                <env_name>_environment.py
                app.py
                Dockerfile
                requirements.txt

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
    This function does NOT import openenv-core. It writes text files only and
    can run on any Python >= 3.10 regardless of whether openenv-core is
    installed. Only loading the generated server requires openenv-core.
    """
    from ..cli import _safe_filename_stem

    # scenario_attr is interpolated into generated source; reject non-identifiers
    # so it cannot inject braces or a misleading name (security review K10).
    if not scenario_attr.isidentifier():
        raise ValueError(
            f"scenario_attr must be a valid Python identifier, got {scenario_attr!r}."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive env_name from scenario.id using the same K9 guard as the CLI.
    raw_stem = _safe_filename_stem(scenario.id)
    env_name = raw_stem.replace(" ", "-").replace("_", "-")
    env_module = env_name.replace("-", "_")

    # scenario.id is author-controlled. repr() produces a syntactically valid
    # Python string literal for ANY id content (triple-quotes, newlines,
    # backslashes). The sanitized label is used only in comments.
    scenario_id_repr = repr(scenario.id)
    scenario_id_label = _sanitize_id_for_comment(scenario.id)

    # Shared substitution context for templates that need env_name or env_module.
    ctx = dict(
        env_name=env_name,
        env_module=env_module,
        scenario_id_label=scenario_id_label,
        scenario_id_repr=scenario_id_repr,
        scenario_attr=scenario_attr,
    )

    # models.py
    (out_dir / "models.py").write_text(
        _MODELS_PY.format(**ctx), encoding="utf-8"
    )

    # __init__.py
    (out_dir / "__init__.py").write_text(
        _PACKAGE_INIT_PY.format(**ctx), encoding="utf-8"
    )

    # client.py
    (out_dir / "client.py").write_text(
        _CLIENT_PY.format(**ctx), encoding="utf-8"
    )

    # openenv.yaml
    (out_dir / "openenv.yaml").write_text(
        _OPENENV_YAML.format(**ctx), encoding="utf-8"
    )

    # pyproject.toml
    (out_dir / "pyproject.toml").write_text(
        _PYPROJECT_TOML.format(**ctx), encoding="utf-8"
    )

    # README.md
    (out_dir / "README.md").write_text(
        _README_MD.format(**ctx), encoding="utf-8"
    )

    # server/
    server_dir = out_dir / "server"
    server_dir.mkdir(parents=True, exist_ok=True)

    (server_dir / "__init__.py").write_text(
        _SERVER_INIT_PY.format(**ctx), encoding="utf-8"
    )

    (server_dir / f"{env_module}_environment.py").write_text(
        _ENVIRONMENT_PY.format(**ctx), encoding="utf-8"
    )

    (server_dir / "app.py").write_text(
        _APP_PY.format(**ctx), encoding="utf-8"
    )

    (server_dir / "Dockerfile").write_text(
        _DOCKERFILE.format(**ctx), encoding="utf-8"
    )

    (server_dir / "requirements.txt").write_text(
        _REQUIREMENTS_TXT.format(**ctx), encoding="utf-8"
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
