# Korrel to OpenEnv mapping (v0.3)

Status: specification. The exporter code follows this document, not the reverse. Changes to the
mapping land here first, then in `src/korrel/exporters/`.

This document defines how a Korrel `Scenario` is translated into an OpenEnv environment server. It
is the authoring-spec contract: the canonical Korrel transcript types and the rubric
reward-function signature are fixed, and the translation adapts at the boundary rather than
redesigning either side. OpenEnv's action and observation shapes are author-defined, so the
near-identity property holds at the message level even where the runtime envelope differs from
verifiers.

## Pinned upstream version

Confirmed against: openenv-core==0.3.0. All interface claims in this document were confirmed
against the installed source of that version (`importlib.metadata.version("openenv-core") ==
"0.3.0"`; the package exposes no `openenv.__init__.__version__`), not the published docs. Each
claim cites the module and symbol it was read from. A prior "verified" note is a claim to re-check
against the then-current installed version; re-confirm before relying on any clause here.

openenv-core 0.3.0 was published to PyPI on 2026-05-11 and declares `Requires-Python: >=3.10`. It
installs and imports on Python 3.14, so the interface was confirmed in a throwaway Python 3.14
environment and cross-checked in a Python 3.12 environment.

### Import contract

The canonical import root is `openenv` / `openenv.core`. The top-level `openenv_core` module is a
deprecated alias that emits a `DeprecationWarning` on import. The exporter and the generated
package import server types from `openenv.core.env_server.*`, never from `openenv_core`.

### requires-python consequence

openenv-core 0.3.0 declares `Requires-Python: >=3.10` with no upper bound, so it installs into
Korrel's primary Python 3.14 repository environment, unlike `verifiers==0.1.14` which is capped at
`<3.14`. Consequences for the exporter work:

- OpenEnv exporter tests that import `openenv.core` run in the default test matrix; they are not
  confined to a version-capped job the way the verifiers exporter tests are.
- openenv-core is not a Korrel runtime dependency. It is a target of translation. The exporter
  produces a standalone OpenEnv package that declares `openenv-core` as its own dependency; Korrel
  does not import `openenv.core` at scenario-authoring or CI-run time.
- The default offline, no-key unit suite does not start a server or make a live persona/judge call.
  Any test that exercises the generated server end to end is gated and stays out of the default
  offline suite.

## Public API surface (contract, not final symbols)

The exporter exposes one translation entry point and a set of artifact files. The exact function
and module names are set by `exporter-engineer`; this document fixes the contract, not the
spelling.

- A single translation call with the shape `Scenario -> Environment subclass`. Given a Korrel
  `Scenario`, it produces an `openenv.core.env_server.interfaces.Environment` subclass (plus the
  author-defined `Action` and `Observation` subclasses the environment consumes and returns).
- An export-artifact emitter that writes the OpenEnv package described under "Packaging and
  deployment contract": the `openenv init` file set, with `server/app.py` wiring the environment
  into a FastAPI app via `create_app`.

The canonical transcript types in `src/korrel/types.py` and the rubric reward-function signature in
`src/korrel/rubric.py` are NOT changed by this mapping. Where the runtime shapes differ, the
exporter adapts at the boundary.

## Runtime model: agent is the policy, persona and judge are the environment

In the CI runtime, the agent under test is an adapter Korrel calls and the persona drives the
simulated user (`src/korrel/runtime.py`, `run_scenario`). In the OpenEnv runtime the roles invert
at the process boundary: the agent under test is the external RL policy that posts actions to the
server, and the persona plus the rubric judge run server-side inside the environment container.
Each `step` receives the policy's turn as an action and returns the environment's reply as an
observation.

OpenEnv's `Environment` is a synchronous Gym-style interface
(`openenv.core.env_server.interfaces`, `class Environment(ABC, Generic[ActT, ObsT, StateT])` with
`__init__(self, transform=None, rubric=None)`). Both lifecycle methods return an observation
directly:

- `reset(self, seed=None, episode_id=None, **kwargs) -> ObsT` (abstract).
- `step(self, action, timeout_s=None, **kwargs) -> ObsT` (abstract).
- `state` (abstract `@property`) returns the `StateT`.

There is no server-side `StepResult`. `done` and `reward` are FIELDS on the returned observation
(`openenv.core.env_server.types`, `class Observation(BaseModel)`: `done: bool = False`,
`reward: bool | int | float | None = None`). The client-side `StepResult`
(`openenv.core.client_types`, a dataclass `{observation, reward=None, done=False}`) is built by
`EnvClient`, never by the server. On the wire the HTTP layer lifts reward and done to the top level
of `StepResponse` / `ResetResponse` alongside the observation dict
(`openenv.core.env_server.types`, `class StepResponse`, `class ResetResponse`), but the
environment code returns one observation object.

## Concept mapping

| Korrel concept | OpenEnv target | Source clause (openenv-core 0.3.0) |
| --- | --- | --- |
| `Scenario` | an `Environment` subclass | `core/env_server/interfaces.py`: `class Environment(ABC, Generic[ActT, ObsT, StateT])`, `__init__(self, transform=None, rubric=None)` |
| `Scenario.system` + `Scenario.opening_message` | the seed observation returned by `reset` | `core/env_server/interfaces.py`: abstract `reset(self, seed=None, episode_id=None, **kwargs) -> ObsT` |
| agent turn (assistant content + tool calls) | an author-defined `Action` subclass passed to `step` | `core/env_server/types.py`: `class Action(BaseModel)` (`extra="forbid"`); no built-in conversational action exists, only the `EchoAction(message: str)` template |
| environment reply (user message or tool results) | an author-defined `Observation` subclass returned by `step` | `core/env_server/types.py`: `class Observation(BaseModel)`, fields `done`, `reward`, `metadata` |
| running Korrel `Message` list | episode-scoped `State` | `core/env_server/types.py`: `class State(BaseModel)` (`extra="allow"`), `episode_id`, `step_count` |
| `Persona.next_message(messages)` | the user-turn branch of `step` | `core/env_server/interfaces.py`: `step(...) -> ObsT` returns the next observation |
| `MockTool` execution (mirror `runtime.py::_resolve_tool_call`) | the tool-resolution branch of `step` | `core/env_server/interfaces.py`: `step(...) -> ObsT` |
| `Rubric` aggregate score | terminal `observation.reward` at `done=True` | `core/env_server/types.py`: `Observation.reward: bool | int | float | None` |
| `Scenario.info` | rubric ground truth held in `State` | `core/env_server/types.py`: `State` (`extra="allow"`) |
| package + server wiring | `openenv init` file set, `create_app` in `server/app.py` | `core/env_server/http_server.py`: `create_app(env, action_cls, observation_cls, env_name=None, ...) -> FastAPI` |

OpenEnv has no built-in conversational action and no built-in `CallToolAction` in
`core/env_server/types`; the only action in the init template is `EchoAction(message: str)`
(`cli/templates/openenv_env/models.py`). Korrel therefore authors its own `Action` and
`Observation` subclasses in the generated `models.py` rather than reusing an upstream type.

### Scenario to environment

`Scenario` fields (`src/korrel/scenario.py`, `class Scenario`): `id, system, persona,
opening_message, tools, max_turns (>=1, default 1), max_tool_rounds (>=1, default 8), seed, info,
rubric`.

The translation builds:

- The `Environment` subclass with `reset`, `step`, and the `state` property implemented.
- The author-defined `Action` subclass carrying the agent turn (assistant `content` and
  `tool_calls`) and the `Observation` subclass carrying the environment reply messages plus the
  `done` and `reward` fields.
- The mock-tool table from `scenario.tools` for the tool-resolution branch of `step`.
- The terminal reward from `scenario.rubric` (see "Reward to terminal observation reward").

`scenario.id` names the export artifact and the generated environment class; see "Packaging and
deployment contract".

### Reset to seed observation

`reset` builds the conversation seed from `scenario.system` and `scenario.opening_message`,
mirroring how `run_scenario` seeds `messages` with a system message (when `scenario.system` is set)
followed by the opening user message (`src/korrel/runtime.py`, `run_scenario`: the `if
scenario.system` system append and the `pending_user = Message(role="user",
content=scenario.opening_message)` seed). The episode-scoped `State` holds the running list of
Korrel canonical `Message` objects, starting from that seed, plus `scenario.info` for terminal
scoring. `reset` resets `step_count` to zero (`core/env_server/types.py`, `State.step_count: int =
0`) and returns an `Observation` with `done=False` and `reward=None` carrying the opening user
message for the policy to answer.

### Step to environment reply

`step(action)` branches EXACTLY as the verifiers exporter's `env_response` does, so the CI runtime,
the verifiers export, and the OpenEnv export resolve a turn identically. The branch mirrors the
inner loop of `run_scenario` and the helper `_resolve_tool_call` (`src/korrel/runtime.py`):

1. Convert the action into a canonical assistant `Message` and append it to the `State` message
   list.
2. If the action carries tool calls: resolve each call against the scenario mock tools (the tool
   branch below), append the tool-result messages to the `State` list, increment a per-user-turn
   tool-round counter, and return an `Observation` with the tool-result messages, `done=False`,
   `reward=None`. When the counter reaches `scenario.max_tool_rounds`, stop returning tool results
   for that turn (mirroring `run_scenario`'s `if tool_round >= scenario.max_tool_rounds` break with
   `stop_reason="max_tool_rounds"`).
3. Else (a plain assistant text turn): call `persona.next_message(history)`
   (`src/korrel/persona.py`, `Persona.next_message(messages) -> Optional[str]`), append the
   returned text as a user `Message`, and return an `Observation` carrying that user message with
   `done=False`, `reward=None`.

Termination and terminal reward:

- When the persona returns empty (the `if not next_user: break` in `run_scenario`) OR the policy
  has received exactly `scenario.max_turns` assistant turns, the episode ends. The turn budget
  matches `runtime.py::run_scenario` exactly: the policy gets `scenario.max_turns` assistant turns
  and the persona is called at most `scenario.max_turns - 1` times (run_scenario breaks at
  `turn_index == scenario.max_turns - 1` without calling the persona on the final turn). `step`
  returns an `Observation` with `done=True` and `reward` set to the Korrel rubric aggregate,
  `scenario.rubric.score(messages, scenario.info).score`
  (`src/korrel/rubric.py`, `Rubric.score` returning `RubricResult.score`).
- Persona exhaustion sets `done=True`. The terminal reward is the single rubric aggregate returned
  once, on the `done=True` step. Every intermediate step carries `reward=None`.

### Mock tools to the tool-resolution branch

`MockTool` (`src/korrel/tools.py`, `class MockTool`): `name`, `schema` (chat-completions
`{"type":"function","function":{name,description,parameters}}`), `respond(arguments, state)`, and
`call(arguments, state)`. The tool branch of `step` mirrors `runtime.py::_resolve_tool_call`
exactly so the runtimes agree on every input, including malformed input:

- Parse arguments with `json.loads(call.function.arguments or "{}")`; on `JSONDecodeError`, use
  `{}` (`src/korrel/runtime.py`, `_resolve_tool_call`).
- Resolve the tool by name; on unknown tool, return the result `{"error": f"unknown tool:
  {name}"}` (`_resolve_tool_call`).
- Call `MockTool.call(arguments, state)`, where `state` is the per-episode mutable dict held in
  `State`.
- Serialize a non-string result with `json.dumps(...)`; pass a string result through
  (`_resolve_tool_call`).
- Return the result in a canonical tool `Message` with `tool_call_id` set to the originating call
  id, carried inside the returned `Observation`.

### Reward to terminal observation reward

Korrel `Rubric` (`src/korrel/rubric.py`, `class Rubric`): `Rubric(funcs, pass_threshold=0.5,
judge=None)`; reward functions have the signature `(completion: list[korrel.Message], info: dict,
**kwargs) -> float` (sync or async); `Rubric.score(completion, info)` aggregates by MEAN
(`aggregate = sum(scores.values()) / len(scores)`) and returns a `RubricResult` whose `score` is
that mean.

OpenEnv exposes its own rubric base (`openenv.core.rubrics.base`, `class Rubric` with `forward(self,
action, observation) -> float`, invoked as `rubric(action, observation)` through
`Environment._apply_rubric`). That signature is `(action, observation)`, incompatible with Korrel's
`(completion, info, **kwargs)`. The exporter does NOT adopt OpenEnv's `Rubric` class and does NOT
change Korrel's reward-function signature. Instead `step` computes the terminal scalar directly by
calling `scenario.rubric.score(messages, scenario.info)` and assigns `RubricResult.score` to
`observation.reward` on the `done=True` step. This keeps the canonical reward path identical to CI
and to the verifiers export.

## Message-value shim (OpenEnv action and observation to Korrel Message)

The reward-function SIGNATURE is already compatible and is unchanged. The boundary types differ, so
the exporter converts between the author-defined OpenEnv action/observation and Korrel canonical
`Message` objects, reusing the v0.2 flat/nested conversion the verifiers exporter already performs
(`src/korrel/exporters/verifiers.py`, `_to_korrel_messages`; see "Reward-function value shim" in
`korrel-to-verifiers.md`).

- The OpenEnv action carries the agent turn. Its assistant content maps to a canonical
  `Message(role="assistant", content=...)` (`src/korrel/types.py`, `class Message`). Tool calls on
  the action map to the canonical NESTED `ToolCall{id, type:"function", function: ToolFunction{name,
  arguments}}` (`src/korrel/types.py`, `class ToolCall`, `class ToolFunction`).
- The OpenEnv observation carries the environment reply: a user `Message(role="user", content=...)`
  on a persona turn, or tool `Message(role="tool", tool_call_id=..., content=...)` entries on a tool
  turn.
- Content narrows to the canonical `Optional[str]` (`src/korrel/types.py`, `Message.content`); see
  "Content shape narrowing".

The conversion is the only adaptation; it does not alter either type definition. Korrel canonical
types and the reward signature remain the export target and remain fixed.

## Lossy edges

Each edge is a place where the two runtimes do not align one-to-one. The exporter resolves each
explicitly so the OpenEnv reward reproduces Korrel's intent.

### Terminal reward vs dense per-step reward

Korrel produces one aggregate score for the whole transcript (`src/korrel/rubric.py`,
`Rubric.score`). OpenEnv supports a reward on every observation, and 0.3.0 ships RFC 004 trajectory
rubrics for dense, credit-assigned shaping (`openenv.core.rubrics.trajectory`,
`TrajectoryRubric` and `ExponentialDiscountingTrajectoryRubric`).

Resolution: the exporter emits the terminal scalar only. `observation.reward` is `None` on every
intermediate step and is set to the Korrel aggregate on the `done=True` step
(`core/env_server/types.py`, `Observation.reward` and `Observation.done`). Dense shaping is
available upstream through `TrajectoryRubric` if a future Korrel version wants per-step reward; this
mapping does not adopt it, because Korrel's rubric contract is a single terminal aggregate.

### Persona and judge make live model calls inside the container

`Persona.next_message` is LLM-driven and makes a live Claude call by default
(`src/korrel/persona.py`, `Persona.next_message` calls `self._provider().complete(...)`), and the
judge reward function calls a provider during scoring (`src/korrel/rubric.py`, the inner `judge`
calls `prov.complete(...)`). In the OpenEnv runtime these calls run server-side inside the
container at `step` time, because the persona and judge are the environment. The key is read from a
container environment variable at call time (BYO keys), supplied to a deployed Hugging Face Space
through `openenv push --secret ANTHROPIC_API_KEY=...`
(`openenv.cli.commands.push`, the `--secret KEY=VALUE` channel routed to `add_space_secret`, value
never logged). No key is ever written into any emitted file.

The judge remains hardened: the transcript is serialized as untrusted data inside a fixed rubric and
is never treated as instructions (`src/korrel/rubric.py`, `_JUDGE_SYSTEM`). Offline and gated tests
inject a fake persona exposing the same `next_message(messages) -> str | None` interface and
substitute a deterministic reward function for the judge, exactly as `run_scenario` accepts a
`persona` override (`src/korrel/runtime.py`, `run_scenario` `persona` parameter). The default
offline suite never makes the live call.

### Content shape narrowing

The author-defined OpenEnv observation and action carry content as the fields the exporter declares
on them; the conversion to canonical `Message` narrows any non-string content to a string, because
Korrel canonical `content` is `Optional[str]` (`src/korrel/types.py`, `Message.content`). Korrel
authoring produces string content, so the forward direction is lossless; the narrowing only affects
shapes a Korrel scenario does not author. This is the same narrowing the verifiers export performs
(`korrel-to-verifiers.md`, "Content shape narrowing").

### Tool-call arguments: JSON string vs structured field

Korrel canonical `ToolCall.arguments` is a JSON-encoded STRING (`src/korrel/types.py`,
`ToolFunction.arguments: str`), matching the chat-completions wire format. OpenEnv's own
`CallToolAction` carries `arguments: Dict[str, Any]` (`openenv.core.env_server.mcp_types`,
`class CallToolAction`), a structured dict, but the exporter does not use that MCP action for
conversational tool calls. The author-defined Korrel action keeps `arguments` as a JSON-string
field, and the tool branch parses it with `json.loads(... or "{}")` at resolution time, mirroring
`runtime.py::_resolve_tool_call` exactly (`src/korrel/runtime.py`, `_resolve_tool_call`). Keeping
the JSON string on the action preserves the canonical shape and the tolerant-parse behavior across
the boundary; a bare non-object JSON value (for example a number) would not fit `Dict[str, Any]`,
which is a further reason not to route through `CallToolAction`.

### Strict action and observation schemas (extra forbidden)

OpenEnv `Action` and `Observation` set `extra="forbid"` (`core/env_server/types.py`, `class
Action`, `class Observation`). Any data the exporter wants to carry alongside the turn must be a
declared field on the author-defined subclass or live in the `metadata: Dict[str, Any]` field both
base classes provide (`core/env_server/types.py`, `Action.metadata`, `Observation.metadata`). The
exporter declares explicit fields for the conversational payload rather than relying on extra keys.

## Packaging and deployment contract

`openenv init <name>` scaffolds an environment package (`openenv.cli.commands.init`, copying
`cli/templates/openenv_env/`). The emitted file set is:

```
<env_name>/
  __init__.py
  client.py
  models.py
  openenv.yaml
  pyproject.toml
  README.md
  uv.lock                 (generated by uv lock)
  server/
    __init__.py
    app.py
    Dockerfile
    requirements.txt
    <env_name>_environment.py
```

The export artifact matches this layout, with Korrel-specific content in three files:

- `models.py`: the author-defined `Action` and `Observation` subclasses for the scenario
  (`cli/templates/openenv_env/models.py` is the `EchoAction` / `EchoObservation` template the
  exporter replaces).
- `server/<env_name>_environment.py`: the `Environment` subclass implementing `reset`, `step`, and
  `state` for the scenario.
- `server/app.py`: wires the environment into a FastAPI app. It calls
  `create_app(<Env>, <Action>, <Observation>, env_name="<env_name>", ...)`, passing the environment
  CLASS as the factory (`core/env_server/http_server.py`,
  `create_app(env: Callable[[], Environment], action_cls, observation_cls, env_name=None, ...) ->
  FastAPI`), and defines `def main(host="0.0.0.0", port=8000)` under an `if __name__ == "__main__"`
  guard (`cli/templates/openenv_env/server/app.py`).

The manifest `openenv.yaml` declares `spec_version: 1`, `name`, `type: space`, `runtime: fastapi`,
`app: server.app:app`, `port: 8000` (`cli/templates/openenv_env/openenv.yaml`).

`openenv validate` enforces the multi-mode deployment shape (`openenv.cli._validation`,
`validate_multi_mode_deployment`): a `[project.scripts]` entry `server = "<pkg>.server.app:main"`,
an `openenv` / `openenv-core>=0.2.0` dependency, a `server/app.py` defining `def main(` under a
`__main__` guard, and a `Dockerfile`. The generated package satisfies each: the exporter pins
`openenv-core>=0.3.0` (the version this spec is confirmed against), and `server/app.py` carries the
`main` entry point and `__main__` guard from the template.

`openenv push` packages the directory as a Hugging Face Space with `space_sdk="docker"`
(`openenv.cli.commands.push`, `huggingface_hub.upload_folder`). The BYO-key channel is `openenv
push --secret KEY=VALUE`, which sets a Space secret through `add_space_secret` (value never logged).
No key is written into the emitted package; the persona and judge read the key from the container
environment at call time.

## Why the OpenEnv path targets openenv-core directly

`verifiers==0.1.14` consumes OpenEnv but does not export to it: `vf.OpenEnvEnv` wraps an external
OpenEnv server through `openenv.core.generic_client.GenericEnvClient`
(`verifiers/envs/integrations/openenv_env.py`). There is no verifiers code path that emits an
OpenEnv environment. The Korrel-to-OpenEnv path therefore targets `openenv-core` directly, which is
the scope of this document.
