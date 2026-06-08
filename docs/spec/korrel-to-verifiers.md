# Korrel to verifiers mapping (v0.2)

Status: specification. The exporter code follows this document, not the reverse. Changes to the
mapping land here first, then in `src/korrel/exporters/`.

This document defines how a Korrel `Scenario` is translated into a `verifiers` environment. It is
the authoring-spec contract: the canonical Korrel transcript types and the rubric reward-function
signature are fixed, and the translation is a near-identity restatement of an already-compatible
shape, not a redesign.

## Pinned upstream version

All interface claims in this document were confirmed against the installed source of
`verifiers==0.1.14` (`__version__ = "0.1.14"`, `verifiers-0.1.14.dist-info/METADATA`), not the
published docs. Each claim cites the module and symbol it was read from. A prior "verified" note
is a claim to re-check against the then-current installed version; re-confirm before relying on
any clause here.

The exporter pins `verifiers>=0.1.14`. The generated export artifact declares the same lower
bound in its `pyproject.toml` dependencies.

### requires-python consequence

`verifiers==0.1.14` declares `Requires-Python: <3.14,>=3.10` (`METADATA`, `Requires-Python`
field). Korrel's own repository environment is Python 3.14, which is outside that range, so
`verifiers` cannot be installed into the repo's primary 3.14 virtual environment. Consequences
for the exporter work:

- The `verifiers` interface was confirmed in a separate Python 3.12 inspection environment.
- `verifiers` is not a Korrel runtime dependency. It is a target of translation. The exporter
  produces a standalone, pip-installable package that declares `verifiers` as its own dependency;
  Korrel does not import `verifiers` at scenario-authoring or CI-run time.
- Any test that imports `verifiers` runs only where a `<3.14` interpreter is available. Korrel's
  offline, no-key unit test suite does not import `verifiers`. Exporter tests that exercise the
  generated package against a live `verifiers` install are gated on interpreter version and stay
  out of the default offline suite.

## Public API surface (contract, not final symbols)

The exporter exposes one translation entry point and a small set of artifact files. The exact
function and module names are set by `exporter-engineer`; this document fixes the contract, not
the spelling.

- A single translation call with the shape `Scenario -> vf.Environment`. Given a Korrel
  `Scenario`, it returns a constructed `verifiers` environment instance (a `MultiTurnEnv`
  subclass, or a `SingleTurnEnv` in the narrow case defined below).
- An export-artifact emitter that writes the pip-installable package described under
  "Packaging and discovery contract": a `pyproject.toml` and an environment module exposing a
  top-level `load_environment(**kwargs) -> Environment`.

The canonical transcript types in `src/korrel/types.py` and the rubric reward-function signature
in `src/korrel/rubric.py` are NOT changed by this mapping. Where the runtime shapes differ, the
exporter adapts at the boundary.

## Concept mapping

| Korrel concept | verifiers target | Source clause (verifiers 0.1.14) |
| --- | --- | --- |
| `Scenario` (persona-driven or with tools) | a `MultiTurnEnv` subclass | `envs/multiturn_env.py`: `class MultiTurnEnv(vf.Environment)`, abstract `async def env_response(self, messages, state, **kwargs) -> Messages` |
| `Scenario` (single exchange, no tools, no follow-up) | a `SingleTurnEnv` | `envs/singleturn_env.py`: `class SingleTurnEnv(vf.MultiTurnEnv)`, `__init__` does `super().__init__(max_turns=1)` |
| `Persona.next_message(messages)` | the user-turn branch of `env_response` | `envs/multiturn_env.py`: `env_response(...) -> Messages` returns the next environment messages |
| `MockTool` execution (mirror `runtime.py::_resolve_tool_call`) | the tool-execution branch of `env_response` | `envs/tool_env.py`: `env_response` parses `last_msg.tool_calls`, calls the tool, returns `ToolMessage`s |
| `MockTool.schema` (chat-completions tool schema) | a `vf.Tool` def `{name, description, parameters, strict?}` | `types.py`: `class Tool(CustomBaseModel)` with `name, description, parameters, strict` |
| `Rubric` reward functions | `vf.Rubric(funcs=..., weights=...)` | `rubrics/rubric.py`: `Rubric.__init__(self, funcs=None, weights=None, parser=None)` |
| `Scenario.system` | dataset row system, via env `system_prompt` or the prompt's system message | `envs/environment.py`: `Environment.__init__(..., system_prompt=None, ...)` |
| `Scenario.opening_message` | the user message in the dataset row's `prompt` | `types.py`: `RolloutInput` required `prompt: Messages` |
| `Scenario.info` | the dataset row's `info` | `types.py`: `RolloutInput` optional `info: Info | str` |

The environment requires a `dataset` or `eval_dataset` or raises `ValueError`
(`envs/environment.py`, `Environment.__init__`). A single scenario maps to a one-row Hugging Face
`Dataset`; the row carries `prompt`, `info`, and optionally `answer`.

### Scenario to environment

`Scenario` fields (`src/korrel/scenario.py`): `id, system, persona, opening_message, tools,
max_turns (>=1, default 1), max_tool_rounds (>=1, default 8), seed, info, rubric`.

The translation builds:

- The environment class (`MultiTurnEnv` subclass, or `SingleTurnEnv`; see the decision rule).
- A one-row dataset built from `system`, `opening_message`, `info` (see "Dataset row").
- The tool defs from `scenario.tools`, each `MockTool.schema` reshaped to `vf.Tool`.
- The rubric from `scenario.rubric` (see "Reward functions").

`scenario.id` names the export artifact (`env_id`); see "Packaging and discovery contract".

### Persona to env_response (user turn)

`Persona.next_message(messages: list[korrel.Message]) -> str | None` produces the next simulated
user message (`src/korrel/persona.py`). In the `MultiTurnEnv` it becomes the user-turn branch of
`env_response(self, messages, state, **kwargs) -> Messages`
(`envs/multiturn_env.py`, abstract method): when the last assistant message carries no tool
calls, the env converts the conversation so far into `korrel.Message` form, calls
`persona.next_message`, and returns a single `UserMessage` (`types.py`,
`class UserMessage`). A `None` return signals no further user turn; the env then signals
completion (no further messages to return), consistent with Korrel's loop, which stops when
`next_message` is falsy (`runtime.py`, `run_scenario`).

Completion is driven by `@vf.stop`-decorated methods, not an `is_completed` override
(`envs/multiturn_env.py`: built-in stops `has_error` priority 100, `prompt_too_long`,
`max_turns_reached`, `max_total_completion_tokens_reached`, `has_final_env_response`). The
exporter does not override `is_completed`; it relies on `max_turns_reached` plus the persona
exhaustion path. To make persona exhaustion terminate the rollout, the env sets
`state["final_env_response"]` when the persona returns `None`, which the built-in
`has_final_env_response` stop observes (`envs/multiturn_env.py`,
`async def has_final_env_response`).

### Mock tools to env_response (tool turn)

`MockTool` (`src/korrel/tools.py`): `name`, `schema` (chat-completions
`{"type":"function","function":{name,description,parameters}}`), `respond(arguments, state)`, and
`call(arguments, state)`.

`ToolEnv` is the template for the tool-execution branch (`envs/tool_env.py`). Its `env_response`:

1. Reads `last_msg.tool_calls` (verifiers `AssistantMessage.tool_calls: list[ToolCall]`,
   `types.py`).
2. For each call, takes `tool_call.id`, `tool_call.name`, and
   `json.loads(tool_call.arguments)` (verifiers `ToolCall` is the FLAT shape
   `{id, name, arguments}`, `types.py`, `class ToolCall(CustomBaseModel)`).
3. Calls the tool callable and returns `ToolMessage(role="tool", content=..., tool_call_id=...)`
   (`types.py`, `class ToolMessage`).

The exporter's tool branch mirrors `runtime.py::_resolve_tool_call` exactly so CI and RL resolve
tools identically:

- Parse arguments with `json.loads(arguments or "{}")`; on `JSONDecodeError`, use `{}`
  (Korrel `_resolve_tool_call`). Verifiers' `ToolEnv` raises or error-formats on bad JSON; the
  exporter follows Korrel's tolerant behavior so the two runtimes agree.
- Resolve the tool by name; on unknown tool, return the result
  `{"error": f"unknown tool: {name}"}` (Korrel `_resolve_tool_call`).
- Call `MockTool.call(arguments, state)`, where `state` is the per-run mutable dict.
- Serialize a non-string result with `json.dumps(...)`; pass a string result through
  (Korrel `_resolve_tool_call`).
- Return a `ToolMessage` with `tool_call_id` set to the originating call id.

`MockTool.schema` is reshaped to a `vf.Tool`. Korrel stores the chat-completions
`{"type":"function","function":{name,description,parameters}}`; verifiers' `Tool` is the flat,
provider-agnostic `{name, description, parameters, strict?}` (`types.py`, `class Tool`). The
legacy `{type, function}` shape is rejected by verifiers (`_normalize_tool_defs`), so the
exporter unwraps `schema["function"]` into the flat `Tool`. The canonical Korrel tool schema does
not change.

### Reward functions to vf.Rubric

Korrel `Rubric` (`src/korrel/rubric.py`): `Rubric(funcs, pass_threshold=0.5, judge=None)`; reward
functions have the signature `(completion: list[korrel.Message], info: dict, **kwargs) -> float`
(sync or async); `make_judge(...)` returns a hardened reward function named `"judge"`.

verifiers `Rubric(funcs=None, weights=None, parser=None)` (`rubrics/rubric.py`). Each reward
function is invoked with a merged kwargs mapping; a function declaring `**kwargs` receives the
full mapping, otherwise only matching parameters (`rubrics/rubric.py`,
`_call_individual_reward_func`: the `VAR_KEYWORD` branch passes `**merged`). The merged mapping
includes `completion`, `info`, `prompt`, `answer`, `state`, `task`, `parser`, and class objects
(`rubrics/rubric.py`, `score_objects`). The result is coerced with `float(...)`.

Korrel reward functions already declare `(completion, info, **kwargs)`, so they accept the
verifiers merged-kwargs call without a signature change. The reward-function signature is the
contract and is NOT changed.

## SingleTurn vs MultiTurn decision rule

The exporter chooses the environment base class deterministically from the scenario:

- A scenario that is persona-driven or has tools, or has `max_turns > 1`, maps to a
  `MultiTurnEnv` subclass.
- A scenario with `max_turns == 1` AND no tools AND no persona follow-up (a single user message,
  one assistant reply, then stop) maps to a `SingleTurnEnv`.

Tools force `MultiTurnEnv`. `SingleTurnEnv.env_response` raises `NotImplementedError`
(`envs/singleturn_env.py`: `raise NotImplementedError("env_response is not implemented for
SingleTurnEnv")`), so a `SingleTurnEnv` cannot resolve a tool call: it has no environment turn in
which to run the mock tool and return a `ToolMessage`. Any scenario whose agent may emit a tool
call therefore requires `MultiTurnEnv`, whose `env_response` runs the tool branch.

`SingleTurnEnv` is the narrow optimization for a scenario with no environment turn to perform. In
practice most Korrel scenarios are persona-driven or tool-bearing and map to `MultiTurnEnv`.

## Reward-function value shim (verifiers Message to korrel Message)

The reward-function SIGNATURE is already compatible and is unchanged. The `completion` VALUE
differs between the two runtimes, so the exporter converts before calling a Korrel reward
function:

- verifiers passes `completion` as `Messages` (`types.py`, `Messages = list[Message]`), where
  the message union is `SystemMessage | UserMessage | AssistantMessage | ToolMessage |
  TextMessage` and a tool call is the FLAT `ToolCall{id, name, arguments}`
  (`types.py`, `class ToolCall`).
- Korrel reward functions expect `completion: list[korrel.Message]`, where an assistant tool call
  is the NESTED `ToolCall{id, type:"function", function: ToolFunction{name, arguments}}`
  (`src/korrel/types.py`).

Field-by-field conversion, verifiers `Message` to korrel `Message`:

- system: `SystemMessage{role:"system", content}` to `Message(role="system", content=...)`.
  `content` may be a string or a list of content parts in verifiers; the exporter joins or
  serializes non-string content to the canonical string `content`.
- user: `UserMessage{role:"user", content}` to `Message(role="user", content=...)`.
- assistant without tool calls: `AssistantMessage{role:"assistant", content}` to
  `Message(role="assistant", content=...)`.
- assistant with tool calls: each FLAT `ToolCall{id, name, arguments}` is nested into the
  canonical `ToolCall(id=id, type="function", function=ToolFunction(name=name,
  arguments=arguments))`. `arguments` stays a JSON-encoded string on both sides
  (verifiers `ToolCall.arguments: str`; Korrel `ToolFunction.arguments` is a JSON string).
- tool result: `ToolMessage{role:"tool", tool_call_id, content}` to
  `Message(role="tool", tool_call_id=..., content=...)`.

The conversion is the only adaptation; it does not alter either type definition. Korrel canonical
types and the reward signature remain the export target and remain fixed.

## Lossy edges

Each edge is a place where the two runtimes do not align one-to-one. The exporter resolves each
explicitly so the verifiers scalar reproduces Korrel's intent.

### max_turns unit mismatch

Korrel counts a turn as one user-to-assistant exchange and runs an inner tool loop of up to
`max_tool_rounds` per turn (`runtime.py`, `run_scenario`: the outer
`for turn_index in range(scenario.max_turns)` and the inner
`while assistant.tool_calls` bounded by `max_tool_rounds`). verifiers counts `max_turns` as
model-response steps: `max_turns_reached` fires when `len(state["trajectory"]) >= self.max_turns`
(`envs/multiturn_env.py`, `max_turns_reached`), and the trajectory grows by one step per model
response (`add_model_response`). One Korrel turn with tool calls produces several verifiers steps.

Resolution: the exporter does not pass `scenario.max_turns` directly as verifiers `max_turns`. It
sets verifiers `max_turns` to a model-step budget that covers Korrel's worst case,
`scenario.max_turns * (1 + scenario.max_tool_rounds)`, and reproduces Korrel's per-turn tool cap
inside `env_response` by tracking a per-user-turn tool-round counter in `state` and ceasing to
return tool results once `max_tool_rounds` is reached for that turn (mirroring Korrel's
`max_tool_rounds` break with `stop_reason="max_tool_rounds"`). The user-turn count remains
governed by persona exhaustion and the derived step budget. This keeps both runtimes bounded by
the same two scenario limits even though the unit differs.

### Aggregation mismatch (weighted sum vs mean)

Korrel aggregates reward functions by MEAN: `aggregate = sum(scores.values()) / len(scores)`
(`src/korrel/rubric.py`, `Rubric.score`). verifiers aggregates by WEIGHTED SUM into
`state["reward"]`: `reward=sum(reward * weight for ...)` (`rubrics/rubric.py`, `score_rollout`).

Resolution: the exporter constructs `vf.Rubric(funcs=fs, weights=[1/n]*n)` where `n` is the count
of reward functions. With uniform weights `1/n`, the verifiers weighted sum equals
`sum(scores)/n`, which is Korrel's mean. The verifiers scalar then reproduces Korrel's aggregate
score. If a scenario later attaches non-uniform reward weights, the exporter normalizes them to
sum to one so the weighted sum stays a weighted mean.

pass_threshold: Korrel compares the aggregate against `pass_threshold` for overall pass/fail and
against each function's score for the failed-function list (`src/korrel/rubric.py`,
`Rubric.score`). verifiers carries `pass_threshold` on the environment
(`envs/environment.py`, `Environment.__init__(..., pass_threshold=0.5, ...)`) and applies it to
the aggregate reward at the eval layer. The exporter passes `scenario.rubric.pass_threshold`
through to the environment `pass_threshold` so the pass boundary matches. Per-function
pass/fail (Korrel's failed-function list) has no verifiers reward-scalar equivalent; verifiers
surfaces each function as a named metric keyed by `func.__name__` (`rubrics/rubric.py`,
`score_rollout` metrics), which is the closest available signal.

### Persona makes a live model call inside env_response

`Persona.next_message` is LLM-driven and makes a live Claude call by default with a key read from
the environment at call time (BYO keys). Inside the verifiers `env_response`, that call runs
during the rollout, so the environment is non-deterministic and network-dependent. This is a
property of the persona, not a defect of the mapping. Offline tests inject a fake persona with
the same `next_message(messages) -> str | None` interface, exactly as Korrel's runtime accepts a
`persona` override for fakes (`runtime.py`, `run_scenario` `persona` parameter). The default
offline suite never makes the live call.

### The judge reward function makes a live model call

`make_judge(...)` returns a reward function that calls a provider during scoring
(`src/korrel/rubric.py`, the inner `judge` calls `prov.complete(...)`). When that function is
registered in the `vf.Rubric`, scoring is non-deterministic and network-dependent. The judge
remains hardened: the transcript is serialized as untrusted data inside a fixed rubric and is
never treated as instructions (`src/korrel/rubric.py`, `_JUDGE_SYSTEM`). Offline tests substitute
a deterministic reward function for the judge. As with the persona, this is inherent to
LLM-as-judge and is not introduced by the export.

### Tool-call JSON-decode tolerance

Korrel tolerates malformed tool-call arguments by defaulting to `{}`
(`runtime.py`, `_resolve_tool_call`). verifiers' `ToolEnv` either raises `ToolParseError` (when
the error type is in `stop_errors`) or returns an error-formatted `ToolMessage`
(`envs/tool_env.py`, `env_response`). The exporter follows Korrel's tolerant behavior in its
`env_response` so the two runtimes agree on bad input; it does not adopt the `ToolEnv`
error-formatting default.

### Content shape narrowing

verifiers message `content` may be a list of content parts (text, image, audio)
(`types.py`, `MessageContent: TypeAlias = str | list[ContentPart]`). Korrel canonical `content`
is `Optional[str]` (`src/korrel/types.py`, `Message.content`). The exporter narrows non-string
content to a string when converting verifiers messages to Korrel messages for scoring. Korrel
authoring produces string content, so the forward direction is lossless; the narrowing only
affects multimodal content that a Korrel scenario does not author.

## Packaging and discovery contract

verifiers discovers an environment by `vf.load_environment(env_id, **env_args)`, which imports the
module `env_id.replace("-","_").split("/")[-1]` and requires a top-level
`load_environment(**kwargs) -> Environment` (`utils/env_utils.py`, `load_environment`; it raises
`AttributeError` when the module has no `load_environment`, and constructs the env via
`env_load_func(**call_env_args)`).

The export artifact is a pip-installable package, matching the `vf-init` scaffold
(`scripts/init.py`):

```
<env_id_dir>/
  pyproject.toml
  <env_module>.py
```

- `pyproject.toml`:
  - `[project] name = "<env_id>"` (scaffold `PYPROJECT_TEMPLATE`: `name = "{env_id}"`).
  - `[project] dependencies = ["verifiers>=0.1.14", "korrel"]`. The scaffold declares
    `verifiers>={vf.__version__}`; the exporter pins `>=0.1.14` and adds `korrel` because the
    generated module imports Korrel to build mock tools, persona, and rubric.
  - `requires-python = ">=3.10"` (scaffold). The generated package targets `<3.14` only insofar
    as `verifiers` constrains it transitively; the package itself states `>=3.10`.
  - `[build-system] requires = ["hatchling"]`, `build-backend = "hatchling.build"` (scaffold).
  - `[tool.hatch.build] include = ["<env_module>.py", "pyproject.toml"]` (scaffold
    `PYPROJECT_TEMPLATE`: `include = ["{env_file}.py", "pyproject.toml"]`).
- `<env_module>.py` exposes `def load_environment(**kwargs) -> vf.Environment` returning the
  constructed environment for the scenario (the `Scenario -> vf.Environment` call, with the
  scenario reconstructed or referenced from the module).

The module name is `env_id.replace("-","_")` (verifiers' import rule). The exporter derives both
`env_id` and module name from `scenario.id`; `scenario.id` is already sanitized for filesystem use
upstream, so the exporter applies the same dash-to-underscore rule verifiers uses at import time.

## OpenEnv via verifiers (scope note)

`verifiers==0.1.14` consumes OpenEnv; it does not export to it. `vf.OpenEnvEnv` is an environment
that wraps an external OpenEnv server: it imports `openenv.core.generic_client.GenericEnvClient`
and `prime_sandboxes`, and connects to a running OpenEnv contract
(`envs/integrations/openenv_env.py`, the `OpenEnvEnv` class and its
`from openenv.core.generic_client import GenericEnvClient` import; exported as `OpenEnvEnv` in
`verifiers/__init__.py`). There is no verifiers code path that emits an OpenEnv environment from a
verifiers environment. Conclusion: a Korrel-to-OpenEnv path cannot route through verifiers and
must target `openenv-core` directly. That is the scope of Dispatch D; this document defines no
OpenEnv design beyond this note.
