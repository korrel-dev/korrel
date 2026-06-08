# korrel

OSS Python SDK for agent simulation. Define a multi-turn agent test once, run it as a pytest CI gate, export it as a verifiers/OpenEnv RL environment.

## Install

```
uv add korrel
```

Bring your own provider keys. Korrel reads keys from the environment at call time and stores none. The default provider is Claude via the `anthropic` SDK; set `ANTHROPIC_API_KEY`. OpenAI support is an optional extra (`korrel[openai]`).

## Quickstart

The following is a complete path from installation to a passing scenario run.

**1. Write a scenario module** (`support_refund.py`):

```python
from korrel import MockTool, Persona, Rubric, Scenario, adapter_from_provider
from korrel.providers import AnthropicProvider
from korrel.types import Message

def lookup_order(arguments, state):
    state["called"] = True
    orders = {"A1001": {"order_id": "A1001", "amount": 49.99, "refundable": True}}
    return orders.get(arguments.get("order_id", ""), {"error": "not found"})

orders = MockTool(
    name="lookup_order",
    schema={"type": "function", "function": {
        "name": "lookup_order",
        "description": "Look up an order by its id.",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]},
    }},
    respond=lookup_order,
)

def confirmed(completion, info, **kwargs):
    amount = f"{info['amount']:.2f}"
    return 1.0 if any(
        m.role == "assistant" and m.content
        and "refund" in m.content.lower() and amount in m.content
        for m in completion
    ) else 0.0

scenario = Scenario(
    id="support_refund",
    system="You are a support agent. Resolve refunds using lookup_order.",
    persona=Persona(goal="Get a refund for order A1001.", behavior="Polite but firm."),
    opening_message="My order A1001 arrived broken and I want a refund.",
    tools=[orders],
    max_turns=3,
    seed=7,
    info={"order_id": "A1001", "amount": 49.99},
    rubric=Rubric(funcs=[confirmed], pass_threshold=0.5),
)

# AnthropicProvider reads ANTHROPIC_API_KEY at call time, not at import time.
adapter = adapter_from_provider(AnthropicProvider())
```

**2. Run it:**

```
korrel run support_refund.py
```

**Output on pass:**

```
scenario  : support_refund
score     : 1.0000
status    : pass
transcript: .korrel/support_refund.transcript.json
```

**Output on failure** (exit code 1):

```
scenario  : support_refund
score     : 0.0000
status    : fail
failed    : confirmed
clusters  : confirmed(zero)
transcript: .korrel/support_refund.transcript.json
```

The CLI exits zero on pass and non-zero on failure. The full conversation transcript is written to `.korrel/<scenario-id>.transcript.json`.

**CLI flags:**

```
korrel run SCENARIO_PY [--out DIR] [--seed N]
                       [--scenario-attr NAME] [--adapter-attr NAME]
```

`--out` overrides the transcript directory (default `.korrel/`). `--seed` overrides the scenario seed. `--scenario-attr` and `--adapter-attr` override the module attribute names (defaults: `scenario`, `adapter`).

## The pytest CI gate

Name scenario files `*_scenario.py` and place them in your test tree. The korrel pytest plugin (registered automatically via the `pytest11` entry-point, no conftest needed) discovers and runs them:

```
uv run pytest
```

A passing scenario produces a green dot. A failing scenario produces a normal pytest failure block showing score, threshold, failed rubric function names, and the transcript path.

**Discovery options:**

- `--korrel-glob GLOB`: override which file name pattern the plugin picks up (command-line flag)
- `korrel_glob = *_scenario.py`: the corresponding `pytest.ini` / `pyproject.toml` option

Each scenario file in the CI gate must expose both a module-level `scenario` and a module-level `adapter`. If `adapter` is absent, the item is reported as an error immediately.

## The four objects

- `Scenario`: a code-first test definition. Holds the system prompt, a `Persona`, the opening message, mock tools, `max_turns`, `max_tool_rounds`, a `seed`, ground-truth `info`, and a `Rubric`.
- `Persona`: the LLM-driven user-simulator. Given the conversation so far, it produces the next user message. Defaults to Claude.
- `MockTool`: a programmable tool. Holds a chat-completions tool schema and a `respond` callable that takes parsed arguments and a mutable per-run state and returns a result.
- `Rubric`: reward functions plus an optional hardened LLM judge. Reward signatures mirror verifiers: `(completion, info, **kwargs) -> float`. The judge treats the transcript as data, never as instructions.

## The module convention

A scenario module exposes two module-level names:

- `scenario`: a `Scenario` instance.
- `adapter`: any callable `(messages: list[Message], tools: list[ToolSchema]) -> Message`. Both `korrel run` and the pytest plugin read these names (overridable via `--scenario-attr` / `--adapter-attr`).

The built-in helper `adapter_from_provider(provider)` wraps any `Provider` (such as `AnthropicProvider`) as an adapter. Because `AnthropicProvider` reads the API key from the environment only at call time, constructing `adapter_from_provider(AnthropicProvider())` at module level is import-safe: no key is required to import the module.

## Running the Python API directly

```python
from korrel import run_scenario
result = run_scenario(scenario, adapter)
print(result.score, result.passed, result.failed_functions)
```

`examples/support_refund.py` holds a runnable scenario definition with a real `AnthropicProvider` adapter.

## Export to verifiers

A Korrel scenario can be translated into a [`verifiers`](https://github.com/willccbb/verifiers) RL training environment. The full mapping is specified in [`docs/spec/korrel-to-verifiers.md`](docs/spec/korrel-to-verifiers.md).

### Install

`verifiers` is an optional extra. `import korrel` works without it.

```
pip install 'korrel[verifiers]'
# or
uv add 'korrel[verifiers]'
```

`verifiers==0.1.14` requires Python `<3.14`, so the extra installs only on Python 3.10-3.13.

### Python API

```python
from korrel.exporters.verifiers import to_verifiers_env

env = to_verifiers_env(scenario)
```

`to_verifiers_env` returns a constructed verifiers environment: a `MultiTurnEnv` subclass for most scenarios (persona-driven or tool-bearing), or a `SingleTurnEnv` for a single-exchange scenario with no tools and no persona follow-up. Using the support-refund scenario from the quickstart, the call returns a `MultiTurnEnv` because the scenario has both a persona and a mock tool.

### CLI export

```
korrel export support_refund.py --to verifiers --out ./support_refund_env
```

This writes a pip-installable package that `verifiers` discovers via `load_environment`:

```
support_refund_env/
  pyproject.toml
  support_refund.py   # environment module exposing load_environment()
  _scenario.py        # copy of the original scenario source
```

Install the package in a Python 3.10-3.13 environment and load it:

```python
import verifiers as vf

env = vf.load_environment("support_refund")
```

When `--out` is omitted, the package is written to `.korrel/export/<scenario-id>/`.

### Concept mapping

| Korrel concept | verifiers target |
|---|---|
| `Scenario` (persona or tools or `max_turns > 1`) | `MultiTurnEnv` subclass |
| `Scenario` (single exchange, no tools) | `SingleTurnEnv` |
| `Persona` | user-turn generation inside `env_response` |
| `MockTool` | tool-execution branch of `env_response` |
| Rubric reward functions | `verifiers.Rubric(funcs=..., weights=...)` |
| `Scenario.system` + `opening_message` + `info` | dataset row (`prompt`, `info`) |

The spec holds the full field-level detail.

### What survives the translation

The reward-function signature `(completion, info, **kwargs) -> float` is unchanged. The canonical transcript types in `korrel.types` are unchanged. The only adaptation at the boundary is converting the `completion` value from the verifiers message shape to the korrel canonical shape before calling each reward function.

Lossy edges (summarized; see the spec's [Lossy edges](docs/spec/korrel-to-verifiers.md#lossy-edges) section for the complete list):

- The persona generates user turns with a live model call inside `env_response`. BYO key, non-deterministic. Offline tests substitute a fake persona.
- The judge reward function likewise makes a live model call during scoring.
- Korrel aggregates reward functions by mean; verifiers uses a weighted sum. The exporter sets weights to `1/n` to reproduce the mean.
- Korrel `max_turns` counts user-to-assistant exchanges; verifiers `max_turns` counts individual model-response steps. The exporter derives the verifiers step budget from `scenario.max_turns` and `scenario.max_tool_rounds`.

## Export to OpenEnv

A Korrel scenario can be translated into an [OpenEnv](https://github.com/huggingface/openenv) environment server. The full mapping is specified in [`docs/spec/korrel-to-openenv.md`](docs/spec/korrel-to-openenv.md).

### Install

`openenv-core` is an optional extra. `import korrel` works without it.

```
pip install 'korrel[openenv]'
# or
uv add 'korrel[openenv]'
```

`openenv-core>=0.3.0` declares `Requires-Python: >=3.10` with no upper bound, so it installs on Python 3.10 through 3.14, including the repo's default Python 3.14. This contrasts with the verifiers extra, which is capped at Python `<3.14`.

### Python API

```python
from korrel.exporters.openenv import build_environment_class

EnvironmentCls = build_environment_class(scenario, observation_cls, action_cls)
```

`build_environment_class` returns a concrete `openenv.core.env_server.interfaces.Environment` subclass. The caller supplies the author-defined `Action` and `Observation` subclasses (generated by the CLI export; see below). The `persona` keyword argument accepts an override for offline testing without live model calls.

The primary deployment path is the CLI export: the generated package is the artifact that gets deployed to a Hugging Face Space.

### CLI export

```
korrel export support_refund.py --to openenv --out ./support_refund_env
```

This writes a pip-installable OpenEnv environment package:

```
support_refund_env/
    __init__.py
    client.py
    models.py
    openenv.yaml
    pyproject.toml
    README.md
    _scenario.py                    # copy of the original scenario source
    server/
        __init__.py
        support_refund_environment.py
        app.py
        Dockerfile
        requirements.txt
```

When `--out` is omitted, the package is written to `.korrel/export/<scenario-id>/`.

Deploy the generated package to a Hugging Face Space:

```
openenv push --secret ANTHROPIC_API_KEY=<your-key>
```

The persona and judge make live model calls inside the container. The API key is supplied at runtime via `openenv push --secret` and is never written into any emitted file.

### Concept mapping

| Korrel concept | OpenEnv target |
|---|---|
| `Scenario` | `Environment` subclass |
| `scenario.system` + `opening_message` | `reset()` seed observation |
| agent turn | author-defined `Action` subclass |
| environment reply | author-defined `Observation` subclass |
| `MockTool` execution | tool branch of `step()` |
| `Persona.next_message` | persona branch of `step()` |
| `Rubric` aggregate | terminal `observation.reward` (`done=True`) |

The spec holds the full field-level detail.

### What survives the translation

The reward-function signature `(completion, info, **kwargs) -> float` is unchanged. The canonical transcript types in `korrel.types` are unchanged. The OpenEnv `Rubric` class (incompatible action/observation signature) is not used; korrel computes reward with its own `Rubric.score`.

Lossy edges (summarized; see the spec's [Lossy edges](docs/spec/korrel-to-openenv.md#lossy-edges) section for the complete list):

- Reward is terminal, not dense. Every intermediate step carries `reward=None`; the final step (`done=True`) carries the rubric aggregate.
- The persona and judge run server-side inside the container; they make live model calls, which are non-deterministic and require a key at runtime.
- Content-shape narrowing: `Observation.messages` carries dicts; the full canonical `Message` type is used internally and serialized at the boundary.
- Tool-call arguments stay a JSON string (`ToolFunction.arguments`), matching the chat-completions wire format.

## Determinism

Every run takes a seed and records the model and request parameters. The seed pins scenario setup and any sampling Korrel controls. LLM calls are not bit-reproducible; provider nondeterminism is outside the seed.

## Telemetry

Korrel includes opt-in telemetry. No scenario content, persona text, transcripts, prompts, tool schemas, file paths, or model names tied to a customer are ever collected. The event carries only aggregate counters and version metadata.

**What the `run` event sends** (every field, nothing more):

| Field | Description |
|---|---|
| `event` | Always `"run"` |
| `schema_version` | Event schema version (currently `"1"`) |
| `korrel_version` | Installed korrel version string |
| `python_version` | CPython version string |
| `scenario_count` | Number of scenarios in the run |
| `total_turns` | Total turns across all scenarios |
| `pass_count` | Number of passing scenarios |
| `fail_count` | Number of failing scenarios |
| `duration_s` | Wall-clock duration in seconds |
| `install_id` | Anonymous, randomly generated UUID (created once, stored locally) |

No key, scenario id, path, persona, transcript, prompt, tool schema, or model name is present in the event.

**Opt-outs (any one disables telemetry):**

- Set `KORREL_TELEMETRY=0` (also accepts `false`, `no`, `off`) in the environment.
- Set `DO_NOT_TRACK=1` in the environment.
- Telemetry is automatically off in CI (detected via `CI`, `GITHUB_ACTIONS`, `TRAVIS`, `CIRCLECI`, `GITLAB_CI`, `JENKINS_URL`, `BUILDKITE`, `TF_BUILD`, `TEAMCITY_VERSION`, `BITBUCKET_BUILD_NUMBER`).
- On the first interactive run outside CI, Korrel prompts once for consent and persists the answer. Declining disables telemetry permanently for that install. Non-interactive sessions default to off with no prompt.

**No endpoint configured means no data sent.** Without `KORREL_TELEMETRY_ENDPOINT` set in the environment, the event is built and dropped. Set `KORREL_TELEMETRY_DEBUG=1` to write the event JSON to stderr for inspection. No endpoint is hardcoded.

Consent and the anonymous install id are stored in `%APPDATA%\korrel\config.json` (Windows) or `$XDG_CONFIG_HOME/korrel/config.json` / `~/.config/korrel/config.json` (Linux/macOS). No key, scenario content, or identifying information is ever written there.

## Data model and chat-completions compatibility

The canonical transcript types in `korrel.types` are provider-neutral and wire-compatible with the OpenAI chat-completions message schema. They are the v0.2 verifiers/OpenEnv export target. Their attribute names and wire shapes are a contract.

### Types

**`Message`** - one message in a conversation:

```python
Message(
    role="assistant",        # "system" | "user" | "assistant" | "tool"
    content="Your refund...", # text body; None when only tool_calls is set
    tool_calls=[...],        # present on assistant messages that call tools
    tool_call_id="call_1",   # links a role="tool" message to the call it answers
    name=None,               # optional speaker name
)
```

**`ToolCall`** - a single tool invocation inside an assistant message:

```python
ToolCall(
    id="call_1",
    type="function",          # always "function"
    function=ToolFunction(
        name="lookup_order",
        arguments='{"order_id": "A1001"}',  # JSON-encoded string, not a dict
    ),
)
```

`arguments` is a JSON-encoded string, matching the chat-completions wire format (OpenAI API reference, `tool_calls[].function.arguments`). Parse with `json.loads()` to recover the call arguments.

**`ToolSchema`** - a tool definition passed to an adapter, in chat-completions tool-schema shape:

```python
{"type": "function", "function": {
    "name": "lookup_order",
    "description": "Look up an order by its id.",
    "parameters": {"type": "object",
                   "properties": {"order_id": {"type": "string"}},
                   "required": ["order_id"]},
}}
```

### Chat-completions mapping table

| Canonical field | Chat-completions wire field | Notes |
|---|---|---|
| `Message.role` | `role` | `"system"`, `"user"`, `"assistant"`, `"tool"` |
| `Message.content` | `content` | `None` when only tool calls are present |
| `Message.tool_calls` | `tool_calls` | array of `ToolCall` objects |
| `Message.tool_call_id` | `tool_call_id` | on `role="tool"` messages |
| `ToolCall.id` | `tool_calls[].id` | |
| `ToolCall.type` | `tool_calls[].type` | always `"function"` |
| `ToolCall.function.name` | `tool_calls[].function.name` | |
| `ToolCall.function.arguments` | `tool_calls[].function.arguments` | JSON string, not a dict |

The Anthropic provider (`AnthropicProvider` in `korrel.providers`) translates between `tool_use`/`tool_result` blocks and these canonical types. Nothing in `korrel.types` depends on the `openai` package.

## License

MIT.
