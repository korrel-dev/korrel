# Changelog

## 0.1.3 (2026-06-11)

- A mock tool that raises no longer escapes `run_scenario` as a raw traceback with the transcript lost. The runtime raises the typed `ToolExecutionError` (exported from the package root) carrying the tool name, the original exception, and the partial transcript up to and including the assistant tool call that triggered it. The pytest plugin catches it, writes the partial transcript, and reports a clean failure block naming the tool and the transcript path.
- The runtime records why a run ended. `Transcript.stop_reason` and `RunResult.stop_reason` carry `"max_turns"` (turn budget exhausted) or `"persona_ended"` (the simulated user ended the conversation). `RunResult.tool_rounds_capped` counts turns whose tool loop hit `max_tool_rounds`. All fields are additive with defaults; scoring, exit codes, seeding, and the telemetry payload are unchanged.
- `korrel run` never prints a raw traceback. A missing provider key (the new `MissingAPIKeyError`, a `RuntimeError` subclass raised by `AnthropicProvider._get_client` and exported from the package root) prints one instructive `error:` line; a raising mock tool prints `error: tool '<name>' raised: <message>` and writes the partial transcript; any other failure prints `error: run failed: <type>: <message>`. All exit 1.
- `korrel run --model NAME` overrides the persona model always, and the agent model when the adapter exposes a provider. `adapter_from_provider` attaches its provider to the returned callable as `.provider`. The run summary adds a `model` line (`unknown` for a scripted or custom adapter without a provider), a `stop reason : max_turns` line when the turn budget cut the run, and a `tool rounds : capped` line when any turn hit `max_tool_rounds`.
- Both exporters load the bundled scenario source under a unique in-memory module name, `_korrel_scenario_<env_module>`, registered in `sys.modules`, instead of the shared literal `"_scenario"`. Two exported environments imported in one process no longer collide on module identity. All identifier and path-sanitization guards are unchanged.
- The fidelity benchmark gains a make-free, cross-platform front door: `python benchmarks/run.py` reproduces `make benchmark` (sync, selftest, labels, fidelity) using only the standard library. CI gains a `benchmark` job that runs it offline with no key and asserts the 80/80, 0-verdict-flip result.
- Documentation: `Scenario.persona` is documented as required even for single-turn scenarios (at `max_turns=1` it is never invoked, so no model call and no key); the persona-override contract is documented as accepting any object exposing `next_message(messages) -> Optional[str]` across `run_scenario`, `to_verifiers_env`, `build_environment_class`, and the generated `load_environment`; the README Go-live example drops an unused import; `examples/support_refund.py`'s reward function gains the `**kwargs` the verifiers export contract requires.
- README example output blocks match the 0.1.3 CLI, including the new `model` and `stop reason` lines, and document the `--model` flag and the clean error behavior.

## 0.1.2 (2026-06-11)

- The OpenEnv exporter emits an installable package for scenario ids that are not valid Python identifiers: every Python-path position (packages, package-dir, the `project.scripts` entry point, the README import) uses the sanitized module form, and the hyphenated form is reserved for the distribution name.

## 0.1.1 (2026-06-11)

- `import korrel` no longer emits a pydantic `UserWarning` for the `MockTool.schema` field; the value moved to an aliased field with a read property, keeping the public surface unchanged.

## 0.1.0 (2026-06-11)

- First public release: code-first `Scenario`/`Persona`/`MockTool`/`Rubric` authoring, the `run_scenario` loop, the `korrel run` CLI, the pytest plugin, the verifiers and OpenEnv exporters, opt-in content-free telemetry, and the tau2 fidelity benchmark.
