# korrel scenarios gallery

A catalog of self-contained korrel scenarios. Each entry is one agent-testing
pattern, written once and runnable two ways: as a pytest-style CI gate today,
and as a verifiers/OpenEnv RL environment when exported. That dual use is what
separates a korrel scenario from an RL-only environment: the same file that
gates a pull request offline becomes a training environment for the same task.

Every entry runs offline with no provider key and no network. A scripted adapter
stands in for the agent under test so the gate is deterministic. Each entry
documents the one-line swap to a live provider, and the export-verified entries
have been exported to a verifiers and an OpenEnv package and loaded.

## Run any entry

```
korrel run gallery/<entry>.py
```

The CLI exits 0 on pass and non-zero on failure, and writes the transcript to
`.korrel/<entry>.transcript.json`. No key is needed. To export an entry to an RL
environment:

```
korrel export gallery/<entry>.py --to verifiers --out ./out
korrel export gallery/<entry>.py --to openenv   --out ./out
```

The verifiers extra installs on Python 3.10 through 3.13; the openenv extra
installs on 3.10 through 3.14. See the main [README](../README.md) for the
export details.

## Entries

| Entry | Pattern | Offline | Export-verified |
|---|---|---|---|
| [support_hours.py](support_hours.py) | Single-turn, scripted, plain reward function (no tools, no judge). The smallest gate. | no key | verifiers, openenv |
| [password_reset.py](password_reset.py) | Multi-turn persona with one mock tool. The simulated user drives the conversation; the agent verifies, then confirms. | no key | run-only |
| [order_triage.py](order_triage.py) | Multi-tool, multi-round. Two mock tools feed each other inside one turn's tool loop. | no key | verifiers, openenv |
| [refund_denial_judge.py](refund_denial_judge.py) | Judge-scored rubric (`make_judge`), with a deterministic offline reward. The judge is shown as a keyed live-swap and never run in CI. | no key | verifiers, openenv |
| [tool_failure_showcase.py](tool_failure_showcase.py) | Failure-mode showcase. A raising mock tool surfaces as a clean named error plus a partial transcript; the run exits non-zero by design. | no key | run-only |

"Offline: no key" means the entry runs in CI with `ANTHROPIC_API_KEY` unset and
no network. "Export-verified" means the gallery CI job exports the entry to that
target and loads the generated package offline. "run-only" means the entry is
validated by its offline run, not by export.

## How the gallery is validated

Entries live outside the package's `testpaths`, so `uv run pytest` does not
collect them. The gallery has its own CI job (`.github/workflows/ci.yml`, the
`gallery` job) that runs [`_ci_check.py`](_ci_check.py): it runs every entry
offline and asserts its documented outcome, and exports every export-marked
entry to verifiers and openenv and loads it. The job runs with no key and no
network. Reproduce it locally:

```
uv run --extra verifiers --extra openenv python gallery/_ci_check.py
```

## Add an entry

See [CONTRIBUTING.md](CONTRIBUTING.md). Copy [TEMPLATE.py](TEMPLATE.py), ship a
scripted adapter so the entry runs offline, add a row to the table above, and
open a pull request. The `gallery` CI job is the bar.
