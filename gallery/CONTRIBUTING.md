# Contributing a gallery entry

The bar for a gallery entry is the `gallery` CI job. It runs every entry offline
with no provider key and no network, because contributed entries are code that
runs in CI. An entry that needs a key for a live run is welcome, but its keyed
path is never executed in CI: the job validates the offline run and, where
marked, the export.

## Steps

1. Copy [TEMPLATE.py](TEMPLATE.py) to `gallery/<your_entry>.py`. Pick a name
   that reads as a catalog entry (the pattern or domain), not as a test.
2. Fill in the scenario: the system prompt, a `Persona`, the opening message,
   any mock tools, the rubric, and ground-truth `info`.
3. Ship a scripted `adapter` (a plain callable that replays fixed assistant
   messages) so the entry runs offline and deterministically. Do not put a
   provider or a key in the entry. For a tool-calling turn, build the message
   with `tool_calls` (see [order_triage.py](order_triage.py)). For a multi-turn
   persona conversation, subclass `Persona` with a scripted `next_message` (see
   [password_reset.py](password_reset.py)).
4. Write the top docstring: the pattern the entry demonstrates, whether a live
   run needs a key, and which export targets it is verified against. Include the
   one-line swap to a live provider as a comment.
5. Confirm it runs offline:

   ```
   korrel run gallery/<your_entry>.py
   ```

6. Add a row to the table in [README.md](README.md).
7. Open a pull request. The `gallery` CI job must pass.

## index.json

[`index.json`](index.json) holds per-entry expectations for the CI job. An entry
that is not listed defaults to `{"run": "pass", "export": []}`: it must exit 0
offline and is not export-checked. Add a row only to:

- mark export targets (`"export": ["verifiers", "openenv"]`), so CI exports the
  entry and loads the generated package, or
- declare a non-pass run outcome (`"run": "tool_error"` for an entry whose mock
  tool raises and therefore exits non-zero by design).

## Writing rules

Entry prose and comments follow the project's public-content rules: no
em-dashes, no emoji, no exclamation marks, declarative. Keep the entry
self-contained; do not import shared gallery helpers.

## Reproduce the CI job locally

```
uv run --extra verifiers --extra openenv python gallery/_ci_check.py
```

This is exactly what the `gallery` job runs. It needs no key and no network.
