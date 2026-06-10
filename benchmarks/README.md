# korrel tau2 fidelity benchmark

**Status: foundations committed; transcripts and results not yet generated.**
dx-doc-writer will expand this into full prose. The pin facts, task-id list,
rerun mechanics, and layout map below are authoritative.

---

## What this measures

A single claim: one Korrel scenario definition reproduces tau2-bench's gold
reward across three runtimes (pytest CI gate, verifiers env, OpenEnv server)
within floating-point noise. This is a fidelity proof, not a head-to-head
scoring contest.

---

## tau2-bench pin

| Field | Value |
|---|---|
| Repository | https://github.com/sierra-research/tau2-bench |
| Tag | v1.0.0 |
| Commit SHA | 17e07b1da2bbc0cadfddeea36412686e0604127b |
| Published | 2026-03-18 |
| Python requirement | >=3.12,<3.14 (from tau2 pyproject.toml) |
| Package name | tau2 (version 0.2.1-dev at that tag) |

SHA verified on 2026-06-10 via:

```
gh api repos/sierra-research/tau2-bench/git/refs/tags/v1.0.0
# -> object.sha = 17e07b1da2bbc0cadfddeea36412686e0604127b
```

The dependency is pinned as a git source install in `pyproject.toml`:

```
tau2 @ git+https://github.com/sierra-research/tau2-bench.git@17e07b1da2bbc0cadfddeea36412686e0604127b
```

---

## Domain and task slice

Domain: **retail** (114 tasks total in the base split at the pinned SHA).

Slice rule: first 40 task IDs by ascending numeric sort of the `id` field in
`data/tau2/domains/retail/tasks.json`. The field name in tasks.json is `id`
(not `task_id`); the evaluator uses `.id` to look up a task.

```
Task IDs (40 total):
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
```

These cover both the `train` split (0-4, 6-8, 10-11, 13-16, 19-25, 28-31,
34-35, 37) and the `test` split (5, 9, 12, 17, 18, 26, 27, 32, 33, 36, 38,
39) from split_tasks.json. No cherry-picking: the slice is the first 40 by
numeric id, independent of split membership.

---

## Agent models

Two models generate trajectories so the label set spans pass and fail:

| Slug | tau2 / litellm model string | Role |
|---|---|---|
| sonnet | `anthropic/claude-sonnet-4-6` | strong agent |
| haiku | `anthropic/claude-3-5-haiku-20241022` | weak agent |

Both use `temperature=0` and `seed=300` (tau2's DEFAULT_SEED). A single
`ANTHROPIC_API_KEY` is required for generation only. The user simulator also
uses these same model strings (tau2's litellm routing: `anthropic/` prefix).

---

## Evaluation function

`tau2.evaluator.evaluator.evaluate_simulation` at the pinned SHA.

Signature (confirmed from installed source):

```python
def evaluate_simulation(
    simulation: SimulationRun,
    task: Task,
    evaluation_type: EvaluationType,
    solo_mode: bool,
    domain: str,
    mode: CommunicationMode = CommunicationMode.HALF_DUPLEX,
    env_kwargs: dict = None,
) -> RewardInfo:
```

`EvaluationType` members:
- `ENV` -- DB state + env assertions (no LLM)
- `COMMUNICATE` -- communication checks (no LLM)
- `ACTION` -- tool-call checks (no LLM)
- `ALL` -- product of the above, filtered by each task's reward_basis
- `NL_ASSERTIONS` -- WIP, requires LLM; excluded from this benchmark
- `ALL_WITH_NL_ASSERTIONS` -- WIP; excluded
- `ALL_IGNORE_BASIS` -- product of all three regardless of reward_basis
- `ALL_WITH_NL_ASSERTIONS_IGNORE_BASIS` -- WIP; excluded

This benchmark uses `EvaluationType.ALL_IGNORE_BASIS`.

Rationale: 38 of the 40 benchmark tasks (ids 0-39) have `NL_ASSERTION` in
their `reward_basis`. Confirmed from `data/tau2/domains/retail/tasks.json` at
the pinned SHA. `EvaluationType.ALL` raises `ValueError` when `NL_ASSERTION`
is in the basis but not being evaluated (confirmed from evaluator.py source,
line 229: `raise ValueError("NL assertions are part of the reward basis...")`).
`EvaluationType.ALL_WITH_NL_ASSERTIONS` (WIP) requires a live LLM call.
`EvaluationType.ALL_IGNORE_BASIS` multiplies `ENV * ACTION * COMMUNICATE`
rewards unconditionally, ignoring each task's declared `reward_basis`. This
runs offline with no keys.

Caveat: the `NL_ASSERTION` component is not scored. The fidelity claim covers
only the structural (DB state, action, communicate) components. A reader who
reruns with `ALL_WITH_NL_ASSERTIONS` will get different numbers; this is
expected and documented.

`gold_pass` is defined as `gold_reward >= 1.0`. tau2's product-of-components
reward: the only value satisfying all three components (DB, action, communicate)
is 1.0. Partial failures yield a product below 1.0. This definition is read
from the evaluator source; tau2 does not publish a separate pass threshold
constant.

---

## Data provisioning

tau2 ships its data files under `data/` in the repo. When installed via the
git SHA pin above, the `data/` tree is present in the uv git checkout cache.
tau2's `DATA_DIR` resolution (from `src/tau2/utils/utils.py`):

1. If the `TAU2_DATA_DIR` environment variable is set, use it.
2. Otherwise, use `Path(__file__).parents[3] / "data"` (three parents up from
   `src/tau2/utils/utils.py`).

When tau2 is installed as a wheel into a venv (as uv does), `parents[3]` of
`site-packages/tau2/utils/utils.py` is `site-packages/tau2/../../../` which
does not contain `data/`. The `TAU2_DATA_DIR` environment variable must be
set to the uv git cache checkout path.

uv caches git checkouts under:
  `~/.local/share/uv/python/` (Linux/macOS) or
  `%LOCALAPPDATA%\uv\cache\git-v0\checkouts\` (Windows)

The `make benchmark` target sets `TAU2_DATA_DIR` automatically by locating
the tau2 package source path within the venv and resolving upward to the
git checkout root. See the Makefile `find-tau2-data` helper target.

The retail domain reads:
- `data/tau2/domains/retail/tasks.json` (114 tasks)
- `data/tau2/domains/retail/db.json` (customer/order database)
- `data/tau2/domains/retail/policy.md`

---

## SimulationRun serialization

Each frozen run is a `tau2.data_model.simulation.SimulationRun` serialized
with `.model_dump_json()` (pydantic v2). Key fields used by the fidelity
harness:

| Field | Type | Notes |
|---|---|---|
| `id` | str | run UUID |
| `task_id` | str | matches task `.id` field |
| `termination_reason` | TerminationReason enum | must be AGENT_STOP or USER_STOP for reward > 0 |
| `messages` | list[Message] | half-duplex trajectory (HALF_DUPLEX mode used here) |
| `reward_info` | RewardInfo | populated by evaluate_simulation; frozen with the run |
| `mode` | str | "half_duplex" |
| `seed` | int | 300 |

`TerminationReason` values: `AGENT_STOP`, `USER_STOP`, `MAX_STEPS`,
`TIMEOUT`, `TOO_MANY_ERRORS`, `AGENT_ERROR`, `USER_ERROR`,
`INFRASTRUCTURE_ERROR`, `CONTEXT_WINDOW_EXCEEDED`, `UNEXPECTED_ERROR`.

Runs with `termination_reason` outside `{AGENT_STOP, USER_STOP}` receive
`reward=0.0` from `evaluate_simulation` (verified from evaluator source).

---

## Frozen transcript format

```
transcripts/retail/<task_id>__<model-slug>.run.json
```

Example: `transcripts/retail/0__sonnet.run.json`

Model slugs: `sonnet`, `haiku` (matching the slug column in the agents table
above).

---

## labels.jsonl

`transcripts/labels.jsonl` -- one JSON line per run:

```json
{"task_id": "0", "run": "0__sonnet", "domain": "retail", "model": "sonnet",
 "gold_reward": 1.0, "gold_pass": true}
```

Generated by `regenerate/compute_labels.py` (deterministic, no keys).
Lines are sorted by `(task_id_int, model)` for stable diffs.

---

## Fidelity claim

For each row in `labels.jsonl`, the Korrel scenario defined in
`fidelity/scenario_retail.py` must reproduce `gold_reward` to floating-point
equality (or within 1e-9 relative tolerance for float arithmetic) on the
corresponding frozen transcript, across all three runtimes.

---

## Prerequisites

- uv on PATH.
- Python 3.13 (uv will install it if absent via `.python-version`).
- On Windows: enable git long-path support before running `make sync`:
    ```
    git config --global core.longpaths true
    ```
  tau2's `web/leaderboard/` subdirectory contains filenames that exceed the
  Windows 260-character MAX_PATH limit. uv sync will fail without this setting.

## How to rerun

**Fidelity harness (offline, no keys):**

```bash
cd benchmarks
make benchmark
# reads transcripts/retail/*.run.json
# writes results/results.json
```

**Recompute labels only:**

```bash
cd benchmarks
make labels
```

**Regenerate transcripts (founder only, requires ANTHROPIC_API_KEY):**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cd benchmarks
make transcripts
# generates transcripts/retail/*.run.json
# then run: make labels
```

---

## Layout

```
benchmarks/
  README.md            -- this file (pin facts, task list, rerun instructions)
  pyproject.toml       -- uv project; Python 3.13; tau2 pinned by SHA
  .python-version      -- 3.13
  Makefile             -- make benchmark | labels | transcripts
  .gitignore           -- excludes .venv/
  transcripts/
    LICENSE            -- tau2-bench MIT license attribution
    retail/
      *.run.json       -- FROZEN; committed after founder generation run
      .gitkeep
    labels.jsonl       -- FROZEN; committed after compute_labels run
  regenerate/
    gen_transcripts.py -- keyed; calls tau2 runner; requires ANTHROPIC_API_KEY
    compute_labels.py  -- offline; reads *.run.json; writes labels.jsonl
  fidelity/
    scenario_retail.py -- Korrel Scenario whose rubric reproduces tau2 reward
    _convert.py        -- tau2 Message <-> korrel canonical Message converters
    run_fidelity.py    -- offline harness; reads labels.jsonl; writes results.json
  results/
    .gitkeep
    results.json       -- written by run_fidelity.py after founder run
    summary.md         -- written by run_fidelity.py after founder run
```
