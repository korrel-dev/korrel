# korrel tau2 fidelity benchmark

**Status: foundations committed; frozen transcripts and results not yet generated.**

This artifact proves one claim: a single Korrel scenario definition reproduces
tau2-bench's gold reward across three runtimes, korrel pytest CI, verifiers
exported environment, and OpenEnv exporter, to exact float equality per frozen
transcript. The selftest (two committed synthetic fixtures) is the standing
evidence until the founder runs transcript generation.

---

## What this measures and what it deliberately does not

**What it measures.** One Korrel scenario definition authored once in
`fidelity/scenario_retail.py` is scored four ways against each frozen
transcript:

1. Korrel CI: `scenario.rubric.score(completion, info).score`
2. verifiers: korrel's verifiers exporter wrapped-reward path, called directly
   over the frozen transcript without an LLM rollout (verifiers==0.1.14,
   confirmed from `fidelity/run_fidelity.py`).
3. OpenEnv: korrel's OpenEnv exporter `_compute_reward(scenario, messages)`
   path (openenv-core==0.3.0, confirmed from `fidelity/run_fidelity.py`).
4. tau2 gold: `gold_reward` from `transcripts/labels.jsonl`, computed offline
   by `regenerate/compute_labels.py` using
   `tau2.evaluator.evaluator.evaluate_simulation` at the pinned SHA.

All four values must be identical (exact float equality, tolerance 1e-9) for
every frozen transcript. The harness also asserts two round-trips: the
verifiers message-conversion path (`_to_korrel_messages` applied to
chat-completions-shaped dicts reproduces canonical messages) and the
tau2->korrel->tau2 path (evaluator-relevant fields preserved). Each transcript
is scored 20 times and zero verdict flips are required.

**What it deliberately does not measure.** This is not a head-to-head scoring
contest against other evaluation frameworks. No framework's aggregate pass rate
is compared. The claim is narrower: Korrel's authoring layer can express a real
benchmark's reward function once and carry it across CI and RL runtimes without
drift. The two agent models (strong and weak) are chosen to produce a label set
spanning pass and fail, not to rank those models.

---

## tau2-bench pin

| Field | Value |
|---|---|
| Repository | https://github.com/sierra-research/tau2-bench |
| Tag | v1.0.0 |
| Commit SHA | 17e07b1da2bbc0cadfddeea36412686e0604127b |
| Published | 2026-03-18 |
| License | MIT (notice in `transcripts/LICENSE`) |
| Python requirement | >=3.13,<3.14 (benchmarks pyproject.toml) |
| Package name | tau2 (version 0.2.1-dev at that tag) |

SHA verified on 2026-06-10:

```
gh api repos/sierra-research/tau2-bench/git/refs/tags/v1.0.0
# -> object.sha = 17e07b1da2bbc0cadfddeea36412686e0604127b
```

The dependency is pinned as a git source install in `benchmarks/pyproject.toml`:

```toml
tau2 = { git = "https://github.com/sierra-research/tau2-bench",
         rev = "17e07b1da2bbc0cadfddeea36412686e0604127b" }
```

tau2's own imports (`tau2.runner`, `tau2.registry`) require the `voice` and
`knowledge` extras unconditionally at import time. Both are listed in the
benchmarks dependency declaration.

---

## Four-leg fidelity claim and method

The central assertion of `make benchmark` is:

```
for each row in transcripts/labels.jsonl:
  korrel_ci == verifiers_wrapped == openenv_reward == gold_reward
```

to exact float equality. The fidelity harness (`fidelity/run_fidelity.py`)
implements this check as follows.

**Korrel CI leg.** `scenario.rubric.score(completion, info).score`. The rubric
wraps `tau2_retail_reward`, which reconstructs a `SimulationRun` from the
canonical korrel messages and the `info` dict, looks up the retail task by
`task_id`, and calls `evaluate_simulation` with
`EvaluationType.ALL_IGNORE_BASIS`. No LLM calls; fully offline.

**verifiers leg.** `korrel.exporters.verifiers._wrap_reward_fn` is called
directly for each reward function in the rubric. The canonical messages are
first converted to flat verifiers-style dicts (ToolCall shape: `{id, name,
arguments}`), then the wrapped function converts them back via
`_to_korrel_messages` and forwards to the underlying korrel reward function.
The average across all wrapped functions is computed, matching korrel's
`Rubric.score` arithmetic mean. No LLM rollout occurs. Confirmed against
verifiers==0.1.14 calling convention (see `fidelity/run_fidelity.py` module
docstring).

**OpenEnv leg.** `korrel.exporters.openenv._compute_reward(scenario, messages)`
is called with the canonical messages and the per-run `info` temporarily
assigned to `scenario.info`. Confirmed against openenv-core==0.3.0 (see
`fidelity/run_fidelity.py` module docstring).

**tau2 gold leg.** `gold_reward` from `transcripts/labels.jsonl`. This value
is computed offline by `regenerate/compute_labels.py`, which calls
`evaluate_simulation` directly on each frozen `SimulationRun` with
`EvaluationType.ALL_IGNORE_BASIS`.

**Equality check.** Exact float equality (`gold_reward == korrel_reward ==
verifiers_reward == openenv_reward`). The four values are produced by the same
underlying computation path; float arithmetic noise is not expected. The harness
sets `REL_TOL = ABS_TOL = 1e-9` as a safety margin but the intent is exact
equality.

**Selftest.** Before scoring frozen transcripts, `make benchmark` runs a
selftest against two committed synthetic fixtures:

- `fidelity/fixtures/synthetic_task0.run.json`: `termination_reason =
  unexpected_error`, expected reward `0.0` on all four legs. Exercises the fail
  path and all conversion helpers.
- `fidelity/fixtures/synthetic_task0_pass.run.json`: `termination_reason =
  agent_stop`, all five required actions present with correct arguments and
  exact tool responses, expected reward `1.0` on all four legs. Exercises the
  ENV and ACTION evaluator paths and proves the harness distinguishes pass from
  fail.

Each fixture is scored with 20 reruns and zero verdict flips are required. An
additional guard asserts that the pass fixture's agreed reward is greater than
`0.0`, which catches a broken leg that always returns `0.0` (four-way equality
at `0.0` would otherwise pass silently).

---

## Layout map

```
benchmarks/
  README.md              -- this file
  pyproject.toml         -- uv project; Python 3.13; tau2 pinned by SHA
  .python-version        -- 3.13
  Makefile               -- make benchmark | labels | transcripts | selftest
  .gitignore             -- excludes .venv/

  transcripts/
    LICENSE              -- tau2-bench MIT license attribution
    retail/
      *.run.json         -- FROZEN; committed after founder generation run
      .gitkeep
    labels.jsonl         -- FROZEN; committed after compute_labels run

  regenerate/
    gen_transcripts.py   -- keyed; calls tau2 runner; requires ANTHROPIC_API_KEY
    compute_labels.py    -- offline; reads *.run.json; writes labels.jsonl

  fidelity/
    scenario_retail.py   -- Korrel Scenario whose rubric reproduces tau2 reward
    _convert.py          -- tau2 Message <-> korrel canonical Message converters
    _tau2_data.py        -- TAU2_DATA_DIR auto-discovery helper
    run_fidelity.py      -- offline harness; reads labels.jsonl; writes results.json
    fixtures/
      synthetic_task0.run.json       -- committed; fail fixture (expected 0.0)
      synthetic_task0_pass.run.json  -- committed; pass fixture (expected 1.0)

  tests/
    test_convert.py          -- unit + property-based tests for _convert.py
    test_rubric_fidelity.py  -- rubric-matches-evaluate_simulation assertions
    test_labels_shape.py     -- labels.jsonl record schema and sorting
    test_tau2_data.py        -- TAU2_DATA_DIR discovery
    test_no_tau2_in_korrel_core.py  -- guard: tau2 must not appear in korrel src

  results/
    .gitkeep
    results.json   -- written by run_fidelity.py (main benchmark run)
    summary.md     -- written by run_fidelity.py (main benchmark run)
    selftest.json  -- written by run_fidelity.py --selftest
```

---

## Domain and task slice

Domain: **retail** (114 tasks total in the base split at the pinned SHA).

Slice rule: first 40 task IDs by ascending numeric sort of the `id` field in
`data/tau2/domains/retail/tasks.json` (confirmed from `fidelity/scenario_retail.py`
and `regenerate/gen_transcripts.py`).

```
Task IDs (40 total):
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
```

These span both the `train` split (0-4, 6-8, 10-11, 13-16, 19-25, 28-31,
34-35, 37) and the `test` split (5, 9, 12, 17, 18, 26, 27, 32, 33, 36, 38,
39) from split_tasks.json. The slice is the first 40 by numeric id,
independent of split membership. No cherry-picking.

---

## Agent models

Two models generate trajectories so the label set spans pass and fail:

| Slug | tau2 / litellm model string | Role |
|---|---|---|
| sonnet | `anthropic/claude-sonnet-4-6` | strong agent |
| haiku | `anthropic/claude-3-5-haiku-20241022` | weak agent |

Both use `temperature=0` and `seed=300` (tau2's `DEFAULT_SEED` from
`src/tau2/config.py` at the pinned SHA). tau2 routes through litellm;
the `anthropic/` prefix routes to Anthropic's API. A single
`ANTHROPIC_API_KEY` read from the environment at call time is sufficient
for transcript generation. Keys are never stored.

Runs with `termination_reason` outside `{AGENT_STOP, USER_STOP}` receive
`reward=0.0` from `evaluate_simulation` (verified from tau2 v1.0.0 evaluator
source; confirmed in `regenerate/gen_transcripts.py` and `fidelity/scenario_retail.py`).
The generation script retries up to three times for a run with a good
termination, then skips and logs to a `skipped.json` sidecar.

---

## Evaluation function

`tau2.evaluator.evaluator.evaluate_simulation` at the pinned SHA.

Signature (confirmed from installed source; cited in `fidelity/run_fidelity.py`):

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

This benchmark uses `EvaluationType.ALL_IGNORE_BASIS`: multiplies
`ENV * ACTION * COMMUNICATE` rewards unconditionally, ignoring each task's
declared `reward_basis`. Runs offline with no LLM calls.

`gold_pass` is defined as `gold_reward >= 1.0`. tau2's product-of-components
reward reaches `1.0` only when all three components are satisfied; partial
failures yield a product below `1.0`. This definition is read from the
evaluator source; tau2 does not publish a separate pass-threshold constant.

---

## How to rerun

### Prerequisites

- uv on PATH.
- Python 3.13 (uv will install it if absent via `.python-version`).
- Windows: enable git long-path support before `make sync`:
  ```
  git config --global core.longpaths true
  ```
  tau2's `web/leaderboard/` subdirectory has filenames exceeding the Windows
  260-character MAX_PATH limit. `uv sync` will fail without this setting.

### Hostile-reader path (offline, no keys)

```bash
cd benchmarks
make benchmark
```

`make benchmark` runs `make selftest` first (synthetic fixtures, committed),
then scores every frozen transcript in `transcripts/retail/` across all four
legs, writes `results/results.json` and `results/summary.md`, and exits
nonzero if any transcript fails.

If frozen transcripts are not yet present, `make selftest` alone runs the
harness against the two committed synthetic fixtures:

```bash
cd benchmarks
make selftest
```

### Keyed regeneration path (founder run; requires ANTHROPIC_API_KEY)

Frozen transcripts are not yet committed. The founder generates them once:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cd benchmarks
make transcripts   # calls tau2 runner; writes transcripts/retail/*.run.json
make labels        # calls evaluate_simulation offline; writes transcripts/labels.jsonl
```

After those files are committed, `make benchmark` runs the full fidelity
harness offline with no keys.

### Recompute labels only (offline, no keys)

```bash
cd benchmarks
make labels
```

Reads all `transcripts/retail/*.run.json`, calls `evaluate_simulation` with
`EvaluationType.ALL_IGNORE_BASIS`, and writes `transcripts/labels.jsonl`
sorted by `(task_id_int, model)` for stable diffs.

---

## Caveats

### ALL_IGNORE_BASIS: NL assertions are off

This benchmark uses `EvaluationType.ALL_IGNORE_BASIS`, not
`EvaluationType.ALL`. The reason: 38 of the 40 benchmark tasks (ids 0-39)
list `NL_ASSERTION` in their `reward_basis` (confirmed from
`data/tau2/domains/retail/tasks.json` at the pinned SHA). At tau2 v1.0.0,
`EvaluationType.ALL` raises `ValueError` when `NL_ASSERTION` is in the
basis but is not being evaluated (confirmed from `evaluator.py` source,
line 229). `EvaluationType.ALL_WITH_NL_ASSERTIONS` is marked WIP and
requires a live LLM call.

`EvaluationType.ALL_IGNORE_BASIS` multiplies `ENV * ACTION * COMMUNICATE`
unconditionally. The NL assertion component is not scored. The fidelity
claim covers only the structural (DB state, action, communicate) components.

A reader who reruns with `NL_ASSERTION` evaluation enabled will get different
numbers. This is expected and documented here and in `fidelity/run_fidelity.py`.

### Frozen-transcript status

Frozen transcripts (`transcripts/retail/*.run.json`) and labels
(`transcripts/labels.jsonl`) are not yet committed. The fidelity table in
`results/summary.md` is produced by `make benchmark` and will land with the
frozen transcripts after the founder's generation run. Until then, the
selftest against the two committed synthetic fixtures is the standing evidence
that the four-leg harness is correct.

### Data provisioning: TAU2_DATA_DIR

tau2 ships its data files under `data/` in the repository. When installed via
the git SHA pin, `data/` is present in uv's git checkout cache. tau2's
`DATA_DIR` resolution (from `src/tau2/utils/utils.py`) checks
`TAU2_DATA_DIR` first, then falls back to `Path(__file__).parents[3] / "data"`.
When tau2 is installed as a wheel into a venv, that fallback path does not
contain `data/`. The `fidelity/_tau2_data.py` helper auto-discovers the correct
path; the Makefile sets it automatically. If auto-discovery fails (for example,
uv cache in a non-standard location), set `TAU2_DATA_DIR` manually to the git
checkout root before running any script.

---

## Capability matrix

The rows below describe what each tool does. This table credits tools for their
stated capabilities; it does not score or rank them.

| Tool | Capabilities |
|---|---|
| DeepEval (https://deepeval.com) | Open-source LLM evaluation framework; 50+ research-backed metrics covering hallucination, faithfulness, answer relevancy, conversational metrics, and others; pytest-native CI integration; conversation simulator for synthetic golden generation across user personas. |
| Coval (https://www.coval.ai) | Voice agent evaluation and production monitoring platform; simulation, benchmarking, and real-time observability for voice AI across the deployment lifecycle; closed commercial SaaS, so results produced on its platform cannot be independently reproduced outside it. |
| Braintrust (https://www.braintrust.dev) | Tracing, logging, and evaluation platform for LLM applications; captures execution traces, supports experiments comparing model versions and prompt variants, and provides a CLI (`bt`) for repeatable evaluation workflows. |
| tau2-bench (https://github.com/sierra-research/tau2-bench) | Multi-domain agent benchmark with retail, airline, banking, and telco domains; structured task definitions, gold evaluator (`evaluate_simulation`), and split-based task organization; MIT licensed; this artifact builds on it directly. |
| Korrel (https://github.com/sierra-research/tau2-bench) | Define a multi-turn agent test once (scenario, persona, mock tools, rubric); run it as a pytest CI gate today and export it as a verifiers or OpenEnv RL environment when ready; this artifact is the fidelity proof for that claim. |

---

## SimulationRun serialization reference

Each frozen run is a `tau2.data_model.simulation.SimulationRun` serialized
with `.model_dump_json()` (pydantic v2). Fields used by the fidelity harness:

| Field | Type | Notes |
|---|---|---|
| `id` | str | run UUID |
| `task_id` | str | matches task `.id` field |
| `termination_reason` | TerminationReason enum | must be AGENT_STOP or USER_STOP for reward > 0 |
| `messages` | list[Message] | half-duplex trajectory (HALF_DUPLEX mode) |
| `reward_info` | RewardInfo | populated at generation time; harness recomputes |
| `mode` | str | "half_duplex" |
| `seed` | int | 300 |

Transcript filename pattern: `transcripts/retail/<task_id>__<model-slug>.run.json`
(example: `transcripts/retail/0__sonnet.run.json`).

`labels.jsonl` format (one JSON line per run, sorted by `(task_id_int, model)`):

```json
{"task_id": "0", "run": "0__sonnet", "domain": "retail", "model": "sonnet",
 "gold_reward": 1.0, "gold_pass": true}
```
