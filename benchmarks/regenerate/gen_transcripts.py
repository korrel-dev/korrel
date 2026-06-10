"""Generate frozen tau2 retail transcripts for the fidelity benchmark.

REQUIRES: ANTHROPIC_API_KEY environment variable set at run time.
Key is read from the environment at call time and never stored.

Runs tau2's runner on the 40-task retail slice (task ids 0-39, first 40 by
ascending numeric sort) with two agent models:
  strong: anthropic/claude-sonnet-4-6
  weak:   anthropic/claude-3-5-haiku-20241022

tau2 uses litellm under the hood. Provider prefix: "anthropic/" routes via
litellm's Anthropic provider. A single ANTHROPIC_API_KEY is sufficient.

For each (task, model), keeps runs with termination_reason in
{AGENT_STOP, USER_STOP}; retries up to MAX_RETRIES times for other outcomes;
skips and logs if all retries exhaust. Skipped tasks are written to a
skipped.json sidecar for auditability.

Output: transcripts/retail/<task_id>__<model-slug>.run.json
One file per (task, model). Files are serialized with SimulationRun.model_dump_json().

Determinism knobs:
  - seed=300 (tau2 DEFAULT_SEED, applied to both agent and user)
  - temperature=0 for both agent and user (tau2 DEFAULT_LLM_TEMPERATURE_*)
  - tau2's litellm routing passes seed/temperature to Anthropic's API.

Run with:
  export ANTHROPIC_API_KEY=sk-ant-...
  uv run --project . python regenerate/gen_transcripts.py
  # or via: make transcripts
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Resolve paths relative to benchmarks/
BENCHMARKS_DIR = Path(__file__).parent.parent

# Set TAU2_DATA_DIR before any tau2 import (tau2.utils.utils reads it at load time).
sys.path.insert(0, str(BENCHMARKS_DIR))
from fidelity._tau2_data import ensure_tau2_data_dir  # noqa: E402
_tau2_data = ensure_tau2_data_dir()
TRANSCRIPTS_DIR = BENCHMARKS_DIR / "transcripts" / "retail"
SKIPPED_PATH = BENCHMARKS_DIR / "transcripts" / "retail" / "skipped.json"

# Task slice: first 40 retail task ids by ascending numeric sort.
# Confirmed from data/tau2/domains/retail/tasks.json at pinned SHA:
# 114 tasks with integer id strings "0".."113"; first 40 = "0".."39".
TASK_IDS = [str(i) for i in range(40)]

# Model configs: (slug, litellm_model_string)
# tau2 uses litellm; "anthropic/" prefix routes to Anthropic's API.
AGENT_MODELS = [
    ("sonnet", "anthropic/claude-sonnet-4-6"),
    ("haiku", "anthropic/claude-3-5-haiku-20241022"),
]

# User simulator model (same key, anthropic prefix)
USER_MODEL = "anthropic/claude-sonnet-4-6"

# tau2 defaults confirmed from src/tau2/config.py at pinned SHA
SEED = 300
TEMPERATURE = 0.0
MAX_STEPS = 200
MAX_ERRORS = 10
MAX_RETRIES = 3


def _check_api_key() -> None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or not key.strip():
        print(
            "ERROR: ANTHROPIC_API_KEY is not set. "
            "Export it before running gen_transcripts.py.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    _check_api_key()

    try:
        from tau2.data_model.simulation import SimulationRun, TerminationReason, TextRunConfig
        from tau2.domains.retail.environment import get_tasks as retail_get_tasks
        from tau2.evaluator.evaluator import EvaluationType
        from tau2.runner import build_text_orchestrator, run_simulation
    except ImportError as exc:
        print(
            f"ERROR: import failed ({exc}). "
            "Run `make sync` to install the benchmarks environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    # Acceptable termination reasons; others are retried or skipped.
    GOOD_TERMINATIONS = {
        TerminationReason.AGENT_STOP,
        TerminationReason.USER_STOP,
    }

    # Load all retail tasks once.
    all_tasks = retail_get_tasks(task_split_name=None)
    task_by_id = {task.id: task for task in all_tasks}

    # Validate all task ids are present.
    missing = [tid for tid in TASK_IDS if tid not in task_by_id]
    if missing:
        print(f"ERROR: task ids not found in retail data: {missing}", file=sys.stderr)
        sys.exit(1)

    skipped: list[dict] = []
    generated = 0
    already_present = 0

    for model_slug, llm_model in AGENT_MODELS:
        print(f"\n=== model: {model_slug} ({llm_model}) ===")
        for task_id in TASK_IDS:
            out_path = TRANSCRIPTS_DIR / f"{task_id}__{model_slug}.run.json"
            if out_path.exists():
                print(f"  SKIP (exists) {task_id}__{model_slug}")
                already_present += 1
                continue

            task = task_by_id[task_id]
            config = TextRunConfig(
                domain="retail",
                task_ids=[task_id],
                llm_agent=llm_model,
                llm_args_agent={"temperature": TEMPERATURE, "seed": SEED},
                llm_user=USER_MODEL,
                llm_args_user={"temperature": TEMPERATURE, "seed": SEED},
                max_steps=MAX_STEPS,
                max_errors=MAX_ERRORS,
                seed=SEED,
                num_trials=1,
                max_concurrency=1,
                max_retries=0,  # we handle retries ourselves below
                log_level="ERROR",
            )

            simulation: SimulationRun | None = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    orchestrator = build_text_orchestrator(
                        config, task, seed=SEED
                    )
                    run = run_simulation(
                        orchestrator,
                        # ALL_IGNORE_BASIS: offline component. The frozen run
                        # stores reward_info for reference; compute_labels.py
                        # recomputes from scratch for the labels file.
                        evaluation_type=EvaluationType.ALL_IGNORE_BASIS,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  attempt {attempt}/{MAX_RETRIES} FAILED "
                        f"{task_id}__{model_slug}: {exc}",
                        file=sys.stderr,
                    )
                    if attempt == MAX_RETRIES:
                        skipped.append(
                            {
                                "task_id": task_id,
                                "model_slug": model_slug,
                                "reason": f"exception: {exc}",
                            }
                        )
                    continue

                if run.termination_reason in GOOD_TERMINATIONS:
                    simulation = run
                    break
                else:
                    print(
                        f"  attempt {attempt}/{MAX_RETRIES} bad termination "
                        f"{run.termination_reason.value} for {task_id}__{model_slug}",
                        file=sys.stderr,
                    )
                    if attempt == MAX_RETRIES:
                        skipped.append(
                            {
                                "task_id": task_id,
                                "model_slug": model_slug,
                                "reason": (
                                    f"termination_reason={run.termination_reason.value} "
                                    "after all retries"
                                ),
                            }
                        )

            if simulation is None:
                print(
                    f"  SKIPPED {task_id}__{model_slug} "
                    f"(see {SKIPPED_PATH.name})",
                    file=sys.stderr,
                )
                continue

            # Freeze: serialize with pydantic v2 model_dump_json (tau2's own format)
            json_str = simulation.model_dump_json(indent=2)
            out_path.write_text(json_str, encoding="utf-8")
            reward = simulation.reward_info.reward if simulation.reward_info else "n/a"
            print(
                f"  SAVED {task_id}__{model_slug}  "
                f"termination={simulation.termination_reason.value}  "
                f"reward={reward}"
            )
            generated += 1

    # Write skipped sidecar
    if skipped:
        SKIPPED_PATH.write_text(
            json.dumps(skipped, indent=2), encoding="utf-8"
        )
        print(
            f"\nWARNING: {len(skipped)} task(s) skipped. See {SKIPPED_PATH}.",
            file=sys.stderr,
        )

    print(
        f"\nDone. Generated={generated}, already_present={already_present}, "
        f"skipped={len(skipped)}."
    )
    print(
        "Next step: run `make labels` to compute gold labels from the frozen transcripts."
    )


if __name__ == "__main__":
    main()
