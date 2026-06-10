"""Compute tau2 gold labels from frozen SimulationRun transcripts.

For each transcripts/retail/*.run.json, calls tau2.evaluator.evaluator.evaluate_simulation
with EvaluationType.ALL_IGNORE_BASIS (no LLM calls: product of ENV*ACTION*COMMUNICATE,
ignoring each task's declared reward_basis) and writes transcripts/labels.jsonl.

Rationale for ALL_IGNORE_BASIS:
  38 of 40 benchmark retail tasks (task ids 0-39) have NL_ASSERTION in reward_basis.
  EvaluationType.ALL raises ValueError when NL_ASSERTION is in the basis but not
  being evaluated (confirmed from tau2 v1.0.0 evaluator.py source).
  EvaluationType.ALL_WITH_NL_ASSERTIONS requires a live LLM call and is marked WIP.
  EvaluationType.ALL_IGNORE_BASIS computes DB*ACTION*COMMUNICATE offline.
  This caveat is documented in the README and in run_fidelity.py output.

Output format (one JSON line per run, sorted by (task_id_int, model)):
  {"task_id": "0", "run": "0__sonnet", "domain": "retail",
   "model": "sonnet", "gold_reward": 1.0, "gold_pass": true}

gold_pass = gold_reward >= 1.0
tau2's product-of-components reward: the only value satisfying all required
components is 1.0; partial failures yield a value in (0.0, 1.0).
Runs whose termination_reason is outside {AGENT_STOP, USER_STOP} receive
reward=0.0 from evaluate_simulation (verified from evaluator source).

Deterministic, offline, no keys.
Run with: uv run --project . python regenerate/compute_labels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Resolve paths relative to benchmarks/
BENCHMARKS_DIR = Path(__file__).parent.parent

# Set TAU2_DATA_DIR before any tau2 import (tau2.utils.utils reads it at load time).
sys.path.insert(0, str(BENCHMARKS_DIR))
from fidelity._tau2_data import ensure_tau2_data_dir  # noqa: E402
_tau2_data = ensure_tau2_data_dir()
TRANSCRIPTS_DIR = BENCHMARKS_DIR / "transcripts" / "retail"
LABELS_PATH = BENCHMARKS_DIR / "transcripts" / "labels.jsonl"

# gold_pass threshold: reward must equal 1.0 (product of all required components)
PASS_THRESHOLD = 1.0


def main() -> None:
    try:
        from tau2.data_model.simulation import SimulationRun
        from tau2.domains.retail.environment import get_tasks as retail_get_tasks
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
        from tau2.orchestrator.modes import CommunicationMode
    except ImportError as exc:
        print(
            f"ERROR: import failed ({exc}). "
            "Run `make sync` to install the benchmarks environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_files = sorted(TRANSCRIPTS_DIR.glob("*.run.json"))
    if not run_files:
        print(
            f"ERROR: no *.run.json files found in {TRANSCRIPTS_DIR}. "
            "Run `make transcripts` first (requires ANTHROPIC_API_KEY).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load all retail tasks once (avoids repeated file reads)
    all_tasks = retail_get_tasks(task_split_name=None)
    task_by_id = {task.id: task for task in all_tasks}

    rows = []
    errors = []

    for path in run_files:
        # Filename pattern: <task_id>__<model-slug>.run.json
        stem = path.stem  # e.g. "0__sonnet"
        parts = stem.split("__", 1)
        if len(parts) != 2:
            print(
                f"WARNING: unexpected filename {path.name}, skipping",
                file=sys.stderr,
            )
            continue
        task_id, model_slug = parts[0], parts[1]

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        try:
            simulation = SimulationRun.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            msg = f"parse error for {path.name}: {exc}"
            print(f"ERROR: {msg}", file=sys.stderr)
            errors.append(msg)
            continue

        task = task_by_id.get(task_id)
        if task is None:
            msg = f"task_id '{task_id}' not found in retail tasks"
            print(f"ERROR: {msg}", file=sys.stderr)
            errors.append(msg)
            continue

        try:
            reward_info = evaluate_simulation(
                simulation=simulation,
                task=task,
                # ALL_IGNORE_BASIS: offline, no LLM. See module docstring.
                evaluation_type=EvaluationType.ALL_IGNORE_BASIS,
                solo_mode=False,
                domain="retail",
                mode=CommunicationMode.HALF_DUPLEX,
                env_kwargs=None,
            )
            gold_reward = float(reward_info.reward)
        except Exception as exc:  # noqa: BLE001
            msg = f"evaluate_simulation failed for {path.name}: {exc}"
            print(f"ERROR: {msg}", file=sys.stderr)
            errors.append(msg)
            continue

        gold_pass = gold_reward >= PASS_THRESHOLD
        rows.append(
            {
                "task_id": task_id,
                "run": stem,
                "domain": "retail",
                "model": model_slug,
                "gold_reward": gold_reward,
                "gold_pass": gold_pass,
            }
        )
        status = "PASS" if gold_pass else "FAIL"
        print(f"{status}  {stem:40s}  reward={gold_reward:.6f}")

    if errors:
        print(f"\n{len(errors)} error(s) encountered; labels.jsonl not written.",
              file=sys.stderr)
        sys.exit(1)

    # Sort by (task_id_int, model) for stable diffs
    rows.sort(key=lambda r: (int(r["task_id"]), r["model"]))

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            # Stable float formatting: repr gives enough precision without noise.
            line = json.dumps(row, ensure_ascii=False)
            f.write(line + "\n")

    n_pass = sum(1 for r in rows if r["gold_pass"])
    n_fail = len(rows) - n_pass
    print(
        f"\nWrote {len(rows)} labels to {LABELS_PATH} "
        f"({n_pass} pass, {n_fail} fail)."
    )


if __name__ == "__main__":
    main()
