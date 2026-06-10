"""Fidelity harness: verify that the korrel scenario reproduces tau2 gold rewards.

For each row in transcripts/labels.jsonl:
  1. Load the corresponding frozen *.run.json.
  2. Convert tau2 messages to korrel canonical messages.
  3. Call scenario.rubric.score(completion, info) to get the korrel reward.
  4. Compare against gold_reward from labels.jsonl.

Pass criterion: abs(korrel_reward - gold_reward) <= max(1e-9, 1e-9 * abs(gold_reward))
(i.e., floating-point equality within 1e-9 relative tolerance).

Writes results/results.json and results/summary.md.
Exits with code 1 if any task fails.

Run offline with no keys:
  uv run --project . python fidelity/run_fidelity.py
  # or via: make benchmark
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Resolve paths relative to benchmarks/
BENCHMARKS_DIR = Path(__file__).parent.parent

# Set TAU2_DATA_DIR before any tau2 import (tau2.utils.utils reads it at load time).
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))
from fidelity._tau2_data import ensure_tau2_data_dir  # noqa: E402
_tau2_data = ensure_tau2_data_dir()
LABELS_PATH = BENCHMARKS_DIR / "transcripts" / "labels.jsonl"
TRANSCRIPTS_DIR = BENCHMARKS_DIR / "transcripts" / "retail"
RESULTS_DIR = BENCHMARKS_DIR / "results"
RESULTS_JSON = RESULTS_DIR / "results.json"
SUMMARY_MD = RESULTS_DIR / "summary.md"

# Tolerance: floating-point equality within 1e-9 relative tolerance.
REL_TOL = 1e-9
ABS_TOL = 1e-9


def _check_close(a: float, b: float) -> bool:
    if a == b:
        return True
    denom = max(abs(b), ABS_TOL)
    return abs(a - b) <= REL_TOL * denom


def main() -> None:
    if not LABELS_PATH.exists():
        print(
            f"ERROR: {LABELS_PATH} not found. "
            "Run `make labels` (or `make transcripts && make labels`) first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Import scenario and converter here (deferred) so that the import error
    # message is clear if tau2 is not installed.
    try:
        from fidelity._convert import tau2_messages_to_korrel
        from fidelity.scenario_retail import scenario

        from tau2.data_model.simulation import SimulationRun
    except ImportError as exc:
        print(
            f"ERROR: import failed ({exc}). "
            "Ensure the benchmarks environment is synced: `make sync`.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load labels
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = [json.loads(line) for line in f if line.strip()]

    results: list[dict[str, Any]] = []
    failures: list[str] = []

    for label in labels:
        task_id: str = label["task_id"]
        run_slug: str = label["run"]
        gold_reward: float = label["gold_reward"]
        model: str = label["model"]

        run_path = TRANSCRIPTS_DIR / f"{run_slug}.run.json"
        if not run_path.exists():
            msg = f"MISSING transcript: {run_path}"
            print(msg, file=sys.stderr)
            failures.append(f"{run_slug}: {msg}")
            results.append(
                {
                    "run": run_slug,
                    "task_id": task_id,
                    "model": model,
                    "gold_reward": gold_reward,
                    "korrel_reward": None,
                    "passed": False,
                    "error": msg,
                }
            )
            continue

        # Load frozen SimulationRun
        with open(run_path, encoding="utf-8") as f:
            raw = json.load(f)
        simulation = SimulationRun.model_validate(raw)

        # Convert tau2 messages to korrel canonical messages
        messages = simulation.messages or []
        completion = tau2_messages_to_korrel(messages)

        # Build info dict for the reward function
        info: dict[str, Any] = {
            "task_id": task_id,
            "termination_reason": simulation.termination_reason.value,
            "run_id": simulation.id,
            "seed": simulation.seed,
            "domain": "retail",
        }

        # Score with korrel scenario rubric
        try:
            rubric_result = scenario.rubric.score(completion, info)
            korrel_reward = rubric_result.score
        except Exception as exc:  # noqa: BLE001
            msg = f"rubric raised {type(exc).__name__}: {exc}"
            print(f"ERROR {run_slug}: {msg}", file=sys.stderr)
            failures.append(f"{run_slug}: {msg}")
            results.append(
                {
                    "run": run_slug,
                    "task_id": task_id,
                    "model": model,
                    "gold_reward": gold_reward,
                    "korrel_reward": None,
                    "passed": False,
                    "error": msg,
                }
            )
            continue

        passed = _check_close(korrel_reward, gold_reward)
        status = "PASS" if passed else "FAIL"
        print(
            f"{status} {run_slug:40s}  gold={gold_reward:.6f}  korrel={korrel_reward:.6f}"
        )

        if not passed:
            delta = abs(korrel_reward - gold_reward)
            failures.append(
                f"{run_slug}: gold={gold_reward} korrel={korrel_reward} delta={delta}"
            )

        results.append(
            {
                "run": run_slug,
                "task_id": task_id,
                "model": model,
                "gold_reward": gold_reward,
                "korrel_reward": korrel_reward,
                "passed": passed,
                "error": None,
            }
        )

    # Aggregate
    n_total = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    n_fail = n_total - n_pass

    # Write results.json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "tau2_pin": {
            "tag": "v1.0.0",
            "sha": "17e07b1da2bbc0cadfddeea36412686e0604127b",
            "published": "2026-03-18",
        },
        "evaluation_type": "EvaluationType.ALL_IGNORE_BASIS",
        "caveat": (
            "EvaluationType.ALL_IGNORE_BASIS: multiplies ENV*ACTION*COMMUNICATE "
            "without NL_ASSERTION (LLM-based, WIP). 38/40 tasks have NL_ASSERTION "
            "in reward_basis; EvaluationType.ALL raises ValueError for those tasks."
        ),
        "domain": "retail",
        "task_slice": "first 40 task ids by ascending numeric sort (0-39)",
        "pass_threshold": 1.0,
        "fidelity_tolerance": f"abs <= max({ABS_TOL}, {REL_TOL} * abs(gold))",
        "n_total": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "runs": results,
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nresults written to {RESULTS_JSON}")

    # Write summary.md (dx-doc-writer expands this)
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("# Fidelity benchmark results\n\n")
        f.write(f"tau2-bench pin: v1.0.0 @ 17e07b1da2bbc0cadfddeea36412686e0604127b\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Total runs | {n_total} |\n")
        f.write(f"| Pass (korrel == gold) | {n_pass} |\n")
        f.write(f"| Fail | {n_fail} |\n")
        if failures:
            f.write("\n## Failures\n\n")
            for msg in failures:
                f.write(f"- {msg}\n")
    print(f"summary written to {SUMMARY_MD}")

    if failures:
        print(f"\nFAIL: {n_fail}/{n_total} runs did not reproduce gold reward.",
              file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nPASS: all {n_total} runs reproduce gold reward.")


if __name__ == "__main__":
    # Ensure benchmarks/ is on sys.path so `from fidelity._convert import ...` works.
    benchmarks_dir = str(Path(__file__).parent.parent)
    if benchmarks_dir not in sys.path:
        sys.path.insert(0, benchmarks_dir)
    main()
