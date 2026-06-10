"""Fidelity harness: verify that the korrel scenario reproduces tau2 gold rewards.

Four-way equality check for every frozen transcript in transcripts/retail/:
  1. korrel CI: scenario.rubric.score(completion, info).score
  2. verifiers: wrapped reward from korrel's verifiers exporter, called directly
     over the transcript without an LLM rollout.
  3. openenv: korrel's openenv exporter _compute_reward(scenario, messages) path.
  4. tau2 gold: gold_reward from transcripts/labels.jsonl.

Equality: exact float equality. The four values must be identical.

Additional assertions:
  - verifiers-path round-trip: _to_korrel_messages applied to chat-completions-
    shaped transcript reproduces the canonical messages.
  - tau2->korrel->tau2 round-trip preserves every field the evaluator reads.
  - N=20 reruns per transcript assert 0 verdict flips.

Writes results/results.json and results/summary.md (stable ordering, diffable).
Exits nonzero if any transcript fails or if no transcripts are present.

Run offline with no keys:
  uv run --project . python fidelity/run_fidelity.py
  uv run --project . python fidelity/run_fidelity.py --selftest
  # or via: make benchmark / make selftest

Built against:
  verifiers==0.1.14  -- wrapped-rubric calling convention confirmed below.
  openenv-core==0.3.0 -- reward entrypoint: _compute_reward(scenario, messages).
  tau2-bench v1.0.0 @ 17e07b1da2bbc0cadfddeea36412686e0604127b

verifiers wrapped-rubric convention (confirmed against verifiers==0.1.14
src: verifiers/types.py Rubric._call_individual_reward_func):
  score_objects(state) builds a merged dict containing:
    completion = state["completion"]  (list of vf.Message objects)
    info       = state.get("info", {})
    answer     = state.get("answer", "")
    prompt     = state["prompt"]
    ...and class_objects.
  _call_individual_reward_func then calls the function via:
    If func has **kwargs: func(**merged)
    Otherwise: func(**{k: v for k, v in merged.items() if k in sig.parameters})
  The korrel wrapper sync_wrapper(completion=None, info=None, **kwargs) has
  **kwargs, so it receives the full merged dict with completion and info.
  To drive the wrapper directly without a rollout: call it with
    wrapped_fn(completion=<vf-shaped messages>, info=<info dict>)
  which the wrapper converts via _to_korrel_messages and forwards to the
  underlying korrel reward function.

openenv reward entrypoint (confirmed against openenv-core==0.3.0 and
src: korrel/exporters/openenv.py _compute_reward):
  _compute_reward(scenario, messages) -> float
  Calls scenario.rubric.score(messages, scenario.info or {}).score.
  messages is list[korrel.types.Message] (canonical korrel messages).
  Called at episode termination; intermediate steps carry reward=None.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

# Resolve paths relative to benchmarks/
BENCHMARKS_DIR = Path(__file__).parent.parent

# Set TAU2_DATA_DIR before any tau2 import.
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))
from fidelity._tau2_data import ensure_tau2_data_dir  # noqa: E402

_tau2_data = ensure_tau2_data_dir()

LABELS_PATH = BENCHMARKS_DIR / "transcripts" / "labels.jsonl"
TRANSCRIPTS_DIR = BENCHMARKS_DIR / "transcripts" / "retail"
FIXTURES_DIR = BENCHMARKS_DIR / "fidelity" / "fixtures"
RESULTS_DIR = BENCHMARKS_DIR / "results"
RESULTS_JSON = RESULTS_DIR / "results.json"
SUMMARY_MD = RESULTS_DIR / "summary.md"

# Selftest fixture
SELFTEST_FIXTURE = FIXTURES_DIR / "synthetic_task0.run.json"

# Tolerance: exact float equality (no relaxation; if this fails, stop and report)
REL_TOL = 1e-9
ABS_TOL = 1e-9

# Number of reruns per transcript for the stability assertion.
N_RERUNS = 20

TAU2_PIN = {
    "tag": "v1.0.0",
    "sha": "17e07b1da2bbc0cadfddeea36412686e0604127b",
    "published": "2026-03-18",
}


# ---------------------------------------------------------------------------
# korrel CI leg
# ---------------------------------------------------------------------------


def _score_korrel_ci(
    scenario: Any, completion: list[Any], info: dict[str, Any]
) -> float:
    """Score via korrel Rubric.score(completion, info).score."""
    result = scenario.rubric.score(completion, info)
    return float(result.score)


# ---------------------------------------------------------------------------
# verifiers leg
# ---------------------------------------------------------------------------


def _to_verifiers_dicts(completion: list[Any]) -> list[dict[str, Any]]:
    """Convert canonical korrel Messages to verifiers-style flat dicts.

    verifiers ToolCall has flat fields {id, name, arguments} (confirmed from
    verifiers==0.1.14 src/verifiers/types.py). korrel ToolCall is nested:
    {id, type, function: {name, arguments}}. _to_korrel_messages expects the
    flat verifiers shape, so we must flatten before passing to the wrapper.

    This is what verifiers passes as state["completion"] to reward functions:
    a list of message dicts where assistant tool_calls use the flat shape.
    """
    result: list[dict[str, Any]] = []
    for m in completion:
        d: dict[str, Any] = {"role": m.role}
        if m.content is not None:
            d["content"] = m.content
        if m.tool_calls:
            flat_tcs = []
            for tc in m.tool_calls:
                flat_tc: dict[str, Any] = {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                flat_tcs.append(flat_tc)
            d["tool_calls"] = flat_tcs
        if m.tool_call_id is not None:
            d["tool_call_id"] = m.tool_call_id
        if m.name is not None:
            d["name"] = m.name
        result.append(d)
    return result


def _score_verifiers(
    scenario: Any,
    completion: list[Any],
    info: dict[str, Any],
) -> float:
    """Score via korrel's verifiers exporter wrapped-reward path.

    Confirmed against verifiers==0.1.14:
    Rubric._call_individual_reward_func builds score_objects(state) which puts
    completion and info in the merged dict. The korrel wrapper
    sync_wrapper(completion=None, info=None, **kwargs) receives these via
    **kwargs. We drive the wrapper directly without a rollout.

    The wrapped function converts completion via _to_korrel_messages (which
    expects the flat verifiers ToolCall shape {id, name, arguments}), then
    calls the underlying korrel reward function. The result must equal the
    korrel CI leg (same function, same messages after conversion).

    Note on uniform weight: to_verifiers_env sets weights=[1/n]*n. We call
    each wrapped function directly and average, matching Rubric.score() in
    korrel/rubric.py (arithmetic mean over all funcs).
    """
    from korrel.exporters.verifiers import _wrap_reward_fn

    all_fns = list(scenario.rubric.funcs)
    if scenario.rubric.judge is not None:
        all_fns.append(scenario.rubric.judge)

    # Convert to flat verifiers-style dicts (ToolCall: {id, name, arguments})
    # so the wrapper's _to_korrel_messages exercises the full conversion path.
    completion_dicts = _to_verifiers_dicts(completion)

    scores: list[float] = []
    for i, fn in enumerate(all_fns):
        fn_name = getattr(fn, "__name__", None) or f"reward_{i}"
        wrapped = _wrap_reward_fn(fn, fn_name)
        score = wrapped(completion=completion_dicts, info=info)
        scores.append(float(score))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# openenv leg
# ---------------------------------------------------------------------------


def _score_openenv(
    scenario: Any, completion: list[Any], info: dict[str, Any]
) -> float:
    """Score via korrel's OpenEnv exporter _compute_reward path.

    Confirmed against openenv-core==0.3.0 and korrel/exporters/openenv.py:
    _compute_reward(scenario, messages) calls
    scenario.rubric.score(messages, scenario.info or {}).score.
    messages is list[korrel.types.Message] (canonical korrel messages).

    The fidelity harness stores task metadata in the per-run info dict, not in
    scenario.info. _compute_reward uses scenario.info, so we temporarily set
    scenario.info to the per-run info for this scoring call.
    """
    from korrel.exporters.openenv import _compute_reward

    orig_info = scenario.info
    try:
        scenario.info = info
        reward = _compute_reward(scenario, completion)
    finally:
        scenario.info = orig_info
    return float(reward)


# ---------------------------------------------------------------------------
# Conversion round-trip assertions
# ---------------------------------------------------------------------------


def _assert_verifiers_round_trip(completion: list[Any]) -> None:
    """Assert that _to_korrel_messages(flat_dicts) reproduces completion.

    Drives the verifiers exporter's message conversion path: converts canonical
    korrel Messages to flat verifiers-style dicts (ToolCall: {id, name, arguments}),
    then converts back via _to_korrel_messages, and asserts field-by-field equality.

    Uses _to_verifiers_dicts (defined above) rather than model_dump() because
    korrel model_dump() produces the nested {id, type, function: {name, arguments}}
    shape, while _to_korrel_messages expects the flat verifiers shape.
    """
    from korrel.exporters._shared import _to_korrel_messages

    completion_dicts = _to_verifiers_dicts(completion)
    recovered = _to_korrel_messages(completion_dicts)

    assert len(recovered) == len(completion), (
        f"Round-trip message count mismatch: {len(recovered)} != {len(completion)}"
    )
    for i, (orig, rec) in enumerate(zip(completion, recovered)):
        assert orig.role == rec.role, (
            f"[{i}] role mismatch: {orig.role} != {rec.role}"
        )
        assert orig.content == rec.content, (
            f"[{i}] content mismatch: {orig.content!r} != {rec.content!r}"
        )
        if orig.tool_calls:
            assert rec.tool_calls is not None, f"[{i}] tool_calls lost"
            assert len(orig.tool_calls) == len(rec.tool_calls), (
                f"[{i}] tool_calls count: {len(orig.tool_calls)} != {len(rec.tool_calls)}"
            )
            for j, (otc, rtc) in enumerate(zip(orig.tool_calls, rec.tool_calls)):
                assert otc.id == rtc.id, (
                    f"[{i}][{j}] tc.id: {otc.id} != {rtc.id}"
                )
                assert otc.function.name == rtc.function.name, (
                    f"[{i}][{j}] tc.name: {otc.function.name} != {rtc.function.name}"
                )
                assert otc.function.arguments == rtc.function.arguments, (
                    f"[{i}][{j}] tc.arguments: "
                    f"{otc.function.arguments!r} != {rtc.function.arguments!r}"
                )
        if orig.tool_call_id is not None:
            assert orig.tool_call_id == rec.tool_call_id, (
                f"[{i}] tool_call_id: {orig.tool_call_id} != {rec.tool_call_id}"
            )


def _assert_tau2_round_trip(tau2_messages: list[Any], completion: list[Any]) -> None:
    """Assert that tau2->korrel->tau2 conversion preserves evaluator-relevant fields.

    The evaluator reads: role, content, tool_calls (name, id, arguments), and
    the ToolMessage.id that links back to the tool call. We verify these are
    preserved through the round-trip.

    MultiToolMessage flattens to individual ToolMessages; the reverse direction
    produces individual ToolMessages (no MultiToolMessage), which is correct:
    the tau2 evaluator flattens MultiToolMessages internally.
    """
    from fidelity._convert import korrel_messages_to_tau2
    from tau2.data_model.message import MultiToolMessage, ToolMessage

    tau2_round_tripped = korrel_messages_to_tau2(completion)

    def _flatten(msgs: list[Any]) -> list[Any]:
        result = []
        for m in msgs:
            if isinstance(m, MultiToolMessage):
                result.extend(m.tool_messages)
            else:
                result.append(m)
        return result

    orig_flat = _flatten(tau2_messages)
    rt_flat = _flatten(tau2_round_tripped)

    assert len(orig_flat) == len(rt_flat), (
        f"tau2 round-trip message count: {len(orig_flat)} != {len(rt_flat)}"
    )

    for i, (orig, rt) in enumerate(zip(orig_flat, rt_flat)):
        assert orig.role == rt.role, f"[{i}] role: {orig.role} != {rt.role}"
        orig_content = getattr(orig, "content", None) or ""
        rt_content = getattr(rt, "content", None) or ""
        assert orig_content == rt_content, (
            f"[{i}] content: {orig_content!r} != {rt_content!r}"
        )
        orig_tcs = getattr(orig, "tool_calls", None) or []
        rt_tcs = getattr(rt, "tool_calls", None) or []
        assert len(orig_tcs) == len(rt_tcs), (
            f"[{i}] tool_calls count: {len(orig_tcs)} != {len(rt_tcs)}"
        )
        for j, (otc, rtc) in enumerate(zip(orig_tcs, rt_tcs)):
            assert otc.id == rtc.id, f"[{i}][{j}] tc.id: {otc.id} != {rtc.id}"
            assert otc.name == rtc.name, (
                f"[{i}][{j}] tc.name: {otc.name} != {rtc.name}"
            )
            assert otc.arguments == rtc.arguments, (
                f"[{i}][{j}] tc.arguments: {otc.arguments!r} != {rtc.arguments!r}"
            )
        if isinstance(orig, ToolMessage):
            orig_id = getattr(orig, "id", None) or ""
            rt_id = getattr(rt, "id", None) or ""
            assert orig_id == rt_id, (
                f"[{i}] ToolMessage.id: {orig_id} != {rt_id}"
            )


# ---------------------------------------------------------------------------
# Per-transcript scoring
# ---------------------------------------------------------------------------


def _score_transcript(
    scenario: Any,
    simulation: Any,
    label: dict[str, Any],
) -> dict[str, Any]:
    """Score one frozen SimulationRun across all four legs.

    Returns a result dict with all four reward values, equality booleans,
    round-trip pass status, flip count, and error info.
    """
    from fidelity._convert import tau2_messages_to_korrel

    task_id: str = label["task_id"]
    run_slug: str = label["run"]
    gold_reward: float = float(label["gold_reward"])
    model: str = label.get("model", "unknown")

    tau2_messages = simulation.messages or []
    completion = tau2_messages_to_korrel(tau2_messages)

    info: dict[str, Any] = {
        "task_id": task_id,
        "termination_reason": simulation.termination_reason.value,
        "run_id": simulation.id,
        "seed": simulation.seed,
        "domain": label.get("domain", "retail"),
    }

    # Score korrel CI
    try:
        korrel_reward = _score_korrel_ci(scenario, completion, info)
    except Exception as exc:  # noqa: BLE001
        return {
            "run": run_slug,
            "task_id": task_id,
            "model": model,
            "domain": label.get("domain", "retail"),
            "gold_reward": gold_reward,
            "korrel_reward": None,
            "verifiers_reward": None,
            "openenv_reward": None,
            "eq_korrel_gold": False,
            "eq_verifiers_korrel": False,
            "eq_openenv_korrel": False,
            "eq_all_four": False,
            "round_trip_verifiers": False,
            "round_trip_tau2_korrel_tau2": False,
            "flip_count": 0,
            "passed": False,
            "error": f"korrel CI raised {type(exc).__name__}: {exc}",
        }

    # Score verifiers leg
    verifiers_error: str | None = None
    try:
        verifiers_reward: float | None = _score_verifiers(scenario, completion, info)
    except Exception as exc:  # noqa: BLE001
        verifiers_reward = None
        verifiers_error = f"verifiers leg raised {type(exc).__name__}: {exc}"

    # Score openenv leg
    openenv_error: str | None = None
    try:
        openenv_reward: float | None = _score_openenv(scenario, completion, info)
    except Exception as exc:  # noqa: BLE001
        openenv_reward = None
        openenv_error = f"openenv leg raised {type(exc).__name__}: {exc}"

    # Conversion round-trips
    rt_verifiers_ok = False
    rt_verifiers_error: str | None = None
    try:
        _assert_verifiers_round_trip(completion)
        rt_verifiers_ok = True
    except AssertionError as exc:
        rt_verifiers_error = f"verifiers round-trip: {exc}"

    rt_tau2_ok = False
    rt_tau2_error: str | None = None
    try:
        _assert_tau2_round_trip(tau2_messages, completion)
        rt_tau2_ok = True
    except AssertionError as exc:
        rt_tau2_error = f"tau2 round-trip: {exc}"

    # Stability: N_RERUNS reruns assert 0 verdict flips
    flip_count = 0
    ref_passed = korrel_reward >= scenario.rubric.pass_threshold
    for _ in range(N_RERUNS - 1):
        try:
            rescore = _score_korrel_ci(scenario, completion, info)
        except Exception:  # noqa: BLE001
            break
        if rescore != korrel_reward or (rescore >= scenario.rubric.pass_threshold) != ref_passed:
            flip_count += 1

    # Equality checks (exact float equality)
    eq_korrel_gold = gold_reward == korrel_reward
    eq_verifiers_korrel = (
        verifiers_reward is not None and verifiers_reward == korrel_reward
    )
    eq_openenv_korrel = (
        openenv_reward is not None and openenv_reward == korrel_reward
    )
    eq_all_four = (
        verifiers_reward is not None
        and openenv_reward is not None
        and gold_reward == korrel_reward == verifiers_reward == openenv_reward
    )

    errors: list[str] = []
    for e in [verifiers_error, openenv_error, rt_verifiers_error, rt_tau2_error]:
        if e:
            errors.append(e)

    passed = (
        eq_all_four
        and rt_verifiers_ok
        and rt_tau2_ok
        and flip_count == 0
    )

    return {
        "run": run_slug,
        "task_id": task_id,
        "model": model,
        "domain": label.get("domain", "retail"),
        "gold_reward": gold_reward,
        "korrel_reward": korrel_reward,
        "verifiers_reward": verifiers_reward,
        "openenv_reward": openenv_reward,
        "eq_korrel_gold": eq_korrel_gold,
        "eq_verifiers_korrel": eq_verifiers_korrel,
        "eq_openenv_korrel": eq_openenv_korrel,
        "eq_all_four": eq_all_four,
        "round_trip_verifiers": rt_verifiers_ok,
        "round_trip_tau2_korrel_tau2": rt_tau2_ok,
        "flip_count": flip_count,
        "passed": passed,
        "error": "; ".join(errors) if errors else None,
    }


# ---------------------------------------------------------------------------
# Selftest: four-way assertion against the synthetic fixture
# ---------------------------------------------------------------------------


def _korrel_version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("korrel")
    except Exception:  # noqa: BLE001
        return "unknown"


def run_selftest() -> None:
    """Run the four-way assertion against the committed synthetic fixture.

    The fixture is benchmarks/fidelity/fixtures/synthetic_task0.run.json.
    It carries termination_reason=unexpected_error so the tau2 evaluator
    returns 0.0 deterministically, exercising all conversion paths with no
    DB access for the env check and no LLM calls.

    Asserts:
    - All four legs return exactly 0.0.
    - verifiers round-trip: _to_korrel_messages(dicts) reproduces canonical messages.
    - tau2 round-trip: tau2->korrel->tau2 preserves evaluator-relevant fields.
    - 0 verdict flips over N=20 reruns.

    Exits nonzero on any assertion failure.
    """
    try:
        from fidelity.scenario_retail import scenario
        from tau2.data_model.simulation import SimulationRun
    except ImportError as exc:
        print(
            f"ERROR: import failed ({exc}). "
            "Ensure the benchmarks environment is synced: `make sync`.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not SELFTEST_FIXTURE.exists():
        print(
            f"ERROR: selftest fixture not found: {SELFTEST_FIXTURE}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"selftest: loading fixture {SELFTEST_FIXTURE.name}")
    with open(SELFTEST_FIXTURE, encoding="utf-8") as f:
        raw = json.load(f)
    simulation = SimulationRun.model_validate(raw)
    print(f"  task_id={simulation.task_id}, messages={len(simulation.messages)}")
    print(f"  termination_reason={simulation.termination_reason.value}")

    label: dict[str, Any] = {
        "task_id": simulation.task_id,
        "run": "synthetic_task0",
        "model": "synthetic",
        "domain": "retail",
        "gold_reward": 0.0,
        "gold_pass": False,
    }

    print(f"  scoring with N={N_RERUNS} stability reruns...")
    result = _score_transcript(scenario, simulation, label)

    failures: list[str] = []

    # All four legs must be 0.0
    for leg, key in [
        ("korrel CI", "korrel_reward"),
        ("verifiers", "verifiers_reward"),
        ("openenv", "openenv_reward"),
        ("tau2 gold", "gold_reward"),
    ]:
        val = result[key]
        if val != 0.0:
            failures.append(f"{leg} reward expected 0.0, got {val!r}")
        else:
            print(f"  {leg}: {val} (ok)")

    if not result["eq_all_four"]:
        failures.append(
            f"four-way equality failed: "
            f"korrel={result['korrel_reward']}, "
            f"verifiers={result['verifiers_reward']}, "
            f"openenv={result['openenv_reward']}, "
            f"gold={result['gold_reward']}"
        )
    else:
        print("  four-way equality: ok")

    if not result["round_trip_verifiers"]:
        failures.append("verifiers round-trip assertion failed")
    else:
        print("  verifiers round-trip: ok")

    if not result["round_trip_tau2_korrel_tau2"]:
        failures.append("tau2->korrel->tau2 round-trip assertion failed")
    else:
        print("  tau2 round-trip: ok")

    if result["flip_count"] > 0:
        failures.append(
            f"stability: {result['flip_count']} verdict flips over {N_RERUNS} reruns"
        )
    else:
        print(f"  stability: 0 flips over {N_RERUNS} reruns (ok)")

    if result["error"]:
        failures.append(f"errors: {result['error']}")

    # Write selftest results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selftest_output: dict[str, Any] = {
        "mode": "selftest",
        "fixture": SELFTEST_FIXTURE.name,
        "tau2_pin": TAU2_PIN,
        "verifiers_version": "0.1.14",
        "openenv_core_version": "0.3.0",
        "korrel_version": _korrel_version(),
        "python_version": platform.python_version(),
        "n_reruns_stability": N_RERUNS,
        "result": result,
        "selftest_passed": len(failures) == 0,
    }
    selftest_json = RESULTS_DIR / "selftest.json"
    with open(selftest_json, "w", encoding="utf-8") as f:
        json.dump(selftest_output, f, indent=2, sort_keys=True)
    print(f"\nselftest results written to {selftest_json}")

    if failures:
        print(f"\nSELFTEST FAIL ({len(failures)} failures):", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nSELFTEST PASS: synthetic fixture reproduces 0.0 on all four legs.")


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


def main() -> None:
    if not LABELS_PATH.exists():
        print(
            f"ERROR: {LABELS_PATH} not found.\n"
            "Generate frozen transcripts first (requires ANTHROPIC_API_KEY):\n"
            "  make transcripts && make labels\n"
            "To validate the harness offline without frozen transcripts, run:\n"
            "  make selftest\n"
            "  (or: uv run --project . python fidelity/run_fidelity.py --selftest)",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from fidelity.scenario_retail import scenario
        from tau2.data_model.simulation import SimulationRun
    except ImportError as exc:
        print(
            f"ERROR: import failed ({exc}). "
            "Ensure the benchmarks environment is synced: `make sync`.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = [json.loads(line) for line in f if line.strip()]

    if not labels:
        print(
            "ERROR: labels.jsonl is empty.\n"
            "Run `make transcripts && make labels` to generate frozen transcripts.\n"
            "To validate the harness offline, run: make selftest",
            file=sys.stderr,
        )
        sys.exit(1)

    results: list[dict[str, Any]] = []
    failures: list[str] = []

    for label in labels:
        task_id = label["task_id"]
        run_slug = label["run"]
        gold_reward: float = float(label["gold_reward"])

        run_path = TRANSCRIPTS_DIR / f"{run_slug}.run.json"
        if not run_path.exists():
            msg = f"MISSING transcript: {run_path}"
            print(msg, file=sys.stderr)
            failures.append(f"{run_slug}: {msg}")
            results.append(
                {
                    "run": run_slug,
                    "task_id": task_id,
                    "model": label.get("model", "unknown"),
                    "domain": label.get("domain", "retail"),
                    "gold_reward": gold_reward,
                    "korrel_reward": None,
                    "verifiers_reward": None,
                    "openenv_reward": None,
                    "eq_korrel_gold": False,
                    "eq_verifiers_korrel": False,
                    "eq_openenv_korrel": False,
                    "eq_all_four": False,
                    "round_trip_verifiers": False,
                    "round_trip_tau2_korrel_tau2": False,
                    "flip_count": 0,
                    "passed": False,
                    "error": msg,
                }
            )
            continue

        with open(run_path, encoding="utf-8") as f:
            raw = json.load(f)
        simulation = SimulationRun.model_validate(raw)

        result = _score_transcript(scenario, simulation, label)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        k = result["korrel_reward"]
        v = result["verifiers_reward"]
        o = result["openenv_reward"]
        g = result["gold_reward"]
        print(
            f"{status} {run_slug:40s}  "
            f"gold={g:.6f}  korrel={k:.6f}  "
            f"vf={v:.6f}  oe={o:.6f}  "
            f"flips={result['flip_count']}"
        )

        if not result["passed"]:
            details: list[str] = []
            if not result["eq_all_four"]:
                details.append(f"four-way equality failed: k={k} vf={v} oe={o} gold={g}")
            if not result["round_trip_verifiers"]:
                details.append("verifiers round-trip failed")
            if not result["round_trip_tau2_korrel_tau2"]:
                details.append("tau2 round-trip failed")
            if result["flip_count"] > 0:
                details.append(f"{result['flip_count']} verdict flips")
            if result["error"]:
                details.append(result["error"])
            failures.append(f"{run_slug}: " + "; ".join(details))

    # Aggregate
    n_total = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    n_fail = n_total - n_pass

    # Write results.json (stable key ordering, deterministic)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {
        "mode": "benchmark",
        "tau2_pin": TAU2_PIN,
        "evaluation_type": "EvaluationType.ALL_IGNORE_BASIS",
        "caveat": (
            "EvaluationType.ALL_IGNORE_BASIS: multiplies ENV*ACTION*COMMUNICATE "
            "without NL_ASSERTION (LLM-based, WIP). 38/40 retail tasks list "
            "NL_ASSERTION in reward_basis; EvaluationType.ALL raises ValueError "
            "for those tasks. ALL_IGNORE_BASIS is the correct offline type."
        ),
        "domain": "retail",
        "task_slice": "first 40 task ids by ascending numeric sort (0-39)",
        "pass_threshold": 1.0,
        "fidelity_tolerance": "exact float equality",
        "n_reruns_stability": N_RERUNS,
        "verifiers_version": "0.1.14",
        "openenv_core_version": "0.3.0",
        "korrel_version": _korrel_version(),
        "python_version": platform.python_version(),
        "n_total": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "runs": sorted(results, key=lambda r: r["run"]),
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    print(f"\nresults written to {RESULTS_JSON}")

    # Write summary.md
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("# Fidelity benchmark results\n\n")
        f.write(
            "tau2-bench pin: v1.0.0 @ 17e07b1da2bbc0cadfddeea36412686e0604127b\n\n"
        )
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Total runs | {n_total} |\n")
        f.write(f"| Pass (all four legs equal, flips=0) | {n_pass} |\n")
        f.write(f"| Fail | {n_fail} |\n")
        f.write("| verifiers version | 0.1.14 |\n")
        f.write("| openenv-core version | 0.3.0 |\n")
        f.write(f"| korrel version | {_korrel_version()} |\n")
        if failures:
            f.write("\n## Failures\n\n")
            for msg in failures:
                f.write(f"- {msg}\n")
    print(f"summary written to {SUMMARY_MD}")

    if failures:
        print(
            f"\nFAIL: {n_fail}/{n_total} runs did not pass all four-way fidelity checks.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print(
            f"\nPASS: all {n_total} runs pass all four-way fidelity checks (flips=0)."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    benchmarks_dir = str(Path(__file__).parent.parent)
    if benchmarks_dir not in sys.path:
        sys.path.insert(0, benchmarks_dir)

    parser = argparse.ArgumentParser(
        description="korrel fidelity harness: four-way equality against tau2 gold.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help=(
            "Run against the committed synthetic fixture only (offline, no keys). "
            "Exits nonzero if any assertion fails."
        ),
    )
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
    else:
        main()
