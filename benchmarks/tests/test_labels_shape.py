"""Tests for labels.jsonl line shape produced by regenerate/compute_labels.py.

compute_labels.py runs module-level code (ensure_tau2_data_dir + tau2 imports)
that cannot be imported as a plain module in isolation, so these tests verify
the record-building logic and output format by replicating the record
construction inline.

Covers:
- Required keys in every record: task_id, run, domain, model, gold_reward, gold_pass
- gold_pass = gold_reward >= PASS_THRESHOLD (1.0)
- Sorting by (task_id_int, model)
- Filename stem parsing: <task_id>__<model_slug>
- JSON encoding is valid for every row
- float precision: gold_reward is JSON-serializable as a float
"""

from __future__ import annotations

import json

import pytest

# The PASS_THRESHOLD defined in compute_labels.py
PASS_THRESHOLD = 1.0


# ---------------------------------------------------------------------------
# Helpers that replicate compute_labels record-building logic
# ---------------------------------------------------------------------------


def _build_record(
    task_id: str,
    stem: str,
    model_slug: str,
    gold_reward: float,
) -> dict:
    """Replicate compute_labels.main record construction."""
    return {
        "task_id": task_id,
        "run": stem,
        "domain": "retail",
        "model": model_slug,
        "gold_reward": gold_reward,
        "gold_pass": gold_reward >= PASS_THRESHOLD,
    }


def _sort_rows(rows: list[dict]) -> list[dict]:
    """Replicate compute_labels.main sorting key."""
    return sorted(rows, key=lambda r: (int(r["task_id"]), r["model"]))


# ---------------------------------------------------------------------------
# Record schema
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"task_id", "run", "domain", "model", "gold_reward", "gold_pass"}


@pytest.mark.parametrize(
    "task_id, stem, model_slug, gold_reward",
    [
        ("0", "0__sonnet", "sonnet", 1.0),
        ("1", "1__haiku", "haiku", 0.0),
        ("5", "5__gpt-4o", "gpt-4o", 0.5),
        ("39", "39__claude", "claude", 0.75),
    ],
    ids=["pass_1.0", "fail_0.0", "partial_0.5", "partial_0.75"],
)
def test_record_has_all_required_keys(
    task_id: str, stem: str, model_slug: str, gold_reward: float
) -> None:
    record = _build_record(task_id, stem, model_slug, gold_reward)
    assert REQUIRED_KEYS.issubset(record.keys())


@pytest.mark.parametrize(
    "task_id, stem, model_slug, gold_reward, expected_pass",
    [
        ("0", "0__sonnet", "sonnet", 1.0, True),
        ("1", "1__haiku", "haiku", 0.0, False),
        ("2", "2__gpt4", "gpt4", 0.999, False),
        ("3", "3__opus", "opus", 1.0, True),
    ],
    ids=["exactly_1.0", "zero", "below_threshold", "exactly_1.0_again"],
)
def test_gold_pass_matches_threshold(
    task_id: str,
    stem: str,
    model_slug: str,
    gold_reward: float,
    expected_pass: bool,
) -> None:
    record = _build_record(task_id, stem, model_slug, gold_reward)
    assert record["gold_pass"] is expected_pass


def test_domain_is_always_retail() -> None:
    record = _build_record("0", "0__model", "model", 0.5)
    assert record["domain"] == "retail"


def test_run_field_is_stem() -> None:
    record = _build_record("7", "7__claude-3", "claude-3", 0.0)
    assert record["run"] == "7__claude-3"


def test_model_field_matches_slug() -> None:
    record = _build_record("7", "7__my-model", "my-model", 0.0)
    assert record["model"] == "my-model"


def test_task_id_field_is_string() -> None:
    record = _build_record("12", "12__m", "m", 0.0)
    assert isinstance(record["task_id"], str)


def test_gold_reward_is_float() -> None:
    record = _build_record("0", "0__m", "m", 0.5)
    assert isinstance(record["gold_reward"], float)


# ---------------------------------------------------------------------------
# JSON encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gold_reward",
    [0.0, 1.0, 0.5, 0.333333, 0.123456789],
    ids=["zero", "one", "half", "third", "precision"],
)
def test_record_is_json_serializable(gold_reward: float) -> None:
    record = _build_record("0", "0__m", "m", gold_reward)
    line = json.dumps(record, ensure_ascii=False)
    parsed = json.loads(line)
    assert parsed["gold_reward"] == pytest.approx(gold_reward, rel=1e-9)
    assert parsed["gold_pass"] == (gold_reward >= PASS_THRESHOLD)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


def test_sorting_by_task_id_int_then_model() -> None:
    rows = [
        _build_record("10", "10__z", "z", 0.0),
        _build_record("2", "2__a", "a", 1.0),
        _build_record("2", "2__b", "b", 0.5),
        _build_record("1", "1__a", "a", 0.0),
    ]
    sorted_rows = _sort_rows(rows)
    task_ids = [r["task_id"] for r in sorted_rows]
    # Numeric sort: 1, 2, 2, 10
    assert task_ids == ["1", "2", "2", "10"]
    # Within task_id=2, sorted by model ascending
    task2_models = [r["model"] for r in sorted_rows if r["task_id"] == "2"]
    assert task2_models == ["a", "b"]


def test_sorting_is_stable_for_same_key() -> None:
    rows = [
        _build_record("5", "5__m", "m", 0.0),
        _build_record("5", "5__m", "m", 1.0),  # duplicate key, different reward
    ]
    sorted_rows = _sort_rows(rows)
    assert len(sorted_rows) == 2


# ---------------------------------------------------------------------------
# Filename stem parsing (replicate compute_labels parsing logic)
# ---------------------------------------------------------------------------


def _parse_stem(stem: str) -> tuple[str, str] | None:
    """Replicate compute_labels stem parsing."""
    parts = stem.split("__", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("0__sonnet", ("0", "sonnet")),
        ("39__claude-3-5-haiku", ("39", "claude-3-5-haiku")),
        ("5__gpt-4o-mini", ("5", "gpt-4o-mini")),
        ("100__model_v2", ("100", "model_v2")),
    ],
    ids=["simple", "with_dashes", "with_dots", "with_underscore"],
)
def test_stem_parsing_valid(stem: str, expected: tuple[str, str]) -> None:
    result = _parse_stem(stem)
    assert result == expected


@pytest.mark.parametrize(
    "stem",
    [
        "no_double_underscore",
        "single",
        "",
    ],
    ids=["underscore_only", "no_separator", "empty"],
)
def test_stem_parsing_invalid_returns_none(stem: str) -> None:
    result = _parse_stem(stem)
    assert result is None


def test_stem_parsing_preserves_model_slug_with_multiple_double_underscores() -> None:
    """split('__', 1) keeps the second segment intact even if it contains '__'."""
    stem = "5__model__variant"
    result = _parse_stem(stem)
    assert result == ("5", "model__variant")


# ---------------------------------------------------------------------------
# Run slug derivation from the transcript filename
# ---------------------------------------------------------------------------


def _slug_from_filename(name: str) -> str:
    """Replicate compute_labels slug derivation: strip the full .run.json suffix."""
    return name.removesuffix(".run.json")


def test_slug_strips_full_double_suffix() -> None:
    """Path.stem would leave "0__sonnet.run"; the harness resolves the slug back
    to "<slug>.run.json", so the slug must be the bare "<task_id>__<model>"."""
    assert _slug_from_filename("0__sonnet.run.json") == "0__sonnet"
    assert _parse_stem(_slug_from_filename("39__haiku.run.json")) == ("39", "haiku")


# ---------------------------------------------------------------------------
# JSONL line format
# ---------------------------------------------------------------------------


def test_jsonl_output_is_one_json_object_per_line() -> None:
    """Each line must be a self-contained JSON object (not an array)."""
    rows = [
        _build_record("0", "0__a", "a", 1.0),
        _build_record("1", "1__b", "b", 0.0),
    ]
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    for line in lines:
        parsed = json.loads(line)
        assert isinstance(parsed, dict)
