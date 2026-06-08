"""Unit tests for rubric aggregation, thresholds, and failed-function lists."""

from korrel import Message, Rubric


def msgs() -> list[Message]:
    return [Message(role="assistant", content="hello")]


def always_one(completion, info, **kwargs) -> float:
    return 1.0


def always_zero(completion, info, **kwargs) -> float:
    return 0.0


async def async_half(completion, info, **kwargs) -> float:
    return 0.5


def test_aggregate_is_mean_of_scores():
    rubric = Rubric(funcs=[always_one, always_zero], pass_threshold=0.5)
    result = rubric.score(msgs(), {})
    assert result.score == 0.5
    assert result.scores == {"always_one": 1.0, "always_zero": 0.0}


def test_pass_threshold_pass_and_fail():
    passing = Rubric(funcs=[always_one], pass_threshold=0.9).score(msgs(), {})
    assert passing.passed is True
    assert passing.failed_functions == []

    failing = Rubric(funcs=[always_zero], pass_threshold=0.5).score(msgs(), {})
    assert failing.passed is False
    assert failing.failed_functions == ["always_zero"]


def test_failed_functions_lists_only_below_threshold():
    rubric = Rubric(funcs=[always_one, always_zero], pass_threshold=0.5)
    result = rubric.score(msgs(), {})
    assert result.failed_functions == ["always_zero"]


def test_async_reward_function_is_awaited():
    rubric = Rubric(funcs=[async_half, always_one], pass_threshold=0.5)
    result = rubric.score(msgs(), {})
    assert result.scores["async_half"] == 0.5
    assert result.score == 0.75
    assert result.passed is True


def test_judge_is_scored_alongside_funcs():
    def fake_judge(completion, info, **kwargs) -> float:
        return 0.0

    fake_judge.__name__ = "judge"
    rubric = Rubric(funcs=[always_one], pass_threshold=0.5, judge=fake_judge)
    result = rubric.score(msgs(), {})
    assert "judge" in result.scores
    assert result.score == 0.5
    assert "judge" in result.failed_functions
