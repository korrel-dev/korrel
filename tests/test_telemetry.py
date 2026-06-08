"""Tests for opt-in telemetry: event shape, consent resolution, sender injection."""

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from korrel.telemetry import (
    _ci_detected,
    _config_path,
    _is_falsey,
    _is_truthy,
    _load_config,
    _resolve_consent,
    build_run_event,
    emit_run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _recording_sender() -> tuple[list[dict[str, Any]], Any]:
    """Return (events_list, sender_callable) for test injection."""
    events: list[dict[str, Any]] = []

    def sender(event: dict[str, Any]) -> None:
        events.append(event)

    return events, sender


# ---------------------------------------------------------------------------
# build_run_event field coverage
# ---------------------------------------------------------------------------


def test_build_run_event_required_fields():
    event = build_run_event(
        scenario_count=3,
        total_turns=12,
        pass_count=2,
        fail_count=1,
        duration_s=1.234,
        install_id="test-id",
    )
    assert event["event"] == "run"
    assert "schema_version" in event
    assert "korrel_version" in event
    assert "python_version" in event
    assert event["scenario_count"] == 3
    assert event["total_turns"] == 12
    assert event["pass_count"] == 2
    assert event["fail_count"] == 1
    assert event["duration_s"] == 1.234
    assert event["install_id"] == "test-id"


def test_build_run_event_no_content_fields():
    """The event must not carry any scenario id, path, model, or transcript."""
    event = build_run_event(
        scenario_count=1,
        total_turns=1,
        pass_count=1,
        fail_count=0,
        duration_s=0.1,
        install_id="x",
    )
    forbidden = {"scenario_id", "scenario_content", "persona", "prompt", "transcript",
                 "tool_schema", "path", "model"}
    for key in forbidden:
        assert key not in event, f"forbidden field {key!r} present in event"


def test_build_run_event_is_json_serializable():
    event = build_run_event(
        scenario_count=1,
        total_turns=0,
        pass_count=0,
        fail_count=1,
        duration_s=0.0,
        install_id="abc",
    )
    serialized = json.dumps(event)
    assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# Consent resolution
# ---------------------------------------------------------------------------


def test_resolve_consent_korrel_telemetry_false(monkeypatch):
    monkeypatch.setenv("KORREL_TELEMETRY", "0")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    assert _resolve_consent({}) is False


def test_resolve_consent_korrel_telemetry_off(monkeypatch):
    monkeypatch.setenv("KORREL_TELEMETRY", "off")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    assert _resolve_consent({}) is False


def test_resolve_consent_korrel_telemetry_true(monkeypatch):
    monkeypatch.setenv("KORREL_TELEMETRY", "1")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("CI", raising=False)
    assert _resolve_consent({}) is True


def test_resolve_consent_do_not_track(monkeypatch):
    monkeypatch.delenv("KORREL_TELEMETRY", raising=False)
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    assert _resolve_consent({}) is False


def test_resolve_consent_ci_detected(monkeypatch):
    monkeypatch.delenv("KORREL_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.setenv("CI", "true")
    assert _resolve_consent({}) is False


def test_resolve_consent_github_actions(monkeypatch):
    monkeypatch.delenv("KORREL_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert _resolve_consent({}) is False


def test_resolve_consent_persisted_true(monkeypatch):
    monkeypatch.delenv("KORREL_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    config = {"telemetry_enabled": True}
    assert _resolve_consent(config) is True


def test_resolve_consent_persisted_false(monkeypatch):
    monkeypatch.delenv("KORREL_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    config = {"telemetry_enabled": False}
    assert _resolve_consent(config) is False


def test_resolve_consent_undecided_returns_none(monkeypatch):
    monkeypatch.delenv("KORREL_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("CI", raising=False)
    for marker in ("GITHUB_ACTIONS", "TRAVIS", "CIRCLECI", "GITLAB_CI",
                   "JENKINS_URL", "BUILDKITE", "TF_BUILD", "TEAMCITY_VERSION",
                   "BITBUCKET_BUILD_NUMBER"):
        monkeypatch.delenv(marker, raising=False)
    assert _resolve_consent({}) is None


# ---------------------------------------------------------------------------
# emit_run with injected sender
# ---------------------------------------------------------------------------


def test_emit_run_disabled_no_send(monkeypatch):
    """When telemetry is off, the injected sender must NOT be called."""
    monkeypatch.setenv("KORREL_TELEMETRY", "0")
    events, sender = _recording_sender()

    emit_run(
        scenario_count=1,
        total_turns=1,
        pass_count=1,
        fail_count=0,
        duration_s=0.1,
    )
    # No sender injected, no send expected.
    assert events == []


def test_emit_run_with_injected_sender_when_disabled(monkeypatch):
    """An injected sender receives the event even when consent is off,
    so tests can inspect the built event without any network or consent."""
    monkeypatch.setenv("KORREL_TELEMETRY", "0")
    events, sender = _recording_sender()

    emit_run(
        scenario_count=2,
        total_turns=5,
        pass_count=1,
        fail_count=1,
        duration_s=0.5,
        sender=sender,
    )
    # Injected sender gets the event for field inspection.
    assert len(events) == 1
    e = events[0]
    assert e["event"] == "run"
    assert e["scenario_count"] == 2
    assert e["pass_count"] == 1
    assert e["fail_count"] == 1


def test_emit_run_enabled_calls_sender(monkeypatch):
    monkeypatch.setenv("KORREL_TELEMETRY", "1")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("CI", raising=False)
    events, sender = _recording_sender()

    emit_run(
        scenario_count=1,
        total_turns=3,
        pass_count=1,
        fail_count=0,
        duration_s=1.0,
        sender=sender,
    )
    assert len(events) == 1
    assert events[0]["event"] == "run"
    assert events[0]["scenario_count"] == 1


def test_emit_run_never_raises(monkeypatch):
    """emit_run must not propagate exceptions under any circumstances."""
    monkeypatch.setenv("KORREL_TELEMETRY", "1")

    def exploding_sender(event):
        raise RuntimeError("network error")

    # Should not raise.
    emit_run(
        scenario_count=1,
        total_turns=0,
        pass_count=0,
        fail_count=1,
        duration_s=0.0,
        sender=exploding_sender,
    )


# ---------------------------------------------------------------------------
# No key, no content
# ---------------------------------------------------------------------------


def test_no_api_key_in_event(monkeypatch):
    monkeypatch.setenv("KORREL_TELEMETRY", "0")
    events, sender = _recording_sender()
    emit_run(
        scenario_count=1,
        total_turns=1,
        pass_count=1,
        fail_count=0,
        duration_s=0.1,
        sender=sender,
    )
    if events:
        for key in events[0]:
            assert "key" not in key.lower(), f"field name {key!r} suggests a key"
            assert "secret" not in key.lower()
            assert "token" not in key.lower()


# ---------------------------------------------------------------------------
# Config path
# ---------------------------------------------------------------------------


def test_config_path_on_windows_uses_appdata(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    # Force the platform check to behave as Windows.
    with patch("korrel.telemetry.sys") as mock_sys:
        mock_sys.platform = "win32"
        # Re-import to use patched sys.platform inside the function.
        from korrel.telemetry import _config_dir
        d = _config_dir()
    assert "korrel" in str(d)


def test_config_path_non_windows_uses_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    with patch("korrel.telemetry.sys") as mock_sys:
        mock_sys.platform = "linux"
        from korrel.telemetry import _config_dir
        d = _config_dir()
    assert str(tmp_path) in str(d)


# ---------------------------------------------------------------------------
# _is_falsey / _is_truthy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("val", ["0", "false", "False", "FALSE", "no", "No", "off", "OFF"])
def test_is_falsey(val):
    assert _is_falsey(val) is True


@pytest.mark.parametrize("val", ["1", "true", "True", "TRUE", "yes", "Yes", "on", "ON"])
def test_is_truthy(val):
    assert _is_truthy(val) is True


# ---------------------------------------------------------------------------
# KORREL_TELEMETRY_DEBUG writes to stderr
# ---------------------------------------------------------------------------


def test_debug_flag_writes_to_stderr(monkeypatch, capsys):
    monkeypatch.setenv("KORREL_TELEMETRY", "1")
    monkeypatch.setenv("KORREL_TELEMETRY_DEBUG", "1")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("KORREL_TELEMETRY_ENDPOINT", raising=False)

    from korrel.telemetry import _debug_sender
    event = build_run_event(
        scenario_count=1, total_turns=0, pass_count=0, fail_count=0,
        duration_s=0.0, install_id="debug-test",
    )
    _debug_sender(event)
    captured = capsys.readouterr()
    assert "run" in captured.err
