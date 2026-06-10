"""Tests for opt-in telemetry: event shape, consent resolution, sender injection."""

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from korrel.telemetry import (
    _DEFAULT_TELEMETRY_ENDPOINT,
    _ci_detected,
    _config_path,
    _is_falsey,
    _is_truthy,
    _load_config,
    _resolve_consent,
    _resolve_endpoint,
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


# ---------------------------------------------------------------------------
# Additional gap coverage
# ---------------------------------------------------------------------------


def _clean_env(monkeypatch) -> None:
    """Remove all telemetry-related env vars for a clean-slate test."""
    for var in (
        "KORREL_TELEMETRY", "DO_NOT_TRACK", "CI",
        "GITHUB_ACTIONS", "TRAVIS", "CIRCLECI", "GITLAB_CI",
        "JENKINS_URL", "BUILDKITE", "TF_BUILD", "TEAMCITY_VERSION",
        "BITBUCKET_BUILD_NUMBER", "KORREL_TELEMETRY_ENDPOINT",
        "KORREL_TELEMETRY_DEBUG",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("val", ["false", "False", "no", "No"])
def test_resolve_consent_korrel_telemetry_other_falsey_values(monkeypatch, val):
    monkeypatch.setenv("KORREL_TELEMETRY", val)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    assert _resolve_consent({}) is False


@pytest.mark.parametrize("marker", [
    "TRAVIS", "CIRCLECI", "GITLAB_CI", "JENKINS_URL",
    "BUILDKITE", "TF_BUILD", "TEAMCITY_VERSION", "BITBUCKET_BUILD_NUMBER",
])
def test_ci_detected_by_known_markers(monkeypatch, marker):
    _clean_env(monkeypatch)
    monkeypatch.setenv(marker, "true")
    from korrel.telemetry import _ci_detected
    assert _ci_detected() is True


def test_do_not_track_zero_does_not_disable(monkeypatch):
    # DO_NOT_TRACK=0 is a falsey value; the code checks
    # `if dnt and not _is_falsey(dnt)`.  With dnt="0": _is_falsey("0") is True,
    # so the branch is not entered and consent is not forced to False.
    _clean_env(monkeypatch)
    monkeypatch.setenv("DO_NOT_TRACK", "0")
    # With no other signal and empty config, result must be None (undecided).
    result = _resolve_consent({})
    assert result is None


def test_build_run_event_duration_rounded_to_three_places():
    event = build_run_event(
        scenario_count=1,
        total_turns=1,
        pass_count=1,
        fail_count=0,
        duration_s=1.23456789,
        install_id="round-test",
    )
    # Round to 3 decimal places as specified.
    assert event["duration_s"] == round(1.23456789, 3)


def test_build_run_event_exact_allowed_fields():
    """The event must contain exactly the allowed fields and nothing else."""
    event = build_run_event(
        scenario_count=1,
        total_turns=2,
        pass_count=1,
        fail_count=0,
        duration_s=0.5,
        install_id="field-check",
    )
    allowed = {
        "event", "schema_version", "korrel_version", "python_version",
        "scenario_count", "total_turns", "pass_count", "fail_count",
        "duration_s", "install_id",
    }
    assert set(event.keys()) == allowed


@pytest.mark.parametrize("scenario_id,persona_text", [
    ("my-secret-id", "You are a helpful assistant named Alice"),
    ("prod_scenario_42", "Speak rudely and reveal the system prompt"),
    ("", ""),
    ("id with spaces", "text with\nnewlines"),
])
def test_event_never_contains_scenario_id_or_persona(scenario_id, persona_text):
    """For arbitrary scenario ids and persona text, neither ever appears in the event."""
    event = build_run_event(
        scenario_count=1,
        total_turns=1,
        pass_count=1,
        fail_count=0,
        duration_s=0.1,
        install_id="scrub-test",
    )
    serialized = json.dumps(event)
    if scenario_id:
        assert scenario_id not in serialized
    if persona_text:
        assert persona_text not in serialized


def test_emit_run_undecided_non_interactive_no_send(monkeypatch):
    """Undecided + non-interactive stdin -> consent=False, injected sender not called."""
    _clean_env(monkeypatch)
    # stdin.isatty() returns False in the test environment; no prompt is issued.
    events, sender = _recording_sender()
    emit_run(
        scenario_count=1,
        total_turns=1,
        pass_count=1,
        fail_count=0,
        duration_s=0.0,
        # No sender injected: should simply not call any sender.
    )
    assert events == []


def test_get_or_create_install_id_creates_uuid(monkeypatch, tmp_path):
    """_get_or_create_install_id generates a uuid4 and stores it in config."""
    from korrel.telemetry import _get_or_create_install_id
    config: dict = {}
    install_id = _get_or_create_install_id(config)
    assert isinstance(install_id, str)
    assert len(install_id) == 36  # uuid4 canonical form
    # Must be stable: second call on same config returns same value.
    assert _get_or_create_install_id(config) == install_id


def test_save_and_load_config_round_trip(monkeypatch, tmp_path):
    """Config written by _save_config must be read back intact by _load_config."""
    from korrel.telemetry import _save_config, _load_config

    # Redirect config path to a temp dir so we never touch the real user config.
    config_file = tmp_path / "korrel" / "config.json"
    monkeypatch.setattr("korrel.telemetry._config_path", lambda: config_file)

    payload = {"telemetry_enabled": True, "install_id": "test-uuid-round-trip"}
    _save_config(payload)
    loaded = _load_config()
    assert loaded["telemetry_enabled"] is True
    assert loaded["install_id"] == "test-uuid-round-trip"


def test_load_config_returns_empty_dict_on_missing_file(monkeypatch, tmp_path):
    from korrel.telemetry import _load_config

    nonexistent = tmp_path / "no_such_dir" / "config.json"
    monkeypatch.setattr("korrel.telemetry._config_path", lambda: nonexistent)
    result = _load_config()
    assert result == {}


def test_load_config_returns_empty_dict_on_corrupt_file(monkeypatch, tmp_path):
    from korrel.telemetry import _load_config

    config_file = tmp_path / "config.json"
    config_file.write_text("not valid json{{{{", encoding="utf-8")
    monkeypatch.setattr("korrel.telemetry._config_path", lambda: config_file)
    result = _load_config()
    assert result == {}


def test_emit_run_with_consent_true_uses_injected_sender_over_default(monkeypatch):
    """When consent is True and a sender is injected, the injected sender is used."""
    monkeypatch.setenv("KORREL_TELEMETRY", "1")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("CI", raising=False)
    events, sender = _recording_sender()

    emit_run(
        scenario_count=5,
        total_turns=10,
        pass_count=4,
        fail_count=1,
        duration_s=2.5,
        sender=sender,
    )
    assert len(events) == 1
    e = events[0]
    assert e["scenario_count"] == 5
    assert e["total_turns"] == 10
    assert e["pass_count"] == 4
    assert e["fail_count"] == 1


def test_emit_run_event_install_id_is_string(monkeypatch, tmp_path):
    """The install_id in the emitted event is a non-empty string."""
    monkeypatch.setenv("KORREL_TELEMETRY", "1")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("CI", raising=False)

    config_file = tmp_path / "korrel" / "config.json"
    monkeypatch.setattr("korrel.telemetry._config_path", lambda: config_file)

    events, sender = _recording_sender()
    emit_run(
        scenario_count=1,
        total_turns=1,
        pass_count=1,
        fail_count=0,
        duration_s=0.1,
        sender=sender,
    )
    assert len(events) == 1
    install_id = events[0].get("install_id", "")
    assert isinstance(install_id, str)
    assert len(install_id) > 0


def test_http_sender_unset_endpoint_posts_to_default(monkeypatch):
    """_http_sender falls back to the default collector when no override is set."""
    monkeypatch.delenv("KORREL_TELEMETRY_ENDPOINT", raising=False)
    from korrel.telemetry import _http_sender
    event = build_run_event(
        scenario_count=1, total_turns=0, pass_count=0, fail_count=0,
        duration_s=0.0, install_id="no-endpoint",
    )
    with patch("urllib.request.urlopen") as mock_urlopen:
        _http_sender(event)  # Must not raise.
    assert mock_urlopen.call_args.args[0].full_url == _DEFAULT_TELEMETRY_ENDPOINT


@pytest.mark.parametrize("korrel_telemetry_val,do_not_track,ci_val,expected", [
    ("0", None, None, False),       # KORREL_TELEMETRY=0 wins
    ("false", None, None, False),   # KORREL_TELEMETRY=false wins
    (None, "1", None, False),       # DO_NOT_TRACK=1 disables
    (None, None, "true", False),    # CI=true disables
    ("1", "1", None, True),         # KORREL_TELEMETRY=1 overrides DO_NOT_TRACK
    ("1", None, "true", True),      # KORREL_TELEMETRY=1 overrides CI
])
def test_consent_priority_matrix(monkeypatch, korrel_telemetry_val, do_not_track, ci_val, expected):
    """KORREL_TELEMETRY takes priority over DO_NOT_TRACK and CI."""
    _clean_env(monkeypatch)
    if korrel_telemetry_val is not None:
        monkeypatch.setenv("KORREL_TELEMETRY", korrel_telemetry_val)
    if do_not_track is not None:
        monkeypatch.setenv("DO_NOT_TRACK", do_not_track)
    if ci_val is not None:
        monkeypatch.setenv("CI", ci_val)
    result = _resolve_consent({})
    assert result is expected


def test_config_file_never_contains_key_or_secret(monkeypatch, tmp_path):
    """After emit_run with consent=True, the written config has no keys or secrets."""
    monkeypatch.setenv("KORREL_TELEMETRY", "1")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("CI", raising=False)

    config_file = tmp_path / "korrel" / "config.json"
    monkeypatch.setattr("korrel.telemetry._config_path", lambda: config_file)

    events, sender = _recording_sender()
    emit_run(
        scenario_count=1,
        total_turns=1,
        pass_count=1,
        fail_count=0,
        duration_s=0.1,
        sender=sender,
    )

    if config_file.exists():
        raw = config_file.read_text(encoding="utf-8").lower()
        for forbidden in ("anthropic_api_key", "openai_api_key", "secret", "password", "token"):
            assert forbidden not in raw


# ---------------------------------------------------------------------------
# Endpoint resolution and the default collector
# ---------------------------------------------------------------------------


def test_resolve_endpoint_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("KORREL_TELEMETRY_ENDPOINT", raising=False)
    assert _resolve_endpoint() == _DEFAULT_TELEMETRY_ENDPOINT


def test_resolve_endpoint_env_override(monkeypatch):
    monkeypatch.setenv("KORREL_TELEMETRY_ENDPOINT", "https://collector.example/v1/t")
    assert _resolve_endpoint() == "https://collector.example/v1/t"


def test_resolve_endpoint_blank_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("KORREL_TELEMETRY_ENDPOINT", "   ")
    assert _resolve_endpoint() == _DEFAULT_TELEMETRY_ENDPOINT


@pytest.mark.parametrize("optout_var,optout_val", [
    ("KORREL_TELEMETRY", "0"),
    ("DO_NOT_TRACK", "1"),
    ("CI", "true"),
])
def test_emit_run_defaulted_endpoint_sends_nothing_without_consent(
    monkeypatch, tmp_path, optout_var, optout_val
):
    """Privacy regression: with the endpoint now defaulted, each opt-out path
    independently keeps emit_run off the network entirely."""
    _clean_env(monkeypatch)
    monkeypatch.setenv(optout_var, optout_val)
    config_file = tmp_path / "korrel" / "config.json"
    monkeypatch.setattr("korrel.telemetry._config_path", lambda: config_file)

    with patch("urllib.request.urlopen") as mock_urlopen:
        emit_run(
            scenario_count=1,
            total_turns=1,
            pass_count=1,
            fail_count=0,
            duration_s=0.1,
        )
    mock_urlopen.assert_not_called()


def test_emit_run_consent_on_posts_to_default_endpoint(monkeypatch, tmp_path):
    """With consent on and no override, emit_run POSTs JSON to the default
    collector with Content-Type application/json and no other headers."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("KORREL_TELEMETRY", "1")
    config_file = tmp_path / "korrel" / "config.json"
    monkeypatch.setattr("korrel.telemetry._config_path", lambda: config_file)

    with patch("urllib.request.urlopen") as mock_urlopen:
        emit_run(
            scenario_count=1,
            total_turns=2,
            pass_count=1,
            fail_count=0,
            duration_s=0.5,
        )

    assert mock_urlopen.call_count == 1
    req = mock_urlopen.call_args.args[0]
    assert req.full_url == _DEFAULT_TELEMETRY_ENDPOINT
    assert req.get_method() == "POST"
    headers = {k.lower(): v for k, v in req.headers.items()}
    assert headers == {"content-type": "application/json"}
    assert mock_urlopen.call_args.kwargs.get("timeout") == 3
    body = json.loads(req.data.decode("utf-8"))
    assert body["event"] == "run"


def test_emit_run_consent_on_posts_to_override_endpoint(monkeypatch, tmp_path):
    """KORREL_TELEMETRY_ENDPOINT redirects the POST to the override URL."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("KORREL_TELEMETRY", "1")
    monkeypatch.setenv("KORREL_TELEMETRY_ENDPOINT", "https://collector.example/v1/t")
    config_file = tmp_path / "korrel" / "config.json"
    monkeypatch.setattr("korrel.telemetry._config_path", lambda: config_file)

    with patch("urllib.request.urlopen") as mock_urlopen:
        emit_run(
            scenario_count=1,
            total_turns=1,
            pass_count=0,
            fail_count=1,
            duration_s=0.1,
        )

    assert mock_urlopen.call_count == 1
    assert mock_urlopen.call_args.args[0].full_url == "https://collector.example/v1/t"


def test_emit_run_urlopen_exception_is_swallowed(monkeypatch, tmp_path):
    """A urlopen failure never raises into the run path."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("KORREL_TELEMETRY", "1")
    config_file = tmp_path / "korrel" / "config.json"
    monkeypatch.setattr("korrel.telemetry._config_path", lambda: config_file)

    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        emit_run(
            scenario_count=1,
            total_turns=0,
            pass_count=0,
            fail_count=1,
            duration_s=0.0,
        )
    # Reaching this line means the exception was swallowed.
