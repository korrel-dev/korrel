"""Opt-in telemetry for korrel.

Design goals:
- Best-effort. Telemetry must never raise into the run path.
- No content. The event carries only aggregate numeric counters and
  metadata; never scenario ids, persona text, transcripts, prompts, tool
  schemas, paths, or anything identifying.
- Consent-first. Multiple opt-out signals are respected, checked in order
  before any prompt or send attempt.
- Pluggable sender. Tests inject a recording sender so they can assert on
  the built event without any network I/O.

Consent resolution order (first match wins):
1. KORREL_TELEMETRY in environment and falsey ("0", "false", "no", "off") -> off.
2. DO_NOT_TRACK truthy in environment -> off.
3. CI detected (CI truthy, or known CI env vars present) -> off, no prompt.
4. Interactive TTY and no persisted decision -> prompt once, persist answer.
5. Non-interactive or undecided -> off, no prompt.

The persisted config file stores only:
- "telemetry_enabled": bool
- "install_id": a uuid4 string generated once

No key, scenario id, model name, path, prompt, transcript, or tool schema
is ever written to the config file.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

# The "run" event schema version. Increment when the field set changes.
_EVENT_SCHEMA_VERSION = "1"


# ---------------------------------------------------------------------------
# Config-file path resolution
# ---------------------------------------------------------------------------


def _config_dir() -> Path:
    """Return the OS-appropriate config directory for korrel."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "korrel"
        return Path.home() / "AppData" / "Roaming" / "korrel"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "korrel"
    return Path.home() / ".config" / "korrel"


def _config_path() -> Path:
    return _config_dir() / "config.json"


# ---------------------------------------------------------------------------
# Persisted config I/O
# ---------------------------------------------------------------------------


def _load_config() -> dict[str, Any]:
    path = _config_path()
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_config(config: dict[str, Any]) -> None:
    path = _config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    except Exception:
        pass


def _get_or_create_install_id(config: dict[str, Any]) -> str:
    """Return the persisted install id, creating one if absent."""
    if "install_id" not in config:
        config["install_id"] = str(uuid.uuid4())
    return config["install_id"]


# ---------------------------------------------------------------------------
# Consent resolution
# ---------------------------------------------------------------------------


def _is_falsey(value: str) -> bool:
    return value.strip().lower() in ("0", "false", "no", "off")


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _ci_detected() -> bool:
    """Return True when a CI environment is detected."""
    ci_env = os.environ.get("CI", "")
    # Any non-empty, non-falsey value of CI counts as a CI environment.
    if ci_env.strip() and not _is_falsey(ci_env):
        return True
    # Known CI-specific env vars.
    ci_markers = (
        "GITHUB_ACTIONS",
        "TRAVIS",
        "CIRCLECI",
        "GITLAB_CI",
        "JENKINS_URL",
        "BUILDKITE",
        "TF_BUILD",
        "TEAMCITY_VERSION",
        "BITBUCKET_BUILD_NUMBER",
    )
    return any(os.environ.get(m, "") for m in ci_markers)


def _resolve_consent(config: dict[str, Any]) -> Optional[bool]:
    """Determine consent from env and config without prompting.

    Returns True (enabled), False (disabled), or None (undecided, no
    prompt appropriate).
    """
    # 1. Explicit KORREL_TELEMETRY override.
    kt = os.environ.get("KORREL_TELEMETRY", "")
    if kt:
        if _is_falsey(kt):
            return False
        if _is_truthy(kt):
            return True

    # 2. DO_NOT_TRACK.
    dnt = os.environ.get("DO_NOT_TRACK", "")
    if dnt and not _is_falsey(dnt):
        return False

    # 3. CI -> off without prompt.
    if _ci_detected():
        return False

    # 4. Persisted decision.
    if "telemetry_enabled" in config:
        return bool(config["telemetry_enabled"])

    # 5. Undecided.
    return None


def _prompt_consent(config: dict[str, Any]) -> bool:
    """Ask the user once on an interactive TTY. Persist the answer."""
    try:
        if not sys.stdin.isatty() or not sys.stderr.isatty():
            return False
        sys.stderr.write(
            "\nkorrel collects anonymous usage data (run counts, pass/fail "
            "aggregates) to improve the project. No scenario content, "
            "prompts, transcripts, or identifying information is sent.\n"
            "Allow? [y/N] "
        )
        sys.stderr.flush()
        answer = sys.stdin.readline().strip().lower()
        enabled = answer in ("y", "yes")
        config["telemetry_enabled"] = enabled
        _save_config(config)
        return enabled
    except Exception:
        return False


def is_enabled() -> bool:
    """Return True if telemetry is enabled, without prompting."""
    config = _load_config()
    result = _resolve_consent(config)
    return result is True


# ---------------------------------------------------------------------------
# Event construction
# ---------------------------------------------------------------------------


def build_run_event(
    *,
    scenario_count: int,
    total_turns: int,
    pass_count: int,
    fail_count: int,
    duration_s: float,
    install_id: str,
) -> dict[str, Any]:
    """Build a ``run`` telemetry event.

    Only aggregate numeric counters and korrel/python version metadata are
    included. No scenario id, persona, transcript, prompt, tool schema,
    path, or model name is ever present in this event.
    """
    from . import __version__

    return {
        "event": "run",
        "schema_version": _EVENT_SCHEMA_VERSION,
        "korrel_version": __version__,
        "python_version": platform.python_version(),
        "scenario_count": scenario_count,
        "total_turns": total_turns,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "duration_s": round(duration_s, 3),
        "install_id": install_id,
    }


# ---------------------------------------------------------------------------
# Sender
# ---------------------------------------------------------------------------

# The default sender uses urllib.request. Tests inject a recording sender.
Sender = Callable[[dict[str, Any]], None]

# The public, unauthenticated, content-free collector (per the cloud
# collector design). KORREL_TELEMETRY_ENDPOINT overrides it for
# self-hosting.
_DEFAULT_TELEMETRY_ENDPOINT = (
    "https://tsbvccnafjkqgizbbdwy.supabase.co/functions/v1/telemetry"
)


def _resolve_endpoint() -> str:
    """Return the collector URL: the env override if set, else the default."""
    return (
        os.environ.get("KORREL_TELEMETRY_ENDPOINT", "").strip()
        or _DEFAULT_TELEMETRY_ENDPOINT
    )


def _http_sender(event: dict[str, Any]) -> None:
    """POST the event JSON to the resolved telemetry endpoint."""
    import urllib.request

    endpoint = _resolve_endpoint()
    try:
        payload = json.dumps(event).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=3)
    except Exception:
        pass


def _debug_sender(event: dict[str, Any]) -> None:
    """Write the event JSON to stderr (used when KORREL_TELEMETRY_DEBUG is set)."""
    try:
        sys.stderr.write(f"[korrel telemetry] {json.dumps(event)}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _default_sender(event: dict[str, Any]) -> None:
    """POST the event to the resolved collector, or write it to stderr
    when KORREL_TELEMETRY_DEBUG is set."""
    if os.environ.get("KORREL_TELEMETRY_DEBUG", ""):
        _debug_sender(event)
    else:
        _http_sender(event)


# ---------------------------------------------------------------------------
# Public emit function
# ---------------------------------------------------------------------------


def emit_run(
    *,
    scenario_count: int,
    total_turns: int,
    pass_count: int,
    fail_count: int,
    duration_s: float,
    sender: Optional[Sender] = None,
) -> None:
    """Build and (conditionally) send a ``run`` telemetry event.

    Consent is checked first. If disabled or undecided, the event is
    built only when a recording sender is injected (for test inspection),
    and nothing is sent. Never raises.
    """
    try:
        config = _load_config()
        consent = _resolve_consent(config)

        if consent is None:
            # Attempt a one-time interactive prompt.
            consent = _prompt_consent(config)

        if not consent:
            if sender is not None:
                # Injected sender (tests): build and deliver the event so
                # assertions can verify the field set, but only when the
                # caller explicitly passes a sender.
                _get_or_create_install_id(config)
                event = build_run_event(
                    scenario_count=scenario_count,
                    total_turns=total_turns,
                    pass_count=pass_count,
                    fail_count=fail_count,
                    duration_s=duration_s,
                    install_id=config.get("install_id", "test"),
                )
                sender(event)
            return

        install_id = _get_or_create_install_id(config)
        _save_config(config)

        event = build_run_event(
            scenario_count=scenario_count,
            total_turns=total_turns,
            pass_count=pass_count,
            fail_count=fail_count,
            duration_s=duration_s,
            install_id=install_id,
        )

        active_sender = sender if sender is not None else _default_sender
        active_sender(event)

    except Exception:
        pass
