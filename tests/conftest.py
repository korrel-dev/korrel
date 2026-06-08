"""Shared test fixtures: deterministic fakes, no network, no keys."""

import sys
from pathlib import Path
from typing import Optional

from korrel.types import Message

# Make the example scenario importable without installing it as a package.
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))


class ScriptedAdapter:
    """An agent adapter that replays a fixed queue of assistant messages."""

    def __init__(self, messages: list[Message]) -> None:
        self._queue = list(messages)
        self.calls = 0

    def __call__(self, messages: list[Message], tools: list[dict]) -> Message:
        self.calls += 1
        if self._queue:
            return self._queue.pop(0)
        return Message(role="assistant", content="")


class ScriptedPersona:
    """A user-simulator that replays a fixed queue of user messages."""

    model = "fake-persona"
    params = {"provider": "fake"}

    def __init__(self, messages: list[str]) -> None:
        self._queue = list(messages)
        self.calls = 0

    def next_message(self, messages: list[Message]) -> Optional[str]:
        self.calls += 1
        if self._queue:
            return self._queue.pop(0)
        return None
