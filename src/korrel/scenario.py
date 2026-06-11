"""Scenario: a code-first definition of a multi-turn agent test."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .persona import Persona
from .rubric import Rubric
from .tools import MockTool


class Scenario(BaseModel):
    """A single agent-simulation scenario.

    ``system`` is the system prompt for the agent under test. ``persona`` drives
    the simulated user; it is required even for single-turn scenarios, and at
    ``max_turns=1`` it is constructed but never invoked (no model call, no
    key). ``opening_message`` is the first user turn.
    ``tools`` are the mock tools available to the agent. ``info`` is the ground
    truth handed to the rubric (for example expected answers). ``rubric`` scores
    the transcript after the run. ``seed`` pins scenario setup and any sampling
    Korrel controls; it does not make provider calls bit-reproducible.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    system: str
    persona: Persona
    opening_message: str
    tools: list[MockTool] = Field(default_factory=list)
    max_turns: int = Field(default=1, ge=1)
    max_tool_rounds: int = Field(default=8, ge=1)
    seed: int = 0
    info: Optional[dict[str, Any]] = None
    rubric: Optional[Rubric] = None
