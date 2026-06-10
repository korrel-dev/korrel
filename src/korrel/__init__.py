"""korrel: an OSS Python SDK for agent simulation.

Define a multi-turn agent test once as a code-first Scenario, run it against a
bring-your-own agent adapter, and score the transcript with a Rubric. Keys are
read from the environment at call time and stored nowhere.
"""

from .adapter import AgentAdapter, adapter_from_provider
from .persona import Persona
from .rubric import Rubric, RewardFn, make_judge
from .runtime import (
    FailureCluster,
    RunResult,
    Transcript,
    Turn,
    run_scenario,
)
from .scenario import Scenario
from .tools import MockTool
from .types import Message, ToolCall, ToolFunction, ToolSchema

__all__ = [
    "Scenario",
    "Persona",
    "MockTool",
    "Rubric",
    "RewardFn",
    "make_judge",
    "run_scenario",
    "RunResult",
    "Transcript",
    "Turn",
    "FailureCluster",
    "AgentAdapter",
    "adapter_from_provider",
    "Message",
    "ToolCall",
    "ToolFunction",
    "ToolSchema",
]

__version__ = "0.1.1"
