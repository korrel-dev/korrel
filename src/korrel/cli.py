"""korrel CLI entry point.

Subcommands:
  run <scenario.py>  Run a scenario module and print the result.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from typing import Optional

from .adapter import AgentAdapter
from .runtime import RunResult, run_scenario
from .scenario import Scenario


# ---------------------------------------------------------------------------
# Shared transcript writer (used by CLI and pytest plugin)
# ---------------------------------------------------------------------------


def write_transcript_for(result: RunResult, scenario_id: str, out_dir: Path) -> Path:
    """Write ``result.transcript`` as JSON and return the file path.

    The file is ``<out_dir>/<scenario_id>.transcript.json``. ``out_dir`` is
    created if it does not exist.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scenario_id}.transcript.json"
    path.write_text(result.transcript.model_dump_json(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Module loader (shared with pytest plugin)
# ---------------------------------------------------------------------------


def load_module_from_path(file_path: Path) -> types.ModuleType:
    """Import a Python source file by path and return the module object."""
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def load_scenario_and_adapter(
    module: types.ModuleType,
    scenario_attr: str,
    adapter_attr: str,
) -> tuple[Scenario, AgentAdapter]:
    """Extract a Scenario and AgentAdapter from a loaded module."""
    scenario = getattr(module, scenario_attr, None)
    if scenario is None:
        raise AttributeError(
            f"Module {module.__name__!r} has no attribute {scenario_attr!r}. "
            "Define a module-level `scenario = Scenario(...)` in the file."
        )
    if not isinstance(scenario, Scenario):
        raise TypeError(
            f"{scenario_attr!r} in {module.__name__!r} is not a Scenario "
            f"(got {type(scenario).__name__})."
        )

    adapter = getattr(module, adapter_attr, None)
    if adapter is None:
        raise AttributeError(
            f"Module {module.__name__!r} has no attribute {adapter_attr!r}. "
            "Define a module-level `adapter = adapter_from_provider(...)` or "
            "a custom callable in the file."
        )
    if not callable(adapter):
        raise TypeError(
            f"{adapter_attr!r} in {module.__name__!r} is not callable "
            f"(got {type(adapter).__name__})."
        )

    return scenario, adapter  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# `korrel run` command
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"error: file not found: {file_path}", file=sys.stderr)
        return 1

    try:
        module = load_module_from_path(file_path)
    except Exception as exc:
        print(f"error: could not load {file_path}: {exc}", file=sys.stderr)
        return 1

    try:
        scenario, adapter = load_scenario_and_adapter(
            module,
            scenario_attr=args.scenario_attr,
            adapter_attr=args.adapter_attr,
        )
    except (AttributeError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    seed: Optional[int] = args.seed
    result = run_scenario(scenario, adapter, seed=seed)

    out_dir = Path(args.out) if args.out else Path(".korrel")
    transcript_path = write_transcript_for(result, scenario.id, out_dir)

    # Print result summary.
    status = "pass" if result.passed else "fail"
    print(f"scenario : {scenario.id}")
    print(f"score    : {result.score:.4f}")
    print(f"status   : {status}")
    if result.failed_functions:
        print(f"failed   : {', '.join(result.failed_functions)}")
    if result.clusters:
        cluster_strs = [f"{c.function}({c.signature})" for c in result.clusters]
        print(f"clusters : {', '.join(cluster_strs)}")
    print(f"transcript: {transcript_path}")

    return 0 if result.passed else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="korrel",
        description="korrel: agent simulation SDK.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    run_parser = sub.add_parser("run", help="Run a scenario module.")
    run_parser.add_argument("file", metavar="SCENARIO_PY", help="Path to the scenario module.")
    run_parser.add_argument(
        "--out",
        metavar="DIR",
        default=None,
        help="Directory for transcript output (default: .korrel/).",
    )
    run_parser.add_argument(
        "--seed",
        metavar="N",
        type=int,
        default=None,
        help="Override the scenario seed.",
    )
    run_parser.add_argument(
        "--scenario-attr",
        metavar="NAME",
        default="scenario",
        help="Module attribute name for the Scenario (default: scenario).",
    )
    run_parser.add_argument(
        "--adapter-attr",
        metavar="NAME",
        default="adapter",
        help="Module attribute name for the AgentAdapter (default: adapter).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        sys.exit(_cmd_run(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
