"""korrel CLI entry point.

Subcommands:
  run <scenario.py>  Run a scenario module and print the result.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import types
from pathlib import Path
from typing import Optional

from .adapter import AgentAdapter
from .runtime import RunResult, Transcript, run_scenario
from .scenario import Scenario


# ---------------------------------------------------------------------------
# Shared transcript writer (used by CLI and pytest plugin)
# ---------------------------------------------------------------------------


def _safe_filename_stem(scenario_id: str) -> str:
    """Reduce a scenario id to a single safe path component.

    A scenario id is author-controlled. Used raw as a filename it could carry
    path separators or an absolute path and escape ``out_dir`` (security review
    K9). ``Path(...).name`` strips any directory part on both POSIX and Windows;
    the fallback covers ids that reduce to nothing usable (``""``, ``.``, ``..``).
    """
    stem = Path(scenario_id).name
    if stem in ("", ".", ".."):
        return "scenario"
    return stem


def write_transcript(transcript: Transcript, scenario_id: str, out_dir: Path) -> Path:
    """Write a transcript as JSON and return the file path.

    The file is ``<out_dir>/<scenario_id>.transcript.json``. ``scenario_id`` is
    reduced to a single safe path component first. ``out_dir`` is created if it
    does not exist. Also used for the partial transcript carried by
    ``ToolExecutionError``, which has no ``RunResult``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_safe_filename_stem(scenario_id)}.transcript.json"
    path.write_text(transcript.model_dump_json(), encoding="utf-8")
    return path


def write_transcript_for(result: RunResult, scenario_id: str, out_dir: Path) -> Path:
    """Write ``result.transcript`` as JSON and return the file path."""
    return write_transcript(result.transcript, scenario_id, out_dir)


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


def load_scenario_only(
    module: types.ModuleType,
    scenario_attr: str,
) -> Scenario:
    """Extract a Scenario from a loaded module (no adapter required)."""
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
    return scenario


# ---------------------------------------------------------------------------
# `korrel export` command
# ---------------------------------------------------------------------------

_SUPPORTED_TARGETS = ("verifiers", "openenv")


def _cmd_export(args: argparse.Namespace) -> int:
    target: str = args.to
    if target not in _SUPPORTED_TARGETS:
        print(
            f"error: unsupported export target {target!r}. "
            f"Supported: {', '.join(_SUPPORTED_TARGETS)}.",
            file=sys.stderr,
        )
        return 1

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"error: file not found: {file_path}", file=sys.stderr)
        return 1

    try:
        module = load_module_from_path(file_path)
    except Exception as exc:
        print(f"error: could not load {file_path}: {exc}", file=sys.stderr)
        return 1

    if not args.scenario_attr.isidentifier():
        print(
            f"error: --scenario-attr must be a valid Python identifier, "
            f"got {args.scenario_attr!r}.",
            file=sys.stderr,
        )
        return 1

    try:
        scenario = load_scenario_only(module, scenario_attr=args.scenario_attr)
    except (AttributeError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # scenario.id is author-controlled; reduce it to a single safe path component
    # before using it as the default output directory name (security review K9:
    # an id carrying separators or an absolute path could otherwise escape
    # .korrel/export/). When --out is given the operator owns that path.
    out_dir = (
        Path(args.out)
        if args.out
        else Path(".korrel") / "export" / _safe_filename_stem(scenario.id)
    )

    if target == "verifiers":
        # Import the artifact emitter (does not import verifiers itself).
        from .exporters.verifiers import write_verifiers_env

        produced = write_verifiers_env(
            scenario,
            out_dir,
            scenario_source_path=file_path,
            scenario_attr=args.scenario_attr,
        )
        pyproject = produced / "pyproject.toml"
        env_module = _safe_filename_stem(scenario.id).replace("-", "_").replace(" ", "_")
        env_module_py = produced / f"{env_module}.py"
        scenario_py = produced / "_scenario.py"
        print(f"{'target':<12}: verifiers")
        print(f"{'scenario':<12}: {scenario.id}")
        print(f"{'out_dir':<12}: {produced}")
        print(f"{'pyproject':<12}: {pyproject}")
        print(f"{'env_module':<12}: {env_module_py}")
        print(f"{'scenario_src':<12}: {scenario_py}")

    elif target == "openenv":
        # Import the artifact emitter (does not import openenv-core itself).
        from .exporters.openenv import write_openenv_env

        produced = write_openenv_env(
            scenario,
            out_dir,
            scenario_source_path=file_path,
            scenario_attr=args.scenario_attr,
        )
        env_name = _safe_filename_stem(scenario.id).replace("_", "-")
        env_module = env_name.replace("-", "_")
        pyproject = produced / "pyproject.toml"
        models_py = produced / "models.py"
        env_py = produced / "server" / f"{env_module}_environment.py"
        scenario_py = produced / "_scenario.py"
        print(f"{'target':<12}: openenv")
        print(f"{'scenario':<12}: {scenario.id}")
        print(f"{'out_dir':<12}: {produced}")
        print(f"{'pyproject':<12}: {pyproject}")
        print(f"{'models':<12}: {models_py}")
        print(f"{'environment':<12}: {env_py}")
        print(f"{'scenario_src':<12}: {scenario_py}")

    return 0


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
    t_start = time.monotonic()
    result = run_scenario(scenario, adapter, seed=seed)
    duration_s = time.monotonic() - t_start

    out_dir = Path(args.out) if args.out else Path(".korrel")
    transcript_path = write_transcript_for(result, scenario.id, out_dir)

    # Print result summary. The "model calls" label is wider than the prior
    # 10-char pad, so the pad is widened to 12 across every line to stay aligned.
    status = "pass" if result.passed else "fail"
    print(f"{'scenario':<12}: {scenario.id}")
    print(f"{'score':<12}: {result.score:.4f}")
    print(f"{'status':<12}: {status}")
    print(f"{'model calls':<12}: {result.model_calls}")
    if result.failed_functions:
        print(f"{'failed':<12}: {', '.join(result.failed_functions)}")
    if result.clusters:
        cluster_strs = [f"{c.function}({c.signature})" for c in result.clusters]
        print(f"{'clusters':<12}: {', '.join(cluster_strs)}")
    print(f"{'transcript':<12}: {transcript_path}")

    # Emit telemetry. Best-effort: errors are swallowed inside emit_run.
    from .telemetry import emit_run

    total_turns = len(result.transcript.turns)
    emit_run(
        scenario_count=1,
        total_turns=total_turns,
        pass_count=1 if result.passed else 0,
        fail_count=0 if result.passed else 1,
        duration_s=duration_s,
    )

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

    export_parser = sub.add_parser(
        "export",
        help="Export a scenario as an RL training environment.",
    )
    export_parser.add_argument(
        "file",
        metavar="SCENARIO_PY",
        help="Path to the scenario module.",
    )
    export_parser.add_argument(
        "--to",
        metavar="TARGET",
        required=True,
        help=f"Export target. Supported: {', '.join(_SUPPORTED_TARGETS)}.",
    )
    export_parser.add_argument(
        "--out",
        metavar="DIR",
        default=None,
        help="Output directory for the generated package (default: .korrel/export/<scenario-id>/).",
    )
    export_parser.add_argument(
        "--scenario-attr",
        metavar="NAME",
        default="scenario",
        help="Module attribute name for the Scenario (default: scenario).",
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
    elif args.command == "export":
        sys.exit(_cmd_export(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
