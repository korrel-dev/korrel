"""The emitted _load_scenario must use a unique in-memory module name.

Two exported environments imported in one process previously both registered
the bundled scenario source as sys.modules["_scenario"] and collided. The
emitters now derive the spec_from_file_location name from the sanitized
env_module; the on-disk filename stays _scenario.py.
"""

import ast
import importlib.util
import sys
import textwrap
from pathlib import Path

from korrel import Persona, Scenario
from korrel.exporters.openenv import write_openenv_env
from korrel.exporters.verifiers import write_verifiers_env


def _scenario(scenario_id: str) -> Scenario:
    return Scenario(
        id=scenario_id,
        system="You are a support agent.",
        persona=Persona(goal="Get help.", behavior="Polite."),
        opening_message="Hello.",
        max_turns=2,
    )


def _scenario_source(scenario_id: str) -> str:
    return textwrap.dedent(f"""\
        from korrel import Persona, Scenario

        scenario = Scenario(
            id={scenario_id!r},
            system="You are a support agent.",
            persona=Persona(goal="Get help.", behavior="Polite."),
            opening_message="Hello.",
            max_turns=2,
        )
    """)


def test_verifiers_emitted_module_names_are_distinct(tmp_path: Path) -> None:
    out_a = write_verifiers_env(_scenario("ns-alpha"), tmp_path / "a")
    out_b = write_verifiers_env(_scenario("ns-beta"), tmp_path / "b")

    src_a = (out_a / "ns_alpha.py").read_text(encoding="utf-8")
    src_b = (out_b / "ns_beta.py").read_text(encoding="utf-8")

    assert "'_korrel_scenario_ns_alpha'" in src_a
    assert "'_korrel_scenario_ns_beta'" in src_b
    # The literal "_scenario" module name is gone from the loader call.
    assert 'spec_from_file_location("_scenario"' not in src_a
    assert "spec_from_file_location('_scenario'" not in src_a
    ast.parse(src_a)
    ast.parse(src_b)


def test_openenv_emitted_module_names_are_distinct(tmp_path: Path) -> None:
    out_a = write_openenv_env(_scenario("ns-alpha"), tmp_path / "a")
    out_b = write_openenv_env(_scenario("ns-beta"), tmp_path / "b")

    src_a = (out_a / "server" / "ns_alpha_environment.py").read_text(encoding="utf-8")
    src_b = (out_b / "server" / "ns_beta_environment.py").read_text(encoding="utf-8")

    assert "'_korrel_scenario_ns_alpha'" in src_a
    assert "'_korrel_scenario_ns_beta'" in src_b
    assert 'spec_from_file_location("_scenario"' not in src_a
    assert "spec_from_file_location('_scenario'" not in src_a
    ast.parse(src_a)
    ast.parse(src_b)


def test_verifiers_non_identifier_id_yields_identifier_module_name(
    tmp_path: Path,
) -> None:
    # The verifiers env_module path only replaces separators, so an id with
    # other punctuation is not identifier-safe there. The in-memory module
    # name must still reduce to a valid identifier.
    out = write_verifiers_env(_scenario("weird.id!x"), tmp_path / "env")
    src = (out / "weird.id!x.py").read_text(encoding="utf-8")
    assert "'_korrel_scenario_weird_id_x'" in src
    ast.parse(src)


def test_two_emitted_verifiers_packages_load_without_collision(
    tmp_path: Path,
) -> None:
    # Functional check, offline: load both emitted env modules in this
    # process and call their _load_scenario(). Each must see its own bundled
    # source, registered under its own sys.modules key. (_load_scenario does
    # not import verifiers; only load_environment does.)
    src_a = tmp_path / "alpha_src.py"
    src_a.write_text(_scenario_source("ns-alpha"), encoding="utf-8")
    src_b = tmp_path / "beta_src.py"
    src_b.write_text(_scenario_source("ns-beta"), encoding="utf-8")

    out_a = write_verifiers_env(
        _scenario("ns-alpha"), tmp_path / "a", scenario_source_path=src_a
    )
    out_b = write_verifiers_env(
        _scenario("ns-beta"), tmp_path / "b", scenario_source_path=src_b
    )

    def _load(env_path: Path, name: str):
        spec = importlib.util.spec_from_file_location(name, env_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    env_a = _load(out_a / "ns_alpha.py", "test_ns_env_alpha")
    env_b = _load(out_b / "ns_beta.py", "test_ns_env_beta")

    scenario_a = env_a._load_scenario()
    scenario_b = env_b._load_scenario()

    assert scenario_a.id == "ns-alpha"
    assert scenario_b.id == "ns-beta"
    assert "_korrel_scenario_ns_alpha" in sys.modules
    assert "_korrel_scenario_ns_beta" in sys.modules
    assert (
        sys.modules["_korrel_scenario_ns_alpha"]
        is not sys.modules["_korrel_scenario_ns_beta"]
    )
