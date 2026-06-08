"""korrel.exporters: optional export targets for Korrel scenarios.

Each submodule implements an exporter for a specific RL training framework.
None of them import the target framework at module load; imports are deferred
to function call time so that ``import korrel`` works without the target
framework installed.

Available submodules:

- ``korrel.exporters.verifiers``: export a Scenario as a verifiers
  Environment (requires the ``verifiers`` optional extra, Python <3.14).
- ``korrel.exporters.openenv``: export a Scenario as an OpenEnv environment
  package (requires the ``openenv`` optional extra; openenv-core>=0.3.0).
"""
