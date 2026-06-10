# benchmarks/Makefile
#
# Targets:
#   make benchmark   -- reproducible, no keys, no LLM calls.
#                       Runs selftest, recomputes labels, reads frozen
#                       transcripts/retail/*.run.json, scores each through the
#                       fidelity harness, writes results/results.json.
#
#   make selftest    -- offline, no keys. Runs the four-way fidelity assertion
#                       against the committed synthetic fixtures only.
#
#   make labels      -- recompute transcripts/labels.jsonl from frozen runs.
#                       Deterministic, offline, no keys.
#
#   make transcripts -- REQUIRES ANTHROPIC_API_KEY. Calls tau2's runner to
#                       generate trajectories for the 40-task retail slice with
#                       two agent models (strong + weak). Run once; commit output.
#
# TAU2_DATA_DIR is resolved automatically by each Python script via
# benchmarks/fidelity/_tau2_data.py. No env var needed unless auto-discovery
# fails (e.g. uv cache in a non-standard location).
#
# Prerequisites: uv must be on PATH (or set UV= on the command line).
#
# Windows note: run `git config --global core.longpaths true` before `make sync`.
# tau2's leaderboard/ subdirectory has filenames exceeding Windows MAX_PATH.

UV ?= uv
PYTHON = $(UV) run --project . python

.PHONY: benchmark labels transcripts sync selftest

# --- selftest: offline, no keys, synthetic fixture only -----------------------

selftest: sync
	@echo "--- running selftest (synthetic fixture, no frozen transcripts needed) ---"
	$(PYTHON) fidelity/run_fidelity.py --selftest

# --- primary target: offline, no keys ------------------------------------------
# Runs selftest as a pre-step before the frozen-transcript benchmark.

benchmark: selftest labels
	@echo "--- running fidelity harness ---"
	$(PYTHON) fidelity/run_fidelity.py

# --- offline label computation -------------------------------------------------

labels: sync
	@echo "--- computing labels from frozen transcripts ---"
	$(PYTHON) regenerate/compute_labels.py

# --- keyed transcript generation (founder runs once) --------------------------

transcripts: sync
	@echo "--- generating transcripts (requires ANTHROPIC_API_KEY) ---"
	$(PYTHON) regenerate/gen_transcripts.py

# --- environment setup --------------------------------------------------------

sync:
	$(UV) sync --project .
