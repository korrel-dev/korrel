# benchmarks/Makefile
#
# Targets:
#   make benchmark   -- reproducible, no keys, no LLM calls.
#                       Reads frozen transcripts/retail/*.run.json, scores each,
#                       runs the fidelity harness, writes results/results.json.
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

.PHONY: benchmark labels transcripts sync

# --- primary target: offline, no keys ------------------------------------------

benchmark: sync labels
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
