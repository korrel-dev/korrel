# benchmarks/Makefile
#
# Targets:
#   make benchmark   -- reproducible, no keys, no LLM calls.
#                       Reads frozen transcripts/retail/*.run.json, scores each
#                       with compute_labels.py (already done if labels.jsonl
#                       exists), runs the fidelity harness, writes
#                       results/results.json and results/summary.md.
#
#   make labels      -- recompute transcripts/labels.jsonl from frozen runs.
#                       Deterministic, offline, no keys.
#
#   make transcripts -- REQUIRES ANTHROPIC_API_KEY. Calls tau2's runner to
#                       generate trajectories for the 40-task retail slice with
#                       two agent models (strong + weak). Writes frozen run
#                       files to transcripts/retail/. Run once; commit output.
#
# tau2 data provisioning:
#   When tau2 is installed from the git SHA pin, its data/ directory lives in
#   the uv git cache, not in site-packages. TAU2_DATA_DIR must point to it.
#   The `find-tau2-data` target resolves the path by inspecting the installed
#   tau2 package source location and walking up to the git checkout root.
#
# Prerequisites: uv must be on PATH (or adjust UV below).

UV ?= uv
PYTHON = $(UV) run --project . python

.PHONY: benchmark labels transcripts sync find-tau2-data

# --- primary target: offline, no keys ------------------------------------------

benchmark: sync
	@echo "--- locating tau2 data directory ---"
	$(eval TAU2_DATA := $(shell $(PYTHON) -c "import tau2.utils.utils as u; import pathlib; p = pathlib.Path(u.__file__).parent; cand = p.parents[3] / 'data'; print(cand if cand.exists() else '')"))
	@if [ -z "$(TAU2_DATA)" ]; then \
		echo "ERROR: could not auto-locate tau2 data/. Set TAU2_DATA_DIR manually."; \
		exit 1; \
	fi
	@echo "  TAU2_DATA_DIR=$(TAU2_DATA)"
	@echo "--- computing labels from frozen transcripts ---"
	TAU2_DATA_DIR="$(TAU2_DATA)" $(PYTHON) regenerate/compute_labels.py
	@echo "--- running fidelity harness ---"
	TAU2_DATA_DIR="$(TAU2_DATA)" $(PYTHON) fidelity/run_fidelity.py

# --- offline label computation -------------------------------------------------

labels: sync
	$(eval TAU2_DATA := $(shell $(PYTHON) -c "import tau2.utils.utils as u; import pathlib; p = pathlib.Path(u.__file__).parent; cand = p.parents[3] / 'data'; print(cand if cand.exists() else '')"))
	@if [ -z "$(TAU2_DATA)" ]; then \
		echo "ERROR: could not auto-locate tau2 data/. Set TAU2_DATA_DIR manually."; \
		exit 1; \
	fi
	TAU2_DATA_DIR="$(TAU2_DATA)" $(PYTHON) regenerate/compute_labels.py

# --- keyed transcript generation (founder runs once) --------------------------

transcripts: sync
	$(eval TAU2_DATA := $(shell $(PYTHON) -c "import tau2.utils.utils as u; import pathlib; p = pathlib.Path(u.__file__).parent; cand = p.parents[3] / 'data'; print(cand if cand.exists() else '')"))
	@if [ -z "$(TAU2_DATA)" ]; then \
		echo "ERROR: could not auto-locate tau2 data/. Set TAU2_DATA_DIR manually."; \
		exit 1; \
	fi
	@echo "--- generating transcripts (requires ANTHROPIC_API_KEY) ---"
	TAU2_DATA_DIR="$(TAU2_DATA)" $(PYTHON) regenerate/gen_transcripts.py

# --- print the resolved tau2 data path (diagnostic) --------------------------

find-tau2-data: sync
	$(eval TAU2_DATA := $(shell $(PYTHON) -c "import tau2.utils.utils as u; import pathlib; p = pathlib.Path(u.__file__).parent; cand = p.parents[3] / 'data'; print(cand if cand.exists() else 'NOT FOUND')"))
	@echo "TAU2_DATA_DIR would be set to: $(TAU2_DATA)"

# --- environment setup --------------------------------------------------------

sync:
	$(UV) sync --project .
