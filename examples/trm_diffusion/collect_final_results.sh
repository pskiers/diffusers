#!/bin/bash -l
# Thin wrapper: the real logic (including merging per-shape maze jsons) is in
# collect_final_results.py, because the scorer writes VLM maze results as one
# file per shape inside GEN_DIR, not one file per run.
set -uo pipefail
ROOT="${PROJECT_ROOT:-$PWD}"
cd "${ROOT}"
PROJECT_ROOT="${ROOT}" python3 collect_final_results.py
