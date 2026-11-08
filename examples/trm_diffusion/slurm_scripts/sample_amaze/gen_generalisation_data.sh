#!/bin/bash -l
#SBATCH --job-name=amaze_gen_subset
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#
# Generate ONE training subset for the generalisation experiments
# (definitions in generalisation_subsets.sh). Only the train stage runs: the
# models are scored on the existing full test sets in data/amaze/maze.
#
# Usage (from the trm_diffusion dir):
#   sbatch slurm_scripts/sample_amaze/gen_generalisation_data.sh <sizes_5_8_13|circle_only|hexagon_only|square_8>
# Env: GEN_ROOT (data/amaze_generalisation), OVERWRITE (false),
#      GEN_NPROC (generator workers; default: all CPUs of the job — set it when several
#      subsets share one job, see gen_generalisation_data_all.sh)
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PROJECT_ROOT}/slurm_scripts/sample_amaze/generalisation_subsets.sh"

SUBSET="${1:?usage: sbatch gen_generalisation_data.sh <subset: ${GEN_SUBSETS[*]}>}"
SHAPES="$(subset_shapes "${SUBSET}")"
SAMPLES="$(subset_samples "${SUBSET}")"
OUT_ROOT="${GEN_ROOT}/${SUBSET}"

source "${PROJECT_ROOT}/slurm_scripts/cluster_env.sh"
cluster_activate --node          # Python venv + nodejs (cluster_env.sh: NODE_MODULES)
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs
command -v node >/dev/null || { echo "ERROR: no 'node' on PATH (set NODE_MODULES for ${CLUSTER})" >&2; exit 1; }
echo ">> node $(node --version) on $(uname -m)"

if [[ ! -d third_party/ear_amaze/mazes-generator/node_modules ]]; then
  echo "ERROR: third_party/ear_amaze/mazes-generator/node_modules is missing." >&2
  echo "Install the generator's packages for THIS cluster's architecture, e.g. on a machine" >&2
  echo "with npm:  (cd third_party/ear_amaze/mazes-generator && npm ci --os=linux --cpu=<arm64|x64> --libc=glibc)" >&2
  echo "(Helios compute nodes are arm64, lem is x64.)" >&2
  exit 1
fi

NPROC_ARGS=()
[[ -n "${GEN_NPROC:-}" ]] && NPROC_ARGS=( -o "nproc=${GEN_NPROC}" )

echo ">> subset ${SUBSET}: shapes ${SHAPES}, rows per shape ${SAMPLES}, workers ${GEN_NPROC:-all} -> ${OUT_ROOT}"
python scripts/gen_amaze.py --task maze --stage train --output-root "${OUT_ROOT}" \
  -o "maze.shapes=${SHAPES}" \
  -o "maze.train.samples=${SAMPLES}" \
  -o "overwrite=${OVERWRITE:-false}" \
  ${NPROC_ARGS[@]+"${NPROC_ARGS[@]}"}

# Fail loudly if the subset does not have the intended size, or does not look
# like the original data. The start/end markers are text glyphs ('●', '⨯' at a
# fixed font size), so how big they come out depends on the fonts of the machine
# that renders them: rendered on macOS they cover ~1780 px instead of the
# original ~2700 px. Their area is the same for every shape and size, so the
# median over the first rows is compared with an original test set.
TRAIN_PARQUET="$(subset_train_parquet "${SUBSET}")"
ORIGINAL_PARQUET="${OUT_ROOT}/maze/maze_dataset_train_original.parquet"
REFERENCE_PARQUET="${REFERENCE_PARQUET:-${PROJECT_ROOT}/data/amaze/maze/square/n8_test.parquet}"
python - "${TRAIN_PARQUET}" "${GEN_TRAIN_ROWS}" "${ORIGINAL_PARQUET}" "${REFERENCE_PARQUET}" <<'EOF'
import base64, io, sys
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

path, want, original, reference = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
rows = pq.ParquetFile(path).metadata.num_rows
print(f">> {path}: {rows} rows (expected {want})")
if rows != want:
    sys.exit(f"ERROR: {path} has {rows} rows, expected {want}")

def marker_px(raw) -> int:
    if isinstance(raw, str):
        raw = base64.b64decode(raw.split(",", 1)[-1])
    a = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")).astype(int)
    return int(((a[..., 0] > 200) & (a[..., 1] < 80) & (a[..., 2] < 80)).sum())

def median_marker_px(parquet: str, n: int = 50) -> float:
    batch = next(pq.ParquetFile(parquet).iter_batches(batch_size=n, columns=["m_original_img"]))
    return float(np.median([marker_px(v) for v in batch.column(0).to_pylist()]))

generated, expected = median_marker_px(original), median_marker_px(reference)
ratio = generated / expected
print(f">> marker area: generated {generated:.0f} px, original data {expected:.0f} px (ratio {ratio:.2f})")
if not 0.9 <= ratio <= 1.1:
    sys.exit("ERROR: start/end markers are rendered at a different size than in the original data "
             "(font difference on this node?). Not training on this subset.")
EOF

echo "gen_generalisation_data ${SUBSET} done"
