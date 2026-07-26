#!/bin/bash
# Generate an Amaze dataset directory that AmazeDataset can load.
# Each (task, shape/scale) gets its own directory, because AmazeDataset loads one fixed-name maze_dataset_{train,test}.parquet per directory.
#
# Usage:
#   scripts/generate_amaze_datasets.sh [maze_square_8x8|maze_hex_8x8|queens_7x7|all]
#
# Override sized to generate via env variables:
# MAZE_TRAIN=20 MAZE_TEST=6 QUEEN_TRAIN=20 QUEEN_TEST_PER_SCALE=2

set -euo pipefail

TRM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAZE_GEN="${TRM_ROOT}/third_party/amaze/mazes-generator"
QUEEN_GEN="${TRM_ROOT}/third_party/amaze/queen-generator"
PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-${TRM_ROOT}/data/amaze}"

WHICH="${1:-all}"
NPROC="${NPROC:-$(nproc)}"           # parallel maze workers (hexagon is ~1s/sample)
MAZE_TRAIN="${MAZE_TRAIN:-30000}";   MAZE_TEST="${MAZE_TEST:-3000}"
QUEEN_TRAIN="${QUEEN_TRAIN:-30000}"; QUEEN_TEST_PER_SCALE="${QUEEN_TEST_PER_SCALE:-430}"  # ~3010 over n=4..10

# gen_maze <name> <shape> <width> <height> <algo1,algo2,...>
gen_maze() {
  local name="$1" shape="$2" w="$3" h="$4" algos="$5"
  [[ -f "${OUT_ROOT}/${name}/maze_dataset_train.parquet" ]] && { echo ">> ${name} already exists, skipping"; return; }
  echo ">> ${name}: generating $((MAZE_TRAIN + MAZE_TEST)) mazes on ${NPROC} workers"
  if [[ ! -d "${MAZE_GEN}/node_modules" ]]; then
    echo "   installing node deps (needs internet — run on a login node, not an offline compute node)…"
    ( cd "${MAZE_GEN}" && npm install )
  fi

  local work; work="$(mktemp -d)"

  # We use maze generator from AMAZE paper. The generator takes a config file listing each maze to generate. It is built here, splitted to n_proc shards.
  MAZE_N=$((MAZE_TRAIN + MAZE_TEST)) SHAPE="${shape}" W="${w}" H="${h}" ALGOS="${algos}" \
  NAME="${name}" NPROC="${NPROC}" "${PYTHON}" - "${work}" <<'PY'
import json, os, sys
work = sys.argv[1]; n = int(os.environ["MAZE_N"]); nproc = int(os.environ["NPROC"])
algos = os.environ["ALGOS"].split(",")
shards = [[] for _ in range(nproc)]
for i in range(n):
    shards[i % nproc].append({"shape": os.environ["SHAPE"], "width": int(os.environ["W"]),
        "height": int(os.environ["H"]), "algorithm": algos[i % len(algos)], "exitConfig": "hardest",
        "seed": 1_000_000 + i, "filename": f"{os.environ['NAME']}_{i:07d}.png"})
for k, m in enumerate(shards):
    if m:
        json.dump({"mazes": m}, open(os.path.join(work, f"shard_{k:03d}.json"), "w"))
PY

  # Run all shards in parallel; they share ./generated_* (filenames are disjoint).
  # Worker output goes to per-shard logs, so print a heartbeat to stdout — otherwise
  # the terminal looks frozen for the whole run.
  ( cd "${work}"
    for f in shard_*.json; do
      node "${MAZE_GEN}/batch-maze-generator.js" config "${f}" >"${f}.log" 2>&1 &
    done
    local total=$((MAZE_TRAIN + MAZE_TEST))
    while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
      sleep 15
      echo "   … $(ls generated_mazes_no_markers 2>/dev/null | wc -l)/${total} mazes rendered"
    done
    wait )

  local ratio; ratio="$("${PYTHON}" -c "print(${MAZE_TRAIN}/(${MAZE_TRAIN}+${MAZE_TEST}))")"
  "${PYTHON}" "${MAZE_GEN}/process_maze_into_parquet.py" \
    --maze-dir "${work}/generated_mazes" \
    --no-markers-dir "${work}/generated_mazes_no_markers" \
    --solution-dir "${work}/generated_solutions" \
    --metadata-dir "${work}/generated_metadata" \
    --output "${OUT_ROOT}/${name}/maze_dataset.parquet" \
    --train-ratio "${ratio}" --seed 42
  rm -rf "${work}"
}

# gen_queens: train = n=7, test = multi-scale n=4..10 (AMAZE-paper style).
gen_queens() {
  [[ -f "${OUT_ROOT}/queens_7x7/maze_dataset_train.parquet" ]] && { echo ">> queens_7x7 already exists, skipping"; return; }
  echo ">> queens_7x7: generating ${QUEEN_TRAIN} train (n=7) + multi-scale test (n=4..10)"
  local tr te tep; tr="$(mktemp -d)"; te="$(mktemp -d)"; tep="$(mktemp -d)"

  "${PYTHON}" "${QUEEN_GEN}/generate_queens_puzzle.py" \
    --n 7 --count "${QUEEN_TRAIN}" --outdir "${tr}" --seed 5000000 \
    --cell-size 64 --queen-radius 16 --image-format png
  for n in 4 5 6 7 8 9 10; do
    "${PYTHON}" "${QUEEN_GEN}/generate_queens_puzzle.py" \
      --n "${n}" --count "${QUEEN_TEST_PER_SCALE}" --outdir "${te}" --seed $((6000000 + n)) \
      --cell-size 64 --queen-radius 16 --image-format png
  done

  # Convert each pool with --test-ratio 0 (everything lands in *_train.parquet),
  # then place them as the train / test splits of the final dataset dir.
  "${PYTHON}" "${QUEEN_GEN}/convert_queen_to_parquet.py" \
    --queen-outdir "${tr}" --dataset-outdir "${OUT_ROOT}/queens_7x7" --test-ratio 0 --seed 42
  "${PYTHON}" "${QUEEN_GEN}/convert_queen_to_parquet.py" \
    --queen-outdir "${te}" --dataset-outdir "${tep}" --test-ratio 0 --seed 42
  mv "${tep}/maze_dataset_train.parquet" "${OUT_ROOT}/queens_7x7/maze_dataset_test.parquet"
  rm -rf "${tr}" "${te}" "${tep}"
}

SQUARE_ALGOS="recursiveBacktrack,simplifiedPrims,truePrims,wilson,aldousBroder,huntAndKill,kruskal,binaryTree,sidewinder,ellers"
HEX_ALGOS="recursiveBacktrack,simplifiedPrims,truePrims,wilson,aldousBroder,huntAndKill"

case "${WHICH}" in
  maze_square_8x8) gen_maze maze_square_8x8 square  8 8 "${SQUARE_ALGOS}" ;;
  maze_hex_8x8)    gen_maze maze_hex_8x8    hexagon 8 8 "${HEX_ALGOS}" ;;
  queens_7x7)      gen_queens ;;
  all)
    gen_maze maze_square_8x8 square  8 8 "${SQUARE_ALGOS}"
    gen_maze maze_hex_8x8    hexagon 8 8 "${HEX_ALGOS}"
    gen_queens ;;
  *) echo "Unknown dataset: ${WHICH} (use maze_square_8x8|maze_hex_8x8|queens_7x7|all)" >&2; exit 1 ;;
esac
echo "Done -> ${OUT_ROOT}"
