#!/usr/bin/env bash
set -euo pipefail

# 关卡尺寸变化（N 变化），但格子和皇后像素大小固定
N_LIST=(7)
COUNT_PER_N=6400
CELL_SIZE=64
QUEEN_RADIUS=16
BASE_OUTDIR="/media/raid/workspace/zhaoyanpeng/code/queen/train_7_6400"
SEED=1
MAX_ATTEMPTS=300

for n in "${N_LIST[@]}"; do
  python generate_queens_puzzle.py \
    --n "${n}" \
    --count "${COUNT_PER_N}" \
    --outdir "${BASE_OUTDIR}" \
    --seed "${SEED}" \
    --max-attempts "${MAX_ATTEMPTS}" \
    --cell-size "${CELL_SIZE}" \
    --queen-radius "${QUEEN_RADIUS}"
done
