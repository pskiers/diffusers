# Shared definitions for the PaTh generalisation experiments (sourced, not run).
#
# Every subset has 29,988 training mazes, the size of the original full training set
# (4 shapes x 7 sizes x 1071), so "less data" is not mixed into the comparison.
#
# gen_amaze.py MERGES -o overrides into configs/data/amaze_generation.yaml, so a
# size left out of maze.train.samples would silently keep its original 1071 rows.
# Every size is therefore listed, with 0 for the ones a subset leaves out
# (gen_amaze.py skips sizes with a count of 0).

GEN_SUBSETS=(sizes_5_8_13 circle_only hexagon_only square_8)
GEN_TRAIN_ROWS=29988

# Where the subsets are written: <GEN_ROOT>/<subset>/maze/maze_dataset_train_144.parquet
GEN_ROOT="${GEN_ROOT:-${PROJECT_ROOT}/data/amaze_generalisation}"

subset_shapes() {
  case "$1" in
    sizes_5_8_13) echo "[square,hexagon,triangle,circle]" ;;
    circle_only)  echo "[circle]" ;;
    hexagon_only) echo "[hexagon]" ;;
    square_8)     echo "[square]" ;;
    *) echo "unknown subset '$1' (use: ${GEN_SUBSETS[*]})" >&2; return 1 ;;
  esac
}

subset_samples() {   # rows PER SHAPE for each size
  case "$1" in
    sizes_5_8_13) echo "{5: 2499, 7: 0, 8: 2499, 9: 0, 11: 0, 13: 2499, 16: 0}" ;;     # 4 shapes x 3 sizes x 2499
    circle_only)  echo "{5: 4284, 7: 4284, 8: 4284, 9: 4284, 11: 4284, 13: 4284, 16: 4284}" ;;  # 7 x 4284
    hexagon_only) echo "{5: 4284, 7: 4284, 8: 4284, 9: 4284, 11: 4284, 13: 4284, 16: 4284}" ;;  # 7 x 4284
    square_8)     echo "{5: 0, 7: 0, 8: 29988, 9: 0, 11: 0, 13: 0, 16: 0}" ;;
    *) echo "unknown subset '$1' (use: ${GEN_SUBSETS[*]})" >&2; return 1 ;;
  esac
}

subset_train_parquet() {
  echo "${GEN_ROOT}/$1/maze/maze_dataset_train_144.parquet"
}

# The test cells ("<shape>_<size>") a subset trains on -> +seen_cells for the metrics
# script (wandb gen/seen/* vs gen/unseen/*). Must match subset_shapes/subset_samples.
subset_seen_cells() {
  local shapes sizes cells=() g s
  case "$1" in
    sizes_5_8_13) shapes="square hexagon triangle circle"; sizes="5 8 13" ;;
    circle_only)  shapes="circle";  sizes="5 7 8 9 11 13 16" ;;
    hexagon_only) shapes="hexagon"; sizes="5 7 8 9 11 13 16" ;;
    square_8)     shapes="square";  sizes="8" ;;
    *) echo "unknown subset '$1' (use: ${GEN_SUBSETS[*]})" >&2; return 1 ;;
  esac
  for g in ${shapes}; do for s in ${sizes}; do cells+=("${g}_${s}"); done; done
  (IFS=,; echo "${cells[*]}")
}
