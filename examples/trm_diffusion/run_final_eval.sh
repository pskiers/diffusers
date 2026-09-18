#!/bin/bash -l
#
# Re-run every AMAZE evaluation needed for the new metrics (Pass = max(0,cov-viol),
# Exact = set equality), with OOD disabled and VLMs scored at native resolution.
#
#   bash run_final_eval.sh            # DRY RUN - prints what it would submit
#   bash run_final_eval.sh submit     # actually sbatch everything
#   bash run_final_eval.sh submit A   # only stage A (see below)
#
# Stages, cheapest first:
#   A  RE-SCORE ONLY   Janus + Janus-Test. Generations already on disk -> no inference.
#   B  RE-SAMPLE       DiT + PT. Diffusion sampling is cheap; generations are not kept.
#   C  GENERATE        Bagel. 7B autoregressive, nothing usable on disk -> the expensive one.
#
# After everything finishes:  bash collect_final_results.sh
set -uo pipefail

MODE="${1:-dry}"
ONLY="${2:-ABC}"
ROOT="${PROJECT_ROOT:-$PWD}"
cd "${ROOT}"

SHAPES="square triangle hexagon circle"
JOBS=()

run() {
    local label="$1"; shift
    if [[ "${MODE}" == "submit" ]]; then
        local out; out=$("$@" 2>&1)
        echo "  [submitted] ${label}: ${out}"
        JOBS+=("${out##* }")
    else
        echo "  [dry-run]   ${label}"
        echo "              $*"
    fi
}

# ---------------------------------------------------------------- checkpoints
# Newest checkpoint_final.pt per run directory, verified 2026-09-17.
DIT_MAZE_CKPT="runs/dit_maze_final/checkpoint_final.pt"                 # Aug 22 23:08
PT_MAZE_THINKER="runs/pt_maze_final_thinker/checkpoint_final.pt"        # Aug 23 14:42
PT_MAZE_PAINTER="runs/pt_maze_final_painter/checkpoint_final.pt"        # Aug 22 18:04
DIT_QUEENS_CKPT="runs/queens_dit_baseline/checkpoint_final.pt"          # Aug 22 21:41
PT_QUEENS_THINKER="runs/pt_queens_final_thinker/checkpoint_final.pt"    # Aug 22 21:41
PT_QUEENS_PAINTER="${PT_QUEENS_PAINTER:-runs/queens_painter_v2/checkpoint_final.pt}"  # CONFIRM

# ---------------------------------------------------------------- generations
JANUS_MAZE_GEN="third_party/ear-amaze/inference_results/maze"           # 19700 png @384
JANUS_QUEENS_GEN="third_party/ear-amaze/inference_results/queens_std"   # 2475 png
JANUSTEST_QUEENS_GEN="third_party/ear-amaze/inference_results/queens_n4" # 2475 png
BAGEL_QUEENS_GEN="runs/bagel_infer/bagel_queens_2026.09.16_10.19.55_50/generated_images"  # 1750 attempt jpgs

echo "=============================================================="
echo " MODE=${MODE}   stages=${ONLY}   root=${ROOT}"
echo "=============================================================="

# ============================================================ A: RE-SCORE ONLY
if [[ "${ONLY}" == *A* ]]; then
echo
echo "--- A. RE-SCORE ONLY (no model inference) ---------------------"

for SHAPE in ${SHAPES}; do
    run "janus/maze/${SHAPE}" env \
        IMAGE_SIZE=384 MAZE_OOD_SCALES= \
        RUN_NAME="final_janus_maze" \
        GEN_DIR="${JANUS_MAZE_GEN}/${SHAPE}" \
        sbatch slurm_scripts/score_generated.sh maze "${SHAPE}"
done

run "janus/queens" env \
    IMAGE_SIZE=384 QUEEN_OOD_SCALES= \
    RUN_NAME="final_janus_queens" \
    GEN_DIR="${JANUS_QUEENS_GEN}" \
    sbatch slurm_scripts/score_generated.sh queens

run "janus_test/queens(n4)" env \
    IMAGE_SIZE=384 QUEEN_OOD_SCALES= \
    RUN_NAME="final_janus_test_queens" \
    GEN_DIR="${JANUSTEST_QUEENS_GEN}" \
    sbatch slurm_scripts/score_generated.sh queens

run "bagel/queens" env \
    IMAGE_SIZE=640 QUEEN_OOD_SCALES= \
    RUN_NAME="final_bagel_queens" \
    GEN_DIR="${BAGEL_QUEENS_GEN}" \
    sbatch slurm_scripts/score_generated.sh queens

echo "  [SKIPPED]   janus_test/maze - no matching generation dir found."
echo "              inference_results/maze3_* is n3-only (the OOD size being dropped)."
echo "              Set JANUSTEST_MAZE_GEN and add a block here once confirmed."
fi

# ============================================================== B: RE-SAMPLE
if [[ "${ONLY}" == *B* ]]; then
echo
echo "--- B. RE-SAMPLE DiT / PT (cheap GPU) -------------------------"

run "dit/maze" env \
    MAZE_OOD_SCALES= RUN_NAME="final_dit_maze" \
    sbatch slurm_scripts/sample_amaze/eval_amaze_dit_pt.sh dit maze "${DIT_MAZE_CKPT}"

run "pt/maze" env \
    MAZE_OOD_SCALES= RUN_NAME="final_pt_maze" \
    sbatch slurm_scripts/sample_amaze/eval_amaze_dit_pt.sh trm maze "${PT_MAZE_THINKER}" "${PT_MAZE_PAINTER}"

run "dit/queens" env \
    QUEEN_OOD_SCALES= RUN_NAME="final_dit_queens" \
    sbatch slurm_scripts/sample_amaze/eval_amaze_dit_pt.sh dit queens "${DIT_QUEENS_CKPT}"

run "pt/queens" env \
    QUEEN_OOD_SCALES= RUN_NAME="final_pt_queens" \
    sbatch slurm_scripts/sample_amaze/eval_amaze_dit_pt.sh trm queens "${PT_QUEENS_THINKER}" "${PT_QUEENS_PAINTER}"
fi

# =============================================================== C: GENERATE
if [[ "${ONLY}" == *C* ]]; then
echo
echo "--- C. GENERATE Bagel MAZE (expensive: 7B autoregressive) -----"
echo "    (bagel queens is already generated and is re-scored in stage A)"
: "${BAGEL_MODEL_PATH:?set BAGEL_MODEL_PATH=\$SCRATCH/models/BAGEL-7B-MoT}"
: "${BAGEL_CKPT:?set BAGEL_CKPT=<fine-tuned bagel checkpoint>}"

for SHAPE in ${SHAPES}; do
    run "bagel/maze/${SHAPE}" env \
        BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH}" CHECKPOINT="${BAGEL_CKPT}" \
        IMAGE_SIZE=1024 MAZE_OOD_SCALES= \
        RUN_NAME="final_bagel_maze" \
        sbatch slurm_scripts/sample_amaze/eval_bagel.sh maze "${SHAPE}"
done

fi

echo
echo "=============================================================="
if [[ "${MODE}" == "submit" ]]; then
    DEP=$(IFS=:; echo "${JOBS[*]}")
    COLLECT=$(sbatch --parsable --dependency=afterany:"${DEP}" \
        --job-name=amaze_collect --account=plgdiffusion3-gpu-gh200 \
        --partition=plgrid-gpu-gh200 --nodes=1 --ntasks-per-node=1 \
        --cpus-per-task=2 --mem=8G --time=00:15:00 \
        --output=slurm_outputs/%x_%j.out --error=slurm_outputs/%x_%j.err \
        --wrap="cd ${ROOT} && bash collect_final_results.sh && cd ${ROOT} && rm -f final_results.zip && zip -r final_results.zip final_results -x '*.DS_Store'")
    echo " submitted ${#JOBS[@]} eval jobs: ${JOBS[*]}"
    echo " collect+zip job ${COLLECT} will run after they all finish"
    echo " watch:   squeue -u \$USER"
    echo " result:  ${ROOT}/final_results.zip"
else
    echo " DRY RUN - nothing was submitted."
    echo " re-run as:  bash run_final_eval.sh submit"
fi
echo "=============================================================="
