#!/bin/bash -l
#SBATCH --job-name=amaze_flops
#SBATCH --account=plgdiffusion3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#
# FLOPs-vs-accuracy + "where is reasoning needed" analysis (the sudoku analysis, ported to
# AMAZE maze / queens). See experiments/amaze_flops_sweep.py for what every stage computes.
#
# Stages (STAGES, space separated). Default = what was agreed for MAZE:
#   held-out REPORT boards (offset REPORT_OFFSET, REPORT_N per shape x size) -- feed the final plot:
#     dm        DiT baseline(s), accuracy vs #denoising steps, with real FLOPs   (DIT_CKPTS: one or more "label=path")
#     trm_flat  trained PaTh, accuracy vs n_sup / cycles, with real FLOPs
#   TUNING boards (offset 0, TUNE_N per shape x size) -- phase 1, used to DEFINE the "wise" TRM:
#     profile   per-step z_H convergence profile + step-to-step z_H cosine (regime boundary / carry map)
#     cut_at_k  reason on the first k% of denoising steps only
#     converge  exit inside the reasoning loop at relative z_H change < eps (3 thresholds), fresh / carry-in-regime
#     carry     fresh vs carry-everywhere vs carry-within-regime, over n_sup
#   later (after phase 1 is discussed): trm_wise, alloc;   report: figures + wandb images from the JSONs
#   NEW_ABLATIONS=1 (v2 design): `carry` = reset every 1/2/5/20 + measured regime boundary + cosine reset map over
#     n_sup 1..16; `alloc` = bump grid (5 centres x 3 widths) + the schedule from the z_H-change profile at an average
#     budget of ALLOC_N (2) iterations/step. Both read PROFILE_JSON (default OUT_DIR/tune_split/<task>_profile.json),
#     COS_THR (0.93) and ALLOC_EPS (0.01). SWEEP_WANDB=false keeps the sweeps out of wandb (the report logs one run).
#
# Usage (from the trm_diffusion dir; slurm_scripts/submit.sh swaps account/partition for Athena):
#   bash slurm_scripts/submit.sh slurm_scripts/sample_amaze/flops_analysis.sh maze
#   bash slurm_scripts/sample_amaze/submit_flops_both.sh maze      # both clusters, cancels the loser
# Env: STAGES, OUT_DIR (runs/flops_analysis/<task>), TUNE_N (16) REPORT_OFFSET (16) REPORT_N (32), WANDB_PROJECT (PaTh-rebuttal), WANDB_GROUP (flops),
#      DIT_CKPTS ("base=runs/dit_maze_final/checkpoint_final.pt 20m=runs/dit_maze_20m/... 200m=..."), TRM_CKPT PAINTER_CKPT,
#      SMOKE=1 (tiny grids/boards, no wandb -- a plumbing check only)
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
TASK="${1:?usage: flops_analysis.sh <maze|queens>}"
[[ "${TASK}" == "maze" || "${TASK}" == "queens" ]] || { echo "TASK must be maze|queens" >&2; exit 1; }

source "${PROJECT_ROOT}/slurm_scripts/cluster_env.sh"
cluster_activate
cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
echo ">> task=${TASK}"

STAGES="${STAGES:-dm trm_flat profile cut_at_k converge carry report}"
OUT_DIR="${OUT_DIR:-runs/flops_analysis/${TASK}}"
TUNE_N="${TUNE_N:-16}"; REPORT_OFFSET="${REPORT_OFFSET:-16}"; REPORT_N="${REPORT_N:-32}"
WANDB_PROJECT="${WANDB_PROJECT:-PaTh-rebuttal}"
WANDB_GROUP="${WANDB_GROUP:-flops}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/amaze}"
[[ "${TASK}" == "maze" ]] || { echo "only maze is set up for now (agreed: mazes first)" >&2; exit 1; }

TRM_CKPT="${TRM_CKPT:-runs/pt_maze_final_thinker/checkpoint_final.pt}"
PAINTER_CKPT="${PAINTER_CKPT:-runs/pt_maze_final_painter/checkpoint_final.pt}"
DIT_CKPTS="${DIT_CKPTS:-base=runs/dit_maze_final/checkpoint_final.pt}"
EVAL_CB=amaze

COMMON=( +sweep.task="${TASK}" +sweep.batch_size=64 +sweep.wandb_project="${WANDB_PROJECT}"
         +sweep.wandb_group="${WANDB_GROUP}" +data_root="${DATA_ROOT}" )
TUNE=( +sweep.puzzles_per_combo="${TUNE_N}" +sweep.offset=0 )
REPORT=( +sweep.puzzles_per_combo="${REPORT_N}" +sweep.offset="${REPORT_OFFSET}" )
SMOKE_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  # plumbing check: 1 board of 1 shape x 1 size, tiny grids, no wandb, separate output dir
  COMMON=( +sweep.task="${TASK}" +sweep.batch_size=8 +sweep.wandb=false +data_root="${DATA_ROOT}"
           '+sweep.geometries=[square]' '+sweep.scales=[5]' )
  TUNE=( +sweep.puzzles_per_combo=2 +sweep.offset=0 ); REPORT=( +sweep.puzzles_per_combo=2 +sweep.offset=2 )
  OUT_DIR="${OUT_DIR}_smoke"
  SMOKE_ARGS=( '+sweep.steps=[2,5]' '+sweep.n_sup_values=[1,2]' '+sweep.cut_fracs=[0.5,1.0]'
               '+sweep.eps_values=[0.1]' '+sweep.carry_n_values=[2]' '+sweep.num_inference_steps=4'
               '+sweep.regime_boundary_step=2' )
fi
mkdir -p "${OUT_DIR}/report_split" "${OUT_DIR}/tune_split"
SWEEP_WANDB="${SWEEP_WANDB:-true}"
[[ "${SMOKE:-0}" != "1" && "${SWEEP_WANDB}" == "false" ]] && COMMON+=( +sweep.wandb=false )
PROFILE_JSON="${PROFILE_JSON:-${OUT_DIR}/tune_split/${TASK}_profile.json}"
CARRY_ARGS=(); ALLOC_ARGS=()
if [[ "${NEW_ABLATIONS:-0}" == "1" ]]; then
  CARRY_ARGS=( +sweep.profile_json="${PROFILE_JSON}" +sweep.cos_threshold="${COS_THR:-0.93}" )
  ALLOC_ARGS=( +sweep.profile_json="${PROFILE_JSON}" +sweep.alloc_profile_eps="${ALLOC_EPS:-0.01}"
               "+sweep.alloc_flat_n_values=[${ALLOC_N:-2}]" '+sweep.alloc_directions=[bump]'
               '+sweep.alloc_peaks=[0.1,0.3,0.5,0.7,0.9]' '+sweep.alloc_widths=[0.08,0.2,0.4]' )
  [[ "${SMOKE:-0}" == "1" ]] || CARRY_ARGS+=( '+sweep.carry_n_values=[1,2,4,8,16]' )
fi
want() { [[ " ${STAGES} " == *" $1 "* ]]; }

TRM_ARGS=( experiment=amaze_thinker_v2_controlnet painter.checkpoint="${PAINTER_CKPT}" +checkpoint="${TRM_CKPT}"
           eval_callbacks="${EVAL_CB}" )

FAILED=()
run_stage() {  # run_stage <name> <cmd...>; one failing stage must not discard the others
  local name="$1"; shift
  echo "=== [flops/${TASK}] stage ${name} ($(date +%H:%M:%S)) ==="
  if ! "$@"; then echo "!!! stage ${name} FAILED" >&2; FAILED+=("${name}"); fi
}
trm_stage() {  # trm_stage <mode> <split: tune|report> [extra overrides...]
  local mode="$1" split="$2"; shift 2
  local SPLIT_ARGS=( "${TUNE[@]}" ); [[ "${split}" == "report" ]] && SPLIT_ARGS=( "${REPORT[@]}" )
  run_stage "${mode}" srun python experiments/amaze_flops_sweep.py "${TRM_ARGS[@]}" "${COMMON[@]}" "${SPLIT_ARGS[@]}" \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} "$@" +sweep.mode="${mode}" +sweep.out="${OUT_DIR}/${split}_split/${TASK}_${mode}.json" \
    +sweep.wandb_name="flops_${TASK}_${mode}"
}

if want dm || want trm_flat || want trm_wise || want alloc || want profile || want cut_at_k || want converge || want carry; then
  for ck in "${TRM_CKPT}" "${PAINTER_CKPT}"; do [[ -s "${ck}" ]] || { echo "missing ${ck}" >&2; exit 1; }; done
fi

# ---- held-out report boards ----
if want dm; then
  for entry in ${DIT_CKPTS}; do
    label="${entry%%=*}"; ck="${entry#*=}"
    [[ -s "${ck}" ]] || { echo "missing DiT checkpoint ${ck}" >&2; exit 1; }
    # The checkpoints hold weights only: rebuild each DiT at its own size (train_dit_sized.sh).
    case "${label}" in
      base)   DIT_ARCH=() ;;
      20m)    DIT_ARCH=( backbone.num_layers=12 backbone.num_attention_heads=6 backbone.attention_head_dim=48 ) ;;
      medium) DIT_ARCH=( backbone.num_layers=16 backbone.num_attention_heads=12 ) ;;
      200m)   DIT_ARCH=( backbone.num_layers=18 backbone.num_attention_heads=12 ) ;;
      large)  DIT_ARCH=( backbone.num_layers=24 backbone.num_attention_heads=16 ) ;;
      *) echo "unknown DiT label '${label}' (base|20m|medium|200m|large)" >&2; exit 1 ;;
    esac
    run_stage "dm_${label}" srun python experiments/amaze_flops_sweep.py experiment="amaze_dit_${TASK}" \
      ${DIT_ARCH[@]+"${DIT_ARCH[@]}"} +checkpoint="${ck}" "${COMMON[@]}" "${REPORT[@]}" ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
      +sweep.mode=dm +sweep.out="${OUT_DIR}/report_split/${TASK}_dm_${label}.json" \
      +sweep.wandb_name="flops_${TASK}_dm_${label}" "+sweep.wandb_tags=[dit_${label}]"
  done
fi
want trm_flat && trm_stage trm_flat report
# ---- tuning boards: phase 1 ----
for m in profile cut_at_k converge; do want "${m}" && trm_stage "${m}" tune; done
want carry && trm_stage carry tune ${CARRY_ARGS[@]+"${CARRY_ARGS[@]}"}
# ---- later stages (not in the default list) ----
want trm_wise && trm_stage trm_wise report
want alloc && trm_stage alloc tune ${ALLOC_ARGS[@]+"${ALLOC_ARGS[@]}"}

if want report; then
  R=( --dir "${OUT_DIR}" --tasks "${TASK}" --wandb-project "${WANDB_PROJECT}" --wandb-group "${WANDB_GROUP}" )
  [[ "${SMOKE:-0}" == "1" ]] && R+=( --no-wandb )
  run_stage report python experiments/amaze_flops_report.py "${R[@]}"
fi

if (( ${#FAILED[@]} )); then echo "DONE with failed stages: ${FAILED[*]}" >&2; exit 1; fi
echo "DONE flops_analysis ${TASK} -> ${OUT_DIR}"
