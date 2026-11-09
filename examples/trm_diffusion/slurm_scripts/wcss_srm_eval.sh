#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --account=hpc-kamildeja-1773916679
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion/slurm_outputs/stdout/output_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion/slurm_outputs/stderr/error_%A_%a.err

# ABOUTME: Job array for rebuttal exp 01 (docs/rebuttal/experiments/01_ipr_srm_baselines.md): full-protocol SRM + IPR
# ABOUTME: evals of the SRM checkpoints (examples/srm_sudoku, env_srm), one manifest row (task + scale group) per task.
#
#   sbatch --array=1-16 slurm_scripts/wcss_srm_eval.sh slurm_scripts/srm_eval/full.tsv
#   check: DRY_RUN=1 SLURM_ARRAY_TASK_ID=<i> bash slurm_scripts/wcss_srm_eval.sh slurm_scripts/srm_eval/full.tsv
#
# Manifest columns (whitespace separated, '#' lines ignored; row i = array task i):
#   task        maze | queens | clevr
#   checkpoint  path relative to examples/srm_sudoku
#   scales      in-distribution scales of this shard (MAZE_SCALES / QUEEN_SCALES), '-' for none
#   ood         OOD scales of this shard (MAZE_OOD_SCALES), '-' for none
# Protocol (the paper's AMAZE eval): every test board, 5 SRM samples per board, IPR budgets 10/25/50 on 1 of them;
# SRM at 10 denoising steps per patch (3, IPR's Sudoku setting, leaves speckle noise on AMAZE/CLEVR), IPR at its
# README "MS" settings. CLEVR: 1500 seeded val scenes, CFG 2.0; generation only (scored afterwards with
# clevr_score_samples.py in the trm env).
# Output: examples/srm_sudoku/results/srm_eval/full/<task>_<scales>[_ood<ood>].json (AMAZE; merge the shards with
# amaze_srm_merge.py) or results/srm_eval/full/clevr/ (images per budget).

MANIFEST=$1
LINE=$(grep -v '^#' "$MANIFEST" | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
read -r TASK CKPT SCALES OOD <<< "$LINE"
[[ -n $OOD ]] || { echo "no manifest row $SLURM_ARRAY_TASK_ID in $MANIFEST" >&2; exit 1; }
[[ $SCALES == - ]] && SCALES=""
[[ $OOD == - ]] && OOD=""

OUTDIR=results/srm_eval/full
COMMON=(--steps-per-patch 10 --score-budgets 10,25,50)
case $TASK in
  maze)
    NAME=maze${SCALES:+_${SCALES//,/-}}; [[ -n $OOD ]] && NAME+=_ood${OOD//,/-}
    ENVV=(MAZE_SCALES=$SCALES MAZE_OOD_SCALES=$OOD)
    CMD=(env_srm/bin/python amaze_srm_eval.py "$CKPT" --task maze --out $OUTDIR/$NAME.json "${COMMON[@]}") ;;
  queens)
    NAME=queens${SCALES:+_${SCALES//,/-}}; [[ -n $OOD ]] && NAME+=_ood${OOD//,/-}
    ENVV=(QUEEN_SCALES=$SCALES QUEEN_OOD_SCALES=$OOD)
    CMD=(env_srm/bin/python amaze_srm_eval.py "$CKPT" --task queens --out $OUTDIR/$NAME.json "${COMMON[@]}") ;;
  clevr)
    ENVV=()
    CMD=(env_srm/bin/python clevr_srm_eval.py "$CKPT" --out-dir $OUTDIR/clevr --clevr-root ../trm_diffusion/data/clevr
         --n-scenes 1500 --batch-size 64 "${COMMON[@]}") ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
if [[ -n $DRY_RUN ]]; then echo "${ENVV[*]} ${CMD[*]}"; exit 0; fi

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0  # env_srm's python links into this module
ROOT=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers
cd $ROOT/examples/srm_sudoku
mkdir -p $OUTDIR
echo "$(date -Is) $SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID $(hostname): ${ENVV[*]} ${CMD[*]}"
env "${ENVV[@]}" "${CMD[@]}"
echo "$(date -Is) done"
