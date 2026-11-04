#!/bin/bash
# TEMPORARY (Claude): z_H interventions on side loops (interp_loops_zh.py) and the loop test for plain TRM variants
# (interp_loops_first.py + interp_loops_junction.py). Usage: zh_variants_titan.sh <gpu (nvidia-smi index)> <plain|path>
set -uo pipefail
cd ~/diffusers_interp/examples/trm_diffusion
GPU=$1; KIND=$2
source ~/trm_helios_venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments:$HOME/diffusers_interp/stubs"
B=runs/claude_zh
for spec in "longest 10" "shortest 0"; do
  set -- $spec; OUT=$B/n11_$1; mkdir -p $OUT
  SCRIPT=interp_loops_zh.py bash experiments/loops_first_titan.sh $GPU $KIND $OUT \
    ++interprebability_config.data_parquet=data/amaze/maze/square/n11_test.parquet +interprebability_config.maze_size=11 \
    +interprebability_config.side_pick=$1 +interprebability_config.min_side=$2 > $OUT/log_$KIND.txt 2>&1
done
if [ "$KIND" = plain ]; then
  for V in "step4000 runs/claude_plain_trm/checkpoint_step4000.pt" "noisy runs/claude_plain_trm_noisy/checkpoint_final.pt" "mse runs/claude_plain_trm_mse/checkpoint_final.pt"; do
    set -- $V; NAME=$1; CK=$2
    for spec in "11 longest 10" "8 random 0"; do
      set -- $spec; OUT=$B/variants/${NAME}_n$1_$2; mkdir -p $OUT
      bash experiments/loops_first_titan.sh $GPU plain $OUT ++interprebability_config.plain_checkpoint=$CK \
        ++interprebability_config.data_parquet=data/amaze/maze/square/n$1_test.parquet +interprebability_config.maze_size=$1 \
        +interprebability_config.side_pick=$2 +interprebability_config.min_side=$3 +interprebability_config.interventions=false > $OUT/log_plain.txt 2>&1
      python experiments/interp_loops_junction.py --run_dir $OUT --kind plain --n $1 --side_pick $2 --min_side $3 > $OUT/junction_plain.txt 2>&1
    done
  done
fi
echo ALLDONE > $B/done_$KIND
