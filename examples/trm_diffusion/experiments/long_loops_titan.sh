#!/bin/bash
# TEMPORARY (Claude): long vs short side loops (interp_loops_first.py + interp_loops_junction.py), one model on one GPU.
# Usage: long_loops_titan.sh <gpu (nvidia-smi index)> <plain|path>
set -uo pipefail
cd ~/diffusers_interp/examples/trm_diffusion
GPU=$1; KIND=$2
source ~/trm_helios_venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments:$HOME/diffusers_interp/stubs"
for spec in "11 longest 10" "11 shortest 0" "8 longest 8"; do
  set -- $spec; N=$1; PICK=$2; MIN=$3
  OUT=runs/claude_long_loops/n${N}_${PICK}
  mkdir -p $OUT
  bash experiments/loops_first_titan.sh $GPU $KIND $OUT ++interprebability_config.data_parquet=data/amaze/maze/square/n${N}_test.parquet \
    +interprebability_config.maze_size=$N +interprebability_config.side_pick=$PICK +interprebability_config.min_side=$MIN \
    +interprebability_config.interventions=false > $OUT/log_$KIND.txt 2>&1
  python experiments/interp_loops_junction.py --run_dir $OUT --kind $KIND --n $N --side_pick $PICK --min_side $MIN > $OUT/junction_$KIND.txt 2>&1
done
echo ALLDONE > runs/claude_long_loops/done_$KIND
