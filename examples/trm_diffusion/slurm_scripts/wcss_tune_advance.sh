#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --account=hpc-kamildeja-1773916679
#SBATCH --partition=lem-cpu-short

# Fallback for advancing a tuning session: runs after a round's jobs end
# (tuning/driver.py submits it with --dependency=afterany) and scores the
# stage / submits the next one, unless the last run's own end-of-job advance
# already did. $1 = tuning family, $2 = round (empty: act unconditionally).

source /usr/local/sbin/modules.sh
module load Python/3.10.4-GCCcore-11.3.0
ROOT=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers
source $ROOT/trm_torch26_env/bin/activate
cd $ROOT/examples/trm_diffusion
python -m tuning.driver advance "$1" ${2:+--round "$2"}
