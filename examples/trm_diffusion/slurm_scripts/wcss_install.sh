#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=2:00:00
#SBATCH --account=hpc-kamildeja-1773916679
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion/slurm_outputs/stdout/output_%j.out
#SBATCH --error=/lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers/examples/trm_diffusion/slurm_outputs/stderr/error_%j.err


source /usr/local/sbin/modules.sh
module load Python/3.10.4-GCCcore-11.3.0
cd /lustre/pd03/hpc-kamildeja-1773916679/pawel/diffusers
virtualenv env
source env/bin/activate
module load CUDA/12.4.0

pip install .
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
cd examples/trm_diffusion
pip install -r requirements.txt
