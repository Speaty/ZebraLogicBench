#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --output=gpu-test.out
#SBATCH --error=gpu-test.err
#SBATCH --ntasks=1
#SBATCH --gres gpu:1
#SBATCH --partition general
#SBATCH --time=00:05:00
#SBATCH --mail-user js2042@sussex.ac.uk
#SBATCH --mail-type=ALL

module load anaconda
source /mnt/lustre/users/inf/js2042/ZebraLogicBench/.venv/bin/activate

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('CUDA device count:', torch.cuda.device_count())"

