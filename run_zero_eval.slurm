#!/bin/bash
#SBATCH --job-name=zero-eval
#SBATCH --output=/mnt/lustre/scratch/inf/js2042/logs/zero_eval_%j.out
#SBATCH --error=/mnt/lustre/scratch/inf/js2042/logs/zero_eval_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Load modules or activate your conda environment
source .venv/bin/activate

# Optional: ensure Hugging Face models download to Lustre
export HF_HOME=/mnt/lustre/users/inf/js2042/models/cache
export TRANSFORMERS_CACHE=$HF_HOME

# Change to the directory of your repo
cd /mnt/lustre/users/inf/js2042/ZebraLogicBench/  # Adjust as needed

# Run the experiment
bash zero_eval_local.sh \
  -d zebra-grid-test \                # Dataset
  -m Qwen/Qwen2-7B-Instruct \         # Model
  -p Qwen2-7B-Instruct \              # Pretty name
  -s 2 \                              # n shards
  -F json \                           # format
  -r test                             # file name
