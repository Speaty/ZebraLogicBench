#!/bin/bash
#SBATCH --job-name=zero-eval
#SBATCH --output=/mnt/lustre/scratch/inf/js2042/logs/zero_eval_%j.out
#SBATCH --error=/mnt/lustre/scratch/inf/js2042/logs/zero_eval_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --mail-user js2042@sussex.ac.uk
#SBATCH --mail-type=ALL
# export CUDA_VISIBLE_DEVICES=0

set -x

module purge
module load Python/3.10.8-GCCcore-12.2.0

# Load modules or activate your conda environment
source /mnt/lustre/users/inf/js2042/ZebraLogicBench/.venv/bin/activate

echo "======= ENVIRONMENT DEBUG ======="
which python
python --version
python -c "import torch; print('Torch Version: ', torch.__version__)"
python -c 'import torch; print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])'
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "================================="

# Optional: ensure Hugging Face models download to Lustre
export HF_HOME=/mnt/lustre/users/inf/js2042/models/cache

# Change to the directory of your repo
cd /mnt/lustre/users/inf/js2042/ZebraLogicBench/  

python /mnt/lustre/users/inf/js2042/ZebraLogicBench/src/unified_infer.py \
	--engine vllm \
	--data_name "zebra-grid-test" \
	--model_name "Qwen/Qwen2-7B-Instruct" \
	--run_name "test"\
	--gpu_memory_utilization 0.8\
	--max_model_len -1\
	--use_hf_conv_template --use_imend_stop \
	--download_dir $HF_HOME\
	--tensor_parallel_size 1\
	--dtype bfloat16 \
	--model_pretty_name "test"\
	--top_p 1.0 --temperature 0\
	--repetition_penalty 1.0 \
	--batch_size 4 \
	--output_folder "/mnt/lustre/users/inf/js2042/ZebraLogicBench/result_dirs/" \
	--format "json"\

# Run the experiment
#bash zero_eval_local.sh \
#  -d zebra-grid-test \
#  -m Qwen/Qwen2-7B-Instruct \
#  -p Qwen2-7B-Instruct \
#  -s 1 \
#  -F json \
#  -r test
