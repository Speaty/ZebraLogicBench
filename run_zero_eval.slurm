#!/bin/bash
#SBATCH --job-name=zero-eval
#SBATCH --output=/mnt/lustre/scratch/inf/js2042/logs/zero_eval_%j.out
#SBATCH --error=/mnt/lustre/scratch/inf/js2042/logs/zero_eval_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --mail-user js2042@sussex.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --export=HF_HUB_TOKEN
# export CUDA_VISIBLE_DEVICES=0

set -x

export XDG_CACHE_HOME=/mnt/lustre/users/inf/js2042/vllm_cache
mkdir -p $XDG_CACHE_HOME

# Optional, but good to have
export TORCH_COMPILE_CACHE=$XDG_CACHE_HOME/torch_compile_cache
mkdir -p $TORCH_COMPILE_CACHE

module purge
module load Python/3.10.8-GCCcore-12.2.0

# Load modules or activate your conda environment
source /mnt/lustre/users/inf/js2042/ZebraLogicBench/.venv/bin/activate
PYTHON=$(which python)

# non‐interactive HF login
if [ -z "${HF_HUB_TOKEN-}" ]; then
  echo "❌ HF_HUB_TOKEN is not set; private models may fail to download"
  exit 1
fi
echo "$HF_HUB_TOKEN" | huggingface-cli login --stdin


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
# === COFIGURATION ===
DATA_NAME="zebra-grid-mini"
RUN_NAME="mini"
MODELS=("Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-Math-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-coder-14B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")
DTYPES=("json" "xml")
BASE_DIR="/mnt/lustre/users/inf/js2042/ZebraLogicBench/result_dirs"

for MODEL in "${MODELS[@]}"; do
	# replace '/' with _ for folder name
	SAFE_MODEL_NAME="$(echo $MODEL | tr '/' '_')"
	for DT in "${DTYPES[@]}"; do
		
		# Create output directory if it doesn't exist
		OUT_DIR="$BASE_DIR/$DATA_NAME/$DT/$SAFE_MODEL_NAME/"
		mkdir -p "$OUT_DIR"

		echo "=== RUNNING MODEL = $SAFE_MODEL_NAME === FORMAT = $DT ==="
		# Run the evaluation script
		srun --gpus=2 --cpus-per-task=8 \
			$PYTHON /mnt/lustre/users/inf/js2042/ZebraLogicBench/src/unified_infer.py \
			--engine vllm \
			--data_name "$DATA_NAME" \
			--model_name "$MODEL" \
			--run_name "$RUN_NAME"\
			--gpu_memory_utilization 0.8\
			--max_model_len -1\
			--use_hf_conv_template --use_imend_stop \
			--download_dir $HF_HOME\
			--tensor_parallel_size 1\
			--dtype bfloat16 \
			--model_pretty_name "$SAFE_MODEL_NAME"\
			--top_p 1.0 --temperature 0\
			--repetition_penalty 1.0 \
			--batch_size 4 \
			--output_folder "$OUT_DIR" \
			--format "$DT"
	done
done



