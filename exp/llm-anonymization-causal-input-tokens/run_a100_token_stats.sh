#!/bin/bash
#SBATCH -p gpu2
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -n 8
#SBATCH -t 7-00:00:00
#SBATCH -J vllm_token_stats
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# A100 x2 on gpu2; walltime = partition MaxTime (7 days).
# Job end (success/fail/cancel) releases GPUs automatically.

set -euo pipefail

cd /home/xzhang5364/llm-anonymization-causal-input-tokens
mkdir -p logs

cleanup() {
  echo "[$(date '+%F %T')] job ending (exit=$?); Slurm will release GPUs."
}
trap cleanup EXIT

# Use conda env: verl
source /home/xzhang5364/miniconda3/etc/profile.d/conda.sh
conda activate verl
PYTHON=/home/xzhang5364/miniconda3/envs/verl/bin/python

# Within a 2-GPU allocation, devices are 0,1 (not 2,3).
export CUDA_VISIBLE_DEVICES=0,1

echo "[$(date '+%F %T')] host=$(hostname) job=${SLURM_JOB_ID:-NA}"
echo "[$(date '+%F %T')] conda env=verl python=$PYTHON"
echo "[$(date '+%F %T')] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
which python
"$PYTHON" -c "import vllm; print('vllm', vllm.__version__)"
nvidia-smi || true

"$PYTHON" zxz_run_causal_vllm_token_stats.py \
  --output-dir results/vllm_llama3.1-8B_all300_tokens \
  --model-path /home/xzhang5364/llm-ckpt/Llama/Llama-3.1-8B-Instruct \
  --model-name /home/xzhang5364/llm-ckpt/Llama/Llama-3.1-8B-Instruct \
  --vllm-port 8010 \
  --profile-workers 8 \
  --max-model-len 32768 \
  --max-output-tokens 8192 \
  --gpu-memory-utilization 0.85

echo "[$(date '+%F %T')] python finished successfully."
