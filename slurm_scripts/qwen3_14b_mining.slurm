#!/bin/bash
#SBATCH --job-name=qwen3_14b_mining
#SBATCH --partition=Bunescu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --output=qwen3_14b_mining-%j.out
#SBATCH --error=qwen3_14b_mining-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ealhossa@charlotte.edu

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "Working directory: $PWD"
echo "========================================="

# Change to the project directory
cd /projects/rbunescu_research/erfan_smita_space/artificial_students

# Activate virtual environment
source activate socratic_env

# CRITICAL: Set environment variables for vLLM
export VLLM_USE_V1=0
export VLLM_USE_V0=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_HOME=/usr/local/cuda

# Print environment info
echo "Environment variables:"
echo "  VLLM_USE_V1=$VLLM_USE_V1"
echo "  VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo ""

# Print GPU information
nvidia-smi
echo "========================================="

# Create directories for outputs and logs
mkdir -p job_logs
mkdir -p mining_misconceptions/test_results/vllm_qwen3-14b_thinking/{single,multi}
mkdir -p mining_misconceptions/test_results/vllm_qwen3-14b_no-thinking/{single,multi}

# Set corrupted codes directory
CORRUPTED_DIR="mining_misconceptions/data/corrupted_codes/corrupted_codes_best"
PROBLEMS_FILE="mining_misconceptions/data/problems_processed.json"

echo "Running Qwen3-14B experiments with vLLM offline mode..."
echo ""

# ============ WITH THINKING MODE ============
echo -e "\n[1/4] Running single-code analysis WITH thinking mode (BATCH MODE)..."
python mining_misconceptions/run_infer_misc.py \
    --llm vllm \
    --vllm-model Qwen/Qwen3-14B \
    --vllm-enable-thinking \
    --use-batch \
    --template zeroshot-no-reasoning \
    --input-dir $CORRUPTED_DIR \
    --problems-file $PROBLEMS_FILE \
    --output-dir mining_misconceptions/test_results/vllm_qwen3-14b_thinking/single \
    || echo "Single-code with thinking failed, continuing..."

echo -e "\n[2/4] Running multi-code analysis WITH thinking mode (BATCH MODE)..."
python mining_misconceptions/run_infer_misc_multi.py \
    --llm vllm \
    --vllm-model Qwen/Qwen3-14B \
    --vllm-enable-thinking \
    --use-batch \
    --template zeroshot-no-reasoning-multi \
    --input-dir $CORRUPTED_DIR \
    --problems-file $PROBLEMS_FILE \
    --output-dir mining_misconceptions/test_results/vllm_qwen3-14b_thinking/multi \
    --min-codes-per-group 4 \
    --max-codes-per-group 8 \
    --correct-bags-ratio 0.15 \
    || echo "Multi-code with thinking failed, continuing..."

# ============ WITHOUT THINKING MODE (Non-thinking for efficiency) ============
echo -e "\n[3/4] Running single-code analysis WITHOUT thinking mode (BATCH MODE)..."
python mining_misconceptions/run_infer_misc.py \
    --llm vllm \
    --vllm-model Qwen/Qwen3-14B \
    --use-batch \
    --template zeroshot \
    --input-dir $CORRUPTED_DIR \
    --problems-file $PROBLEMS_FILE \
    --output-dir mining_misconceptions/test_results/vllm_qwen3-14b_no-thinking/single \
    || echo "Single-code without thinking failed, continuing..."

echo -e "\n[4/4] Running multi-code analysis WITHOUT thinking mode (BATCH MODE)..."
python mining_misconceptions/run_infer_misc_multi.py \
    --llm vllm \
    --vllm-model Qwen/Qwen3-14B \
    --use-batch \
    --template zeroshot-multi \
    --input-dir $CORRUPTED_DIR \
    --problems-file $PROBLEMS_FILE \
    --output-dir mining_misconceptions/test_results/vllm_qwen3-14b_no-thinking/multi \
    --min-codes-per-group 4 \
    --max-codes-per-group 8 \
    --correct-bags-ratio 0.15 \
    || echo "Multi-code without thinking failed, continuing..."

echo "========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"

# Move the logs to the job_logs directory
mv qwen3_14b_mining-*.out job_logs/ 2>/dev/null || true
mv qwen3_14b_mining-*.err job_logs/ 2>/dev/null || true

echo "Results saved to: mining_misconceptions/test_results/vllm_qwen3-14b_*/"
echo "Logs moved to: job_logs/"

