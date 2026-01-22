#!/bin/bash
#SBATCH --job-name=openai_mining
#SBATCH --partition=Orion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --time=26:00:00
#SBATCH --mem=8GB
#SBATCH --output=openai_mining-%j.out
#SBATCH --error=openai_mining-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ealhossa@charlotte.edu

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Start time: $(date)"
echo "Working directory: $PWD"
echo "========================================="

# Change to the project directory
cd /projects/rbunescu_research/erfan_smita_space/artificial_students

# Create directories for outputs and logs
mkdir -p job_logs
mkdir -p mining_misconceptions/test_results/openai_o3-mini_effort-low/{single,multi}
mkdir -p mining_misconceptions/test_results/openai_o3-mini_effort-medium/{single,multi}
mkdir -p mining_misconceptions/test_results/openai_o3-mini_effort-high/{single,multi}

# Activate virtual environment
source activate socratic_env

# Set corrupted codes directory
CORRUPTED_DIR="mining_misconceptions/data/corrupted_codes/corrupted_codes_best"
PROBLEMS_FILE="mining_misconceptions/data/problems_processed.json"

echo "Running OpenAI o3-mini experiments with different reasoning effort levels..."

# ============ LOW EFFORT ============
echo -e "\n[1/6] Running single-code analysis with LOW reasoning effort..."
python mining_misconceptions/run_infer_misc.py \
    --llm openai \
    --openai-model o3-mini \
    --template zeroshot-no-reasoning \
    --reasoning \
    --reasoning-effort low \
    --input-dir $CORRUPTED_DIR \
    --problems-file $PROBLEMS_FILE \
    --output-dir mining_misconceptions/test_results/openai_o3-mini_effort-low/single \
    || echo "Single-code with low effort failed, continuing..."

echo -e "\n[2/6] Running multi-code analysis with LOW reasoning effort..."
python mining_misconceptions/run_infer_misc_multi.py \
    --llm openai \
    --openai-model o3-mini \
    --template zeroshot-no-reasoning-multi \
    --reasoning \
    --reasoning-effort low \
    --input-dir $CORRUPTED_DIR \
    --problems-file $PROBLEMS_FILE \
    --output-dir mining_misconceptions/test_results/openai_o3-mini_effort-low/multi \
    --min-codes-per-group 4 \
    --max-codes-per-group 8 \
    --correct-bags-ratio 0.15 \
    || echo "Multi-code with low effort failed, continuing..."

# ============ MEDIUM EFFORT ============
echo -e "\n[3/6] Running single-code analysis with MEDIUM reasoning effort..."
python mining_misconceptions/run_infer_misc.py \
    --llm openai \
    --openai-model o3-mini \
    --template zeroshot-no-reasoning \
    --reasoning \
    --reasoning-effort medium \
    --input-dir $CORRUPTED_DIR \
    --problems-file $PROBLEMS_FILE \
    --output-dir mining_misconceptions/test_results/openai_o3-mini_effort-medium/single \
    || echo "Single-code with medium effort failed, continuing..."

echo -e "\n[4/6] Running multi-code analysis with MEDIUM reasoning effort..."
python mining_misconceptions/run_infer_misc_multi.py \
    --llm openai \
    --openai-model o3-mini \
    --template zeroshot-no-reasoning-multi \
    --reasoning \
    --reasoning-effort medium \
    --input-dir $CORRUPTED_DIR \
    --problems-file $PROBLEMS_FILE \
    --output-dir mining_misconceptions/test_results/openai_o3-mini_effort-medium/multi \
    --min-codes-per-group 4 \
    --max-codes-per-group 8 \
    --correct-bags-ratio 0.15 \
    || echo "Multi-code with medium effort failed, continuing..."

# # ============ HIGH EFFORT ============
# echo -e "\n[5/6] Running single-code analysis with HIGH reasoning effort..."
# python mining_misconceptions/run_infer_misc.py \
#     --llm openai \
#     --openai-model o3-mini \
#     --template zeroshot-no-reasoning \
#     --reasoning \
#     --reasoning-effort high \
#     --input-dir $CORRUPTED_DIR \
#     --problems-file $PROBLEMS_FILE \
#     --output-dir mining_misconceptions/test_results/openai_o3-mini_effort-high/single \
#     || echo "Single-code with high effort failed, continuing..."

# echo -e "\n[6/6] Running multi-code analysis with HIGH reasoning effort..."
# python mining_misconceptions/run_infer_misc_multi.py \
#     --llm openai \
#     --openai-model o3-mini \
#     --template zeroshot-no-reasoning-multi \
#     --reasoning \
#     --reasoning-effort high \
#     --input-dir $CORRUPTED_DIR \
#     --problems-file $PROBLEMS_FILE \
#     --output-dir mining_misconceptions/test_results/openai_o3-mini_effort-high/multi \
#     --min-codes-per-group 4 \
#     --max-codes-per-group 8 \
#     --correct-bags-ratio 0.15 \
#     || echo "Multi-code with high effort failed, continuing..."

echo "========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"

# Move the logs to the job_logs directory
mv openai_mining-*.out job_logs/ 2>/dev/null || true
mv openai_mining-*.err job_logs/ 2>/dev/null || true

echo "Results saved to:"
echo "  - mining_misconceptions/test_results/openai_o3-mini_effort-low/"
echo "  - mining_misconceptions/test_results/openai_o3-mini_effort-medium/"
# echo "  - mining_misconceptions/test_results/openai_o3-mini_effort-high/"
echo "Logs moved to: job_logs/" 