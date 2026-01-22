#!/bin/bash
#SBATCH --job-name=gemini_mining
#SBATCH --partition=Orion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --time=26:00:00
#SBATCH --mem=8GB
#SBATCH --output=gemini_mining-%j.out
#SBATCH --error=gemini_mining-%j.err
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
mkdir -p mining_misconceptions/test_results/gemini_2.5-flash_reasoning/{single,multi}
mkdir -p mining_misconceptions/test_results/gemini_2.5-flash_no-reasoning/{single,multi}

# Activate virtual environment
source activate socratic_env

# Set corrupted codes directory
CORRUPTED_DIR="mining_misconceptions/data/corrupted_codes/corrupted_codes_best"
PROBLEMS_FILE="mining_misconceptions/data/problems_processed.json"

echo "Running Gemini 2.5-flash experiments..."

# ============ WITH REASONING ============
echo -e "\n[1/4] Running single-code analysis WITH reasoning..."
python mining_misconceptions/run_infer_misc.py \
    --llm gemini \
    --gemini-model gemini-2.5-flash \
    --template zeroshot-no-reasoning \
    --reasoning \
    --input-dir $CORRUPTED_DIR \
    --problems-file $PROBLEMS_FILE \
    --output-dir mining_misconceptions/test_results/gemini_2.5-flash_reasoning/single \
    || echo "Single-code with reasoning failed, continuing..."

echo -e "\n[2/4] Running multi-code analysis WITH reasoning..."
python mining_misconceptions/run_infer_misc_multi.py \
    --llm gemini \
    --gemini-model gemini-2.5-flash \
    --template zeroshot-no-reasoning-multi \
    --reasoning \
    --thinking-budget 2000 \
    --input-dir $CORRUPTED_DIR \
    --problems-file $PROBLEMS_FILE \
    --output-dir mining_misconceptions/test_results/gemini_2.5-flash_reasoning/multi \
    --min-codes-per-group 4 \
    --max-codes-per-group 8 \
    --correct-bags-ratio 0.15 \
    || echo "Multi-code with reasoning failed, continuing..."

# ============ WITHOUT REASONING ============
echo -e "\n[3/4] Running single-code analysis WITHOUT reasoning..."
python mining_misconceptions/run_infer_misc.py \
    --llm gemini \
    --gemini-model gemini-2.5-flash \
    --template zeroshot \
    --input-dir $CORRUPTED_DIR \
    --problems-file $PROBLEMS_FILE \
    --output-dir mining_misconceptions/test_results/gemini_2.5-flash_no-reasoning/single \
    || echo "Single-code without reasoning failed, continuing..."

echo -e "\n[4/4] Running multi-code analysis WITHOUT reasoning..."
python mining_misconceptions/run_infer_misc_multi.py \
    --llm gemini \
    --gemini-model gemini-2.5-flash \
    --template zeroshot-multi \
    --input-dir $CORRUPTED_DIR \
    --problems-file $PROBLEMS_FILE \
    --output-dir mining_misconceptions/test_results/gemini_2.5-flash_no-reasoning/multi \
    --min-codes-per-group 4 \
    --max-codes-per-group 8 \
    --correct-bags-ratio 0.15 \
    || echo "Multi-code without reasoning failed, continuing..."

echo "========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"

# Move the logs to the job_logs directory
mv gemini_mining-*.out job_logs/ 2>/dev/null || true
mv gemini_mining-*.err job_logs/ 2>/dev/null || true

echo "Results saved to: mining_misconceptions/test_results/gemini_2.5-flash_*/"
echo "Logs moved to: job_logs/" 