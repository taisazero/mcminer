#!/bin/bash
#SBATCH --job-name=eval_corrupt_code
#SBATCH --partition=Orion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=01-4:00:00
#SBATCH --mem=8GB
#SBATCH --output=eval_corrupt_code-%j.out
#SBATCH --error=eval_corrupt_code-%j.err
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
cd $SLURM_SUBMIT_DIR

# Create directories for outputs and logs
mkdir -p job_logs
mkdir -p mining_misconceptions/results

# Activate virtual environment
source activate socratic_env

# Run the evaluation on manual_eval_sample
echo "Starting evaluation of corrupted_codes_best..."
echo "This will evaluate whether corrupted codes exhibit their intended misconceptions"


python mining_misconceptions/run_mcinject_judge.py \
  --corrupted-codes-dir mining_misconceptions/data/corrupted_codes/corrupted_codes_best \
   --misconceptions-file mining_misconceptions/data/misconception.json \
  --output-file mining_misconceptions/results/corrupted_codes_best_evaluation.json \
  --apply-filtering \
  --skip-rationale \
  --filtered-output-dir mining_misconceptions/data/corrupted_codes/corrupted_codes_best_filtered \
#   --use-batch


# python mining_misconceptions/run_mcinject_judge.py \
#     --corrupted-codes-dir mining_misconceptions/data/corrupted_codes/corrupted_codes_best \
#     --misconceptions-file mining_misconceptions/data/misconception.json \
#     --prompt-template mining_misconceptions/prompt_templates/evaluation/check_misconception_exhibited.md \
#     --output-file mining_misconceptions/results/corrupted_codes_best_evaluation.json \

echo "========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"

# Move the logs to the job_logs directory
mv eval_corrupt_code-*.out job_logs/ 2>/dev/null || true
mv eval_corrupt_code-*.err job_logs/ 2>/dev/null || true

echo "Results saved to: mining_misconceptions/results/corrupted_codes_best_evaluation.json"
echo "Logs moved to: job_logs/" 