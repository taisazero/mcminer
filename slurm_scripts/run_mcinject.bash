#!/bin/bash
#SBATCH --job-name=inject_misc
#SBATCH --partition=Orion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=1-12:00:00
#SBATCH --mem=8GB
#SBATCH --output=inject_misc-%j.out
#SBATCH --error=inject_misc-%j.err
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

# Change to the project directory (adjust if needed)
cd $SLURM_SUBMIT_DIR


mkdir -p job_logs

# Activate virtual environment if you have one (uncomment and modify as needed)
source activate socratic_env

# python mining_misconceptions/run_inject_misc.py --llm anthropic --max-misconceptions 25 --max-problems 5 --output-dir mining_misconceptions/data/corrupted_codes/corrupted_codes_sonnet-4-0


# python  mining_misconceptions/run_inject_misc.py --llm anthropic --random-seed 42 --max-problems 200 --max-misconceptions 25 --max-solutions-per-problem 1 --output-dir mining_misconceptions/data/corrupted_codes/corrupted_codes_sonnet-4-0
# python mining_misconceptions/run_inject_misc.py --llm anthropic --anthropic-model claude-sonnet-4-0 --reasoning --prompt-template mining_misconceptions/prompt_templates/injection/zeroshot-no-reasoning.md --max-problems 5 --max-solutions-per-problem 1 --debug-prompt
# python mining_misconceptions/run_inject_misc.py --llm anthropic --anthropic-model claude-sonnet-4-5 --reasoning --prompt-template mining_misconceptions/prompt_templates/injection/zeroshot-no-reasoning.md --enable-feedback-loop --max-feedback-iterations 1 --max-problems 4 --max-solutions-per-problem 1 --debug-prompt
# FINAL:
python mining_misconceptions/run_inject_misc.py --llm anthropic --anthropic-model claude-sonnet-4-5 --reasoning --prompt-template mining_misconceptions/prompt_templates/injection/zeroshot-no-reasoning.md --enable-feedback-loop --max-feedback-iterations 1 --max-problems 25 --max-solutions-per-problem 1 --debug-prompt

echo "========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds" 

# move the logs to the job_logs directory
mv inject_misc-*.out job_logs/
mv inject_misc-*.err job_logs/