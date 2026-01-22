#!/bin/bash
#SBATCH --job-name=evaluate_all
#SBATCH --partition=Orion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --time=26:00:00
#SBATCH --mem=8GB
#SBATCH --output=evaluate_all-%j.out
#SBATCH --error=evaluate_all-%j.err
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
mkdir -p mining_misconceptions/test_results/evaluations

# Activate virtual environment
source activate socratic_env

# Set common paths
MISCONCEPTIONS_FILE="mining_misconceptions/data/misconception.json"
CORRUPTED_DIR="mining_misconceptions/data/corrupted_codes/corrupted_codes_best"

# Set evaluation model (using Claude Sonnet 4.5 as specified in appendix.tex)
EVAL_MODEL="claude-sonnet-4-5"

echo "Evaluating all mining results with ${EVAL_MODEL}..."

# Function to evaluate results
evaluate_results() {
    local provider=$1
    local model=$2
    local reasoning_mode=$3
    local format=$4  # single or multi
    
    local input_file="mining_misconceptions/test_results/${provider}_${model}_${reasoning_mode}/${format}/predictions.json"
    if [ "$format" == "multi" ] || [ "$format" == "multi_old" ]; then
        input_file="mining_misconceptions/test_results/${provider}_${model}_${reasoning_mode}/${format}/multi_predictions.json"
    fi
    
    local output_dir="mining_misconceptions/test_results/evaluations/${provider}_${model}_${reasoning_mode}/${format}"
    
    if [ -f "$input_file" ]; then
        echo -e "\n[$provider-$model-$reasoning_mode-$format] Evaluating..."
        mkdir -p "$output_dir"
        
        python mining_misconceptions/compute_eval_metrics_multi.py \
            --predictions-file "$input_file" \
            --misconceptions-file "$MISCONCEPTIONS_FILE" \
            --input-dir "$CORRUPTED_DIR" \
            --output-dir "$output_dir" \
            --use-claude-eval \
            --eval-model "$EVAL_MODEL" \
            || echo "Evaluation failed for $provider-$model-$reasoning_mode-$format, continuing..."
    else
        echo "[$provider-$model-$reasoning_mode-$format] No predictions file found at $input_file, skipping..."
    fi
}

# Evaluate OpenAI o3-mini results (all reasoning effort levels)
echo -e "\n=== Evaluating OpenAI o3-mini results ==="
# Low effort
evaluate_results "openai" "o3-mini" "effort-low" "single"
evaluate_results "openai" "o3-mini" "effort-low" "multi"
# Medium effort
evaluate_results "openai" "o3-mini" "effort-medium" "single"
evaluate_results "openai" "o3-mini" "effort-medium" "multi"
# High effort
# evaluate_results "openai" "o3-mini" "effort-high" "single"
# evaluate_results "openai" "o3-mini" "effort-high" "multi"

# Evaluate Anthropic claude-sonnet-4-5 results
echo -e "\n=== Evaluating Anthropic claude-sonnet-4-5 results ==="
evaluate_results "anthropic" "claude-sonnet-4-5" "reasoning" "single"
evaluate_results "anthropic" "claude-sonnet-4-5" "reasoning" "multi"
# evaluate_results "anthropic" "claude-sonnet-4-5" "reasoning" "multi_old"

evaluate_results "anthropic" "claude-sonnet-4-5" "no-reasoning" "single"
evaluate_results "anthropic" "claude-sonnet-4-5" "no-reasoning" "multi"
# evaluate_results "anthropic" "claude-sonnet-4-5" "no-reasoning" "multi_old"

# Evaluate Gemini 2.5-flash results
echo -e "\n=== Evaluating Gemini 2.5-flash results ==="
evaluate_results "gemini" "2.5-flash" "reasoning" "single"
evaluate_results "gemini" "2.5-flash" "reasoning" "multi"
# evaluate_results "gemini" "2.5-flash" "reasoning" "multi_old"
evaluate_results "gemini" "2.5-flash" "no-reasoning" "single"
evaluate_results "gemini" "2.5-flash" "no-reasoning" "multi"
# evaluate_results "gemini" "2.5-flash" "no-reasoning" "multi_old"

# Create summary report
echo -e "\n=== Creating summary report ==="
python -c "
import json
import os
from pathlib import Path

eval_dir = Path('mining_misconceptions/test_results/evaluations')
summary = {}

for provider_dir in eval_dir.glob('*'):
    if not provider_dir.is_dir():
        continue
    
    provider_name = provider_dir.name
    
    for mode_dir in provider_dir.glob('*'):
        if not mode_dir.is_dir():
            continue
            
        mode = mode_dir.name  # single or multi
        summary_file = mode_dir / 'evaluation_summary.json'
        
        if summary_file.exists():
            with open(summary_file) as f:
                data = json.load(f)
                
            key = f'{provider_name}_{mode}'
            
            # Extract key metrics (supporting both old and new format)
            claude_eval = data.get('claude_evaluation_metrics', {})
            summary[key] = {
                'provider': provider_name,
                'mode': mode,
                'claude_eval': claude_eval,
                'gt_based': data.get('gt_based_metrics', {}),
                'total_predictions': data.get('recall_metrics', {}).get('total_predictions_analyzed', 0)
            }

# Save summary
with open('mining_misconceptions/test_results/evaluations/overall_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Print summary
print('\n📊 Evaluation Summary:')
for key, metrics in summary.items():
    print(f'\n{key}:')
    if metrics.get('claude_eval'):
        ce = metrics['claude_eval']
        
        # Check if new format (with standard and with_novel)
        if 'standard' in ce:
            ce_std = ce.get('standard', {})
            ce_novel = ce.get('with_novel', {})
            
            print(f'  Standard Metrics:')
            print(f'    P: {ce_std.get(\"precision\", 0):.2%}, R: {ce_std.get(\"recall\", 0):.2%}, F1: {ce_std.get(\"f1_score\", 0):.2%}')
            
            if ce_novel:
                print(f'  With Novel Misconceptions:')
                print(f'    P: {ce_novel.get(\"precision\", 0):.2%}, R: {ce_novel.get(\"recall\", 0):.2%}, F1: {ce_novel.get(\"f1_score\", 0):.2%}')
                if 'true_positives_breakdown' in ce_novel:
                    breakdown = ce_novel['true_positives_breakdown']
                    print(f'    TPs: {ce_novel.get(\"true_positives\", 0)} (std: {breakdown.get(\"standard\", 0)}, novel: {breakdown.get(\"novel\", 0)})')
        else:
            # Old format (backward compatibility)
            print(f'  Claude Evaluation - P: {ce.get(\"precision\", 0):.2%}, R: {ce.get(\"recall\", 0):.2%}, F1: {ce.get(\"f1_score\", 0):.2%}')
    
    if metrics.get('gt_based'):
        gt = metrics['gt_based']
        print(f'  GT-based Accuracy: {gt.get(\"overall_accuracy\", 0):.2%}')
    print(f'  Total predictions: {metrics.get(\"total_predictions\", 0)}')
"

echo "========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"

# Move the logs to the job_logs directory
mv evaluate_all-*.out job_logs/ 2>/dev/null || true
mv evaluate_all-*.err job_logs/ 2>/dev/null || true

echo -e "\nEvaluation results saved to: mining_misconceptions/test_results/evaluations/"
echo "Overall summary saved to: mining_misconceptions/test_results/evaluations/overall_summary.json"
echo "Logs moved to: job_logs/" 