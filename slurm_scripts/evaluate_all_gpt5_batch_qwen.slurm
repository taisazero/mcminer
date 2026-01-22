#!/bin/bash
#SBATCH --job-name=eval_gpt5_batch_qwen
#SBATCH --partition=Orion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --time=48:00:00
#SBATCH --mem=8GB
#SBATCH --output=eval_gpt5_batch_qwen-%j.out
#SBATCH --error=eval_gpt5_batch-qwen-%j.err
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
echo "USING GPT-5 RESPONSES BATCH API FOR EVALUATION"
echo "========================================="

# Change to the project directory
cd /projects/rbunescu_research/erfan_smita_space/artificial_students

# Create directories for outputs and logs
mkdir -p job_logs
mkdir -p mining_misconceptions/test_results/evaluations_gpt5_batch

# Activate virtual environment
source activate socratic_env

# Set common paths
MISCONCEPTIONS_FILE="mining_misconceptions/data/misconception.json"
CORRUPTED_DIR="mining_misconceptions/data/corrupted_codes/corrupted_codes_best"

# Set evaluation model to GPT-5
EVAL_MODEL="gpt-5"

echo "Evaluating all mining results with ${EVAL_MODEL} using RESPONSES BATCH API..."
echo "Note: Batch processing will poll for completion (max 48 hours)"

# Function to evaluate results with BATCH API
evaluate_results() {
    local provider=$1
    local model=$2
    local reasoning_mode=$3
    local format=$4  # single or multi
    
    local input_file="mining_misconceptions/test_results/${provider}_${model}_${reasoning_mode}/${format}/predictions.json"
    if [ "$format" == "multi" ] || [ "$format" == "multi_old" ]; then
        input_file="mining_misconceptions/test_results/${provider}_${model}_${reasoning_mode}/${format}/multi_predictions.json"
    fi
    
    # Save to separate directory for GPT-5 BATCH evaluations
    local output_dir="mining_misconceptions/test_results/evaluations_gpt5_batch/${provider}_${model}_${reasoning_mode}/${format}"
    
    if [ -f "$input_file" ]; then
        echo -e "\n[$provider-$model-$reasoning_mode-$format] Evaluating with ${EVAL_MODEL} RESPONSES BATCH API..."
        mkdir -p "$output_dir"
        
        # Enable batch processing with --claude-batch flag
        python mining_misconceptions/compute_eval_metrics_multi.py \
            --predictions-file "$input_file" \
            --misconceptions-file "$MISCONCEPTIONS_FILE" \
            --input-dir "$CORRUPTED_DIR" \
            --output-dir "$output_dir" \
            --use-claude-eval \
            --eval-model "$EVAL_MODEL" \
            --claude-batch \
            || echo "Evaluation failed for $provider-$model-$reasoning_mode-$format, continuing..."
    else
        echo "[$provider-$model-$reasoning_mode-$format] No predictions file found at $input_file, skipping..."
    fi
}


# Evaluate Qwen3-8B results
echo -e "\n=== Evaluating Qwen3-8B results ==="
evaluate_results "vllm" "qwen3-8b" "thinking" "single"
evaluate_results "vllm" "qwen3-8b" "thinking" "multi"
evaluate_results "vllm" "qwen3-8b" "no-thinking" "single"
evaluate_results "vllm" "qwen3-8b" "no-thinking" "multi"

# Evaluate Qwen3-14B results
echo -e "\n=== Evaluating Qwen3-14B results ==="
evaluate_results "vllm" "qwen3-14b" "thinking" "single"
evaluate_results "vllm" "qwen3-14b" "thinking" "multi"
evaluate_results "vllm" "qwen3-14b" "no-thinking" "single"
evaluate_results "vllm" "qwen3-14b" "no-thinking" "multi"

# Create summary report
echo -e "\n=== Creating summary report ==="
python -c "
import json
import os
from pathlib import Path

eval_dir = Path('mining_misconceptions/test_results/evaluations_gpt5_batch')
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
with open('mining_misconceptions/test_results/evaluations_gpt5_batch/overall_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Print summary
print('\n📊 GPT-5 RESPONSES BATCH API Evaluation Summary:')
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
mv eval_gpt5_batch-*.out job_logs/ 2>/dev/null || true
mv eval_gpt5_batch-*.err job_logs/ 2>/dev/null || true

echo -e "\nBATCH API Evaluation results saved to: mining_misconceptions/test_results/evaluations_gpt5_batch/"
echo "Overall summary saved to: mining_misconceptions/test_results/evaluations_gpt5_batch/overall_summary.json"
echo "Logs moved to: job_logs/"
echo ""

