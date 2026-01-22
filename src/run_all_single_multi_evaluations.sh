#!/bin/bash
# Script to run single-multi evaluations for all model configurations

echo "========================================="
echo "Running Single-Multi Evaluations for All Models"
echo "Started at: $(date)"
echo "========================================="

# Track successes and failures
declare -a successful=()
declare -a failed=()

# Function to run evaluation for a specific model
run_evaluation() {
    local provider=$1
    local model=$2
    local reasoning=$3
    
    echo -e "\n🔄 Processing: ${provider}_${model}_${reasoning}"
    echo "----------------------------------------"
    
    ./mining_misconceptions/run_single_multi_evaluation.sh \
        --provider "$provider" \
        --model "$model" \
        --reasoning "$reasoning" \
        --group-dir "multi"
    
    if [[ $? -eq 0 ]]; then
        successful+=("${provider}_${model}_${reasoning}")
        echo "✅ Success: ${provider}_${model}_${reasoning}"
    else
        failed+=("${provider}_${model}_${reasoning}")
        echo "❌ Failed: ${provider}_${model}_${reasoning}"
    fi
}

# Run evaluations for all model configurations
echo -e "\n📊 Running evaluations for all models...\n"

# OpenAI models (all reasoning effort levels)
run_evaluation "openai" "o3-mini" "effort-low"
run_evaluation "openai" "o3-mini" "effort-medium"
# run_evaluation "openai" "o3-mini" "effort-high"

# Anthropic models
run_evaluation "anthropic" "claude-sonnet-4-5" "reasoning"
run_evaluation "anthropic" "claude-sonnet-4-5" "no-reasoning"

# Gemini models
run_evaluation "gemini" "2.5-flash" "reasoning"
run_evaluation "gemini" "2.5-flash" "no-reasoning"

# Qwen3
run_evaluation "vllm" "qwen3-8b" "thinking"
run_evaluation "vllm" "qwen3-8b" "no-thinking"

run_evaluation "vllm" "qwen3-14b" "thinking"
run_evaluation "vllm" "qwen3-14b" "no-thinking"

# Summary
echo -e "\n========================================="
echo "EVALUATION SUMMARY"
echo "========================================="
echo "Completed at: $(date)"
echo -e "\n✅ Successful (${#successful[@]}):"
for model in "${successful[@]}"; do
    echo "   - $model"
done

if [[ ${#failed[@]} -gt 0 ]]; then
    echo -e "\n❌ Failed (${#failed[@]}):"
    for model in "${failed[@]}"; do
        echo "   - $model"
    done
fi

echo -e "\n📁 Results saved in:"
echo "   - Grouped predictions: mining_misconceptions/test_results/{model}/single_multi/"
echo "   - Evaluation metrics: mining_misconceptions/test_results/evaluations/{model}/single_multi/"

# Generate a combined summary report
echo -e "\n📊 Generating combined summary report..."
python3 -c "
import json
import os
from pathlib import Path

summary_data = {}
eval_base = Path('mining_misconceptions/test_results/evaluations')

# Collect metrics from all models
for model_dir in eval_base.iterdir():
    if model_dir.is_dir() and 'single_multi' in os.listdir(model_dir):
        metrics_file = model_dir / 'single_multi' / 'evaluation_metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                model_name = model_dir.name
                
                # Use with_novel_metrics if available, otherwise standard_metrics
                if 'with_novel_metrics' in metrics:
                    metric_data = metrics['with_novel_metrics']['overall_metrics']
                    metric_type = 'with_novel'
                elif 'standard_metrics' in metrics:
                    metric_data = metrics['standard_metrics']['overall_metrics']
                    metric_type = 'standard'
                else:
                    # Old format (backward compatibility)
                    metric_data = metrics.get('overall_metrics', {})
                    metric_type = 'legacy'
                
                summary_data[model_name] = {
                    'overall_accuracy': metric_data.get('overall_accuracy', 0),
                    'correct_only_accuracy': metric_data.get('correct_only_accuracy', 0),
                    'misconception_accuracy': metric_data.get('misconception_accuracy', 0),
                    'total_bags': metric_data.get('total_bags', 0),
                    'metric_type': metric_type
                }

# Save combined summary
output_file = eval_base / 'single_multi_combined_summary.json'
with open(output_file, 'w') as f:
    json.dump(summary_data, f, indent=2)

print(f'Combined summary saved to: {output_file}')

# Print summary table
print('\nModel Performance Summary:')
print('=' * 95)
print('{:<40} {:>10} {:>10} {:>10} {:>8} {:>12}'.format('Model', 'Overall', 'Correct', 'Misc', 'Bags', 'Metric Type'))
print('-' * 95)
for model, metrics in sorted(summary_data.items()):
    print('{:<40} {:>9.1%} {:>9.1%} {:>9.1%} {:>8} {:>12}'.format(
        model, 
        metrics['overall_accuracy'],
        metrics['correct_only_accuracy'],
        metrics['misconception_accuracy'],
        metrics['total_bags'],
        metrics.get('metric_type', 'unknown')
    ))
print('=' * 95)
"

echo -e "\n✅ All evaluations completed!" 