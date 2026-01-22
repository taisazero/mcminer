#!/bin/bash
# Script to run the complete single-multi evaluation pipeline

# Default values
PROVIDER="openai"
MODEL="o3-mini"
REASONING="reasoning"
GROUP_DIR="multi_old"
OUTPUT_BASE="mining_misconceptions/test_results/evaluations_gemini_pro_batch"
# EVAL_DIR="mining_misconceptions/test_results/evaluations_gpt5_batch"
EVAL_DIR="mining_misconceptions/test_results/evaluations_gemini_pro_batch"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --reasoning)
            REASONING="$2"
            shift 2
            ;;
        --group-dir)
            GROUP_DIR="$2"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--provider PROVIDER] [--model MODEL] [--reasoning REASONING] [--group-dir GROUP_DIR] [--output-base OUTPUT_BASE]"
            exit 1
            ;;
    esac
done

# Construct paths
MODEL_DIR="${PROVIDER}_${MODEL}_${REASONING}"
MULTI_PRED_FILE="mining_misconceptions/test_results/${MODEL_DIR}/${GROUP_DIR}/multi_predictions.json"
SINGLE_PRED_DIR="mining_misconceptions/test_results/${MODEL_DIR}/single"
GROUPED_PRED_DIR="mining_misconceptions/test_results/${MODEL_DIR}/single_multi"
GROUPED_PRED_FILE="${GROUPED_PRED_DIR}/single_multi_predictions.json"
EVAL_OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_DIR}/single_multi"

echo "========================================="
echo "Single-Multi Evaluation Pipeline"
echo "========================================="
echo "Provider: $PROVIDER"
echo "Model: $MODEL"
echo "Reasoning: $REASONING"
echo "Group directory: $GROUP_DIR"
echo "Multi predictions: $MULTI_PRED_FILE"
echo "Single predictions: $SINGLE_PRED_DIR"
echo "Grouped predictions: $GROUPED_PRED_DIR"
echo "Evaluation output: $EVAL_OUTPUT_DIR"
echo "========================================="

# Check if files exist
if [ ! -f "$MULTI_PRED_FILE" ]; then
    echo "ERROR: Multi predictions file not found: $MULTI_PRED_FILE"
    exit 1
fi

if [ ! -d "$SINGLE_PRED_DIR" ]; then
    echo "ERROR: Single predictions directory not found: $SINGLE_PRED_DIR"
    exit 1
fi

# Step 1: Create grouped predictions
echo -e "\n📦 Step 1: Creating grouped predictions..."
python mining_misconceptions/create_single_multi_predictions.py \
    --multi-predictions-file "$MULTI_PRED_FILE" \
    --single-predictions-dir "$SINGLE_PRED_DIR" \
    --output-file "$GROUPED_PRED_FILE" \
    --pretty-print

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create grouped predictions"
    exit 1
fi

# Step 2: Evaluate grouped predictions
echo -e "\n📊 Step 2: Evaluating grouped predictions..."
python mining_misconceptions/evaluate_single_multi_predictions.py \
    --grouped-predictions-file "$GROUPED_PRED_FILE" \
    --evaluations-dir "$EVAL_DIR" \
    --output-dir "$EVAL_OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to evaluate grouped predictions"
    exit 1
fi

echo -e "\n✅ Pipeline completed successfully!"
echo "Grouped predictions saved to: $GROUPED_PRED_DIR"
echo "Evaluation results saved to: $EVAL_OUTPUT_DIR" 