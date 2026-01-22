#!/usr/bin/env python3
"""
Script to evaluate grouped single predictions using bag-level logic.

This script:
1. Loads grouped predictions created by create_single_multi_predictions.py
2. Uses existing Claude evaluations when available
3. For misconception samples without evaluations, runs Claude evaluation
4. For correct samples, simply checks if predicted_misconceptions is empty
5. Applies bag-level evaluation logic
6. Generates detailed analytics by misconception and problem

Usage:
    python evaluate_single_multi_predictions.py \
        --grouped-predictions-file test_results/openai_o3-mini_reasoning/single_multi_predictions.json \
        --evaluations-dir test_results/evaluations \
        --output-dir test_results/evaluations/single_multi
"""

import argparse
import json
import os
import sys
import subprocess
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

def load_json_file(file_path: str) -> Any:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_claude_evaluation(prediction_id: str, evaluations_dir: str, provider: str, model: str, reasoning: str) -> Optional[Dict]:
    """
    Find Claude evaluation result for a specific prediction ID.
    """
    # Construct path to Claude evaluation results
    eval_file = os.path.join(evaluations_dir, f"{provider}_{model}_{reasoning}", "single", "claude_evaluation_results.json")
    
    if not os.path.exists(eval_file):
        return None
    
    try:
        eval_results = load_json_file(eval_file)
        
        # Search for this prediction in evaluation details
        for detail in eval_results.get("evaluation_details", []):
            if detail.get("prediction_id") == prediction_id:
                return detail
        
    except Exception as e:
        print(f"Error loading evaluation file {eval_file}: {e}")
    
    return None

def run_claude_evaluation_for_misconceptions(prediction_ids: List[str], 
                                           predictions_file: str,
                                           misconceptions_file: str,
                                           input_dir: str) -> Dict[str, Dict]:
    """
    Run Claude evaluation only for specified misconception predictions.
    Returns a dict mapping prediction_id to evaluation result.
    """
    print(f"Running Claude evaluation for {len(prediction_ids)} misconception predictions...")
    
    # Create a temporary predictions file with only the needed predictions
    temp_predictions = []
    all_predictions = load_json_file(predictions_file)
    
    for pred in all_predictions:
        if pred.get("prediction_id") in prediction_ids:
            temp_predictions.append(pred)
    
    if not temp_predictions:
        print("No predictions found for Claude evaluation")
        return {}
    
    # Save temporary predictions file
    temp_file = predictions_file.replace('.json', '_temp_eval.json')
    with open(temp_file, 'w') as f:
        json.dump(temp_predictions, f)
    
    try:
        # Run compute_eval_metrics_multi.py
        cmd = [
            "python", "mining_misconceptions/compute_eval_metrics_multi.py",
            "--predictions-file", temp_file,
            "--misconceptions-file", misconceptions_file,
            "--input-dir", input_dir,
            "--output-dir", "temp_eval_output",
            "--use-claude-eval"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running Claude evaluation: {result.stderr}")
            return {}
        
        # Load the results
        claude_results_file = "temp_eval_output/claude_evaluation_results.json"
        if os.path.exists(claude_results_file):
            claude_results = load_json_file(claude_results_file)
            
            # Create mapping
            eval_map = {}
            for detail in claude_results.get("evaluation_details", []):
                pred_id = detail.get("prediction_id")
                if pred_id:
                    eval_map[pred_id] = detail
            
            # Cleanup
            os.remove(temp_file)
            if os.path.exists("temp_eval_output"):
                import shutil
                shutil.rmtree("temp_eval_output")
            
            return eval_map
        
    except Exception as e:
        print(f"Error in Claude evaluation: {e}")
        
    # Cleanup on error
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return {}

def evaluate_single_prediction(single_pred: Dict, 
                             group_type: str,
                             claude_eval: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Evaluate a single prediction with both standard and with_novel matching.
    For correct samples: check if predicted_misconceptions is empty
    For misconception samples: use Claude evaluation
    """
    result = {
        "prediction_id": single_pred.get("prediction_id"),
        "match_standard": False,
        "match_with_novel": False,
        "evaluation_method": "unknown"
    }
    
    if group_type == "correct_only":
        # For correct samples, just check if predictions are empty
        has_no_predictions = (
            len(single_pred.get("predicted_misconceptions", [])) == 0 or
            single_pred.get("no_predicted_misconceptions", False)
        )
        result["match_standard"] = has_no_predictions
        result["match_with_novel"] = has_no_predictions
        result["evaluation_method"] = "empty_check"
        
    else:  # misconception group
        if claude_eval:
            # Use Claude evaluation result
            result["match_standard"] = claude_eval.get("match", False)
            result["match_with_novel"] = claude_eval.get("match_with_novel", claude_eval.get("match", False))
            result["confidence"] = claude_eval.get("confidence", "unknown")
            result["evaluation_method"] = "claude_existing"
        else:
            # No Claude evaluation available - this shouldn't happen if we run evaluations
            result["match_standard"] = False
            result["match_with_novel"] = False
            result["evaluation_method"] = "no_evaluation"
            result["error"] = "No Claude evaluation available"
    
    return result

def evaluate_bag(group: Dict, single_evaluations: List[Dict]) -> Dict[str, Any]:
    """
    Apply bag-level evaluation logic for both standard and with_novel.
    Returns results for both metric types.
    - For misconception groups: count matches vs non-matches (ignore NONEs)
    - For correct-only groups: ALL must be NONE for TP
    """
    group_type = group.get("group_type")
    gt_misconception = group.get("group_info", {}).get("gt_misconception")
    
    # Compute for both standard and with_novel
    results = {}
    
    for metric_type in ["standard", "with_novel"]:
        match_field = f"match_{metric_type}"
        
        if group_type == "correct_only":
            # For correct-only: ALL samples must have no predictions (strict criteria)
            has_any_predictions = False
            for single_pred in group["single_predictions"]:
                has_predictions = (
                    len(single_pred.get("predicted_misconceptions", [])) > 0 and
                    not single_pred.get("no_predicted_misconceptions", False)
                )
                if has_predictions:
                    has_any_predictions = True
                    break
            
            # Bag is correct only if NO samples have predictions
            bag_match = not has_any_predictions
            
            # Count how many samples had no predictions
            match_count = sum(1 for e in single_evaluations if e[match_field])
            total = len(single_evaluations)
            
            results[metric_type] = {
                "bag_match": bag_match,
                "single_matches": match_count,
                "single_total": total,
                "match_ratio": match_count / total if total > 0 else 0,
                "has_any_predictions": has_any_predictions
            }
            
        else:  # misconception group
            # For misconception: count matches vs non-matches (ignore NONEs)
            match_count = 0
            non_match_count = 0
            none_count = 0
            
            for i, single_eval in enumerate(single_evaluations):
                single_pred = group["single_predictions"][i]
                
                # Check if this prediction is NONE/empty
                has_no_predictions = (
                    len(single_pred.get("predicted_misconceptions", [])) == 0 or
                    single_pred.get("no_predicted_misconceptions", False)
                )
                
                if has_no_predictions:
                    none_count += 1
                else:
                    # This sample has predictions - check if it matches GT
                    if single_eval[match_field]:
                        match_count += 1
                    else:
                        non_match_count += 1
            
            # Bag is correct if matches > non-matches
            bag_match = match_count > non_match_count
            
            results[metric_type] = {
                "bag_match": bag_match,
                "match_count": match_count,
                "non_match_count": non_match_count,
                "none_count": none_count,
                "decision_detail": f"{match_count} matches > {non_match_count} non-matches = {bag_match}"
            }
    
    return {
        "group_id": group.get("group_id"),
        "group_type": group_type,
        "gt_misconception": gt_misconception,
        "standard": results["standard"],
        "with_novel": results["with_novel"],
        "single_total": len(single_evaluations),
        "evaluation_method": "both_standard_and_novel"
    }

def extract_model_info(grouped_predictions_file: str) -> Tuple[str, str, str]:
    """
    Extract provider, model, and reasoning mode from file path.
    Example: test_results/openai_o3-mini_reasoning/single_multi_predictions.json
    """
    path_parts = Path(grouped_predictions_file).parts
    
    # Find the test_results directory and get the next part
    for i, part in enumerate(path_parts):
        if part == "test_results" and i + 1 < len(path_parts):
            model_dir = path_parts[i + 1]
            # Parse format: provider_model_reasoning
            parts = model_dir.split('_')
            if len(parts) >= 3:
                provider = parts[0]
                model = '_'.join(parts[1:-1])  # Handle models with underscores
                reasoning = parts[-1]
                return provider, model, reasoning
    
    return "unknown", "unknown", "unknown"

def calculate_detailed_metrics(bag_evaluations: List[Dict], grouped_predictions: List[Dict]) -> Dict[str, Any]:
    """
    Calculate detailed metrics for both standard and with_novel approaches.
    """
    metrics = {"standard": {}, "with_novel": {}}
    
    for metric_type in ["standard", "with_novel"]:
        # Overall metrics
        total_bags = len(bag_evaluations)
        correct_bags = sum(1 for e in bag_evaluations if e[metric_type]["bag_match"])
        
        # Group by type
        correct_only_evals = [e for e in bag_evaluations if e["group_type"] == "correct_only"]
        misconception_evals = [e for e in bag_evaluations if e["group_type"] == "misconception"]
        
        # Calculate accuracies
        correct_only_accuracy = (
            sum(1 for e in correct_only_evals if e[metric_type]["bag_match"]) / len(correct_only_evals)
            if correct_only_evals else 0
        )
        misconception_accuracy = (
            sum(1 for e in misconception_evals if e[metric_type]["bag_match"]) / len(misconception_evals)
            if misconception_evals else 0
        )
    
        # Breakdown by misconception ID
        by_misconception = defaultdict(lambda: {"total": 0, "correct": 0})
        by_problem = defaultdict(lambda: {"total": 0, "correct": 0, "misconceptions": set()})
        
        for eval_result in bag_evaluations:
            # Get problem IDs from the original group info
            group_id = eval_result.get("group_id")
            # Find the original group to get problem IDs
            for group in grouped_predictions:
                if group.get("group_id") == group_id:
                    problem_ids = group.get("group_info", {}).get("problem_ids", [])
                    for problem_id in problem_ids:
                        by_problem[problem_id]["total"] += 1
                        if eval_result[metric_type]["bag_match"]:
                            by_problem[problem_id]["correct"] += 1
                        if eval_result["group_type"] == "misconception":
                            by_problem[problem_id]["misconceptions"].add(eval_result["gt_misconception"])
                    break
            
            if eval_result["group_type"] == "misconception":
                misc_id = str(eval_result["gt_misconception"])
                by_misconception[misc_id]["total"] += 1
                if eval_result[metric_type]["bag_match"]:
                    by_misconception[misc_id]["correct"] += 1
        
        # Calculate accuracy per misconception
        misconception_metrics = {}
        for misc_id, data in by_misconception.items():
            accuracy = data["correct"] / data["total"] if data["total"] > 0 else 0
            misconception_metrics[misc_id] = {
                "accuracy": accuracy,
                "total_groups": data["total"],
                "correct_groups": data["correct"]
            }
        
        # Sort misconceptions by accuracy
        sorted_misconceptions = sorted(misconception_metrics.items(), key=lambda x: x[1]["accuracy"])
        
        metrics[metric_type] = {
            "overall_metrics": {
                "total_bags": total_bags,
                "correct_bags": correct_bags,
                "overall_accuracy": correct_bags / total_bags if total_bags > 0 else 0,
                "correct_only_accuracy": correct_only_accuracy,
                "misconception_accuracy": misconception_accuracy,
                "correct_only_count": len(correct_only_evals),
                "misconception_count": len(misconception_evals)
            },
            "by_misconception": dict(misconception_metrics),
            "top_5_best_misconceptions": [
                {"misconception_id": m[0], **m[1]} for m in sorted_misconceptions[-5:][::-1]
            ],
            "top_5_worst_misconceptions": [
                {"misconception_id": m[0], **m[1]} for m in sorted_misconceptions[:5]
            ]
        }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate grouped single predictions using bag-level logic"
    )
    
    parser.add_argument(
        "--grouped-predictions-file",
        required=True,
        help="Path to grouped predictions JSON file"
    )
    parser.add_argument(
        "--evaluations-dir",
        default="mining_misconceptions/test_results/evaluations",
        help="Directory containing Claude evaluation results"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--misconceptions-file",
        default="mining_misconceptions/data/misconception.json",
        help="Path to misconceptions database"
    )
    parser.add_argument(
        "--input-dir",
        default="mining_misconceptions/data/corrupted_codes/corrupted_codes_best",
        help="Directory containing original code files"
    )
    parser.add_argument(
        "--force-claude-eval",
        action="store_true",
        help="Force re-evaluation with Claude even if results exist"
    )
    
    args = parser.parse_args()
    
    # Load grouped predictions
    print(f"Loading grouped predictions from: {args.grouped_predictions_file}")
    grouped_predictions = load_json_file(args.grouped_predictions_file)
    print(f"  Loaded {len(grouped_predictions)} groups")
    
    # Extract model information
    provider, model, reasoning = extract_model_info(args.grouped_predictions_file)
    print(f"  Model info: {provider} / {model} / {reasoning}")
    
    # Get the original single predictions file path
    # The grouped predictions are in test_results/{model}/single_multi/
    # We need to go up one level to find the single/ directory
    grouped_dir = os.path.dirname(args.grouped_predictions_file)
    model_dir = os.path.dirname(grouped_dir)  # Go up from single_multi to model dir
    single_pred_file = os.path.join(model_dir, "single", "predictions.json")
    
    print(f"  Looking for single predictions at: {single_pred_file}")
    if not os.path.exists(single_pred_file):
        print(f"ERROR: Single predictions file not found at: {single_pred_file}")
        return 1
    
    # Collect all prediction IDs that need Claude evaluation
    misconception_pred_ids = []
    for group in grouped_predictions:
        if group["group_type"] == "misconception":
            for single_pred in group["single_predictions"]:
                misconception_pred_ids.append(single_pred["prediction_id"])
    
    # Check for existing Claude evaluations and collect missing ones
    missing_claude_ids = []
    existing_claude_evals = {}
    
    if not args.force_claude_eval:
        print("\nChecking for existing Claude evaluations...")
        for pred_id in misconception_pred_ids:
            claude_eval = find_claude_evaluation(pred_id, args.evaluations_dir, provider, model, reasoning)
            if claude_eval:
                existing_claude_evals[pred_id] = claude_eval
            else:
                missing_claude_ids.append(pred_id)
        
        print(f"  Found {len(existing_claude_evals)} existing evaluations")
        print(f"  Need to evaluate {len(missing_claude_ids)} misconception predictions")
    else:
        missing_claude_ids = misconception_pred_ids
        print(f"\nForce mode: Will evaluate all {len(missing_claude_ids)} misconception predictions")
    
    # Run Claude evaluation for missing predictions
    new_claude_evals = {}
    if missing_claude_ids:
        new_claude_evals = run_claude_evaluation_for_misconceptions(
            missing_claude_ids,
            single_pred_file,
            args.misconceptions_file,
            args.input_dir
        )
        print(f"  Obtained {len(new_claude_evals)} new Claude evaluations")
    
    # Combine evaluations
    all_claude_evals = {**existing_claude_evals, **new_claude_evals}
    
    # Evaluate each group
    print("\nEvaluating groups...")
    bag_evaluations = []
    
    for group in grouped_predictions:
        single_evaluations = []
        
        # Evaluate each single prediction in the group
        for single_pred in group["single_predictions"]:
            pred_id = single_pred["prediction_id"]
            claude_eval = all_claude_evals.get(pred_id)
            
            single_eval = evaluate_single_prediction(
                single_pred,
                group["group_type"],
                claude_eval
            )
            single_evaluations.append(single_eval)
        
        # Apply bag-level logic
        if single_evaluations:
            bag_eval = evaluate_bag(group, single_evaluations)
            bag_eval["single_evaluations"] = single_evaluations
            bag_evaluations.append(bag_eval)
    
    # Calculate detailed metrics
    print("\nCalculating detailed metrics...")
    metrics = calculate_detailed_metrics(bag_evaluations, grouped_predictions)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save detailed evaluation results
    eval_file = os.path.join(args.output_dir, "bag_evaluation_results.json")
    with open(eval_file, 'w') as f:
        json.dump({
            "evaluation_timestamp": datetime.now().isoformat(),
            "grouped_predictions_file": args.grouped_predictions_file,
            "model_info": {
                "provider": provider,
                "model": model,
                "reasoning": reasoning
            },
            "bag_evaluations": bag_evaluations
        }, f, indent=2)
    
    print(f"Saved evaluation results to: {eval_file}")
    
    # Save metrics summary
    metrics_file = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            "standard_metrics": metrics["standard"],
            "with_novel_metrics": metrics["with_novel"],
            "comparison": {
                "accuracy_improvement": metrics["with_novel"]["overall_metrics"]["overall_accuracy"] - 
                                       metrics["standard"]["overall_metrics"]["overall_accuracy"]
            }
        }, f, indent=2)
    
    print(f"Saved metrics to: {metrics_file}")
    
    # Print summary for both metric types
    for metric_type in ["standard", "with_novel"]:
        print("\n" + "="*60)
        print(f"📊 EVALUATION SUMMARY ({metric_type.upper().replace('_', ' ')})")
        print("="*60)
        
        overall = metrics[metric_type]["overall_metrics"]
        print(f"\n🎯 Overall Performance:")
        print(f"  - Total bags evaluated: {overall['total_bags']}")
        print(f"  - Overall accuracy: {overall['overall_accuracy']:.2%}")
        print(f"  - Correct bags: {overall['correct_bags']}/{overall['total_bags']}")
        
        print(f"\n📈 By Group Type:")
        print(f"  - Correct-only groups: {overall['correct_only_accuracy']:.2%} ({overall['correct_only_count']} groups)")
        print(f"    → Evaluation: ALL samples must have NO predictions")
        print(f"  - Misconception groups: {overall['misconception_accuracy']:.2%} ({overall['misconception_count']} groups)")
        print(f"    → Evaluation: matches > non-matches (ignoring NONEs)")
    
        # Add breakdown of misconception group results (only for standard, as it's the same info)
        if metric_type == "standard":
            misconception_breakdown = {"matches_win": 0, "non_matches_win": 0, "tie": 0}
            for eval_result in bag_evaluations:
                if eval_result["group_type"] == "misconception":
                    match_count = eval_result[metric_type].get("match_count", 0)
                    non_match_count = eval_result[metric_type].get("non_match_count", 0)
                    if match_count > non_match_count:
                        misconception_breakdown["matches_win"] += 1
                    elif non_match_count > match_count:
                        misconception_breakdown["non_matches_win"] += 1
                    else:
                        misconception_breakdown["tie"] += 1
            
            print(f"\n📊 Misconception Groups Breakdown:")
            print(f"  - Matches > Non-matches: {misconception_breakdown['matches_win']} groups")
            print(f"  - Non-matches > Matches: {misconception_breakdown['non_matches_win']} groups")
            print(f"  - Tie (equal counts): {misconception_breakdown['tie']} groups")
        
        print(f"\n🏆 Top 5 Best Performing Misconceptions:")
        for i, misc in enumerate(metrics[metric_type]["top_5_best_misconceptions"], 1):
            print(f"  {i}. Misconception {misc['misconception_id']}: {misc['accuracy']:.2%} ({misc['correct_groups']}/{misc['total_groups']})")
        
        print(f"\n⚠️  Top 5 Worst Performing Misconceptions:")
        for i, misc in enumerate(metrics[metric_type]["top_5_worst_misconceptions"], 1):
            print(f"  {i}. Misconception {misc['misconception_id']}: {misc['accuracy']:.2%} ({misc['correct_groups']}/{misc['total_groups']})")
    
    print("\n✅ Evaluation completed successfully!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 