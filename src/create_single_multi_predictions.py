#!/usr/bin/env python3
"""
Script to create grouped predictions by combining multi-group structures with corresponding single predictions.

This script:
1. Loads multi predictions (bag-of-code analysis)
2. Finds corresponding single predictions for each code in the bag
3. Creates a new structure that maintains groupings for evaluation

Usage:
    python create_single_multi_predictions.py \
        --multi-predictions-file test_results/openai_o3-mini_reasoning/multi_old/multi_predictions.json \
        --single-predictions-dir test_results/openai_o3-mini_reasoning/single \
        --output-file test_results/openai_o3-mini_reasoning/single_multi_predictions.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict

def load_json_file(file_path: str) -> Any:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_single_prediction_correct(single_predictions: List[Dict], problem_id: int) -> Optional[Dict]:
    """
    Find a single prediction for a correct code sample.
    Looking for ground_truth_misconception.id == "NONE" and matching problem_id.
    """
    for pred in single_predictions:
        if (pred.get("problem_id") == problem_id and 
            pred.get("ground_truth_misconception", {}).get("id") == "NONE"):
            return pred
    return None

def find_single_prediction_misconception(single_predictions: List[Dict], 
                                       misconception_id: int, 
                                       problem_id: int) -> Optional[Dict]:
    """
    Find a single prediction for a misconception code sample.
    Match by both misconception_id and problem_id.
    """
    for pred in single_predictions:
        # Check original_misconception field (common in single predictions)
        orig_misc = pred.get("original_misconception", {})
        if (orig_misc.get("id") == misconception_id and 
            pred.get("problem_id") == problem_id):
            return pred
        
        # Also check ground_truth_misconception field
        gt_misc = pred.get("ground_truth_misconception", {})
        if (gt_misc.get("id") == misconception_id and 
            pred.get("problem_id") == problem_id):
            return pred
    
    return None

def create_grouped_predictions(multi_predictions: List[Dict], 
                             single_predictions: List[Dict]) -> List[Dict]:
    """
    Create grouped predictions by matching multi groups with single predictions.
    """
    grouped_predictions = []
    missing_predictions = []
    
    # Create lookup indices for faster searching
    print("Creating lookup indices for single predictions...")
    single_by_problem = defaultdict(list)
    single_by_misc_problem = defaultdict(list)
    
    for pred in single_predictions:
        problem_id = pred.get("problem_id")
        if problem_id is not None:
            single_by_problem[problem_id].append(pred)
            
            # Index by misconception + problem
            orig_misc_id = pred.get("original_misconception", {}).get("id")
            if orig_misc_id is not None:
                key = (orig_misc_id, problem_id)
                single_by_misc_problem[key].append(pred)
    
    print(f"\nProcessing {len(multi_predictions)} multi-group predictions...")
    
    for multi_pred in multi_predictions:
        group_info = multi_pred.get("group_info", {})
        group_type = multi_pred.get("group_type", "unknown")
        problem_ids = group_info.get("problem_ids", [])
        gt_misconception = group_info.get("gt_misconception")
        
        # Create new grouped prediction
        grouped_pred = {
            "group_id": multi_pred.get("prediction_id"),
            "group_type": group_type,
            "multi_prediction": multi_pred,  # Keep original multi prediction
            "single_predictions": [],
            "group_info": {
                "num_codes": group_info.get("num_codes", 0),
                "problem_ids": problem_ids,
                "gt_misconception": gt_misconception,
                "source_files": group_info.get("source_files", [])
            }
        }
        
        # Find corresponding single predictions
        if group_type == "correct_only":
            # For correct-only groups, find predictions with NONE misconception
            for problem_id in problem_ids:
                candidates = single_by_problem.get(problem_id, [])
                found = False
                for cand in candidates:
                    if cand.get("ground_truth_misconception", {}).get("id") == "NONE":
                        grouped_pred["single_predictions"].append({
                            "prediction_id": cand.get("prediction_id"),
                            "problem_id": problem_id,
                            "predicted_misconceptions": cand.get("predicted_misconceptions", []),
                            "no_predicted_misconceptions": cand.get("no_predicted_misconceptions", False),
                            "parse_success": cand.get("parse_success", False),
                            "source_file": cand.get("source_file", "")
                        })
                        found = True
                        break
                
                if not found:
                    missing_predictions.append({
                        "group_id": grouped_pred["group_id"],
                        "problem_id": problem_id,
                        "type": "correct_only"
                    })
        
        else:  # misconception group
            # Extract misconception ID
            if isinstance(gt_misconception, (int, str)) and gt_misconception != "NONE":
                misc_id = int(gt_misconception) if isinstance(gt_misconception, str) else gt_misconception
                
                for problem_id in problem_ids:
                    key = (misc_id, problem_id)
                    candidates = single_by_misc_problem.get(key, [])
                    
                    if candidates:
                        cand = candidates[0]  # Take first match
                        grouped_pred["single_predictions"].append({
                            "prediction_id": cand.get("prediction_id"),
                            "problem_id": problem_id,
                            "misconception_id": misc_id,
                            "predicted_misconceptions": cand.get("predicted_misconceptions", []),
                            "no_predicted_misconceptions": cand.get("no_predicted_misconceptions", False),
                            "parse_success": cand.get("parse_success", False),
                            "source_file": cand.get("source_file", "")
                        })
                    else:
                        missing_predictions.append({
                            "group_id": grouped_pred["group_id"],
                            "problem_id": problem_id,
                            "misconception_id": misc_id,
                            "type": "misconception"
                        })
        
        grouped_predictions.append(grouped_pred)
    
    # Report missing predictions
    if missing_predictions:
        print(f"\n⚠️  WARNING: {len(missing_predictions)} missing single predictions:")
        for miss in missing_predictions[:10]:  # Show first 10
            print(f"  - Group: {miss['group_id']}, Problem: {miss['problem_id']}, Type: {miss['type']}")
        if len(missing_predictions) > 10:
            print(f"  ... and {len(missing_predictions) - 10} more")
    
    return grouped_predictions

def generate_summary_stats(grouped_predictions: List[Dict]) -> Dict[str, Any]:
    """Generate summary statistics for the grouped predictions."""
    stats = {
        "total_groups": len(grouped_predictions),
        "correct_only_groups": 0,
        "misconception_groups": 0,
        "total_single_predictions": 0,
        "groups_with_complete_singles": 0,
        "groups_with_missing_singles": 0,
        "coverage_by_group_type": {},
        "misconception_distribution": defaultdict(int)
    }
    
    for group in grouped_predictions:
        group_type = group["group_type"]
        expected_count = group["group_info"]["num_codes"]
        actual_count = len(group["single_predictions"])
        
        if group_type == "correct_only":
            stats["correct_only_groups"] += 1
        else:
            stats["misconception_groups"] += 1
            misc_id = group["group_info"]["gt_misconception"]
            if misc_id != "NONE":
                stats["misconception_distribution"][str(misc_id)] += 1
        
        stats["total_single_predictions"] += actual_count
        
        if actual_count == expected_count:
            stats["groups_with_complete_singles"] += 1
        else:
            stats["groups_with_missing_singles"] += 1
        
        # Coverage by group type
        if group_type not in stats["coverage_by_group_type"]:
            stats["coverage_by_group_type"][group_type] = {
                "total_groups": 0,
                "total_expected_singles": 0,
                "total_found_singles": 0
            }
        
        stats["coverage_by_group_type"][group_type]["total_groups"] += 1
        stats["coverage_by_group_type"][group_type]["total_expected_singles"] += expected_count
        stats["coverage_by_group_type"][group_type]["total_found_singles"] += actual_count
    
    # Convert defaultdict to regular dict
    stats["misconception_distribution"] = dict(stats["misconception_distribution"])
    
    return stats

def main():
    parser = argparse.ArgumentParser(
        description="Create grouped predictions from multi and single prediction files"
    )
    
    parser.add_argument(
        "--multi-predictions-file",
        required=True,
        help="Path to multi predictions JSON file"
    )
    parser.add_argument(
        "--single-predictions-dir",
        required=True,
        help="Directory containing single predictions.json file"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output path for grouped predictions JSON file"
    )
    parser.add_argument(
        "--pretty-print",
        action="store_true",
        help="Pretty print the output JSON"
    )
    
    args = parser.parse_args()
    
    # Load multi predictions
    print(f"Loading multi predictions from: {args.multi_predictions_file}")
    multi_predictions = load_json_file(args.multi_predictions_file)
    print(f"  Loaded {len(multi_predictions)} multi-group predictions")
    
    # Load single predictions
    single_pred_file = os.path.join(args.single_predictions_dir, "predictions.json")
    print(f"\nLoading single predictions from: {single_pred_file}")
    
    if not os.path.exists(single_pred_file):
        print(f"ERROR: Single predictions file not found: {single_pred_file}")
        return 1
    
    single_predictions = load_json_file(single_pred_file)
    print(f"  Loaded {len(single_predictions)} single predictions")
    
    # Create grouped predictions
    print("\nCreating grouped predictions...")
    grouped_predictions = create_grouped_predictions(multi_predictions, single_predictions)
    
    # Generate summary statistics
    stats = generate_summary_stats(grouped_predictions)
    
    # Save output
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving grouped predictions to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        if args.pretty_print:
            json.dump(grouped_predictions, f, indent=2)
        else:
            json.dump(grouped_predictions, f)
    
    # Save summary stats
    stats_file = args.output_file.replace('.json', '_summary.json')
    print(f"Saving summary statistics to: {stats_file}")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n📊 Summary Statistics:")
    print(f"  Total groups: {stats['total_groups']}")
    print(f"    - Correct-only groups: {stats['correct_only_groups']}")
    print(f"    - Misconception groups: {stats['misconception_groups']}")
    print(f"  Total single predictions matched: {stats['total_single_predictions']}")
    print(f"  Groups with complete coverage: {stats['groups_with_complete_singles']}")
    print(f"  Groups with missing singles: {stats['groups_with_missing_singles']}")
    
    print("\n  Coverage by group type:")
    for gtype, cov in stats['coverage_by_group_type'].items():
        coverage_pct = (cov['total_found_singles'] / cov['total_expected_singles'] * 100) if cov['total_expected_singles'] > 0 else 0
        print(f"    - {gtype}: {cov['total_found_singles']}/{cov['total_expected_singles']} ({coverage_pct:.1f}%)")
    
    print("\n✅ Grouped predictions created successfully!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 