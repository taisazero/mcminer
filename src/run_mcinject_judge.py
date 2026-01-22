#!/usr/bin/env python3
"""
Script to evaluate the quality of corrupted code samples by checking if they exhibit their intended misconceptions.

This script:
1. Loads corrupted code samples from a specified directory
2. Loads misconceptions from misconception.json
3. Uses Claude Sonnet 4.5 with reasoning to check each code sample
4. Computes data quality metrics
5. Optionally filters out codes that don't exhibit misconceptions (converts to "NONE")

Usage:
    # Evaluation only
    python run_mcinject_judge.py --corrupted-codes-dir mining_misconceptions/data/corrupted_codes/corrupted_codes_best_sample
    
    # Evaluation with filtering
    python run_mcinject_judge.py --corrupted-codes-dir data/corrupted_codes/full --apply-filtering --filtered-output-dir data/corrupted_codes/filtered
    
    # Use simple prompt (no rationale) for faster/cheaper evaluation
    python run_mcinject_judge.py --corrupted-codes-dir results/ --skip-rationale --use-batch
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_clients import AnthropicClient


def load_json_file(file_path: str) -> Any:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_prompt_template(template_path: str) -> str:
    """Load the prompt template from file."""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def create_evaluation_prompt(template: str, misconception: Dict[str, Any], code: str) -> str:
    """Create a specific prompt by filling in the template."""
    prompt = template
    
    # Replace placeholders
    prompt = prompt.replace("{misconception_description}", misconception["description"])
    prompt = prompt.replace("{misconception_example}", misconception.get("example", "No example provided"))
    prompt = prompt.replace("{code_to_analyze}", code)
    
    return prompt


def parse_evaluation_response(response: str, skip_rationale: bool = False) -> Dict[str, Any]:
    """Parse the evaluation response and extract the answer.
    
    Args:
        response: The LLM response text
        skip_rationale: If True, don't require rationale for parse_success
    """
    result = {
        "exhibits_misconception": None,
        "reasoning": "",
        "rationale": "",
        "feedback": "",
        "confidence": "",
        "misconception_type": None,  # benign or harmful
        "raw_response": response,
        "parse_success": False
    }
    
    try:
        # First, try to extract reasoning from the full response (it appears before <answer>)
        reasoning_match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', response, re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        
        # Extract answer section
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
        
        if answer_match:
            answer_content = answer_match.group(1)
            
            # Extract exhibits_misconception
            exhibits_match = re.search(r'<exhibits_misconception>\s*([YN])\s*</exhibits_misconception>', answer_content)
            if exhibits_match:
                result["exhibits_misconception"] = exhibits_match.group(1).upper() == 'Y'
            
            # If reasoning wasn't found in full response, try within answer section (backward compatibility)
            if not result["reasoning"]:
                reasoning_match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', answer_content, re.DOTALL)
                if reasoning_match:
                    result["reasoning"] = reasoning_match.group(1).strip()
            
            # Extract rationale (optional field)
            rationale_match = re.search(r'<rationale>\s*(.*?)\s*</rationale>', answer_content, re.DOTALL)
            if rationale_match:
                result["rationale"] = rationale_match.group(1).strip()
            
            # Extract feedback (optional field)
            feedback_match = re.search(r'<feedback>\s*(.*?)\s*</feedback>', answer_content, re.DOTALL)
            if feedback_match:
                result["feedback"] = feedback_match.group(1).strip()
            
            # Extract confidence
            confidence_match = re.search(r'<confidence>\s*(high|medium|low)\s*</confidence>', answer_content)
            if confidence_match:
                result["confidence"] = confidence_match.group(1)
            
            # Try to extract misconception type from reasoning or rationale
            text_to_check = (result["reasoning"] + " " + result["rationale"]).lower()
            if text_to_check:
                if "benign" in text_to_check:
                    result["misconception_type"] = "benign"
                elif "harmful" in text_to_check:
                    result["misconception_type"] = "harmful"
            
            # Check if we got the essential parts
            if skip_rationale:
                # For simple prompt: only need exhibits_misconception and confidence
                if result["exhibits_misconception"] is not None and result["confidence"]:
                    result["parse_success"] = True
            else:
                # For full prompt: rationale is required
                if result["exhibits_misconception"] is not None and result["rationale"]:
                    result["parse_success"] = True
                # Backward compatibility: accept reasoning if rationale not present
                elif result["exhibits_misconception"] is not None and result["reasoning"]:
                    result["parse_success"] = True
                
    except Exception as e:
        result["parse_error"] = str(e)
        print(f"Warning: Failed to parse evaluation response: {e}")
    
    return result


def collect_corrupted_code_samples(corrupted_codes_dir: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """Collect all corrupted code samples from the directory."""
    samples = []
    
    # Get all JSON files in the directory
    json_files = sorted(Path(corrupted_codes_dir).glob("problem_*_misc_*.json"))
    
    if max_samples:
        json_files = json_files[:max_samples]
    
    print(f"Found {len(json_files)} corrupted code files")
    
    for json_file in json_files:
        try:
            data = load_json_file(json_file)
            
            # Extract samples from each solution
            for solution in data.get("solutions", []):
                if solution.get("generated_code") and solution["generated_code"] != "NONE":
                    sample = {
                        "file_name": json_file.name,
                        "problem_id": data["problem_id"],
                        "misconception_id": data["misconception_id"],
                        "misconception_description": data["misconception_description"],
                        "solution_index": solution["solution_index"],
                        "code": solution["generated_code"],
                        "generation_metadata": solution.get("metadata", {})
                    }
                    samples.append(sample)
                    
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Collected {len(samples)} non-NONE code samples")
    return samples


def evaluate_samples_batch(samples: List[Dict[str, Any]], misconceptions: Dict[int, Dict[str, Any]], 
                          template: str, client: AnthropicClient, skip_rationale: bool = False) -> List[Dict[str, Any]]:
    """Evaluate samples using batch processing."""
    print("Preparing batch evaluation...")
    
    # Prepare batch messages
    batch_messages = []
    metadata_list = []
    
    for sample in samples:
        misconception_id = sample["misconception_id"]
        misconception = misconceptions.get(misconception_id)
        
        if not misconception:
            print(f"Warning: Misconception {misconception_id} not found")
            continue
        
        # Create prompt
        prompt = create_evaluation_prompt(template, misconception, sample["code"])
        messages = [{"role": "user", "content": prompt}]
        
        batch_messages.append(messages)
        metadata_list.append(sample)
    
    # Process batch with reasoning enabled (unless using simple prompt)
    print(f"Submitting batch of {len(batch_messages)} evaluations to Claude Sonnet 4.5...")
    responses = client.create_batch_messages(
        batch_messages,
        reasoning=not skip_rationale,  # Skip reasoning for simple prompt
        budget_tokens=2000 if not skip_rationale else 1000,
        model="claude-sonnet-4-5",
        temperature=0.1,
        max_tokens=2000 if not skip_rationale else 1000
    )
    
    # Parse responses and combine with metadata
    results = []
    for sample, response in zip(metadata_list, responses):
        parsed = parse_evaluation_response(response, skip_rationale=skip_rationale)
        result = {**sample, "evaluation": parsed}
        results.append(result)
    
    return results


def evaluate_samples_individual(samples: List[Dict[str, Any]], misconceptions: Dict[int, Dict[str, Any]], 
                               template: str, client: AnthropicClient, skip_rationale: bool = False) -> List[Dict[str, Any]]:
    """Evaluate samples individually."""
    results = []
    
    for sample in tqdm(samples, desc="Evaluating samples"):
        misconception_id = sample["misconception_id"]
        misconception = misconceptions.get(misconception_id)
        
        if not misconception:
            print(f"Warning: Misconception {misconception_id} not found")
            continue
        
        # Create prompt
        prompt = create_evaluation_prompt(template, misconception, sample["code"])
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Call Claude with reasoning enabled (unless using simple prompt)
            max_tok = 2000 if not skip_rationale else 1000
            budget = 2000 if not skip_rationale else 1000
            
            response = client.create_message(
                messages,
                kwargs={
                    "model": "claude-sonnet-4-5",
                    "temperature": 0.1,
                    "max_tokens": max_tok
                },
                reasoning=not skip_rationale,
                budget_tokens=budget
            )
            
            # Parse response
            parsed = parse_evaluation_response(response, skip_rationale=skip_rationale)
            
        except Exception as e:
            print(f"Error evaluating sample: {e}")
            parsed = {
                "exhibits_misconception": None,
                "reasoning": f"Error: {e}",
                "confidence": "low",
                "raw_response": str(e),
                "parse_success": False
            }
        
        result = {**sample, "evaluation": parsed}
        results.append(result)
    
    return results


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute data quality metrics from evaluation results."""
    total_samples = len(results)
    successful_parses = sum(1 for r in results if r["evaluation"]["parse_success"])
    exhibits_misconception = sum(1 for r in results 
                                if r["evaluation"]["parse_success"] 
                                and r["evaluation"]["exhibits_misconception"])
    
    # Count benign vs harmful
    benign_count = sum(1 for r in results 
                      if r["evaluation"]["parse_success"] 
                      and r["evaluation"]["exhibits_misconception"]
                      and r["evaluation"].get("misconception_type") == "benign")
    harmful_count = sum(1 for r in results 
                       if r["evaluation"]["parse_success"] 
                       and r["evaluation"]["exhibits_misconception"]
                       and r["evaluation"].get("misconception_type") == "harmful")
    unclassified_count = exhibits_misconception - benign_count - harmful_count
    
    # Group by misconception
    by_misconception = defaultdict(lambda: {"total": 0, "exhibits": 0, "parse_success": 0, "benign": 0, "harmful": 0})
    
    for result in results:
        misc_id = result["misconception_id"]
        by_misconception[misc_id]["total"] += 1
        
        if result["evaluation"]["parse_success"]:
            by_misconception[misc_id]["parse_success"] += 1
            if result["evaluation"]["exhibits_misconception"]:
                by_misconception[misc_id]["exhibits"] += 1
                
                # Track type
                misc_type = result["evaluation"].get("misconception_type")
                if misc_type == "benign":
                    by_misconception[misc_id]["benign"] += 1
                elif misc_type == "harmful":
                    by_misconception[misc_id]["harmful"] += 1
    
    # Calculate per-misconception metrics
    misconception_metrics = {}
    for misc_id, stats in by_misconception.items():
        if stats["parse_success"] > 0:
            exhibit_rate = stats["exhibits"] / stats["parse_success"]
        else:
            exhibit_rate = 0.0
            
        misconception_metrics[misc_id] = {
            "total_samples": stats["total"],
            "successful_evaluations": stats["parse_success"],
            "exhibits_misconception": stats["exhibits"],
            "exhibit_rate": exhibit_rate,
            "benign_count": stats["benign"],
            "harmful_count": stats["harmful"]
        }
    
    # Calculate overall metrics
    if successful_parses > 0:
        overall_exhibit_rate = exhibits_misconception / successful_parses
    else:
        overall_exhibit_rate = 0.0
    
    # Group by confidence levels
    confidence_dist = defaultdict(int)
    for result in results:
        if result["evaluation"]["parse_success"]:
            confidence = result["evaluation"]["confidence"]
            confidence_dist[confidence] += 1
    
    metrics = {
        "total_samples": total_samples,
        "successful_evaluations": successful_parses,
        "parse_success_rate": successful_parses / total_samples if total_samples > 0 else 0.0,
        "samples_exhibiting_misconception": exhibits_misconception,
        "overall_exhibit_rate": overall_exhibit_rate,
        "benign_misconceptions": benign_count,
        "harmful_misconceptions": harmful_count,
        "unclassified_misconceptions": unclassified_count,
        "confidence_distribution": dict(confidence_dist),
        "per_misconception_metrics": misconception_metrics
    }
    
    return metrics


def save_results(results: List[Dict[str, Any]], metrics: Dict[str, Any], output_file: str):
    """Save evaluation results and metrics to file."""
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "detailed_results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")


def print_metrics_summary(metrics: Dict[str, Any]):
    """Print a summary of the metrics."""
    print("\n" + "="*60)
    print("EVALUATION METRICS SUMMARY")
    print("="*60)
    
    print(f"\nTotal code samples evaluated: {metrics['total_samples']}")
    print(f"Successful evaluations: {metrics['successful_evaluations']} ({metrics['parse_success_rate']:.1%})")
    print(f"\nSamples exhibiting their intended misconception: {metrics['samples_exhibiting_misconception']}")
    print(f"Overall exhibit rate: {metrics['overall_exhibit_rate']:.1%}")
    
    print("\nMisconception type breakdown:")
    print(f"  Benign (working but suboptimal): {metrics['benign_misconceptions']}")
    print(f"  Harmful (causing errors): {metrics['harmful_misconceptions']}")
    print(f"  Unclassified: {metrics['unclassified_misconceptions']}")
    
    print("\nConfidence distribution:")
    for level, count in sorted(metrics['confidence_distribution'].items()):
        print(f"  {level}: {count}")
    
    print("\nTop 10 misconceptions by exhibit rate:")
    sorted_misconceptions = sorted(
        metrics['per_misconception_metrics'].items(),
        key=lambda x: x[1]['exhibit_rate'],
        reverse=True
    )[:10]
    
    for misc_id, stats in sorted_misconceptions:
        type_info = f" (B:{stats['benign_count']} H:{stats['harmful_count']})" if stats['benign_count'] + stats['harmful_count'] > 0 else ""
        print(f"  Misconception {misc_id}: {stats['exhibit_rate']:.1%} "
              f"({stats['exhibits_misconception']}/{stats['successful_evaluations']}){type_info}")
    
    print("\nBottom 10 misconceptions by exhibit rate:")
    bottom_misconceptions = sorted(
        metrics['per_misconception_metrics'].items(),
        key=lambda x: x[1]['exhibit_rate']
    )[:10]
    
    for misc_id, stats in bottom_misconceptions:
        type_info = f" (B:{stats['benign_count']} H:{stats['harmful_count']})" if stats['benign_count'] + stats['harmful_count'] > 0 else ""
        print(f"  Misconception {misc_id}: {stats['exhibit_rate']:.1%} "
              f"({stats['exhibits_misconception']}/{stats['successful_evaluations']}){type_info}")
    
    print("="*60)


def should_filter_sample(evaluation: Dict[str, Any], filter_low_confidence: bool) -> Tuple[bool, str]:
    """Determine if a sample should be filtered based on evaluation results.
    
    Returns:
        (should_filter, reason) tuple
    """
    if not evaluation.get("parse_success"):
        return False, ""
    
    # Default filtering: exhibits_misconception == False
    if not evaluation.get("exhibits_misconception"):
        return True, "exhibits_misconception: False"
    
    # Optional filtering: low confidence exhibits
    if filter_low_confidence and evaluation.get("confidence") == "low":
        return True, "exhibits_misconception: True, confidence: low"
    
    return False, ""


def apply_filtering_to_files(results: List[Dict[str, Any]], corrupted_codes_dir: str, 
                             filtered_output_dir: str, filter_low_confidence: bool) -> Dict[str, Any]:
    """Apply filtering to original JSON files and save to output directory.
    
    Args:
        results: Evaluation results
        corrupted_codes_dir: Input directory
        filtered_output_dir: Output directory for filtered files
        filter_low_confidence: Whether to also filter low-confidence exhibits
    
    Returns:
        Filtering statistics
    """
    from pathlib import Path
    import shutil
    
    # Create output directory
    os.makedirs(filtered_output_dir, exist_ok=True)
    
    # Group results by file
    results_by_file = defaultdict(list)
    for result in results:
        results_by_file[result["file_name"]].append(result)
    
    stats = {
        "total_files": 0,
        "total_samples": 0,
        "samples_filtered": 0,
        "filter_reasons": defaultdict(int),
        "per_misconception": defaultdict(lambda: {"total": 0, "filtered": 0})
    }
    
    print(f"\n🔄 Applying filtering to {len(results_by_file)} files...")
    
    for file_name, file_results in tqdm(results_by_file.items(), desc="Filtering files"):
        # Load original file
        input_path = Path(corrupted_codes_dir) / file_name
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {file_name}: {e}")
            continue
        
        stats["total_files"] += 1
        
        # Create lookup of evaluations by solution_index
        evaluations = {r["solution_index"]: r["evaluation"] for r in file_results}
        
        # Apply filtering to solutions
        for solution in data.get("solutions", []):
            stats["total_samples"] += 1
            misc_id = data["misconception_id"]
            stats["per_misconception"][misc_id]["total"] += 1
            
            solution_idx = solution["solution_index"]
            evaluation = evaluations.get(solution_idx)
            
            if evaluation:
                should_filter, reason = should_filter_sample(evaluation, filter_low_confidence)
                
                if should_filter:
                    # Convert to NONE with metadata
                    original_code = solution.get("generated_code", "")
                    solution["generated_code"] = "NONE"
                    
                    # Add filtering metadata
                    if "metadata" not in solution:
                        solution["metadata"] = {}
                    
                    solution["metadata"]["filtered"] = True
                    solution["metadata"]["filter_reason"] = reason
                    solution["metadata"]["filter_timestamp"] = datetime.now().isoformat()
                    solution["metadata"]["original_evaluation"] = evaluation
                    solution["metadata"]["original_code_length"] = len(original_code)
                    
                    stats["samples_filtered"] += 1
                    stats["filter_reasons"][reason] += 1
                    stats["per_misconception"][misc_id]["filtered"] += 1
        
        # Save filtered file
        output_path = Path(filtered_output_dir) / file_name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Calculate rates
    stats["filter_rate"] = stats["samples_filtered"] / stats["total_samples"] if stats["total_samples"] > 0 else 0.0
    
    return dict(stats)


def save_filtering_report(stats: Dict[str, Any], output_dir: str, filter_low_confidence: bool):
    """Save filtering report to JSON file."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "filter_criteria": {
            "default": "exhibits_misconception == False",
            "low_confidence_enabled": filter_low_confidence
        },
        "statistics": stats
    }
    
    report_path = Path(output_dir) / "filtering_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Filtering report saved to {report_path}")


def print_filtering_summary(stats: Dict[str, Any]):
    """Print filtering statistics."""
    print("\n" + "="*60)
    print("FILTERING SUMMARY")
    print("="*60)
    print(f"\nTotal files processed: {stats['total_files']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Samples filtered: {stats['samples_filtered']} ({stats['filter_rate']:.1%})")
    
    print("\nFilter reasons:")
    for reason, count in sorted(stats['filter_reasons'].items()):
        print(f"  {reason}: {count}")
    
    print("\nTop misconceptions by filter rate:")
    misc_stats = [(k, v) for k, v in stats['per_misconception'].items() if v['total'] > 0]
    misc_stats.sort(key=lambda x: x[1]['filtered'] / x[1]['total'], reverse=True)
    
    for misc_id, ms in misc_stats[:10]:
        rate = ms['filtered'] / ms['total']
        print(f"  Misconception {misc_id}: {rate:.1%} ({ms['filtered']}/{ms['total']})")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate quality of corrupted code samples")
    
    # Data paths
    parser.add_argument("--corrupted-codes-dir", 
                       default="mining_misconceptions/data/corrupted_codes/corrupted_codes_best_sample",
                       help="Directory containing corrupted code JSON files")
    parser.add_argument("--misconceptions-file",
                       default="mining_misconceptions/data/misconception.json",
                       help="Path to misconceptions JSON file")
    parser.add_argument("--prompt-template",
                       default="mining_misconceptions/prompt_templates/evaluation/check_misconception_exhibited.md",
                       help="Path to evaluation prompt template")
    
    # Output
    parser.add_argument("--output-file", default="injection_evaluation_results.json",
                       help="Output file for evaluation results")
    
    # Processing options
    parser.add_argument("--use-batch", action="store_true",
                       help="Use batch processing (faster)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Limit number of files to process (for testing)")
    parser.add_argument("--skip-rationale", action="store_true",
                       help="Use simple prompt without rationale (faster, cheaper)")
    
    # Filtering options
    parser.add_argument("--apply-filtering", action="store_true",
                       help="Filter corrupted codes that don't exhibit misconceptions")
    parser.add_argument("--filter-low-confidence", action="store_true",
                       help="Also filter samples with exhibits=True but confidence=low")
    parser.add_argument("--filtered-output-dir",
                       help="Output directory for filtered corrupted codes (required if --apply-filtering)")
    
    args = parser.parse_args()
    
    # Validation
    if args.apply_filtering and not args.filtered_output_dir:
        parser.error("--filtered-output-dir is required when --apply-filtering is enabled")
    
    # Auto-select simple prompt if skip-rationale is set
    if args.skip_rationale and args.prompt_template == "mining_misconceptions/prompt_templates/evaluation/check_misconception_exhibited.md":
        args.prompt_template = "mining_misconceptions/prompt_templates/evaluation/check_misconception_exhibited_simple.md"
        print("ℹ️  Using simple prompt template (no rationale required)")
    
    # Load data
    print("Loading data...")
    try:
        misconceptions_data = load_json_file(args.misconceptions_file)
        template = load_prompt_template(args.prompt_template)
        
        # Convert misconceptions to dict keyed by ID
        misconceptions = {m["id"]: m for m in misconceptions_data}
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    print(f"Loaded {len(misconceptions)} misconceptions")
    
    # Collect corrupted code samples
    samples = collect_corrupted_code_samples(args.corrupted_codes_dir, args.max_samples)
    
    if not samples:
        print("No code samples found to evaluate")
        return 1
    
    # Create Anthropic client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return 1
    
    client = AnthropicClient(api_key=api_key)
    reasoning_status = "disabled" if args.skip_rationale else "enabled"
    print(f"Initialized Claude Sonnet 4.5 client with reasoning {reasoning_status}")
    
    # Evaluate samples
    if args.use_batch:
        print(f"Evaluating {len(samples)} samples using batch processing...")
        results = evaluate_samples_batch(samples, misconceptions, template, client, 
                                         skip_rationale=args.skip_rationale)
    else:
        print(f"Evaluating {len(samples)} samples individually...")
        results = evaluate_samples_individual(samples, misconceptions, template, client,
                                             skip_rationale=args.skip_rationale)
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Save evaluation results
    save_results(results, metrics, args.output_file)
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Apply filtering if requested
    if args.apply_filtering:
        print(f"\n🔍 Applying filtering to corrupted codes...")
        if args.filter_low_confidence:
            print("   Filtering: exhibits=False OR (exhibits=True AND confidence=low)")
        else:
            print("   Filtering: exhibits=False only")
        
        filter_stats = apply_filtering_to_files(
            results, 
            args.corrupted_codes_dir, 
            args.filtered_output_dir,
            args.filter_low_confidence
        )
        
        # Save filtering report
        save_filtering_report(filter_stats, args.filtered_output_dir, args.filter_low_confidence)
        
        # Print filtering summary
        print_filtering_summary(filter_stats)
    
    print(f"\n✅ Evaluation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 