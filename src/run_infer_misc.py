#!/usr/bin/env python3
"""
Script to infer misconceptions from corrupted Python code using LLMs.

This script:
1. Loads corrupted codes from a specified directory
2. Loads problems from problems_processed.json for context
3. Uses LLMs to predict what misconceptions are present in the code
4. Saves predictions as JSON files

Usage:
    # Individual processing (default)
    python run_infer_misc.py --llm anthropic --input-dir data/corrupted_codes/corrupted_codes_anthropic --output-dir results/inferred_misconceptions/
    python run_infer_misc.py --llm openai --openai-model gpt-4o
    python run_infer_misc.py --llm gemini --gemini-model gemini-2.5-pro-preview-06-05
    python run_infer_misc.py --llm vllm --vllm-base-url http://localhost:8000/v1
    
    # Template selection
    python run_infer_misc.py --template zeroshot-no-reasoning  # Direct template specification
    python run_infer_misc.py --no-reasoning-template  # Convenience flag for no-reasoning variant
    python run_infer_misc.py --template-dir prompt_templates/custom  # Custom template directory
    
    # Reasoning mode (for compatible models only)
    python run_infer_misc.py --llm openai --openai-model o3-mini --reasoning --reasoning-effort high
    python run_infer_misc.py --llm anthropic --anthropic-model claude-3-7-sonnet-latest --reasoning
    
    # Batch processing (faster & cheaper for large datasets)
    python run_infer_misc.py --llm anthropic --use-batch
    python run_infer_misc.py --llm openai --use-batch
    
    # NONE-only processing (for previously skipped samples)
    python run_infer_misc.py --llm anthropic --none-only --append-results --output-dir existing_results/
    python run_infer_misc.py --llm openai --none-only --append-results  # Append NONE samples to existing predictions
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(override=True)
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.llm_clients import OpenAIClient, VLLMClient, AnthropicClient, GeminiClient


def load_prompt_template(template_name: str = "zeroshot", template_dir: str = "prompt_templates/mining") -> str:
    """Load mining prompt template from external markdown file."""
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Map template names to files
    template_files = {
        "zeroshot": "zeroshot.md",
        "zeroshot-no-reasoning": "zeroshot-no-reasoning.md",
        "fewshot": "fewshot.md",  # backward compatibility
        "fewshot-no-reasoning": "fewshot-no-reasoning.md"  # backward compatibility
    }
    
    if template_name not in template_files:
        raise ValueError(f"Unknown template name: {template_name}. Available: {list(template_files.keys())}")
    
    template_file = template_files[template_name]
    template_path = os.path.join(script_dir, template_dir, template_file)
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Remove markdown header if present (first line starting with #)
        lines = template_content.split('\n')
        if lines and lines[0].startswith('#'):
            lines = lines[1:]  # Remove header line
            if lines and lines[0].strip() == '':  # Remove empty line after header
                lines = lines[1:]
        
        return '\n'.join(lines)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Template file not found: {template_path}")


def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_corrupted_codes(input_dir: str) -> List[Dict[str, Any]]:
    """Load all corrupted code files from the input directory."""
    corrupted_codes = []
    
    # Look for JSON files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json') and filename not in ['summary.json', 'filtering_report.json']:
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                # Extract problem_id from filename if not in data
                if 'problem_id' not in data:
                    match = re.search(r'problem_(\d+)', filename)
                    if match:
                        data['problem_id'] = int(match.group(1))
                
                # Add filename for tracking
                data['source_file'] = filename
                corrupted_codes.append(data)
    
    return corrupted_codes


def create_mining_prompt(template: str, problem: Dict[str, Any], student_code: str) -> str:
    """Create a prompt for inferring misconceptions from code."""
    prompt = template
    
    # Replace placeholders
    prompt = prompt.replace("{problem_description}", problem.get("description", ""))
    prompt = prompt.replace("{problem_title}", f"Problem {problem.get('id', 'Unknown')}")
    prompt = prompt.replace("{student_code}", student_code)
    
    return prompt


def parse_mining_response(response: str) -> Dict[str, Any]:
    """Parse LLM response for inferred misconceptions."""
    result = {
        "reasoning": "",
        "misconceptions": [],
        "metadata": {},
        "raw_response": response,
        "parse_success": False,
        "analysis": ""
    }
    
    try:
        # Extract reasoning (if present)
        reasoning_match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', response, re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        
        # Extract misconception directly (no wrapper tag)
        misconception_match = re.search(r'<misconception>\s*(.*?)\s*</misconception>', response, re.DOTALL)
        if misconception_match:
            misconception_text = misconception_match.group(1).strip()
            
            if misconception_text.upper() == "NONE":
                result["misconceptions"] = []
                result["no_predicted_misconceptions"] = True
            else:
                misconception = {}
                
                # Extract fields (description and explanation plus metadata fields)
                desc_match = re.search(r'<description>\s*(.*?)\s*</description>', misconception_text, re.DOTALL)
                if desc_match:
                    misconception['description'] = desc_match.group(1).strip()
                
                explanation_match = re.search(r'<explanation>\s*(.*?)\s*</explanation>', misconception_text, re.DOTALL)
                if explanation_match:
                    misconception['explanation'] = explanation_match.group(1).strip()
                
                # Extract metadata fields (for consistency with multi-code version)
                # First check if they're in the misconception block
                type_match = re.search(r'<type>\s*(.*?)\s*</type>', misconception_text, re.DOTALL)
                if type_match:
                    misconception['misconception_type'] = type_match.group(1).strip()
                
                error_match = re.search(r'<error_type>\s*(.*?)\s*</error_type>', misconception_text, re.DOTALL)
                if error_match:
                    misconception['error_type'] = error_match.group(1).strip()
                    
                confidence_match = re.search(r'<confidence>\s*(.*?)\s*</confidence>', misconception_text, re.DOTALL)
                if confidence_match:
                    misconception['confidence'] = confidence_match.group(1).strip()
                
                if 'description' in misconception:  # Only add if we found at least a description
                    result["misconceptions"].append(misconception)
                
                result["no_predicted_misconceptions"] = len(result["misconceptions"]) == 0
        
        # Extract metadata
        metadata_match = re.search(r'<metadata>\s*(.*?)\s*</metadata>', response, re.DOTALL)
        if metadata_match:
            metadata_text = metadata_match.group(1).strip()
            result["metadata"]["raw"] = metadata_text
            
            # Parse simple key: value pairs
            for line in metadata_text.split('\n'):
                line = line.strip()
                if ':' in line and not line.startswith('#'):
                    key, value = line.split(':', 1)
                    result["metadata"][key.strip()] = value.strip()
            
            # If misconceptions exist but are missing metadata fields, copy from global metadata
            if result["misconceptions"]:
                for misconception in result["misconceptions"]:
                    if "misconception_type" not in misconception and "misconception_type" in result["metadata"]:
                        misconception["misconception_type"] = result["metadata"]["misconception_type"]
                    if "error_type" not in misconception and "error_type" in result["metadata"]:
                        misconception["error_type"] = result["metadata"]["error_type"]
                    if "confidence" not in misconception and "confidence_level" in result["metadata"]:
                        misconception["confidence"] = result["metadata"]["confidence_level"]
        
        # Check if parsing was successful
        if misconception_match:
            result["parse_success"] = True
            
        # Ensure no_predicted_misconceptions field is always present
        if "no_predicted_misconceptions" not in result:
            result["no_predicted_misconceptions"] = len(result["misconceptions"]) == 0
            
    except Exception as e:
        result["metadata"]["parse_error"] = str(e)
        print(f"Warning: Failed to parse LLM response: {e}")
    
    return result


def get_model_name(args, llm_client=None) -> str:
    """Get the model name being used."""
    if args.llm == "anthropic":
        return args.anthropic_model
    elif args.llm == "openai":
        return args.openai_model
    elif args.llm == "gemini":
        return args.gemini_model
    else:  # vllm
        if llm_client and hasattr(llm_client, 'model_name') and llm_client.model_name:
            return llm_client.model_name
        else:
            return "vllm-served-model"


def get_llm_kwargs(args) -> Dict[str, Any]:
    """Get LLM-specific kwargs based on the selected provider."""
    if args.llm == "anthropic":
        reasoning_supported = args.anthropic_model in ["claude-3-7-sonnet-latest", "claude-sonnet-4-0", "claude-opus-4-1", "claude-sonnet-4-5"]
        use_reasoning = args.reasoning and reasoning_supported
        return {
            "model": args.anthropic_model,
            "temperature": 0.1,
            "max_tokens": 4000,
            "reasoning": use_reasoning,
            "budget_tokens": 2000 if use_reasoning else 1000
        }
    elif args.llm == "openai":
        reasoning_supported = args.openai_model in ["o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"]
        use_reasoning = args.reasoning and reasoning_supported
        return {
            "model": args.openai_model,
            "temperature": 0.1,
            "max_tokens": 4000,
            "reasoning": use_reasoning,
            "reasoning_effort": args.reasoning_effort
        }
    elif args.llm == "gemini":
        return {
            "model": args.gemini_model,
            "temperature": 0.1,
            "max_tokens": 4000
        }
    else:  # vllm
        # Use Qwen3-recommended parameters if thinking mode is enabled
        if hasattr(args, 'vllm_enable_thinking') and args.vllm_enable_thinking:
            # Qwen3 thinking mode parameters
            # https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes
            return {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "max_tokens": 4000
            }
        else:
            # Qwen3 non-thinking or standard vLLM parameters
            return {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "max_tokens": 4000
            }


def create_llm_client(args) -> Any:
    """Create the appropriate LLM client based on arguments."""
    if args.llm == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return AnthropicClient(api_key=api_key)
    
    elif args.llm == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAIClient(api_key=api_key)
    
    elif args.llm == "vllm":
        if args.vllm_model:
            # Offline mode with direct model loading
            print(f"Initializing vLLM in offline mode with model: {args.vllm_model}")
            return VLLMClient(
                model=args.vllm_model,
                offline_mode=True,
                gpu_memory_utilization=0.9,
                max_model_len=32768 if '8b' in args.vllm_model.lower() else None
            )
        else:
            # Server mode (existing behavior)
            return VLLMClient(
                api_key=args.vllm_api_key or "NONE",
                base_url=args.vllm_base_url
            )
    
    elif args.llm == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        return GeminiClient(api_key=api_key, model=args.gemini_model)
    
    else:
        raise ValueError(f"Unsupported LLM: {args.llm}")


def generate_mining_batches(corrupted_codes: List[Dict[str, Any]], problems: Dict[str, Any], 
                              template: str, none_only: bool = False) -> List[Tuple[Dict[str, Any], List[Dict[str, str]]]]:
    """Generate batches of prompts for inferring misconceptions."""
    batches = []
    
    mode_str = "NONE-only" if none_only else "all"
    print(f"Preparing mining prompts ({mode_str} mode)...")
    
    for code_data in tqdm(corrupted_codes, desc="Processing corrupted codes"):
        problem_id = code_data.get('problem_id')
        
        # Skip if no problem_id
        if problem_id is None:
            continue
        
        # Get problem context
        problem = None
        
        # Handle dict format with string or int keys
        if isinstance(problems, dict):
            # Try string key first
            problem = problems.get(str(problem_id))
            # Try int key if string didn't work
            if not problem:
                problem = problems.get(problem_id)
        elif isinstance(problems, list):
            # Handle list format
            for p in problems:
                if p.get('id') == problem_id:
                    problem = p
                    break
        
        # If no problem found, create minimal context
        if not problem:
            problem = {'id': problem_id, 'description': f'Problem {problem_id}'}
        
        # Process each solution in the corrupted code file
        solutions = code_data.get('solutions', [])
        for sol_idx, solution in enumerate(solutions):
            generated_code = solution.get('generated_code', '')
            original_code = generated_code  # Store original value for metadata
            
            # Check if this is a NONE sample
            is_none_sample = not original_code or original_code == 'NONE'
            
            # Apply filtering based on mode
            if none_only and not is_none_sample:
                # Skip non-NONE samples when in NONE-only mode
                continue
            elif not none_only and is_none_sample:
                # Handle NONE cases by substituting with correct solution (existing behavior)
                if problem and 'solutions' in problem and problem['solutions']:
                    # Use the first correct solution as substitute
                    generated_code = problem['solutions'][0]
                    print(f"Substituting NONE with correct solution for problem {problem_id}, solution {sol_idx}")
                else:
                    # If no correct solution available, skip
                    print(f"Warning: No correct solution available for problem {problem_id}, skipping NONE sample")
                    continue
            elif none_only and is_none_sample:
                # Process NONE samples in NONE-only mode - substitute with correct solution
                if problem and 'solutions' in problem and problem['solutions']:
                    generated_code = problem['solutions'][0]
                    print(f"Processing NONE sample with correct solution for problem {problem_id}, solution {sol_idx}")
                else:
                    print(f"Warning: No correct solution available for problem {problem_id}, skipping NONE sample")
                    continue
            
            # Create mining prompt
            prompt = create_mining_prompt(template, problem, generated_code)
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Store metadata
            # For NONE-substituted samples, the ground truth becomes "no misconception" since we're analyzing correct code
            if is_none_sample:
                # Original labels (what failed to generate)
                original_misc_id = code_data.get('misconception_id')
                original_misc_desc = code_data.get('misconception_description', '')
                # Ground truth for analysis (correct code = no misconception)
                gt_misconception = "NONE"
                gt_misconception_desc = "No misconception - correct code"
            else:
                # Regular corrupted code - ground truth matches original
                original_misc_id = code_data.get('misconception_id')
                original_misc_desc = code_data.get('misconception_description', '')
                gt_misconception = original_misc_id
                gt_misconception_desc = original_misc_desc
            
            metadata = {
                "source_file": code_data.get('source_file', ''),
                "problem_id": problem_id,
                "solution_index": sol_idx,
                "original_misconception_id": original_misc_id,
                "original_misconception_desc": original_misc_desc,
                "gt_misconception": gt_misconception,  # Ground truth for evaluation
                "gt_misconception_desc": gt_misconception_desc,
                "was_none_substituted": is_none_sample,
                "code_type": "correct" if is_none_sample else "corrupted",
            }
            
            batches.append((metadata, messages))
    
    print(f"Prepared {len(batches)} mining prompts ({mode_str} mode)")
    return batches


def save_mining_results(results: List[Dict[str, Any]], output_dir: str, 
                          llm_info: Dict[str, str], run_info: Dict[str, Any], 
                          append_mode: bool = False):
    """Save mining results to JSON files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual results
    new_predictions = []
    
    for i, result in enumerate(results):
        metadata = result["metadata"]
        parsed = result["parsed_response"]
        
        # Copy metadata fields into each misconception for compatibility
        enhanced_misconceptions = []
        for misc in parsed["misconceptions"]:
            enhanced_misc = misc.copy()
            # Add metadata fields that are now outside misconception tag
            if "misconception_type" in parsed["metadata"]:
                enhanced_misc["misconception_type"] = parsed["metadata"]["misconception_type"]
            if "error_type" in parsed["metadata"]:
                enhanced_misc["error_type"] = parsed["metadata"]["error_type"]
            if "confidence_level" in parsed["metadata"]:
                enhanced_misc["confidence"] = parsed["metadata"]["confidence_level"]  # Map to 'confidence' for compatibility
            enhanced_misconceptions.append(enhanced_misc)
        
        prediction = {
            "prediction_id": f"{metadata['source_file']}_{metadata['solution_index']}",
            "source_file": metadata["source_file"],
            "problem_id": metadata["problem_id"],
            "solution_index": metadata["solution_index"],
            "original_misconception": {
                "id": metadata.get("original_misconception_id"),
                "description": metadata.get("original_misconception_desc")
            },
            "ground_truth_misconception": {
                "id": metadata.get("gt_misconception"),
                "description": metadata.get("gt_misconception_desc")
            },
            "predicted_misconceptions": enhanced_misconceptions,
            "no_predicted_misconceptions": parsed.get("no_predicted_misconceptions", False),
            "reasoning": parsed["reasoning"],
            "analysis": parsed["analysis"],
            "parse_success": parsed["parse_success"],
            "was_none_substituted": metadata.get("was_none_substituted", False),
            "code_type": metadata.get("code_type", "corrupted"),
            "metadata": parsed["metadata"]
        }
        
        new_predictions.append(prediction)
    
    # Handle appending to existing predictions
    output_file = os.path.join(output_dir, "predictions.json")
    all_predictions = []
    
    if append_mode and os.path.exists(output_file):
        # Load existing predictions
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_predictions = json.load(f)
            
            # Create set of existing prediction IDs to avoid duplicates
            existing_ids = {pred["prediction_id"] for pred in existing_predictions}
            
            # Add existing predictions
            all_predictions.extend(existing_predictions)
            
            # Add new predictions, skipping duplicates
            duplicates_skipped = 0
            for pred in new_predictions:
                if pred["prediction_id"] not in existing_ids:
                    all_predictions.append(pred)
                else:
                    duplicates_skipped += 1
            
            print(f"Appended {len(new_predictions) - duplicates_skipped} new predictions, skipped {duplicates_skipped} duplicates")
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load existing predictions ({e}), saving new predictions only")
            all_predictions = new_predictions
    else:
        all_predictions = new_predictions
    
    # Save all predictions to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(all_predictions)} total predictions to {output_file}")
    
    # Calculate statistics - use new predictions for run-specific stats
    new_predictions_count = len(new_predictions)
    total_predictions = len(all_predictions)
    successful_parses = sum(1 for p in new_predictions if p["parse_success"])
    total_misconceptions_found = sum(len(p["predicted_misconceptions"]) for p in new_predictions)
    avg_misconceptions = total_misconceptions_found / new_predictions_count if new_predictions_count > 0 else 0
    
    # Count NONE substitutions in new predictions
    none_substitutions = sum(1 for result in results if result["metadata"].get("was_none_substituted", False))
    
    # Count code types
    correct_codes = sum(1 for p in new_predictions if p.get("code_type") == "correct")
    corrupted_codes = sum(1 for p in new_predictions if p.get("code_type") == "corrupted")
    
    # Count codes with no predicted misconceptions by type
    correct_codes_no_misc = sum(1 for p in new_predictions if p.get("code_type") == "correct" and len(p["predicted_misconceptions"]) == 0)
    corrupted_codes_no_misc = sum(1 for p in new_predictions if p.get("code_type") == "corrupted" and len(p["predicted_misconceptions"]) == 0)
    
    # Save summary
    summary = {
        "run_timestamp": run_info["timestamp"],
        "input_directory": run_info["input_dir"],
        "output_directory": output_dir,
        "llm_provider": llm_info["provider"],
        "llm_model": llm_info["model"],
        "processing_mode": llm_info["processing_mode"],
        "reasoning_enabled": llm_info.get("reasoning_enabled", False),
        "template_type": run_info["template_type"],
        "statistics": {
            "new_codes_analyzed": new_predictions_count,
            "total_codes_in_file": total_predictions,
            "correct_codes_analyzed": correct_codes,
            "corrupted_codes_analyzed": corrupted_codes,
            "successful_parses": successful_parses,
            "parse_success_rate": successful_parses / new_predictions_count if new_predictions_count > 0 else 0,
            "total_misconceptions_found": total_misconceptions_found,
            "average_misconceptions_per_code": avg_misconceptions,
            "codes_with_no_misconceptions": sum(1 for p in new_predictions if len(p["predicted_misconceptions"]) == 0),
            "correct_codes_no_misconceptions": correct_codes_no_misc,
            "corrupted_codes_no_misconceptions": corrupted_codes_no_misc,
            "none_substitutions": none_substitutions,
            "none_substitution_rate": none_substitutions / new_predictions_count if new_predictions_count > 0 else 0,
            "append_mode": append_mode
        }
    }
    
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary Statistics:")
    print(f"  - New codes analyzed: {summary['statistics']['new_codes_analyzed']}")
    print(f"  - Total codes in file: {summary['statistics']['total_codes_in_file']}")
    print(f"  - Code types: {summary['statistics']['correct_codes_analyzed']} correct, {summary['statistics']['corrupted_codes_analyzed']} corrupted")
    print(f"  - Parse success rate: {summary['statistics']['parse_success_rate']:.2%}")
    print(f"  - Total misconceptions found: {total_misconceptions_found}")
    print(f"  - Average misconceptions per code: {avg_misconceptions:.2f}")
    print(f"  - Codes with no misconceptions: {summary['statistics']['codes_with_no_misconceptions']}")
    if correct_codes > 0:
        print(f"    - Correct codes: {summary['statistics']['correct_codes_no_misconceptions']}/{correct_codes} ({summary['statistics']['correct_codes_no_misconceptions']/correct_codes*100:.1f}%)")
    if corrupted_codes > 0:
        print(f"    - Corrupted codes: {summary['statistics']['corrupted_codes_no_misconceptions']}/{corrupted_codes} ({summary['statistics']['corrupted_codes_no_misconceptions']/corrupted_codes*100:.1f}%)")
    print(f"  - NONE substitutions: {summary['statistics']['none_substitutions']} ({summary['statistics']['none_substitution_rate']:.1%})")
    if append_mode:
        print(f"  - Mode: Appended to existing results")


def main():
    parser = argparse.ArgumentParser(description="Infer misconceptions from corrupted code using LLMs")
    
    # LLM selection
    parser.add_argument("--llm", choices=["anthropic", "openai", "vllm", "gemini"], 
                       default="anthropic", help="LLM provider to use")
    
    # LLM-specific arguments
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1",
                       help="Base URL for vLLM server (server mode)")
    parser.add_argument("--vllm-api-key", default="NONE",
                       help="API key for vLLM server (if needed)")
    parser.add_argument("--vllm-model", default=None,
                       help="vLLM model for offline mode (e.g., Qwen/Qwen3-8B). If specified, uses offline mode instead of server.")
    parser.add_argument("--vllm-enable-thinking", action="store_true",
                       help="Enable thinking mode for Qwen3 models in vLLM offline mode")
    parser.add_argument("--openai-model", default="gpt-4o",
                        choices=["gpt-4o", "gpt-4o-mini", "gpt-4.1", "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"],
                       help="OpenAI model name")
    parser.add_argument("--anthropic-model", default="claude-sonnet-4-5",
                        choices=["claude-sonnet-4-0", "claude-opus-4-1", "claude-3-7-sonnet-latest", "claude-sonnet-4-5"],
                       help="Anthropic model name")
    parser.add_argument("--gemini-model", 
                       choices=["gemini-2.5-pro-preview-06-05", "gemini-2.5-flash", "gemini-2.0-flash"],
                       default="gemini-2.5-flash",
                       help="Gemini model name")
    
    # Reasoning settings
    parser.add_argument("--reasoning", action="store_true",
                       help="Enable reasoning mode (only for compatible models)")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium",
                       help="Reasoning effort level for OpenAI models")
    parser.add_argument("--no-reasoning-template", action="store_true",
                       help="Use the no-reasoning variant of the mining template")
    
    # Template settings
    parser.add_argument("--template", 
                       choices=["zeroshot", "zeroshot-no-reasoning", "fewshot", "fewshot-no-reasoning"],
                       help="Specify template name explicitly (overrides --no-reasoning-template)")
    parser.add_argument("--template-dir", default="prompt_templates/mining",
                       help="Directory containing template files")
    
    # Data paths
    parser.add_argument("--input-dir", 
                       default="mining_misconceptions/data/corrupted_codes/corrupted_codes_anthropic",
                       help="Directory containing corrupted code files")
    parser.add_argument("--problems-file", 
                       default="mining_misconceptions/data/problems_processed.json",
                       help="Path to problems JSON file")
    
    # Output
    parser.add_argument("--output-dir", 
                       default="mining_misconceptions/test_results/mined_misconceptions",
                       help="Output directory for results")
    
    # Processing options
    parser.add_argument("--use-batch", action="store_true",
                       help="Enable batch processing")
    parser.add_argument("--max-files", type=int, default=None,
                       help="Limit number of files to process (for testing)")
    parser.add_argument("--debug-prompt", action="store_true",
                       help="Save first generated prompt to debug_mining.txt")
    parser.add_argument("--none-only", action="store_true",
                       help="Process only NONE samples (codes that were previously skipped)")
    parser.add_argument("--append-results", action="store_true",
                       help="Append results to existing predictions file instead of overwriting")
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.none_only and not args.append_results:
        print("⚠️  Warning: --none-only is typically used with --append-results to add to existing predictions")
    
    if args.none_only:
        print("🔍 Running in NONE-only mode: processing only samples that were previously 'NONE'")
    
    # Validate reasoning compatibility
    if args.reasoning:
        if not args.no_reasoning_template:
            print("⚠️  Warning: --reasoning is typically used with --no-reasoning-template")
        
        # Check model compatibility
        openai_reasoning_models = ["o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"]
        anthropic_reasoning_models = ["claude-3-7-sonnet-latest", "claude-sonnet-4-0", "claude-opus-4-1", "claude-sonnet-4-5"]
        
        if args.llm == "openai" and args.openai_model not in openai_reasoning_models:
            print(f"❌ Error: --reasoning requires OpenAI reasoning models: {', '.join(openai_reasoning_models)}")
            return 1
        elif args.llm == "anthropic" and args.anthropic_model not in anthropic_reasoning_models:
            print(f"❌ Error: --reasoning requires Anthropic reasoning models: {', '.join(anthropic_reasoning_models)}")
            return 1
        elif args.llm not in ["openai", "anthropic", "gemini"]:
            print(f"❌ Error: --reasoning is only supported for OpenAI and Anthropic")
            return 1
    
    # Load data
    print("Loading data...")
    try:
        problems = load_json_data(args.problems_file)
        corrupted_codes = load_corrupted_codes(args.input_dir)
        
        if args.max_files:
            corrupted_codes = corrupted_codes[:args.max_files]
            print(f"Limited to {len(corrupted_codes)} files for testing")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    print(f"Loaded {len(corrupted_codes)} corrupted code files")
    print(f"Loaded {len(problems)} problems for context")
    
    # Load appropriate template
    if args.template:
        template_type = args.template
    else:
        template_type = "zeroshot-no-reasoning" if args.no_reasoning_template else "zeroshot"
    
    try:
        template = load_prompt_template(template_type, args.template_dir)
    except FileNotFoundError as e:
        print(f"❌ Template loading error: {e}")
        return 1
    
    # Create LLM client
    print(f"Initializing {args.llm} LLM client...")
    try:
        llm_client = create_llm_client(args)
    except Exception as e:
        print(f"Error creating LLM client: {e}")
        return 1
    
    # Generate mining prompts
    batches = generate_mining_batches(corrupted_codes, problems, template, args.none_only)
    
    if not batches:
        print("No valid codes to analyze.")
        return 1
    
    # Debug first prompt if requested
    if args.debug_prompt and batches:
        debug_prompt = batches[0][1][0]["content"]
        with open("debug_mining.txt", 'w', encoding='utf-8') as f:
            f.write(debug_prompt)
        print(f"🔍 Debug: First prompt saved to debug_mining.txt ({len(debug_prompt)} chars)")
    
    # Process mining requests
    all_results = []
    
    if args.use_batch and args.llm != "gemini":
        print(f"🔄 Processing {len(batches)} requests in batch mode")
        
        all_metadata = [item[0] for item in batches]
        all_messages = [item[1] for item in batches]
        
        try:
            kwargs = get_llm_kwargs(args)
            print(f"Submitting batch to {args.llm}...")
            
            if args.llm == "anthropic":
                reasoning = kwargs.pop("reasoning", False)
                budget_tokens = kwargs.pop("budget_tokens", 1000)
                responses = llm_client.create_batch_messages(all_messages, reasoning=reasoning, budget_tokens=budget_tokens, **kwargs)
            elif args.llm == "openai":
                reasoning = kwargs.pop("reasoning", False)
                reasoning_effort = kwargs.pop("reasoning_effort", "medium")
                responses = llm_client.create_batch_messages(all_messages, reasoning=reasoning, reasoning_effort=reasoning_effort, **kwargs)
            elif args.llm == "vllm" and hasattr(args, 'vllm_model') and args.vllm_model:
                # Use vLLM offline mode with thinking control for Qwen3
                enable_thinking = getattr(args, 'vllm_enable_thinking', False)
                responses = llm_client.create_batch_messages_with_thinking(all_messages, enable_thinking=enable_thinking, **kwargs)
            else:
                responses = llm_client.create_batch_messages(all_messages, **kwargs)
                
            print(f"✅ Batch processing completed")
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            return 1
        
        # Parse responses
        for metadata, response in zip(all_metadata, responses):
            parsed_response = parse_mining_response(response)
            all_results.append({
                "metadata": metadata,
                "parsed_response": parsed_response
            })
    
    else:
        # Individual processing
        print(f"🔄 Processing {len(batches)} requests individually")
        
        for metadata, messages in tqdm(batches, desc="Inferring misconceptions"):
            try:
                kwargs = get_llm_kwargs(args)
                
                if hasattr(llm_client, 'create_message'):
                    if args.llm == "anthropic":
                        reasoning = kwargs.pop("reasoning", False)
                        budget_tokens = kwargs.pop("budget_tokens", 1000)
                        response = llm_client.create_message(messages, kwargs=kwargs, reasoning=reasoning, budget_tokens=budget_tokens)
                    elif args.llm == "openai":
                        reasoning = kwargs.pop("reasoning", False)
                        reasoning_effort = kwargs.pop("reasoning_effort", "medium")
                        response = llm_client.create_message(messages, kwargs=kwargs, reasoning=reasoning, reasoning_effort=reasoning_effort)
                    else:
                        response = llm_client.create_message(messages, kwargs=kwargs)
                else:
                    # Fallback to batch with single item
                    if args.llm == "anthropic":
                        reasoning = kwargs.pop("reasoning", False)
                        budget_tokens = kwargs.pop("budget_tokens", 1000)
                        responses = llm_client.create_batch_messages([messages], reasoning=reasoning, budget_tokens=budget_tokens, **kwargs)
                    elif args.llm == "openai":
                        reasoning = kwargs.pop("reasoning", False)
                        reasoning_effort = kwargs.pop("reasoning_effort", "medium")
                        responses = llm_client.create_batch_messages([messages], reasoning=reasoning, reasoning_effort=reasoning_effort, **kwargs)
                    elif args.llm == "vllm" and hasattr(args, 'vllm_model') and args.vllm_model:
                        # Use vLLM offline mode with thinking control for Qwen3
                        enable_thinking = getattr(args, 'vllm_enable_thinking', False)
                        responses = llm_client.create_batch_messages_with_thinking([messages], enable_thinking=enable_thinking, **kwargs)
                    else:
                        responses = llm_client.create_batch_messages([messages], **kwargs)
                    response = responses[0]
                    
            except Exception as e:
                print(f"Error processing request: {e}")
                response = ""
            
            parsed_response = parse_mining_response(response)
            all_results.append({
                "metadata": metadata,
                "parsed_response": parsed_response
            })
    
    # Save results
    print("\nSaving results...")
    
    llm_info = {
        "provider": args.llm,
        "model": get_model_name(args, llm_client),
        "processing_mode": "batch" if args.use_batch else "individual",
        "reasoning_enabled": args.reasoning
    }
    
    run_info = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": args.input_dir,
        "template_type": template_type
    }
    
    save_mining_results(all_results, args.output_dir, llm_info, run_info, args.append_results)
    
    print("✅ Misconception Mining completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())