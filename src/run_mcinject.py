#!/usr/bin/env python3
"""
Script to generate Python code exhibiting specific misconceptions using LLMs.

This script:
1. Loads problems from problems_processed.json
2. Loads misconceptions from misconception.json  
3. Cycles through all solutions for each problem
4. Generates misconception code using LLMs via batch processing
5. Saves results as JSON files

Usage:
    # Individual processing (default)
    python run_inject_misc.py --llm anthropic --output-dir results/
    python run_inject_misc.py --llm vllm --vllm-base-url http://localhost:8000/v1
    python run_inject_misc.py --llm openai --openai-model gpt-4o
    python run_inject_misc.py --llm gemini --gemini-model gemini-2.5-pro-preview-06-05
    python run_inject_misc.py --llm anthropic --prompt-template mining_misconceptions/prompt_templates/injection/fewshot.md
    
    # Reasoning mode (for compatible models only)
    python run_inject_misc.py --llm openai --openai-model o3-mini --reasoning --reasoning-effort high
    python run_inject_misc.py --llm anthropic --anthropic-model claude-3-7-sonnet-latest --reasoning
    
    # Batch processing (faster & cheaper for large datasets, entire dataset as one batch)
    python run_inject_misc.py --llm anthropic --use-batch
    python run_inject_misc.py --llm openai --use-batch
    python run_inject_misc.py --llm vllm --use-batch
    # Note: Gemini automatically switches to individual mode even with --use-batch
    
    # Random sampling options (per-problem misconceptions)
    python run_inject_misc.py --llm anthropic --random-seed 42 --max-problems 5 --max-misconceptions 10 --max-solutions-per-problem 3
    # This generates: 5 problems * 10 misconceptions * 3 solutions = 150 total instances
    
    # Exclude misconception examples from prompts (examples are included by default)
    python run_inject_misc.py --llm anthropic --exclude-examples
    
    # Keep inline comments in generated code (comments are removed by default)
    python run_inject_misc.py --llm anthropic --keep-comments
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re
import random
import ast
import tokenize
from io import StringIO
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.llm_clients import OpenAIClient, VLLMClient, AnthropicClient, GeminiClient


def load_prompt_template(template_path: str) -> str:
    """Load the prompt template from file."""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_template_content(template: str) -> str:
    """Return the entire template content - LLM needs all instructions and examples."""
    # Return the full template so LLM sees XML formatting instructions and examples
    return template


def create_prompt(template_content: str, problem: Dict[str, Any], solution: str, misconception: Dict[str, Any], include_example: bool = True) -> str:
    """Create a specific prompt by filling in the template."""
    
    # Replace all placeholders in the template consistently
    # Templates use {text} for main template variables and [text] for placeholders in examples
    
    prompt = template_content
    
    # Prepare misconception description (with or without example)
    misconception_desc = misconception["description"]
    if include_example and "example" in misconception:
        misconception_desc += f"\n\n**Example:**\n{misconception['example']}"
    
    # Replace {curly_brace} placeholders (actual template variables)
    prompt = prompt.replace("{correct_solution}", solution)
    prompt = prompt.replace("{misconception_desc}", misconception_desc)
    prompt = prompt.replace("{problem_description}", problem.get("description", ""))
    prompt = prompt.replace("{problem_title}", f"Problem {problem.get('id', 'Unknown')}")
    
    
    return prompt


def remove_inline_comments(code: str) -> str:
    """Remove inline comments from Python code using tokenize module.
    
    Args:
        code: Python source code as string
        
    Returns:
        Code with inline comments removed, or original code if processing fails
    """
    if not code or code.strip() == "NONE":
        return code
    
    try:
        # Create a StringIO object from the code
        code_io = StringIO(code)
        
        # Tokenize the code
        tokens = list(tokenize.generate_tokens(code_io.readline))
        
        # Filter out comment tokens
        filtered_tokens = []
        for token in tokens:
            if token.type != tokenize.COMMENT:
                filtered_tokens.append(token)
        
        # Reconstruct the code from filtered tokens
        result_code = tokenize.untokenize(filtered_tokens)
        
        # Clean up any extra whitespace that might have been introduced
        lines = result_code.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove trailing whitespace but preserve leading indentation
            cleaned_line = line.rstrip()
            cleaned_lines.append(cleaned_line)
        
        # Remove any trailing empty lines but preserve structure
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
            
        result_code = '\n'.join(cleaned_lines)
        
        # Validate that the result is still valid Python syntax
        try:
            ast.parse(result_code)
            return result_code
        except SyntaxError:
            # If removing comments broke the syntax, return original
            print("Warning: Comment removal resulted in syntax error, keeping original code")
            return code
            
    except Exception as e:
        # If any error occurs during processing, return original code
        print(f"Warning: Failed to remove comments from code: {e}")
        return code


def parse_llm_response(response: str, remove_comments: bool = True) -> Dict[str, Any]:
    """Parse LLM response and extract reasoning, code, and metadata."""
    result = {
        "reasoning": "",
        "code": "",
        "metadata": {},
        "raw_response": response,
        "parse_success": False
    }
    
    try:
        # Pre-process: Remove markdown code fences if present
        # LLMs sometimes wrap XML in ```xml ... ``` or use ``` instead of </code>
        cleaned_response = response
        
        # Step 1: Replace ``` that appears between <code> and <metadata> with </code>
        # This is the main fix - LLM sometimes uses ``` instead of </code>
        # Pattern: <code>...content...```\n\n<metadata>
        if "</code>" not in response:
            cleaned_response = re.sub(
                r'(<code>.*?)```\s*\n+\s*(<metadata>)',
                r'\1</code>\n\n\2',
                cleaned_response,
                flags=re.DOTALL
            )
        
        # Step 2: Remove opening ```xml or ```python fence before <code>
        # Only match the specific pattern right before <code>
        cleaned_response = re.sub(r'```(?:xml|python)?\s*\n\s*(<code>)', r'\1', cleaned_response)
        
        # Step 3: Remove closing ``` at the very end after </metadata>
        cleaned_response = re.sub(r'(</metadata>)\s*\n```\s*$', r'\1', cleaned_response)
        
        # Try to extract XML content with improved regex
        reasoning_match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', cleaned_response, re.DOTALL)
        code_match = re.search(r'<code>\s*(.*?)\s*</code>', cleaned_response, re.DOTALL)
        metadata_match = re.search(r'<metadata>\s*(.*?)\s*</metadata>', cleaned_response, re.DOTALL)
        
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        
        if code_match:
            extracted_code = code_match.group(1).strip()
            # Post-process: remove inline comments from the code if requested
            if remove_comments:
                result["code"] = remove_inline_comments(extracted_code)
            else:
                result["code"] = extracted_code
        
        if metadata_match:
            metadata_text = metadata_match.group(1).strip()
            # Parse simple key: value pairs from metadata
            result["metadata"] = {"raw": metadata_text}
            for line in metadata_text.split('\n'):
                line = line.strip()
                if ':' in line and not line.startswith('#'):  # Skip comments
                    key, value = line.split(':', 1)
                    result["metadata"][key.strip()] = value.strip()
        
        # Check if we got the essential parts (reasoning and either code or "NONE")
        if reasoning_match and code_match:
            result["parse_success"] = True
        elif reasoning_match and code_match and result["code"] == "NONE":
            # Handle inapplicable cases
            result["parse_success"] = True
            
    except Exception as e:
        result["metadata"]["parse_error"] = str(e)
        print(f"Warning: Failed to parse LLM response: {e}")
        
        # Fallback: try to extract content even without proper XML
        if "reasoning" in response.lower() and ("code" in response.lower() or "none" in response.upper()):
            result["parse_success"] = True
            result["reasoning"] = "Fallback parsing used - check raw_response"
            if "NONE" in response.upper():
                result["code"] = "NONE"
    
    return result


def parse_evaluation_response(response: str) -> Dict[str, Any]:
    """Parse evaluation response and extract the answer and feedback."""
    result = {
        "exhibits_misconception": None,
        "reasoning": "",
        "rationale": "",
        "feedback": "",
        "confidence": "",
        "misconception_type": None,
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
            
            # Extract rationale (new field)
            rationale_match = re.search(r'<rationale>\s*(.*?)\s*</rationale>', answer_content, re.DOTALL)
            if rationale_match:
                result["rationale"] = rationale_match.group(1).strip()
            
            # Extract feedback (new field)
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
            
            # Check if we got the essential parts (rationale is now required)
            if result["exhibits_misconception"] is not None and result["rationale"]:
                result["parse_success"] = True
            # Backward compatibility: accept reasoning if rationale not present
            elif result["exhibits_misconception"] is not None and result["reasoning"]:
                result["parse_success"] = True
                
    except Exception as e:
        result["parse_error"] = str(e)
        print(f"Warning: Failed to parse evaluation response: {e}")
    
    return result


def create_evaluation_prompt(template: str, misconception: Dict[str, Any], code: str) -> str:
    """Create evaluation prompt by filling in the template."""
    prompt = template
    
    # Replace placeholders
    prompt = prompt.replace("{misconception_description}", misconception["description"])
    prompt = prompt.replace("{misconception_example}", misconception.get("example", "No example provided"))
    prompt = prompt.replace("{code_to_analyze}", code)
    
    return prompt


def evaluate_generated_code(code: str, misconception: Dict[str, Any], 
                           eval_template: str, eval_client: AnthropicClient) -> Dict[str, Any]:
    """Evaluate generated code and return parsed evaluation response."""
    # Create evaluation prompt
    prompt = create_evaluation_prompt(eval_template, misconception, code)
    messages = [{"role": "user", "content": prompt}]
    
    try:
        # Call Claude with reasoning enabled
        response = eval_client.create_message(
            messages,
            kwargs={
                "model": "claude-sonnet-4-5",
                "temperature": 0.1,
                "max_tokens": 2000
            },
            reasoning=True,
            budget_tokens=2000
        )
        
        # Parse response
        parsed = parse_evaluation_response(response)
        
    except Exception as e:
        print(f"Error evaluating code: {e}")
        parsed = {
            "exhibits_misconception": None,
            "reasoning": f"Error: {e}",
            "rationale": "",
            "feedback": "",
            "confidence": "low",
            "raw_response": str(e),
            "parse_success": False
        }
    
    return parsed


def create_feedback_message(original_prompt: str, previous_attempt: str, feedback: str) -> List[Dict[str, str]]:
    """Create a multi-turn conversation with feedback for regeneration."""
    feedback_instruction = f"""Your previous code was evaluated and needs improvement. Here is the feedback:

{feedback}

Please generate an improved version that addresses this feedback while still exhibiting the misconception as specified in the original instructions."""
    
    messages = [
        {"role": "user", "content": original_prompt},
        {"role": "assistant", "content": previous_attempt},
        {"role": "user", "content": feedback_instruction}
    ]
    
    return messages


def get_prompting_method_description(template_path: str, reasoning_enabled: bool) -> str:
    """Generate a concise description of the prompting method based on template file name and settings."""
    template_name = os.path.basename(template_path).replace('.md', '')
    
    # Extract method from filename
    if 'fewshot' in template_name:
        method = "Few-shot prompting"
    elif 'zeroshot' in template_name:
        method = "Zero-shot prompting"
    else:
        method = "Custom prompting"
    
    # Add reasoning information
    if reasoning_enabled:
        method += " with model reasoning"
    elif 'no-reasoning' in template_name:
        method += " without explicit reasoning instructions"
    else:
        method += " with explicit reasoning instructions"
    
    return method


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
            # Use the model_name that was set during VLLMClient initialization
            return llm_client.model_name
        elif llm_client and hasattr(llm_client, 'get_llm_server_modelname'):
            # Fallback to querying the server if model_name is not available
            model_name = llm_client.get_llm_server_modelname()
            return model_name if model_name else "vllm-unknown-model"
        else:
            return "vllm-served-model"


def get_llm_kwargs(args) -> Dict[str, Any]:
    """Get LLM-specific kwargs based on the selected provider."""
    if args.llm == "anthropic":
        # Check if model supports reasoning and if reasoning is enabled
        reasoning_supported = args.anthropic_model in ["claude-3-7-sonnet-latest", "claude-sonnet-4-0", "claude-opus-4-0", "claude-sonnet-4-5", "claude-opus-4-1"]
        use_reasoning = args.reasoning and reasoning_supported
        return {
            "model": args.anthropic_model,
            "temperature": 0.1,
            "max_tokens": 4000,
            "reasoning": use_reasoning,
            "budget_tokens": 2000 if use_reasoning else 1000
        }
    elif args.llm == "openai":
        # Check if model supports reasoning and if reasoning is enabled
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
        # Check if model supports thinking/reasoning and if reasoning is enabled
        gemini_thinking_models = ["gemini-2.5-flash-preview", "gemini-2.5-flash-exp", "gemini-2.5-flash", 
                                 "gemini-2.5-pro", "gemini-2.5-pro-exp", "gemini-2.5-pro-exp-03-25"]
        reasoning_supported = any(thinking_model in args.gemini_model for thinking_model in gemini_thinking_models)
        use_reasoning = args.reasoning and reasoning_supported
        return {
            "model": args.gemini_model,
            "temperature": 0.1,
            "max_tokens": 4000,
            "reasoning": use_reasoning,
            "thinking_budget": 2000 if use_reasoning else 1000
        }
    else:  # vllm
        return {
            "temperature": 0.1,
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


def generate_misconception_batches(problems: Dict[str, Any], misconceptions: List[Dict[str, Any]], 
                                  template_content: str, max_problems: int = None, 
                                  max_misconceptions: int = None, max_solutions_per_problem: int = None,
                                  random_seed: int = None, include_example: bool = True) -> List[Tuple[Dict[str, Any], List[Dict[str, str]]]]:
    """Generate batches of prompts using random sampling strategy with max_misconceptions per problem."""
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        print(f"🎲 Random seed set to: {random_seed}")
    
    batches = []
    
    print("Preparing prompts with random sampling strategy...")
    
    # Handle both list and dict formats for problems
    if isinstance(problems, list):
        problem_items = [(problem.get("id", idx), problem) for idx, problem in enumerate(problems)]
    else:
        problem_items = list(problems.items())
    
    # Check if misconception ID 35 is present
    has_misconception_35 = any(misc.get("id") == 35 for misc in misconceptions)
    
    # Randomly shuffle the problems to get random sampling order
    random.shuffle(problem_items)
    
    # Special handling: If misconception 35 is present, ensure problem 501 is included
    if has_misconception_35:
        problem_501_item = None
        # Find problem 501 in the items
        for item in problem_items:
            if str(item[0]) == "501" or item[0] == 501:
                problem_501_item = item
                break
        
        # If problem 501 exists but wasn't in the shuffled list, we need to find it in the original data
        if problem_501_item is None:
            if isinstance(problems, list):
                for idx, problem in enumerate(problems):
                    if str(problem.get("id", idx)) == "501" or problem.get("id", idx) == 501:
                        problem_501_item = (problem.get("id", idx), problem)
                        break
            else:
                for problem_id, problem in problems.items():
                    if str(problem_id) == "501" or problem_id == 501:
                        problem_501_item = (problem_id, problem)
                        break
        
        # If problem 501 was found, ensure it's at the beginning of our selection
        if problem_501_item:
            # Remove problem 501 from its current position if it exists
            problem_items = [item for item in problem_items if str(item[0]) != "501" and item[0] != 501]
            # Add problem 501 at the beginning
            problem_items.insert(0, problem_501_item)
            print(f"🎯 Misconception 35 detected: Problem 501 guaranteed to be included")
        else:
            print(f"⚠️ Misconception 35 detected but Problem 501 not found in dataset")
    
    # Limit problems if specified
    if max_problems:
        problem_items = problem_items[:max_problems]
        print(f"🎯 Limited to {len(problem_items)} randomly sampled problems")
    
    # Set target misconceptions per problem
    target_misconceptions_per_problem = max_misconceptions if max_misconceptions else len(misconceptions)
    
    print(f"📊 Target: {target_misconceptions_per_problem} misconception instances PER problem across {len(problem_items)} problems")
    
    global_sampling_order = 0
    global_misconception_index = 0  # Global misconception index that continues across problems
    
    # Iterate through randomly sampled problems
    for problem_idx, (problem_id, problem) in enumerate(problem_items):
        solutions = problem.get("solutions", [])
        
        # Limit solutions per problem if specified
        if max_solutions_per_problem and len(solutions) > max_solutions_per_problem:
            solutions = solutions[:max_solutions_per_problem]
            print(f"📝 Problem {problem_id}: Using {len(solutions)} solutions (limited from {len(problem.get('solutions', []))})")
        
        if not solutions:
            print(f"⚠️ Problem {problem_id}: No solutions available, skipping")
            continue
        
        print(f"🔄 Processing problem {problem_id} ({problem_idx + 1}/{len(problem_items)}) with {len(solutions)} solutions")
        
        # Track misconceptions processed for this specific problem
        misconceptions_generated_for_problem = 0
        misconceptions_used_for_problem = []  # Track which misconceptions are used for this problem
        
        # Generate exactly target_misconceptions_per_problem instances for this problem
        while misconceptions_generated_for_problem < target_misconceptions_per_problem:
            # Get current misconception (cycling through the list, continuing globally)
            current_misconception = misconceptions[global_misconception_index % len(misconceptions)]
            misconceptions_used_for_problem.append(current_misconception["id"])
            
            # Process all solutions for this problem-misconception pair
            for sol_idx, solution in enumerate(solutions):
                if misconceptions_generated_for_problem >= target_misconceptions_per_problem:
                    break
                    
                # Create the prompt
                prompt = create_prompt(template_content, problem, solution, current_misconception, include_example)
                
                # Create messages format for LLM
                messages = [
                    {"role": "user", "content": prompt}
                ]
                
                # Store metadata about this request
                metadata = {
                    "problem_id": problem_id,
                    "problem_title": f"Problem {problem_id}",
                    "solution_index": sol_idx,
                    "misconception_id": current_misconception["id"],
                    "misconception_description": current_misconception["description"],
                    "sampling_order": global_sampling_order + 1,
                    "random_seed_used": random_seed,
                    "problem_misconception_index": misconceptions_generated_for_problem + 1,
                    "global_misconception_index": global_misconception_index
                }
                
                batches.append((metadata, messages))
                misconceptions_generated_for_problem += 1
                global_sampling_order += 1
            
            # Move to next misconception globally (continues across problems)
            global_misconception_index += 1
            
            # Safety check to prevent infinite loop if we run out of unique misconceptions
            if misconceptions_generated_for_problem >= len(misconceptions) * len(solutions):
                print(f"⚠️  Problem {problem_id}: Reached maximum possible unique combinations")
                break
        
        print(f"✅ Problem {problem_id}: Generated {misconceptions_generated_for_problem} instances using misconceptions {misconceptions_used_for_problem}")
    
    # Calculate expected vs actual totals
    expected_total = len(problem_items) * target_misconceptions_per_problem
    actual_total = len(batches)
    
    print(f"Generated {actual_total} total prompts using random sampling strategy")
    print(f"Expected: {expected_total} instances ({len(problem_items)} problems * {target_misconceptions_per_problem} misconceptions)")
    print(f"Coverage: {len(set(batch[0]['problem_id'] for batch in batches))} unique problems, "
          f"{len(set(batch[0]['misconception_id'] for batch in batches))} unique misconceptions")
    
    return batches


def save_results(results: List[Dict[str, Any]], output_dir: str, llm_info: Dict[str, str], prompt_info: Dict[str, str]):
    """Save results to JSON files organized by problem and misconception."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by problem and misconception
    grouped = {}
    
    for result in results:
        problem_id = result["metadata"]["problem_id"]
        misconception_id = result["metadata"]["misconception_id"]
        
        key = f"problem_{problem_id}_misc_{misconception_id}"
        
        if key not in grouped:
            grouped[key] = {
                "problem_id": problem_id,
                "problem_title": result["metadata"]["problem_title"],
                "misconception_id": misconception_id,
                "misconception_description": result["metadata"]["misconception_description"],
                "solutions": []
            }
        
        solution_data = {
            "solution_index": result["metadata"]["solution_index"],
            "generated_code": result["parsed_response"]["code"],
            "reasoning": result["parsed_response"]["reasoning"],
            "metadata": result["parsed_response"]["metadata"],
            "raw_response": result["parsed_response"]["raw_response"],
            "parse_success": result["parsed_response"]["parse_success"],
            "sampling_order": result["metadata"].get("sampling_order"),
            "random_seed_used": result["metadata"].get("random_seed_used"),
            "problem_misconception_index": result["metadata"].get("problem_misconception_index"),
            "global_misconception_index": result["metadata"].get("global_misconception_index")
        }
        
        # Add feedback loop information if present
        if "feedback_loop" in result:
            solution_data["feedback_loop"] = result["feedback_loop"]
        
        grouped[key]["solutions"].append(solution_data)
    
    # Save each group to a separate file
    for key, data in grouped.items():
        output_file = os.path.join(output_dir, f"{key}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(grouped)} result files to {output_dir}")
    
    # Calculate sampling statistics
    unique_problems = len(set(r["metadata"]["problem_id"] for r in results))
    unique_misconceptions = len(set(r["metadata"]["misconception_id"] for r in results))
    random_seed_used = results[0]["metadata"].get("random_seed_used") if results else None
    
    # Calculate feedback loop statistics
    feedback_enabled = any("feedback_loop" in r and r["feedback_loop"].get("enabled") for r in results)
    total_with_feedback = sum(1 for r in results if "feedback_loop" in r and r["feedback_loop"].get("iterations", 0) > 0)
    total_iterations = sum(r.get("feedback_loop", {}).get("iterations", 0) for r in results)
    avg_iterations = total_iterations / total_with_feedback if total_with_feedback > 0 else 0
    
    # Success metrics for feedback loop
    samples_with_final_eval = sum(1 for r in results if "feedback_loop" in r and r["feedback_loop"].get("final_evaluation"))
    exhibits_after_feedback = sum(
        1 for r in results 
        if "feedback_loop" in r 
        and r["feedback_loop"].get("final_evaluation") 
        and r["feedback_loop"]["final_evaluation"].get("exhibits_misconception")
    )
    
    # Save summary statistics
    summary = {
        "total_requests": len(results),
        "total_files": len(grouped),
        "unique_problems_sampled": unique_problems,
        "unique_misconceptions_used": unique_misconceptions,
        "parse_success_rate": sum(1 for r in results if r["parsed_response"]["parse_success"]) / len(results),
        "inapplicable_rate": sum(1 for r in results if r["parsed_response"]["code"] == "NONE") / len(results),
        "llm_provider": llm_info["provider"],
        "llm_model": llm_info["model"],
        "processing_mode": llm_info["processing_mode"],
        "reasoning_enabled": llm_info.get("reasoning_enabled", False),
        "reasoning_effort": llm_info.get("reasoning_effort"),
        "prompt_template_path": prompt_info["template_path"],
        "prompt_template_content": prompt_info["template_content"],
        "prompting_method": prompt_info["method_description"],
        "sampling_strategy": "random_problems_iterative_misconceptions_per_problem",
        "random_seed_used": random_seed_used,
        "max_problems_limit": llm_info.get("max_problems_limit"),
        "max_misconceptions_per_problem_limit": llm_info.get("max_misconceptions_limit"),
        "max_solutions_per_problem_limit": llm_info.get("max_solutions_per_problem_limit"),
        "examples_included_in_prompts": llm_info.get("examples_included", True),
        "inline_comments_removed": llm_info.get("comments_removed", True),
        "feedback_loop_enabled": feedback_enabled,
        "feedback_loop_stats": {
            "total_samples_regenerated": total_with_feedback,
            "total_iterations": total_iterations,
            "avg_iterations_per_sample": avg_iterations,
            "samples_with_final_evaluation": samples_with_final_eval,
            "exhibits_misconception_after_feedback": exhibits_after_feedback,
            "success_rate_after_feedback": exhibits_after_feedback / samples_with_final_eval if samples_with_final_eval > 0 else 0
        } if feedback_enabled else None
    }
    
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary: {summary['parse_success_rate']:.2%} parse success, {summary['inapplicable_rate']:.2%} inapplicable")
    print(f"LLM: {llm_info['provider']} ({llm_info['model']}) - {llm_info['processing_mode']} mode")
    print(f"Sampling: {unique_problems} problems, {unique_misconceptions} misconceptions, seed: {random_seed_used}")
    
    if feedback_enabled and summary.get("feedback_loop_stats"):
        stats = summary["feedback_loop_stats"]
        print(f"\nFeedback Loop Stats:")
        print(f"  Samples regenerated: {stats['total_samples_regenerated']}")
        print(f"  Total iterations: {stats['total_iterations']}")
        print(f"  Avg iterations per sample: {stats['avg_iterations_per_sample']:.2f}")
        print(f"  Success rate after feedback: {stats['success_rate_after_feedback']:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Generate misconception code using LLMs")
    
    # LLM selection
    parser.add_argument("--llm", choices=["anthropic", "openai", "vllm", "gemini"], 
                       default="anthropic", help="LLM provider to use")
    
    # LLM-specific arguments
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1",
                       help="Base URL for vLLM server")
    parser.add_argument("--vllm-api-key", default="NONE",
                       help="API key for vLLM server (if needed)")
    parser.add_argument("--openai-model", default="gpt-4o",
                        choices=["gpt-4o", "gpt-4o-mini", "gpt-4.1", "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"],
                       help="OpenAI model name")
    parser.add_argument("--anthropic-model", default="claude-sonnet-4-5",
                        choices=["claude-sonnet-4-5", "claude-sonnet-4-0", "claude-opus-4-0", "claude-3-7-sonnet-latest"],
                       help="Anthropic model name")
    parser.add_argument("--gemini-model", 
                       choices=["gemini-2.5-pro-preview-06-05", "gemini-2.5-flash", "gemini-2.0-flash",
                               "gemini-2.5-flash-preview", "gemini-2.5-flash-exp", "gemini-2.5-flash",
                               "gemini-2.5-pro", "gemini-2.5-pro-exp", "gemini-2.5-pro-exp-03-25"],
                       default="gemini-2.5-flash",
                       help="Gemini model name")
    
    # Reasoning settings
    parser.add_argument("--reasoning", action="store_true",
                       help="Enable reasoning mode (only compatible with OpenAI o1/o3/o4 models and Anthropic reasoning models)")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium",
                       help="Set reasoning effort level for OpenAI reasoning models (default: medium)")
    
    # Data paths
    parser.add_argument("--problems-file", 
                       default="mining_misconceptions/data/problems_processed.json",
                       help="Path to problems JSON file")
    parser.add_argument("--misconceptions-file",
                       default="mining_misconceptions/data/misconception.json", 
                       help="Path to misconceptions JSON file")
    parser.add_argument("--prompt-template",
                       default="mining_misconceptions/prompt_templates/injection/zeroshot.md",
                       help="Path to prompt template file (default: zeroshot.md. Alternatives: fewshot.md, zeroshot-no-reasoning.md, and fewshot-no-reasoning.md)")
    
    # Output
    parser.add_argument("--output-dir", default="mining_misconceptions/data/corrupted_codes/corrupted_codes_best",
                       help="Output directory for results")
    
    # Processing options
    parser.add_argument("--use-batch", action="store_true",
                       help="Enable batch processing (processes entire dataset as one batch, excludes Gemini)")
    parser.add_argument("--max-problems", type=int, default=None,
                       help="Limit number of problems to randomly sample (for testing)")
    parser.add_argument("--max-misconceptions", type=int, default=None,
                       help="Limit number of misconception instances to generate PER problem")
    parser.add_argument("--max-solutions-per-problem", type=int, default=None,
                       help="Limit number of solutions to use per problem (if not set, use all solutions)")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducible problem sampling (default: 42)")
    parser.add_argument("--debug-prompt", action="store_true",
                       help="Save first generated prompt to debug.txt for inspection")
    parser.add_argument("--exclude-examples", action="store_true",
                       help="Exclude misconception examples from prompts (include by default)")
    parser.add_argument("--keep-comments", action="store_true",
                       help="Keep inline comments in generated code (removed by default)")
    
    # Feedback loop options
    parser.add_argument("--enable-feedback-loop", action="store_true",
                       help="Enable evaluation and regeneration based on feedback from Claude")
    parser.add_argument("--max-feedback-iterations", type=int, default=1,
                       help="Maximum feedback retry attempts (default: 1, max: 3)")
    parser.add_argument("--evaluation-prompt-template",
                       default="mining_misconceptions/prompt_templates/evaluation/check_misconception_exhibited.md",
                       help="Path to evaluation prompt template")
    
    args = parser.parse_args()
    
    # Validate feedback loop arguments
    if args.enable_feedback_loop:
        if args.max_feedback_iterations < 1 or args.max_feedback_iterations > 3:
            print(f"❌ Error: --max-feedback-iterations must be between 1 and 3")
            print(f"   Current value: {args.max_feedback_iterations}")
            return 1
        
        if args.use_batch:
            print("⚠️  Warning: Feedback loop is not compatible with batch mode. Disabling batch processing.")
            args.use_batch = False
    
    # Validate reasoning argument compatibility
    if args.reasoning:
        # Define compatible models for reasoning
        openai_reasoning_models = ["o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"]
        anthropic_reasoning_models = ["claude-3-7-sonnet-latest", "claude-sonnet-4-0", "claude-opus-4-0", "claude-sonnet-4-5", "claude-opus-4-1"]
        gemini_thinking_models = ["gemini-2.5-flash-preview", "gemini-2.5-flash-exp", "gemini-2.5-flash",
                                 "gemini-2.5-pro", "gemini-2.5-pro-exp", "gemini-2.5-pro-exp-03-25"]
        
        # Validate reasoning is only used with no-reasoning templates
        if "no-reasoning" not in args.prompt_template:
            print(f"❌ Error: --reasoning can only be used with 'no-reasoning' prompt templates.")
            print(f"   Current template: {args.prompt_template}")
            print(f"   Please use a template with 'no-reasoning' in the filename when using --reasoning mode.")
            return 1
        
        if args.llm == "openai":
            if args.openai_model not in openai_reasoning_models:
                print(f"❌ Error: --reasoning is only compatible with OpenAI reasoning models: {', '.join(openai_reasoning_models)}")
                print(f"   Current model: {args.openai_model}")
                return 1
        elif args.llm == "anthropic":
            if args.anthropic_model not in anthropic_reasoning_models:
                print(f"❌ Error: --reasoning is only compatible with Anthropic reasoning models: {', '.join(anthropic_reasoning_models)}")
                print(f"   Current model: {args.anthropic_model}")
                return 1
        elif args.llm == "gemini":
            if not any(thinking_model in args.gemini_model for thinking_model in gemini_thinking_models):
                print(f"❌ Error: --reasoning is only compatible with Gemini thinking models: {', '.join(gemini_thinking_models)}")
                print(f"   Current model: {args.gemini_model}")
                return 1
        else:
            print(f"❌ Error: --reasoning is only compatible with OpenAI, Anthropic, and Gemini LLMs.")
            print(f"   Current LLM: {args.llm}")
            print(f"   Compatible OpenAI models: {', '.join(openai_reasoning_models)}")
            print(f"   Compatible Anthropic models: {', '.join(anthropic_reasoning_models)}")
            print(f"   Compatible Gemini models: {', '.join(gemini_thinking_models)}")
            return 1
    
    # Load data
    print("Loading data...")
    try:
        problems = load_json_data(args.problems_file)
        misconceptions_data = load_json_data(args.misconceptions_file)
        template = load_prompt_template(args.prompt_template)
        
        # Extract misconceptions list from the data structure
        if isinstance(misconceptions_data, dict):
            misconceptions = list(misconceptions_data.values())
        else:
            misconceptions = misconceptions_data
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    print(f"Loaded {len(problems)} problems and {len(misconceptions)} misconceptions")
    
    # Report example inclusion status
    if args.exclude_examples:
        print("📝 Misconception examples will be EXCLUDED from prompts")
    else:
        print("📝 Misconception examples will be INCLUDED in prompts (default)")
    
    # Report comment processing status
    if args.keep_comments:
        print("💬 Inline comments will be KEPT in generated code")
    else:
        print("💬 Inline comments will be REMOVED from generated code (default)")
    
    # Extract template content
    template_content = extract_template_content(template)
    
    print(f"Template loaded successfully. Length: {len(template_content)} characters")
    
    # Verify XML formatting instructions are included
    if  "<code>" in template_content and "<metadata>" in template_content:
        print("✅ XML formatting instructions found in template")
    else:
        print("⚠️ Warning: XML formatting instructions not found in template!")
    
    # Create LLM client
    print(f"Initializing {args.llm} LLM client...")
    if args.reasoning:
        if args.llm == "openai":
            print(f"🧠 Reasoning mode enabled with {args.openai_model} (effort: {args.reasoning_effort})")
        elif args.llm == "anthropic":
            print(f"🧠 Reasoning mode enabled with {args.anthropic_model}")
    try:
        llm_client = create_llm_client(args)
    except Exception as e:
        print(f"Error creating LLM client: {e}")
        return 1
    
    # Initialize evaluation client and load template if feedback loop is enabled
    eval_client = None
    eval_template = None
    if args.enable_feedback_loop:
        print(f"🔄 Feedback loop enabled (max iterations: {args.max_feedback_iterations})")
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("❌ Error: ANTHROPIC_API_KEY required for feedback loop evaluation")
                return 1
            eval_client = AnthropicClient(api_key=api_key)
            eval_template = load_prompt_template(args.evaluation_prompt_template)
            print(f"✅ Evaluation client initialized with Claude Sonnet 4.5")
        except Exception as e:
            print(f"Error initializing evaluation client: {e}")
            return 1
    
    # Generate all prompts using the new random sampling strategy
    batches = generate_misconception_batches(
        problems, misconceptions, template_content,
        max_problems=args.max_problems,
        max_misconceptions=args.max_misconceptions,
        max_solutions_per_problem=args.max_solutions_per_problem,
        random_seed=args.random_seed,
        include_example=not args.exclude_examples
    )
    
    # Debug: save first prompt if requested
    if args.debug_prompt and batches:
        debug_prompt = batches[0][1][0]["content"]  # Get first prompt content
        with open("debug_prompt.txt", 'w', encoding='utf-8') as f:
            f.write(debug_prompt)
        print(f"🔍 Debug: First prompt saved to debug_prompt.txt ({len(debug_prompt)} chars)")
    
    if not batches:
        print("No prompts generated. Check your input data.")
        return 1
    
    # Process messages (batch or individual)
    all_results = []
    
    if args.use_batch:
        # Check if provider supports batch processing
        if args.llm == "gemini":
            print("⚠️ Gemini does not support batch processing. Switching to individual mode.")
            args.use_batch = False
        else:
            print(f"🔄 Processing {len(batches)} requests as one large batch")
            
            # Process entire dataset as one batch
            all_metadata = [item[0] for item in batches]
            all_messages = [item[1] for item in batches]
            
            try:
                # Set up kwargs based on the LLM type
                kwargs = get_llm_kwargs(args)
                print(f"Submitting batch of {len(all_messages)} requests to {args.llm}...")
                
                if args.llm == "anthropic":
                    reasoning = kwargs.pop("reasoning", False)
                    budget_tokens = kwargs.pop("budget_tokens", 1000)
                    responses = llm_client.create_batch_messages(all_messages, reasoning=reasoning, budget_tokens=budget_tokens, **kwargs)
                elif args.llm == "openai":
                    reasoning = kwargs.pop("reasoning", False)
                    reasoning_effort = kwargs.pop("reasoning_effort", "medium")
                    responses = llm_client.create_batch_messages(all_messages, reasoning=reasoning, reasoning_effort=reasoning_effort, **kwargs)
                elif args.llm == "gemini":
                    reasoning = kwargs.pop("reasoning", False)
                    thinking_budget = kwargs.pop("thinking_budget", 1000)
                    responses = llm_client.create_batch_messages(all_messages, reasoning=reasoning, thinking_budget=thinking_budget, **kwargs)
                else:
                    responses = llm_client.create_batch_messages(all_messages, **kwargs)
                print(f"✅ Batch processing completed, received {len(responses)} responses")
            except Exception as e:
                print(f"Error processing batch: {e}")
                return 1
            
            # Parse and store results
            for metadata, response in zip(all_metadata, responses):
                parsed_response = parse_llm_response(response, remove_comments=not args.keep_comments)
                
                result = {
                    "metadata": metadata,
                    "parsed_response": parsed_response
                }
                all_results.append(result)
    
    if not args.use_batch:
        print(f"🔄 Processing {len(batches)} requests individually")
        
        # Process each request individually
        for i, (metadata, messages) in enumerate(tqdm(batches, desc="Processing requests")):
            try:
                # Set up kwargs based on the LLM type
                kwargs = get_llm_kwargs(args)
                
                # Process single message
                if hasattr(llm_client, 'create_message'):
                    if args.llm == "anthropic":
                        reasoning = kwargs.pop("reasoning", False)
                        budget_tokens = kwargs.pop("budget_tokens", 1000)
                        response = llm_client.create_message(messages, kwargs=kwargs, reasoning=reasoning, budget_tokens=budget_tokens)
                    elif args.llm == "openai":
                        reasoning = kwargs.pop("reasoning", False)
                        reasoning_effort = kwargs.pop("reasoning_effort", "medium")
                        response = llm_client.create_message(messages, kwargs=kwargs, reasoning=reasoning, reasoning_effort=reasoning_effort)
                    elif args.llm == "gemini":
                        reasoning = kwargs.pop("reasoning", False)
                        thinking_budget = kwargs.pop("thinking_budget", 1000)
                        response = llm_client.create_message(messages, kwargs=kwargs, reasoning=reasoning, thinking_budget=thinking_budget)
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
                    elif args.llm == "gemini":
                        reasoning = kwargs.pop("reasoning", False)
                        thinking_budget = kwargs.pop("thinking_budget", 1000)
                        responses = llm_client.create_batch_messages([messages], reasoning=reasoning, thinking_budget=thinking_budget, **kwargs)
                    else:
                        responses = llm_client.create_batch_messages([messages], **kwargs)
                    response = responses[0]
                    
            except Exception as e:
                print(f"Error processing request {i+1}: {e}")
                # Create a failed result
                parsed_response = {
                    "reasoning": "",
                    "code": "",
                    "metadata": {"error": str(e)},
                    "raw_response": f"Error: {e}",
                    "parse_success": False
                }
                response = ""
            
            if response:
                parsed_response = parse_llm_response(response, remove_comments=not args.keep_comments)
            
            # Initialize feedback loop tracking
            evaluation_history = []
            feedback_iterations = 0
            final_evaluation = None
            
            # Implement feedback loop if enabled
            if args.enable_feedback_loop and parsed_response.get("code") and parsed_response["code"] != "NONE":
                # Get misconception for evaluation
                misconception_id = metadata["misconception_id"]
                # Find the misconception in the list
                misconception = None
                for misc in misconceptions:
                    if misc["id"] == misconception_id:
                        misconception = misc
                        break
                
                if misconception and eval_client and eval_template:
                    current_code = parsed_response["code"]
                    current_response = response
                    original_prompt = messages[0]["content"]  # Save original prompt
                    
                    for iteration in range(args.max_feedback_iterations):
                        # Evaluate the current code
                        evaluation = evaluate_generated_code(
                            current_code, 
                            misconception, 
                            eval_template, 
                            eval_client
                        )
                        evaluation_history.append(evaluation)
                        
                        # Check if feedback suggests improvement
                        if evaluation.get("parse_success") and evaluation.get("feedback"):
                            feedback_text = evaluation["feedback"].strip()
                            
                            # Check if feedback is "NONE" (no improvements needed)
                            if feedback_text.upper() == "NONE":
                                final_evaluation = evaluation
                                break
                            
                            # Feedback exists and is not NONE - regenerate
                            feedback_iterations += 1
                            print(f"  📝 Iteration {feedback_iterations}: Regenerating based on feedback")
                            
                            try:
                                # Create multi-turn conversation with feedback
                                feedback_messages = create_feedback_message(
                                    original_prompt,
                                    current_response,
                                    feedback_text
                                )
                                
                                # Regenerate with feedback
                                kwargs = get_llm_kwargs(args)
                                
                                if hasattr(llm_client, 'create_message'):
                                    if args.llm == "anthropic":
                                        reasoning = kwargs.pop("reasoning", False)
                                        budget_tokens = kwargs.pop("budget_tokens", 1000)
                                        new_response = llm_client.create_message(feedback_messages, kwargs=kwargs, reasoning=reasoning, budget_tokens=budget_tokens)
                                    elif args.llm == "openai":
                                        reasoning = kwargs.pop("reasoning", False)
                                        reasoning_effort = kwargs.pop("reasoning_effort", "medium")
                                        new_response = llm_client.create_message(feedback_messages, kwargs=kwargs, reasoning=reasoning, reasoning_effort=reasoning_effort)
                                    elif args.llm == "gemini":
                                        reasoning = kwargs.pop("reasoning", False)
                                        thinking_budget = kwargs.pop("thinking_budget", 1000)
                                        new_response = llm_client.create_message(feedback_messages, kwargs=kwargs, reasoning=reasoning, thinking_budget=thinking_budget)
                                    else:
                                        new_response = llm_client.create_message(feedback_messages, kwargs=kwargs)
                                else:
                                    # Fallback to batch with single item
                                    if args.llm == "anthropic":
                                        reasoning = kwargs.pop("reasoning", False)
                                        budget_tokens = kwargs.pop("budget_tokens", 1000)
                                        responses = llm_client.create_batch_messages([feedback_messages], reasoning=reasoning, budget_tokens=budget_tokens, **kwargs)
                                    elif args.llm == "openai":
                                        reasoning = kwargs.pop("reasoning", False)
                                        reasoning_effort = kwargs.pop("reasoning_effort", "medium")
                                        responses = llm_client.create_batch_messages([feedback_messages], reasoning=reasoning, reasoning_effort=reasoning_effort, **kwargs)
                                    elif args.llm == "gemini":
                                        reasoning = kwargs.pop("reasoning", False)
                                        thinking_budget = kwargs.pop("thinking_budget", 1000)
                                        responses = llm_client.create_batch_messages([feedback_messages], reasoning=reasoning, thinking_budget=thinking_budget, **kwargs)
                                    else:
                                        responses = llm_client.create_batch_messages([feedback_messages], **kwargs)
                                    new_response = responses[0]
                                
                                # Parse the new response
                                new_parsed = parse_llm_response(new_response, remove_comments=not args.keep_comments)
                                
                                # Update current code and response for next iteration
                                if new_parsed.get("code") and new_parsed["code"] != "NONE":
                                    current_code = new_parsed["code"]
                                    current_response = new_response
                                    parsed_response = new_parsed  # Update final parsed response
                                else:
                                    # If regeneration failed, keep previous version
                                    print(f"  ⚠️  Regeneration produced NONE or invalid code, keeping previous version")
                                    break
                                    
                            except Exception as e:
                                print(f"  ❌ Error during regeneration: {e}")
                                break
                        else:
                            # No valid feedback or parse failed
                            final_evaluation = evaluation
                            break
                    
                    # Do a final evaluation if we made changes
                    if feedback_iterations > 0 and not final_evaluation:
                        final_evaluation = evaluate_generated_code(
                            parsed_response["code"],
                            misconception,
                            eval_template,
                            eval_client
                        )
                        evaluation_history.append(final_evaluation)
            
            result = {
                "metadata": metadata,
                "parsed_response": parsed_response,
                "feedback_loop": {
                    "enabled": args.enable_feedback_loop,
                    "iterations": feedback_iterations,
                    "evaluation_history": evaluation_history,
                    "final_evaluation": final_evaluation
                }
            }
            all_results.append(result)
    
    # Save results
    print("Saving results...")
    
    # Prepare LLM info for summary
    llm_info = {
        "provider": args.llm,
        "model": get_model_name(args, llm_client),
        "processing_mode": "batch" if args.use_batch else "individual",
        "reasoning_enabled": args.reasoning,
        "reasoning_effort": args.reasoning_effort if args.reasoning and args.llm == "openai" else None,
        "max_problems_limit": args.max_problems,
        "max_misconceptions_limit": args.max_misconceptions,
        "max_solutions_per_problem_limit": args.max_solutions_per_problem,
        "examples_included": not args.exclude_examples,
        "comments_removed": not args.keep_comments
    }
    
    # Prepare prompt info for summary
    prompt_info = {
        "template_path": args.prompt_template,
        "template_content": template_content,
        "method_description": get_prompting_method_description(args.prompt_template, args.reasoning)
    }
    
    save_results(all_results, args.output_dir, llm_info, prompt_info)
    
    print("✅ Misconception generation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 