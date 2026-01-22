#!/usr/bin/env python3
"""
McMiner-S Streamlit Interface

An interactive web app for identifying programming misconceptions in student code
using the McMiner-S single-instance mining approach.
"""

import streamlit as st
import os
import sys
import re
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add parent directories to path for imports
current_dir = Path(__file__).resolve().parent
mining_dir = current_dir
project_root = mining_dir.parent

# Add project root to path so we can import shared module
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from utils.llm_clients import OpenAIClient, AnthropicClient, GeminiClient
except ImportError as e:
    print(f"Error importing shared.llm_clients: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Mining directory: {mining_dir}")
    print(f"Project root: {project_root}")
    print(f"sys.path: {sys.path[:3]}")
    raise





def load_prompt_template() -> str:
    """Load the zeroshot-no-reasoning mining prompt template."""
    template_path = mining_dir / "prompt_templates" / "mining" / "zeroshot-no-reasoning.md"
    
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    # Remove markdown header if present
    lines = template_content.split('\n')
    if lines and lines[0].startswith('#'):
        lines = lines[1:]
        if lines and lines[0].strip() == '':
            lines = lines[1:]
    
    return '\n'.join(lines)


def create_mining_prompt(template: str, problem_title: str, problem_description: str, student_code: str) -> str:
    """Create a prompt for inferring misconceptions from code."""
    prompt = template
    prompt = prompt.replace("{problem_title}", problem_title)
    prompt = prompt.replace("{problem_description}", problem_description)
    prompt = prompt.replace("{student_code}", student_code)
    return prompt


def parse_mining_response(response: str) -> Dict[str, Any]:
    """Parse LLM response for inferred misconceptions."""
    result = {
        "reasoning": "",
        "misconception_found": False,
        "description": "",
        "explanation": "",
        "raw_response": response,
        "parse_success": False
    }
    
    try:
        # Extract reasoning (if present)
        reasoning_match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', response, re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        
        # Extract misconception
        misconception_match = re.search(r'<misconception>\s*(.*?)\s*</misconception>', response, re.DOTALL)
        if misconception_match:
            misconception_text = misconception_match.group(1).strip()
            
            if misconception_text.upper() == "NONE":
                result["misconception_found"] = False
                result["description"] = "No misconception detected"
            else:
                result["misconception_found"] = True
                
                # Extract description
                desc_match = re.search(r'<description>\s*(.*?)\s*</description>', misconception_text, re.DOTALL)
                if desc_match:
                    result["description"] = desc_match.group(1).strip()
                
                # Extract explanation
                explanation_match = re.search(r'<explanation>\s*(.*?)\s*</explanation>', misconception_text, re.DOTALL)
                if explanation_match:
                    result["explanation"] = explanation_match.group(1).strip()
            
            result["parse_success"] = True
            
    except Exception as e:
        result["parse_error"] = str(e)
    
    return result


def create_llm_client(provider: str, model: str):
    """Create the appropriate LLM client."""
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            st.error("❌ ANTHROPIC_API_KEY not found in environment variables")
            return None
        return AnthropicClient(api_key=api_key)
    
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ OPENAI_API_KEY not found in environment variables")
            return None
        return OpenAIClient(api_key=api_key)
    
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("❌ GEMINI_API_KEY not found in environment variables")
            return None
        return GeminiClient(api_key=api_key, model=model)
    
    return None


def get_llm_config(provider: str, model: str) -> Dict[str, Any]:
    """Get LLM configuration including reasoning settings."""
    # Reasoning-capable models
    anthropic_reasoning = ["claude-sonnet-4-5", "claude-sonnet-4-0", "claude-opus-4-1", "claude-3-7-sonnet-latest"]
    openai_reasoning = ["o3-mini", "o1-mini", "o1", "o1-preview", "o3", "o4-mini"]
    gemini_reasoning = ["gemini-2.5-flash", "gemini-2.5-pro-preview-06-05"]
    
    config = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 4000,
        "reasoning": False,
    }
    
    # Enable reasoning for compatible models
    if provider == "anthropic" and model in anthropic_reasoning:
        config["reasoning"] = True
        config["budget_tokens"] = 2000
    elif provider == "openai" and model in openai_reasoning:
        config["reasoning"] = True
        config["reasoning_effort"] = "medium"
    elif provider == "gemini" and model in gemini_reasoning:
        config["reasoning"] = True
        config["thinking_budget"] = 2000
    
    return config


def analyze_code(provider: str, model: str, problem_title: str, problem_description: str, student_code: str):
    """Analyze student code for misconceptions."""
    # Load template
    template = load_prompt_template()
    
    # Create prompt
    prompt = create_mining_prompt(template, problem_title, problem_description, student_code)
    
    # Create LLM client
    client = create_llm_client(provider, model)
    if not client:
        return None
    
    # Get configuration
    config = get_llm_config(provider, model)
    
    # Create messages
    messages = [{"role": "user", "content": prompt}]
    
    # Call LLM
    try:
        with st.spinner(f"Analyzing with {model}..."):
            if provider == "anthropic":
                response = client.create_message(
                    messages,
                    kwargs={"model": config["model"], "temperature": config["temperature"], "max_tokens": config["max_tokens"]},
                    reasoning=config.get("reasoning", False),
                    budget_tokens=config.get("budget_tokens", 2000)
                )
            elif provider == "openai":
                response = client.create_message(
                    messages,
                    kwargs={"model": config["model"], "temperature": config["temperature"], "max_tokens": config["max_tokens"]},
                    reasoning=config.get("reasoning", False),
                    reasoning_effort=config.get("reasoning_effort", "medium")
                )
            elif provider == "gemini":
                response = client.create_message(
                    messages,
                    kwargs={"max_output_tokens": config["max_tokens"], "temperature": config["temperature"]},
                    reasoning=config.get("reasoning", False),
                    thinking_budget=config.get("thinking_budget", 2000)
                )
            else:
                st.error(f"Unsupported provider: {provider}")
                return None
        
        # Parse response
        parsed = parse_mining_response(response)
        return parsed
        
    except Exception as e:
        st.error(f"Error analyzing code: {str(e)}")
        return None


def main():
    st.set_page_config(
        page_title="McMiner-S: Misconception Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 McMiner-S: Programming Misconception Detector")
    st.markdown("Identify potential programming misconceptions in student Python code using LLMs.")
    
    # Sidebar for model selection
    st.sidebar.header("⚙️ Model Settings")
    
    # Model selection
    model_options = {
        "Anthropic": {
            "claude-sonnet-4-5": "Claude Sonnet 4.5 (Default)",
            "claude-sonnet-4-0": "Claude Sonnet 4.0",
            "claude-opus-4-1": "Claude Opus 4.1",
            "claude-3-7-sonnet-latest": "Claude 3.7 Sonnet",
        },
        "OpenAI": {
            "o3-mini": "o3-mini",
            "o1-mini": "o1-mini",
            "gpt-4o": "GPT-4o",
            "gpt-4o-mini": "GPT-4o Mini",
        },
        "Gemini": {
            "gemini-2.5-flash": "Gemini 2.5 Flash",
            "gemini-2.5-pro-preview-06-05": "Gemini 2.5 Pro Preview",
        }
    }
    
    provider = st.sidebar.selectbox(
        "Provider",
        options=list(model_options.keys()),
        index=0  # Default to Anthropic
    )
    
    model = st.sidebar.selectbox(
        "Model",
        options=list(model_options[provider].keys()),
        format_func=lambda x: model_options[provider][x],
        index=0  # Default to first model in provider
    )
    
    # Show reasoning status
    config = get_llm_config(provider.lower(), model)
    if config.get("reasoning"):
        st.sidebar.success("✅ Reasoning enabled for this model")
    else:
        st.sidebar.info("ℹ️ Reasoning not available for this model")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**About McMiner-S**")
    st.sidebar.markdown("McMiner-S uses single-instance mining to identify programming misconceptions from individual code samples.")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input")
        
        # Problem title
        problem_title = st.text_input(
            "Problem Title",
            value="Factorial Function",
            help="Short title for the programming problem"
        )
        
        # Problem description
        problem_description = st.text_area(
            "Problem Description",
            value="Write the factorial(n) function that computes the factorial n! of a natural number n. Additionally, if the input n is negative, the function should return 0.",
            height=150,
            help="Detailed description of what the code should accomplish"
        )
        
        # Student code
        student_code = st.text_area(
            "Student Code",
            value="""def factorial(n):
    if n < 0:
        return 0
    fact = 1
    for i in range(n):
        fact = fact * i
    return fact""",
            height=300,
            help="The Python code to analyze for misconceptions"
        )
        
        # Analyze button
        analyze_button = st.button("🔍 Analyze Code", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if analyze_button:
            if not student_code.strip():
                st.warning("⚠️ Please enter some student code to analyze.")
            elif not problem_description.strip():
                st.warning("⚠️ Please enter a problem description.")
            else:
                result = analyze_code(
                    provider.lower(),
                    model,
                    problem_title,
                    problem_description,
                    student_code
                )
                
                if result:
                    if result["parse_success"]:
                        if result["misconception_found"]:
                            st.error("🐛 Misconception Detected")
                            
                            st.markdown("**Description:**")
                            st.info(result["description"])
                            
                            st.markdown("**Explanation:**")
                            st.markdown(result["explanation"])
                            
                            if result["reasoning"]:
                                with st.expander("🧠 Model Reasoning (Click to expand)"):
                                    st.markdown(result["reasoning"])
                        else:
                            st.success("✅ No Misconception Detected")
                            st.markdown("The code appears to be correct or contains only minor issues that don't indicate a programming misconception.")
                            
                            if result["reasoning"]:
                                with st.expander("🧠 Model Reasoning (Click to expand)"):
                                    st.markdown(result["reasoning"])
                    else:
                        st.error("❌ Failed to parse model response")
                        with st.expander("View Raw Response"):
                            st.code(result["raw_response"])
        else:
            st.info("👈 Enter code and click 'Analyze Code' to get started")
            
            st.markdown("---")
            st.markdown("**Example Misconception:**")
            st.markdown("The default example code exhibits a common misconception about Python's `range()` function:")
            st.info("**The student believes** that `range(n)` produces values from 1 to n inclusive, when in fact it produces values from 0 to n-1.")


if __name__ == "__main__":
    main()

