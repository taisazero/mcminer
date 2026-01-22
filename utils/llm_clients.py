from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Union
from openai import OpenAI
import json
import anthropic
from google import genai
import time
from pathlib import Path
import requests
from typing import Optional
from tqdm import tqdm
import tempfile
import os
import subprocess


class BaseLLMClient(ABC):
    @abstractmethod
    async def create_message(self, messages: List[Dict[str, str]], with_tools: bool = False) -> Any:
        pass

    @abstractmethod
    def get_tool_info(self, tool: Dict[str, Any]) -> Dict[str, str]:
        pass

    @abstractmethod
    def process_tool_result(self, tool_use: Any) -> Tuple[str, Dict[str, Any]]:
        pass
    
    @abstractmethod
    def create_batch_messages(self, message_batches: List[List[Dict[str, str]]], **kwargs) -> List[Any]:
        """Process multiple message conversations in batch for efficiency
        
        Common parameters:
            polling_interval: Seconds between status checks (default: 1800 = 30 minutes)
            max_duration: Maximum time to wait in seconds (default: 172800 = 48 hours)
        """
        pass



class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gpt-5", base_url: str = None, use_responses_api: bool = True):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model identifier (default: gpt-5)
            base_url: Optional base URL for API calls
            use_responses_api: If True, use Responses API by default. If False, use Chat Completions API.
                             Responses API is recommended for GPT-5 and provides better performance.
        """
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        self.model = model
        self.use_responses_api = use_responses_api

    def create_message(self, messages: List[Dict[str, str]], tools: List = None, schema: Union[str, Dict[str, Any]] = None, kwargs: dict = None, reasoning: bool = False, reasoning_effort: str = "medium", verbosity: str = "medium", use_responses_api: bool = None) -> Any:
        """
        Create a message using either Responses API or Chat Completions API.
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions
            schema: Optional schema for structured outputs
            kwargs: Additional parameters for the API call
            reasoning: Whether to enable reasoning mode (for Anthropic/Gemini, ignored for GPT-5/o-series)
            reasoning_effort: Reasoning effort level - 'minimal', 'low', 'medium', 'high' (GPT-5) or 'low', 'medium', 'high' (o-series)
            verbosity: Text verbosity level - 'low', 'medium', 'high' (GPT-5 only)
            use_responses_api: Override default API choice. If None, uses instance default.
        """
        # Determine which API to use
        if use_responses_api is None:
            use_responses_api = self.use_responses_api
        
        # Set default kwargs
        if kwargs is None:
            kwargs = {
                "model": self.model,
                "max_tokens": 4000,
            }
        else:
            kwargs.setdefault("model", self.model)
            kwargs.setdefault("max_tokens", 4000)
        
        model_name = kwargs["model"]
        
        # Check if this is a reasoning model
        is_o_series_model = any(model_name.startswith(prefix) for prefix in ["o1", "o3", "o4"])
        is_gpt5_model = any(model_name.startswith(prefix) for prefix in ["gpt-5"])
        
        # GPT-5 models MUST use Responses API (they're reasoning models)
        # o-series models MUST use Chat Completions API
        if is_gpt5_model:
            # GPT-5 always uses Responses API and doesn't support temperature
            return self._create_message_responses_api(messages, tools, schema, kwargs, reasoning_effort, verbosity)
        elif is_o_series_model:
            # o-series uses Chat Completions API
            return self._create_message_chat_completions(messages, tools, schema, kwargs, reasoning, reasoning_effort, True)
        else:
            # Other models: use API preference, set temperature if not set
            if "temperature" not in kwargs:
                kwargs["temperature"] = 1.0
            if use_responses_api:
                return self._create_message_responses_api(messages, tools, schema, kwargs, reasoning_effort, verbosity)
            else:
                return self._create_message_chat_completions(messages, tools, schema, kwargs, reasoning, reasoning_effort, False)
    
    def _create_message_responses_api(self, messages: List[Dict[str, str]], tools: List = None, schema: Union[str, Dict[str, Any]] = None, kwargs: dict = None, reasoning_effort: str = "medium", verbosity: str = "medium") -> Any:
        """Create a message using the Responses API."""
        # Separate system instruction from messages
        system_instruction = None
        input_messages = []
        
        for message in messages:
            if message.get("role") == "system":
                system_instruction = message.get("content", "")
            else:
                input_messages.append(message)
        
        model_name = kwargs["model"]
        is_gpt5_model = any(model_name.startswith(prefix) for prefix in ["gpt-5"])
        
        # Build request parameters
        request_params = {
            "model": model_name,
            "input": input_messages if len(input_messages) > 1 else (input_messages[0]["content"] if input_messages else ""),
        }
        
        # Add system instruction if present
        if system_instruction:
            request_params["instructions"] = system_instruction
        
        # Add tools if provided (convert to Responses API format)
        if tools:
            converted_tools = []
            for tool in tools:
                if "function" in tool:
                    # Convert from Chat Completions format to Responses format
                    converted_tools.append({
                        "type": "function",
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "parameters": tool["function"].get("parameters", {})
                    })
                else:
                    converted_tools.append(tool)
            request_params["tools"] = converted_tools
        
        # Add structured outputs if provided (use text.format instead of response_format)
        if schema:
            if isinstance(schema, dict):
                if "text" not in request_params:
                    request_params["text"] = {}
                request_params["text"]["format"] = schema
            else:
                # If schema is a string, assume it's a JSON schema
                if "text" not in request_params:
                    request_params["text"] = {}
                request_params["text"]["format"] = {
                    "type": "json_schema",
                    "json_schema": json.loads(schema) if isinstance(schema, str) else schema
                }
        
        # GPT-5 specific parameters
        if is_gpt5_model:
            # GPT-5 always has reasoning - control effort level
            request_params["reasoning"] = {"effort": reasoning_effort}
            
            # Add verbosity control
            if "text" not in request_params:
                request_params["text"] = {}
            request_params["text"]["verbosity"] = verbosity
            
            # GPT-5 doesn't support temperature, top_p, logprobs
            # Use max_output_tokens instead of max_tokens
            request_params["max_output_tokens"] = kwargs.get("max_tokens", 4000)
        else:
            # Non-GPT-5 models
            request_params["max_output_tokens"] = kwargs.get("max_tokens", 4000)
            
            # Add temperature if model supports it
            if "temperature" in kwargs:
                request_params["temperature"] = kwargs["temperature"]
        
        # Make the API call
        response = self.client.responses.create(**request_params)
        
        # Handle tool calls
        if tools and hasattr(response, 'output'):
            for item in response.output:
                if hasattr(item, 'type') and item.type == 'function_call':
                    return {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": item.name,
                                    "arguments": item.arguments,
                                }
                            }
                        ]
                    }
        
        # Extract text content with reasoning if present
        if hasattr(response, 'output_text'):
            # Check if there's reasoning content in the output
            reasoning_content = ""
            text_content = response.output_text
            
            if hasattr(response, 'output'):
                for item in response.output:
                    if hasattr(item, 'type') and item.type == 'reasoning':
                        if hasattr(item, 'summary') and item.summary:
                            for summary_item in item.summary:
                                if hasattr(summary_item, 'text'):
                                    reasoning_content += summary_item.text
            
            # Combine reasoning and text if both present
            if reasoning_content:
                return f"<reasoning>\n{reasoning_content}\n</reasoning>\n\n{text_content}"
            return text_content
        
        # Fallback to string representation
        return str(response)
    
    def _create_message_chat_completions(self, messages: List[Dict[str, str]], tools: List = None, schema: Union[str, Dict[str, Any]] = None, kwargs: dict = None, reasoning: bool = False, reasoning_effort: str = "medium", is_reasoning_model: bool = False) -> Any:
        """Create a message using the Chat Completions API."""
        # Build request parameters
        request_params = {
            "model": kwargs["model"],
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4000),
            "temperature": kwargs.get("temperature", 0.1),
        }
        
        # Handle reasoning models (o1, o3, o4 series)
        if is_reasoning_model:
            # For reasoning models, use max_completion_tokens instead of max_tokens
            base_tokens = request_params.pop("max_tokens")
            if reasoning:
                # Allocate different amounts based on reasoning effort level
                effort_multipliers = {
                    "low": 3000,
                    "medium": 5000,
                    "high": 8000
                }
                extra_tokens = effort_multipliers.get(reasoning_effort, 5000)
                request_params["max_completion_tokens"] = base_tokens + extra_tokens
            else:
                request_params["max_completion_tokens"] = base_tokens
            
            # Add reasoning effort
            if reasoning:
                request_params["reasoning_effort"] = reasoning_effort
            
            # Remove unsupported parameters
            unsupported_params = ["temperature", "top_p", "presence_penalty", "frequency_penalty", 
                                "logprobs", "top_logprobs", "logit_bias"]
            for param in unsupported_params:
                request_params.pop(param, None)

        # Add tools if provided
        if tools:
            request_params["tools"] = tools

        # Add schema if provided
        if schema:
            request_params["extra_body"] = {
                "guided_json": schema,
                "guided_decoding_backend": "lm-format-enforcer",
            }
        
        # Make the API call
        response = self.client.chat.completions.create(**request_params)
        
        # Handle reasoning models response parsing
        if is_reasoning_model:
            response_content = response.choices[0].message.content
            
            # Check if we have reasoning token information
            if (reasoning and hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens_details') 
                and hasattr(response.usage.completion_tokens_details, 'reasoning_tokens') 
                and response.usage.completion_tokens_details.reasoning_tokens > 0):
                reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
                reasoning_content = f"The model used {reasoning_tokens} reasoning tokens to process this request."
                return f"<reasoning>\n{reasoning_content}\n</reasoning>\n\n{response_content}"
            
                return response_content
        
        # Handle tool calls
        response_content = response.choices[0].message.content
        if tools and response.choices[0].message.tool_calls:
            return {
                "tool_calls": [
                    {
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                        }
                    }
                    for tool_call in response.choices[0].message.tool_calls
                ]
            }
        elif tools:
            tool_call = None
            # Check if response_content contains indicators of a tool call
            if response_content and isinstance(response_content, str) and '"name":' in response_content:
                cleaned_content = response_content.strip()
                for token in ["<|eom_id|>", "<|eot_id|>"]:
                    cleaned_content = cleaned_content.replace(token, "")
                
                try:
                    tool_call = json.loads(cleaned_content)
                except json.decoder.JSONDecodeError:
                    try:
                        tool_call = json.loads(cleaned_content + "}")
                    except json.decoder.JSONDecodeError:
                        pass
            
            if tool_call is not None:
                return {
                    "tool_calls": [
                        {
                            "function": {
                                "name": tool_call["name"],
                                "arguments": tool_call["parameters"],
                            }
                        }
                    ]
                }
            
        return response_content
    
    def get_tool_info(self, tool: Dict[str, Any]) -> Dict[str, str]:
        return {
            "name": tool["function"]["name"],
            "description": tool["function"]["description"]
        }

    def process_tool_result(self, tool_use: Any) -> Tuple[str, Dict[str, Any]]:
        return tool_use['function']['name'], json.loads(tool_use['function']['arguments'])
    
    def create_batch_messages(self, message_batches: List[List[Dict[str, str]]], reasoning: bool = False, reasoning_effort: str = "medium", verbosity: str = "medium", polling_interval: int = 1800, max_duration: int = 172800, use_responses_api: bool = None, **kwargs) -> List[Any]:
        """
        Process multiple message conversations using OpenAI's Batch API.
        
        Args:
            message_batches: List of message conversations to process
            reasoning: Whether to enable reasoning mode (for Anthropic/Gemini, ignored for GPT-5/o-series)
            reasoning_effort: Reasoning effort level - 'minimal', 'low', 'medium', 'high' (GPT-5) or 'low', 'medium', 'high' (o-series)
            verbosity: Text verbosity level - 'low', 'medium', 'high' (GPT-5 only)
            polling_interval: Seconds between status checks (default: 1800 = 30 minutes)
            max_duration: Maximum time to wait in seconds (default: 172800 = 48 hours)
            use_responses_api: Override default API choice. If None, uses instance default.
            **kwargs: Additional parameters for the API call
        """
        # Determine which API to use
        if use_responses_api is None:
            use_responses_api = self.use_responses_api
        
        model_name = kwargs.get("model", self.model)
        is_o_series_model = any(model_name.startswith(prefix) for prefix in ["o1", "o3", "o4"])
        is_gpt5_model = any(model_name.startswith(prefix) for prefix in ["gpt-5"])
        
        # GPT-5 models MUST use Responses API, o-series MUST use Chat Completions
        if is_gpt5_model:
            return self._create_batch_messages_responses_api(message_batches, reasoning_effort, verbosity, polling_interval, max_duration, **kwargs)
        elif is_o_series_model:
            return self._create_batch_messages_chat_completions(message_batches, reasoning, reasoning_effort, polling_interval, max_duration, **kwargs)
        else:
            # Other models: use API preference
            if use_responses_api:
                return self._create_batch_messages_responses_api(message_batches, reasoning_effort, verbosity, polling_interval, max_duration, **kwargs)
            else:
                return self._create_batch_messages_chat_completions(message_batches, reasoning, reasoning_effort, polling_interval, max_duration, **kwargs)
    
    def _create_batch_messages_responses_api(self, message_batches: List[List[Dict[str, str]]], reasoning_effort: str = "medium", verbosity: str = "medium", polling_interval: int = 1800, max_duration: int = 172800, **kwargs) -> List[Any]:
        """Process multiple message conversations using OpenAI's Responses Batch API."""
        try:
            # Prepare batch input for Responses API
            batch_requests = []
            model_name = kwargs.get("model", self.model)
            is_gpt5_model = any(model_name.startswith(prefix) for prefix in ["gpt-5"])
            
            for i, messages in enumerate(message_batches):
                # Separate system instruction from messages
                system_instruction = None
                input_messages = []
                
                for message in messages:
                    if message.get("role") == "system":
                        system_instruction = message.get("content", "")
                    else:
                        input_messages.append(message)
                
                # Build request body for Responses API
                body = {
                    "model": kwargs.get("model", self.model),
                    "input": input_messages if len(input_messages) > 1 else (input_messages[0]["content"] if input_messages else ""),
                }
                
                # Add system instruction if present
                if system_instruction:
                    body["instructions"] = system_instruction
                
                # GPT-5 specific parameters
                if is_gpt5_model:
                    # GPT-5 always has reasoning - control effort level
                    body["reasoning"] = {"effort": reasoning_effort}
                    
                    # Add verbosity control
                    body["text"] = {"verbosity": verbosity}
                    
                    # GPT-5 doesn't support temperature
                    body["max_output_tokens"] = kwargs.get("max_tokens", 4000)
                else:
                    # Non-GPT-5 models
                    body["max_output_tokens"] = kwargs.get("max_tokens", 4000)
                    
                    # Add temperature if supported
                    if "temperature" in kwargs:
                        body["temperature"] = kwargs.get("temperature")
                
                batch_requests.append({
                    "custom_id": str(i),
                    "method": "POST", 
                    "url": "/v1/responses",
                    "body": body
                })
            
            # Write batch to a temp file
            with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".jsonl") as f:
                for req in batch_requests:
                    f.write(json.dumps(req) + "\n")
                batch_file_path = f.name
            
            # Upload batch file
            with open(batch_file_path, "rb") as batch_file:
                batch_input_file = self.client.files.create(
                    file=batch_file,
                    purpose="batch"
                )
            
            # Create batch
            batch = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/responses",
                completion_window="24h"
            )
            
            print(f"OpenAI Responses batch created: {batch.id}")
            print(f"Polling every {polling_interval} seconds (max duration: {max_duration} seconds)")
            
            # Poll for completion
            start_time = time.time()
            while batch.status not in ("completed", "failed", "expired", "cancelled", "cancelling"):
                elapsed_time = time.time() - start_time
                if elapsed_time > max_duration:
                    print(f"Batch processing exceeded maximum duration of {max_duration} seconds")
                    return [f"Batch timeout after {max_duration} seconds"] * len(message_batches)
                
                print(f"Batch status: {batch.status}, waiting {polling_interval} seconds... (elapsed: {int(elapsed_time)}s)")
                time.sleep(polling_interval)
                batch = self.client.batches.retrieve(batch.id)
            
            if batch.status != "completed":
                error_msg = f"Batch failed with status: {batch.status}"
                print(f"❌ {error_msg}")
                raise Exception(error_msg)
            
            # Download results
            output_file_id = batch.output_file_id
            result = self.client.files.content(output_file_id)
            
            # Parse results and map by custom_id
            results = {}
            succeeded_count = 0
            errored_count = 0
            
            for line in result.content.decode('utf-8').strip().split('\n'):
                if line.strip():
                    obj = json.loads(line)
                    if "custom_id" in obj:
                        # Check for errors
                        if obj.get("error"):
                            errored_count += 1
                            error = obj["error"]
                            error_msg = error.get("message", str(error))
                            error_code = error.get("code", "unknown_error")
                            print(f"Error for request {obj['custom_id']}: {error_msg}")
                            results[obj["custom_id"]] = f"Error: {error_code} - {error_msg}"
                        # Extract response content
                        elif obj.get("response") and obj["response"].get("body"):
                            response_body = obj["response"]["body"]
                            if response_body.get("error"):
                                errored_count += 1
                                error = response_body["error"]
                                error_msg = error.get("message", str(error))
                                error_code = error.get("code", "unknown_error")
                                print(f"Error for request {obj['custom_id']}: {error_msg}")
                                results[obj["custom_id"]] = f"Error: {error_code} - {error_msg}"
                            elif response_body.get("output"):
                                succeeded_count += 1
                                # Extract text from output items
                                output_text = ""
                                reasoning_content = ""
                                
                                for item in response_body["output"]:
                                    if item.get("type") == "reasoning" and item.get("summary"):
                                        for summary_item in item["summary"]:
                                            if summary_item.get("text"):
                                                reasoning_content += summary_item["text"]
                                    elif item.get("type") == "message" and item.get("content"):
                                        for content_item in item["content"]:
                                            if content_item.get("type") == "output_text":
                                                output_text += content_item.get("text", "")
                                
                                # Combine reasoning and text if both present
                                if reasoning_content:
                                    content = f"<reasoning>\n{reasoning_content}\n</reasoning>\n\n{output_text}"
                                else:
                                    content = output_text
                                
                                results[obj["custom_id"]] = content
                            else:
                                errored_count += 1
                                results[obj["custom_id"]] = f"Error: Unexpected response format - {str(response_body)}"
                        else:
                            errored_count += 1
                            results[obj["custom_id"]] = f"Error: No response body - {str(obj)}"
            
            # Clean up temp file
            os.unlink(batch_file_path)
            
            print(f"Results summary: {succeeded_count} succeeded, {errored_count} errored")
            
            # Check if all requests failed
            total_requests = len(message_batches)
            if errored_count == total_requests:
                error_msg = f"All {total_requests} batch requests failed"
                print(f"❌ {error_msg}")
                raise Exception(error_msg)
            elif errored_count > 0:
                print(f"⚠️  Warning: {errored_count}/{total_requests} requests failed")
            
            # Return in input order
            return [results.get(str(i), f"No result for index {i}") for i in range(len(message_batches))]
            
        except Exception as e:
            print(f"OpenAI Responses batch processing failed: {e}")
            # Fallback to sequential processing
            print("Falling back to sequential processing...")
            responses = []
            with tqdm(total=len(message_batches), desc="Processing OpenAI Responses requests (sequential)", unit="req") as pbar:
                for i, messages in enumerate(message_batches):
                    try:
                        response = self.create_message(messages, kwargs=kwargs, reasoning_effort=reasoning_effort, verbosity=verbosity, use_responses_api=True)
                        responses.append(response)
                    except Exception as e:
                        print(f"Error processing batch item {i}: {e}")
                        responses.append(f"Error: {str(e)}")
                    pbar.update(1)
            return responses
    
    def _create_batch_messages_chat_completions(self, message_batches: List[List[Dict[str, str]]], reasoning: bool = False, reasoning_effort: str = "medium", polling_interval: int = 1800, max_duration: int = 172800, **kwargs) -> List[Any]:
        """Process multiple message conversations using OpenAI's Chat Completions Batch API."""
        try:
            # Prepare batch input as per OpenAI Batch API
            batch_requests = []
            for i, messages in enumerate(message_batches):
                model_name = kwargs.get("model", self.model)
                is_reasoning_model = any(model_name.startswith(prefix) for prefix in ["o1", "o3", "o4"])
                
                body = {
                    "model": model_name,
                    "messages": messages,
                }
                
                if is_reasoning_model:
                    # For reasoning models, use max_completion_tokens with extra allocation
                    base_tokens = kwargs.get("max_tokens", 4000)
                    if reasoning:
                        # Allocate different amounts based on reasoning effort level
                        effort_multipliers = {
                            "low": 3000,      # Conservative for low effort
                            "medium": 5000,   # Moderate for medium effort  
                            "high": 8000      # Generous for high effort
                        }
                        extra_tokens = effort_multipliers.get(reasoning_effort, 5000)
                        body["max_completion_tokens"] = base_tokens + extra_tokens
                        body["reasoning_effort"] = reasoning_effort
                    else:
                        body["max_completion_tokens"] = base_tokens
                else:
                    # For regular models, use max_tokens and temperature
                    body["max_tokens"] = kwargs.get("max_tokens", 4000)
                    body["temperature"] = kwargs.get("temperature", 0.1)
                
                batch_requests.append({
                    "custom_id": str(i),
                    "method": "POST", 
                    "url": "/v1/chat/completions",
                    "body": body
                })
            
            # Write batch to a temp file
            with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".jsonl") as f:
                for req in batch_requests:
                    f.write(json.dumps(req) + "\n")
                batch_file_path = f.name
            
            # Upload batch file
            with open(batch_file_path, "rb") as batch_file:
                batch_input_file = self.client.files.create(
                    file=batch_file,
                    purpose="batch"
                )
            
            # Create batch
            batch = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            print(f"OpenAI Chat Completions batch created: {batch.id}")
            print(f"Polling every {polling_interval} seconds (max duration: {max_duration} seconds)")
            
            # Poll for completion with configurable interval and max duration
            start_time = time.time()
            while batch.status not in ("completed", "failed", "expired", "cancelled", "cancelling"):
                elapsed_time = time.time() - start_time
                if elapsed_time > max_duration:
                    print(f"Batch processing exceeded maximum duration of {max_duration} seconds")
                    return [f"Batch timeout after {max_duration} seconds"] * len(message_batches)
                
                print(f"Batch status: {batch.status}, waiting {polling_interval} seconds... (elapsed: {int(elapsed_time)}s)")
                time.sleep(polling_interval)
                batch = self.client.batches.retrieve(batch.id)
            
            if batch.status != "completed":
                error_msg = f"Batch failed with status: {batch.status}"
                print(f"❌ {error_msg}")
                raise Exception(error_msg)
            
            # Download results
            output_file_id = batch.output_file_id
            result = self.client.files.content(output_file_id)
            
            # Parse results and map by custom_id
            results = {}
            succeeded_count = 0
            errored_count = 0
            
            for line in result.content.decode('utf-8').strip().split('\n'):
                if line.strip():
                    obj = json.loads(line)
                    if "custom_id" in obj:
                        # Check for errors in the response
                        if obj.get("error"):
                            errored_count += 1
                            error = obj["error"]
                            error_msg = error.get("message", str(error))
                            error_code = error.get("code", "unknown_error")
                            print(f"Error for request {obj['custom_id']}: {error_msg}")
                            results[obj["custom_id"]] = f"Error: {error_code} - {error_msg}"
                        # Extract the actual response content
                        elif obj.get("response") and obj["response"].get("body"):
                            response_body = obj["response"]["body"]
                            if response_body.get("error"):
                                errored_count += 1
                                error = response_body["error"]
                                error_msg = error.get("message", str(error))
                                error_code = error.get("code", "unknown_error")
                                print(f"Error for request {obj['custom_id']}: {error_msg}")
                                results[obj["custom_id"]] = f"Error: {error_code} - {error_msg}"
                            elif response_body.get("choices"):
                                succeeded_count += 1
                                content = response_body["choices"][0]["message"]["content"]
                                results[obj["custom_id"]] = content
                            else:
                                errored_count += 1
                                results[obj["custom_id"]] = f"Error: Unexpected response format - {str(response_body)}"
                        else:
                            errored_count += 1
                            results[obj["custom_id"]] = f"Error: No response body - {str(obj)}"
            
            # Clean up temp file
            os.unlink(batch_file_path)
            
            print(f"Results summary: {succeeded_count} succeeded, {errored_count} errored")
            
            # Check if all requests failed
            total_requests = len(message_batches)
            if errored_count == total_requests:
                error_msg = f"All {total_requests} batch requests failed"
                print(f"❌ {error_msg}")
                raise Exception(error_msg)
            elif errored_count > 0:
                print(f"⚠️  Warning: {errored_count}/{total_requests} requests failed")
            
            # Return in input order
            return [results.get(str(i), f"No result for index {i}") for i in range(len(message_batches))]
            
        except Exception as e:
            print(f"OpenAI Chat Completions batch processing failed: {e}")
            # Fallback to sequential processing
            print("Falling back to sequential processing...")
            responses = []
            with tqdm(total=len(message_batches), desc="Processing OpenAI Chat Completions requests (sequential)", unit="req") as pbar:
                for i, messages in enumerate(message_batches):
                    try:
                        response = self.create_message(messages, kwargs=kwargs, reasoning=reasoning, reasoning_effort=reasoning_effort, use_responses_api=False)
                        responses.append(response)
                    except Exception as e:
                        print(f"Error processing batch item {i}: {e}")
                        responses.append(f"Error: {str(e)}")
                    pbar.update(1)
            return responses
    

class VLLMClient(OpenAIClient):
    def __init__(self, api_key: str = "NONE", 
                 model: str = "",
                 model_name: str = "", 
                 base_url: str = "http://localhost:8000/v1",
                 offline_mode: bool = False,
                 **llm_kwargs
                 ) -> None:
        # CRITICAL: Set environment variables FIRST, before any vLLM imports
        # This must happen at the very start to prevent V1 engine initialization
        if offline_mode:
            # Force disable V1 engine (multiple methods for redundancy)
            os.environ['VLLM_USE_V1'] = '0'
            os.environ['VLLM_USE_V0'] = '1'  # Explicitly enable legacy engine
            os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'
            os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'  # Avoid fork issues
        
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._llm_instance = None  # For offline mode
        self._offline_mode = offline_mode
        
        # Support both 'model' and 'model_name' parameters for backward compatibility
        requested_model = model or model_name
        
        if offline_mode:
            # In offline mode, use the requested model directly
            if not requested_model:
                raise ValueError("Model name must be specified for offline mode")
            self.model_name = requested_model
            print(f"vLLM offline mode: will use model '{self.model_name}'")
            # Initialize offline mode
            self.init_offline_mode(**llm_kwargs)
        else:
            # Server mode: check server for available model
            available_model = self.get_llm_server_modelname()
            
            if requested_model:
                if requested_model != available_model:
                    print(f"Requested model '{requested_model}' is not available. Using the available model: '{available_model}'")
                self.model_name = requested_model
            elif available_model:
                self.model_name = available_model
                print(f"Language model name not set, using the available model: '{available_model}'")
            else:
                raise ValueError("No model is available on the VLLM server. Please ensure that the VLLM server is running and is serving a language model.")
        
        # Initialize OpenAI client parent class (only in server mode)
        if not offline_mode:
            super().__init__(api_key=api_key, model=self.model_name, base_url=base_url)
        else:
            # In offline mode, we don't need OpenAI client
            self.model = self.model_name

    def set_base_url(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        if not self._offline_mode:
            self.client.base_url = self.base_url
    
    def init_offline_mode(self, **llm_kwargs):
        """
        Initialize vLLM in offline mode for direct batch inference.
        This loads the model directly into GPU memory for optimal batch processing.
        
        Args:
            **llm_kwargs: Additional arguments passed to vLLM.LLM constructor, such as:
                - tensor_parallel_size: Number of GPUs to use (default: auto-detect from SLURM)
                - gpu_memory_utilization: GPU memory fraction to use (default: 0.9)
                - max_model_len: Maximum sequence length (default: model's max)
                - trust_remote_code: Whether to trust remote code (default: True for custom models)
                - dtype: Model dtype (default: 'auto')
                - disable_log_stats: Disable logging statistics (default: True)
        """
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM is required for offline mode. Install with: pip install vllm"
            )
        
        # Auto-detect GPU count from SLURM environment
        if 'tensor_parallel_size' not in llm_kwargs:
            gpu_count = self._get_gpu_count_from_slurm()
            if gpu_count > 1:
                llm_kwargs['tensor_parallel_size'] = gpu_count
                print(f"Auto-detected {gpu_count} GPUs from SLURM, using tensor parallelism")
        
        # Set default parameters optimized for batch inference
        llm_kwargs.setdefault('gpu_memory_utilization', 0.9)
        llm_kwargs.setdefault('trust_remote_code', True)
        llm_kwargs.setdefault('dtype', 'auto')
        llm_kwargs.setdefault('disable_log_stats', True)
        
        print(f"Initializing vLLM offline mode with model: {self.model_name}")
        print(f"vLLM parameters: {llm_kwargs}")
        print(f"Environment: VLLM_USE_V1={os.environ.get('VLLM_USE_V1')}, VLLM_ATTENTION_BACKEND={os.environ.get('VLLM_ATTENTION_BACKEND')}")
        
        self._llm_instance = LLM(
            model=self.model_name,
            **llm_kwargs
        )
        
        print("✓ vLLM model loaded successfully in offline mode")
    
    def _get_gpu_count_from_slurm(self) -> int:
        """Detect the number of GPUs allocated by SLURM."""
        # Check CUDA_VISIBLE_DEVICES (most reliable)
        cuda_devices = os.getenv('CUDA_VISIBLE_DEVICES')
        if cuda_devices:
            gpu_count = len([d for d in cuda_devices.split(',') if d.strip()])
            if gpu_count > 0:
                return gpu_count
        
        # Check SLURM_GPUS_ON_NODE
        slurm_gpus = os.getenv('SLURM_GPUS_ON_NODE')
        if slurm_gpus:
            try:
                return int(slurm_gpus)
            except ValueError:
                pass
        
        # Check SLURM_STEP_GPUS
        step_gpus = os.getenv('SLURM_STEP_GPUS')
        if step_gpus:
            return len([d for d in step_gpus.split(',') if d.strip()])
        
        # Default to 1 GPU
        return 1
    
    def _format_messages_as_prompt(self, messages: List[Dict[str, str]], enable_thinking: bool = True) -> str:
        """
        Format message list into a single prompt string using the model's chat template.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            enable_thinking: For Qwen3, controls thinking mode (default: True)
        
        Returns:
            Formatted prompt string
        """
        # Try to use the tokenizer's chat template if available
        if self._llm_instance and hasattr(self._llm_instance, 'llm_engine'):
            try:
                tokenizer = self._llm_instance.llm_engine.tokenizer
                if hasattr(tokenizer, 'apply_chat_template'):
                    # For Qwen3 models, pass enable_thinking parameter
                    # https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes
                    if 'qwen3' in self.model_name.lower():
                        return tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True,
                            enable_thinking=enable_thinking  # Qwen3 thinking mode control
                        )
                    else:
                        return tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
            except Exception as e:
                print(f"Warning: Could not use chat template: {e}")
                print("Falling back to simple concatenation")
        
        # Fallback: simple concatenation with role prefixes
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts) + "\n\nAssistant:"
        
    def create_message(self, messages: List[Dict[str, str]], tools: List = None, schema: Union[str, Dict[str, Any]] = None, kwargs: dict = None, reasoning: bool = False, reasoning_effort: str = "medium") -> Any:
        if self._offline_mode:
            # Use offline mode for single message
            if self._llm_instance is None:
                raise RuntimeError("Offline mode not initialized. Call init_offline_mode() first.")
            
            from vllm import SamplingParams
            
            # Set up kwargs with defaults
            if kwargs is None:
                kwargs = {}
            kwargs.setdefault("max_tokens", 4000)
            kwargs.setdefault("temperature", 0.1)
            kwargs.setdefault("top_p", 0.9)
            
            # Format messages as prompt
            prompt = self._format_messages_as_prompt(messages)
            
            # Set sampling parameters
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", 0.1),
                top_p=kwargs.get("top_p", 0.9),
                max_tokens=kwargs.get("max_tokens", 4000),
            )
            
            # Generate response
            outputs = self._llm_instance.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text
        else:
            # Use server mode
            if kwargs is None:
                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": 4000,
                    "temperature": 0.1,
                    "top_p": 0.9,
                }
            else:
                kwargs["model"] = self.model_name if "model" not in kwargs else kwargs["model"]
                # Always use the messages parameter, not whatever is in kwargs
                kwargs["messages"] = messages
                kwargs["max_tokens"] = 4000 if "max_tokens" not in kwargs else kwargs["max_tokens"]
                kwargs["temperature"] = 0.1 if "temperature" not in kwargs else kwargs["temperature"]
                kwargs["top_p"] = 0.9 if "top_p" not in kwargs else kwargs["top_p"]
            return super().create_message(messages, tools, schema, kwargs, reasoning, reasoning_effort)

    def get_llm_server_modelname(self) -> Optional[str]:
        base_url = self.base_url.replace("/v1", "").rstrip("/")
        try:
            if self.api_key:
                response = requests.get(
                    f"{base_url}/v1/models", headers={"Authorization": f"Bearer {self.api_key}"}
                )
            else:
                response = requests.get(f"{base_url}/v1/models")
            if response.status_code == 200:
                models = [m for m in response.json()["data"] if m["object"] == "model"]
                if len(models) == 0:
                    print("The vLLM server is running but not hosting any models.")
                    return None
                model_name = models[0]["id"]
                print(f"vLLM server is running. Selecting: {model_name}.")
                return model_name
            else:
                print(f"vLLM server is running but could not get the list of models. Status code: {response.status_code}")
                return None
        except requests.exceptions.ConnectionError:
            print("No vLLM server running at the specified URL.")
            return None
        except Exception as e:
            print(f"Error while trying to get the vLLM model name: {e}")
            return None

    def create_batch_messages(self, message_batches: List[List[Dict[str, str]]], timeout: int = 172800, reasoning: bool = False, reasoning_effort: str = "medium", **kwargs) -> List[Any]:
        """
        Process multiple message conversations using vLLM's batch processing.
        
        In offline mode: Uses vLLM's direct LLM interface for optimal batch processing (RECOMMENDED)
        In server mode: Uses vLLM's OpenAI-compatible batch file format
        
        Based on: https://docs.vllm.ai/en/stable/examples/offline_inference/openai_batch.html
        
        Args:
            message_batches: List of message conversations to process
            timeout: Maximum time to wait in seconds (default: 172800 = 48 hours)
            reasoning: Whether to enable reasoning mode (not used for vLLM)
            reasoning_effort: Reasoning effort level (not used for vLLM)
            **kwargs: Additional parameters for the API call
        """
        if self._offline_mode:
            return self._create_batch_messages_offline(message_batches, **kwargs)
        else:
            return self._create_batch_messages_server(message_batches, timeout, **kwargs)
    
    def _create_batch_messages_offline(self, message_batches: List[List[Dict[str, str]]], enable_thinking: bool = True, **kwargs) -> List[Any]:
        """
        Process batches using vLLM's direct LLM interface (most efficient).
        This bypasses the server and uses vLLM's optimized continuous batching.
        
        Args:
            message_batches: List of message conversations
            enable_thinking: For Qwen3, controls thinking mode (default: True)
            **kwargs: Sampling parameters (temperature, top_p, max_tokens, etc.)
        """
        if self._llm_instance is None:
            raise RuntimeError("Offline mode not initialized. Call init_offline_mode() first.")
        
        try:
            from vllm import SamplingParams
            
            print(f"Processing {len(message_batches)} conversations in offline batch mode...")
            if 'qwen3' in self.model_name.lower():
                print(f"Qwen3 thinking mode: {'ENABLED' if enable_thinking else 'DISABLED'}")
            
            # Convert message conversations to prompts
            prompts = []
            print("Converting messages to prompts...")
            for i, messages in enumerate(tqdm(message_batches, desc="Formatting prompts", unit="conv")):
                prompt = self._format_messages_as_prompt(messages, enable_thinking=enable_thinking)
                prompts.append(prompt)
            
            # Set sampling parameters with top_k support
            sampling_kwargs = {
                'temperature': kwargs.get('temperature', 0.1),
                'top_p': kwargs.get('top_p', 0.9),
                'max_tokens': kwargs.get('max_tokens', 4000),
            }
            
            # Add top_k if specified (Qwen3 recommends top_k=20)
            if 'top_k' in kwargs:
                sampling_kwargs['top_k'] = kwargs['top_k']
            
            sampling_params = SamplingParams(**sampling_kwargs)
            
            print(f"Generating responses with sampling params: temperature={sampling_params.temperature}, "
                  f"top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}")
            if hasattr(sampling_params, 'top_k') and sampling_params.top_k is not None:
                print(f"  top_k={sampling_params.top_k}")
            
            # Generate all responses in batch (vLLM handles optimization automatically)
            # vLLM will use continuous batching and PagedAttention for efficiency
            outputs = self._llm_instance.generate(prompts, sampling_params, use_tqdm=True)
            
            # Extract results in order
            results = [output.outputs[0].text for output in outputs]
            
            print(f"✓ Successfully generated {len(results)} responses")
            return results
            
        except Exception as e:
            print(f"vLLM offline batch processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to sequential processing
            print("Falling back to sequential processing...")
            responses = []
            with tqdm(total=len(message_batches), desc="Processing vLLM requests (sequential)", unit="req") as pbar:
                for i, messages in enumerate(message_batches):
                    try:
                        response = self.create_message(messages, kwargs=kwargs)
                        responses.append(response)
                    except Exception as e:
                        print(f"Error processing batch item {i}: {e}")
                        responses.append(f"Error: {str(e)}")
                    pbar.update(1)
            return responses
    
    def create_batch_messages_with_thinking(self, message_batches: List[List[Dict[str, str]]], enable_thinking: bool = True, **kwargs) -> List[Any]:
        """
        Process batches with explicit thinking mode control (for Qwen3 models).
        
        For Qwen3, use recommended sampling parameters:
        - Thinking mode: temperature=0.6, top_p=0.95, top_k=20
        - Non-thinking mode: temperature=0.7, top_p=0.8, top_k=20
        
        Args:
            message_batches: List of message conversations to process
            enable_thinking: Enable/disable thinking mode for Qwen3 (default: True)
            **kwargs: Additional parameters (temperature, top_p, top_k, max_tokens, etc.)
        
        Returns:
            List of responses
        """
        if not self._offline_mode:
            raise ValueError("create_batch_messages_with_thinking is only available in offline mode")
        
        return self._create_batch_messages_offline(message_batches, enable_thinking=enable_thinking, **kwargs)
    
    def _create_batch_messages_server(self, message_batches: List[List[Dict[str, str]]], timeout: int = 172800, **kwargs) -> List[Any]:
        """
        Process batches using vLLM's OpenAI-compatible batch file format.
        This requires spawning a subprocess and uses file I/O.
        """
        try:
            # Prepare batch input as per vLLM OpenAI batch format
            batch_requests = []
            for i, messages in enumerate(message_batches):
                batch_requests.append({
                    "custom_id": f"request-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions", 
                    "body": {
                        "model": kwargs.get("model", self.model_name),
                        "messages": messages,
                        "max_completion_tokens": kwargs.get("max_tokens", 4000),
                        "temperature": kwargs.get("temperature", 0.1),
                    }
                })
            
            # Write batch to temp file
            with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".jsonl") as input_file:
                for req in batch_requests:
                    input_file.write(json.dumps(req) + "\n")
                input_file_path = input_file.name
            
            # Create output file path
            output_file_path = input_file_path.replace(".jsonl", "_output.jsonl")
            
            # Run vLLM batch processing
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.run_batch",
                "-i", input_file_path,
                "-o", output_file_path,
                "--model", kwargs.get("model", self.model_name)
            ]
            
            print(f"Running vLLM batch command: {' '.join(cmd)}")
            print(f"vLLM batch timeout: {timeout} seconds ({timeout/3600:.1f} hours)")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode != 0:
                print(f"vLLM batch processing failed: {result.stderr}")
                raise Exception(f"vLLM batch failed: {result.stderr}")
            
            # Read results
            results = {}
            if os.path.exists(output_file_path):
                with open(output_file_path, "r") as f:
                    for line in f:
                        if line.strip():
                            obj = json.loads(line)
                            if "custom_id" in obj:
                                # Extract response content
                                if obj.get("response") and obj["response"].get("body"):
                                    response_body = obj["response"]["body"]
                                    if response_body.get("choices"):
                                        content = response_body["choices"][0]["message"]["content"]
                                        # Extract request index from custom_id
                                        request_id = obj["custom_id"].replace("request-", "")
                                        results[request_id] = content
                                    else:
                                        request_id = obj["custom_id"].replace("request-", "")
                                        results[request_id] = str(response_body)
            
            # Clean up temp files
            os.unlink(input_file_path)
            if os.path.exists(output_file_path):
                os.unlink(output_file_path)
            
            # Return in input order
            return [results.get(str(i), f"No result for index {i}") for i in range(len(message_batches))]
            
        except Exception as e:
            print(f"vLLM batch processing failed: {e}")
            # Fallback to sequential processing
            print("Falling back to sequential processing...")
            responses = []
            with tqdm(total=len(message_batches), desc="Processing vLLM requests (sequential)", unit="req") as pbar:
                for i, messages in enumerate(message_batches):
                    try:
                        response = self.create_message(messages, kwargs=kwargs)
                        responses.append(response)
                    except Exception as e:
                        print(f"Error processing batch item {i}: {e}")
                        responses.append(f"Error: {str(e)}")
                    pbar.update(1)
            return responses

class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def create_message(self, messages: List[Dict[str, str]], tools: List = None, schema: Union[str, Dict[str, Any]] = None, kwargs: dict = None, reasoning: bool = False, budget_tokens: int = 2000) -> Any:
        if kwargs is None:
            kwargs = {
                "model": self.model,
                "max_tokens": 4000,
                "temperature": 0.1,
            }
        else: 
            kwargs["model"] = kwargs.get("model", self.model)
            kwargs["max_tokens"] = kwargs.get("max_tokens", 4000)
            kwargs["temperature"] = kwargs.get("temperature", 0.1)

        # Convert messages format for Anthropic
        # Anthropic expects 'content' instead of 'content' and handles system messages differently
        anthropic_messages = []
        system_message = None
        
        for message in messages:
            if message.get("role") == "system":
                system_message = message.get("content", "")
            else:
                anthropic_messages.append({
                    "role": message.get("role"),
                    "content": message.get("content", "")
                })

        # Set up the request parameters
        request_params = {
            "model": kwargs["model"],
            "messages": anthropic_messages,
            "max_tokens": kwargs["max_tokens"],
            "temperature": kwargs["temperature"]
        }

        if system_message:
            request_params["system"] = system_message

        if tools:
            # Convert tools to Anthropic format
            anthropic_tools = []
            for tool in tools:
                if "function" in tool:
                    anthropic_tools.append({
                        "name": tool["function"]["name"],
                        "description": tool["function"]["description"],
                        "input_schema": tool["function"]["parameters"]
                    })
            request_params["tools"] = anthropic_tools

        # Add extended thinking if reasoning is enabled
        if reasoning:
            request_params['temperature'] = 1.0
            request_params['max_tokens'] = request_params['max_tokens'] + budget_tokens
            request_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }

        # Enable streaming for long operations (>10k tokens total)
        total_tokens = request_params.get('max_tokens', 0)
        use_streaming = total_tokens > 10000
        
        if use_streaming:
            print(f"🔄 Using streaming for long operation ({total_tokens} max tokens)")

        try:
            if use_streaming:
                # Use streaming for long operations with proper thinking block handling
                stream = self.client.messages.create(
                    stream=True,
                    **request_params
                )
                
                # Collect streaming response with proper block tracking
                thinking_content = ""
                text_content = ""
                current_block_type = None
                
                for chunk in stream:
                    if hasattr(chunk, 'type'):
                        if chunk.type == 'content_block_start':
                            if hasattr(chunk, 'content_block') and hasattr(chunk.content_block, 'type'):
                                current_block_type = chunk.content_block.type
                        elif chunk.type == 'content_block_delta':
                            if hasattr(chunk, 'delta'):
                                if hasattr(chunk.delta, 'text') and current_block_type == 'text':
                                    text_content += chunk.delta.text
                                elif hasattr(chunk.delta, 'thinking') and current_block_type == 'thinking':
                                    thinking_content += chunk.delta.thinking
                
                # Create a mock response object for consistent handling
                class MockResponse:
                    def __init__(self, thinking, text):
                        self.content = []
                        if thinking:
                            self.content.append(type('ThinkingBlock', (), {'type': 'thinking', 'thinking': thinking})())
                        if text:
                            self.content.append(type('TextBlock', (), {'type': 'text', 'text': text})())
                
                response = MockResponse(thinking_content, text_content)
            else:
                # Standard non-streaming call
                response = self.client.messages.create(**request_params)
            
            # Handle tool calls
            if tools and hasattr(response, 'content') and response.content:
                for content_block in response.content:
                    if hasattr(content_block, 'type') and content_block.type == 'tool_use':
                        return {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": content_block.name,
                                        "arguments": content_block.input,
                                    }
                                }
                            ]
                        }
            
            # Extract thinking and text content
            if hasattr(response, 'content') and response.content:
                thinking_content = ""
                text_content = ""
                
                for content_block in response.content:
                    if hasattr(content_block, 'type'):
                        if content_block.type == 'thinking':
                            thinking_content += content_block.thinking
                        elif content_block.type == 'text':
                            text_content += content_block.text
                
                # Combine thinking and text with XML formatting
                combined_content = ""
                if thinking_content:
                    combined_content += f"<reasoning>\n{thinking_content}\n</reasoning>\n\n"
                combined_content += text_content
                
                return combined_content if combined_content else ""
            
            return ""
            
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return f"Error: {str(e)}"

    def get_tool_info(self, tool: Dict[str, Any]) -> Dict[str, str]:
        return {
            "name": tool["function"]["name"],
            "description": tool["function"]["description"]
        }

    def process_tool_result(self, tool_use: Any) -> Tuple[str, Dict[str, Any]]:
        return tool_use['function']['name'], tool_use['function']['arguments']
    
    def create_batch_messages(self, message_batches: List[List[Dict[str, str]]], reasoning: bool = False, budget_tokens: int = 2000, polling_interval: int = 1800, max_duration: int = 172800, **kwargs) -> List[Any]:
        """
        Process multiple message conversations using Anthropic's Message Batches API.
        
        Based on: https://docs.anthropic.com/en/docs/build-with-claude/message-batches
        
        Args:
            message_batches: List of message conversations to process
            reasoning: Whether to enable reasoning mode
            budget_tokens: Token budget for reasoning (when enabled)
            polling_interval: Seconds between status checks (default: 1800 = 30 minutes)
            max_duration: Maximum time to wait in seconds (default: 172800 = 48 hours)
            **kwargs: Additional parameters for the API call
        
        Returns:
            List of responses in the same order as input message_batches
        """
        try:
            from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
            from anthropic.types.messages.batch_create_params import Request
            
            # Prepare batch requests using SDK types
            batch_requests = []
            for i, messages in enumerate(message_batches):
                # Convert messages for Anthropic format
                anthropic_messages = []
                system_message = None
                
                for message in messages:
                    if message.get("role") == "system":
                        system_message = message.get("content", "")
                    else:
                        anthropic_messages.append({
                            "role": message.get("role"),
                            "content": message.get("content", "")
                        })
                
                # Build params dict for MessageCreateParamsNonStreaming
                params_dict = {
                    "model": kwargs.get("model", self.model),
                    "max_tokens": kwargs.get("max_tokens", 4000),
                    "temperature": kwargs.get("temperature", 0.1),
                    "messages": anthropic_messages
                }
                
                if system_message:
                    params_dict["system"] = system_message
                
                # Add extended thinking if reasoning is enabled
                if reasoning:
                    params_dict['max_tokens'] = params_dict['max_tokens'] + budget_tokens
                    params_dict['temperature'] = 1.0
                    params_dict["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget_tokens,
                    }
                
                # Create Request object
                batch_requests.append(
                    Request(
                        custom_id=str(i),
                        params=MessageCreateParamsNonStreaming(**params_dict)
                    )
                )
            
            # Create batch using SDK
            print(f"Creating Anthropic batch with {len(batch_requests)} requests...")
            message_batch = self.client.messages.batches.create(requests=batch_requests)
            
            batch_id = message_batch.id
            print(f"Anthropic batch created: {batch_id}")
            print(f"Processing status: {message_batch.processing_status}")
            print(f"Request counts: {message_batch.request_counts}")
            print(f"Polling every {polling_interval} seconds (max duration: {max_duration} seconds)")
            
            # Poll for completion
            start_time = time.time()
            while message_batch.processing_status == "in_progress":
                elapsed_time = time.time() - start_time
                if elapsed_time > max_duration:
                    print(f"Batch processing exceeded maximum duration of {max_duration} seconds")
                    raise Exception(f"Batch timeout after {max_duration} seconds")
                
                print(f"Batch status: {message_batch.processing_status}, waiting {polling_interval} seconds... (elapsed: {int(elapsed_time)}s)")
                print(f"  Request counts: {message_batch.request_counts}")
                time.sleep(polling_interval)
                
                # Retrieve updated batch status
                message_batch = self.client.messages.batches.retrieve(batch_id)
            
            # Check final status
            if message_batch.processing_status != "ended":
                print(f"Batch completed with unexpected status: {message_batch.processing_status}")
            
            print(f"Batch processing ended. Final request counts: {message_batch.request_counts}")
            
            # Stream results using SDK
            print("Retrieving batch results...")
            results = {}
            succeeded_count = 0
            errored_count = 0
            expired_count = 0
            canceled_count = 0
            
            for result in self.client.messages.batches.results(batch_id):
                custom_id = result.custom_id
                
                # Handle different result types
                if result.result.type == "succeeded":
                    succeeded_count += 1
                    message = result.result.message
                    
                    # Extract thinking and text content
                    thinking_content = ""
                    text_content = ""
                    
                    for content_block in message.content:
                        if content_block.type == "thinking":
                            thinking_content += content_block.thinking
                        elif content_block.type == "text":
                            text_content += content_block.text
                    
                    # Combine thinking and text with XML formatting
                    combined_content = ""
                    if thinking_content:
                        combined_content += f"<reasoning>\n{thinking_content}\n</reasoning>\n\n"
                    combined_content += text_content
                    
                    final_content = combined_content if combined_content else text_content
                    results[custom_id] = final_content
                    
                elif result.result.type == "errored":
                    errored_count += 1
                    error = result.result.error
                    error_type = error.type if hasattr(error, 'type') else "unknown_error"
                    error_message = error.message if hasattr(error, 'message') else str(error)
                    
                    if error_type == "invalid_request":
                        # Request body must be fixed before re-sending request
                        print(f"Validation error for request {custom_id}: {error_message}")
                    else:
                        # Request can be retried directly (e.g., server error)
                        print(f"Server error for request {custom_id}: {error_message}")
                    
                    results[custom_id] = f"Error: {error_type} - {error_message}"
                    
                elif result.result.type == "canceled":
                    canceled_count += 1
                    print(f"Request {custom_id} was canceled")
                    results[custom_id] = "Error: Request was canceled"
                    
                elif result.result.type == "expired":
                    expired_count += 1
                    print(f"Request {custom_id} expired")
                    results[custom_id] = "Error: Request expired (batch exceeded 24 hour limit)"
            
            print(f"Results summary: {succeeded_count} succeeded, {errored_count} errored, {expired_count} expired, {canceled_count} canceled")
            
            # Check if all requests failed
            total_requests = len(message_batches)
            failed_requests = errored_count + expired_count + canceled_count
            if failed_requests == total_requests:
                error_msg = f"All {total_requests} batch requests failed: {errored_count} errored, {expired_count} expired, {canceled_count} canceled"
                print(f"❌ {error_msg}")
                raise Exception(error_msg)
            elif failed_requests > 0:
                print(f"⚠️  Warning: {failed_requests}/{total_requests} requests failed")
            
            # Return in input order (results may not match input order)
            return [results.get(str(i), f"No result for index {i}") for i in range(len(message_batches))]
            
        except Exception as e:
            print(f"Anthropic batch processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to sequential processing
            print("Falling back to sequential processing...")
            responses = []
            with tqdm(total=len(message_batches), desc="Processing Anthropic requests (sequential)", unit="req") as pbar:
                for i, messages in enumerate(message_batches):
                    try:
                        response = self.create_message(messages, kwargs=kwargs, reasoning=reasoning, budget_tokens=budget_tokens)
                        responses.append(response)
                    except Exception as e:
                        print(f"Error processing batch item {i}: {e}")
                        responses.append(f"Error: {str(e)}")
                    pbar.update(1)
            return responses


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.model = model
        self.client = genai.Client(api_key=api_key, vertexai=False)
        
        # Define models that support thinking/reasoning
        self.thinking_supported_models = {
            "gemini-2.5-flash-preview",
            "gemini-2.5-flash-exp",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-pro-exp",
            "gemini-2.5-pro-exp-03-25"
        }

    def _supports_thinking(self) -> bool:
        """Check if the current model supports thinking/reasoning capabilities."""
        model_name = self.model.lower()
        return any(supported_model in model_name for supported_model in self.thinking_supported_models)

    def create_message(self, messages: List[Dict[str, str]], tools: List = None, schema: Union[str, Dict[str, Any]] = None, kwargs: dict = None, reasoning: bool = False, thinking_budget: int = 1000, max_retries: int = 5) -> Any:
        # Set up default kwargs if none provided
        if kwargs is None:
            kwargs = {
                "max_output_tokens": 4000,
                "temperature": 0.1,
            }
        else:
            kwargs.setdefault("max_output_tokens", 4000)
            kwargs.setdefault("temperature", 0.1)
        
        # Convert messages to the format expected by the new genai client
        contents = []
        system_instruction = None
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                # System messages are handled via system_instruction in the new API
                system_instruction = content
            else:
                contents.append(content)
        
        # Combine all content parts
        combined_content = "\n".join(contents)
        
        # Set up the config
        config = {
            "max_output_tokens": kwargs["max_output_tokens"],
            "temperature": kwargs["temperature"],
        }
        
        # Add system instruction if provided
        if system_instruction:
            config["system_instruction"] = system_instruction
        
        # Add thinking config if reasoning is enabled
        if reasoning:
            if not self._supports_thinking():
                print(f"⚠️  Warning: Model '{self.model}' does not support thinking/reasoning capabilities.")
                print(f"   Supported thinking models: {', '.join(sorted(self.thinking_supported_models))}")
                print(f"   Proceeding without thinking config...")
                reasoning = False
            else:
                config["thinking_config"] = {
                    "include_thoughts": True,
                    "thinking_budget": thinking_budget
                }
                # Increase max tokens to account for thinking
                config["max_output_tokens"] = config["max_output_tokens"] + thinking_budget
        
        # Retry logic with exponential backoff for rate limits
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=combined_content,
                    config=config
                )
                
                # Handle thinking content in response
                if reasoning and hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    thinking_content = ""
                    text_content = ""
                    
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            # Check if this part has text content
                            if hasattr(part, 'text') and part.text:
                                # Check if this is a thinking part
                                if hasattr(part, 'thought') and part.thought:
                                    thinking_content += part.text
                                else:
                                    text_content += part.text
                    
                    # Combine thinking and text with XML formatting (similar to Anthropic)
                    if thinking_content:
                        combined_response = f"<reasoning>\n{thinking_content}\n</reasoning>\n\n{text_content}"
                        return combined_response
                    else:
                        return text_content if text_content else response.text if hasattr(response, "text") else str(response)
                
                # Standard response handling
                return response.text if hasattr(response, "text") else str(response)
                
            except Exception as e:
                error_str = str(e)
                last_error = e
                
                # Check if this is a rate limit error (429 RESOURCE_EXHAUSTED)
                is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower()
                
                if is_rate_limit and attempt < max_retries - 1:
                    # Extract suggested retry delay from error if available
                    import re
                    retry_match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_str)
                    if retry_match:
                        retry_delay = float(retry_match.group(1))
                    else:
                        # Use exponential backoff: 60, 120, 240, 480 seconds
                        retry_delay = 60 * (2 ** attempt)
                    
                    print(f"⚠️  Rate limit hit (attempt {attempt + 1}/{max_retries})")
                    print(f"   Waiting {retry_delay:.1f} seconds before retry...")
                    time.sleep(retry_delay)
                    continue
                else:
                    # Not a rate limit error or out of retries
                    break
        
        # All retries failed, return error
        return f"Error: {str(last_error)}"

    def get_tool_info(self, tool: Dict[str, Any]) -> Dict[str, str]:
        return {"name": tool.get("name", ""), "description": tool.get("description", "")}

    def process_tool_result(self, tool_use: Any) -> Tuple[str, Dict[str, Any]]:
        return tool_use.get("name", ""), tool_use.get("arguments", {})

    def create_batch_messages(self, message_batches: List[List[Dict[str, str]]], reasoning: bool = False, thinking_budget: int = 1000, polling_interval: int = 1800, max_duration: int = 172800, **kwargs) -> List[Any]:
        """
        Process multiple message conversations using Gemini's Batch API.
        
        Based on: https://ai.google.dev/gemini-api/docs/batch-api
        
        Gemini Batch API supports two input methods:
        1. Inline requests (< 20MB): Requests passed directly as a list
        2. File input (larger batches): Requests uploaded as JSONL file via Files API
        
        This implementation tries inline requests first, and falls back to file-based
        method if the batch is too large.
        
        Args:
            message_batches: List of message conversations to process
            reasoning: Whether to enable reasoning mode
            thinking_budget: Token budget for reasoning (when enabled)
            polling_interval: Seconds between status checks (default: 1800 = 30 minutes)
            max_duration: Maximum time to wait in seconds (default: 172800 = 48 hours)
            **kwargs: Additional parameters for the API call
        
        Returns:
            List of responses in the same order as input message_batches
        """
        # Validate thinking support once for the batch
        if reasoning and not self._supports_thinking():
            print(f"⚠️  Warning: Model '{self.model}' does not support thinking/reasoning capabilities for batch processing.")
            print(f"   Supported thinking models: {', '.join(sorted(self.thinking_supported_models))}")
            print(f"   Proceeding without thinking config...")
            reasoning = False
        
        try:
            from google.genai import types
            
            # Prepare batch requests
            batch_requests = []
            
            for i, messages in enumerate(message_batches):
                # Convert messages to Gemini format
                contents = []
                system_instruction = None
                
                for message in messages:
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    
                    if role == "system":
                        system_instruction = content
                    else:
                        contents.append(content)
                
                # Combine all content parts
                combined_content = "\n".join(contents)
                
                # Build the request
                request_dict = {
                    'contents': [{
                        'parts': [{'text': combined_content}],
                        'role': 'user'
                    }]
                }
                
                # Add generation config
                generation_config = {
                    'max_output_tokens': kwargs.get('max_output_tokens', 4000),
                    'temperature': kwargs.get('temperature', 0.1),
                }
                
                # Add system instruction if provided
                if system_instruction:
                    request_dict['system_instruction'] = {'parts': [{'text': system_instruction}]}
                
                # Add thinking config if reasoning is enabled
                if reasoning:
                    generation_config['thinking_config'] = {
                        'include_thoughts': True,
                        'thinking_budget': thinking_budget
                    }
                    # Increase max tokens to account for thinking
                    generation_config['max_output_tokens'] = generation_config['max_output_tokens'] + thinking_budget
                
                request_dict['generation_config'] = generation_config
                
                batch_requests.append(request_dict)
            
            # Gemini supports two input methods:
            # 1. Inline requests (for batches < 20MB): Pass list directly as src
            # 2. File input (for larger batches): Upload JSONL file via Files API
            
            # Try inline requests first (suitable for most use cases)
            # Each request is a GenerateContentRequest dict
            try:
                print(f"Creating Gemini batch with {len(batch_requests)} inline requests...")
                
                batch_job = self.client.batches.create(
                    model=self.model,
                    src=batch_requests,  # Pass list directly for inline requests
                    config={
                        'display_name': f'batch-{int(time.time())}',
                    }
                )
            except Exception as inline_error:
                # If inline fails (e.g., batch too large), try file-based approach
                print(f"Inline request failed: {inline_error}")
                print(f"Falling back to file-based batch method...")
                
                # Create JSONL file with key-request structure for file input
                with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".jsonl") as f:
                    for i, request in enumerate(batch_requests):
                        f.write(json.dumps({"key": f"request-{i}", "request": request}) + "\n")
                    batch_file_path = f.name
                
                # Upload the file via Files API
                uploaded_file = self.client.files.upload(
                    file=batch_file_path,
                    config=types.UploadFileConfig(
                        display_name=f'batch-input-{int(time.time())}',
                        mime_type='application/jsonl'
                    )
                )
                
                print(f"Uploaded file: {uploaded_file.name}")
                
                # Create batch with uploaded file
                batch_job = self.client.batches.create(
                    model=self.model,
                    src=uploaded_file.name,
                    config={
                        'display_name': f'batch-{int(time.time())}',
                    }
                )
                
                # Clean up temp file
                os.unlink(batch_file_path)
            
            batch_name = batch_job.name
            print(f"Gemini batch created: {batch_name}")
            print(f"State: {batch_job.state}")
            print(f"Polling every {polling_interval} seconds (max duration: {max_duration} seconds)")
            
            # Poll for completion
            start_time = time.time()
            while batch_job.state not in ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
                elapsed_time = time.time() - start_time
                if elapsed_time > max_duration:
                    print(f"Batch processing exceeded maximum duration of {max_duration} seconds")
                    raise Exception(f"Batch timeout after {max_duration} seconds")
                
                print(f"Batch state: {batch_job.state}, waiting {polling_interval} seconds... (elapsed: {int(elapsed_time)}s)")
                time.sleep(polling_interval)
                
                # Retrieve updated batch status
                batch_job = self.client.batches.get(name=batch_name)
            
            # Check final status
            if batch_job.state != "JOB_STATE_SUCCEEDED":
                error_msg = f"Batch failed with state: {batch_job.state}"
                if hasattr(batch_job, 'error') and batch_job.error:
                    error_msg += f", error: {batch_job.error}"
                print(error_msg)
                raise Exception(error_msg)
            
            print(f"Batch processing completed successfully!")
            
            # Extract results
            results = {}
            succeeded_count = 0
            errored_count = 0
            
            # Check if results are inline or in a file
            if hasattr(batch_job, 'dest') and batch_job.dest:
                if hasattr(batch_job.dest, 'inlined_responses') and batch_job.dest.inlined_responses:
                    # Inline responses
                    print("Extracting inline responses...")
                    for i, inline_response in enumerate(batch_job.dest.inlined_responses):
                        if hasattr(inline_response, 'response') and inline_response.response:
                            # Extract thinking and text content
                            thinking_content = ""
                            text_content = ""
                            
                            response = inline_response.response
                            if hasattr(response, 'candidates') and response.candidates:
                                candidate = response.candidates[0]
                                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            if hasattr(part, 'thought') and part.thought:
                                                thinking_content += part.text
                                            else:
                                                text_content += part.text
                            
                            # Combine thinking and text
                            if thinking_content:
                                combined_response = f"<reasoning>\n{thinking_content}\n</reasoning>\n\n{text_content}"
                                results[str(i)] = combined_response
                            else:
                                results[str(i)] = text_content
                            succeeded_count += 1
                        elif hasattr(inline_response, 'error') and inline_response.error:
                            error_msg = str(inline_response.error)
                            print(f"Error for request {i}: {error_msg}")
                            results[str(i)] = f"Error: {error_msg}"
                            errored_count += 1
                
                elif hasattr(batch_job.dest, 'responses_file') and batch_job.dest.responses_file:
                    # File-based responses
                    print(f"Downloading results from file: {batch_job.dest.responses_file}")
                    
                    # Download the results file
                    response_file = self.client.files.get(name=batch_job.dest.responses_file)
                    
                    # Read the file content
                    file_content = self.client.files.download(name=batch_job.dest.responses_file)
                    
                    # Parse JSONL results
                    for line in file_content.decode('utf-8').strip().split('\n'):
                        if line.strip():
                            obj = json.loads(line)
                            if "key" in obj:
                                key = obj["key"]
                                # Extract request index from key
                                request_idx = key.replace("request-", "")
                                
                                if "response" in obj:
                                    response = obj["response"]
                                    # Extract thinking and text content
                                    thinking_content = ""
                                    text_content = ""
                                    
                                    if "candidates" in response and response["candidates"]:
                                        candidate = response["candidates"][0]
                                        if "content" in candidate and "parts" in candidate["content"]:
                                            for part in candidate["content"]["parts"]:
                                                if "text" in part:
                                                    if part.get("thought", False):
                                                        thinking_content += part["text"]
                                                    else:
                                                        text_content += part["text"]
                                    
                                    # Combine thinking and text
                                    if thinking_content:
                                        combined_response = f"<reasoning>\n{thinking_content}\n</reasoning>\n\n{text_content}"
                                        results[request_idx] = combined_response
                                    else:
                                        results[request_idx] = text_content
                                    succeeded_count += 1
                                elif "error" in obj:
                                    error_msg = str(obj['error'])
                                    print(f"Error for request {request_idx}: {error_msg}")
                                    results[request_idx] = f"Error: {error_msg}"
                                    errored_count += 1
            
            print(f"Retrieved {len(results)} results")
            print(f"Results summary: {succeeded_count} succeeded, {errored_count} errored")
            
            # Check if all requests failed
            total_requests = len(message_batches)
            if errored_count == total_requests:
                error_msg = f"All {total_requests} batch requests failed"
                print(f"❌ {error_msg}")
                raise Exception(error_msg)
            elif errored_count > 0:
                print(f"⚠️  Warning: {errored_count}/{total_requests} requests failed")
            
            # Return in input order
            return [results.get(str(i), f"No result for index {i}") for i in range(len(message_batches))]
            
        except Exception as e:
            print(f"Gemini batch processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to sequential processing with retry logic
            print("Falling back to sequential processing with retry logic...")
            responses = []
            with tqdm(total=len(message_batches), desc="Processing Gemini requests (sequential)", unit="req") as pbar:
                for i, messages in enumerate(message_batches):
                    response = self.create_message(
                        messages, 
                        kwargs=kwargs, 
                        reasoning=reasoning, 
                        thinking_budget=thinking_budget,
                        max_retries=5  # Enable retries for rate limits
                    )
                    responses.append(response)
                    pbar.update(1)
            return responses