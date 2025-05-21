"""Integration with Simon Willison's LLM CLI tool.

This module provides integration with the LLM CLI tool 
(https://llm.datasette.io/) for further processing OpenVLM outputs.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

try:
    import llm
except ImportError:
    llm = None

# OpenVLM command family
if llm is not None:
    @llm.hookimpl
    def register_commands(cli):
        """Register OpenVLM commands with the LLM CLI tool."""
        # Create a command group for OpenVLM commands
        @cli.group(name="openvlm")
        def openvlm_group():
            """Commands for working with OpenVLM outputs."""
            pass
        
        # Command to analyze OpenVLM results
        @openvlm_group.command(name="analyze")
        @llm.option(
            "-i", "--input", 
            type=llm.Path(exists=True), 
            required=True, 
            help="Path to the JSON/JSONL file with OpenVLM results"
        )
        @llm.option(
            "-o", "--output", 
            type=str, 
            help="Path to write output summary (optional, prints to stdout if not specified)"
        )
        @llm.option(
            "--model", 
            type=str, 
            default="", 
            help="LLM model to use for analysis (default: use current model)"
        )
        @llm.option(
            "--system", 
            type=str, 
            default="You are an AI assistant analyzing OpenVLM outputs. Provide a concise summary of the key findings.",
            help="System prompt for analysis"
        )
        def analyze(input, output, model, system):
            """Analyze OpenVLM results using LLM to generate a summary report.
            
            This command takes OpenVLM results (JSON/JSONL) and analyzes them with an LLM model,
            generating a summary of the insights from the vision analysis.
            """
            # Load the results
            results = _load_results(input)
            if not results:
                print(f"No valid results found in {input}")
                return
            
            # Convert results to a concise text representation
            formatted_input = _format_results_for_llm(results)
            
            # Prepare the prompt
            prompt = f"""
            Analyze the following OpenVLM results from vision analysis:
            
            {formatted_input}
            
            Provide a concise summary of the key findings, patterns, and notable insights.
            """
            
            # Run the analysis with LLM
            response = _run_llm_analysis(prompt, system, model)
            
            # Write or print the response
            if output:
                with open(output, "w") as f:
                    f.write(response)
                print(f"Analysis written to {output}")
            else:
                print(response)
        
        # Command to extract structured data from results
        @openvlm_group.command(name="extract")
        @llm.option(
            "-i", "--input", 
            type=llm.Path(exists=True), 
            required=True, 
            help="Path to the JSON/JSONL file with OpenVLM results"
        )
        @llm.option(
            "-o", "--output", 
            type=str, 
            required=True, 
            help="Path to write extracted data (as JSON)"
        )
        @llm.option(
            "--format", 
            type=llm.Choice(["json", "csv", "md"]), 
            default="json", 
            help="Output format"
        )
        @llm.option(
            "--model", 
            type=str, 
            default="", 
            help="LLM model to use for extraction (default: use current model)"
        )
        @llm.option(
            "--extract", 
            type=str, 
            default="measurements",
            help="Type of data to extract: measurements, components, key_values, etc."
        )
        def extract(input, output, format, model, extract):
            """Extract structured data from OpenVLM results using LLM.
            
            This command processes OpenVLM results to extract specific structured data
            like measurements, component lists, or key-value pairs.
            """
            # Load the results
            results = _load_results(input)
            if not results:
                print(f"No valid results found in {input}")
                return
            
            # Convert results to a concise text representation
            formatted_input = _format_results_for_llm(results)
            
            # Prepare the extraction prompt based on what's being extracted
            extraction_prompts = {
                "measurements": "Extract all measurements mentioned in the results, including values and units.",
                "components": "Extract a list of all components or parts mentioned in the results.",
                "key_values": "Extract all key-value pairs (e.g., specifications, parameters) from the results."
            }
            
            prompt_text = extraction_prompts.get(
                extract, 
                f"Extract the following from the results: {extract}"
            )
            
            prompt = f"""
            Analyze the following OpenVLM results from vision analysis:
            
            {formatted_input}
            
            {prompt_text}
            
            Return the extracted data as a structured JSON object.
            """
            
            # Set the system prompt for extraction
            system = f"You are a data extraction assistant. Extract {extract} from the text and return a JSON object."
            
            # Run the extraction with LLM
            response = _run_llm_analysis(prompt, system, model)
            
            # Process the response
            extracted_data = _process_extraction_response(response)
            
            # Output in the appropriate format
            if format == "json":
                with open(output, "w") as f:
                    json.dump(extracted_data, f, indent=2)
            elif format == "csv":
                import csv
                # Handle different structures
                if isinstance(extracted_data, list) and extracted_data and isinstance(extracted_data[0], dict):
                    fieldnames = list(extracted_data[0].keys())
                    with open(output, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(extracted_data)
                else:
                    # Fallback for non-list-of-dicts
                    with open(output, "w") as f:
                        f.write(json.dumps(extracted_data, indent=2))
            elif format == "md":
                markdown = _convert_to_markdown(extracted_data)
                with open(output, "w") as f:
                    f.write(markdown)
            
            print(f"Extracted data written to {output} in {format} format")

def _load_results(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load OpenVLM results from a JSON or JSONL file.
    
    Args:
        file_path: Path to the JSON/JSONL file
        
    Returns:
        List of result objects
    """
    try:
        from open_vlm.utils import load_json_or_jsonl
        return load_json_or_jsonl(file_path)
    except ImportError:
        # Fallback implementation if open_vlm utils are not available
        file_path = Path(file_path)
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() == '.jsonl':
                return [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    return [data]

def _format_results_for_llm(results: List[Dict[str, Any]]) -> str:
    """Format OpenVLM results for LLM consumption.
    
    Args:
        results: List of result objects
        
    Returns:
        Formatted text representation
    """
    formatted_lines = []
    for i, result in enumerate(results, 1):
        formatted_lines.append(f"## Result {i}")
        
        # Add key fields
        for key in ["image_id", "prompt", "response"]:
            if key in result:
                formatted_lines.append(f"{key}: {result[key]}")
        
        # Add other fields
        for key, value in result.items():
            if key not in ["image_id", "prompt", "response"]:
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, indent=2)
                    formatted_lines.append(f"{key}:")
                    for line in value_str.split("\n"):
                        formatted_lines.append(f"  {line}")
                else:
                    formatted_lines.append(f"{key}: {value}")
        
        formatted_lines.append("")  # Add blank line between results
    
    return "\n".join(formatted_lines)

def _run_llm_analysis(prompt: str, system: str, model: str) -> str:
    """Run LLM analysis on the given prompt.
    
    Args:
        prompt: Prompt text
        system: System prompt
        model: LLM model name (if empty, uses default)
        
    Returns:
        LLM response text
    """
    if llm is None:
        return "Error: LLM package is not installed. Please install with: pip install llm"
    
    try:
        # Get the model instance
        llm_model = llm.get_model(model) if model else None
        
        # Get the response
        response = llm.invoke(
            prompt, 
            system=system,
            model=llm_model
        )
        
        return str(response)
    except Exception as e:
        return f"Error running LLM analysis: {str(e)}"

def _process_extraction_response(response: str) -> Any:
    """Process the LLM extraction response to extract the JSON object.
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted data as a Python object
    """
    # Look for JSON content within the response
    try:
        # First try to parse the entire response as JSON
        return json.loads(response)
    except json.JSONDecodeError:
        # If that fails, look for JSON blocks in the response
        try:
            # Find content between triple backticks
            import re
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
            if json_match:
                json_content = json_match.group(1)
                return json.loads(json_content)
            
            # If no match, look for content that looks like a JSON object or array
            json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", response)
            if json_match:
                json_content = json_match.group(1)
                return json.loads(json_content)
        except (json.JSONDecodeError, AttributeError):
            pass
    
    # If all else fails, return the raw response
    return {"raw_response": response}

def _convert_to_markdown(data: Any) -> str:
    """Convert extracted data to Markdown format.
    
    Args:
        data: Extracted data
        
    Returns:
        Markdown text
    """
    if isinstance(data, list) and data and isinstance(data[0], dict):
        # Create a markdown table
        headers = list(data[0].keys())
        markdown = "| " + " | ".join(headers) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        for item in data:
            row = []
            for header in headers:
                value = item.get(header, "")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                row.append(str(value))
            markdown += "| " + " | ".join(row) + " |\n"
        
        return markdown
    else:
        # Just dump as JSON with markdown code blocks
        return f"```json\n{json.dumps(data, indent=2)}\n```"

# Functions to be called by register_commands
def register_commands(cli=None):
    """Register OpenVLM commands with the LLM CLI."""
    if llm is None:
        print("LLM package not found. Please install with: pip install llm", file=sys.stderr)
        return {}
    
    return {
        "name": "OpenVLM LLM Integration",
        "version": "0.1.0",
        "description": "Commands for working with OpenVLM results using LLM."
    } 