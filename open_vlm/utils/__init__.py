"""Common utilities for the OpenVLM framework."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

logger = logging.getLogger(__name__)

def load_json_or_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load data from a JSON or JSONL file.
    
    This function detects the file format based on extension and content
    and returns a list of dictionaries.
    
    Args:
        file_path: Path to the JSON or JSONL file
        
    Returns:
        List of dictionaries with the loaded data
        
    Raises:
        ValueError: If the file format is invalid or cannot be determined
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # Empty file
            if not content:
                return []
            
            # Check if it's a JSONL file (each line is a valid JSON object)
            if file_path.suffix.lower() == '.jsonl' or content.startswith('{') and '\n{' in content:
                results = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            results.append(json.loads(line))
                return results
            
            # Otherwise, treat as regular JSON
            data = json.loads(content)
            
            # If it's already a list, return it directly
            if isinstance(data, list):
                return data
            # If it's a dict, wrap it in a list
            elif isinstance(data, dict):
                return [data]
            else:
                raise ValueError(f"Unsupported JSON structure in {file_path}")
                
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {file_path}: {str(e)}")

def get_version() -> str:
    """Get the current version of OpenVLM.
    
    Returns:
        The version string
    """
    try:
        from open_vlm import __version__
        return __version__
    except (ImportError, AttributeError):
        return "unknown" 