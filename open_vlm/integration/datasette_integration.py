"""Datasette integration for OpenVLM.

This module provides a Datasette plugin that adds custom functionality for
exploring and visualizing OpenVLM results.
"""

from datasette import hookimpl
import markupsafe
import json
from typing import Dict, Any, List, Optional

@hookimpl
def render_cell(value, column, table, database, datasette):
    """Render cells with custom visualization based on column content.
    
    This hook improves the display of OpenVLM results in Datasette by:
    1. Pretty-printing JSON content (especially for measurements, bounding boxes)
    2. Displaying inline thumbnails for image paths
    3. Formatting confidence scores nicely
    """
    # Handle possible JSON content
    if isinstance(value, str) and (
        (value.startswith("{") and value.endswith("}")) or
        (value.startswith("[") and value.endswith("]"))
    ):
        try:
            data = json.loads(value)
            # Pretty format JSON
            formatted_json = json.dumps(data, indent=2)
            return markupsafe.Markup(
                f"<details><summary>JSON data</summary><pre>{formatted_json}</pre></details>"
            )
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Handle image paths
    if isinstance(value, str) and column == "image_path" and (
        value.endswith(".jpg") or 
        value.endswith(".jpeg") or 
        value.endswith(".png") or 
        value.endswith(".gif")
    ):
        # Create a thumbnail link
        thumbnail_html = f"""
        <a href="{value}" target="_blank">
            <img src="{value}" style="max-width: 100px; max-height: 100px;">
        </a>
        """
        return markupsafe.Markup(thumbnail_html)
    
    # Format confidence scores
    if isinstance(value, (float, int)) and column == "confidence":
        confidence = float(value)
        if 0 <= confidence <= 1:
            # Change color based on confidence level
            color = "green" if confidence >= 0.8 else "orange" if confidence >= 0.5 else "red"
            percentage = f"{confidence * 100:.1f}%"
            return markupsafe.Markup(
                f'<span style="color: {color}; font-weight: bold;">{percentage}</span>'
            )
    
    return None  # Return None to use default rendering

@hookimpl
def extra_css_urls():
    """Add custom CSS for OpenVLM visualizations."""
    return [
        "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
    ]

@hookimpl
def extra_js_urls():
    """Add custom JavaScript for interactive visualizations."""
    return [
        {
            "url": "https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js",
            "defer": True,
        },
        {
            "url": "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js",
            "defer": True,
        }
    ]

@hookimpl
def extra_template_vars(template, database, table, columns, view_name, request, datasette):
    """Add custom template variables for OpenVLM-specific visualizations."""
    # Only add these variables for tables that look like OpenVLM results
    openvlm_columns = {"image_id", "prompt", "response", "confidence", "processing_time"}
    
    # Check if this looks like an OpenVLM results table
    if table and columns and len(openvlm_columns.intersection(set(columns))) >= 3:
        return {
            "is_openvlm_results": True,
            "openvlm_version": _get_openvlm_version(),
        }
    
    return {}

def _get_openvlm_version() -> str:
    """Get the current OpenVLM version."""
    try:
        from open_vlm import __version__
        return __version__
    except (ImportError, AttributeError):
        return "unknown"

# Register the plugin
metadata = {
    "name": "OpenVLM Datasette Plugin", 
    "description": "Enhances Datasette for visualizing OpenVLM results",
    "version": "0.1.0",
    "license": "Apache-2.0",
    "author": "Jina AI",
    "source": "https://github.com/jina-ai/open-vlm",
    "documentation": "https://github.com/jina-ai/open-vlm/tree/main/docs",
}

# This will be picked up by the datasette.plugins entry point
plugin = metadata 