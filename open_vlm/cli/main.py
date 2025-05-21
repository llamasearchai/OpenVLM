"""Main CLI entry point for OpenVLM."""

import os
import sys
import logging
import argparse
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

import torch
from PIL import Image

from open_vlm.config import (
    TrainingMethod,
    BaseTrainingConfig,
    VisionSFTConfig,
    QuantitativeReasoningConfig,
    GUIInteractionConfig,
    SpatialUnderstandingConfig,
    TechnicalDiagramConfig,
    MultimodalRLHFConfig
)

from open_vlm.core import VLMEngineer

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False):
    """Set up logging configuration.
    
    Args:
        verbose: Whether to enable debug logging.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("open_vlm.log"),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing the configuration.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise

def create_training_config(config_dict: Dict[str, Any]) -> BaseTrainingConfig:
    """Create a training configuration from a dictionary.
    
    Args:
        config_dict: Dictionary containing configuration.
        
    Returns:
        Training configuration instance.
        
    Raises:
        ValueError: If training method is invalid.
    """
    method_str = config_dict.get("training", {}).get("method", "")
    try:
        method = TrainingMethod(method_str)
    except ValueError:
        valid_methods = ", ".join([m.value for m in TrainingMethod])
        raise ValueError(f"Invalid training method: {method_str}. Valid methods: {valid_methods}")
    
    # Base configuration params
    base_params = {
        "method": method,
        "model_name": config_dict.get("model", {}).get("name", ""),
        "vision_model_name": config_dict.get("model", {}).get("vision_model_name", "openai/clip-vit-large-patch14"),
    }
    
    # Add training params
    training_params = config_dict.get("training", {})
    base_params.update({k: v for k, v in training_params.items() if k != "method"})
    
    # Add data params
    data_params = config_dict.get("data", {})
    base_params.update(data_params)
    
    # Create specific config class based on method
    if method == TrainingMethod.VISION_SFT:
        return VisionSFTConfig(**base_params)
    elif method == TrainingMethod.QUANTITATIVE_REASONING:
        return QuantitativeReasoningConfig(**base_params)
    elif method == TrainingMethod.GUI_INTERACTION:
        return GUIInteractionConfig(**base_params)
    elif method == TrainingMethod.SPATIAL_UNDERSTANDING:
        return SpatialUnderstandingConfig(**base_params)
    elif method == TrainingMethod.TECHNICAL_DIAGRAM_ANALYSIS:
        return TechnicalDiagramConfig(**base_params)
    elif method == TrainingMethod.MULTIMODAL_RLHF:
        return MultimodalRLHFConfig(**base_params)
    else:
        raise ValueError(f"Unsupported training method: {method}")

def train_command(args):
    """Handle train command.
    
    Args:
        args: Command line arguments.
    """
    # Load and parse config
    config_dict = load_config(args.config)
    training_config = create_training_config(config_dict)
    
    # Create agent and run training
    agent = VLMEngineer(training_config)
    model = agent.run()
    
    logger.info(f"Training completed, model saved to {training_config.output_dir}")

def infer_command(args):
    """Handle infer command.
    
    Args:
        args: Command line arguments.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from PIL import Image
    
    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        logger.info(f"Model loaded from {args.model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Load image if specified
    image = None
    if args.image_path:
        try:
            image = Image.open(args.image_path).convert("RGB")
            logger.info(f"Image loaded from {args.image_path}")
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return
    
    # Generate response
    try:
        from open_vlm.core import VLMPostProcessor
        
        inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
        
        # Generate text
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        # Decode output
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-processing
        post_processor = VLMPostProcessor(
            tokenizer=tokenizer,
            numerical_precision=6,
            unit_conversion=True,
            detect_inconsistencies=True
        )
        
        task_type = "general"
        if args.task_type in ["quantitative", "spatial", "diagram"]:
            task_type = args.task_type
        
        processed_output = post_processor.process(output_text, task_type=task_type)
        
        # Print result
        print(processed_output)
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return

def serve_command(args):
    """Handle serve command for launching the REST API.
    
    Args:
        args: Command line arguments.
    """
    try:
        # Import here to avoid dependency on FastAPI if not using this command
        import uvicorn
        from open_vlm.api import create_app
        
        logger.info(f"Starting API server with model: {args.model_path or 'None (lazy loading)'}")
        logger.info(f"Server will be available at http://{args.host}:{args.port}")
        logger.info(f"API documentation will be available at http://{args.host}:{args.port}/docs")
        
        # Create an app instance with the specified model path
        app = create_app(
            model_path=args.model_path,
            enable_cors=args.cors,
            title="OpenVLM API" if not args.title else args.title,
            description="API for Vision-Language Models specialized for engineering applications"
        )
        
        # Launch the server with uvicorn
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            log_level="debug" if args.verbose else "info"
        )
        
    except ImportError:
        logger.error("FastAPI and/or uvicorn not installed. Please install with: pip install 'open-vlm[api]'")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error starting API server: {str(e)}")
        sys.exit(1)

def export_command(args):
    """Handle export command for exporting results to various formats.
    
    Args:
        args: Command line arguments.
    """
    try:
        from open_vlm.utils import load_json_or_jsonl
        from open_vlm.integration.datasette_utils import DatasetteExporter
        
        # Load the results
        results = load_json_or_jsonl(args.results_file)
        
        # Load metadata if provided
        metadata = None
        if args.metadata:
            import json
            with open(args.metadata, 'r') as f:
                metadata = json.load(f)
        
        # Handle different output formats
        if args.format == "sqlite":
            exporter = DatasetteExporter(args.output)
            exporter.export_results(results, table_name=args.table, metadata=metadata)
        elif args.format == "json":
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f)
        elif args.format == "csv":
            import pandas as pd
            pd.DataFrame(results).to_csv(args.output, index=False)
        
        logger.info(f"Successfully exported {len(results)} results to {args.output}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="OpenVLM: Vision-Language Model Framework")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", "-c", required=True, help="Path to configuration file")
    train_parser.set_defaults(func=train_command)
    
    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model-path", "-m", required=True, help="Path to model directory")
    infer_parser.add_argument("--image-path", "-i", help="Path to input image (optional)")
    infer_parser.add_argument("--prompt", "-p", required=True, help="Text prompt")
    infer_parser.add_argument("--task-type", "-t", default="general", 
                             choices=["general", "quantitative", "spatial", "diagram"],
                             help="Type of task for post-processing")
    infer_parser.set_defaults(func=infer_command)
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Launch REST API server")
    serve_parser.add_argument("--model-path", "-m", help="Path to model or HF model name")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server (default: 0.0.0.0)")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to run the server on (default: 8000)")
    serve_parser.add_argument("--workers", "-w", type=int, default=1, help="Number of worker processes (default: 1)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    serve_parser.add_argument("--title", help="API title")
    serve_parser.add_argument("--cors", action="store_true", help="Enable CORS middleware")
    serve_parser.set_defaults(func=serve_command)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export results to various formats")
    export_parser.add_argument("results_file", help="Path to JSON/JSONL results file")
    export_parser.add_argument("--output", "-o", default="vlm_outputs.db", help="Output file path")
    export_parser.add_argument("--table", "-t", default="results", help="Table name (for SQLite)")
    export_parser.add_argument("--metadata", "-m", help="Path to JSON metadata file")
    export_parser.add_argument("--format", "-f", choices=["sqlite", "json", "csv"], default="sqlite", 
                              help="Output format (default: sqlite)")
    export_parser.set_defaults(func=export_command)
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    if args.command is None:
        parser.print_help()
        return
    
    # Call the appropriate function
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 