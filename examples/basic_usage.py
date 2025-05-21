#!/usr/bin/env python
"""Basic usage example for the open-vlm package."""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
from PIL import Image

# Add parent directory to path for importing from the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from open_vlm.config import VisionSFTConfig
from open_vlm.core import VLMEngineer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OpenVLM basic usage example")
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Path to the pretrained model or model name on HuggingFace"
    )
    parser.add_argument(
        "--vision_model_path",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Path to the vision model or model name on HuggingFace"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this technical diagram in detail. Include any measurements and components visible:",
        help="Prompt to use for the model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use LoRA for efficient adaptation"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate"
    )
    
    return parser.parse_args()

def main():
    """Run the example."""
    args = parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        logger.error(f"Image not found at path: {args.image_path}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize configuration
    logger.info("Initializing configuration")
    config = VisionSFTConfig(
        model_name=args.model_path,
        vision_model_name=args.vision_model_path,
        adapter_mode="parallel",
        use_lora=args.use_lora,
        numerical_precision=6,
        use_physical_constraints=True,
        unit_conversion=True,
        output_dir=args.output_dir
    )
    
    # Initialize VLM Engineer
    logger.info(f"Initializing VLM Engineer with model {args.model_path}")
    vlm_engineer = VLMEngineer(
        config=config,
        device=args.device
    )
    
    # Generate response
    logger.info(f"Generating response for image {args.image_path}")
    response = vlm_engineer.generate_response(
        prompt=args.prompt,
        image_path=args.image_path,
        max_length=args.max_tokens
    )
    
    # Print response
    print("\n" + "="*50)
    print("GENERATED RESPONSE:")
    print("="*50)
    print(response)
    print("="*50 + "\n")
    
    # Save response to file
    output_path = os.path.join(args.output_dir, "response.txt")
    with open(output_path, "w") as f:
        f.write(response)
    
    logger.info(f"Response saved to {output_path}")

if __name__ == "__main__":
    main() 