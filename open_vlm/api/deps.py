"""Dependency injection functions for FastAPI."""

import logging
from typing import Optional

from fastapi import Request, Depends, HTTPException
import torch

from open_vlm.core import VLMEngineer
from open_vlm.config import VisionSFTConfig

logger = logging.getLogger(__name__)

def get_vlm_engineer(request: Request) -> VLMEngineer:
    """Dependency to get the VLMEngineer instance.
    
    If the VLMEngineer is not already initialized, it will be created.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Initialized VLMEngineer instance
    """
    # Check if the model is already initialized
    if not hasattr(request.app.state, "vlm_engineer") or request.app.state.vlm_engineer is None:
        # Get model path from app state
        model_path = request.app.state.model_path
        if not model_path:
            raise HTTPException(
                status_code=500,
                detail="No model path specified. Set the model_path when creating the app."
            )
        
        try:
            # Initialize configuration
            device = torch.device(request.app.state.device)
            
            # Check if this is a directory (likely a saved model) or HF model name
            import os
            if os.path.isdir(model_path):
                # Load saved model
                try:
                    # Try to load config.json if it exists
                    import json
                    config_path = os.path.join(model_path, "config.json")
                    if os.path.exists(config_path):
                        with open(config_path, "r") as f:
                            config_dict = json.load(f)
                        config = VisionSFTConfig(**config_dict)
                    else:
                        # Create default config pointing to the saved model path
                        config = VisionSFTConfig(
                            model_name=model_path,
                            adapter_mode="parallel"
                        )
                except Exception as e:
                    logger.warning(f"Error loading config from {model_path}: {str(e)}")
                    # Fallback to default config
                    config = VisionSFTConfig(
                        model_name=model_path,
                        adapter_mode="parallel"
                    )
            else:
                # Create default config with HF model name
                config = VisionSFTConfig(
                    model_name=model_path,
                    adapter_mode="parallel"
                )
            
            # Initialize the VLMEngineer
            logger.info(f"Initializing VLMEngineer with model {model_path} on {device}")
            vlm_engineer = VLMEngineer(config=config, device=device)
            
            # Store in app state
            request.app.state.vlm_engineer = vlm_engineer
            request.app.state.model_initialized = True
            
            return vlm_engineer
            
        except Exception as e:
            logger.exception(f"Error initializing VLMEngineer")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize model: {str(e)}"
            )
    
    # Return existing VLMEngineer instance
    return request.app.state.vlm_engineer 