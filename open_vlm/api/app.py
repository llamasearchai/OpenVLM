"""FastAPI application for serving OpenVLM models."""

import os
import logging
from typing import Dict, Optional, List, Union

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import time

from open_vlm.api.models import (
    ImagePromptRequest,
    ImagePromptResponse,
    ErrorResponse,
    ModelInfo,
    HealthResponse
)
from open_vlm.api.deps import get_vlm_engineer
from open_vlm.core import VLMEngineer
from open_vlm.config import VisionSFTConfig

logger = logging.getLogger(__name__)

def create_app(
    model_path: Optional[str] = None,
    enable_cors: bool = True,
    title: str = "OpenVLM API",
    description: str = "API for Vision-Language Models specialized for engineering applications",
) -> FastAPI:
    """Create a FastAPI application for serving OpenVLM models.
    
    Args:
        model_path: Path to the model directory or HF model name
        enable_cors: Whether to enable CORS middleware
        title: API title
        description: API description
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        description=description,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        responses={
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse}
        },
    )
    
    # Add CORS middleware if enabled
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Store the model path for dependency injection
    app.state.model_path = model_path
    app.state.device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.inference_times = []  # For monitoring performance
    
    # Add routes
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def health_check():
        """Check if the API is running."""
        return {
            "status": "ok",
            "device": app.state.device,
            "model_loaded": hasattr(app.state, "model_initialized") and app.state.model_initialized,
        }
    
    @app.get("/model-info", response_model=ModelInfo, tags=["System"])
    def get_model_info(vlm_engineer: VLMEngineer = Depends(get_vlm_engineer)):
        """Get information about the loaded model."""
        config = vlm_engineer.config
        
        # Calculate average inference time if available
        avg_time = None
        if app.state.inference_times:
            avg_time = sum(app.state.inference_times) / len(app.state.inference_times)
            
        # Return model information
        return {
            "model_name": config.model_name,
            "vision_model_name": config.vision_model_name,
            "device": app.state.device,
            "quantized": getattr(config, "use_8bit_quantization", False),
            "avg_inference_time": avg_time,
            "adapter_type": "lora" if getattr(config, "use_lora", False) else "full",
        }
    
    @app.post("/analyze", response_model=ImagePromptResponse, tags=["Inference"])
    async def analyze_image(
        background_tasks: BackgroundTasks,
        request: ImagePromptRequest,
        vlm_engineer: VLMEngineer = Depends(get_vlm_engineer)
    ):
        """Generate a response for an image with a prompt.
        
        The request can include either a base64-encoded image or an image URL.
        """
        start_time = time.time()
        
        try:
            # Handle base64-encoded image
            if request.image_data:
                try:
                    image_data = base64.b64decode(request.image_data)
                    image = Image.open(io.BytesIO(image_data))
                except Exception as e:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid base64 image data: {str(e)}"
                    )
            
            # Handle image URL (requires a utility function to download and load)
            elif request.image_url:
                try:
                    from open_vlm.utils.image_utils import load_image_from_url
                    image = await load_image_from_url(request.image_url)
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to load image from URL: {str(e)}"
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Either image_data or image_url must be provided"
                )
            
            # Save image temporarily if needed
            temp_image_path = None
            if vlm_engineer.requires_file_path:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    temp_image_path = tmp.name
                    image.save(temp_image_path)
            
            # Generate response
            if temp_image_path:
                response = vlm_engineer.generate_response(
                    prompt=request.prompt,
                    image_path=temp_image_path,
                    max_length=request.max_tokens or 256,
                    temperature=request.temperature or 0.7,
                )
                # Clean up in background
                background_tasks.add_task(os.unlink, temp_image_path)
            else:
                response = vlm_engineer.generate_response(
                    prompt=request.prompt,
                    image=image,
                    max_length=request.max_tokens or 256,
                    temperature=request.temperature or 0.7,
                )
            
            # Calculate elapsed time and update metrics
            elapsed_time = time.time() - start_time
            app.state.inference_times.append(elapsed_time)
            # Keep only the last 100 inference times
            if len(app.state.inference_times) > 100:
                app.state.inference_times.pop(0)
            
            return {
                "response": response,
                "processing_time": elapsed_time
            }
            
        except Exception as e:
            logger.exception("Error processing request")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating response: {str(e)}"
            )
    
    @app.post("/upload", response_model=ImagePromptResponse, tags=["Inference"])
    async def upload_image(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        prompt: str = None,
        max_tokens: Optional[int] = 256,
        temperature: Optional[float] = 0.7,
        vlm_engineer: VLMEngineer = Depends(get_vlm_engineer)
    ):
        """Generate a response for an uploaded image file with a prompt."""
        start_time = time.time()
        
        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="Prompt is required"
            )
        
        try:
            # Read and process the uploaded file
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            # Save image temporarily if needed
            temp_image_path = None
            if vlm_engineer.requires_file_path:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    temp_image_path = tmp.name
                    image.save(temp_image_path)
            
            # Generate response
            if temp_image_path:
                response = vlm_engineer.generate_response(
                    prompt=prompt,
                    image_path=temp_image_path,
                    max_length=max_tokens,
                    temperature=temperature,
                )
                # Clean up in background
                background_tasks.add_task(os.unlink, temp_image_path)
            else:
                response = vlm_engineer.generate_response(
                    prompt=prompt,
                    image=image,
                    max_length=max_tokens,
                    temperature=temperature,
                )
            
            # Calculate elapsed time and update metrics
            elapsed_time = time.time() - start_time
            app.state.inference_times.append(elapsed_time)
            # Keep only the last 100 inference times
            if len(app.state.inference_times) > 100:
                app.state.inference_times.pop(0)
            
            return {
                "response": response,
                "processing_time": elapsed_time
            }
            
        except Exception as e:
            logger.exception("Error processing uploaded file")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing uploaded file: {str(e)}"
            )
    
    # Add a route to export analytics to Datasette
    @app.get("/export-analytics", tags=["Utility"])
    async def export_analytics(
        output_path: str = "openvlm_analytics.db",
        background_tasks: BackgroundTasks
    ):
        """Export analytics data to a SQLite database for Datasette."""
        try:
            from open_vlm.integration.datasette_utils import DatasetteExporter
            
            # Collect analytics data
            analytics = {
                "inference_times": app.state.inference_times,
                "avg_time": sum(app.state.inference_times) / len(app.state.inference_times) if app.state.inference_times else None,
                "device": app.state.device,
                "model_path": app.state.model_path,
            }
            
            # Export in background to avoid blocking the response
            def export_data():
                exporter = DatasetteExporter(output_path)
                exporter.export_results([analytics], "analytics")
                logger.info(f"Analytics exported to {output_path}")
            
            background_tasks.add_task(export_data)
            
            return {"status": "Analytics export scheduled", "output_path": output_path}
        except Exception as e:
            logger.exception("Error exporting analytics")
            raise HTTPException(
                status_code=500,
                detail=f"Error exporting analytics: {str(e)}"
            )
            
    return app 