"""Pydantic models for OpenVLM API request and response types."""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, HttpUrl, validator

class ImagePromptRequest(BaseModel):
    """Request model for image analysis with a prompt."""
    
    prompt: str = Field(..., description="The prompt text to guide the model's analysis")
    image_data: Optional[str] = Field(None, description="Base64-encoded image data")
    image_url: Optional[HttpUrl] = Field(None, description="URL of the image to analyze")
    max_tokens: Optional[int] = Field(256, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Temperature for generation sampling (0.0-1.0)")
    
    @validator("prompt")
    def prompt_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v
    
    @validator("temperature")
    def validate_temperature(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Temperature must be between 0.0 and 1.0")
        return v
    
    @validator("max_tokens")
    def validate_max_tokens(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_tokens must be greater than 0")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Describe this technical diagram in detail, including any measurements and component labels.",
                "image_url": "https://example.com/image.jpg",
                "max_tokens": 256,
                "temperature": 0.7
            }
        }

class ImagePromptResponse(BaseModel):
    """Response model for image analysis with a prompt."""
    
    response: str = Field(..., description="The generated text response")
    processing_time: float = Field(..., description="Time taken to process the request in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "The technical diagram shows a gear assembly with two interlocking gears. The input gear has 24 teeth and a diameter of 48mm, while the output gear has 36 teeth and a diameter of 72mm. The gear ratio is 1.5:1, providing a mechanical advantage for torque amplification...",
                "processing_time": 1.25
            }
        }

class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    detail: str = Field(..., description="Error message")
    
    class Config:
        schema_extra = {
            "example": {
                "detail": "Failed to load image from URL: Connection timeout"
            }
        }

class ModelInfo(BaseModel):
    """Model information response."""
    
    model_name: str = Field(..., description="Name of the language model")
    vision_model_name: str = Field(..., description="Name of the vision encoder model")
    device: str = Field(..., description="Device used for inference (cuda/cpu)")
    quantized: bool = Field(..., description="Whether the model is quantized")
    avg_inference_time: Optional[float] = Field(None, description="Average inference time in seconds")
    adapter_type: str = Field(..., description="Type of adapter used (lora/full)")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "meta-llama/Llama-2-7b-hf",
                "vision_model_name": "openai/clip-vit-large-patch14",
                "device": "cuda",
                "quantized": False,
                "avg_inference_time": 1.35,
                "adapter_type": "lora"
            }
        }

class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status (ok/error)")
    device: str = Field(..., description="Device used for inference (cuda/cpu)")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "ok",
                "device": "cuda",
                "model_loaded": True
            }
        } 