"""Tests for the OpenVLM FastAPI integration."""

import sys
import json
import base64
from io import BytesIO
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from open_vlm.api.app import create_app
from open_vlm.api.models import (
    ImagePromptRequest,
    ImagePromptResponse,
    ErrorResponse,
    ModelInfo,
    HealthResponse
)

@pytest.fixture
def mock_vlm_engineer():
    """Fixture to create a mock VLMEngineer."""
    mock = MagicMock()
    mock.config = MagicMock()
    mock.config.model_name = "mock-llama-7b"
    mock.config.vision_model_name = "mock-clip"
    mock.config.use_8bit_quantization = False
    mock.config.use_lora = True
    mock.generate_response.return_value = "This is a mock response describing the image."
    mock.requires_file_path = False
    return mock

@pytest.fixture
def app(mock_vlm_engineer):
    """Fixture to create a test FastAPI app."""
    with patch("open_vlm.api.deps.get_vlm_engineer", return_value=mock_vlm_engineer):
        app = create_app(model_path="mock_model_path")
        app.state.model_initialized = True
        app.state.vlm_engineer = mock_vlm_engineer
        yield app

@pytest.fixture
def client(app):
    """Fixture to create a test client."""
    return TestClient(app)

@pytest.fixture
def sample_image():
    """Fixture to create a sample image for testing."""
    # Create a simple 10x10 red image
    img = Image.new('RGB', (10, 10), color='red')
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

def test_health_endpoint(client):
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["device"] in ["cuda", "cpu"]
    assert data["model_loaded"] is True

def test_model_info_endpoint(client):
    """Test the model info endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "mock-llama-7b"
    assert data["vision_model_name"] == "mock-clip"
    assert data["adapter_type"] == "lora"

def test_analyze_endpoint_with_base64(client, sample_image, mock_vlm_engineer):
    """Test the analyze endpoint with base64-encoded image."""
    request_data = {
        "prompt": "Describe this image",
        "image_data": sample_image,
        "max_tokens": 100,
        "temperature": 0.5
    }
    
    response = client.post("/analyze", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "This is a mock response describing the image."
    assert "processing_time" in data
    
    # Verify the mock was called with expected args
    mock_vlm_engineer.generate_response.assert_called_once()
    args, kwargs = mock_vlm_engineer.generate_response.call_args
    assert kwargs["prompt"] == "Describe this image"
    assert kwargs["max_length"] == 100
    assert kwargs["temperature"] == 0.5

def test_analyze_endpoint_missing_image(client):
    """Test the analyze endpoint with missing image data."""
    request_data = {
        "prompt": "Describe this image"
    }
    
    response = client.post("/analyze", json=request_data)
    assert response.status_code == 400
    data = response.json()
    assert "Either image_data or image_url must be provided" in data["detail"]

def test_analyze_endpoint_invalid_base64(client):
    """Test the analyze endpoint with invalid base64 data."""
    request_data = {
        "prompt": "Describe this image",
        "image_data": "not-valid-base64"
    }
    
    response = client.post("/analyze", json=request_data)
    assert response.status_code == 400
    data = response.json()
    assert "Invalid base64 image data" in data["detail"]

@pytest.mark.asyncio
async def test_upload_endpoint(client, mock_vlm_engineer):
    """Test the upload endpoint with a file upload."""
    # Create a simple test image
    img = Image.new('RGB', (10, 10), color='blue')
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    
    response = client.post(
        "/upload",
        files={"file": ("image.png", buffer, "image/png")},
        data={"prompt": "Describe this blue image", "max_tokens": 150, "temperature": 0.8}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "This is a mock response describing the image."
    assert "processing_time" in data
    
    # Verify the mock was called with expected args
    mock_vlm_engineer.generate_response.assert_called_once()
    args, kwargs = mock_vlm_engineer.generate_response.call_args
    assert kwargs["prompt"] == "Describe this blue image"
    assert kwargs["max_length"] == 150
    assert kwargs["temperature"] == 0.8 