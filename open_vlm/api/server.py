"""Server module for OpenVLM API.

This module is the entry point for uvicorn when running the API server.
"""

import os
import logging
from open_vlm.api.app import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("open_vlm.api")

# Get configuration from environment variables
model_path = os.environ.get("OPENVLM_MODEL_PATH")
host = os.environ.get("OPENVLM_HOST", "0.0.0.0")
port = int(os.environ.get("OPENVLM_PORT", "8000"))

# Create the FastAPI app
app = create_app(
    model_path=model_path,
    enable_cors=True,
    title="OpenVLM API",
    description="API for Vision-Language Models specialized for engineering applications",
)

logger.info(f"OpenVLM API initialized with model path: {model_path or 'None (lazy loading)'}")

# This app instance will be imported by uvicorn 