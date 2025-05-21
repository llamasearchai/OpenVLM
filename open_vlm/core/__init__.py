"""Core implementation of the OpenVLM framework."""

from open_vlm.core.vlm_engineer import VLMEngineer
from open_vlm.core.adapters import EngineeringVLAdapter, EngineeringVisionEncoder
from open_vlm.core.post_processor import VLMPostProcessor
from open_vlm.core.cpp_tensor_processor import CppTensorProcessor

# Specialized modules
from open_vlm.core.quantitative_reasoning import QuantitativeReasoningModule
from open_vlm.core.gui_interaction import GUIInteractionModule
from open_vlm.core.spatial_understanding import SpatialUnderstandingModule
from open_vlm.core.technical_diagram import TechnicalDiagramModule
from open_vlm.core.multimodal_rlhf import MultimodalRewardModel, MultimodalPPOTrainer

__all__ = [
    "VLMEngineer",
    "EngineeringVLAdapter",
    "EngineeringVisionEncoder",
    "VLMPostProcessor",
    "CppTensorProcessor",
    "QuantitativeReasoningModule",
    "GUIInteractionModule",
    "SpatialUnderstandingModule",
    "TechnicalDiagramModule",
    "MultimodalRewardModel",
    "MultimodalPPOTrainer"
] 