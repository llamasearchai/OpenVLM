"""OpenVLM: Advanced Vision-Language Model Post-Training Framework"""

__version__ = "0.1.0"

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

from open_vlm.core import (
    VLMEngineer,
    EngineeringVLAdapter,
    EngineeringVisionEncoder,
    QuantitativeReasoningModule,
    GUIInteractionModule,
    SpatialUnderstandingModule,
    TechnicalDiagramModule,
    VLMPostProcessor
)

from open_vlm.core.datasets import (
    EngineeringVLMDataset,
    EngineeringVLMSFTDataset,
    QuantitativeReasoningDataset,
    GUIInteractionDataset,
    SpatialUnderstandingDataset,
    TechnicalDiagramDataset,
    MultimodalPreferenceDataset
)

__all__ = [
    "TrainingMethod",
    "BaseTrainingConfig",
    "VisionSFTConfig",
    "QuantitativeReasoningConfig",
    "GUIInteractionConfig",
    "SpatialUnderstandingConfig", 
    "TechnicalDiagramConfig",
    "MultimodalRLHFConfig",
    "VLMEngineer",
    "EngineeringVLAdapter",
    "EngineeringVisionEncoder",
    "QuantitativeReasoningModule",
    "GUIInteractionModule",
    "SpatialUnderstandingModule",
    "TechnicalDiagramModule",
    "VLMPostProcessor",
    "EngineeringVLMDataset",
    "EngineeringVLMSFTDataset",
    "QuantitativeReasoningDataset",
    "GUIInteractionDataset",
    "SpatialUnderstandingDataset",
    "TechnicalDiagramDataset",
    "MultimodalPreferenceDataset"
] 