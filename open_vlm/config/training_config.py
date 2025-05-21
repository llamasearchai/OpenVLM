"""Training configuration for the OpenVLM framework."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Union, Tuple, Optional, Set, Any

class TrainingMethod(Enum):
    """Enum for different training methods."""
    VISION_SFT = "vision_supervised_fine_tuning"
    QUANTITATIVE_REASONING = "quantitative_reasoning"
    GUI_INTERACTION = "gui_interaction"
    SPATIAL_UNDERSTANDING = "spatial_understanding"
    TECHNICAL_DIAGRAM_ANALYSIS = "technical_diagram_analysis"
    MULTIMODAL_RLHF = "multimodal_rlhf"

@dataclass
class BaseTrainingConfig:
    """Base configuration for training parameters."""
    method: TrainingMethod
    model_name: str
    vision_model_name: str = "openai/clip-vit-large-patch14"
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 3
    max_seq_length: int = 512
    max_image_resolution: Tuple[int, int] = (224, 224)
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    output_dir: str = "./output"
    seed: int = 42
    mixed_precision: bool = True
    distributed_training: bool = False
    num_gpus: int = 1
    use_cpp_extension: bool = False
    cpp_extension_path: Optional[str] = None

@dataclass
class VisionSFTConfig(BaseTrainingConfig):
    """Vision Supervised Fine-Tuning configuration."""
    train_file: str = "train.jsonl"
    eval_file: str = "eval.jsonl"
    image_dir: str = "./images"
    vision_backbone: str = "clip"  # clip, blip2
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_lora: bool = True
    use_8bit_quantization: bool = False
    adapter_mode: str = "parallel"  # parallel, serial
    loss_type: str = "token"  # token or sequence
    image_augmentation: bool = True
    augmentation_strength: float = 0.5  # 0.0-1.0 range
    
@dataclass
class QuantitativeReasoningConfig(BaseTrainingConfig):
    """Quantitative reasoning training configuration."""
    train_file: str = "quant_train.jsonl"
    eval_file: str = "quant_eval.jsonl"
    image_dir: str = "./images"
    equation_gen_model: Optional[str] = None
    max_equations_per_example: int = 5
    math_token_handling: str = "special"  # special, standard
    numerical_precision: int = 6  # decimal places to evaluate
    include_symbolic_math: bool = True
    physics_simulation_integration: bool = False
    simulation_backend: Optional[str] = None  # e.g., "scipy", "numpy", "pytorch"
    optimization_target: str = "mse"  # mse, mae, relative_error

@dataclass
class GUIInteractionConfig(BaseTrainingConfig):
    """GUI interaction training configuration."""
    train_file: str = "gui_train.jsonl"
    eval_file: str = "gui_eval.jsonl"
    screenshot_dir: str = "./screenshots"
    gui_action_space: List[str] = field(default_factory=lambda: ["click", "type", "drag", "scroll"])
    simulator_path: Optional[str] = None
    record_intermediate_states: bool = True
    synthetic_data_generation: bool = True
    num_synthetic_examples: int = 1000
    simulator_timeout: int = 30  # seconds
    use_gui_landmarks: bool = True
    include_cursor_history: bool = True
    max_action_sequence_length: int = 50

@dataclass
class SpatialUnderstandingConfig(BaseTrainingConfig):
    """Spatial understanding training configuration."""
    train_file: str = "spatial_train.jsonl"
    eval_file: str = "spatial_eval.jsonl"
    model_dir: str = "./3d_models"
    use_point_cloud: bool = True
    point_cloud_density: int = 10000
    use_depth_maps: bool = True
    include_multiple_viewpoints: bool = True
    num_viewpoints: int = 8
    cad_format: str = "obj"  # obj, stl, step
    render_engine: str = "blender"  # blender, pyrender
    include_measurements: bool = True
    coordinate_system: str = "cartesian"  # cartesian, polar, cylindrical
    generate_cross_sections: bool = True
    
@dataclass
class TechnicalDiagramConfig(BaseTrainingConfig):
    """Technical diagram analysis configuration."""
    train_file: str = "diagram_train.jsonl"
    eval_file: str = "diagram_eval.jsonl"
    diagram_dir: str = "./diagrams"
    include_annotations: bool = True
    detect_symbols: bool = True
    symbol_vocabulary_file: Optional[str] = "symbol_vocabulary.json"
    extract_measurements: bool = True
    ocr_integration: bool = True
    ocr_engine: str = "tesseract"  # tesseract, easyocr
    include_diagram_context: bool = True
    diagram_types: List[str] = field(default_factory=lambda: ["electrical", "mechanical", "architectural"])
    connection_analysis: bool = True
    
@dataclass
class MultimodalRLHFConfig(BaseTrainingConfig):
    """Multimodal RLHF configuration."""
    ppo_epochs: int = 4
    value_loss_coef: float = 0.1
    entropy_coef: float = 0.01
    kl_coef: float = 0.2
    clip_range: float = 0.2
    reward_model_name: Optional[str] = None
    initial_vision_sft_steps: int = 1000
    reference_model_name: Optional[str] = None
    train_preference_file: str = "preferences.jsonl"
    eval_preference_file: str = "eval_preferences.jsonl"
    image_dir: str = "./images"
    include_visual_feedback: bool = True
    preference_collection_method: str = "human"  # human, synthetic, hybrid
    reward_balance_coefficient: float = 0.5 