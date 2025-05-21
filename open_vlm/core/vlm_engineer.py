"""Main VLM Engineer class for engineering reasoning and VLM post-training."""

import os
import torch
import logging
from typing import Dict, List, Optional, Union, Tuple, Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPProcessor,
    CLIPModel,
)

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType
)

from open_vlm.config import (
    BaseTrainingConfig,
    VisionSFTConfig,
    QuantitativeReasoningConfig,
    GUIInteractionConfig,
    SpatialUnderstandingConfig,
    TechnicalDiagramConfig
)

from open_vlm.core.adapters import EngineeringVLAdapter
from open_vlm.core.post_processor import VLMPostProcessor
from open_vlm.core.cpp_tensor_processor import CppTensorProcessor

logger = logging.getLogger(__name__)

class VLMEngineer:
    """Vision-Language Model Engineer.
    
    This class provides the main interface for training and using 
    specialized vision-language models for engineering applications.
    """
    
    def __init__(
        self, 
        config: BaseTrainingConfig,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        vision_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_cpp_extension: bool = False,
        cpp_extension_path: Optional[str] = None,
    ):
        """Initialize the VLM Engineer.
        
        Args:
            config: Configuration for the VLM Engineer.
            model_path: Path to the pretrained model.
            tokenizer_path: Path to the tokenizer.
            vision_model_path: Path to the vision model.
            device: Device to run the model on.
            use_cpp_extension: Whether to use C++ extension for acceleration.
            cpp_extension_path: Path to the C++ extension.
        """
        self.config = config
        self.device = device
        
        # Initialize tokenizer
        tokenizer_path = tokenizer_path or config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        
        # Add special tokens if they don't exist
        special_tokens = {"pad_token": "<PAD>", "eos_token": "</s>", "bos_token": "<s>"}
        for token_name, token_value in special_tokens.items():
            if getattr(self.tokenizer, token_name) is None:
                setattr(self.tokenizer, token_name, token_value)
                
        # Initialize model
        if model_path is None:
            model_path = config.model_name
            
        logger.info(f"Loading language model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        
        # Initialize vision model
        vision_model_path = vision_model_path or config.vision_model_name
        logger.info(f"Loading vision model from {vision_model_path}")
        self.vision_processor = CLIPProcessor.from_pretrained(vision_model_path)
        self.vision_model = CLIPModel.from_pretrained(vision_model_path)
        
        # Initialize the adapter
        self.adapter = EngineeringVLAdapter(
            self.model, 
            self.vision_model,
            adapter_mode=getattr(config, "adapter_mode", "parallel"),
            use_lora=getattr(config, "use_lora", True),
            lora_r=getattr(config, "lora_r", 16),
            lora_alpha=getattr(config, "lora_alpha", 32),
            lora_dropout=getattr(config, "lora_dropout", 0.05),
        )
        
        # Initialize the post-processor
        self.post_processor = VLMPostProcessor(
            self.tokenizer,
            numerical_precision=getattr(config, "numerical_precision", 6),
            unit_conversion=getattr(config, "unit_conversion", True),
            detect_inconsistencies=getattr(config, "detect_inconsistencies", True)
        )
        
        # Initialize the C++ tensor processor if requested
        self.cpp_processor = None
        if use_cpp_extension:
            self.cpp_processor = CppTensorProcessor(cpp_extension_path)
        
        # Move models to device
        self.model.to(self.device)
        self.vision_model.to(self.device)
        
        # Load specialized modules based on config type
        self._initialize_specialized_modules()
        
        logger.info("VLMEngineer initialized successfully")
    
    def _initialize_specialized_modules(self):
        """Initialize specialized modules based on the config type."""
        # Import specialized modules lazily based on config type
        if isinstance(self.config, QuantitativeReasoningConfig):
            from open_vlm.core.quantitative_reasoning import QuantitativeReasoningModule
            self.quantitative_module = QuantitativeReasoningModule(
                use_physical_constraints=self.config.use_physical_constraints,
                unit_conversion=self.config.unit_conversion,
                numerical_precision=self.config.numerical_precision
            )
        
        elif isinstance(self.config, GUIInteractionConfig):
            from open_vlm.core.gui_interaction import GUIInteractionModule
            self.gui_module = GUIInteractionModule(
                gui_action_types=self.config.gui_action_types,
                handle_ocr=self.config.handle_ocr
            )
            
        elif isinstance(self.config, SpatialUnderstandingConfig):
            from open_vlm.core.spatial_understanding import SpatialUnderstandingModule
            self.spatial_module = SpatialUnderstandingModule(
                use_3d_processing=self.config.use_3d_processing,
                point_cloud_resolution=self.config.point_cloud_resolution
            )
            
        elif isinstance(self.config, TechnicalDiagramConfig):
            from open_vlm.core.technical_diagram import TechnicalDiagramModule
            self.diagram_module = TechnicalDiagramModule(
                detect_components=self.config.detect_components,
                extract_measurements=self.config.extract_measurements,
                extract_annotations=self.config.extract_annotations
            )
            
    def train(self, train_dataset, eval_dataset=None, **train_kwargs):
        """Train the model on the dataset.
        
        Args:
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.
            **train_kwargs: Additional training arguments.
            
        Returns:
            Training results.
        """
        from transformers import Trainer, TrainingArguments
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.eval_steps if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
            seed=self.config.seed,
            fp16=self.config.mixed_precision,
            **train_kwargs
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.adapter.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        logger.info("Starting training")
        train_result = trainer.train()
        
        # Save the model
        logger.info(f"Saving model to {self.config.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        return train_result
    
    def process_image(self, image_path):
        """Process an image and return its features.
        
        Args:
            image_path: Path to the image.
            
        Returns:
            Image features.
        """
        from PIL import Image
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Process image with vision processor
        inputs = self.vision_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get image features
        with torch.no_grad():
            image_features = self.vision_model.get_image_features(**inputs)
            
        return image_features
    
    def generate_response(self, prompt, image_path=None, **generation_kwargs):
        """Generate a response from the model.
        
        Args:
            prompt: Text prompt.
            image_path: Path to an image (optional).
            **generation_kwargs: Additional generation arguments.
            
        Returns:
            Generated response.
        """
        # Process image if provided
        image_features = None
        if image_path:
            image_features = self.process_image(image_path)
        
        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        outputs = self.adapter.generate(
            inputs=inputs,
            image_features=image_features,
            **generation_kwargs
        )
        
        # Decode output tokens
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process response
        processed_response = self.post_processor.process_response(
            response, 
            prompt=prompt,
            image_path=image_path
        )
        
        return processed_response
    
    def save_model(self, output_dir):
        """Save the model to the output directory.
        
        Args:
            output_dir: Output directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.adapter.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        import json
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
        
        logger.info(f"Model saved to {output_dir}")
    
    @classmethod
    def from_pretrained(cls, model_dir, device="cuda", **kwargs):
        """Load a model from a pretrained directory.
        
        Args:
            model_dir: Directory containing the model.
            device: Device to load the model on.
            **kwargs: Additional arguments.
            
        Returns:
            VLMEngineer instance.
        """
        import json
        from open_vlm.config import VisionSFTConfig
        
        # Load config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Determine config type
        method = config_dict.get("method", "vision_supervised_fine_tuning")
        if method == "vision_supervised_fine_tuning":
            config = VisionSFTConfig(**config_dict)
        else:
            # Handle other config types as needed
            config = VisionSFTConfig(**config_dict)
        
        # Create instance
        instance = cls(
            config=config,
            model_path=model_dir,
            tokenizer_path=model_dir,
            device=device,
            **kwargs
        )
        
        return instance 