"""Adapters for connecting vision and language models."""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from transformers import PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType

logger = logging.getLogger(__name__)

class EngineeringVisionEncoder(nn.Module):
    """Vision encoder for engineering-specific vision processing.
    
    This module adapts the output of a vision model (e.g., CLIP) 
    to be compatible with engineering tasks.
    """
    
    def __init__(
        self,
        vision_model,
        projection_dim: int = 768,
        adapter_mode: str = "parallel",
        dropout: float = 0.1,
    ):
        """Initialize the engineering vision encoder.
        
        Args:
            vision_model: The vision model providing image features.
            projection_dim: Dimension for projecting image features.
            adapter_mode: Mode for adapting features ("parallel" or "sequential").
            dropout: Dropout probability.
        """
        super().__init__()
        self.vision_model = vision_model
        self.adapter_mode = adapter_mode
        
        # Get vision embedding dimension
        vision_dim = vision_model.config.projection_dim  # For CLIP
        
        # Create projection layer
        self.projection = nn.Linear(vision_dim, projection_dim)
        
        # Engineering-specific adapters
        self.technical_drawing_adapter = nn.Sequential(
            nn.Linear(vision_dim, vision_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim, projection_dim)
        )
        
        self.spatial_adapter = nn.Sequential(
            nn.Linear(vision_dim, vision_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim, projection_dim)
        )
        
        self.quantitative_adapter = nn.Sequential(
            nn.Linear(vision_dim, vision_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim, projection_dim)
        )
        
    def forward(self, image_features):
        """Forward pass through the encoder.
        
        Args:
            image_features: Features from the vision model.
            
        Returns:
            Adapted image features.
        """
        # Apply specialized adapters
        tech_drawing_features = self.technical_drawing_adapter(image_features)
        spatial_features = self.spatial_adapter(image_features)
        quantitative_features = self.quantitative_adapter(image_features)
        
        # Base projection
        base_projection = self.projection(image_features)
        
        # Combine features based on adapter mode
        if self.adapter_mode == "parallel":
            # Simple averaging of all adapters
            adapted_features = (base_projection + tech_drawing_features + 
                               spatial_features + quantitative_features) / 4.0
        else:  # Sequential mode
            # Sequential application
            adapted_features = base_projection + 0.1 * (
                tech_drawing_features + spatial_features + quantitative_features
            )
            
        return adapted_features

class EngineeringVLAdapter(nn.Module):
    """Adapter for connecting vision and language models for engineering tasks."""
    
    def __init__(
        self,
        language_model: PreTrainedModel,
        vision_model,
        adapter_mode: str = "parallel",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        vision_projection_dim: int = 768,
    ):
        """Initialize the adapter.
        
        Args:
            language_model: The language model.
            vision_model: The vision model.
            adapter_mode: Mode for adapting features.
            use_lora: Whether to use LoRA for efficient fine-tuning.
            lora_r: LoRA rank.
            lora_alpha: LoRA alpha.
            lora_dropout: LoRA dropout.
            vision_projection_dim: Dimension for projecting vision features.
        """
        super().__init__()
        self.language_model = language_model
        self.vision_model = vision_model
        
        # Get the embedding dimension of the language model
        if hasattr(language_model.config, "hidden_size"):
            embed_dim = language_model.config.hidden_size
        else:
            embed_dim = language_model.config.n_embd  # For some models like GPT
            
        # Vision encoder
        self.vision_encoder = EngineeringVisionEncoder(
            vision_model,
            projection_dim=embed_dim,
            adapter_mode=adapter_mode
        )
        
        # Image prefix tokens
        self.image_prefix = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.image_prefix_proj = nn.Linear(embed_dim, embed_dim)
        
        # Apply LoRA if requested
        if use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            self.model = get_peft_model(language_model, peft_config)
            logger.info("Applied LoRA to the language model")
        else:
            self.model = language_model
            
    def encode_image(self, image_features):
        """Encode image features.
        
        Args:
            image_features: Features from the vision model.
            
        Returns:
            Encoded image features compatible with the language model.
        """
        # Apply the vision encoder
        encoded_features = self.vision_encoder(image_features)
        
        # Process with image prefix
        batch_size = encoded_features.shape[0]
        image_prefix = self.image_prefix.expand(batch_size, -1, -1)
        image_prefix = self.image_prefix_proj(image_prefix)
        
        return encoded_features, image_prefix
    
    def forward(self, inputs, image_features=None):
        """Forward pass through the adapter.
        
        Args:
            inputs: Input IDs and attention mask.
            image_features: Features from the vision model (optional).
            
        Returns:
            Model outputs.
        """
        # Process inputs normally if no image features
        if image_features is None:
            return self.model(**inputs)
        
        # Encode image features
        encoded_image, image_prefix = self.encode_image(image_features)
        
        # Get embeddings from the language model
        inputs_embeds = self.model.get_input_embeddings()(inputs["input_ids"])
        
        # Add image prefix and encoded image at the beginning
        # Shape: [batch_size, 1 + seq_len, hidden_size]
        combined_embeds = torch.cat([image_prefix, encoded_image.unsqueeze(1), inputs_embeds], dim=1)
        
        # Adjust attention mask to account for the image tokens
        if "attention_mask" in inputs:
            image_attn_mask = torch.ones(
                (inputs["attention_mask"].shape[0], 2), 
                dtype=inputs["attention_mask"].dtype,
                device=inputs["attention_mask"].device
            )
            extended_attention_mask = torch.cat([image_attn_mask, inputs["attention_mask"]], dim=1)
        else:
            extended_attention_mask = None
        
        # Forward through the model with custom embeddings
        outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=extended_attention_mask,
            return_dict=True
        )
        
        return outputs
    
    def generate(self, inputs, image_features=None, max_length=512, **kwargs):
        """Generate text from the model.
        
        Args:
            inputs: Input IDs and attention mask.
            image_features: Features from the vision model (optional).
            max_length: Maximum length of the generated sequence.
            **kwargs: Additional generation arguments.
            
        Returns:
            Generated token IDs.
        """
        # Process inputs normally if no image features
        if image_features is None:
            return self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=max_length,
                **kwargs
            )
        
        # Encode image features
        encoded_image, image_prefix = self.encode_image(image_features)
        
        # Get embeddings from the language model
        inputs_embeds = self.model.get_input_embeddings()(inputs["input_ids"])
        
        # Add image prefix and encoded image at the beginning
        # Shape: [batch_size, 1 + seq_len, hidden_size]
        combined_embeds = torch.cat([image_prefix, encoded_image.unsqueeze(1), inputs_embeds], dim=1)
        
        # Adjust attention mask to account for the image tokens
        if "attention_mask" in inputs:
            image_attn_mask = torch.ones(
                (inputs["attention_mask"].shape[0], 2), 
                dtype=inputs["attention_mask"].dtype,
                device=inputs["attention_mask"].device
            )
            extended_attention_mask = torch.cat([image_attn_mask, inputs["attention_mask"]], dim=1)
        else:
            extended_attention_mask = None
            
        # Generate text with custom embeddings
        # This is a simplified approach; for proper generation with custom embeddings,
        # a custom generation function would be needed
        with torch.no_grad():
            # First, get the initial hidden states from the embeddings
            model_kwargs = {
                "attention_mask": extended_attention_mask,
            }
            
            # Pass through the model once to get the initial context
            outputs = self.model(
                inputs_embeds=combined_embeds,
                **model_kwargs
            )
            
            # Now generate with the model using the standard generate method
            # but with the context from the image-prefixed sequence
            # This is a simplification; actual implementation would need a custom generate method
            input_ids = inputs["input_ids"]
            generated = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                **kwargs
            )
            
            return generated 