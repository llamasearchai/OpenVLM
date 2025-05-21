"""Dataset for Vision-Language Supervised Fine-Tuning."""

import torch
from open_vlm.core.datasets.base_dataset import EngineeringVLMDataset

class EngineeringVLMSFTDataset(EngineeringVLMDataset):
    """Dataset for Vision-Language Supervised Fine-Tuning."""
    
    def __getitem__(self, idx):
        """Get a Vision SFT example from the dataset.
        
        Args:
            idx: Index of the example.
            
        Returns:
            Dictionary containing the example data with the following keys:
            - input_ids: Tokenized input text.
            - attention_mask: Attention mask for input.
            - labels: Tokenized target text.
            - pixel_values: Image tensor.
            - instruction: Original instruction text.
            - response: Original response text.
        """
        example = self.examples[idx]
        
        # Load image
        image_path = example["image_path"]
        image_tensor = self._load_image(image_path)
        
        # Format instruction and response
        instruction = example["instruction"]
        response = example["response"]
        
        input_text = f"Instruction: {instruction}\nResponse:"
        target_text = f"Instruction: {instruction}\nResponse: {response}"
        
        # Tokenize
        model_inputs = self.tokenizer(
            input_text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            target_text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )["input_ids"]
        
        # Create attention mask
        attention_mask = model_inputs["attention_mask"].squeeze()
        
        return {
            "input_ids": model_inputs["input_ids"].squeeze(),
            "attention_mask": attention_mask,
            "labels": labels.squeeze(),
            "pixel_values": image_tensor,
            "instruction": instruction,
            "response": response
        } 