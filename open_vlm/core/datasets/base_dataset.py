"""Base dataset class for engineering VLM data."""

import os
import json
import logging
import torch
from typing import Dict, List, Any
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class EngineeringVLMDataset(Dataset):
    """Base dataset class for engineering VLM data."""
    
    def __init__(self, file_path: str, tokenizer, processor=None, 
                 image_dir: str = "./images", max_length: int = 512):
        """Initialize the dataset.
        
        Args:
            file_path: Path to the JSONL data file.
            tokenizer: Hugging Face tokenizer.
            processor: Vision processor (e.g., CLIP processor).
            image_dir: Directory containing images.
            max_length: Maximum sequence length for tokenization.
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.image_dir = image_dir
        self.examples = self._load_data(file_path)
        
    def _load_data(self, file_path: str) -> List[Dict]:
        """Load data from JSONL file.
        
        Args:
            file_path: Path to the JSONL data file.
            
        Returns:
            List of examples as dictionaries.
        """
        examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        return examples
    
    def _load_image(self, image_path: str):
        """Load and preprocess an image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Preprocessed image tensor.
        """
        try:
            if not os.path.exists(image_path):
                # Check in image_dir if path is relative
                image_path = os.path.join(self.image_dir, os.path.basename(image_path))
            
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            
            if self.processor:
                processed = self.processor(images=image, return_tensors="pt")
                return processed.pixel_values.squeeze(0)
            else:
                # Fallback to basic preprocessing
                from torchvision import transforms
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])
                ])
                return preprocess(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a blank tensor as fallback
            return torch.zeros((3, 224, 224))
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get an example from the dataset.
        
        Args:
            idx: Index of the example.
            
        Returns:
            Dictionary containing the example data.
        """
        raise NotImplementedError("Subclasses must implement __getitem__") 