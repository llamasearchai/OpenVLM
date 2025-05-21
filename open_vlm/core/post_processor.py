"""Post-processor for VLM outputs specific to engineering tasks."""

import re
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class VLMPostProcessor:
    """Post-processor for VLM outputs specific to engineering tasks."""
    
    def __init__(self, 
                 tokenizer, 
                 numerical_precision: int = 6, 
                 unit_conversion: bool = True,
                 detect_inconsistencies: bool = True):
        """Initialize the post-processor.
        
        Args:
            tokenizer: Hugging Face tokenizer.
            numerical_precision: Number of decimal places for numerical values.
            unit_conversion: Whether to enable unit conversion.
            detect_inconsistencies: Whether to detect physical inconsistencies.
        """
        self.tokenizer = tokenizer
        self.numerical_precision = numerical_precision
        self.unit_conversion = unit_conversion
        self.detect_inconsistencies = detect_inconsistencies
        
        # Unit conversion mappings (simplified example)
        self.unit_conversions = {
            "mm": {"m": 0.001, "cm": 0.1, "in": 0.0393701},
            "cm": {"m": 0.01, "mm": 10, "in": 0.393701},
            "m": {"cm": 100, "mm": 1000, "in": 39.3701},
            "in": {"cm": 2.54, "mm": 25.4, "m": 0.0254},
            # Add more as needed
        }
    
    def extract_numerical_values(self, text: str) -> List[float]:
        """Extract numerical values from text.
        
        Args:
            text: Input text to extract numerical values from.
            
        Returns:
            List of extracted numerical values.
        """
        # Find all floats or integers in the text
        pattern = r"[-+]?\d*\.\d+|\d+"
        matches = re.findall(pattern, text)
        return [float(match) for match in matches]
    
    def extract_units(self, text: str) -> List[str]:
        """Extract measurement units from text.
        
        Args:
            text: Input text to extract units from.
            
        Returns:
            List of extracted unit strings.
        """
        # Common engineering units
        unit_pattern = r'\b(?:mm|cm|m|km|in|ft|yd|mi|g|kg|lb|oz|N|kN|Pa|kPa|MPa|psi|W|kW|MW|°C|°F|K|rad|deg|°)\b'
        matches = re.findall(unit_pattern, text)
        return matches
    
    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert a value from one unit to another.
        
        Args:
            value: Numerical value to convert.
            from_unit: Source unit.
            to_unit: Target unit.
            
        Returns:
            Converted value.
            
        Raises:
            ValueError: If conversion path not found.
        """
        if from_unit == to_unit:
            return value
        
        if from_unit in self.unit_conversions and to_unit in self.unit_conversions[from_unit]:
            return value * self.unit_conversions[from_unit][to_unit]
        
        # Try two-step conversion if direct conversion not available
        for intermediate_unit in self.unit_conversions:
            if (from_unit in self.unit_conversions[intermediate_unit] and 
                to_unit in self.unit_conversions[intermediate_unit]):
                # Convert from source to intermediate
                intermediate_value = value / self.unit_conversions[intermediate_unit][from_unit]
                # Convert from intermediate to target
                return intermediate_value * self.unit_conversions[intermediate_unit][to_unit]
        
        raise ValueError(f"No conversion path found from {from_unit} to {to_unit}")
    
    def check_physical_consistency(self, prediction: str, expected_units: Optional[List[str]] = None) -> Tuple[bool, str]:
        """Check if the prediction is physically consistent.
        
        Args:
            prediction: Model prediction to check.
            expected_units: Optional list of expected units.
            
        Returns:
            Tuple of (is_consistent, issues_message).
        """
        # Extract values and units
        values = self.extract_numerical_values(prediction)
        units = self.extract_units(prediction)
        
        issues = []
        
        # Check if units are expected
        if expected_units:
            for unit in units:
                if unit not in expected_units:
                    issues.append(f"Unexpected unit: {unit}")
            
            for expected_unit in expected_units:
                if expected_unit not in units:
                    issues.append(f"Missing expected unit: {expected_unit}")
        
        # Check for negative values that should be positive
        # This is domain-specific and would need to be customized
        positive_only_keywords = ["length", "width", "height", "radius", "area", "volume", "mass"]
        for keyword in positive_only_keywords:
            if keyword.lower() in prediction.lower():
                for value in values:
                    if value < 0:
                        issues.append(f"Negative value {value} used with {keyword}")
        
        # Check for unreasonable values
        # This is highly domain-specific and would need to be customized
        
        if issues:
            return False, "\n".join(issues)
        else:
            return True, "No consistency issues detected"
    
    def format_numerical_output(self, output: str) -> str:
        """Format numerical values in the output to consistent precision.
        
        Args:
            output: Output text to format.
            
        Returns:
            Formatted output text.
        """
        def replace_number(match):
            number = float(match.group(0))
            return f"{number:.{self.numerical_precision}f}".rstrip('0').rstrip('.')
        
        pattern = r"[-+]?\d*\.\d+|\d+"
        return re.sub(pattern, replace_number, output)
    
    def process(self, model_output: str, task_type: str = "general") -> str:
        """Process model output based on task type.
        
        Args:
            model_output: Raw model output text.
            task_type: Type of task ("general", "quantitative", "spatial", "diagram").
            
        Returns:
            Processed model output.
        """
        # Apply general formatting
        processed_output = model_output.strip()
        
        # Task-specific processing
        if task_type == "quantitative":
            # Format numerical values
            processed_output = self.format_numerical_output(processed_output)
            
            # Check physical consistency if enabled
            if self.detect_inconsistencies:
                is_consistent, issues = self.check_physical_consistency(processed_output)
                if not is_consistent:
                    logger.warning(f"Consistency issues detected: {issues}")
        
        elif task_type == "spatial":
            # Additional processing for spatial tasks could be added here
            pass
        
        elif task_type == "diagram":
            # Additional processing for diagram analysis could be added here
            pass
        
        return processed_output 