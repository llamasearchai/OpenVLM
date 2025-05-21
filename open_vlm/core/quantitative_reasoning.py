"""Quantitative reasoning module for engineering applications."""

import re
import numpy as np
import logging
import sympy as sp
from typing import Dict, List, Tuple, Optional, Union, Any
from pint import UnitRegistry

logger = logging.getLogger(__name__)

# Initialize unit registry
ureg = UnitRegistry()
Q_ = ureg.Quantity

class PhysicalConstraint:
    """Represents a physical constraint for engineering calculations."""
    
    def __init__(self, name: str, expression: str, description: str = ""):
        """Initialize a physical constraint.
        
        Args:
            name: Name of the constraint.
            expression: SymPy-compatible expression string.
            description: Human-readable description of the constraint.
        """
        self.name = name
        self.expression_str = expression
        self.description = description
        
        # Parse the expression
        try:
            self.expression = sp.sympify(expression)
            self.symbols = list(self.expression.free_symbols)
            logger.info(f"Initialized constraint {name} with expression {expression}")
        except Exception as e:
            logger.error(f"Failed to parse expression {expression}: {e}")
            self.expression = None
            self.symbols = []
    
    def check(self, variable_values: Dict[str, float]) -> bool:
        """Check if the constraint is satisfied.
        
        Args:
            variable_values: Dictionary mapping variable names to values.
            
        Returns:
            True if the constraint is satisfied, False otherwise.
        """
        if self.expression is None:
            return True
        
        # Evaluate the expression
        try:
            # Create a substitution dictionary
            subs_dict = {}
            for sym in self.symbols:
                var_name = sym.name
                if var_name in variable_values:
                    subs_dict[sym] = variable_values[var_name]
                else:
                    logger.warning(f"Variable {var_name} not found in values dict")
                    return True  # Can't check if variables are missing
            
            # Evaluate the expression
            result = float(self.expression.subs(subs_dict))
            
            # Check if the constraint is satisfied (expression >= 0)
            is_satisfied = result >= 0
            if not is_satisfied:
                logger.info(f"Constraint {self.name} violated: {self.expression_str} = {result}")
            
            return is_satisfied
        
        except Exception as e:
            logger.error(f"Error checking constraint {self.name}: {e}")
            return True  # Assume satisfied in case of errors

class QuantitativeReasoningModule:
    """Module for handling quantitative reasoning in engineering tasks."""
    
    def __init__(
        self,
        use_physical_constraints: bool = True,
        unit_conversion: bool = True,
        numerical_precision: int = 6,
        predefined_constraints: Optional[List[Dict[str, str]]] = None,
    ):
        """Initialize the quantitative reasoning module.
        
        Args:
            use_physical_constraints: Whether to use physical constraints.
            unit_conversion: Whether to handle unit conversion.
            numerical_precision: Number of decimal places for numerical values.
            predefined_constraints: List of predefined constraint dictionaries.
        """
        self.use_physical_constraints = use_physical_constraints
        self.unit_conversion = unit_conversion
        self.numerical_precision = numerical_precision
        
        # Initialize constraints
        self.constraints = []
        if predefined_constraints:
            for constraint_dict in predefined_constraints:
                self.add_constraint(
                    constraint_dict["name"],
                    constraint_dict["expression"],
                    constraint_dict.get("description", "")
                )
        
        # Add standard physical constraints
        self._add_standard_constraints()
        
        logger.info(f"Initialized QuantitativeReasoningModule with {len(self.constraints)} constraints")
    
    def _add_standard_constraints(self):
        """Add standard physical constraints to the module."""
        standard_constraints = [
            {
                "name": "positive_mass",
                "expression": "m",
                "description": "Mass must be positive"
            },
            {
                "name": "positive_force",
                "expression": "F",
                "description": "Force magnitude must be positive"
            },
            {
                "name": "positive_length",
                "expression": "l",
                "description": "Length must be positive"
            },
            {
                "name": "positive_area",
                "expression": "A",
                "description": "Area must be positive"
            },
            {
                "name": "positive_volume",
                "expression": "V",
                "description": "Volume must be positive"
            },
            {
                "name": "positive_temperature",
                "expression": "T - 0",
                "description": "Absolute temperature must be positive"
            },
            {
                "name": "speed_limit",
                "expression": "3e8 - v",
                "description": "Speed must be less than the speed of light"
            },
            {
                "name": "positive_pressure",
                "expression": "P",
                "description": "Pressure must be positive"
            },
        ]
        
        for constraint in standard_constraints:
            self.add_constraint(
                constraint["name"],
                constraint["expression"],
                constraint["description"]
            )
    
    def add_constraint(self, name: str, expression: str, description: str = ""):
        """Add a physical constraint to the module.
        
        Args:
            name: Name of the constraint.
            expression: SymPy-compatible expression string.
            description: Human-readable description of the constraint.
        """
        constraint = PhysicalConstraint(name, expression, description)
        self.constraints.append(constraint)
        logger.info(f"Added constraint {name}: {expression}")
    
    def check_constraints(self, variable_values: Dict[str, float]) -> List[str]:
        """Check if all constraints are satisfied.
        
        Args:
            variable_values: Dictionary mapping variable names to values.
            
        Returns:
            List of violated constraint names.
        """
        if not self.use_physical_constraints:
            return []
        
        violated_constraints = []
        for constraint in self.constraints:
            if not constraint.check(variable_values):
                violated_constraints.append(constraint.name)
        
        return violated_constraints
    
    def convert_unit(self, value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """Convert a value from one unit to another.
        
        Args:
            value: The value to convert.
            from_unit: The source unit.
            to_unit: The target unit.
            
        Returns:
            Converted value, or None if conversion failed.
        """
        if not self.unit_conversion:
            return value
        
        try:
            # Create quantity with source unit
            quantity = Q_(value, from_unit)
            
            # Convert to target unit
            converted = quantity.to(to_unit)
            
            # Return the value
            return round(converted.magnitude, self.numerical_precision)
        
        except Exception as e:
            logger.error(f"Unit conversion error: {e}")
            return None
    
    def extract_units(self, text: str) -> Dict[str, str]:
        """Extract units from a text.
        
        Args:
            text: Text to extract units from.
            
        Returns:
            Dictionary mapping variable names to units.
        """
        # Regular expression to match units
        # This is a simplified version and would need expansion for real use
        unit_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*([a-zA-Z]+)'
        
        # Find all matches
        matches = re.finditer(unit_pattern, text)
        
        # Extract variable names and units
        units = {}
        for match in matches:
            variable = match.group(1)
            unit = match.group(3)
            units[variable] = unit
        
        return units
    
    def extract_variables(self, text: str) -> Dict[str, float]:
        """Extract variables and their values from a text.
        
        Args:
            text: Text to extract variables from.
            
        Returns:
            Dictionary mapping variable names to values.
        """
        # Regular expression to match variable assignments
        variable_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
        
        # Find all matches
        matches = re.finditer(variable_pattern, text)
        
        # Extract variable names and values
        variables = {}
        for match in matches:
            variable = match.group(1)
            value = float(match.group(2))
            variables[variable] = value
        
        return variables
    
    def analyze_numerical_reasoning(self, text: str) -> Dict[str, Any]:
        """Analyze numerical reasoning in a text.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Analysis results.
        """
        # Extract variables and units
        variables = self.extract_variables(text)
        units = self.extract_units(text)
        
        # Check constraints
        violated_constraints = self.check_constraints(variables)
        
        # Prepare result
        result = {
            "variables": variables,
            "units": units,
            "violated_constraints": violated_constraints,
            "is_physically_consistent": len(violated_constraints) == 0
        }
        
        return result
    
    def parse_equation(self, equation_str: str) -> Optional[sp.Eq]:
        """Parse an equation string into a SymPy equation.
        
        Args:
            equation_str: Equation string.
            
        Returns:
            SymPy equation, or None if parsing failed.
        """
        try:
            # Split by equals sign
            if "=" in equation_str:
                left_str, right_str = equation_str.split("=", 1)
                left = sp.sympify(left_str.strip())
                right = sp.sympify(right_str.strip())
                return sp.Eq(left, right)
            else:
                # If there's no equals sign, parse as an expression
                expr = sp.sympify(equation_str.strip())
                return sp.Eq(expr, 0)
        
        except Exception as e:
            logger.error(f"Equation parsing error: {e}")
            return None
    
    def solve_equation(self, equation_str: str, target_var: str) -> Optional[float]:
        """Solve an equation for a target variable.
        
        Args:
            equation_str: Equation string.
            target_var: Target variable to solve for.
            
        Returns:
            Solution value, or None if solving failed.
        """
        try:
            # Parse the equation
            equation = self.parse_equation(equation_str)
            if equation is None:
                return None
            
            # Create a symbol for the target variable
            target_symbol = sp.Symbol(target_var)
            
            # Solve the equation
            solution = sp.solve(equation, target_symbol)
            
            # Check if a solution was found
            if solution and len(solution) > 0:
                # Convert to float and round
                solution_value = float(solution[0])
                return round(solution_value, self.numerical_precision)
            else:
                return None
        
        except Exception as e:
            logger.error(f"Equation solving error: {e}")
            return None
    
    def estimate_precision(self, value: float, significant_digits: int = 3) -> float:
        """Estimate the precision of a calculated value.
        
        Args:
            value: The calculated value.
            significant_digits: Number of significant digits to preserve.
            
        Returns:
            Value rounded to the appropriate precision.
        """
        if value == 0:
            return 0
        
        # Estimate the order of magnitude
        order = np.floor(np.log10(abs(value)))
        
        # Calculate the precision
        precision = significant_digits - order - 1
        
        # Round to the appropriate precision
        return round(value, int(precision)) 