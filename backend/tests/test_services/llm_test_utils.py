import numpy as np
from typing import Dict, List, Optional

def validate_parameter_ranges(params: Dict) -> bool:
    """Validate parameter ranges in LLM output."""
    for key, value in params.items():
        if isinstance(value, (int, float)):
            if not -100 <= value <= 100:
                return False
        elif isinstance(value, dict):
            if not validate_parameter_ranges(value):
                return False
    return True

def calculate_parameter_statistics(
    results: List[Dict],
    param: str
) -> Dict:
    """Calculate statistics for a specific parameter across multiple results."""
    values = [r[param] for r in results if param in r]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values))
    }

def compare_parameter_correlation(
    params1: Dict,
    params2: Dict,
    threshold: float = 0.7
) -> bool:
    """Compare correlation between two parameter sets."""
    common_params = set(params1.keys()) & set(params2.keys())
    values1 = [params1[p] for p in common_params]
    values2 = [params2[p] for p in common_params]
    
    if not values1 or not values2:
        return False
        
    correlation = np.corrcoef(values1, values2)[0, 1]
    return abs(correlation) >= threshold

class MockLLaMAResponse:
    """Mock LLaMA model response for testing."""
    
    def __init__(
        self,
        temperature: float = 0.0,
        tint: float = 0.0,
        saturation: float = 0.0,
        contrast: float = 0.0
    ):
        self.parameters = {
            "temperature": temperature,
            "tint": tint,
            "saturation": saturation,
            "contrast": contrast
        }
    
    def to_dict(self) -> Dict:
        """Convert mock response to dictionary."""
        return self.parameters
    
    @classmethod
    def create_random(cls) -> 'MockLLaMAResponse':
        """Create response with random parameters."""
        return cls(
            temperature=np.random.uniform(-100, 100),
            tint=np.random.uniform(-100, 100),
            saturation=np.random.uniform(-100, 100),
            contrast=np.random.uniform(-100, 100)
        )

def create_test_description(
    style: str,
    modifiers: Optional[List[str]] = None
) -> str:
    """Create test description with specific style and modifiers."""
    base = f"Create a {style} look"
    if modifiers:
        modifications = ", ".join(modifiers)
        base += f" with {modifications}"
    return base