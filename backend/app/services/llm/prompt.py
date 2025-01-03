from typing import Dict, Any, Optional
from pydantic import BaseModel

class PromptTemplate(BaseModel):
    """Base model for prompt templates."""
    name: str
    template: str
    description: Optional[str] = None
    examples: Optional[Dict[str, str]] = None

class PromptManager:
    """Manager for LLM prompt templates."""
    
    # LUT Generation Prompt
    LUT_GENERATION = PromptTemplate(
        name="lut_generation",
        description="Generate LUT parameters from text description",
        template="""
You are a professional colorist expert in creating color grading lookup tables (LUTs).
Based on the following description, generate precise color grading parameters.

Description: {description}

Consider these key aspects:
1. Overall Color Temperature:
   - Analyze warmth/coolness requirements
   - Consider any specific color casts mentioned
   - Balance between natural and stylized looks

2. Dynamic Range:
   - Contrast ratio and overall tonality
   - Treatment of highlights and shadows
   - Black point and white point positioning

3. Color Grading:
   - Color balance and saturation levels
   - Specific color modifications
   - Color relationships and complementary colors

4. Technical Requirements:
   - Ensure values stay within processable ranges
   - Consider technical limitations of LUTs
   - Maintain image integrity and avoid artifacts

Generate precise parameters in the following JSON format:
{
    "temperature": <float -100 to 100>,     # Warm (positive) to cool (negative)
    "tint": <float -100 to 100>,            # Magenta (positive) to green (negative)
    "saturation": <float -100 to 100>,      # Global saturation adjustment
    "contrast": <float -100 to 100>,        # Global contrast adjustment
    "highlights": <float -100 to 100>,      # Highlight recovery/boost
    "shadows": <float -100 to 100>,         # Shadow recovery/crush
    "whites": <float -100 to 100>,          # White point adjustment
    "blacks": <float -100 to 100>,          # Black point adjustment
    "color_balance": {
        "shadows": {
            "red": <float -100 to 100>,
            "green": <float -100 to 100>,
            "blue": <float -100 to 100>
        },
        "midtones": {
            "red": <float -100 to 100>,
            "green": <float -100 to 100>,
            "blue": <float -100 to 100>
        },
        "highlights": {
            "red": <float -100 to 100>,
            "green": <float -100 to 100>,
            "blue": <float -100 to 100>
        }
    }
}

Provide only the JSON output without any additional text or explanations.
        """,
        examples={
            "cinematic": """
Description: Create a modern cinematic look with teal and orange color grading, 
deep blacks, and slightly lifted shadows.

{
    "temperature": 15,
    "tint": -5,
    "saturation": 20,
    "contrast": 30,
    "highlights": -10,
    "shadows": 15,
    "whites": -5,
    "blacks": -20,
    "color_balance": {
        "shadows": {
            "red": -10,
            "green": 5,
            "blue": 15
        },
        "midtones": {
            "red": 10,
            "green": 0,
            "blue": 5
        },
        "highlights": {
            "red": 15,
            "green": -5,
            "blue": -10
        }
    }
}
            """
        }
    )

    # Style Analysis Prompt
    STYLE_ANALYSIS = PromptTemplate(
        name="style_analysis",
        description="Analyze visual style description for context",
        template="""
Analyze the following visual style description and extract key characteristics that 
would influence color grading decisions.

Description: {description}

Consider these aspects:
1. Overall Style Direction
2. Color Palette Preferences
3. Mood and Atmosphere
4. Technical Requirements
5. Reference Styles or Influences

Provide analysis in the following JSON format:
{
    "style_type": str,          # Primary style category
    "color_palette": str,       # Main color characteristics
    "mood": str,               # Intended emotional impact
    "technical_notes": [str],   # Technical considerations
    "references": [str],        # Similar styles or influences
    "recommendations": {
        "primary_focus": str,   # Main grading focus
        "key_adjustments": [str], # Suggested primary adjustments
        "cautions": [str]       # Potential pitfalls to avoid
    }
}

Provide only the JSON output without any additional text.
        """
    )

    # Color Matching Prompt
    COLOR_MATCHING = PromptTemplate(
        name="color_matching",
        description="Generate parameters to match reference image",
        template="""
Analyze the provided image characteristics and generate parameters to match its look
while incorporating elements from the description.

Reference Image Analysis:
{image_analysis}

User Description:
{description}

Additional Style Requirements:
{style_requirements}

Generate parameters that would:
1. Match the color characteristics of the reference image
2. Incorporate specific requirements from the description
3. Maintain proper technical standards
4. Ensure reproducible results

Provide the parameters in the standard JSON format with additional matching metadata.
        """
    )

    # Advanced Color Theory Prompt
    COLOR_THEORY = PromptTemplate(
        name="color_theory",
        description="Advanced color theory analysis for grading",
        template="""
Analyze the color requirements and provide advanced color theory recommendations.

Input:
{requirements}

Consider:
1. Color Harmony and Relationships
2. Psychological Impact
3. Technical Feasibility
4. Industry Standards

Provide detailed color theory analysis and recommendations in JSON format.
        """
    )

    # Technical Validation Prompt
    TECHNICAL_VALIDATION = PromptTemplate(
        name="technical_validation",
        description="Validate technical aspects of generated parameters",
        template="""
Validate the following color grading parameters for technical feasibility:

Parameters:
{parameters}

Checks:
1. Value Range Validation
2. Color Space Compatibility
3. Processing Requirements
4. Potential Artifacts

Provide validation results and recommendations in JSON format.
        """
    )

    @staticmethod
    def get_prompt(prompt_type: str, **kwargs) -> str:
        """Get formatted prompt template."""
        prompts = {
            "lut_generation": PromptManager.LUT_GENERATION.template,
            "style_analysis": PromptManager.STYLE_ANALYSIS.template,
            "color_matching": PromptManager.COLOR_MATCHING.template,
            "color_theory": PromptManager.COLOR_THEORY.template,
            "technical_validation": PromptManager.TECHNICAL_VALIDATION.template
        }
        
        if prompt_type not in prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        return prompts[prompt_type].format(**kwargs)

    @staticmethod
    def get_example(prompt_type: str, example_key: str) -> Optional[str]:
        """Get example for specific prompt type."""
        if prompt_type == "lut_generation":
            return PromptManager.LUT_GENERATION.examples.get(example_key)
        return None

    @staticmethod
    def validate_parameters(parameters: Dict[str, Any]) -> bool:
        """Validate generated parameters against expected ranges."""
        try:
            # Basic range validations
            for key in ["temperature", "tint", "saturation", "contrast", 
                       "highlights", "shadows", "whites", "blacks"]:
                value = parameters.get(key)
                if value is None or not -100 <= float(value) <= 100:
                    return False

            # Color balance validation
            if "color_balance" in parameters:
                for zone in ["shadows", "midtones", "highlights"]:
                    for channel in ["red", "green", "blue"]:
                        value = parameters["color_balance"][zone][channel]
                        if not -100 <= float(value) <= 100:
                            return False

            return True
        except (KeyError, ValueError, TypeError):
            return False