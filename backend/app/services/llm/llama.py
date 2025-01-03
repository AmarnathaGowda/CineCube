LUT_GENERATION_PROMPT = """
Analyze the following description and generate appropriate color grading parameters.
Description: {description}

Consider these aspects:
1. Overall color temperature and mood
2. Contrast and dynamic range
3. Color balance and saturation
4. Shadow and highlight treatment

Generate specific color grading parameters that would achieve this look.

Output the parameters in this exact JSON format:
{
    "temperature": float (-100 to 100, warm/cool),
    "tint": float (-100 to 100, green/magenta),
    "saturation": float (-100 to 100),
    "contrast": float (-100 to 100),
    "highlights": float (-100 to 100),
    "shadows": float (-100 to 100),
    "whites": float (-100 to 100),
    "blacks": float (-100 to 100)
}
"""

STYLE_ANALYSIS_PROMPT = """
Analyze this visual style description and extract key characteristics.
Description: {description}

Consider:
- Color palette and temperature
- Mood and atmosphere
- Technical aspects
- Reference styles or genres

Provide analysis in this JSON format:
{
    "style_type": string,
    "color_palette": string,
    "mood": string,
    "technical_notes": array of strings,
    "references": array of strings
}
"""

COLOR_MATCHING_PROMPT = """
Analyze the reference image characteristics and generate matching parameters.
Image analysis data: {image_data}
Description: {description}

Generate parameters that would match the reference image while incorporating
elements from the description.

Output parameters in the standard JSON format with additional matching metadata.
"""

def get_prompt(prompt_type: str, **kwargs) -> str:
    """Get formatted prompt template."""
    prompts = {
        "lut_generation": LUT_GENERATION_PROMPT,
        "style_analysis": STYLE_ANALYSIS_PROMPT,
        "color_matching": COLOR_MATCHING_PROMPT
    }
    
    if prompt_type not in prompts:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    return prompts[prompt_type].format(**kwargs)