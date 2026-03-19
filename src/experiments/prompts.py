from __future__ import annotations
from typing import List

    
prompt_templates = {
        "A": "{ingredient}",
        "B": "a dish containing {ingredient}",
        "C": "a meal with {ingredient} ingredient",
        "D": "a dish with visible {ingredient}",
    }

def make_prompts(vocab: List[str], prompt_type: str) -> List[str]:
    if prompt_type not in prompt_templates:
        raise ValueError(f"Prompt type {prompt_type} must be a valid prompt template!")
    
    template = prompt_templates[prompt_type]
    return [template.format(ingredient=ingredient) for ingredient in vocab]