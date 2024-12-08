from pydantic import *
from typing import *
class raw_hypothesis_template(BaseModel):
    Describing_the_input_grid: str
    Describing_the_size_of_the_output_grid:str
    Describing_how_to_transform_the_grid:str

class summary_hypothesis_template(BaseModel):
    rules: List[str]

class python_implementation_template(BaseModel):
    code:str


def load_templates():
    raw_hypothesis_template_vllm = {
    "type": "object",
    "properties": {
        "Describing_the_input_grid": {
        "type": "string",
        },
        "Describing_the_size_of_the_output_grid": {
        "type": "string",
        },
        "Describing_how_to_transform_the_grid": {
        "type": "string",
        }
    },
    "required": ["Describing_the_input_grid", "Describing_the_size_of_the_output_grid", "Describing_how_to_transform_the_grid"],
    "additionalProperties": False
    }

    summary_hypothesis_template_vllm = {
        "type": "object",
        "properties": {
            "rules": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": ["rules"],
        "additionalProperties": False
    }

    python_implementation_template_vllm = {
    "type": "object",
    "properties": {
        "code": {
        "type": "string",
        },
    },
    "required": ["code"],
    "additionalProperties": False
    }
    return raw_hypothesis_template_vllm, summary_hypothesis_template_vllm, python_implementation_template_vllm