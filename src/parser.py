import json
from pydantic import BaseModel, Field
from typing import Annotated, Any, Dict, List


class Returns(BaseModel):
    """Schema for a function return type definition."""
    type: Annotated[str, Field(min_length=1, max_length=35)]


class Parameters(BaseModel):
    """Schema for a single function parameter definition."""
    type: str


class Function(BaseModel):
    """Schema describing one callable function in the catalog."""
    name: Annotated[str, Field(pattern=r"^fn", min_length=4, max_length=35)]
    description: str
    parameters: Dict[str, Parameters]
    returns: Returns


class Prompt(BaseModel):
    """Schema for one user prompt entry loaded from input."""
    prompt: Annotated[str, Field(min_length=2, max_length=100)]


def parsing_function(function_place: str) -> List[Function]:
    """Load and validate function definitions from JSON input.

    Returns:
        List[Function]: Validated list of function definitions.
    """

    data: List[Function] = []
    with open(function_place, "r") as f:
        temp: List[Dict[str, Any]] = json.load(f)
    for ft in temp:
        data.append(Function(**ft))
    return data


def parsing_prompt(prompt_place: str) -> List[Prompt]:
    """Load and validate prompt entries from JSON input.

    Returns:
        List[Prompt]: Validated list of prompts.
    """

    data: List[Prompt] = []
    with open(prompt_place, "r") as f:
        temp: List[Dict[str, Any]] = json.load(f)
    for prompt in temp:
        data.append(Prompt(**prompt))
    return data
