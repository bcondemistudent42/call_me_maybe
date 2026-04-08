import json

from src.parser import Function, Prompt, parsing_function, parsing_prompt
from src.extractor import call_ai

from pydantic import ValidationError
from typing import Any, Dict, List
from llm_sdk import Small_LLM_Model


def main() -> None:
    """Run the full function-calling pipeline and persist JSON output."""

    try:
        data_function: List[Function] = parsing_function()
        data_prompt: List[Prompt] = parsing_prompt()
    except ValidationError as e:
        print(f"Caught Error: {e.errors()[0]['msg']}")
        return

    my_ai: Small_LLM_Model = Small_LLM_Model()
    output_list: List[Dict[str, Any]] = []
    clean_prompt: List[str] = [
        str(x).strip("prompt=").strip("'") for x in data_prompt
    ]
    for each_prompt in clean_prompt:
        output: Dict[str, Any] = call_ai(my_ai, each_prompt, data_function)
        output_list.append(output)
    with open("data/output/output.json", "w") as f:
        json.dump(output_list, f, indent=4)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        print(f"Error occured : {e}")
