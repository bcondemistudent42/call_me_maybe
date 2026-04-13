import argparse
import json
import os

from .parser import Function, Prompt, parsing_function, parsing_prompt
from .extractor import call_ai

from pydantic import ValidationError
from typing import Any, Dict, List
from llm_sdk import Small_LLM_Model


def main() -> None:
    """Run the full function-calling pipeline and persist JSON output."""

    parser_param = argparse.ArgumentParser()
    parser_param.add_argument("--functions_definition",
                              default="data/input/functions_definition.json",
                              help="<function_definition_file>")
    parser_param.add_argument("--input",
                              default="data/input/function_calling_tests.json",
                              help="<input_file>")
    parser_param.add_argument("--output",
                              default="data/output",
                              help="<output_file>")
    args = parser_param.parse_args()
    try:
        data_ft: List[Function] = parsing_function(args.functions_definition)
        data_prompt: List[Prompt] = parsing_prompt(args.input)
    except ValidationError as e:
        print(f"Caught Error: {e.errors()[0]['msg']}")
        return
    my_ai: Small_LLM_Model = Small_LLM_Model()
    output_list: List[Dict[str, Any]] = []
    clean_prompt: List[str] = [
        str(x).strip("prompt=").strip("'") for x in data_prompt
    ]
    os.system('cls||clear')
    for each_prompt in clean_prompt:
        output: Dict[str, Any] = call_ai(my_ai, each_prompt, data_ft)
        print(each_prompt)
        print(f"Result : {output} \n")
        output_list.append(output)
    output_dir = args.output
    file_path = os.path.join(output_dir, "function_calling_results.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(output_list, f, indent=4)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        print(f"Error occured : {e}")
