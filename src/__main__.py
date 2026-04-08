import json
from typing import Annotated, Any, Dict, List

from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, Field, ValidationError


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


def parsing_function() -> List[Function]:
    """Load and validate function definitions from JSON input.

    Returns:
        List[Function]: Validated list of function definitions.
    """

    data: List[Function] = []
    with open("data/input/functions_definition.json", "r") as f:
        temp: List[Dict[str, Any]] = json.load(f)
    for ft in temp:
        data.append(Function(**ft))
    return data


def parsing_prompt() -> List[Prompt]:
    """Load and validate prompt entries from JSON input.

    Returns:
        List[Prompt]: Validated list of prompts.
    """

    data: List[Prompt] = []
    with open("data/input/function_calling_tests.json", "r") as f:
        temp: List[Dict[str, Any]] = json.load(f)
    for prompt in temp:
        data.append(Prompt(**prompt))
    return data


def get_function_name(
    my_ai: Small_LLM_Model, data: List[Function], usr_prompt: str
) -> str:
    """Infer the function name to call for a user prompt.

    Args:
        my_ai: LLM wrapper used for token generation.
        data: Available function definitions.
        usr_prompt: Prompt to evaluate.

    Returns:
        str: Predicted function name.
    """

    pre_prompt = (
        "<|im_start|>system\n"
        "I give you acess to some function choose the correct one\n"
        "return only the function name that you have to use "
        "then finish your answer\n"
        "the list of function:\n"
        f"{data} \n"
        "<|im_end|>"
    )
    assistant_prompt = "<|im_start|>assistant\n" "function used:"
    prompt = pre_prompt + usr_prompt + assistant_prompt
    encoder_prompt = my_ai.encode(prompt)[0].tolist()
    copy_prompt: list[Any] = []

    i = 0
    while "</think>" not in my_ai.decode(copy_prompt) and i < 350:
        i += 1
        logits = my_ai.get_logits_from_input_ids(encoder_prompt)
        next_token_id = logits.index(max(logits))
        encoder_prompt.append(next_token_id)
        copy_prompt.append(next_token_id)
    ft_name = my_ai.decode(copy_prompt)
    return ft_name.split("\n")[0]


def get_function_args(
    my_ai: Small_LLM_Model,
    function_name: str,
    my_param: str,
    usr_prompt: str,
    prev_answer: str,
) -> str:
    """Infer one argument value for a selected function parameter.

    Args:
        my_ai: LLM wrapper used for token generation.
        function_name: Selected function name.
        my_param: Parameter name to extract.
        usr_prompt: Original prompt content.
        prev_answer: Previously extracted parameters context.

    Returns:
        str: Raw generated argument value.
    """

    pre_prompt = (
        "<|im_start|>system\n"
        f'function name : "{function_name}"\n'
        f'parameter : "{my_param}"\n'
        f'prompt : "{usr_prompt}"\n'
        "Extract parameters from prompt\n"
        "<|im_end|>"
    )
    if len(prev_answer) != 0:
        assistant_prompt = (
            "\n<|im_start|>assistant\n"
            f"extracted parameter {prev_answer} {my_param}:"
        )
    else:
        assistant_prompt = (
            "\n<|im_start|>assistant\n" f"extracted parameter {my_param}:"
        )
    prompt = pre_prompt + assistant_prompt
    encoder_prompt = my_ai.encode(prompt)[0].tolist()
    copy_prompt: list[Any] = []

    i = 0
    while "\n" not in my_ai.decode(copy_prompt) and i < 450:
        i += 1
        logits = my_ai.get_logits_from_input_ids(encoder_prompt)
        next_token_id = logits.index(max(logits))
        encoder_prompt.append(next_token_id)
        copy_prompt.append(next_token_id)
    arg = my_ai.decode(copy_prompt)
    return arg


def call_ai(
    my_ai: Small_LLM_Model, base_prompt: str, data_function: List[Function]
) -> Dict[str, Any]:
    """Build a structured function-call object from one natural prompt.

    Args:
        my_ai: LLM wrapper used for generation.
        base_prompt: User natural language request.
        data_function: Available function definitions.

    Returns:
        Dict[str, Any]: Dictionary containing prompt,
            function name, and parameters.
    """

    usr_prompt = f"<|im_start|> \n {base_prompt} \n <|im_end|>"
    name = get_function_name(my_ai, data_function, usr_prompt).strip(" ")
    i: int = 0
    for func in data_function:
        if func.name == name:
            break
        i += 1
    clear_param: List[str] = (
        str(data_function[i].parameters.keys())
        .strip("dict_keys")
        .strip("()")
        .strip("[]")
        .split(",")
    )
    param_type: str = str(data_function[i].parameters.values()).strip(
        "dict_values("
    )
    for elt in clear_param:
        elt.strip("''")
    j: int = 0
    txt: str = ""
    temp_param: List[str] = []
    parsed_param: Any = []
    for param in clear_param:
        j += 1
        rslt = get_function_args(
            my_ai, name, param.strip(" "), usr_prompt, txt
        )
        temp_param.append(rslt)
        txt += clear_param[j - 1] + ":" + rslt
    args_cleaned: List[str] = [
        x.strip("\n").strip(" ").strip("''") for x in temp_param
    ]
    k: int = 0
    for elt in clear_param:
        parsed_param.append(
            (
                elt.strip(" ").strip("''"),
                args_cleaned[k].strip(" ").strip("''"),
            )
        )
        k += 1
    parsed_param = dict(parsed_param)
    for key, elt in parsed_param.items():
        parsed_param[key] = elt.strip(" \"'")
        try:
            if "number" in param_type:
                parsed_param[key] = float(elt)
            elif "integer" in param_type:
                parsed_param[key] = int(elt)
        except ValueError:
            parsed_param[key] = 1
            pass
    big_dict_data: Dict[str, Any] = {}
    big_dict_data["prompt"] = base_prompt.strip(" \"'")
    big_dict_data["name"] = name
    big_dict_data["parameters"] = parsed_param
    return big_dict_data


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
    main()
