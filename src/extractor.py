import numpy as np

from llm_sdk import Small_LLM_Model
from typing import Any, List, Dict
from .parser import Function


def get_function_name(
    my_ai: Small_LLM_Model, data: List[Function], usr_prompt: str
) -> str:
    """Infer the function name using constrained decoding (logit masking)."""
    pre_prompt = (
        "<|im_start|>system\n"
        "I give you access to some function, choose the correct one. "
        "Return only the function name.\n"
        "Functions:\n"
        f"{[x.name for x in data]} \n"
        "<|im_end|>\n"
    )
    assistant_prompt = "<|im_start|>assistant\nfunction used:"
    prompt = pre_prompt + usr_prompt + assistant_prompt
    encoder_prompt = my_ai.encode(prompt)[0].tolist()
    generated_tokens: List[int] = []
    function_name = [my_ai.encode(x.name)[0].tolist() for x in data]
    eof = my_ai.encode("\n")
    max_steps = 95

    for _ in range(max_steps):
        logits = my_ai.get_logits_from_input_ids(encoder_prompt)
        valid_next_tokens = []
        for tokens in function_name:
            if tokens[:len(generated_tokens)] == generated_tokens:
                if len(tokens) > len(generated_tokens):
                    valid_next_tokens.append(tokens[len(generated_tokens)])
                else:
                    valid_next_tokens.append(eof)
        print(generated_tokens)
        print(my_ai.decode(generated_tokens))
        print()
        masked_logits = np.full(len(logits), -np.inf)
        for token_id in valid_next_tokens:
            masked_logits[token_id] = logits[token_id]
        next_token_id = np.argmax(masked_logits)
        if next_token_id == eof or not valid_next_tokens:
            break
        encoder_prompt.append(next_token_id)
        generated_tokens.append(int(next_token_id))
    ft_name = my_ai.decode(generated_tokens).strip()
    return ft_name


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
